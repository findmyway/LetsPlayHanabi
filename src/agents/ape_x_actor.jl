include("buffers/circular_array_buffer.jl")
include("buffers/hanabi_episode_buffer.jl")

using Random
using Hanabi
using Flux
using LetsPlayHanabi
using LinearAlgebra

struct RainbowActor{Ts, Tn}
    support::Ts
    delta_z::Float32
    Vₘₐₓ::Float32
    n_atoms::Int
    γ::Float32
    update_horizon::Int
    episode_buffer::HanabiEpisodeBuffer
    network::Tn
    ϵ_eval::Float32
end

function RainbowActor(
    ϵ,
    Vₘₐₓ,
    feature_size,
    n_actions;
    γ=0.99f0,
    update_horizon=1,
    n_atoms=51,
    layer_sizes=(512, 512),
    max_steps_of_local_buffer=100)

    support = range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length=n_atoms)

    network = create_network(feature_size, layer_sizes, n_actions * n_atoms)
    episode_buffer = HanabiEpisodeBuffer(max_steps_of_local_buffer, feature_size, n_actions)

    RainbowActor(collect(support), Float32(support.step), Float32(Vₘₐₓ), n_atoms, γ, update_horizon, episode_buffer, network, Float32(ϵ))
end

buffer(agent::RainbowActor) = agent.episode_buffer

function create_network(in, layer_sizes, out)
   Chain(
        Dense(in, layer_sizes[1], relu),
        (Dense(layer_sizes[i], layer_sizes[i+1], relu) for i in 1:(length(layer_sizes)-1))...,
        Dense(layer_sizes[end], out; initW=(out, in) -> (limit=sqrt(sqrt(3.0f0) / in); rand(Float32, out, in) .* limit .* 2 .- limit))
    )
end

function (agent::RainbowActor)(s, legal_action)
    n_actions = length(legal_action)

    q = agent.network(s).data
    q = agent.support .* softmax(reshape(q, :, n_actions))
    probs = vec(sum(q, dims=1)) .+ legal_action |> cpu
    a = rand() < agent.ϵ_eval ? rand(findall(iszero, legal_action)) : argmax(probs)
    a
end

function act_one_episode(env, agents, experiences)
    reset!(env)
    n_step, total_reward = 0, 0
    rewards_in_a_round = zeros(Int32, length(agents))
    is_players_started = fill(false, length(agents))

    # start
    pid = get_cur_player(env)
    obs = observe(env)

    s = push!(buffer(agents[pid]).states, x -> encode_observation!(obs.observation, env, x))
    legal_actions = push!(buffer(agents[pid]).legal_actions, x -> legal_actions!(env, x))
    a = agents[pid](s, legal_actions)
    push!(buffer(agents[pid]).actions, a |> Int32)

    is_players_started[pid] = true

    while true
        interact!(env, a)
        obs = observe(env)

        # !!! TODO: change this API
        total_reward += env.reward.score_gain
        n_step += 1
        # !!! TODO: change this API
        rewards_in_a_round .+= env.reward.score_gain

        if obs.isdone
            break
        end

        pid = get_cur_player(env)

        r = rewards_in_a_round[pid]  # reward since last action
        rewards_in_a_round[pid] = 0

        if is_players_started[pid]
            push!(buffer(agents[pid]).rewards, r)
            push!(buffer(agents[pid]).isdone, obs.isdone)
            s = push!(buffer(agents[pid]).states, x -> encode_observation!(obs.observation, env, x))
            legal_actions = push!(buffer(agents[pid]).legal_actions, x -> legal_actions!(env, x))
            a = agents[pid](s, legal_actions)
            push!(buffer(agents[pid]).actions, a |> Int32)
        else
            is_players_started[pid] = true

            s = push!(buffer(agents[pid]).states, x -> encode_observation!(obs.observation, env, x))
            legal_actions = push!(buffer(agents[pid]).legal_actions, x -> legal_actions!(env, x))
            a = agents[pid](s, legal_actions)
            push!(buffer(agents[pid]).actions, a |> Int32)
        end
    end

    for (agent, i, r) in zip(agents, 1:length(agents), rewards_in_a_round)
        if is_players_started[i]
            push!(buffer(agent).rewards, r)
            push!(buffer(agent).isdone, true)
        end
    end

    for agent in agents
        if isfull(buffer(agent))
            # TODO: send to learner
            put!(experiences, (buffer(agent), get_batch_priorities(agent)))
            empty!(buffer(agent))
        end
    end

    n_step, total_reward
end

function logitcrossentropy_expand(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
  return vec(-sum(y .* logsoftmax(logŷ), dims=1))
end

function get_batch_priorities(actor)
    b = buffer(actor)
    inds = 1:(length(b) - actor.update_horizon)
    next_inds = (1 + actor.update_horizon):length(b)
    n = length(inds)

    states = view(b.states, inds)
    next_states = view(b.states, next_inds)

    next_legal_actions = view(b.legal_actions, next_inds)
    actions = view(b.actions, inds)
    rewards = zeros(Float32, n)
    isdone = fill(false, n)

    discounts = [actor.γ ^ i for i in 0:actor.update_horizon-1]

    for (i, ind) in enumerate(inds)
        isdone_consecutive = view(b.isdone, ind:ind+actor.update_horizon-1)
        d = findfirst(isdone_consecutive)

        if isnothing(d)
            isdone[i] = false
            rewards[i] = dot(discounts, view(b.rewards, ind:ind+actor.update_horizon-1))
        else
            isdone[i] = true
            rewards[i] = dot(@view(discounts[1:d]), @view(b.rewards[ind:ind+d-1]))
        end
    end

    n_atoms, n_actions = actor.n_atoms, size(next_legal_actions, 1)

    γ_with_terminal = (actor.γ ^ actor.update_horizon) .* (1 .- isdone)
    target_support = reshape(rewards, 1, :) .+ (reshape(actor.support, :, 1) * reshape(γ_with_terminal, 1, :))

    logits = reshape(actor.network(states), n_atoms, n_actions, :)
    select_logits = logits[:, [CartesianIndex(a, i) for (i, a) in enumerate(actions)]]

    next_logits = actor.network(next_states).data
    next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)
    next_q = reshape(sum(actor.support .* next_probs, dims=1), n_actions, :)
    next_q_argmax = argmax(cpu(next_q .+ next_legal_actions), dims=1)
    next_prob_select = reshape(next_probs[:, next_q_argmax], n_atoms, :)

    target_distribution = project_distribution(target_support, next_prob_select, actor.support, actor.delta_z, -actor.Vₘₐₓ, actor.Vₘₐₓ)

    losses = logitcrossentropy_expand(select_logits, target_distribution)

    vec(clamp.(sqrt.(losses.data .+ 1f-10), 1.f0, 1.f2))
end

function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(repeat(clampped_support, n_atoms), n_atoms, n_atoms, batch_size)

    projection = clamp.(1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z, 0, 1) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims=1), n_atoms, batch_size)
end

function run_actor(
    ϵ_of_agents,
    experiences,
    latest_params,
    ;seed=123,
    max_steps_of_local_buffer=100,
    n_phase=10
    )

    @info "actor [$(myid())] started..."

    Random.seed!(seed)
    env = HanabiEnv(;seed=seed)
    n_colors, n_ranks, n_hands, n_players, n_actions = num_colors(env.game), num_ranks(env.game), hand_size(env.game), num_players(env.game), max_moves(env.game)

    agents =  [RainbowActor(ϵ_of_agents[i], n_colors * n_ranks, env.observation_length, n_actions;max_steps_of_local_buffer=max_steps_of_local_buffer) for i in 1:n_players]

    network_params = fetch(latest_params)

    for agent in agents
        Flux.loadparams!(agent.network, network_params)
    end

    @info "params initialized on worker[$(myid())] with epsilon=[$ϵ_of_agents], start acting..."

    total_length = 0
    for i in 1:n_phase
        network_params = fetch(latest_params)
        for agent in agents
            Flux.loadparams!(agent.network, network_params)
        end

        while true
            act_one_episode(env, agents, experiences)
            cached_experiences = length(buffer(agents[1]))
            if cached_experiences == 0 
                break
            else
                @debug "[$(myid())] $cached_experiences cached, continue..."
            end
        end
        @debug "worker [$(myid())] finished $i phases"
    end

    close(latest_params)
end