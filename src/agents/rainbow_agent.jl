include("buffers/circular_array_buffer.jl")
include("buffers/hanabi_episode_buffer.jl")
include("buffers/sum_tree.jl")
include("buffers/hanabi_prioritized_buffer.jl")

using LetsPlayHanabi
using Hanabi
using Flux
using StatsBase
using Random
using BSON
using Logging

const log_io = open("test.log", "w+")
const logger = SimpleLogger(log_io)

# using CuArrays
# CuArrays.allowscalar(false)
# include("patch.jl")

mutable struct RuntimeInfo
    training_step::Int
    eval_step::Int
end

RuntimeInfo() = RuntimeInfo(0, 0)

struct RainbowAgent{Tn, To, Tp, Ts}
    support::Ts
    delta_z::Float32
    Vₘₐₓ::Float32
    n_atoms::Int
    γ::Float32
    n_step::Int
    replay_buffer::HanabiPrioritizedBuffer
    episode_buffer::HanabiEpisodeBuffer
    min_replay_history::Int
    online_network::Tn
    target_network::Tn
    optimizer::To
    params::Tp
    update_period::Int
    target_update_period::Int
    batch_size::Int
    ϵ_train::Float32
    ϵ_eval::Float32
    ϵ_decay_period::Int
    runtime_info::RuntimeInfo
end

function RainbowAgent(
    Vₘₐₓ,
    feature_size,
    n_actions;
    γ=0.99f0,
    n_step=1,
    min_replay_history=500,
    n_atoms=51,
    layer_sizes=(512, 512),
    max_steps_per_episode=100,
    replay_capacity=1000000,
    update_period=4,
    target_update_period=500,
    batch_size=32,
    ϵ_train=0.0f0,
    ϵ_eval=0.0f0,
    ϵ_decay_period=1000,
    learning_rate=0.000025,
    ϵ_optimizer=0.00003125)

    support = range(Float32(-Vₘₐₓ), Float32(Vₘₐₓ), length=n_atoms)

    online_network = create_network(feature_size, layer_sizes, n_actions * n_atoms) |> gpu
    target_network = create_network(feature_size, layer_sizes, n_actions * n_atoms) |> gpu
    ps = params(online_network)
    Flux.loadparams!(target_network, ps)

    optimizer = ADAM(learning_rate)

    episode_buffer = HanabiEpisodeBuffer(max_steps_per_episode, feature_size, n_actions)
    replay_buffer = HanabiPrioritizedBuffer(replay_capacity, feature_size, n_actions, n_step, γ)

    RainbowAgent(collect(support)|>gpu, Float32(support.step), Float32(Vₘₐₓ), n_atoms, γ, n_step, replay_buffer, episode_buffer, min_replay_history, online_network, target_network, optimizer, ps, update_period, target_update_period, batch_size, ϵ_train, ϵ_eval, ϵ_decay_period, RuntimeInfo())
end

function get_training_ϵ(agent::RainbowAgent)
    warmup_steps, decay_steps, training_step = agent.min_replay_history, agent.min_replay_history, agent.runtime_info.training_step

    if training_step <= warmup_steps
        1.0
    elseif training_step >= (warmup_steps + decay_steps)
        agent.ϵ_train
    else
        steps_left = warmup_steps + decay_steps - training_step
        agent.ϵ_train + steps_left / decay_steps * (1 - agent.ϵ_train)
    end
end

"share all params except episode buffer"
function RainbowAgent(agent::RainbowAgent;ϵ_eval=nothing)
    RainbowAgent(
        agent.support,
        agent.delta_z,
        agent.Vₘₐₓ,
        agent.n_atoms,
        agent.γ,
        agent.n_step,
        agent.replay_buffer,
        similar(agent.episode_buffer),
        agent.min_replay_history,
        agent.online_network,
        agent.target_network,
        agent.optimizer,
        agent.params,
        agent.update_period,
        agent.target_update_period,
        agent.batch_size,
        agent.ϵ_train,
        isnothing(ϵ_eval) ? agent.ϵ_eval : ϵ_eval,
        agent.ϵ_decay_period,
        agent.runtime_info
    )
end

function update!(agent::RainbowAgent, is_train)
    if is_train
        agent.runtime_info.training_step += 1
        
        if (length(agent.replay_buffer) > agent.min_replay_history) && (agent.runtime_info.training_step % agent.update_period == 0)
            train_once!(agent)
        end

        if agent.runtime_info.training_step % agent.target_update_period == 0
            Flux.loadparams!(agent.target_network, agent.params)
        end
    else
        agent.runtime_info.eval_step += 1
    end
    agent
end

function logitcrossentropy_expand(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
  return vec(-sum(y .* logsoftmax(logŷ), dims=1))
end

function train_once!(agent::RainbowAgent, states, actions, rewards, isdone, next_states, next_legal_actions)
    n_atoms, n_actions = agent.n_atoms, size(next_legal_actions, 1)

    γ_with_terminal = (agent.γ ^ agent.n_step) .* (1 .- isdone)
    target_support = reshape(rewards, 1, :) .+ (reshape(agent.support, :, 1) * reshape(γ_with_terminal, 1, :))

    logits = reshape(agent.online_network(gpu(states)), n_atoms, n_actions, :)
    select_logits = logits[:, [CartesianIndex(a, i) for (i, a) in enumerate(actions)]]

    next_logits = agent.target_network(gpu(next_states)).data
    next_probs = reshape(softmax(reshape(next_logits, n_atoms, :)), n_atoms, n_actions, :)
    next_q = reshape(sum(agent.support .* next_probs, dims=1), n_actions, :)
    next_q_argmax = argmax(cpu(next_q .+ next_legal_actions), dims=1)
    next_prob_select = reshape(next_probs[:, next_q_argmax], n_atoms, :)

    target_distribution = project_distribution(target_support, next_prob_select, agent.support, agent.delta_z, -agent.Vₘₐₓ, agent.Vₘₐₓ)

    losses = logitcrossentropy_expand(select_logits, target_distribution)
    updated_priorities = vec(clamp.(sqrt.(losses.data .+ 1f-10), 1.f0, 1.f2))

    target_priorities = 1.0f0 ./ sqrt.(updated_priorities .+ 1f-10)
    target_priorities ./= maximum(target_priorities)
    weighted_loss = mean(target_priorities .* losses)

    gs = Tracker.gradient(() -> weighted_loss, agent.params)
    Flux.Optimise.update!(agent.optimizer, agent.params, gs)

    updated_priorities
end

function train_once!(agent::RainbowAgent)
    states, actions, rewards, isdone, next_states, next_legal_actions, inds = sample(agent.replay_buffer, agent.batch_size)
    updated_priorities = train_once!(agent, gpu(states), actions, gpu(rewards), gpu(isdone), gpu(next_states), gpu(next_legal_actions))
    agent.replay_buffer.sum_tree[inds] .= updated_priorities |> cpu
end

function project_distribution(supports, weights, target_support, delta_z, vmin, vmax)
    batch_size, n_atoms = size(supports, 2), length(target_support)
    clampped_support = clamp.(supports, vmin, vmax)
    tiled_support = reshape(repeat(clampped_support, n_atoms), n_atoms, n_atoms, batch_size)

    projection = clamp.(1 .- abs.(tiled_support .- reshape(target_support, 1, :)) ./ delta_z, 0, 1) .* reshape(weights, n_atoms, 1, batch_size)
    reshape(sum(projection, dims=1), n_atoms, batch_size)
end

buffer(agent::RainbowAgent) = agent.episode_buffer

function (agent::RainbowAgent)(s, legal_actions, is_train)
    ϵ = is_train ? get_training_ϵ(agent) : agent.ϵ_eval
    if rand() < ϵ
        rand(findall(iszero, legal_actions))
    else
        agent(s, legal_actions)
    end
end

function (agent::RainbowAgent)(s, legal_action)
    s, legal_action = gpu(s), gpu(legal_action)
    n_actions = length(legal_action)

    q = agent.online_network(s).data
    q = agent.support .* softmax(reshape(q, :, n_actions))
    probs = vec(sum(q, dims=1)) .+ legal_action
    argmax(cpu(probs))
end

function create_network(in, layer_sizes, out)
   Chain(
        Dense(in, layer_sizes[1], relu),
        (Dense(layer_sizes[i], layer_sizes[i+1], relu) for i in 1:(length(layer_sizes)-1))...,
        Dense(layer_sizes[end], out; initW=(out, in) -> (limit=sqrt(sqrt(3.0f0) / in); rand(Float32, out, in) .* limit .* 2 .- limit))
    )
end


get_cur_player(env) = cur_player(env) + 1  # pid is 0-based

function run_one_episode(env, agents; is_train=true)
    reset!(env)
    n_step, total_reward = 0, 0
    rewards_in_a_round = zeros(Int32, length(agents))
    is_players_started = fill(false, length(agents))

    # start
    pid = get_cur_player(env)
    obs = observe(env)

    s = push!(buffer(agents[pid]).states, x -> encode_observation!(obs.observation, env, x))
    legal_actions = push!(buffer(agents[pid]).legal_actions, x -> legal_actions!(env, x))
    a = agents[pid](s, legal_actions, is_train) |> Int32
    push!(buffer(agents[pid]).actions, a)

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
            a = agents[pid](s, legal_actions, is_train) |> Int32
            push!(buffer(agents[pid]).actions, a)

            update!(agents[pid], is_train)
        else
            is_players_started[pid] = true

            s = push!(buffer(agents[pid]).states, x -> encode_observation!(obs.observation, env, x))
            legal_actions = push!(buffer(agents[pid]).legal_actions, x -> legal_actions!(env, x))
            a = agents[pid](s, legal_actions, is_train) |> Int32
            push!(buffer(agents[pid]).actions, a)
        end
    end

    for (agent, i, r) in zip(agents, 1:length(agents), rewards_in_a_round)
        if is_players_started[i]
            push!(buffer(agent).rewards, r)
            push!(buffer(agent).isdone, true)
            update!(agents[pid], is_train)
        end
    end


    # TODO: update replay buffer, training

    for agent in agents
        is_train && push!(agent.replay_buffer, agent.episode_buffer)
        empty!(buffer(agent))
    end

    with_logger(logger) do
        @debug "training info" agents[1].runtime_info.training_step n_step total_reward
    end

    n_step, total_reward
end

function run_one_phase(env, agents, min_steps, statistics; is_train=true)
    n_steps, n_episodes, total_rewards = 0, 0, 0.

    while n_steps < min_steps
        n, r = run_one_episode(env, agents; is_train=is_train)
        n_steps += n
        total_rewards += r
        n_episodes += 1
        push!(statistics, (episode_length=n, episode_reward=r))
    end

    n_steps, total_rewards, n_episodes
end

function run_one_iteration(env, agents, min_training_steps, current_iter, statistics; eval_freq=100, n_eval_games=100)
    training_stats = []
    (n_steps, total_rewards, n_episodes), t = @timed run_one_phase(env, agents, min_training_steps, training_stats)
    with_logger(logger) do
        @info "=(------> training info of iter $current_iter" avg_steps_per_second=n_steps/t avg_return_per_episode=total_rewards/n_episodes
    end

    statistics["training_result_$current_iter"] = training_stats

    if current_iter % eval_freq == 1
        eval_stats = []
        for _ in 1:n_eval_games
            n, r = run_one_episode(env, agents; is_train=false)
            push!(eval_stats, (episode_length=n, episode_reward=r))
        end
        with_logger(logger) do
            @info "=(------> evaluation info of iter $current_iter" avg_length=mean(x -> x.episode_length, eval_stats) avg_rewards=mean(x -> x.episode_reward, eval_stats)
        end
        statistics["eval_result_$current_iter"] = eval_stats
    end

    statistics
end

function train_rainbow_agents(
    ;n_iteration=10001,
    save_freq=100,
    min_training_steps_per_iteration=10000,
    eval_freq=100,
    n_eval_games=100,
    seed=123
    )

    Random.seed!(seed)
    env = HanabiEnv(;seed=seed)

    n_colors, n_ranks, n_hands, n_players, n_actions = num_colors(env.game), num_ranks(env.game), hand_size(env.game), num_players(env.game), max_moves(env.game)

    agent = RainbowAgent(
        n_colors * n_ranks,
        env.observation_length,
        n_actions
    )

    agents = (agent, (RainbowAgent(agent) for _ in 1:n_players-1)...)
    statistics = Dict{String, Vector{NamedTuple{(:episode_length, :episode_reward),Tuple{Int64,Int64}}}}()

    for iter in 1:n_iteration
        run_one_iteration(env, agents, min_training_steps_per_iteration, iter, statistics; eval_freq=eval_freq, n_eval_games=n_eval_games)

        if iter % save_freq == 1
            network = cpu(agent.online_network)
            BSON.@save "check_points/agent_online_network_$iter" network
            BSON.@save "check_points/statistics_$iter" statistics
        end

        flush(log_io)
    end

    agents, statistics
end

# train_rainbow_agents()