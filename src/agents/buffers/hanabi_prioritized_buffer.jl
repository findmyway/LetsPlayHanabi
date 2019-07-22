using LinearAlgebra:dot

struct HanabiPrioritizedBuffer
    states::CircularArrayBuffer{Vector{Int32}}
    actions::CircularArrayBuffer{Int32}
    rewards::CircularArrayBuffer{Int32}
    isdone::CircularArrayBuffer{Bool}
    legal_actions::CircularArrayBuffer{Vector{Float32}}
    update_horizon::Int
    gamma::Float32
    discounts::Vector{Float32}
    sum_tree::SumTree{Float32}

    function HanabiPrioritizedBuffer(capacity, feature_size, n_actions, update_horizon, gamma)
        new(
            CircularArrayBuffer{Vector{Int32}}(capacity, (feature_size,)),
            CircularArrayBuffer{Int32}(capacity),
            CircularArrayBuffer{Int32}(capacity),
            CircularArrayBuffer{Bool}(capacity),
            CircularArrayBuffer{Vector{Float32}}(capacity, (n_actions,)),
            update_horizon,
            gamma,
            [gamma ^ i for i in 0:update_horizon-1],
            SumTree{Float32}(capacity)
        )
    end
end

function push!(buffer::HanabiPrioritizedBuffer, b::HanabiEpisodeBuffer, priorities)
    for (i, p) in enumerate(priorities)
        push!(buffer.states, view(b.states, i))
        push!(buffer.actions, b.actions[i])
        push!(buffer.rewards, b.rewards[i])
        push!(buffer.isdone, b.isdone[i])
        push!(buffer.legal_actions, view(b.legal_actions, i))
        push!(buffer.sum_tree, p)
    end

end

function push!(buffer::HanabiPrioritizedBuffer, b::HanabiEpisodeBuffer; default_priority=100f0)
    for i in 1:length(b)
        push!(buffer.states, view(b.states, i))
        push!(buffer.actions, b.actions[i])
        push!(buffer.rewards, b.rewards[i])
        push!(buffer.isdone, b.isdone[i])
        push!(buffer.legal_actions, view(b.legal_actions, i))
        push!(buffer.sum_tree, default_priority)
    end
end

Base.length(buffer::HanabiPrioritizedBuffer) = length(buffer.sum_tree)

isvalid(buffer::HanabiPrioritizedBuffer, i) = (i >= 1) && (i + buffer.update_horizon) <= length(buffer)

function sample_indices(t::HanabiPrioritizedBuffer, n::Int)
    inds = Vector{Int}(undef, n)
    priorities = Vector{Float32}(undef, n)
    for i in 1:n
        ind, p = sample(t.sum_tree)
        while !isvalid(t, ind)
            ind, p = sample(t.sum_tree)
        end
        inds[i], priorities[i] = ind, p
    end
    inds, priorities
end

function sample(t::HanabiPrioritizedBuffer, n::Int)
    inds, priorities = sample_indices(t, n)
    next_inds = inds .+ t.update_horizon

    states_batch = view(t.states, inds)
    next_states_batch = view(t.states, next_inds)
    next_legal_actions_batch = view(t.legal_actions, next_inds)
    actions_batch = view(t.actions, inds)

    rewards_batch = zeros(Float32, n)
    isdone_batch = fill(false, n)

    for i in 1:n
        ind = inds[i]
        isdone_consecutive = view(t.isdone, ind:ind+t.update_horizon-1)
        d = findfirst(isdone_consecutive)

        if isnothing(d)
            isdone_batch[i] = false
            rewards_batch[i] = dot(t.discounts, view(t.rewards, ind:ind+t.update_horizon-1))
        else
            isdone_batch[i] = true
            rewards_batch[i] = dot(@view(t.discounts[1:d]), @view(t.rewards[ind:ind+d-1]))
        end
    end

    states_batch, actions_batch, rewards_batch, isdone_batch, next_states_batch, next_legal_actions_batch, inds
end