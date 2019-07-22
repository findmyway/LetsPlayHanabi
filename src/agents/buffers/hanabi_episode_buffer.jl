struct HanabiEpisodeBuffer
    states::CircularArrayBuffer{Vector{Int32}}
    actions::CircularArrayBuffer{Int32}
    rewards::CircularArrayBuffer{Int32}
    isdone::CircularArrayBuffer{Bool}
    legal_actions::CircularArrayBuffer{Vector{Float32}}
    function HanabiEpisodeBuffer(max_steps_per_episode, feature_size, n_actions)
        new(
            CircularArrayBuffer{Vector{Int32}}(max_steps_per_episode, (feature_size,)),
            CircularArrayBuffer{Int32}(max_steps_per_episode),
            CircularArrayBuffer{Int32}(max_steps_per_episode),
            CircularArrayBuffer{Bool}(max_steps_per_episode),
            CircularArrayBuffer{Vector{Float32}}(max_steps_per_episode, (n_actions,))
        )
    end
end

function Base.similar(buffer::HanabiEpisodeBuffer)
    max_steps_per_episode, feature_size, n_actions = size(buffer.states.buffer, 2), size(buffer.states.buffer, 1), size(buffer.legal_actions.buffer, 1)
    HanabiEpisodeBuffer(max_steps_per_episode, feature_size, n_actions)
end

function Base.empty!(buffer::HanabiEpisodeBuffer)
    for f in fieldnames(HanabiEpisodeBuffer)
        b = getfield(buffer, f)
        # isfull(b) && error("episode shouldn't be full! try increase max_steps_per_episode!")
        empty!(b)
    end
end

Base.length(b::HanabiEpisodeBuffer) = length(b.rewards)

isfull(b::HanabiEpisodeBuffer) = isfull(b.rewards)