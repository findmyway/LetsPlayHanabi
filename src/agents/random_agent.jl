using StatsBase:sample

export RandomAgent, get_action

struct RandomAgent
end

function get_action(agent::RandomAgent, obs)
    sample(obs.legal_actions)
end