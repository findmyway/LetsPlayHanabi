## modified from https://github.com/findmyway/hanabi-learning-environment/blob/master/agents/simple_agent.py

using Hanabi

export SimpleAgent, get_action

struct SimpleAgent
end

function get_action(agent::SimpleAgent, obs)
    game, raw_obs = obs.game, obs.observation
    obs_cur_player_offset(raw_obs) == 0 || throw(ArgumentError("the agent is not the current player"))

    # 1. check if there's any playable card
    for i in 0:(obs_get_hand_size(raw_obs, 0)-1)
        card_knowledge = get_hand_card_knowledge(raw_obs, 0, i)
        if !isnothing(rank(card_knowledge)) || !isnothing(color(card_knowledge))
            return PlayCard(i+1)
        end
    end

    # 2. Check if it's possible to hint a card to your colleagues.
    fireworks = get_fireworks(game, raw_obs)
    if obs_information_tokens(raw_obs) > 0
        for player_offset in 1:(obs_num_players(raw_obs) - 1)
            for card_id in 0:(obs_get_hand_size(raw_obs, player_offset) - 1)
                card = get_hand_card(raw_obs, player_offset, card_id)
                card_knowledge = get_hand_card_knowledge(raw_obs, player_offset, card_id)
                if rank(card) == fireworks[color(card)] && isnothing(color(card_knowledge))
                    return RevealColor(player_offset, color(card))
                end
            end
        end
    end

    # 3. If no card is hintable then discard or play.
    if obs_information_tokens(raw_obs) < max_information_tokens(game)
        DiscardCard(1)  # notice the index
    else
        PlayCard(1)
    end
end