include("rainbow_agent.jl")
using CUDAnative
device!(1)
seed = 123
Random.seed!(seed)
env = HanabiEnv(;seed=seed)

n_colors, n_ranks, n_hands, n_players, n_actions = num_colors(env.game), num_ranks(env.game), hand_size(env.game), num_players(env.game), max_moves(env.game)

agent = RainbowAgent(
    n_colors * n_ranks,
    env.observation_length,
    n_actions
)

using PyCall
using Flux

np = pyimport("numpy")

o_w1 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Online_fully_connected_weights.npy")'
o_b1 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Online_fully_connected_biases.npy")
o_w2 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Online_fully_connected_1_weights.npy")'
o_b2 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Online_fully_connected_1_biases.npy")
o_w3 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Online_fully_connected_2_weights.npy")'
o_b3 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Online_fully_connected_2_biases.npy")

t_w1 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Target_fully_connected_weights.npy")'
t_b1 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Target_fully_connected_biases.npy")
t_w2 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Target_fully_connected_1_weights.npy")'
t_b2 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Target_fully_connected_1_biases.npy")
t_w3 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Target_fully_connected_2_weights.npy")'
t_b3 = np.load("/home/tj/hanabi_rainbow/tf_ckpt-7700/Target_fully_connected_2_biases.npy")

function create_network(in, layer_sizes, out)
   Chain(
        Dense(in, layer_sizes[1], relu),
        (Dense(layer_sizes[i], layer_sizes[i+1], relu) for i in 1:(length(layer_sizes)-1))...,
        Dense(layer_sizes[end], out; initW=(out, in) -> (limit=sqrt(sqrt(3.0f0) / in); rand(Float32, out, in) .* limit .* 2 .- limit))
    ) |> gpu
end

agent.online_network.layers[1].W.data .= o_w1 |> gpu
agent.online_network.layers[1].b.data .= o_b1 |> gpu
agent.online_network.layers[2].W.data .= o_w2 |> gpu
agent.online_network.layers[2].b.data .= o_b2 |> gpu
agent.online_network.layers[3].W.data .= o_w3 |> gpu
agent.online_network.layers[3].b.data .= o_b3 |> gpu

agent.target_network.layers[1].W.data .= t_w1 |> gpu
agent.target_network.layers[1].b.data .= t_b1 |> gpu
agent.target_network.layers[2].W.data .= t_w2 |> gpu
agent.target_network.layers[2].b.data .= t_b2 |> gpu
agent.target_network.layers[3].W.data .= t_w3 |> gpu
agent.target_network.layers[3].b.data .= t_b3 |> gpu

states = np.load("/home/tj/hanabi_rainbow/data4julia/states.npy") |> x -> Array{Float32}(reshape(x, 32, 658)')
actions = np.load("/home/tj/hanabi_rainbow/data4julia/actions.npy")
rewards = np.load("/home/tj/hanabi_rainbow/data4julia/rewards.npy")
isdone = np.load("/home/tj/hanabi_rainbow/data4julia/isdone.npy")
next_states = np.load("/home/tj/hanabi_rainbow/data4julia/next_states.npy") |> x -> Array{Float32}(reshape(x, 32, 658)')
next_legal_actions = np.load("/home/tj/hanabi_rainbow/data4julia/next_legal_actions.npy")'

v_logits = np.load("/home/tj/hanabi_rainbow/data4julia/logits.npy") |> x -> permutedims(x, [3,2,1])
v_target_support = np.load("/home/tj/hanabi_rainbow/data4julia/target_support.npy")'
v_next_prob = np.load("/home/tj/hanabi_rainbow/data4julia/next_prob.npy")'
v_projection = np.load("/home/tj/hanabi_rainbow/data4julia/projection.npy")'