"""
So the idea here is that some of the nodes are fixed to some source:
- Audio, pose graph?, trig functions

and the rest diffuse into the minimal energy states given by the graph laplacian.

TODO:
- [ ] store into array for video rendering
- [ ] Think of a way to clamp nodes into audio:
    - Probably project onto the fourier basis and have nodes represent frequencies.
- [ ] Clean node visuals:
    - [ ] Larger node faces
"""

using Random, Distributions
using GLMakie
using Graphs, GraphMakie

import Graphs.LinAlg.Adjacency

set_theme!(theme_dark())

Random.seed!(42)

include("utils.jl")

x_dim = 300
y_dim = 300
n_pose_graphs = 3
# n_nodes_in_pose_graph = 16
# n_nodes = n_pose_graphs * n_nodes_in_pose_graph
n_nodes = n_pose_graphs * NUM_NODES_PER_POSE_GRAPH

# Fixed nodes

n_fixed_nodes = 0
# fixed_idxs = sample(1:n_nodes, n_fixed_nodes; replace=false)
fixed_mask = [false for i in 1:n_nodes]

if n_fixed_nodes > 0
    fixed_mask[sample(1:n_nodes, n_fixed_nodes; replace=false)] .= true
end

omega = 1
a = 1.

# Diffusion
delta = 0.1

# # Animation loop
dt = 0.1
duration = 20

# Define parameters
num = 20
x_min = -num
x_max = num
y_min = -num
y_max = num
x = LinRange(x_min*1.3, x_max*1.3, x_dim)
y = LinRange(y_min*1.3, y_max*1.3, y_dim)
X = x' .* ones(y_dim)
Y = (ones(x_dim) .* y')'

# Generate random node positions in a 2D space, e.g., within [-2, 2]
head_positions = Observable([rand(x_min:0.1:x_max, 2) for _ in 1:n_pose_graphs])
# node_positions = Observable(rand(-x_min:0.1:x_max, n_nodes, 2))
head_positions_matrix = hcat(head_positions[]...)'

node_positions_matrix = compute_global_positions(head_positions[])
node_positions = Observable([node_positions_matrix[i, :] for i in 1:size(node_positions_matrix)[1]])

# Generate random precisions for the nodes, e.g., within [0.1, 1.0]
precisions = Observable(rand(0.1:0.1:1.0, n_nodes))

Z = Observable(gaussian_mixture_surface(;
            X=X, Y=Y, precisions=precisions[],
            node_positions=node_positions[],
           ))

# construct graph
graph = init_graph!(n_pose_graphs=n_pose_graphs)
lap = laplacian_matrix(graph)

print(lap)

# Close incoming connections to fixed nodes.
if n_fixed_nodes > 0
    lap[fixed_mask,:] = zeros(n_nodes)' .* ones(n_fixed_nodes)
end

# Plot the surface using the observable Z
fig = Figure(size = (800, 600))
ax = Axis3(fig[1, 1], title = "Dynamic Gaussian Mixture Surface")
label = "Time: 0"
# Label(fig[0,:], label)
surface!(ax, x, y, Z, colormap = :viridis, alpha=0.7)

graphplot!(graph, layout=node_positions)

display(fig)

for t in 1:dt:duration
    node_positions_matrix[:,:] -= delta.*(lap * node_positions_matrix)
    node_positions_matrix[fixed_mask,:] += [a*cos(omega*t), a*sin(omega*t)]' .* ones(n_fixed_nodes)
    node_positions[] = [node_positions_matrix[i, :] for i in 1:size(node_positions_matrix, 1)]

    # Recompute surface values
    Z[] = gaussian_mixture_surface(;
            X=X, Y=Y, precisions=precisions[],
            node_positions=node_positions[],
           )

    sleep(dt)  # Control animation speed
    print("time: $t\n")
    print("pos $node_positions_matrix")
end
