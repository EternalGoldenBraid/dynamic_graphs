using Random
using GLMakie
using Graphs, GraphMakie
set_theme!(theme_dark())

include("utils.jl")

x_dim = 300
y_dim = 300
n_nodes = 5

# Define parameters
num = 20
x_min = -num
x_max = num
y_min = -num
y_max = num
x = LinRange(x_min, x_max, x_dim)
y = LinRange(y_min, y_max, y_dim)
# X, Y = meshgrid(x, y)
X = x' .* ones(y_dim)
Y = (ones(x_dim) .* y')'

# Create observables for the Z values
# Initialize node positions and precisions

# Generate random node positions in a 2D space, e.g., within [-2, 2]
# node_positions = Observable(rand(x_min:0.1:x_max, 2) for _ in 1:n_nodes])
node_positions = Observable([rand(x_min:0.1:x_max, 2) for _ in 1:n_nodes])

# Generate random precisions for the nodes, e.g., within [0.1, 1.0]
precisions = Observable(rand(0.1:0.1:1.0, n_nodes))

Z = Observable(gaussian_mixture_surface(;
            X=X, Y=Y, precisions=precisions[],
            node_positions=node_positions[],
           ))
# print("Z: $(Z[])")
# Z = gussian_mixture_surface(X, Y, node_positions, precisions)

graph = init_graph(n_nodes=n_nodes)


# Plot the surface using the observable Z
fig = Figure(size = (800, 600))
ax = Axis3(fig[1, 1], title = "Dynamic Gaussian Mixture Surface")
# label = "Time: 0"
# Label(fig[0,:], label)
# surface!(ax, X, Y, Z, colormap = :viridis)
surface!(ax, x, y, Z, colormap = :viridis, alpha=0.7)
# surface!(ax, x, y, lift(Z) => :z)

graphplot!(graph, layout=node_positions)

# Print vertical bars at every node
# vlines([x_coord for (x_coord, y_coord) in node_positions[]])

display(fig)

# # Animation loop
dt = 0.1
duration = 10
for t in 1:dt:duration
    # Update precisions dynamically
    # precisions[] .= rand(length(precisions[])) .+ 0.1
    node_positions[] = [node_positions[][i] .+ sin(i*t) for i in 1:length(node_positions[])]
    # Recompute surface values
    Z[] = gaussian_mixture_surface(;
            X=X, Y=Y, precisions=precisions[],
            node_positions=node_positions[],
           )

    sleep(dt)  # Control animation speed
    print("time: $t\n")
end

