using GLMakie

x = LinRange(-2, 2, 5)
y = LinRange(-2, 2, 5)
X, Y = x' .* ones(5), ones(5) .* y'
Z = sin.(X .+ Y)  # Example static surface values

fig = Figure(size = (800, 600))
ax = Axis3(fig[1, 1], title = "Static Surface")
surface!(X, Y, Z, axis=ax, colormap = :viridis)
fig

# zs = [cos(x_) * sin(y_) for x_ in x, y_ in y]

