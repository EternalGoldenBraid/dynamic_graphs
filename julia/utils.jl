using Graphs, GraphMakie

function gaussian_mixture_surface(;
        X, Y, 
        # node_positions::Vector{<:Vector{<:Real}},
        node_positions::Vector{<:Vector{<:Real}},
        precisions::Vector{<:Real},
    )
    mu_x = [pos[2] for pos in node_positions]
    mu_y = [pos[1] for pos in node_positions]
    
    sigma = 1 ./ sqrt.(precisions)

    surface = sum(
        exp.(
             -((X .- μx).^2 .+ (Y .- μy).^2) ./ (2 * σ^2)) ./ (2 * π * σ^2)
        for (μx, μy, σ) in zip(mu_x, mu_y, sigma)
    )
    
    return surface
end

function init_graph(;n_nodes)
    # g = wheel_graph(n_nodes)
    g = barabasi_albert(n_nodes, 1)
    # g = complete_graph(n_nodes);
    return g
end
