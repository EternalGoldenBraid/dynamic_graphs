using Graphs, GraphMakie
using LinearAlgebra

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

NUM_NODES_PER_POSE_GRAPH = 17

function add_pose_graph!(graph::SimpleGraph)
    """
    Function to add a pose graph connected
    in a hierarchical manner.
    """
    BODY_PARTS = [
        :head, :neck, :spine1, :spine2, :spine3,
        :left_hip, :right_hip, :left_knee, :right_knee,
        :left_ankle, :right_ankle, :left_shoulder, :right_shoulder,
        :left_elbow, :right_elbow, :left_wrist, :right_wrist
    ]

    # Offset ensures indices don't overlap with existing vertices
    offset = nv(graph)

    # Assign indices to body parts
    body_part_indices = Dict(part => offset + i for (i, part) in enumerate(BODY_PARTS))

    print("body_part_indices: $body_part_indices")

    # Add vertices for the pose graph
    add_vertices!(graph, length(BODY_PARTS))
    println("Added $(length(BODY_PARTS)) vertices. Total vertices: $(nv(graph)).")
    println("Assigned indices: $body_part_indices.")

    # Define edges using body part names
    edges = [
        (:head, :neck),
        (:neck, :spine1), (:spine1, :spine2), (:spine2, :spine3),
        (:spine3, :left_hip), (:spine3, :right_hip),
        (:left_hip, :left_knee), (:right_hip, :right_knee),
        (:left_knee, :left_ankle), (:right_knee, :right_ankle),
        (:neck, :left_shoulder), (:neck, :right_shoulder),
        (:left_shoulder, :left_elbow), (:right_shoulder, :right_elbow),
        (:left_elbow, :left_wrist), (:right_elbow, :right_wrist)
    ]

    # Add edges to the graph using mapped indices
    for (u, v) in edges
        add_edge!(graph, body_part_indices[u], body_part_indices[v])
        println("Added edge ($u, $v) -> ($(body_part_indices[u]), $(body_part_indices[v])). Total edges: $(ne(graph)).")
    end

    println("Pose graph added successfully. Total edges: $(ne(graph)).")
    return graph
end

function init_graph!(;n_pose_graphs)
    # g = barabasi_albert(n_nodes, 1)
    # g = SimpleGraph(NUM_NODES_PER_POSE_GRAPH*n_pose_graphs)
    g = SimpleGraph(0)

    for pose_graph_idx in 1:n_pose_graphs

        # g = add_pose_graph!(g)
        add_pose_graph!(g)

    end
    return g
end


# Function to define a standard T-pose in the local frame
function t_pose_offsets()
    # Define relative offsets for T-pose
    [
        (0, 0),         # Head (origin)
        (0, -1),        # Neck
        (0, -2), (0, -3), (0, -4), # Spine 1, 2, 3
        (-1, -4), (1, -4),         # Left/Right Hip
        (-1, -5), (1, -5),         # Left/Right Knee
        (-1, -6), (1, -6),         # Left/Right Ankle
        (-1, -1), (1, -1),         # Left/Right Shoulder
        (-2, -1), (2, -1),         # Left/Right Elbow
        (-3, -1), (3, -1)          # Left/Right Wrist
    ]
end

# Function to compute global positions
function compute_global_positions(head_positions::Vector{Vector{Float64}})
    # Number of pose graphs
    n = length(head_positions)
    # Number of joints in each pose graph
    offsets = t_pose_offsets()
    # k = length(offsets)
    k = NUM_NODES_PER_POSE_GRAPH

    # Allocate global positions array
    global_positions = zeros(Float64, n * k, 2)
    # global_positions = [zeros(Float64, 2) for i in 1:n*k]

    # Compute global positions for each pose graph
    for i in 1:n
        head_x, head_y = head_positions[i]
        for j in 1:k
            offset_x, offset_y = offsets[j]
            # index = (i - 1) * k + j 
            # print("Index: $index \n")
            # global_positions[index, :] .= [head_x + offset_x, head_y + offset_y]
            global_positions[(i - 1) * k + j, :] .= [head_x + offset_x, head_y + offset_y]
        end
    end

    return global_positions
end
