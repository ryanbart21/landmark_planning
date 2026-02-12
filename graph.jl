using Distributions
using Plots

const VISIBILITY_RAD = 20.0
const UNCERTAINTY_PER_METER = 1.0
const MARKER_PROPORTION = 50.0

struct Landmark
    x::Float64
    y::Float64
    σ::Float64 # uncertainty
end

struct LandmarkGraph
    n::Int
    landmarks::Vector{Landmark}
    distance::Matrix{Float64}
    additional_uncertainty::Matrix{Float64}
end

function generate_graph(landmarks::Vector{Landmark})
    n = length(landmarks)
    dist = zeros(n,n)
    add_unc = zeros(n,n)
    for (i, li) in enumerate(landmarks)
        for (j, lj) in enumerate(landmarks)
            dist[i,j] = sqrt((li.x-lj.x)^2 + (li.y-lj.y)^2)
            add_unc[i,j] = lj.σ + UNCERTAINTY_PER_METER * dist[i,j]
        end
    end
    return LandmarkGraph(n, landmarks, dist, add_unc)
end

function calc_edge_risk(current_σ::Float64, lj_σ::Float64, add_σ::Float64)
    combined_σ = sqrt(current_σ^2 + lj_σ^2 + add_σ^2)
    d = Normal(0.0, combined_σ)
    risk = 2*cdf(d, -VISIBILITY_RAD)
    return risk
end

function dijkstra_risk_prune(graph::LandmarkGraph, max_risk::Float64)
    n = graph.n

    # Track shortest distance to each node
    dist = fill(Inf, n)
    dist[1] = 0.0

    # Track previous node to reconstruct path
    prev = fill(-1, n)

    # Track cumulative risk along the path
    cumulative_risk = zeros(n)
    cumulative_uncertainty = zeros(n)

    # Unvisited nodes
    unvisited = Set(1:n)

    while !isempty(unvisited)
        # Pick unvisited node with minimum distance
        current_node = findmin([dist[i] for i in unvisited])[2]
        current_node = collect(unvisited)[current_node]

        # Stop if goal reached
        if current_node == n
            break
        end

        delete!(unvisited, current_node)

        # Examine neighbors
        for neighbor in unvisited
            edge_distance = graph.distance[current_node, neighbor]
            edge_add_unc = graph.additional_uncertainty[current_node, neighbor]

            # Compute risk of this edge
            edge_risk = calc_edge_risk(cumulative_uncertainty[current_node],
                                       graph.landmarks[neighbor].σ,
                                       edge_add_unc)

            new_risk = 1.0 - (1.0 - cumulative_risk[current_node]) * (1.0 - edge_risk)
            new_dist = dist[current_node] + edge_distance
            new_uncertainty = sqrt(cumulative_uncertainty[current_node]^2 + graph.landmarks[neighbor].σ^2)

            # Only consider neighbor if risk is below threshold
            if new_risk <= max_risk && new_dist < dist[neighbor]
                dist[neighbor] = new_dist
                prev[neighbor] = current_node
                cumulative_risk[neighbor] = new_risk
                cumulative_uncertainty[neighbor] = new_uncertainty
            end
        end
    end

    # Reconstruct path
    path = Int[]
    u = n
    while u != -1
        push!(path, u)
        u = prev[u]
    end
    path = reverse(path)

    # If the first element is not start, path is impossible
    if isempty(path) || path[1] != 1
        return "Impossible", Inf, cumulative_risk[end]
    else
        return path, dist[n], cumulative_risk[end]
    end
end

# Create start, 16 landmarks, end
landmarks = [
    Landmark(0.1, 0.1, 0.1),
    Landmark(2.3,  7.1, 0.1),
    Landmark(5.8,  1.4, 0.1),
    Landmark(9.2,  3.7, 0.1),
    Landmark(1.1,  8.9, 0.1),
    Landmark(6.4,  2.2, 0.1),
    Landmark(3.9,  5.6, 0.1),
    Landmark(7.7,  0.8, 0.1),
    Landmark(4.5,  9.3, 0.1),
    Landmark(8.8,  6.1, 0.1),
    Landmark(0.6,  4.4, 0.1),
    Landmark(10.2, 1.9, 0.1),
    Landmark(12.5, 7.8, 0.1),
    Landmark(14.1, 3.3, 0.1),
    Landmark(11.7, 9.0, 0.1),
    Landmark(13.4, 2.7, 0.1),
    Landmark(14.3, 5.2, 0.1),
    Landmark(14.9, 14.9, 0.0)
]

graph = generate_graph(landmarks)

x_coords = [lm.x for lm in graph.landmarks]
y_coords = [lm.y for lm in graph.landmarks]
marker_sizes = [lm.σ * MARKER_PROPORTION for lm in graph.landmarks]  # scale factor to make it visible

scatter(x_coords[2:end-1], y_coords[2:end-1], label="Landmarks", color=:red, markersize=marker_sizes[2:end-1])
scatter!([x_coords[1]], [y_coords[1]], label="Start", color=:green, markersize=marker_sizes[1])
scatter!([x_coords[end]], [y_coords[end]], label="Goal", marker=:star5, color=:orange, markersize=7)

# Dijkstra's
current_σ = graph.landmarks[1].σ
current_ind = 1

# Run Dijkstra
path, total_distance, total_risk = dijkstra_risk_prune(graph, 0.001)

println("Shortest path indices: ", path)
println("Total distance: ", total_distance)
println("Total risk: ", total_risk)

if path != "Impossible"
    # Extract coordinates along the path
    path_x = [graph.landmarks[i].x for i in path]
    path_y = [graph.landmarks[i].y for i in path]

    # Draw the path on top of the scatter plot
    plot!(path_x, path_y, label="Shortest Path", color=:blue, linewidth=2)
    savefig("myplot_with_path.png")
end