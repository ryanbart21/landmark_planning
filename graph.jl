using Distributions
using Plots
using DataStructures

const VISIBILITY_RAD = 20.0
const UNCERTAINTY_PER_METER = 1.0
const MARKER_PROPORTION = 50.0

struct State
    node::Int
    dist::Float64
    risk::Float64
    unc::Float64
    parent::Int   # index of parent state
end

struct Landmark
    x::Float64
    y::Float64
    σ::Float64
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

function dijkstra_risk_prune_rcsp(graph::LandmarkGraph,
                                  max_risk::Float64)

    n = graph.n

    states = State[]
    node_states = [Int[] for _ in 1:n]

    pq = PriorityQueue{Int, Float64}()

    start = State(1, 0.0, 0.0,
                  graph.landmarks[1].σ, -1)

    push!(states, start)
    push!(node_states[1], 1)
    enqueue!(pq, 1, 0.0)

    goal_state = 0
    best_goal_dist = Inf

    while !isempty(pq)

        si = dequeue!(pq)
        S = states[si]

        v   = S.node
        d   = S.dist
        r   = S.risk
        unc = S.unc

        # Skip dominated states still in queue
        if !(si in node_states[v])
            continue
        end

        # Goal handling (do NOT break yet)
        if v == n
            if d < best_goal_dist
                best_goal_dist = d
                goal_state = si
            end
            continue
        end

        for u in 1:n

            if u == v
                continue
            end

            edge_distance = graph.distance[v, u]
            if !isfinite(edge_distance)
                continue
            end

            edge_add_unc = graph.additional_uncertainty[v, u]

            edge_risk = calc_edge_risk(
                unc,
                graph.landmarks[u].σ,
                edge_add_unc
            )

            new_risk = 1.0 - (1.0 - r) * (1.0 - edge_risk)

            if new_risk > max_risk
                continue
            end

            new_dist = d + edge_distance

            new_unc = sqrt(
                unc^2 +
                (1.0 - edge_risk) *
                graph.landmarks[u].σ^2
            )

            dominated = false
            to_remove = Int[]

            for old_si in node_states[u]
                old = states[old_si]

                if old.dist ≤ new_dist &&
                   old.risk ≤ new_risk &&
                   old.unc  ≤ new_unc
                    dominated = true
                    break
                end

                if new_dist ≤ old.dist &&
                   new_risk ≤ old.risk &&
                   new_unc  ≤ old.unc
                    push!(to_remove, old_si)
                end
            end

            if dominated
                continue
            end

            for rem in to_remove
                deleteat!(node_states[u],
                          findfirst(==(rem),
                                    node_states[u]))
            end

            push!(states,
                  State(u, new_dist,
                        new_risk,
                        new_unc, si))

            new_si = length(states)

            push!(node_states[u], new_si)
            enqueue!(pq, new_si, new_dist)
        end

        # Safe termination condition
        if !isempty(pq) && peek(pq)[2] >= best_goal_dist
            break
        end
    end

    if goal_state == 0
        return "Impossible", Inf, NaN
    end

    path = Int[]
    si = goal_state

    while si != -1
        push!(path, states[si].node)
        si = states[si].parent
    end

    reverse!(path)

    final = states[goal_state]

    return path, final.dist, final.risk
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
path, total_distance, total_risk = dijkstra_risk_prune_rcsp(graph, 0.004)

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