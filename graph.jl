using Distributions
using Plots
using DataStructures
using LinearAlgebra

const VISIBILITY_RAD = 5.0
const DIR_UNCERTAINTY_PER_METER = 1.0
const PERP_UNCERTAINTY_PER_METER = 0.3
const MARKER_PROPORTION = 50.0
const NUM_AGENTS = 3
const SENSOR_NOISE = 0.5

struct State
    node::Int
    dist::Float64
    risk::Float64
    cov::Matrix{Float64}   # current covariance
    parent::Int            # index of parent state
end

struct Landmark
    x::Float64
    y::Float64
    cov::Matrix{Float64}    # 2×2 covariance matrix
end

struct LandmarkGraph
    n::Int
    landmarks::Vector{Landmark}
    distance::Matrix{Float64}
    orientation::Matrix{Float64}              # direction angle from i to j
end

function generate_graph(landmarks::Vector{Landmark})
    n = length(landmarks)
    dist = zeros(n,n)
    orient = zeros(n,n)
    for (i, li) in enumerate(landmarks)
        for (j, lj) in enumerate(landmarks)
            len_x = lj.x - li.x        # direction i->j
            len_y = lj.y - li.y
            dist[i,j] = sqrt((len_x)^2 + (len_y)^2)
            orient[i,j] = atan(len_y, len_x)  # two-argument atan serves as atan2 in Julia
        end
    end
    return LandmarkGraph(n, landmarks, dist, orient)
end

# return the largest eigenvalue of a (real) symmetric 2×2 matrix
function max_eigenvalue(cov::Matrix{Float64})
    # ensure symmetry and real values
    cov_sym = Symmetric((cov + cov')/2)
    vals = eigvals(cov_sym)
    return maximum(real(vals))
end

# rotate a covariance matrix by angle
function rotate_cov(cov::Matrix{Float64}, angle::Float64)
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    return R * cov * R'
end

# Construct anisotropic growth covariance along direction angle
function growth_covariance(distance::Float64, angle::Float64)
    # diagonal in edge frame: major=DIR, minor=PERP
    σ_dir = DIR_UNCERTAINTY_PER_METER * distance
    σ_perp = PERP_UNCERTAINTY_PER_METER * distance
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    return R * Diagonal([σ_dir, σ_perp]) * R'
end

# Predict edge risk using convolved covariance + directional growth
function calc_edge_risk(current_cov::Matrix{Float64}, lj_cov::Matrix{Float64}, edge_distance::Float64, dir_angle::Float64)
    growth_cov = growth_covariance(edge_distance, dir_angle)
    # account for multiple agents by scaling new contributions
    predicted_cov = current_cov + growth_cov + lj_cov
    σ = sqrt(max_eigenvalue(predicted_cov))
    risk = exp(-VISIBILITY_RAD^2 / (2σ^2))
    return risk, predicted_cov
end

@enum Objective MinDistance MinRisk

function dijkstra_rcsp(graph::LandmarkGraph,
                       threshold::Float64,
                       objective::Objective,
                       num_ag::Int = 1)

    n = graph.n

    states = State[]
    node_states = [Int[] for _ in 1:n]
    pq = PriorityQueue{Int, Float64}()

    # initial covariance is the start landmark covariance
    start_cov = graph.landmarks[1].cov
    start = State(1, 0.0, 0.0, start_cov, -1)

    push!(states, start)
    push!(node_states[1], 1)
    enqueue!(pq, 1, 0.0)

    goal_state = 0
    best_goal_value = Inf

    while !isempty(pq)
        si = dequeue!(pq)
        S = states[si]

        v   = S.node
        d   = S.dist
        r   = S.risk
        cov = S.cov

        if !(si in node_states[v])
            continue
        end

        if v == n
            value = objective == MinDistance ? d : r
            if value < best_goal_value
                best_goal_value = value
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

            angle = graph.orientation[v,u]
            edge_risk, combined_cov = calc_edge_risk(
                cov,
                graph.landmarks[u].cov,
                edge_distance,
                angle
            )

            new_risk = 1.0 - (1.0 - r) * (1.0 - edge_risk)
            new_dist = d + edge_distance

            # pruning based on objective
            if objective == MinDistance
                if new_risk > threshold
                    continue
                end
            else
                if new_dist > threshold
                    continue
                end
            end

            P_pred = combined_cov

            θ = angle
            H = [cos(θ) sin(θ)]          # 1×2

            # Measurement noise (sensor) as scalar
            R_sensor = SENSOR_NOISE^2

            # Landmark uncertainty projected along line of sight
            P_L = graph.landmarks[u].cov

            # Wrap R_eff as a 1×1 matrix to avoid scalar/matrix addition issues
            R_eff = R_sensor + (H * P_L * H')[1]
            
            # Probability of successful observation
            p = 1.0 - edge_risk

            # Multi-agent information fusion
            Λ = inv(P_pred)                    # start with predicted covariance
            for agent_id in 1:num_ag
                Λ += p * (H' * inv(R_eff) * H)  # contribution of each agent
            end

            # Updated covariance after multi-agent fusion
            new_cov = inv(Λ)

            # dominance check using largest eigenvalue
            dominated = false
            to_remove = Int[]
            new_metric = max_eigenvalue(new_cov)
            for old_si in node_states[u]
                old = states[old_si]
                old_metric = max_eigenvalue(old.cov)
                if old.dist ≤ new_dist &&
                   old.risk ≤ new_risk &&
                   old_metric ≤ new_metric
                    dominated = true
                    break
                end
                if new_dist ≤ old.dist &&
                   new_risk ≤ old.risk &&
                   new_metric ≤ old_metric
                    push!(to_remove, old_si)
                end
            end

            if dominated
                continue
            end
            for rem in to_remove
                deleteat!(node_states[u],
                          findfirst(==(rem), node_states[u]))
            end

            push!(states,
                  State(u, new_dist,
                        new_risk,
                        new_cov, si))
            new_si = length(states)
            push!(node_states[u], new_si)
            priority = objective == MinDistance ? new_dist : new_risk
            enqueue!(pq, new_si, priority)
        end

        if !isempty(pq) && peek(pq)[2] >= best_goal_value
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
# covariances hardcoded with random elliptical shapes
landmarks = [
    Landmark(0.1, 0.1, [0.1 0.0; 0.0 0.1]),
    Landmark(2.3,  7.1, [0.47 0.1; 0.1 0.22]),
    Landmark(5.8,  1.4, [0.21 -0.05; -0.05 0.12]),
    Landmark(9.2,  3.7, [0.33 0.0; 0.0 0.08]),
    Landmark(1.1,  8.9, [0.05 0.02; 0.02 0.03]),
    Landmark(6.4,  2.2, [0.50 0.0; 0.0 0.50]),
    Landmark(3.9,  5.6, [0.12 -0.02; -0.02 0.06]),
    Landmark(7.7,  0.8, [0.38 0.1; 0.1 0.16]),
    Landmark(4.5,  9.3, [0.29 -0.05; -0.05 0.1]),
    Landmark(8.8,  6.1, [0.08 0.0; 0.0 0.02]),
    Landmark(0.6,  4.4, [0.19 0.05; 0.05 0.09]),
    Landmark(10.2, 1.9, [0.41 -0.1; -0.1 0.2]),
    Landmark(12.5, 7.8, [0.22 0.0; 0.0 0.22]),
    Landmark(14.1, 3.3, [0.49 0.1; 0.1 0.3]),
    Landmark(11.7, 9.0, [0.31 0.0; 0.0 0.15]),
    Landmark(13.4, 2.7, [0.07 0.02; 0.02 0.04]),
    Landmark(14.3, 5.2, [0.25 -0.05; -0.05 0.2]),
    Landmark(14.9, 14.9, zeros(2,2))   # goal
]

graph = generate_graph(landmarks)

x_coords = [lm.x for lm in graph.landmarks]
y_coords = [lm.y for lm in graph.landmarks]
# marker size based on largest eigenvalue of covariance
marker_sizes = [sqrt(max_eigenvalue(lm.cov)) * MARKER_PROPORTION for lm in graph.landmarks]

# helper to draw ellipse given covariance and center
function draw_covariance_ellipse!(plt, x, y, cov; npts=50, nstd=2, color=:red, alpha=0.3)
    vals, vecs = eigen(cov)
    a = nstd * sqrt(vals[1])
    b = nstd * sqrt(vals[2])
    angle = atan(vecs[2,1], vecs[1,1])
    θ = range(0, 2π, length=npts)
    xs = a * cos.(θ)
    ys = b * sin.(θ)
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    pts = R * vcat(xs', ys')   # 2×npts matrix

    # filled polygon instead of line
    plot!(plt, x .+ pts[1,:], y .+ pts[2,:], seriestype=:shape,
          color=color, alpha=alpha, label=false)
end

plt = scatter(x_coords[2:end-1], y_coords[2:end-1], label=false, color=:black, markersize=1)
# draw covariance ellipses for landmarks
for i in 2:length(landmarks)-1
    draw_covariance_ellipse!(plt, landmarks[i].x, landmarks[i].y, landmarks[i].cov,
                             color=:red, alpha=0.3)
end
scatter!(plt, [x_coords[1]], [y_coords[1]], label="Start", color=:green, markersize=marker_sizes[1])
scatter!(plt, [x_coords[end]], [y_coords[end]], label="Goal", marker=:star5, color=:orange, markersize=7)

# Run a Dijkstra RCSP for all agents (min-distance)
path, total_distance, total_risk = dijkstra_rcsp(graph, 0.5, MinDistance, NUM_AGENTS)

println("Shortest path indices: ", path)
println("Total distance: ", total_distance)
println("Total risk: ", total_risk)

if path != "Impossible"
    # Extract coordinates along the path
    path_x = [graph.landmarks[i].x for i in path]
    path_y = [graph.landmarks[i].y for i in path]

    plot!(path_x, path_y, label="Shortest Path", color=:blue, linewidth=2)
    savefig("min_dist.png")
end