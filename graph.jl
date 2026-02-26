using Distributions
using Plots
using DataStructures
using LinearAlgebra

# ----------------------
# Constants & Parameters
# ----------------------
const VISIBILITY_RAD = 5.0
const DIR_UNCERTAINTY_PER_METER = 0.3
const MAJ_MIN_UNC_RATIO = 3
const PERP_UNCERTAINTY_PER_METER = DIR_UNCERTAINTY_PER_METER / MAJ_MIN_UNC_RATIO
const MARKER_PROPORTION = 50.0
const NUM_AGENTS = 3
const SENSOR_NOISE = 0.5
const COMM_RADIUS   = 5.0   # communication radius; weight tapers as Gaussian

# ----------------------
# Data Structures
# ----------------------
struct State
    node::Int
    dist::Float64
    risk::Float64
    cov::Matrix{Float64}   # fused covariance accumulated along this path
    parent::Int
end

struct SupportState
    node::Int
    dist::Float64
    risk::Float64
    cov::Matrix{Float64}   # fused covariance accumulated along this path
    cum_benefit::Float64   # sum of avg_eigenvalue(global_cov[node]) for primary-path nodes visited
    parent::Int
end

struct Landmark
    x::Float64
    y::Float64
    cov::Matrix{Float64}
end

struct LandmarkGraph
    n::Int
    landmarks::Vector{Landmark}
    distance::Matrix{Float64}
    orientation::Matrix{Float64}
end

function generate_graph(landmarks::Vector{Landmark})
    n = length(landmarks)
    dist   = zeros(n, n)
    orient = zeros(n, n)
    for (i, li) in enumerate(landmarks)
        for (j, lj) in enumerate(landmarks)
            dx = lj.x - li.x
            dy = lj.y - li.y
            dist[i,j]   = sqrt(dx^2 + dy^2)
            orient[i,j] = atan(dy, dx)
        end
    end
    return LandmarkGraph(n, landmarks, dist, orient)
end

function max_eigenvalue(cov::Matrix{Float64})
    return maximum(real(eigvals(Symmetric((cov + cov') / 2))))
end

avg_eigenvalue(cov::Matrix{Float64}) = tr(cov) / 2.0

function growth_covariance(distance::Float64, angle::Float64)
    σ_dir  = DIR_UNCERTAINTY_PER_METER  * distance
    σ_perp = PERP_UNCERTAINTY_PER_METER * distance
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    return R * Diagonal([σ_dir^2, σ_perp^2]) * R'
end

function calc_edge_risk(current_cov::Matrix{Float64}, lj_cov::Matrix{Float64},
                        edge_distance::Float64, dir_angle::Float64)
    predicted_cov = current_cov + growth_covariance(edge_distance, dir_angle) + lj_cov
    σ    = sqrt(max_eigenvalue(predicted_cov))
    risk = exp(-VISIBILITY_RAD^2 / (2σ^2))
    return risk, predicted_cov
end

# Fuse current path covariance with the landmark measurement at node u.
# Kalman-style update: new_cov = (path_cov^-1 + (P_L + R_s)^-1)^-1
function fuse_cov(path_cov::Matrix{Float64}, landmark_cov::Matrix{Float64})
    R_s = SENSOR_NOISE^2 * I(2)
    return inv(inv(path_cov) + inv(landmark_cov + R_s))
end

# ---------------------------------------------------------------------------
# Primary-agent search
#
# Minimises total path distance subject to cumulative risk <= threshold.
# `dist_cap` prevents the re-plan from being longer than the provisional path.
# ---------------------------------------------------------------------------
function search_main_agent!(graph::LandmarkGraph,
                             global_cov::Vector{Matrix{Float64}},
                             threshold::Float64;
                             use_heuristic::Bool = false,
                             update_global::Bool  = true,
                             dist_cap::Float64    = Inf)
    n = graph.n
    states      = State[]
    node_states = [Int[] for _ in 1:n]
    pq          = PriorityQueue{Int, Float64}()

    push!(states, State(1, 0.0, 0.0, copy(global_cov[1]), -1))
    push!(node_states[1], 1)
    enqueue!(pq, 1, 0.0)

    heur(v) = use_heuristic ? hypot(graph.landmarks[n].x - graph.landmarks[v].x,
                                    graph.landmarks[n].y - graph.landmarks[v].y) : 0.0

    best_goal_dist = Inf
    goal_state     = 0

    while !isempty(pq)
        si = dequeue!(pq)
        S  = states[si]
        v, d, r, cov = S.node, S.dist, S.risk, S.cov
        si ∉ node_states[v] && continue

        if v == n
            if d < best_goal_dist
                best_goal_dist = d
                goal_state     = si
            end
            continue
        end

        for u in 1:n
            u == v && continue
            # cycle check
            anc = si; cycle = false
            while anc != -1
                states[anc].node == u && (cycle = true; break)
                anc = states[anc].parent
            end
            cycle && continue

            edge_dist = graph.distance[v, u]
            isfinite(edge_dist) || continue
            angle = graph.orientation[v, u]

            new_dist = d + edge_dist
            (new_dist > best_goal_dist || new_dist > dist_cap) && continue

            edge_risk, _ = calc_edge_risk(cov, global_cov[u], edge_dist, angle)
            new_risk = 1.0 - (1.0 - r) * (1.0 - edge_risk)
            new_risk > threshold && continue

            # Correct sequential fusion: fuse path covariance with landmark measurement
            new_cov = fuse_cov(cov, global_cov[u])

            # Pareto dominance on (dist, risk, cov_metric)
            new_metric = avg_eigenvalue(new_cov)
            dominated  = false
            to_remove  = Int[]
            for old_si in node_states[u]
                old = states[old_si]
                old_metric = avg_eigenvalue(old.cov)
                if old.dist <= new_dist && old.risk <= new_risk && old_metric <= new_metric
                    dominated = true; break
                end
                if new_dist <= old.dist && new_risk <= old.risk && new_metric <= old_metric
                    push!(to_remove, old_si)
                end
            end
            dominated && continue
            for rem in to_remove
                deleteat!(node_states[u], findfirst(==(rem), node_states[u]))
            end

            push!(states, State(u, new_dist, new_risk, new_cov, si))
            new_si = length(states)
            push!(node_states[u], new_si)
            enqueue!(pq, new_si, new_dist + heur(u))
        end

        !isempty(pq) && peek(pq)[2] >= best_goal_dist && break
    end

    goal_state == 0 && return Int[], Inf, NaN

    path = Int[]; si = goal_state
    while si != -1
        push!(path, states[si].node)
        si = states[si].parent
    end
    reverse!(path)
    final = states[goal_state]

    if update_global
        for node in path
            global_cov[node] = fuse_cov(global_cov[node], final.cov)
        end
    end

    return path, final.dist, final.risk
end

# ---------------------------------------------------------------------------
# Support-agent search
#
# Travels up to dist_cap and maximises sum of avg_eigenvalue(global_cov[node])
# for nodes on primary_path.  Because global_cov is updated after each agent,
# agent k+1 sees residual uncertainty and naturally takes a different route.
#
# No risk constraint — Pareto dominance is on (dist, benefit) only so diverse
# paths are not pruned away.
# ---------------------------------------------------------------------------
function search_support_agent!(graph::LandmarkGraph,
                                global_cov::Vector{Matrix{Float64}},
                                dist_cap::Float64,
                                primary_path::Vector{Int};
                                use_heuristic::Bool = false)
    n = graph.n

    # Pre-compute communication weight for each node u relative to the primary
    # path.  The weight is the maximum Gaussian taper over all primary-path
    # nodes: w(u) = max_j exp(-d(u,j)^2 / (2*COMM_RADIUS^2)).
    # This means a support agent at u can communicate well with the primary
    # agent at any nearby primary-path node, and benefit tapers smoothly to
    # zero beyond COMM_RADIUS.  Nodes on the primary path itself score w=1.
    comm_weight = zeros(n)
    for u in 1:n
        for pnode in primary_path
            d_comm = graph.distance[u, pnode]
            w = exp(-d_comm^2 / (2 * COMM_RADIUS^2))
            comm_weight[u] = max(comm_weight[u], w)
        end
    end

    states      = SupportState[]
    node_states = [Int[] for _ in 1:n]
    pq          = PriorityQueue{Int, Float64}()

    init_benefit = comm_weight[1] * avg_eigenvalue(global_cov[1])
    push!(states, SupportState(1, 0.0, 0.0, copy(global_cov[1]), init_benefit, -1))
    push!(node_states[1], 1)
    enqueue!(pq, 1, -init_benefit)

    best_benefit = -Inf
    goal_state   = 0

    while !isempty(pq)
        si = dequeue!(pq)
        S  = states[si]
        v, d, r, cov, benefit = S.node, S.dist, S.risk, S.cov, S.cum_benefit
        si ∉ node_states[v] && continue

        # Any node within the cap (not start) is a valid terminal
        if d > 0 && d <= dist_cap && benefit > best_benefit
            best_benefit = benefit
            goal_state   = si
        end

        for u in 1:n
            u == v && continue
            anc = si; cycle = false
            while anc != -1
                states[anc].node == u && (cycle = true; break)
                anc = states[anc].parent
            end
            cycle && continue

            edge_dist = graph.distance[v, u]
            isfinite(edge_dist) || continue

            new_dist = d + edge_dist
            new_dist > dist_cap && continue

            angle    = graph.orientation[v, u]
            edge_risk, _ = calc_edge_risk(cov, global_cov[u], edge_dist, angle)
            new_risk = 1.0 - (1.0 - r) * (1.0 - edge_risk)

            # Rollout benefit: how much does visiting u actually reduce avg uncertainty
            # along *this specific path*, given all prior fusions?  Computed as the
            # drop in avg_eigenvalue before vs after fusing with the landmark at u,
            # weighted by comm_weight so only nodes near the primary path count.
            # This is a true rollout — not a heuristic — with diminishing returns
            # built in: a node already well-observed from prior path fusions gives
            # near-zero benefit even if global_cov[u] is still high.
            new_cov     = fuse_cov(cov, global_cov[u])
            uncertainty_before = avg_eigenvalue(cov)
            uncertainty_after  = avg_eigenvalue(new_cov)
            new_benefit = benefit + comm_weight[u] * max(0.0, uncertainty_before - uncertainty_after)

            # Pareto dominance on (dist, benefit) only — no risk constraint
            dominated = false
            to_remove = Int[]
            for old_si in node_states[u]
                old = states[old_si]
                if old.dist <= new_dist && old.cum_benefit >= new_benefit
                    dominated = true; break
                end
                if new_dist <= old.dist && new_benefit >= old.cum_benefit
                    push!(to_remove, old_si)
                end
            end
            dominated && continue
            for rem in to_remove
                deleteat!(node_states[u], findfirst(==(rem), node_states[u]))
            end

            push!(states, SupportState(u, new_dist, new_risk, new_cov, new_benefit, si))
            new_si = length(states)
            push!(node_states[u], new_si)
            # Maximise benefit; tie-break by shorter distance
            enqueue!(pq, new_si, -new_benefit + 1e-9 * new_dist)
        end
    end

    goal_state == 0 && return Int[], Inf, NaN

    path = Int[]; si = goal_state
    while si != -1
        push!(path, states[si].node)
        si = states[si].parent
    end
    reverse!(path)
    final = states[goal_state]

    # Update global_cov so the next agent and the primary agent see residual
    # uncertainty.  A measurement at node `node` also reduces uncertainty at
    # spatially nearby nodes via spatial correlation, modelled as a Gaussian
    # decay with COMM_RADIUS.  This captures cross-node benefit: the primary
    # agent visiting a neighbour of a support-agent node gets partial credit.
    for node in path
        for v in 1:graph.n
            spatial_w = exp(-graph.distance[node, v]^2 / (2 * COMM_RADIUS^2))
            spatial_w < 1e-4 && continue   # skip negligible contributions
            # Blend: interpolate between current global_cov[v] and the fused value
            # proportional to spatial_w (full fusion only at the visited node itself)
            fused = fuse_cov(global_cov[v], final.cov)
            global_cov[v] = (1.0 - spatial_w) .* global_cov[v] .+ spatial_w .* fused
        end
    end

    return path, final.dist, final.risk
end

# ---------------------------------------------------------------------------
# Multi-agent RCSP orchestration
# Phase 1: Preliminary primary search (no global_cov update) → provisional path + dist
# Phase 2: Support agents sequentially reduce global_cov at primary-path nodes
# Phase 3: Primary agent replans with updated global_cov, capped at provisional_dist
# ---------------------------------------------------------------------------
function multi_agent_rcsp(graph::LandmarkGraph,
                          risk_threshold::Float64,
                          num_ag::Int = 1)

    n = graph.n
    global_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    paths = Vector{Int}[]
    dists = Float64[]
    risks = Float64[]

    # Phase 1
    provisional_path, provisional_dist, provisional_risk =
        search_main_agent!(graph, global_cov, risk_threshold;
                           use_heuristic=true, update_global=false)

    if isempty(provisional_path)
        println("No feasible path found in preliminary search.")
        return paths, dists, risks, global_cov, Int[], Inf, NaN
    end

    # Phase 2
    for _ in 2:num_ag
        path, d, r = search_support_agent!(graph, global_cov,
                                            provisional_dist, provisional_path;
                                            use_heuristic=false)
        push!(paths, path)
        push!(dists, isempty(path) ? Inf : d)
        push!(risks, isempty(path) ? NaN : r)
    end

    # Phase 3: primary agent replans on the updated global_cov (support agents
    # have reduced uncertainty at and near provisional-path nodes).  No dist_cap
    # here — we want the shortest path that satisfies the risk threshold, which
    # may now be shorter than provisional_dist because previously-risky shortcuts
    # are now feasible thanks to the support agents' covariance reductions.
    main_path, main_d, main_r =
        search_main_agent!(graph, global_cov, risk_threshold;
                           use_heuristic=true, update_global=true)
    push!(paths, main_path)
    push!(dists, main_d)
    push!(risks, main_r)

    return paths, dists, risks, global_cov, provisional_path, provisional_dist, provisional_risk
end

# ----------------------
# Landmark graph setup
# ----------------------
landmarks = [
    Landmark(0.1,  0.1,  [0.1 0.0; 0.0 0.1]),
    Landmark(2.3,  7.1,  [0.47 0.1; 0.1 0.22]),
    Landmark(5.8,  1.4,  [0.21 -0.05; -0.05 0.12]),
    Landmark(9.2,  3.7,  [0.33 0.0; 0.0 0.08]),
    Landmark(1.1,  8.9,  [0.05 0.02; 0.02 0.03]),
    Landmark(6.4,  2.2,  [0.50 0.0; 0.0 0.50]),
    Landmark(3.9,  5.6,  [0.12 -0.02; -0.02 0.06]),
    Landmark(7.7,  0.8,  [0.38 0.1; 0.1 0.16]),
    Landmark(4.5,  9.3,  [0.29 -0.05; -0.05 0.1]),
    Landmark(8.8,  6.1,  [0.08 0.0; 0.0 0.02]),
    Landmark(0.6,  4.4,  [0.19 0.05; 0.05 0.09]),
    Landmark(10.2, 1.9,  [0.41 -0.1; -0.1 0.2]),
    Landmark(12.5, 7.8,  [0.22 0.0; 0.0 0.22]),
    Landmark(14.1, 3.3,  [0.49 0.1; 0.1 0.3]),
    Landmark(11.7, 9.0,  [0.31 0.0; 0.0 0.15]),
    Landmark(13.4, 2.7,  [0.07 0.02; 0.02 0.04]),
    Landmark(16.0, 5.2,  [0.25 -0.05; -0.05 0.2]),
    Landmark(14.9, 14.9, 1e-9 * [1.0 0.0; 0.0 1.0])   # goal: tiny cov so fuse_cov stays invertible
]

graph = generate_graph(landmarks)

x_coords     = [lm.x for lm in graph.landmarks]
y_coords     = [lm.y for lm in graph.landmarks]
marker_sizes = [sqrt(max_eigenvalue(lm.cov)) * MARKER_PROPORTION for lm in graph.landmarks]

function draw_covariance_ellipse!(plt, x, y, cov; npts=50, nstd=2, color=:red, alpha=0.3)
    vals, vecs = eigen(Symmetric((cov+cov')/2))
    a     = nstd * sqrt(max(vals[1], 0.0))
    b     = nstd * sqrt(max(vals[2], 0.0))
    angle = atan(vecs[2,1], vecs[1,1])
    θ     = range(0, 2π, length=npts)
    R     = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    pts   = R * vcat((a .* cos.(θ))', (b .* sin.(θ))')
    plot!(plt, x .+ pts[1,:], y .+ pts[2,:], seriestype=:shape,
          color=color, alpha=alpha, label=false)
end

plt = scatter(x_coords[2:end-1], y_coords[2:end-1], label=false, color=:black, markersize=1)
for i in 2:length(landmarks)-1
    draw_covariance_ellipse!(plt, landmarks[i].x, landmarks[i].y, landmarks[i].cov,
                             color=:red, alpha=0.3)
end
scatter!(plt, [x_coords[1]],   [y_coords[1]],   label="Start", color=:green,  markersize=marker_sizes[1])
scatter!(plt, [x_coords[end]], [y_coords[end]],  label="Goal",  marker=:star5, color=:orange, markersize=7)

# ----------------------
# Run
# ----------------------
risk_threshold = 0.1
paths, dists, risks, _, provisional_path, provisional_dist, provisional_risk =
    multi_agent_rcsp(graph, risk_threshold, NUM_AGENTS)

println("Provisional primary path : ", provisional_path,
        "  dist=", round(provisional_dist, digits=3),
        "  risk=", round(provisional_risk, digits=4))

println("\n--- Final Results ---")
for (i, path) in enumerate(paths[1:end-1])
    if isempty(path)
        println("Support agent $i : no path found")
    else
        println("Support agent $i path : ", path,
                "  dist=", round(dists[i], digits=3),
                "  risk=", round(risks[i], digits=4))
    end
end
if isempty(paths[end])
    println("Primary agent : no path found")
else
    println("Primary agent path : ", paths[end],
            "  dist=", round(dists[end], digits=3),
            "  risk=", round(risks[end], digits=4))
end

# ----------------------
# Plot
# ----------------------
agent_colors = [:purple, :teal, :darkorange, :crimson]

for (i, path) in enumerate(paths)
    isempty(path) && continue
    path_x = [graph.landmarks[idx].x for idx in path]
    path_y = [graph.landmarks[idx].y for idx in path]
    if i == length(paths)
        plot!(plt, path_x, path_y, label="Primary agent", color=:blue, linewidth=2)
    else
        clr = i <= length(agent_colors) ? agent_colors[i] : :gray
        plot!(plt, path_x, path_y, label="Support agent $i",
              color=clr, linewidth=1, linestyle=:dash)
    end
end

xlabel!(plt, "x"); ylabel!(plt, "y")
title!(plt, "Multi-Agent Paths (comm-aware support)")
savefig(plt, "multi_agent_paths.png")
println("\nPlot saved to multi_agent_paths.png")