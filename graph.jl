using Distributions
using Plots
using DataStructures
using LinearAlgebra

# ----------------------
# Constants & Parameters
# ----------------------
risk_threshold = 0.24

const VISIBILITY_RAD = 5.0
const DIR_UNCERTAINTY_PER_METER = 0.3
const MAJ_MIN_UNC_RATIO = 3
const PERP_UNCERTAINTY_PER_METER = DIR_UNCERTAINTY_PER_METER / MAJ_MIN_UNC_RATIO
const MARKER_PROPORTION = 50.0
const NUM_AGENTS = 3
const SENSOR_NOISE = 0.5
const COMM_RADIUS   = 10.0   # communication radius; weight tapers as Gaussian

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
# Shortest-path search (no risk constraint) — pure Dijkstra.
# Returns the minimum-distance path from node 1 to node n regardless of risk.
# Used to initialise the provisional path so support agents plan around the
# ideal direct route rather than an already-cautious risk-constrained detour.
# ---------------------------------------------------------------------------
function search_shortest_path(graph::LandmarkGraph)
    n      = graph.n
    dist   = fill(Inf, n)
    parent = fill(-1,  n)
    dist[1] = 0.0
    pq = PriorityQueue{Int, Float64}()
    enqueue!(pq, 1, 0.0)

    while !isempty(pq)
        v = dequeue!(pq)
        for u in 1:n
            u == v && continue
            d = graph.distance[v, u]
            isfinite(d) || continue
            nd = dist[v] + d
            if nd < dist[u]
                dist[u]   = nd
                parent[u] = v
                pq[u]     = nd   # insert or decrease-key
            end
        end
    end

    isinf(dist[n]) && return Int[], Inf

    path = Int[]
    v = n
    while v != -1
        push!(path, v)
        v = parent[v]
    end
    reverse!(path)
    return path, dist[n]
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
# Objective: maximise uncertainty reduction for the primary agent, accounting
# for BOTH spatial proximity (comm radius) AND temporal ordering.
#
# Timing model: all agents travel at the same speed, so time ∝ distance.
# A support-agent observation at node u (reached at distance d_sup) can be
# communicated to the primary agent before it reaches primary-path node k
# (reached at d_prim[k]) iff d_sup <= d_prim[k].  This is a hard gate —
# being early is always fine; there is no penalty for large time slack.
# The observation does NOT need to be at or near the primary path — it just
# needs to be spatially informative about node k (captured by the spatial
# Gaussian exp(-dist(u,k)^2/2σ^2)) and transmitted in time.
#
# Benefit is a rollout (not a heuristic): the actual drop in avg_eigenvalue
# from fusing with the visited landmark, summed over all primary-path nodes
# the observation can still reach in time.
# ---------------------------------------------------------------------------
function search_support_agent!(graph::LandmarkGraph,
                                global_cov::Vector{Matrix{Float64}},
                                dist_cap::Float64,
                                primary_path::Vector{Int};
                                use_heuristic::Bool = false)
    n = graph.n

    # Pre-compute cumulative distance of the primary agent along its path.
    # d_prim[k] = distance the primary agent has traveled when it reaches
    # primary_path[k].  Index into this vector by primary-path position.
    np = length(primary_path)
    d_prim = zeros(np)
    for k in 2:np
        d_prim[k] = d_prim[k-1] + graph.distance[primary_path[k-1], primary_path[k]]
    end

    # joint_weight(u, d_sup):
    #   A support agent at node u, having traveled d_sup, can communicate an
    #   observation of u to the primary agent before it reaches primary-path
    #   node k iff d_sup <= d_prim[k] (hard temporal gate — being early is
    #   always fine, being late means the primary has already passed).
    #   The spatial weight exp(-dist(u,k)^2 / 2σ^2) captures how informative
    #   an observation at u is about the uncertainty at primary-path node k
    #   (nearby landmarks are more correlated).
    #   We sum over all primary-path nodes k that can still be helped, so the
    #   support agent gets credit for reducing uncertainty at multiple nodes.
    function joint_weight(u::Int, d_sup::Float64)
        w_total = 0.0
        for k in 1:np
            d_sup > d_prim[k] && continue   # too late — primary already passed k
            d_spatial = graph.distance[u, primary_path[k]]
            w_spatial = exp(-d_spatial^2 / (2 * COMM_RADIUS^2))
            w_total += w_spatial
        end
        return w_total
    end

    states      = SupportState[]
    node_states = [Int[] for _ in 1:n]
    pq          = PriorityQueue{Int, Float64}()

    init_benefit = joint_weight(1, 0.0) * avg_eigenvalue(global_cov[1])
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

            # Rollout benefit weighted by joint spatio-temporal comm weight.
            # joint_weight(u, new_dist) is nonzero only if the support agent
            # arrives at u early enough to inform the primary agent at some
            # nearby primary-path node.
            new_cov = fuse_cov(cov, global_cov[u])
            uncertainty_drop = max(0.0, avg_eigenvalue(cov) - avg_eigenvalue(new_cov))
            new_benefit = benefit + joint_weight(u, new_dist) * uncertainty_drop

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

    # Update global_cov: for each node the support agent visited, propagate
    # its measurement to primary-path nodes that it could have informed in time
    # (d_sup_node <= d_prim[k]) weighted by spatial proximity.
    # Being early is always valid — only late observations are excluded.
    sup_cum_dist = 0.0
    sup_node_dist = Dict{Int,Float64}()
    for idx in 1:length(path)
        if idx > 1
            sup_cum_dist += graph.distance[path[idx-1], path[idx]]
        end
        sup_node_dist[path[idx]] = sup_cum_dist
    end

    for (node, d_sup_node) in sup_node_dist
        for k in 1:np
            d_sup_node > d_prim[k] && continue   # observation arrived too late
            pnode = primary_path[k]
            spatial_w = exp(-graph.distance[node, pnode]^2 / (2 * COMM_RADIUS^2))
            spatial_w < 1e-4 && continue
            fused = fuse_cov(global_cov[pnode], final.cov)
            global_cov[pnode] = (1.0 - spatial_w) .* global_cov[pnode] .+ spatial_w .* fused
        end
    end

    return path, final.dist, final.risk
end

# ---------------------------------------------------------------------------
# Single iteration: given a fixed provisional_path, run support agents and
# replan the primary agent.  global_cov is reset from scratch each call so
# iterations are independent.  Returns the primary agent's new path + all paths.
# ---------------------------------------------------------------------------
function run_iteration(graph::LandmarkGraph,
                       provisional_path::Vector{Int},
                       provisional_dist::Float64,
                       risk_threshold::Float64,
                       num_ag::Int)

    n          = graph.n
    # Fresh covariances every iteration — no bleed-through between iterations
    global_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    sup_paths  = Vector{Int}[]
    sup_dists  = Float64[]
    sup_risks  = Float64[]

    # Support agents plan around provisional_path, within provisional_dist
    for _ in 2:num_ag
        path, d, r = search_support_agent!(graph, global_cov,
                                            provisional_dist, provisional_path;
                                            use_heuristic=false)
        push!(sup_paths, path)
        push!(sup_dists, isempty(path) ? Inf : d)
        push!(sup_risks,  isempty(path) ? NaN : r)
    end

    # Primary agent replans on the support-updated global_cov
    main_path, main_d, main_r =
        search_main_agent!(graph, global_cov, risk_threshold;
                           use_heuristic=true, update_global=true)

    return main_path, main_d, main_r, sup_paths, sup_dists, sup_risks
end

# ---------------------------------------------------------------------------
# Multi-agent RCSP orchestration with iterative refinement
#
# Iteration 0: provisional path = shortest unconstrained path (ideal target).
#              provisional_dist = solo risk-constrained distance (caps support travel).
# Iteration k: provisional path = primary agent's path from iteration k-1.
#              Repeat until path converges or max_iter reached.
#
# global_cov is reset at the start of each iteration so covariance reductions
# from one iteration don't compound into the next.
# ---------------------------------------------------------------------------
function multi_agent_rcsp(graph::LandmarkGraph,
                          risk_threshold::Float64,
                          num_ag::Int = 1;
                          max_iter::Int = 5)

    n = graph.n

    # --- Bootstrap: shortest path + solo risk-constrained distance ---
    shortest_path, shortest_dist = search_shortest_path(graph)
    if isempty(shortest_path)
        println("No path exists in graph.")
        return Vector{Int}[], Float64[], Float64[], [copy(graph.landmarks[i].cov) for i in 1:n],
               Int[], Inf, NaN, 0
    end

    # Solo risk-constrained dist: how far primary would go alone (caps support travel)
    init_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    _, solo_dist, solo_risk =
        search_main_agent!(graph, init_cov, risk_threshold;
                           use_heuristic=true, update_global=false)

    if isinf(solo_dist)
        println("No feasible risk-constrained path found solo.")
        return Vector{Int}[], Float64[], Float64[], init_cov, Int[], Inf, NaN, 0
    end

    # provisional_path is what support agents target; starts as shortest path.
    # provisional_dist caps how far support agents roam; starts as solo dist.
    # We only update provisional_path when the primary route genuinely changes
    # to a *different* path — if the same path comes back with higher risk (because
    # support agents shifted their coverage), we keep the previous best result.
    provisional_path = shortest_path
    provisional_dist = solo_dist

    best_paths     = Vector{Int}[]
    best_dists     = Float64[]
    best_risks     = Float64[]
    converged_iter = 0

    for iter in 1:max_iter
        main_path, main_d, main_r, sup_paths, sup_dists, sup_risks =
            run_iteration(graph, provisional_path, provisional_dist,
                          risk_threshold, num_ag)

        if isempty(main_path)
            println("  Iter $iter: primary agent found no path — stopping.")
            break
        end

        println("  Iter $iter: primary path = ", main_path,
                "  dist=", round(main_d, digits=3),
                "  risk=", round(main_r, digits=4))

        # Always update with this iteration's result. With provisional_dist fixed
        # at solo_dist, support agents have the same travel budget every iteration
        # and risk should be non-increasing. Warn if it rises (indicates a bug).
        if !isempty(best_paths) && main_r > best_risks[end]
            println("  Warning: risk increased from ", round(best_risks[end], digits=4),
                    " to ", round(main_r, digits=4), " — check coverage.")
        end
        best_paths     = vcat(sup_paths, [main_path])
        best_dists     = vcat(sup_dists, [main_d])
        best_risks     = vcat(sup_risks, [main_r])
        converged_iter = iter

        # Convergence: primary path is unchanged from what support agents
        # were targeting — no new information to exploit.
        if main_path == provisional_path
            println("  Converged at iteration $iter.")
            break
        end

        # Primary path changed — update provisional so next iteration's support
        # agents target the route the primary agent actually wants to take.
        # provisional_dist stays fixed at solo_dist: it represents how far the
        # primary would travel alone and is the correct permanent cap on support
        # travel regardless of how the primary path refines. Shrinking it each
        # iteration was restricting support agents more than iteration 1 and
        # causing them to cover the route less well, raising risk.
        provisional_path = main_path
    end

    return best_paths, best_dists, best_risks,
           [copy(graph.landmarks[i].cov) for i in 1:n],   # return original cov for reference
           shortest_path, shortest_dist, solo_risk, converged_iter
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
paths, dists, risks, _, shortest_path, shortest_dist, solo_risk, converged_iter =
    multi_agent_rcsp(graph, risk_threshold, NUM_AGENTS; max_iter=5)

println("Shortest path (target) : ", shortest_path,
        "  dist=", round(shortest_dist, digits=3))
println("Solo risk-constrained  : risk=", round(solo_risk, digits=4))
println("Converged at iteration : ", converged_iter)

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