using Distributions
using Plots
using DataStructures
using LinearAlgebra
using QuadGK
using SpecialFunctions

# ----------------------
# Constants & Parameters
# ----------------------
# Physical scale: 1 unit = 10m. Graph spans ~200m x 200m.
# Platform: AUV with DVL+IMU dead reckoning, acoustic landmark fixes.
#
# DIR_UNCERTAINTY_PER_METER : 3% dead-reckoning drift (DVL+IMU, along-track)
# MAJ_MIN_UNC_RATIO         : along-track drift ~3x cross-track (DVL characteristic)
# VISIBILITY_RAD            : acoustic transponder detectable within ~30m (3 units)
# SENSOR_NOISE              : USBL/LBL fix accuracy ~2m (0.2 units)
# COMM_RADIUS               : acoustic modem range ~80m (8 units)
# risk_threshold            : max acceptable cumulative risk along primary path

risk_threshold = 0.01

const VISIBILITY_RAD             = 3.0
const DIR_UNCERTAINTY_PER_METER  = 0.10  # raised: direct hop now risky, forces routing through landmarks
const MAJ_MIN_UNC_RATIO          = 3
const PERP_UNCERTAINTY_PER_METER = DIR_UNCERTAINTY_PER_METER / MAJ_MIN_UNC_RATIO
const MARKER_PROPORTION          = 50.0
const NUM_AGENTS                 = 3
const SENSOR_NOISE               = 0.1
const COMM_RADIUS                = 8.0   # reduced: ~40m, gives sharper spatial gradient for support routing

# ----------------------
# Data Structures
# ----------------------
struct State
    node::Int
    dist::Float64
    risk::Float64
    cov::Matrix{Float64}
    parent::Int
end

struct SupportState
    node::Int
    dist::Float64
    risk::Float64
    cov::Matrix{Float64}
    cum_benefit::Float64
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

function prob_outside_disk(Sigma::Matrix{Float64}, r::Float64)::Float64
    # Exact P(||e|| > r) for e ~ N(0, Sigma) via 1D radial integration in eigenbasis.
    # Accounts for full 2D covariance shape, not just the worst-case eigenvalue.
    vals = sort(real(eigvals(Symmetric((Sigma + Sigma') / 2))), rev=true)
    l1 = max(vals[1], 1e-12)
    l2 = max(vals[2], 1e-12)
    a  = (1/l1 + 1/l2) / 4
    b  = abs(1/l2 - 1/l1) / 4
    radial_pdf(ρ) = ρ / sqrt(l1 * l2) * exp(-ρ^2 * a) * besseli(0, ρ^2 * b)
    p_inside, _ = quadgk(radial_pdf, 0.0, r, rtol=1e-6)
    return clamp(1.0 - p_inside, 0.0, 1.0)
end

function calc_edge_risk(current_cov::Matrix{Float64}, lj_cov::Matrix{Float64},
                        edge_distance::Float64, dir_angle::Float64)
    predicted_cov = current_cov + growth_covariance(edge_distance, dir_angle) + lj_cov
    risk = prob_outside_disk(predicted_cov, VISIBILITY_RAD)
    return risk, predicted_cov
end

function fuse_cov(path_cov::Matrix{Float64}, landmark_cov::Matrix{Float64})
    R_s = SENSOR_NOISE^2 * I(2)
    return inv(inv(path_cov) + inv(landmark_cov + R_s))
end

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
                pq[u]     = nd
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

function trace_path_covs(graph::LandmarkGraph,
                          path::Vector{Int},
                          global_cov::Vector{Matrix{Float64}})
    isempty(path) && return Matrix{Float64}[]
    n = graph.n
    goal_node = path[end]
    covs = Matrix{Float64}[]
    current_cov = copy(global_cov[path[1]])
    push!(covs, copy(current_cov))
    for k in 2:length(path)
        v = path[k-1]; u = path[k]
        current_cov = current_cov + growth_covariance(graph.distance[v,u], graph.orientation[v,u])
        if u != goal_node
            current_cov = fuse_cov(current_cov, global_cov[u])
        end
        push!(covs, copy(current_cov))
    end
    return covs
end

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

            # Goal is not a landmark — do not fuse its covariance
            new_cov = (u == n) ? cov + growth_covariance(edge_dist, angle) :
                                  fuse_cov(cov + growth_covariance(edge_dist, angle), global_cov[u])

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
        path_covs = trace_path_covs(graph, path, global_cov)
        for (k, node) in enumerate(path)
            node == n && continue  # goal is not a landmark
            global_cov[node] = fuse_cov(global_cov[node], path_covs[k])
        end
    end

    return path, final.dist, final.risk
end

function search_support_agent!(graph::LandmarkGraph,
                                global_cov::Vector{Matrix{Float64}},
                                dist_cap::Float64,
                                primary_path::Vector{Int};
                                use_heuristic::Bool = false)
    n = graph.n

    np = length(primary_path)
    d_prim = zeros(np)
    for k in 2:np
        d_prim[k] = d_prim[k-1] + graph.distance[primary_path[k-1], primary_path[k]]
    end

    function joint_weight(u::Int, d_sup::Float64)
        w_total = 0.0
        for k in 1:np
            d_sup > d_prim[k] && continue
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

            # Goal is not a landmark — do not fuse its covariance
            new_cov = (u == n) ? cov + growth_covariance(edge_dist, angle) :
                                  fuse_cov(cov + growth_covariance(edge_dist, angle), global_cov[u])
            uncertainty_drop = max(0.0, avg_eigenvalue(cov) - avg_eigenvalue(new_cov))
            new_benefit = benefit + joint_weight(u, new_dist) * uncertainty_drop

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

    # Corrected: each visited support node communicates its landmark measurement
    # to nearby primary-path nodes that haven't been reached yet (temporal gate).
    # Fuse the landmark's own measurement covariance (not the agent's path cov).
    # Distance-inflated noise: far observations are less informative.
    sup_cum_dist = 0.0
    sup_node_dist = Dict{Int,Float64}()
    for idx in 1:length(path)
        if idx > 1
            sup_cum_dist += graph.distance[path[idx-1], path[idx]]
        end
        sup_node_dist[path[idx]] = sup_cum_dist
    end

    for (node, d_sup_node) in sup_node_dist
        landmark_meas_cov = graph.landmarks[node].cov + SENSOR_NOISE^2 * I(2)
        for k in 1:np
            d_sup_node > d_prim[k] && continue
            pnode = primary_path[k]
            spatial_w = exp(-graph.distance[node, pnode]^2 / (2 * COMM_RADIUS^2))
            spatial_w < 1e-4 && continue
            # Inflate measurement noise by proximity — distant observations less useful
            global_cov[pnode] = fuse_cov(global_cov[pnode], landmark_meas_cov / spatial_w)
        end
    end

    return path, final.dist, final.risk
end

function run_iteration(graph::LandmarkGraph,
                       provisional_path::Vector{Int},
                       provisional_dist::Float64,
                       risk_threshold::Float64,
                       num_ag::Int)

    n          = graph.n
    global_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    sup_paths  = Vector{Int}[]
    sup_dists  = Float64[]
    sup_risks  = Float64[]

    for _ in 2:num_ag
        path, d, r = search_support_agent!(graph, global_cov,
                                            provisional_dist, provisional_path;
                                            use_heuristic=false)
        push!(sup_paths, path)
        push!(sup_dists, isempty(path) ? Inf : d)
        push!(sup_risks,  isempty(path) ? NaN : r)
    end

    main_path, main_d, main_r =
        search_main_agent!(graph, global_cov, risk_threshold;
                           use_heuristic=true, update_global=true)

    return main_path, main_d, main_r, sup_paths, sup_dists, sup_risks, global_cov
end

function multi_agent_rcsp(graph::LandmarkGraph,
                          risk_threshold::Float64,
                          num_ag::Int = 1;
                          max_iter::Int = 5)

    n = graph.n

    shortest_path, shortest_dist = search_shortest_path(graph)
    if isempty(shortest_path)
        println("No path exists in graph.")
        return Vector{Int}[], Float64[], Float64[], [copy(graph.landmarks[i].cov) for i in 1:n],
               Int[], Inf, NaN, 0
    end

    init_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    solo_path, solo_dist, solo_risk =
        search_main_agent!(graph, init_cov, risk_threshold;
                           use_heuristic=true, update_global=false)

    if isinf(solo_dist)
        println("No feasible risk-constrained path found solo.")
        return Vector{Int}[], Float64[], Float64[], init_cov, Int[], Inf, NaN, 0
    end

    # Initialise with the solo risk-constrained path, not the unconstrained shortest path.
    # Support agents must cover the path the primary actually needs to take.
    provisional_path = solo_path
    provisional_dist = solo_dist

    best_paths      = Vector{Int}[]
    best_dists      = Float64[]
    best_risks      = Float64[]
    last_global_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    converged_iter  = 0

    for iter in 1:max_iter
        main_path, main_d, main_r, sup_paths, sup_dists, sup_risks, global_cov =
            run_iteration(graph, provisional_path, provisional_dist,
                          risk_threshold, num_ag)

        if isempty(main_path)
            println("  Iter $iter: primary agent found no path — stopping.")
            break
        end

        println("  Iter $iter: primary path = ", main_path,
                "  dist=", round(main_d, digits=3),
                "  risk=", round(main_r, digits=4))

        if !isempty(best_paths) && main_r > best_risks[end]
            println("  Warning: risk increased from ", round(best_risks[end], digits=4),
                    " to ", round(main_r, digits=4), " — check coverage.")
        end
        best_paths      = vcat(sup_paths, [main_path])
        best_dists      = vcat(sup_dists, [main_d])
        best_risks      = vcat(sup_risks, [main_r])
        last_global_cov = global_cov
        converged_iter  = iter

        if main_path == provisional_path
            println("  Converged at iteration $iter.")
            break
        end

        provisional_path = main_path
    end

    return best_paths, best_dists, best_risks,
           last_global_cov,
           shortest_path, shortest_dist, solo_risk, converged_iter
end

# ----------------------
# Landmark graph setup
# ----------------------
landmarks = [
    Landmark(0.5, 0.5, [0.0400 0.0000; 0.0000 0.0400]),
    Landmark(2.0, 2.5, [0.0765 0.0113; 0.0113 0.0435]),
    Landmark(4.5, 0.5, [0.0570 -0.0072; -0.0072 0.0430]),
    Landmark(1.5, 6.0, [0.1065 0.0064; 0.0064 0.0435]),
    Landmark(6.8, 3.2, [0.0519 0.0108; 0.0108 0.0381]),
    Landmark(3.8, 8.8, [0.0875 -0.0095; -0.0095 0.0425]),
    Landmark(8.5, 6.5, [0.0596 0.0118; 0.0118 0.0504]),
    Landmark(10.5, 3.0, [0.0944 -0.0224; -0.0224 0.0656]),
    Landmark(7.2, 11.5, [0.0690 0.0080; 0.0080 0.0310]),
    Landmark(9.0, 9.5, [0.0826 0.0130; 0.0130 0.0574]),
    Landmark(12.8, 5.5, [0.0776 -0.0120; -0.0120 0.0424]),
    Landmark(11.0, 12.5, [0.0587 0.0028; 0.0028 0.0313]),
    Landmark(14.5, 7.0, [0.0857 -0.0275; -0.0275 0.0643]),
    Landmark(13.2, 11.0, [0.0609 0.0093; 0.0093 0.0491]),
    Landmark(15.8, 10.0, [0.0862 -0.0043; -0.0043 0.0438]),
    Landmark(14.0, 14.5, [0.0683 0.0125; 0.0125 0.0317]),
    Landmark(16.8, 12.5, [0.0719 -0.0123; -0.0123 0.0481]),
    Landmark(16.0, 15.0, 1e-9 * [1.0 0.0; 0.0 1.0])
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
paths, dists, risks, final_global_cov, shortest_path, shortest_dist, solo_risk, converged_iter =
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
# Covariance trace using support-updated global_cov
# ----------------------
function trace_path_covariances(graph::LandmarkGraph,
                                 path::Vector{Int},
                                 global_cov::Vector{Matrix{Float64}})
    isempty(path) && return Matrix{Float64}[]

    covs = Matrix{Float64}[]
    current_cov = copy(global_cov[path[1]])
    push!(covs, copy(current_cov))

    goal_node = path[end]
    for k in 2:length(path)
        v = path[k-1]
        u = path[k]
        edge_dist = graph.distance[v, u]
        angle     = graph.orientation[v, u]
        # Always accumulate dead-reckoning drift
        current_cov = current_cov + growth_covariance(edge_dist, angle)
        # Only fuse landmark measurement if this is NOT the goal node
        if u != goal_node
            current_cov = fuse_cov(current_cov, global_cov[u])
        end
        push!(covs, copy(current_cov))
    end

    return covs
end

# ----------------------
# B-Spline Interpolation
# ----------------------
# ----------------------
# B-Spline Interpolation
# ----------------------
# Each graph node gets two flanking control points placed a fraction `tightness`
# of the edge length along the incoming and outgoing edge directions.  This
# pulls the spline tightly through every node while keeping C² continuity.
#
# All agents are sampled at exactly BSPLINE_NPTS points so the resulting
# coordinate vectors are the same length for downstream optimisation.

const BSPLINE_NPTS  = 20   # uniform sample count across ALL agents
const FLANK_RATIO   = 0.18  # minimum flank offset as fraction of edge length
const AUV_TURN_RADIUS = 4.0 # minimum turning radius in graph units (40m / 10m per unit)

# -----------------------------------------------------------------------
# Minimum flank distance to honour a turning radius constraint at a node.
#
# For a symmetric cubic B-spline corner the radius of curvature at the
# node apex is:
#
#   R ≈ (3/4) * h / sin²(α/2)          ... (*)
#
# where h is the flank offset (same on both sides) and α is the exterior
# turning angle (π − interior angle).  Rearranging for the required h:
#
#   h_min = (4/3) * R_min * sin²(α/2)
#
# This is exact for a symmetric corner; for asymmetric flanks (different
# incoming / outgoing edge lengths) we apply the same h_min to both sides,
# which is conservative (actual R ≥ R_min).
# -----------------------------------------------------------------------
function min_flank_for_radius(dx_in, dy_in, dx_out, dy_out,
                               R_min::Float64)::Float64
    # Unit vectors along incoming and outgoing edges
    d_in  = hypot(dx_in,  dy_in)
    d_out = hypot(dx_out, dy_out)
    (d_in < 1e-12 || d_out < 1e-12) && return 0.0

    ux_in  =  dx_in  / d_in;  uy_in  =  dy_in  / d_in
    ux_out =  dx_out / d_out; uy_out =  dy_out / d_out

    # cos of interior angle between the two edge directions
    cos_int = clamp(ux_in * ux_out + uy_in * uy_out, -1.0, 1.0)
    # exterior turning angle α = π − interior angle
    alpha   = π - acos(cos_int)

    sin2_half = sin(alpha / 2)^2           # sin²(α/2)
    return (4.0 / 3.0) * R_min * sin2_half
end

function bspline_basis(t::Float64)
    b0 = (1 - t)^3 / 6
    b1 = (3t^3 - 6t^2 + 4) / 6
    b2 = (-3t^3 + 3t^2 + 3t + 1) / 6
    b3 = t^3 / 6
    return b0, b1, b2, b3
end

# Evaluate the B-spline defined by control points (px, py) at global parameter
# s ∈ [0, 1] mapped uniformly across all spans.
function bspline_eval(px::Vector{Float64}, py::Vector{Float64}, s::Float64)
    m = length(px)
    num_spans = m - 3
    span_s = s * num_spans          # continuous span index
    i = clamp(floor(Int, span_s) + 1, 1, num_spans)
    t = span_s - (i - 1)           # local parameter within span ∈ [0,1)
    b0, b1, b2, b3 = bspline_basis(t)
    x = b0*px[i] + b1*px[i+1] + b2*px[i+2] + b3*px[i+3]
    y = b0*py[i] + b1*py[i+1] + b2*py[i+2] + b3*py[i+3]
    return x, y
end

# Build the expanded control-point sequence with two flanking points per node.
#
# Flank distance at each interior node is:
#   max(FLANK_RATIO * min(d_in, d_out),  h_min_from_turn_radius)
#
# capped at 0.45 * the shorter edge so flanks never cross each other or
# overshoot the midpoint of an edge.
function expand_control_points(xs::Vector{Float64}, ys::Vector{Float64};
                                R_min::Float64 = AUV_TURN_RADIUS)
    n = length(xs)
    px = Float64[]; py = Float64[]

    for k in 1:n
        if k == 1
            if n > 1
                dx = xs[2] - xs[1]; dy = ys[2] - ys[1]
                d  = hypot(dx, dy)
                frac = clamp(FLANK_RATIO, 0.0, 0.45)
                push!(px, xs[1]);                   push!(py, ys[1])
                push!(px, xs[1] + frac * dx);       push!(py, ys[1] + frac * dy)
            else
                push!(px, xs[1]); push!(py, ys[1])
            end

        elseif k == n
            dx = xs[n] - xs[n-1]; dy = ys[n] - ys[n-1]
            d  = hypot(dx, dy)
            frac = clamp(FLANK_RATIO, 0.0, 0.45)
            push!(px, xs[n] - frac * dx);       push!(py, ys[n] - frac * dy)
            push!(px, xs[n]);                   push!(py, ys[n])

        else
            dx_in  = xs[k]   - xs[k-1]; dy_in  = ys[k]   - ys[k-1]
            dx_out = xs[k+1] - xs[k];   dy_out = ys[k+1] - ys[k]
            d_in   = hypot(dx_in,  dy_in)
            d_out  = hypot(dx_out, dy_out)

            # Required flank length to keep R ≥ R_min at this node
            h_turn = min_flank_for_radius(dx_in, dy_in, dx_out, dy_out, R_min)

            # Ratio-based minimum, capped so flanks don't overshoot edge midpoints
            h_ratio = FLANK_RATIO * min(d_in, d_out)
            h       = clamp(max(h_ratio, h_turn), 0.0, 0.45 * min(d_in, d_out))

            # Pre-flank: step back along unit incoming direction
            ux_in = dx_in / d_in; uy_in = dy_in / d_in
            push!(px, xs[k] - h * ux_in);  push!(py, ys[k] - h * uy_in)
            # Node itself
            push!(px, xs[k]);               push!(py, ys[k])
            # Post-flank: step forward along unit outgoing direction
            ux_out = dx_out / d_out; uy_out = dy_out / d_out
            push!(px, xs[k] + h * ux_out); push!(py, ys[k] + h * uy_out)
        end
    end
    return px, py
end

function bspline_curve(xs::Vector{Float64}, ys::Vector{Float64};
                        npts::Int = BSPLINE_NPTS)
    n = length(xs)
    n == 1 && return xs, ys
    n == 2 && return collect(range(xs[1], xs[2], length=npts)),
                     collect(range(ys[1], ys[2], length=npts))

    # Expand to flanked control points then clamp endpoints
    epx, epy = expand_control_points(xs, ys)
    px = [epx[1]; epx[1]; epx; epx[end]; epx[end]]
    py = [epy[1]; epy[1]; epy; epy[end]; epy[end]]

    # Sample at exactly npts uniformly spaced parameter values
    out_x = Vector{Float64}(undef, npts)
    out_y = Vector{Float64}(undef, npts)
    for j in 1:npts
        s = (j - 1) / (npts - 1)
        out_x[j], out_y[j] = bspline_eval(px, py, s)
    end
    return out_x, out_y
end

# ----------------------
# Plot
# ----------------------
agent_colors = [:purple, :teal, :darkorange, :crimson, :magenta, :brown,
                :lime, :navy, :coral, :olive]

for (i, path) in enumerate(paths)
    isempty(path) && continue
    path_x = [graph.landmarks[idx].x for idx in path]
    path_y = [graph.landmarks[idx].y for idx in path]

    # Raw straight-line segments (faint reference)
    if i == length(paths)
        plot!(plt, path_x, path_y,
              label=false, color=:blue, linewidth=0.6, linestyle=:dot, alpha=0.35)
    else
        clr = i <= length(agent_colors) ? agent_colors[i] : :gray
        plot!(plt, path_x, path_y,
              label=false, color=clr, linewidth=0.4, linestyle=:dot, alpha=0.25)
    end

    # B-spline smooth curve — always BSPLINE_NPTS points
    sx, sy = bspline_curve(path_x, path_y; npts=BSPLINE_NPTS)
    if i == length(paths)
        plot!(plt, sx, sy, label="Primary agent (B-spline)", color=:blue, linewidth=2.5)
    else
        clr = i <= length(agent_colors) ? agent_colors[i] : :gray
        plot!(plt, sx, sy, label="Support agent $i (B-spline)",
              color=clr, linewidth=1.4, linestyle=:dash)
    end
end

# Scatter control-point landmarks on top so they remain visible
scatter!(plt, x_coords[2:end-1], y_coords[2:end-1],
         label=false, color=:black, markersize=2.5)

primary_path = paths[end]
if !isempty(primary_path)
    path_covs = trace_path_covariances(graph, primary_path, final_global_cov)
    for (k, node) in enumerate(primary_path)
        lm = graph.landmarks[node]
        draw_covariance_ellipse!(plt, lm.x, lm.y, path_covs[k];
                                 nstd=2, color=:blue, alpha=0.18)
    end
    plot!(plt, [NaN], [NaN], seriestype=:shape,
          color=:blue, alpha=0.18, label="Primary 95% cov")
end

xlabel!(plt, "x (×10m)"); ylabel!(plt, "y (×10m)")
title!(plt, "Underwater AUV Swarm — Continuous B-Spline Trajectories")
savefig(plt, "multi_agent_paths_bspline.png")
println("\nPlot saved to multi_agent_paths_bspline.png")