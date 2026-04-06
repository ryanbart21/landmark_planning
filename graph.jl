using Plots
using DataStructures
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)  # Reproducible jittered grid sampling

# ----------------------
# Constants & Parameters
# ----------------------
# Physical scale: 1 unit = 100m. Graph spans ~1600m x 1500m.
# Platform: AUV with DVL+IMU dead reckoning, acoustic landmark fixes.
#
# DIR_UNCERTAINTY_PER_METER : 10% dead-reckoning drift (DVL+IMU, along-track)
# MAJ_MIN_UNC_RATIO         : along-track drift ~3x cross-track (DVL characteristic)
# SENSOR_NOISE              : USBL/LBL fix accuracy ~10m (0.1 units)
# COMM_RADIUS               : acoustic modem range ~1000m (10 units)
# UNC_RADIUS_THRESHOLD      : max acceptable sqrt(λ_max(Σ_goal)) at goal (~50m = 0.5 units)

const DIR_UNCERTAINTY_PER_METER  = 0.05    # LOWERED from 0.30 to make path length more impactful on final uncertainty
const MAJ_MIN_UNC_RATIO          = 3
const PERP_UNCERTAINTY_PER_METER = DIR_UNCERTAINTY_PER_METER / MAJ_MIN_UNC_RATIO
const MARKER_PROPORTION          = 5.0
const NUM_AGENTS                 = 2
const SENSOR_NOISE               = 0.038   # High-precision bearing-based landmark acoustic fixes
const COMM_RADIUS                = 500.0    # acoustic modem ~300m (3 units); 100m wide tapered Gaussian
const VISIBILITY_SIGMA           = 50.0     # 1σ detection range for landmark observations
const COMM_INTERVAL              = 10.0    # Fixed synchronous communication every 10m of travel
const COMM_SIGMA                 = 150.0    # Gaussian taper: σ=50m means ~1σ=50m, ~3σ=150m (100m ≈ 2σ)
const UNC_RADIUS_THRESHOLD       = 2.964   # Achievable with meaningful lateral detours toward landmarks
const HEX_WIDTH_M                = 80.0
const HEX_RADIUS_M               = HEX_WIDTH_M / sqrt(3.0)  # pointy-top hex: width = sqrt(3)*radius
const SUPPORT_PLOT_OFFSET_M      = 18.0  # visualization-only offset so support paths stay visible

# ε-optimal parameter for primary-cost weighting in A* (f = g + (1+ε)h)
# Set >0 for faster search with bounded suboptimality.
const PRIMARY_EPSILON = 0.0

# Support idling preference used only as a lower-priority tie-break key.
const SUPPORT_IDLE_PENALTY  = 30.0

# Pipeline mode switch:
#  :discrete_then_continuous => run discrete planner first, then continuous refinement
#  :straight_continuous      => skip discrete planner and seed continuous stage with direct start->goal path
const PIPELINE_MODE = :discrete_then_continuous

# When true, primary-only feasibility is not accepted early; supports are still
# searched so you can see whether they improve the primary uncertainty.
const REQUIRE_SUPPORT_REFINEMENT = true

# Number of straight-line control points used to seed primary continuous optimization
# when PIPELINE_MODE == :straight_continuous.
const STRAIGHT_CONT_PRIMARY_WPTS = 11

# Sequential (primary + supports) search debug animation/progress controls.
const DEBUG_SEQ_SEARCH_ANIMATE = true
const DEBUG_SEQ_SEARCH_GIF = "fig_seq_primary_support_search.gif"
const DEBUG_SEQ_PRIMARY_PROGRESS_EVERY = 200
const DEBUG_SEQ_SUPPORT_PROGRESS_EVERY = 200
const DEBUG_SEQ_ANIM_PRIMARY_SAMPLE_EVERY = 20
const DEBUG_SEQ_ANIM_SUPPORT_SAMPLE_EVERY = 20

# ========================================================================================
# KALMAN FILTERING & INFORMATION FUSION APPROACH
# ========================================================================================
# All agents use continuous Kalman filtering with information filter (Joseph form) updates:
#
# 1. DEAD-RECKONING PROPAGATION:
#    - Each step grows covariance by INS drift along and perpendicular to motion
#    - DIR_UNCERTAINTY_PER_METER × distance for along-track (major) axis
#    - PERP_UNCERTAINTY_PER_METER ×distance for cross-track (minor) axis
#    - Heading-aware: covariance rotates with vehicle direction
#
# 2. LANDMARK FUSION (Kalman Measurement Update):
#    - At each waypoint, fuse all visible landmarks using information filter
#    - Information matrix I = inv(S_total) where:
#      * S_sensor = rotated bearing-angle measurement covariance matrix
#      * S_total = S_sensor + landmark_uncertainty, scaled by detection probability
#    - Joseph-form update: J_new = I_landmark + inv(P_prior)
#    - Posterior: P = inv(J_new)
#    - Maintains numerical stability via information form (Joseph equations)
#
# 3. INTER-AGENT COMMUNICATION FUSION:
#    - When agents within COMM_RADIUS, receiver fuses sender's covariance via Kalman update
#    - Weight = exp(-distance² / (2×COMM_RADIUS²)), applied to information matrix
#    - Both primary and support agents participate in full Kalman filtering
#
# 4. DISCRETE EVALUATION:
#    - evaluate_joint_discrete() applies Kalman updates to all agents in parallel
#    - apply_inter_agent_discrete_comms!() fuses information between agents
#
# All computations use symmetric positive-definite 2×2 covariance matrices with
# full Kalman optimality guarantees. Equivalence to standard form ensured via
# Joseph form stability and explicit information matrix inversion.
# ========================================================================================

# Primary agent state: (node, cumulative distance, covariance at node).
# Dominance is over (dist, max_eigenvalue(cov)) — no risk field.
# `visited` is a BitVector per state for cycle detection (supports arbitrary graph sizes).
struct State
    node::Int
    dist::Float64
    cov::Matrix{Float64}
    parent::Int
    visited::BitVector
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
    neighbors::Vector{Vector{Int}}   # sparse adjacency list (empty = fully connected)
    shortest_paths::Matrix{Float64}  # Floyd-Warshall: shortest graph distance between all pairs
end

# Floyd-Warshall: compute shortest-path distances between all pairs via graph edges
function floyd_warshall(dist::Matrix{Float64}, neighbors::Vector{Vector{Int}})
    n = size(dist, 1)
    # Initialize with edge weights (or Inf if no edge)
    sp = fill(Inf, n, n)
    for i in 1:n
        sp[i, i] = 0.0
    end
    for i in 1:n
        for j in neighbors[i]
            sp[i, j] = dist[i, j]
        end
    end
    # Standard Floyd-Warshall: O(n³)
    for k in 1:n
        for i in 1:n
            for j in 1:n
                if sp[i, k] + sp[k, j] < sp[i, j]
                    sp[i, j] = sp[i, k] + sp[k, j]
                end
            end
        end
    end
    return sp
end

function generate_graph(landmarks::Vector{Landmark}; neighbors::Vector{Vector{Int}}=Vector{Int}[])
    n = length(landmarks)
    dist   = zeros(n, n)
    orient = zeros(n, n)
    for (i, li) in enumerate(landmarks)
        for (j, lj) in enumerate(landmarks)
            dx = lj.x - li.x; dy = lj.y - li.y
            dist[i,j]   = sqrt(dx^2 + dy^2)
            orient[i,j] = atan(dy, dx)
        end
    end
    adj = isempty(neighbors) ? [collect(filter(j->j!=i, 1:n)) for i in 1:n] : neighbors
    sp = floyd_warshall(dist, adj)
    return LandmarkGraph(n, landmarks, dist, orient, adj, sp)
end

# Closed-form max eigenvalue for 2×2 symmetric matrix — avoids eigvals overhead
@inline function max_eigenvalue(M::Matrix{Float64})
    a = M[1,1]; d = M[2,2]; b = M[1,2]
    half_tr = (a + d) * 0.5
    return half_tr + sqrt(max(0.0, (a - d)^2 * 0.25 + b * b))
end

unc_radius(cov::Matrix{Float64}) = sqrt(max_eigenvalue(cov))

# Determinant-based scalar uncertainty metric.
# For isotropic Σ = σ²I, this returns σ (same units as position),
# because det(Σ)^(1/4) = (σ⁴)^(1/4) = σ.
@inline function unc_det_radius(cov::Matrix{Float64})
    d = cov[1,1] * cov[2,2] - cov[1,2] * cov[2,1]
    return max(d, 1e-18)^(0.25)
end

# Primary scalar uncertainty used by planning/constraints/reporting.
unc_radius(cov::Matrix{Float64}) = unc_det_radius(cov)

# Covariance partial order for sound dominance pruning:
# cov_a dominates cov_b iff (cov_b - cov_a) is positive semidefinite.
@inline function cov_dominates(cov_a::Matrix{Float64}, cov_b::Matrix{Float64}; tol::Float64=1e-9)
    # Check PSD for symmetric 2x2 matrix D = cov_b - cov_a via principal minors.
    d11 = cov_b[1,1] - cov_a[1,1]
    d12 = cov_b[1,2] - cov_a[1,2]
    d22 = cov_b[2,2] - cov_a[2,2]
    det = d11 * d22 - d12 * d12
    return d11 >= -tol && d22 >= -tol && det >= -tol
end

# Inline: R * diag(sd², sp²) * R' expanded to avoid intermediate allocations
@inline function growth_covariance(distance::Float64, angle::Float64)
    sd2 = (DIR_UNCERTAINTY_PER_METER  * distance)^2
    sp2 = (PERP_UNCERTAINTY_PER_METER * distance)^2
    c = cos(angle); s = sin(angle)
    diff = sd2 - sp2
    return [c*c*sd2 + s*s*sp2  c*s*diff;
            c*s*diff            s*s*sd2 + c*c*sp2]
end

# Inline 2×2 matrix inverse — avoids LinearAlgebra.inv overhead for small matrices
@inline function inv2(M::Matrix{Float64})
    a=M[1,1]; b=M[1,2]; c=M[2,1]; d=M[2,2]
    det = a*d - b*c
    inv_det = 1.0 / det
    return [d*inv_det  -b*inv_det; -c*inv_det  a*inv_det]
end

# Information-filter fuse: Σ_new = (Σ_a⁻¹ + Σ_b⁻¹)⁻¹  — two 2×2 inverses only
@inline function fuse_cov(A::Matrix{Float64}, B::Matrix{Float64})
    return inv2(inv2(A) + inv2(B))
end

# --------------------------------------------------------------------------
# Clamped cubic B-spline helpers for continuous refinement
# --------------------------------------------------------------------------
const SPLINE_DEGREE                    = 3
const SPLINE_SAMPLES_PER_SEG           = 5
const SPLINE_CURVATURE_SAMPLES_PER_SEG = 8
const MIN_TURN_RADIUS_M                = 40.0
const MAX_CURVATURE                    = 1.0 / MIN_TURN_RADIUS_M
const CONT_BARRIER_START               = 20.0
const CONT_BARRIER_DECAY               = 0.35
const CONT_BARRIER_STAGES              = 4
const CONT_LINESEARCH_SHRINK           = 0.5
const CONT_MIN_IMPROVEMENT             = 1e-6

@inline function pt_sub(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64})
    return (a[1] - b[1], a[2] - b[2])
end

@inline function pt_scale(a::Tuple{Float64,Float64}, s::Float64)
    return (a[1] * s, a[2] * s)
end

@inline function pt_dist(a::Tuple{Float64,Float64}, b::Tuple{Float64,Float64})
    return hypot(a[1] - b[1], a[2] - b[2])
end

function bspline_pad_controls(control_pts::Vector{Tuple{Float64,Float64}})
    n = length(control_pts)
    if n >= 4
        return control_pts
    elseif n == 3
        return [control_pts[1], control_pts[1], control_pts[2], control_pts[3], control_pts[3]]
    elseif n == 2
        return [control_pts[1], control_pts[1], control_pts[2], control_pts[2]]
    elseif n == 1
        return [control_pts[1], control_pts[1], control_pts[1], control_pts[1]]
    else
        return Tuple{Float64,Float64}[]
    end
end

function bspline_open_uniform_knots(nctrl::Int, degree::Int)
    nctrl >= degree + 1 || return Float64[]
    knots = Vector{Float64}(undef, nctrl + degree + 1)
    nspans = max(1, nctrl - degree)
    for i in 1:length(knots)
        if i <= degree + 1
            knots[i] = 0.0
        elseif i > nctrl
            knots[i] = 1.0
        else
            knots[i] = (i - degree - 1) / nspans
        end
    end
    return knots
end

@inline function bspline_find_span(u::Float64, knots::Vector{Float64}, degree::Int, nctrl::Int)
    u >= knots[end - degree] && return nctrl
    low = degree + 1
    high = nctrl
    while low <= high
        mid = (low + high) >>> 1
        if u < knots[mid]
            high = mid - 1
        elseif u >= knots[mid + 1]
            low = mid + 1
        else
            return mid
        end
    end
    return clamp(low, degree + 1, nctrl)
end

function bspline_basis_funs(span::Int, u::Float64, degree::Int, knots::Vector{Float64})
    N = zeros(Float64, degree + 1)
    left = zeros(Float64, degree)
    right = zeros(Float64, degree)
    N[1] = 1.0
    for j in 1:degree
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in 1:j
            denom = right[r] + left[j - r + 1]
            temp = abs(denom) < 1e-14 ? 0.0 : N[r] / denom
            N[r] = saved + right[r] * temp
            saved = left[j - r + 1] * temp
        end
        N[j + 1] = saved
    end
    return N
end

function bspline_eval_point(control_pts::Vector{Tuple{Float64,Float64}},
                            knots::Vector{Float64},
                            degree::Int,
                            u::Float64)
    nctrl = length(control_pts)
    nctrl == 0 && return (0.0, 0.0)
    u = clamp(u, 0.0, 1.0)
    span = bspline_find_span(u, knots, degree, nctrl)
    basis = bspline_basis_funs(span, u, degree, knots)
    first = span - degree
    x = 0.0
    y = 0.0
    for j in 0:degree
        pt = control_pts[first + j]
        x += basis[j + 1] * pt[1]
        y += basis[j + 1] * pt[2]
    end
    return (x, y)
end

function bspline_derivative_control_points(control_pts::Vector{Tuple{Float64,Float64}},
                                           knots::Vector{Float64},
                                           degree::Int)
    nctrl = length(control_pts)
    nctrl <= 1 && return Tuple{Float64,Float64}[], Float64[], max(degree - 1, 0)
    dctrl = Vector{Tuple{Float64,Float64}}(undef, nctrl - 1)
    for i in 1:nctrl - 1
        denom = knots[i + degree + 1] - knots[i + 1]
        if abs(denom) < 1e-14
            dctrl[i] = (0.0, 0.0)
        else
            dctrl[i] = pt_scale(pt_sub(control_pts[i + 1], control_pts[i]), degree / denom)
        end
    end
    return dctrl, knots[2:end-1], degree - 1
end

function bspline_sample_path(control_pts::Vector{Tuple{Float64,Float64}};
                             degree::Int = SPLINE_DEGREE,
                             length_samples_per_seg::Int = SPLINE_SAMPLES_PER_SEG,
                             curvature_samples_per_seg::Int = SPLINE_CURVATURE_SAMPLES_PER_SEG)
    controls = bspline_pad_controls(control_pts)
    nctrl = length(controls)
    if nctrl == 0
        return Tuple{Float64,Float64}[], Float64[]
    elseif nctrl < degree + 1
        return copy(controls), zeros(Float64, max(0, length(controls) - 1))
    end

    knots = bspline_open_uniform_knots(nctrl, degree)
    nspans = max(1, nctrl - degree)

    n_pos_samples = nspans * length_samples_per_seg + 1
    pos_samples = Vector{Tuple{Float64,Float64}}(undef, n_pos_samples)
    for i in 1:n_pos_samples
        u = (i - 1) / (n_pos_samples - 1)
        pos_samples[i] = bspline_eval_point(controls, knots, degree, u)
    end

    d1_ctrl, d1_knots, d1_degree = bspline_derivative_control_points(controls, knots, degree)
    d2_ctrl, d2_knots, d2_degree = bspline_derivative_control_points(d1_ctrl, d1_knots, d1_degree)

    curv_us = collect(range(0.0, 1.0; length = nspans * curvature_samples_per_seg + 1))
    if length(curv_us) > 2
        curv_us = curv_us[2:end-1]
    end

    curvatures = Vector{Float64}(undef, length(curv_us))
    for (i, u) in enumerate(curv_us)
        dx, dy = bspline_eval_point(d1_ctrl, d1_knots, d1_degree, u)
        ddx, ddy = bspline_eval_point(d2_ctrl, d2_knots, d2_degree, u)
        speed2 = dx * dx + dy * dy
        curvatures[i] = speed2 < 1e-12 ? Inf : abs(dx * ddy - dy * ddx) / max(speed2^(1.5), 1e-12)
    end

    return pos_samples, curvatures
end

@inline function bspline_path_length(samples::Vector{Tuple{Float64,Float64}})
    total = 0.0
    for i in 2:length(samples)
        total += pt_dist(samples[i - 1], samples[i])
    end
    return total
end

# ==========================================================================
# Trajectory model: straight-line segments between waypoints
# ==========================================================================
# Waypoint positions in continuous space (x, y) can be optimized via gradient descent.
# Continuity and curvature constraints handled by downstream execution layer.

# ----------------------
# A* shortest path
# ----------------------
# Admissible heuristic: straight-line Euclidean distance to goal node n.
function search_shortest_path(graph::LandmarkGraph)
    n    = graph.n
    goal = graph.n   # goal is the last node (routing waypoint appended after landmarks+samples)
    dist = fill(Inf, n); parent = fill(-1, n)
    dist[1] = 0.0
    gx = graph.landmarks[goal].x; gy = graph.landmarks[goal].y
    h(v) = hypot(graph.landmarks[v].x - gx, graph.landmarks[v].y - gy)
    pq = PriorityQueue{Int,Float64}(); enqueue!(pq, 1, h(1))
    while !isempty(pq)
        v = dequeue!(pq)
        v == goal && break
        for u in graph.neighbors[v]
            nd = dist[v] + graph.distance[v, u]
            if nd < dist[u]
                dist[u] = nd; parent[u] = v
                pq[u] = nd + h(u)
            end
        end
    end
    isinf(dist[goal]) && return Int[], Inf
    path = Int[]; v = goal
    while v != -1; push!(path, v); v = parent[v]; end
    return reverse!(path), dist[goal]
end

# ==========================================================================
# K-Shortest Paths (Simplified Yen's Algorithm)
# ==========================================================================
# Computes K distinct simple paths from start→goal, sorted by distance.
# Returns (paths::Vector{Vector{Int}}, distances::Vector{Float64})
function k_shortest_paths(graph::LandmarkGraph, K::Int=100)
    n = graph.n
    goal = n
    start = 1

    # Find shortest path first
    first_path, first_dist = search_shortest_path(graph)
    isempty(first_path) && return Vector{Int}[], Float64[]

    paths = [first_path]
    distances = [first_dist]

    # Candidate pool: (path, distance)
    candidates = Set{Vector{Int}}()  # Track found paths to avoid duplicates

    # For each path found, generate deviations
    for path_idx in 1:min(K-1, length(paths))
        curr_path = paths[path_idx]
        cand_count_before = length(candidates)

        # Try each node as a spur node to generate alternate paths
        for spur_idx in 1:length(curr_path)-1
            spur_node = curr_path[spur_idx]
            root_path = curr_path[1:spur_idx]

            # Create a modified graph that excludes the root path (except spur node)
            # and all edges from previous path nodes to their successors
            excluded_nodes = Set(root_path[1:end-1])
            excluded_edges = Set{Tuple{Int,Int}}()
            for i in 1:spur_idx-1
                push!(excluded_edges, (curr_path[i], curr_path[i+1]))
            end

            # Search for alternate path from spur_node to goal
            dist_spur = fill(Inf, n)
            parent_spur = fill(-1, n)
            dist_spur[spur_node] = 0.0
            gx = graph.landmarks[goal].x
            gy = graph.landmarks[goal].y
            h(v) = hypot(graph.landmarks[v].x - gx, graph.landmarks[v].y - gy)

            pq = PriorityQueue{Int,Float64}()
            enqueue!(pq, spur_node, h(spur_node))

            while !isempty(pq)
                v = dequeue!(pq)
                v == goal && break
                for u in graph.neighbors[v]
                    # Skip excluded nodes (except when arriving at goal)
                    if u in excluded_nodes && u != goal
                        continue
                    end
                    # Skip excluded edges
                    if (v, u) in excluded_edges
                        continue
                    end
                    nd = dist_spur[v] + graph.distance[v, u]
                    if nd < dist_spur[u]
                        dist_spur[u] = nd
                        parent_spur[u] = v
                        pq[u] = nd + h(u)
                    end
                end
            end

            # If alternate path exists, reconstruct it
            if !isinf(dist_spur[goal])
                spur_path = Int[]
                v = goal
                while v != -1
                    push!(spur_path, v)
                    v = parent_spur[v]
                end
                spur_path = reverse!(spur_path)

                # Concatenate root path + spur path (skip first element of spur to avoid duplication)
                new_path = vcat(root_path, spur_path[2:end])

                # Only add if we haven't seen it before
                if new_path ∉ candidates && new_path ∉ paths
                    new_dist = sum(graph.distance[new_path[i], new_path[i+1]] for i in 1:length(new_path)-1)
                    push!(candidates, new_path)
                end
            end
        end

        # Pick the best candidate and add to paths
        if !isempty(candidates)
            best_path = nothing
            best_dist = Inf
            for cand in candidates
                cand_dist = sum(graph.distance[cand[i], cand[i+1]] for i in 1:length(cand)-1)
                if cand_dist < best_dist
                    best_dist = cand_dist
                    best_path = cand
                end
            end
            if best_dist < Inf
                push!(paths, best_path)
                push!(distances, best_dist)
                delete!(candidates, best_path)
            else
                break
            end
        else
            break
        end
    end

    # Sort by distance (should already be mostly sorted)
    perm = sortperm(distances)
    return paths[perm], distances[perm]
end

# ==========================================================================
# Single Support Agent A* (One support at a time)
# ==========================================================================
# Optimize a single support agent's path given:
#   - Fixed primary path
#   - Fixed positions of other supports (if any)
#
# Returns best path for this support that minimizes primary's goal uncertainty

function single_support_astar(graph::LandmarkGraph,
                             lms::Vector{Landmark},
                             primary_path::Vector{Int},
                             primary_path_dist::Float64,
                             other_support_paths::Vector{Vector{Int}},
                             support_idx::Int;
                             unc_threshold::Float64=Inf,
                             max_expansions::Int=100000,
                             debug_support_label::String="",
                             debug_progress_every::Int=0,
                             debug_anim_sample_every::Int=0,
                             debug_on_frame::Union{Nothing, Function}=nothing)
    n = graph.n
    plen = length(primary_path)
    plen == 0 && return Int[], Inf

    # 1σ gate for catch-up feasibility pruning.
    # The support branch is pruned only when it can never re-enter this gate
    # at any future synchronized timestep.
    function in_sigma_range_at_step(support_node::Int, t::Int)
        pnode = primary_path[t]
        sx = graph.landmarks[support_node].x
        sy = graph.landmarks[support_node].y
        px = graph.landmarks[pnode].x
        py = graph.landmarks[pnode].y
        return (sx - px)^2 + (sy - py)^2 <= COMM_SIGMA^2
    end

    # Precompute candidate communication nodes for each primary timestep using 1σ,
    # except the terminal timestep where we allow full COMM_RADIUS.
    comm_nodes_by_t = Vector{Vector{Int}}(undef, plen)
    for t in 1:plen
        pnode = primary_path[t]
        px = graph.landmarks[pnode].x
        py = graph.landmarks[pnode].y
        range_limit = (t == plen) ? COMM_RADIUS : COMM_SIGMA
        comm_nodes = Int[]
        for v in 1:n
            vx = graph.landmarks[v].x
            vy = graph.landmarks[v].y
            if (vx - px)^2 + (vy - py)^2 <= range_limit^2
                push!(comm_nodes, v)
            end
        end
        comm_nodes_by_t[t] = comm_nodes
    end

    # Optimistic hop lower-bound for catch-up check.
    # Uses max edge length so ceil(sp_dist / max_edge_len) is a valid lower bound on hops.
    max_edge_len = 0.0
    for i in 1:n
        for j in graph.neighbors[i]
            max_edge_len = max(max_edge_len, graph.distance[i, j])
        end
    end
    max_edge_len = max(max_edge_len, 1e-9)

    function can_ever_reconnect(node::Int, t::Int)
        for τ in t:plen
            # Immediate in-range match at same timestep.
            if (τ == plen && (graph.landmarks[node].x - graph.landmarks[primary_path[τ]].x)^2 +
                             (graph.landmarks[node].y - graph.landmarks[primary_path[τ]].y)^2 <= COMM_RADIUS^2) ||
               (τ < plen && in_sigma_range_at_step(node, τ))
                return true
            end

            # Optimistic future catch-up test via graph shortest-path distance.
            targets = comm_nodes_by_t[τ]
            isempty(targets) && continue

            dmin = Inf
            for v in targets
                dmin = min(dmin, graph.shortest_paths[node, v])
            end
            isfinite(dmin) || continue

            hops_lb = Int(ceil(dmin / max_edge_len - 1e-9))
            if hops_lb <= (τ - t)
                return true
            end
        end
        return false
    end

    function primary_goal_unc_with_support(candidate_support_path::Vector{Int})
        support_paths = copy(other_support_paths)
        push!(support_paths, pad_path_to_length(candidate_support_path, plen))
        full_paths = vcat(support_paths, [primary_path])
        final_covs, _ = evaluate_full_paths(full_paths, graph, lms, length(full_paths))
        return unc_radius(final_covs[end])
    end

    function reconstruct_support_path(si::Int)
        path = Int[]
        cur = si
        while cur != -1
            pushfirst!(path, state_nodes[cur])
            cur = state_parent[cur]
        end
        return path
    end

    if !can_ever_reconnect(1, 1)
        return Int[], Inf
    end

    state_nodes = Int[1]
    state_t = Int[1]
    state_dist = Float64[0.0]
    state_parent = Int[-1]
    state_unc = Float64[primary_goal_unc_with_support([1])]

    # Uncertainty-first queue: minimize estimated primary uncertainty first,
    # then use support travel distance as a tie-breaker.
    pq = PriorityQueue{Int, Tuple{Float64, Float64}}()
    enqueue!(pq, 1, (state_unc[1], 0.0))

    # Dominance by (t, node): keep all labels that are not strictly worse in
    # estimated primary uncertainty. Equal-uncertainty labels are retained.
    best_unc_at = Dict{Tuple{Int, Int}, Float64}()
    best_unc_at[(1, 1)] = state_unc[1]

    best_terminal_unc = Inf
    best_terminal_si = 0
    best_progress_unc = Inf
    best_progress_si = 0
    expanded = 0

    function partial_unc_estimate(si_est::Int)
        path_est = reconstruct_support_path(si_est)
        padded_est = pad_path_to_length(path_est, plen)
        return primary_goal_unc_with_support(padded_est)
    end

    while !isempty(pq)
        si = dequeue!(pq)
        node = state_nodes[si]
        t = state_t[si]
        dist_so_far = state_dist[si]
        cur_unc = state_unc[si]

        expanded += 1
        if debug_progress_every > 0 && mod(expanded, debug_progress_every) == 0
            if isfinite(cur_unc)
                best_progress_unc = min(best_progress_unc, cur_unc)
                best_progress_si = si
            end
            shown_unc = isfinite(best_terminal_unc) ? best_terminal_unc : min(best_progress_unc, cur_unc)
            shown_unc_txt = isfinite(shown_unc) ? string(round(shown_unc, digits=4)) : "n/a"
            println("  [Support $support_idx] $(debug_support_label) expanded=$expanded, t=$t/$plen, queue=$(length(pq)), best_unc=$(shown_unc_txt)")
        end

        if debug_on_frame !== nothing && debug_anim_sample_every > 0 && mod(expanded, debug_anim_sample_every) == 0
            partial_path = Int[]
            curp = si
            while curp != -1
                pushfirst!(partial_path, state_nodes[curp])
                curp = state_parent[curp]
            end
            debug_on_frame(partial_path, expanded, t)
        end

        if expanded > max_expansions
            break
        end

        if t == plen
            path = Int[]
            cur = si
            while cur != -1
                pushfirst!(path, state_nodes[cur])
                cur = state_parent[cur]
            end

            cand_unc = primary_goal_unc_with_support(path)
            if cand_unc < best_terminal_unc - 1e-9
                best_terminal_unc = cand_unc
                best_terminal_si = si
            end
            if cand_unc <= unc_threshold
                println("  [Support $support_idx] feasible terminal path found at expanded=$expanded, unc=$(round(cand_unc, digits=4))")
            end
            continue
        end

        next_t = t + 1

        # Support can either hold position or move by one graph edge each timestep.
        nbrs = copy(graph.neighbors[node])
        push!(nbrs, node)

        for u in nbrs
            edge_dist = (u == node) ? 0.0 : graph.distance[node, u]
            new_dist = dist_so_far + edge_dist

            # Prune only if the support can never catch up to 1σ communication
            # during the search, or reach COMM_RADIUS at the terminal timestep.
            if !can_ever_reconnect(u, next_t)
                continue
            end

            candidate_path = reconstruct_support_path(si)
            push!(candidate_path, u)
            candidate_path = pad_path_to_length(candidate_path, plen)
            new_unc = primary_goal_unc_with_support(candidate_path)

            key = (next_t, u)
            old_best = get(best_unc_at, key, Inf)
            if new_unc > old_best + 1e-9
                continue
            end

            push!(state_nodes, u)
            push!(state_t, next_t)
            push!(state_dist, new_dist)
            push!(state_parent, si)
            push!(state_unc, new_unc)
            new_si = length(state_nodes)
            if new_unc < old_best - 1e-9
                best_unc_at[key] = new_unc
            elseif !haskey(best_unc_at, key)
                best_unc_at[key] = new_unc
            end
            enqueue!(pq, new_si, (new_unc, new_dist))
        end
    end

    if best_terminal_si == 0
        if isfinite(best_progress_unc)
            progress_path = best_progress_si == 0 ? [1] : reconstruct_support_path(best_progress_si)
            return pad_path_to_length(progress_path, plen), best_progress_unc
        end
        return Int[], Inf
    end

    fallback_path = Int[]
    cur = best_terminal_si
    while cur != -1
        pushfirst!(fallback_path, state_nodes[cur])
        cur = state_parent[cur]
    end
    return pad_path_to_length(fallback_path, plen), best_terminal_unc
end

function first_feasible_primary_with_sequential_supports(graph::LandmarkGraph,
                                                         lms::Vector{Landmark},
                                                         unc_threshold::Float64,
                                                         num_agents::Int;
                                                         max_primary_expansions::Int=800000,
                                                         max_support_expansions::Int=100000,
                                                         max_primary_candidates::Int=2000,
                                                         debug_animate::Bool=false,
                                                         debug_gif_path::String="fig_seq_primary_support_search.gif",
                                                         debug_primary_progress_every::Int=0,
                                                         debug_support_progress_every::Int=0,
                                                         debug_anim_primary_sample_every::Int=0,
                                                         debug_anim_support_sample_every::Int=0)
    n = graph.n
    goal = n
    na = num_agents
    ns = max(na - 1, 0)

    states = PrimaryState[]
    init_vis = falses(n)
    init_vis[1] = true
    push!(states, PrimaryState(1, 0.0, -1, init_vis))

    pq = PriorityQueue{Int, Tuple{Float64, Float64}}()
    h0 = graph.shortest_paths[1, goal]
    enqueue!(pq, 1, (h0, 0.0))

    expanded = 0
    goal_candidates = 0
    anim = debug_animate ? Plots.Animation() : nothing

    function push_seq_frame(stage::String, iter_txt::String,
                            prim_path::Vector{Int}, sup_paths::Vector{Vector{Int}})
        debug_animate || return
        plt = plot_sequential_search_frame(graph, lms, prim_path, sup_paths;
                                           stage_label=stage,
                                           iter_label=iter_txt)
        frame(anim, plt)
    end

    function reconstruct_primary(si::Int)
        path = Int[]
        cur = si
        while cur != -1
            pushfirst!(path, states[cur].node)
            cur = states[cur].parent
        end
        return path
    end

    while !isempty(pq)
        si = dequeue!(pq)
        S = states[si]

        if debug_primary_progress_every > 0 && mod(expanded + 1, debug_primary_progress_every) == 0
            println("  [Primary Search] expanded=$(expanded + 1), queue=$(length(pq)), goal_candidates=$goal_candidates")
        end

        if debug_animate && debug_anim_primary_sample_every > 0 && mod(expanded + 1, debug_anim_primary_sample_every) == 0
            prim_dbg = reconstruct_primary(si)
            push_seq_frame("Primary frontier", "exp=$(expanded + 1)", prim_dbg, [Int[] for _ in 1:ns])
        end

        if S.node == goal
            goal_candidates += 1
            prim_path = reconstruct_primary(si)
            prim_dist = S.g

            println("  [Primary Search] Goal candidate #$goal_candidates dist=$(round(prim_dist, digits=3)), nodes=$(length(prim_path))")
            push_seq_frame("Primary goal candidate", "k=$goal_candidates", prim_path, [Int[] for _ in 1:ns])

            # First check whether this primary candidate is already feasible with
            # supports simply idling at start. This avoids false infeasibility when
            # support pruning is intentionally aggressive.
            idle_supports = [fill(1, length(prim_path)) for _ in 1:ns]
            idle_full_paths = vcat(idle_supports, [prim_path])
            idle_covs, idle_dists = evaluate_full_paths(idle_full_paths, graph, lms, na)
            idle_prim_unc = unc_radius(idle_covs[end])
            if idle_prim_unc <= unc_threshold
                println("  ✓ Primary-only feasible at candidate #$goal_candidates: dist=$(round(idle_dists[end], digits=3)), unc=$(round(idle_prim_unc, digits=4))")
                push_seq_frame("Primary-only feasible", "candidate#$goal_candidates", prim_path, idle_supports)
                if !REQUIRE_SUPPORT_REFINEMENT
                    if debug_animate
                        gif(anim, debug_gif_path, fps=15)
                        println("  [Seq Debug] Animation saved to $debug_gif_path")
                    end
                    return idle_full_paths, idle_dists, idle_prim_unc, goal_candidates
                end
                println("  [Seq Debug] REQUIRE_SUPPORT_REFINEMENT=true; continuing to refine supports anyway")
            end

            support_paths = Vector{Vector{Int}}(undef, ns)
            for sup_idx in 1:ns
                prior_supports = Vector{Vector{Int}}()
                for j in 1:(sup_idx - 1)
                    isempty(support_paths[j]) && continue
                    push!(prior_supports, support_paths[j])
                end
                local_support_frame = function(partial_support_path::Vector{Int}, sup_expanded::Int, step_t::Int)
                    dbg_supports = [Int[] for _ in 1:ns]
                    for jj in 1:(sup_idx - 1)
                        dbg_supports[jj] = support_paths[jj]
                    end
                    dbg_supports[sup_idx] = partial_support_path
                    push_seq_frame("Support $sup_idx search", "exp=$sup_expanded t=$step_t", prim_path, dbg_supports)
                end

                sup_path, _ = single_support_astar(graph, lms, prim_path, prim_dist, prior_supports, sup_idx;
                                                   unc_threshold=unc_threshold,
                                                   max_expansions=max_support_expansions,
                                                   debug_support_label="candidate#$goal_candidates",
                                                   debug_progress_every=debug_support_progress_every,
                                                   debug_anim_sample_every=debug_anim_support_sample_every,
                                                   debug_on_frame=local_support_frame)
                if isempty(sup_path)
                    support_paths[sup_idx] = Int[]
                    println("  [Support $sup_idx] no feasible support path for primary candidate #$goal_candidates")
                    break
                end
                support_paths[sup_idx] = sup_path
                push_seq_frame("Support $sup_idx accepted", "candidate#$goal_candidates", prim_path, support_paths)
            end

            if all(!isempty, support_paths)
                support_paths = pad_support_paths_to_primary(support_paths, prim_path)
                full_paths = vcat(support_paths, [prim_path])
                final_covs, final_dists = evaluate_full_paths(full_paths, graph, lms, na)
                prim_unc = unc_radius(final_covs[end])
                if prim_unc <= unc_threshold
                    println("  ✓ Sequential feasible solution at candidate #$goal_candidates: dist=$(round(final_dists[end], digits=3)), unc=$(round(prim_unc, digits=4))")
                    push_seq_frame("Sequential feasible solution", "candidate#$goal_candidates", prim_path, support_paths)
                    if debug_animate
                        gif(anim, debug_gif_path, fps=15)
                        println("  [Seq Debug] Animation saved to $debug_gif_path")
                    end
                    return full_paths, final_dists, prim_unc, goal_candidates
                end
            end

            if goal_candidates >= max_primary_candidates
                break
            end
            continue
        end

        expanded += 1
        if expanded > max_primary_expansions
            break
        end

        for u in graph.neighbors[S.node]
            S.visited[u] && continue
            new_g = S.g + graph.distance[S.node, u]
            new_h = graph.shortest_paths[u, goal]
            new_vis = copy(S.visited)
            new_vis[u] = true
            push!(states, PrimaryState(u, new_g, si, new_vis))
            new_si = length(states)
            enqueue!(pq, new_si, (new_g + new_h, new_g))
        end
    end

    if debug_animate
        gif(anim, debug_gif_path, fps=15)
        println("  [Seq Debug] Animation saved to $debug_gif_path")
    end

    return Vector{Vector{Int}}(), Float64[], Inf, goal_candidates
end
# Given a fixed primary path, find support paths that minimize primary's endpoint uncertainty.
# Returns (support_paths::Vector{Vector{Int}}, goal_uncertainty::Float64)

struct SupportState
    nodes   :: Vector{Int}          # current node per support agent
    dists   :: Vector{Float64}      # arc-distance per support
    covs    :: Vector{Matrix{Float64}}  # primary's covariance after fusions
    g       :: Float64              # primary's endpoint uncertainty
    parent  :: Int
    visited :: Vector{BitVector}
end

struct PrimaryState
    node::Int
    g::Float64
    parent::Int
    visited::BitVector
end

struct BeamPrimaryState
    path::Vector{Int}
    g::Float64
    cov::Matrix{Float64}
    unc::Float64
    score::Float64
    visited::BitVector
end

function pad_path_to_length(path::Vector{Int}, target_len::Int)
    isempty(path) && return Int[]
    padded = copy(path)
    while length(padded) < target_len
        push!(padded, padded[end])
    end
    return padded
end

function pad_support_paths_to_primary(support_paths::Vector{Vector{Int}}, primary_path::Vector{Int})
    target_len = length(primary_path)
    return [pad_path_to_length(path, target_len) for path in support_paths]
end

function plot_sequential_search_frame(graph::LandmarkGraph,
                                      lms::Vector{Landmark},
                                      primary_path::Vector{Int},
                                      support_paths::Vector{Vector{Int}};
                                      stage_label::String="",
                                      iter_label::String="")
    plt = plot(legend=:outerright, aspect_ratio=:equal,
               xlabel="x (m)", ylabel="y (m)",
               title="Sequential Search: $(stage_label) $(iter_label)")

    # Draw route nodes lightly.
    xs = [lm.x for lm in graph.landmarks]
    ys = [lm.y for lm in graph.landmarks]
    scatter!(plt, xs, ys, label="Nodes", color=:lightgray, markersize=2, markerstrokewidth=0)

    # Draw landmarks as darker points for context.
    if !isempty(lms)
        lx = [lm.x for lm in lms]
        ly = [lm.y for lm in lms]
        scatter!(plt, lx, ly, label="Landmarks", color=:black, markersize=4, markerstrokewidth=0)
    end

    # Plot support paths.
    for (i, sp) in enumerate(support_paths)
        isempty(sp) && continue
        sx = [graph.landmarks[j].x for j in sp]
        sy = [graph.landmarks[j].y for j in sp]
        plot!(plt, sx, sy, color=:purple, linewidth=1.5,
              linestyle=:dash, label="Support $i")
    end

    # Plot primary path.
    if !isempty(primary_path)
        px = [graph.landmarks[j].x for j in primary_path]
        py = [graph.landmarks[j].y for j in primary_path]
        plot!(plt, px, py, color=:blue, linewidth=2.2, label="Primary")
    end

    scatter!(plt, [graph.landmarks[1].x], [graph.landmarks[1].y],
             label="Start", color=:green, markersize=7, markerstrokewidth=0)
    scatter!(plt, [graph.landmarks[graph.n].x], [graph.landmarks[graph.n].y],
             label="Goal", color=:orange, marker=:star5, markersize=9, markerstrokewidth=0)

    return plt
end

function support_astar(graph::LandmarkGraph,
                       lms::Vector{Landmark},
                       primary_path::Vector{Int},
                       primary_path_dist::Float64,
                       unc_threshold::Float64,
                       num_supports::Int;
                       max_expansions::Int=200000)
    n = graph.n
    ns = num_supports

    # Compute primary's covariance trajectory along path (no supports initially)
    primary_covs = [copy(lms[primary_path[1]].cov)]
    for i in 1:length(primary_path)-1
        u, v = primary_path[i], primary_path[i+1]
        edge_cov = edge_cov_continuous(u, v, graph, lms, primary_covs[end])
        push!(primary_covs, edge_cov)
    end

    # Initial state: all supports at start
    init_nodes = fill(1, ns)
    init_dists = zeros(ns)
    init_covs = [copy(primary_covs[end]) for _ in 1:1]  # track primary's cov at goal
    init_visited = [falses(n) for _ in 1:ns]
    for a in 1:ns
        init_visited[a][1] = true
    end

    states = SupportState[]
    node_tuple_states = Dict{Vector{Int}, Vector{Int}}()

    init_state = SupportState(init_nodes, init_dists, init_covs, unc_radius(init_covs[1]), -1, init_visited)
    push!(states, init_state)
    node_tuple_states[copy(init_nodes)] = [1]

    pq = PriorityQueue{Int, Float64}()
    enqueue!(pq, 1, init_state.g)  # f = g (pure cost, no heuristic for now)

    best_goal_g = Inf
    best_goal_si = 0
    best_seen_g = init_state.g
    best_seen_si = 1

    while !isempty(pq)
        si = dequeue!(pq)
        S = states[si]

        tup = S.nodes
        if !haskey(node_tuple_states, tup) || si ∉ node_tuple_states[tup]
            continue
        end

        # Track best feasible solution
        if S.g <= unc_threshold && S.g < best_goal_g
            best_goal_g = S.g
            best_goal_si = si
        end
        if S.g < best_seen_g - 1e-9
            best_seen_g = S.g
            best_seen_si = si
        end

        # Expand: try moving each support
        for moved in 1:ns
            # Don't let supports travel beyond primary
            if S.dists[moved] >= primary_path_dist
                continue
            end

            for u in graph.neighbors[S.nodes[moved]]
                if S.visited[moved][u]
                    continue
                end

                edge_dist = graph.distance[S.nodes[moved], u]
                new_dist = S.dists[moved] + edge_dist

                # Support arc budget
                if new_dist >= primary_path_dist
                    continue
                end

                # Build new state
                new_nodes = copy(S.nodes)
                new_nodes[moved] = u
                new_dists = copy(S.dists)
                new_dists[moved] = new_dist

                # Main trick: track ALL communications between support's path and primary's path
                # A support at node u can communicate with primary at ANY node in primary_path
                # that is within COMM_RADIUS. We fuse at the earliest opportunity.
                ux, uy = graph.landmarks[u].x, graph.landmarks[u].y

                # Compute the best uncertainty reduction from this support position
                new_primary_cov = copy(primary_covs[end])
                best_fusion_point = -1

                for pi in 1:length(primary_path)
                    px = graph.landmarks[primary_path[pi]].x
                    py = graph.landmarks[primary_path[pi]].y
                    d2 = (ux - px)^2 + (uy - py)^2

                    if d2 <= COMM_RADIUS^2
                        # Support can communicate at primary's node pi
                        # Compute support's covariance at time of communication
                        sup_node = u
                        sup_cov = copy(lms[sup_node].cov)

                        # Propagate support covariance along its traveled path
                        if S.dists[moved] > 0
                            # Simplified: average angle along path (in practice, recompute from actual path)
                            sup_angle = graph.orientation[1, sup_node]
                            sup_cov = growth_covariance(S.dists[moved], sup_angle) .+ sup_cov
                        end

                        # Fuse at the primary's node pi
                        primary_cov_at_pi = primary_covs[pi]
                        fused = fuse_cov(primary_cov_at_pi, sup_cov)

                        # Forward-propagate from pi to goal
                        cov_to_goal = fused
                        for pj in pi+1:length(primary_path)-1
                            pj_u, pj_v = primary_path[pj], primary_path[pj+1]
                            cov_to_goal = edge_cov_continuous(pj_u, pj_v, graph, lms, cov_to_goal)
                        end

                        # Take fusion point that gives best (lowest) goal uncertainty
                        eig_to_goal = max_eigenvalue(cov_to_goal)
                        eig_no_fusion = max_eigenvalue(new_primary_cov)
                        if eig_to_goal < eig_no_fusion
                            new_primary_cov = cov_to_goal
                            best_fusion_point = pi
                        end
                    end
                end

                new_covs = [new_primary_cov]
                new_g = unc_radius(new_covs[1])

                # Dominance check
                dominated = false
                to_remove = Int[]
                existing = get(node_tuple_states, new_nodes, Int[])
                for old_si in existing
                    old = states[old_si]
                    if old.g <= new_g
                        dominated = true
                        break
                    end
                    if new_g < old.g
                        push!(to_remove, old_si)
                    end
                end
                if dominated
                    continue
                end

                for rem in to_remove
                    idx = findfirst(==(rem), node_tuple_states[new_nodes])
                    idx !== nothing && deleteat!(node_tuple_states[new_nodes], idx)
                end

                # Add new state
                new_visited = [copy(S.visited[a]) for a in 1:ns]
                new_visited[moved][u] = true
                push!(states, SupportState(new_nodes, new_dists, new_covs, new_g, si, new_visited))
                new_si = length(states)
                if !haskey(node_tuple_states, new_nodes)
                    node_tuple_states[new_nodes] = Int[]
                end
                push!(node_tuple_states[new_nodes], new_si)
                enqueue!(pq, new_si, new_g)
            end
        end
    end

    # Reconstruct support paths
    if best_goal_si == 0
        best_goal_si = best_seen_si
        best_goal_g = best_seen_g
    end

    agent_paths = [Int[] for _ in 1:ns]
    si = best_goal_si
    prev_nodes = fill(-1, ns)
    while si != -1
        S = states[si]
        for a in 1:ns
            if S.nodes[a] != get(prev_nodes, a, -1)
                pushfirst!(agent_paths[a], S.nodes[a])
            end
        end
        prev_nodes = copy(S.nodes)
        si = S.parent
    end

    for a in 1:ns
        unique_path = Int[]
        for nd in agent_paths[a]
            if isempty(unique_path) || nd != unique_path[end]
                push!(unique_path, nd)
            end
        end
        agent_paths[a] = pad_path_to_length(unique_path, length(primary_path))
    end

    return agent_paths, best_goal_g
end

# ---------- physics constants ----------
# BEARING_NOISE_RATIO : ratio of cross-bearing to along-bearing sensor noise
# COMM_INTERVAL_DIST : arc-distance between inter-agent communication events
const BEARING_NOISE_RATIO = 2.2               # cross-range noise 2.2× along-range—tighter sensor
const COMM_INTERVAL_DIST  = 15.0              # comm event every ~1500m of travel (15 units)


# ==========================================================================
# Physics-accurate covariance fusion
# ==========================================================================
#
# landmark_measurement_cov(agent_pos, lm, heading)
#   Returns the effective measurement noise covariance of observing `lm` from
#   `agent_pos` while heading in direction `heading`.
#
#   Model:
#     σ_range   = SENSOR_NOISE  (along line-of-sight)
#     σ_bearing = SENSOR_NOISE * BEARING_NOISE_RATIO  (perpendicular to LOS)
#   These are rotated into world frame by the bearing angle to the landmark.
#   The covariance is then INFLATED by 1/P_detect so that a low-probability
#   observation contributes less to the fusion — a barely-visible landmark
#   effectively has much higher noise.

# Returns the inverse of the measurement noise matrix (information form) directly,
# avoiding the allocation of S and then inv(S) separately.
# Returns nothing if landmark is outside detection range.
@inline function landmark_info(ax::Float64, ay::Float64, lm::Landmark)
    dx = lm.x - ax; dy = lm.y - ay
    d2 = dx*dx + dy*dy

    # Detection probability (soft range gate)
    p_detect = exp(-d2 / (2 * VISIBILITY_SIGMA^2))
    p_detect < 1e-6 && return nothing

    # Bearing angle to landmark (world frame) — R*D*R' expanded inline
    bearing = atan(dy, dx)
    cb = cos(bearing); sb = sin(bearing)
    σ_r2 = SENSOR_NOISE^2
    σ_b2 = (SENSOR_NOISE * BEARING_NOISE_RATIO)^2
    diff = σ_r2 - σ_b2
    # S_sensor = R*diag(σ_r2,σ_b2)*R' inline
    s11 = cb*cb*σ_r2 + sb*sb*σ_b2
    s12 = cb*sb*diff
    s22 = sb*sb*σ_r2 + cb*cb*σ_b2

    # S_total = S_sensor + lm.cov, inflated by 1/p_detect
    inv_p = 1.0 / p_detect
    t11 = (s11 + lm.cov[1,1]) * inv_p
    t12 = (s12 + lm.cov[1,2]) * inv_p
    t22 = (s22 + lm.cov[2,2]) * inv_p

    # Return inv(S_total) directly (2×2 inline inverse)
    det = t11*t22 - t12*t12
    det < 1e-14 && return nothing
    inv_det = 1.0 / det
    return (t22*inv_det, -t12*inv_det, t11*inv_det)  # (I11, I12, I22) of information matrix
end

# propagate_cov_discrete: covariance propagation at discrete waypoints
# Input: positions (x,y) at waypoints
# Outputs: covariance at each waypoint after dead-reckoning + landmark fusion

function propagate_cov_discrete(positions::Vector{Tuple{Float64,Float64}},
                                 lms::Vector{Landmark},
                                 init_cov::Matrix{Float64};
                                 debug_goal_pos::Union{Tuple{Float64,Float64}, Nothing} = nothing,
                                 debug_agent_id::Union{Int, Nothing} = nothing)
    n    = length(positions)
    covs = Vector{Matrix{Float64}}(undef, n)
    cov  = copy(init_cov)
    covs[1] = copy(cov)
    
    fusion_count = 0  # Track number of Kalman fusion events

    for i in 2:n
        x_prev, y_prev = positions[i-1]
        x_curr, y_curr = positions[i]
        
        # Segment distance and heading
        seg = hypot(x_curr - x_prev, y_curr - y_prev)
        heading = atan(y_curr - y_prev, x_curr - x_prev)
        
        # 1. Dead-reckoning growth
        cov = cov + growth_covariance(seg, heading)
        
        # Store pre-fusion covariance for debugging if at goal
        cov_before_fusion = copy(cov)
        
        # 2. Landmark fusion at current position (Kalman update via information filter)
        I11 = 0.0; I12 = 0.0; I22 = 0.0
        for lm in lms
            max_eigenvalue(lm.cov) < 1e-8 && continue
            info = landmark_info(x_curr, y_curr, lm)
            info === nothing && continue
            I11 += info[1]; I12 += info[2]; I22 += info[3]
        end
        if I11 > 0.0 || I22 > 0.0
            # Information filter (Joseph form) Kalman update: combine prior covariance with landmark info
            det_c = cov[1,1]*cov[2,2] - cov[1,2]*cov[2,1]
            inv_det = 1.0 / det_c
            J11 = I11 + cov[2,2]*inv_det
            J12 = I12 - cov[1,2]*inv_det
            J22 = I22 + cov[1,1]*inv_det
            det_j = J11*J22 - J12*J12
            inv_dj = 1.0 / det_j
            cov = [J22*inv_dj  -J12*inv_dj; -J12*inv_dj  J11*inv_dj]
            fusion_count += 1
        end
        covs[i] = copy(cov)
    end
    return covs
end

# Continuous covariance propagation along a B-spline path
# Input: xs, ys (coordinate arrays), lms (landmarks), init_cov (initial covariance)
# Output: covs (covariance at each sample point along path)
function propagate_cov_continuous(xs::Vector{Float64},
                                   ys::Vector{Float64},
                                   lms::Vector{Landmark},
                                   init_cov::Matrix{Float64};
                                   explicit_headings::Union{Vector{Float64}, Nothing} = nothing)
    n    = length(xs)
    covs = Vector{Matrix{Float64}}(undef, n)
    cov  = copy(init_cov)
    covs[1] = copy(cov)

    for i in 2:n
        x_prev, y_prev = xs[i-1], ys[i-1]
        x_curr, y_curr = xs[i], ys[i]
        
        # Segment distance and heading
        seg = hypot(x_curr - x_prev, y_curr - y_prev)
        if seg < 1e-10
            covs[i] = copy(cov)
            continue
        end
        heading = explicit_headings !== nothing ? explicit_headings[i] : atan(y_curr - y_prev, x_curr - x_prev)
        
        # 1. Dead-reckoning growth
        cov = cov + growth_covariance(seg, heading)
        
        # 2. Landmark fusion at current position
        I11 = 0.0; I12 = 0.0; I22 = 0.0
        for lm in lms
            max_eigenvalue(lm.cov) < 1e-8 && continue
            info = landmark_info(x_curr, y_curr, lm)
            info === nothing && continue
            I11 += info[1]; I12 += info[2]; I22 += info[3]
        end
        if I11 > 0.0 || I22 > 0.0
            # Information filter update
            det_c = cov[1,1]*cov[2,2] - cov[1,2]*cov[2,1]
            abs(det_c) < 1e-20 && (covs[i] = copy(cov); continue)
            inv_det = 1.0 / det_c
            J11 = I11 + cov[2,2]*inv_det
            J12 = I12 - cov[1,2]*inv_det
            J22 = I22 + cov[1,1]*inv_det
            det_j = J11*J22 - J12*J12
            abs(det_j) < 1e-20 && (covs[i] = copy(cov); continue)
            inv_dj = 1.0 / det_j
            cov = [J22*inv_dj  -J12*inv_dj; -J12*inv_dj  J11*inv_dj]
        end
        covs[i] = copy(cov)
    end
    return covs
end

# Helper function to convert (xs, ys) lists to position tuples
function xs_ys_to_positions(xs::Vector{Vector{Float64}}, ys::Vector{Vector{Float64}})
    na = length(xs)
    agent_positions = Vector{Vector{Tuple{Float64,Float64}}}(undef, na)
    for a in 1:na
        positions = Vector{Tuple{Float64,Float64}}(undef, length(xs[a]))
        for i in 1:length(xs[a])
            positions[i] = (xs[a][i], ys[a][i])
        end
        agent_positions[a] = positions
    end
    return agent_positions
end

# ==========================================================================
# Discrete multi-agent covariance evaluation
# ==========================================================================
# Evaluate all agents at their waypoint positions with synchronized Kalman communication

function evaluate_joint_discrete(agent_positions::Vector{Vector{Tuple{Float64,Float64}}},
                                  lms::Vector{Landmark},
                                  na::Int;
                                  debug_goal_pos::Union{Tuple{Float64,Float64}, Nothing} = nothing)
    # Evaluate covariance for each agent at their waypoint positions with synchronized comms
    # All agents propagate in lock-step, fusing at fixed distance intervals
    
    # Compute arc lengths for all agents first
    all_arcs = Vector{Vector{Float64}}(undef, na)
    
    for a in 1:na
        if isempty(agent_positions[a])
            all_arcs[a] = [0.0]
        else
            arcs = Vector{Float64}(undef, length(agent_positions[a]))
            arcs[1] = 0.0
            for i in 2:length(agent_positions[a])
                x0, y0 = agent_positions[a][i-1]
                x1, y1 = agent_positions[a][i]
                arcs[i] = arcs[i-1] + hypot(x1-x0, y1-y0)
            end
            all_arcs[a] = arcs
        end
    end
    
    # Propagate covariances with synchronized Kalman fusion
    all_covs = apply_synchronized_propagation!(agent_positions, all_arcs, lms, na; 
                                               debug_goal_pos=debug_goal_pos)
    
    return all_covs, all_arcs
end

function apply_synchronized_propagation!(agent_positions::Vector{Vector{Tuple{Float64,Float64}}},
                                        all_arcs::Vector{Vector{Float64}},
                                        lms::Vector{Landmark},
                                        na::Int;
                                        debug_goal_pos::Union{Tuple{Float64,Float64}, Nothing} = nothing)
    # Propagate each agent independently, then apply synchronized Kalman fusion
    # at fixed communication checkpoints
    
    # First: independent propagation for each agent
    all_covs = Vector{Vector{Matrix{Float64}}}(undef, na)
    
    for a in 1:na
        if isempty(agent_positions[a])
            all_covs[a] = [copy(lms[1].cov)]
        else
            covs = propagate_cov_discrete(agent_positions[a], lms, lms[1].cov;
                                         debug_goal_pos=debug_goal_pos, debug_agent_id=a)
            all_covs[a] = covs
        end
    end
    
    # Second: apply synchronized Kalman fusion at communication checkpoints
    max_arc = maximum(arcs[end] for arcs in all_arcs)
    comm_times = 0.0:COMM_INTERVAL:max_arc
    
    for comm_time in comm_times
        # Get indices of waypoints nearest to this communication time
        agent_indices = Vector{Int}(undef, na)
        agent_positions_at_time = Vector{Union{Tuple{Float64,Float64}, Nothing}}(undef, na)
        
        for a in 1:na
            # Find waypoint nearest to comm_time
            nearest_idx = 1
            min_diff = abs(all_arcs[a][1] - comm_time)
            for i in 2:length(all_arcs[a])
                diff = abs(all_arcs[a][i] - comm_time)
                if diff < min_diff
                    min_diff = diff
                    nearest_idx = i
                end
            end
            agent_indices[a] = nearest_idx
            agent_positions_at_time[a] = agent_positions[a][nearest_idx]
        end
        
        # Apply pairwise synchronized Kalman fusion with tapered Gaussian
        for sender in 1:na
            pos_s = agent_positions_at_time[sender]
            idx_s = agent_indices[sender]
            
            for receiver in sender+1:na
                pos_r = agent_positions_at_time[receiver]
                idx_r = agent_indices[receiver]
                
                dx = pos_s[1] - pos_r[1]
                dy = pos_s[2] - pos_r[2]
                dist = hypot(dx, dy)
                
                # Tapered Gaussian weight: exp(-dist² / (2σ²))
                weight = exp(-dist^2 / (2 * COMM_SIGMA^2))
                
                if weight > 1e-4  # Only fuse if weight is significant
                    # Bidirectional Kalman fusion via information filter
                    S_s = all_covs[sender][idx_s] + SENSOR_NOISE^2 * I(2)
                    S_r = all_covs[receiver][idx_r] + SENSOR_NOISE^2 * I(2)
                    
                    # Receiver fuses sender
                    inv_P_r = inv(all_covs[receiver][idx_r])
                    inv_S_s = inv(S_s)
                    new_inv_P_r = inv_P_r + weight * inv_S_s
                    all_covs[receiver][idx_r] = inv(new_inv_P_r)
                    
                    # Sender fuses receiver (bidirectional)
                    inv_P_s = inv(all_covs[sender][idx_s])
                    inv_S_r = inv(S_r)
                    new_inv_P_s = inv_P_s + weight * inv_S_r
                    all_covs[sender][idx_s] = inv(new_inv_P_s)
                end
            end
        end
    end
    
    return all_covs
end

# ----------------------
# Continuous edge covariance helper
# ----------------------
# Evaluate the covariance at the far end of a single graph edge using the
# same continuous physics as the B-spline evaluator (bearing-angle sensor
# noise, detection probability).  Used by both Dijkstra searches so that
# the search operates on the same uncertainty model as the final evaluation.
#
# EDGE_SAMPLES controls resolution: enough to catch landmark crossings but
# cheap enough to keep search fast.  At graph scale (~1-20 units per edge)
# 8 samples gives sub-0.1% accuracy vs 100 samples.

# SAMPLE_SPACING: physical distance between covariance evaluation points (units).
# Both Dijkstra edges and B-splines use this, guaranteeing identical fusion density.
const SAMPLE_SPACING       = 5.0   # 5 graph units = 500m — used for Bézier/final eval
const SAMPLE_SPACING_SEARCH = 5.0  # same as SAMPLE_SPACING — coarser spacing caused
                                    # landmark fusion misses on long edges (up to 32 units)
                                    # leading to inconsistent evaluations between search
                                    # and final simulation.  Runtime cost is acceptable.

function edge_cov_continuous(v::Int, u::Int,
                              graph::LandmarkGraph,
                              lms::Vector{Landmark},
                              cov_in::Matrix{Float64})
    # Direct covariance propagation along path edge
    vx = graph.landmarks[v].x; vy = graph.landmarks[v].y
    ux = graph.landmarks[u].x; uy = graph.landmarks[u].y
    h_bearing = atan(uy - vy, ux - vx)
    
    # Sample along straight line between waypoints
    xs = [vx, ux]
    ys = [vy, uy]
    covs = propagate_cov_continuous(xs, ys, lms, cov_in; explicit_headings=[h_bearing, h_bearing])
    return covs[end]
end

# ==========================================================================
# Joint A* over all agents simultaneously
# ==========================================================================
#
# STATE
# -----
# A JointState encodes the full multi-agent configuration at one instant:
#   nodes   : current graph node for each agent (length num_agents)
#   dists   : cumulative arc-distance travelled by each agent
#   covs    : position-estimate covariance for each agent
#   g       : true cost = primary arc-distance (agents[end] is primary)
#   parent  : index into the states vector (-1 = root)
#   visited : per-agent BitVector for cycle detection
#
# HEURISTICS
# ----------
# Each agent has its own admissible heuristic:
#
#   Primary (last agent):
#     h_primary(v) = Euclidean distance from v to goal node.
#     This lower-bounds the remaining primary arc-distance, so
#     f = g + h_primary is admissible (never overestimates total primary length).
#
#   Support agent a:
#     The support does not need to reach the goal — it only needs to position
#     itself so that inter-agent comms help the primary. Its heuristic
#     contribution to f is ZERO (supports add no remaining mandatory cost to
#     the primary objective). Using a non-zero heuristic for supports would
#     risk inadmissibility because we cannot bound how far a support must
#     still travel.  h_support = 0 is trivially admissible.
#
# ADMISSIBILITY PROOF
# -------------------
# f(s) = g(s) + h(s)
#       = primary_dist_so_far + h_primary(primary_node)
#       ≤ primary_dist_so_far + remaining_primary_dist   (Euclidean ≤ path dist)
#       ≤ true_total_primary_dist
# So f never overestimates — A* is admissible and returns optimal primary length.
#
# DOMINANCE / PARETO PRUNING
# --------------------------
# Two states at the same joint node-tuple (nodes[1..K]) are compared on:
#   (primary_dist, max_eigenvalue(primary_cov))
# State A dominates B iff A.g ≤ B.g AND λ_max(A.covs[end]) ≤ λ_max(B.covs[end]).
# Dominated states are discarded.  This is sound because:
#   (a) A's remaining primary cost cannot exceed B's (same heuristic from same node)
#   (b) A's primary uncertainty at goal cannot exceed B's (covariance only grows)
# so B can never produce a solution better than A.
#
# INTER-AGENT COMMUNICATION WITHIN THE SEARCH
# --------------------------------------------
# To keep the heuristic consistent with the final evaluate_joint_discrete evaluation we
# apply the same communication model inside edge_cov_continuous — but only
# when the arc-distance gap between agents falls within one COMM_INTERVAL_DIST
# window.  For each edge expansion we compute a lightweight pairwise comm
# update between the newly-expanded agent and each other agent at their
# current arc position, mirroring apply_inter_agent_discrete_comms! at graph resolution.
#
# CYCLE DETECTION
# ---------------
# Each agent carries its own BitVector of visited nodes so that the same node
# is never revisited on one agent's path (prevents infinite loops).  Different
# agents may visit the same node independently.
#
# COMPLEXITY NOTE
# ---------------
# The joint state space is O(n^K) for K agents and n nodes.  With K=3 and
# n≈130 nodes the naive bound is ~2.2M node-tuples.  Pareto pruning keeps the
# live frontier small in practice: each node-tuple retains only its Pareto-
# non-dominated (dist, unc) states, which are few for well-separated landmarks.
# The coarse comm model inside the search (graph-resolution, not Bézier-
# resolution) is an approximation — the Bézier stage later refines it — but
# it is consistent with the heuristic and keeps each expansion O(n_lm) cheap.
#
# EXPANSION STRATEGY
# ------------------
# At each step we advance ALL agents by one edge (round-robin by agent index),
# keeping the joint state synchronised by arc-distance.  Specifically, we
# always expand the agent that is furthest behind in arc-distance, breaking
# ties by agent index.  This mimics simultaneous travel without exponentiating
# the branching factor: at each expansion only one agent moves.
#
# The primary's arc-distance drives the f-value.  Support agents' arc-distance
# is bounded to [0, primary_dist] — supports must not travel further than
# the primary.

struct JointState
    paths   :: Vector{Vector{Int}}  # full path per agent (sequence of nodes); last = primary
    covs    :: Vector{Matrix{Float64}}  # covariance at end of each path
    dists   :: Vector{Float64}      # arc-distance per agent (computed from B-spline)
    g       :: Float64              # primary arc-distance (cost)
    parent  :: Int                  # index in states vector; -1 = root
    visited :: Vector{BitVector}    # per-agent visited sets (cycle detection)
end

# ------------------------------------------------------------------
# Lightweight single-step comm: fuse agent a's cov into agent b's cov
# (and vice-versa) if they are within COMM_RADIUS and a comm event
# threshold has been crossed since the last fusion.
# Returns (new_cov_a, new_cov_b).
# ------------------------------------------------------------------
@inline function pairwise_comm(cov_a::Matrix{Float64}, cov_b::Matrix{Float64},
                                xa::Float64, ya::Float64,
                                xb::Float64, yb::Float64)
    d2 = (xa-xb)^2 + (ya-yb)^2
    w  = exp(-d2 / (2*COMM_RADIUS^2))
    w < 1e-3 && return cov_a, cov_b
    # Information-filter fusion weighted by range
    noise = SENSOR_NOISE^2
    # a receives from b
    Ib = inv2(cov_b .+ noise .* [1.0 0.0; 0.0 1.0])
    new_a = inv2(inv2(cov_a) .+ w .* Ib)
    # b receives from a
    Ia = inv2(cov_a .+ noise .* [1.0 0.0; 0.0 1.0])
    new_b = inv2(inv2(cov_b) .+ w .* Ia)
    return new_a, new_b
end

# ------------------------------------------------------------------
# Apply all pairwise comms for a newly-expanded agent `moved` given
# the updated joint state.  Each pair (moved, other) is checked once.
# Returns updated covs vector (copy).
# ------------------------------------------------------------------
function apply_step_comms(covs::Vector{Matrix{Float64}},
                           nodes::Vector{Int},
                           dists::Vector{Float64},
                           graph::LandmarkGraph,
                           moved::Int)
    na   = length(covs)
    covs = copy(covs)   # shallow copy vector; matrices copied below as needed
    xm   = graph.landmarks[nodes[moved]].x
    ym   = graph.landmarks[nodes[moved]].y
    for b in 1:na
        b == moved && continue
        # Only comm if arc-distances are close enough to share a comm window
        abs(dists[moved] - dists[b]) > COMM_INTERVAL_DIST && continue
        xb = graph.landmarks[nodes[b]].x
        yb = graph.landmarks[nodes[b]].y
        new_m, new_b = pairwise_comm(covs[moved], covs[b], xm, ym, xb, yb)
        covs[moved] = new_m
        covs[b]     = new_b
    end
    return covs
end

# ------------------------------------------------------------------
# Admissible heuristic for the joint state:
#   h = shortest graph path distance from primary's current node to goal.
# This is tighter than Euclidean distance and accounts for graph topology.
# Precomputed via Floyd-Warshall, so lookup is O(1).
# Support heuristics are 0 (see proof above).
# ------------------------------------------------------------------
@inline function joint_heuristic(paths::Vector{Vector{Int}},
                                  goal::Int,
                                  graph::LandmarkGraph)
    primary = length(paths)
    if isempty(paths[primary])
        return 0.0  # no progress yet
    end
    v = paths[primary][end]  # current node of primary
    return graph.shortest_paths[v, goal]
end

# ------------------------------------------------------------------
# Secondary tie-breaker score for support idling.
# Lower is better. Used after weighted primary f-values are compared.
# ------------------------------------------------------------------
@inline function support_idle_score(dists::Vector{Float64})
    na = length(dists)
    idle_count = 0
    for a in 1:(na - 1)
        dists[a] <= 1e-9 && (idle_count += 1)
    end
    return SUPPORT_IDLE_PENALTY * idle_count
end

@inline function joint_node_key(paths::Vector{Vector{Int}})
    nodes = Vector{Int}(undef, length(paths))
    for a in 1:length(paths)
        nodes[a] = paths[a][end]
    end
    return Tuple(nodes)
end

@inline function joint_path_key(paths::Vector{Vector{Int}})
    key = Vector{Tuple{Vararg{Int}}}(undef, length(paths))
    for a in 1:length(paths)
        key[a] = Tuple(paths[a])
    end
    return Tuple(key)
end

@inline function joint_visited_key(visited::Vector{BitVector})
    sig = Vector{Tuple{Vararg{Int}}}(undef, length(visited))
    for a in 1:length(visited)
        sig[a] = Tuple(findall(visited[a]))
    end
    return Tuple(sig)
end

function apply_joint_step_comms(covs::Vector{Matrix{Float64}},
                                nodes::Vector{Int},
                                dists::Vector{Float64},
                                graph::LandmarkGraph)
    na = length(covs)
    updated = copy(covs)
    for a in 1:(na - 1)
        xa = graph.landmarks[nodes[a]].x
        ya = graph.landmarks[nodes[a]].y
        for b in (a + 1):na
            abs(dists[a] - dists[b]) > COMM_INTERVAL_DIST && continue
            xb = graph.landmarks[nodes[b]].x
            yb = graph.landmarks[nodes[b]].y
            new_a, new_b = pairwise_comm(updated[a], updated[b], xa, ya, xb, yb)
            updated[a] = new_a
            updated[b] = new_b
        end
    end
    return updated
end

# ------------------------------------------------------------------
# Evaluate complete paths using discrete waypoint positions
# Input:  agent_paths (node sequences), graph, landmarks, num_agents
# Output: (covs, dists) where each is a vector per agent
#
# This function:
#   1. Converts each node path to (x,y) waypoints
#   2. Evaluates joint covariance via evaluate_joint_discrete (includes inter-agent comms)
#   3. Computes path distances as sum of edge distances
# ------------------------------------------------------------------
function evaluate_full_paths(agent_paths::Vector{Vector{Int}},
                              graph::LandmarkGraph,
                              lms::Vector{Landmark},
                              na::Int)
    # Convert paths to waypoint positions
    all_xs = Vector{Float64}[]
    all_ys = Vector{Float64}[]
    all_dists = Vector{Float64}[]
    
    for a in 1:na
        path = agent_paths[a]
        if isempty(path)
            push!(all_xs, Float64[])
            push!(all_ys, Float64[])
            push!(all_dists, Float64[])
        else
            xs = [graph.landmarks[i].x for i in path]
            ys = [graph.landmarks[i].y for i in path]
            push!(all_xs, xs)
            push!(all_ys, ys)
            
            # Compute cumulative distances along path
            dists = [0.0]
            for i in 2:length(path)
                edge_dist = graph.distance[path[i-1], path[i]]
                push!(dists, dists[end] + edge_dist)
            end
            push!(all_dists, dists)
        end
    end
    
    # Evaluate joint covariance via evaluate_joint_discrete (includes inter-agent communication)
    agent_positions = xs_ys_to_positions(all_xs, all_ys)
    all_covs, _ = evaluate_joint_discrete(agent_positions, lms, na)
    
    # Extract final covariances and distances
    final_covs = [all_covs[a][end] for a in 1:na]
    final_dists = [all_dists[a][end] for a in 1:na]
    
    return final_covs, final_dists
end

# Primary-only uncertainty-constrained A* with optional modeled support effect.
# Returns (path, dist, goal_unc). If no feasible path is found, path is empty.
function primary_uncertainty_astar(graph::LandmarkGraph,
                                   lms::Vector{Landmark},
                                   unc_threshold::Float64,
                                   num_agents::Int;
                                   modeled_support_paths::Union{Nothing, Vector{Vector{Int}}}=nothing,
                                   max_expansions::Int=200000)
    n = graph.n
    goal = n
    na = num_agents
    ns = max(na - 1, 0)

    states = PrimaryState[]
    init_vis = falses(n)
    init_vis[1] = true
    push!(states, PrimaryState(1, 0.0, -1, init_vis))

    pq = PriorityQueue{Int, Float64}()
    enqueue!(pq, 1, graph.shortest_paths[1, goal])

    best_feasible_dist = Inf
    best_feasible_unc = Inf
    best_feasible_si = 0
    expanded = 0

    node_best_g = fill(Inf, n)
    node_best_g[1] = 0.0

    function reconstruct_primary_path(si::Int)
        path = Int[]
        cur = si
        while cur != -1
            pushfirst!(path, states[cur].node)
            cur = states[cur].parent
        end
        return path
    end

    while !isempty(pq)
        si = dequeue!(pq)
        S = states[si]

        # Conservative transposition pruning by node-distance only.
        if S.g > node_best_g[S.node] + 1e-9
            continue
        end

        h = graph.shortest_paths[S.node, goal]
        if S.g + h >= best_feasible_dist - 1e-9
            continue
        end

        if S.node == goal
            prim_path = reconstruct_primary_path(si)

            support_paths = if modeled_support_paths === nothing
                [copy(prim_path) for _ in 1:ns]
            else
                modeled_support_paths
            end

            full_paths = vcat(support_paths, [prim_path])
            final_covs, final_dists = evaluate_full_paths(full_paths, graph, lms, na)
            prim_unc = unc_radius(final_covs[na])
            prim_dist = final_dists[na]

            if prim_unc <= unc_threshold && prim_dist < best_feasible_dist - 1e-9
                best_feasible_dist = prim_dist
                best_feasible_unc = prim_unc
                best_feasible_si = si
            end
            continue
        end

        expanded += 1
        if expanded > max_expansions
            break
        end

        for u in graph.neighbors[S.node]
            S.visited[u] && continue
            edge = graph.distance[S.node, u]
            new_g = S.g + edge
            new_h = graph.shortest_paths[u, goal]
            new_f = new_g + new_h
            if new_f >= best_feasible_dist - 1e-9
                continue
            end

            # Keep only better distance labels at each node.
            if new_g >= node_best_g[u] - 1e-9
                continue
            end

            new_vis = copy(S.visited)
            new_vis[u] = true
            push!(states, PrimaryState(u, new_g, si, new_vis))
            new_si = length(states)
            node_best_g[u] = new_g
            enqueue!(pq, new_si, new_f)
        end
    end

    if best_feasible_si == 0
        return Int[], Inf, Inf
    end
    return reconstruct_primary_path(best_feasible_si), best_feasible_dist, best_feasible_unc
end

function beam_primary_uncertainty_seed(graph::LandmarkGraph,
                                       lms::Vector{Landmark},
                                       unc_threshold::Float64,
                                       num_agents::Int;
                                       modeled_support_paths::Union{Nothing, Vector{Vector{Int}}}=nothing,
                                       beam_width::Int=24,
                                       uncertainty_weight::Float64=6.0,
                                       max_depth::Int=0)
    n = graph.n
    goal = n
    na = num_agents
    ns = max(na - 1, 0)
    max_depth = max_depth > 0 ? max_depth : n

    init_cov = copy(lms[1].cov)
    init_visited = falses(n)
    init_visited[1] = true
    init_score = graph.shortest_paths[1, goal]
    frontier = [BeamPrimaryState([1], 0.0, init_cov, unc_radius(init_cov), init_score, init_visited)]

    best_path = Int[]
    best_dist = Inf
    best_unc = Inf
    best_goal_path = Int[]
    best_goal_dist = Inf
    best_goal_unc = Inf

    for _depth in 1:(max_depth - 1)
        candidates = BeamPrimaryState[]

        for S in frontier
            last_node = S.path[end]
            if last_node == goal
                support_paths = if modeled_support_paths === nothing
                    [copy(S.path) for _ in 1:ns]
                else
                    modeled_support_paths
                end
                full_paths = vcat(support_paths, [S.path])
                final_covs, final_dists = evaluate_full_paths(full_paths, graph, lms, na)
                cur_unc = unc_radius(final_covs[na])
                cur_dist = final_dists[na]
                if cur_dist < best_goal_dist - 1e-9
                    best_goal_path = copy(S.path)
                    best_goal_dist = cur_dist
                    best_goal_unc = cur_unc
                end
                if cur_unc <= unc_threshold && cur_dist < best_dist - 1e-9
                    best_path = copy(S.path)
                    best_dist = cur_dist
                    best_unc = cur_unc
                end
                continue
            end

            for u in graph.neighbors[last_node]
                S.visited[u] && continue
                edge = graph.distance[last_node, u]
                new_g = S.g + edge
                new_h = graph.shortest_paths[u, goal]
                if new_g + new_h >= best_dist - 1e-9
                    continue
                end

                new_cov = edge_cov_continuous(last_node, u, graph, lms, S.cov)
                new_unc = unc_radius(new_cov)
                new_score = new_g + new_h + uncertainty_weight * max(0.0, new_unc - unc_threshold)

                new_path = copy(S.path)
                push!(new_path, u)
                new_visited = copy(S.visited)
                new_visited[u] = true
                push!(candidates, BeamPrimaryState(new_path, new_g, new_cov, new_unc, new_score, new_visited))
            end
        end

        isempty(candidates) && break

        best_by_node = Dict{Int, BeamPrimaryState}()
        for cand in candidates
            last_node = cand.path[end]
            if !haskey(best_by_node, last_node)
                best_by_node[last_node] = cand
                continue
            end
            old = best_by_node[last_node]
            if cand.score < old.score - 1e-9 || (abs(cand.score - old.score) <= 1e-9 && cand.g < old.g - 1e-9)
                best_by_node[last_node] = cand
            end
        end

        frontier = collect(values(best_by_node))
        sort!(frontier, by = s -> (s.score, s.g))
        if length(frontier) > beam_width
            frontier = frontier[1:beam_width]
        end
    end

    if isempty(best_path)
        if !isempty(best_goal_path)
            return best_goal_path, best_goal_dist, best_goal_unc
        end
        return Int[], Inf, Inf
    end
    return best_path, best_dist, best_unc
end

function refine_support_paths_for_primary(graph::LandmarkGraph,
                                          primary_path::Vector{Int},
                                          primary_dist::Float64,
                                          num_agents::Int;
                                          max_support_iters::Int=3)
    ns = max(num_agents - 1, 0)
    support_paths = [Int[] for _ in 1:ns]

    for _support_iter in 1:max_support_iters
        changed = false
        for sup_idx in 1:ns
            other_paths = Vector{Vector{Int}}()
            for j in 1:ns
                j == sup_idx && continue
                isempty(support_paths[j]) && continue
                push!(other_paths, support_paths[j])
            end

            sup_path, _ = single_support_astar(graph, graph.landmarks, primary_path, primary_dist, other_paths, sup_idx)
            new_path = isempty(sup_path) ? [1] : sup_path
            if new_path != support_paths[sup_idx]
                support_paths[sup_idx] = new_path
                changed = true
            end
        end
        !changed && break
    end

    return support_paths
end

# Build a feasible incumbent by alternating:
#  1) beam-search primary solve under the uncertainty bound,
#  2) iterative support optimization against that primary,
#  3) primary re-solve with support effect modeled,
# until primary path stabilizes.
function build_joint_incumbent_seed(graph::LandmarkGraph,
                                    lms::Vector{Landmark},
                                    unc_threshold::Float64,
                                    num_agents::Int;
                                    max_outer_iters::Int=8,
                                    beam_width::Int=24,
                                    support_max_expansions::Int=200000)
    na = num_agents
    primary = na

    best_paths = nothing
    best_dists = nothing
    best_unc = Inf
    best_dist = Inf

    seed_start_t = time()

    # Initial primary plan uses beam search and an optimistic support model.
    prim_path, prim_dist, prim_unc = Int[], Inf, Inf
    for current_beam_width in (beam_width, beam_width * 2, beam_width * 4)
        prim_path, prim_dist, prim_unc = beam_primary_uncertainty_seed(
            graph, lms, unc_threshold, na;
            beam_width=current_beam_width,
            modeled_support_paths=nothing
        )
        !isempty(prim_path) && break
    end
    if isempty(prim_path)
        println("    [Warm-start] beam-search primary solve found no goal-reaching path")
        println("    [Warm-start] complete in $(round(time() - seed_start_t, digits=2))s")
        return nothing
    end
    if prim_unc > unc_threshold
        println("    [Warm-start] initial beam primary is not yet feasible; optimizing supports and re-planning")
    end

    prev_primary = Int[]
    for outer in 1:max_outer_iters
        println("    [Warm-start] iter=$outer primary_dist=$(round(prim_dist, digits=3)) primary_unc=$(round(prim_unc, digits=4))")

        # Step 2: optimize all support agents against the fixed primary path.
        support_paths, support_unc = support_astar(
            graph, lms, prim_path, prim_dist, unc_threshold, na - 1;
            max_expansions=support_max_expansions
        )
        println("    [Warm-start] support refine unc=$(round(support_unc, digits=4))")

        # Evaluate current alternating iterate.
        full_paths = vcat(support_paths, [prim_path])
        final_covs, final_dists = evaluate_full_paths(full_paths, graph, lms, na)
        cur_unc = unc_radius(final_covs[primary])
        cur_dist = final_dists[primary]

        if cur_unc <= unc_threshold && cur_dist < best_dist - 1e-9
            best_paths = full_paths
            best_dists = final_dists
            best_unc = cur_unc
            best_dist = cur_dist
            println("    [Warm-start] feasible dist=$(round(cur_dist, digits=3)) unc=$(round(cur_unc, digits=4))")
        end

        # Step 3: re-plan primary with support effect modeled.
        new_prim_path, new_prim_dist, new_prim_unc = beam_primary_uncertainty_seed(
            graph, lms, unc_threshold, na;
            modeled_support_paths=support_paths,
            beam_width=beam_width
        )

        if isempty(new_prim_path)
            println("    [Warm-start] replan primary produced no feasible path; keep best-so-far")
            break
        end

        # Converged if path does not change.
        if new_prim_path == prim_path || new_prim_path == prev_primary
            prim_path, prim_dist, prim_unc = new_prim_path, new_prim_dist, new_prim_unc
            break
        end

        prev_primary = prim_path
        prim_path, prim_dist, prim_unc = new_prim_path, new_prim_dist, new_prim_unc
    end

    println("    [Warm-start] complete in $(round(time() - seed_start_t, digits=2))s")
    best_paths === nothing && return nothing
    return (best_paths, best_dists, best_unc)
end

function plot_warm_start_solution(graph::LandmarkGraph,
                                  lms::Vector{Landmark},
                                  seed_paths::Vector{Vector{Int}},
                                  seed_dists::Vector{Float64},
                                  seed_unc::Float64;
                                  filename::String="fig2_warm_start.png")
    local_colors = [:purple, :teal, :darkorange, :crimson, :magenta,
                    :brown, :lime, :navy, :coral, :olive]
    _, sensor_mask = node_role_masks(graph)
    plt = plot(legend=:outerright, aspect_ratio=:equal,
               xlabel="x (m)", ylabel="y (m)",
               title="Fig 2: Warm-start [len=$(round(seed_dists[end], digits=2)), unc=$(round(seed_unc, digits=3))]")

    draw_hex_tiles!(plt, graph; fill_color=:aliceblue, line_color=:cadetblue,
                    fill_alpha=0.98, line_alpha=1.0)

    sensor_idx = findall(sensor_mask)
    if !isempty(sensor_idx)
        lx = [graph.landmarks[i].x for i in sensor_idx]
        ly = [graph.landmarks[i].y for i in sensor_idx]
        scatter!(plt, lx, ly, label="Landmarks", color=:black, markersize=4,
                 markerstrokewidth=0)
        for i in sensor_idx
            draw_covariance_ellipse!(plt, graph.landmarks[i].x, graph.landmarks[i].y,
                                     graph.landmarks[i].cov, color=:red, alpha=0.22,
                                     display_scale=400.0)
        end
    end

    for (ai, path) in enumerate(seed_paths)
        isempty(path) && continue
        px = [graph.landmarks[j].x for j in path]
        py = [graph.landmarks[j].y for j in path]
        is_primary = (ai == length(seed_paths))
        clr = is_primary ? :blue : get(local_colors, ai, :gray)
        lbl = is_primary ? "Primary (seed)" : "Support $ai (seed)"
        ls = is_primary ? :solid : (ai == 1 ? :dash : (ai == 2 ? :dashdot : :dot))
        lw = is_primary ? 2.0 : 1.2
        plot!(plt, px, py, label=lbl, color=clr, linewidth=lw, linestyle=ls)
    end

    # Make the warm-start's effect on the primary uncertainty visible on the plot.
    if !isempty(seed_paths)
        seed_xs = Vector{Vector{Float64}}(undef, length(seed_paths))
        seed_ys = Vector{Vector{Float64}}(undef, length(seed_paths))
        for ai in 1:length(seed_paths)
            seed_xs[ai] = [graph.landmarks[j].x for j in seed_paths[ai]]
            seed_ys[ai] = [graph.landmarks[j].y for j in seed_paths[ai]]
        end
        seed_positions = xs_ys_to_positions(seed_xs, seed_ys)
        seed_cov_traj, _ = evaluate_joint_discrete(seed_positions, lms, length(seed_paths))
        prim_covs = seed_cov_traj[end]
        prim_path = seed_paths[end]
        prim_xs = [graph.landmarks[j].x for j in prim_path]
        prim_ys = [graph.landmarks[j].y for j in prim_path]
        for k in 1:min(length(prim_covs), length(prim_xs), length(prim_ys))
            draw_covariance_ellipse!(plt, prim_xs[k], prim_ys[k], prim_covs[k];
                                     nstd=2, color=:blue, alpha=0.10)
        end
    end

    scatter!(plt, [graph.landmarks[1].x], [graph.landmarks[1].y],
             color=:green, markersize=8, markerstrokewidth=0, label="Start")
    scatter!(plt, [graph.landmarks[graph.n].x], [graph.landmarks[graph.n].y],
             color=:orange, marker=:star5, markersize=10, markerstrokewidth=0,
             label="Goal")

    set_hex_world_limits!(plt, graph)
    savefig(plt, filename)
    display(plt)
    println("  Warm-start figure saved: $filename")
end

# ------------------------------------------------------------------
# Constraint-Aware A* Search
# ------------------------------------------------------------------
# Returns (paths, dists, goal_unc) where paths[end] is the primary.
#
# Key Features:
#   - Single joint search on all agents simultaneously
#   - LAZY EVALUATION: only call expensive evaluate_full_paths() when
#     f-value is competitive with best solution found
#   - EPSILON-BOUND ORDERING: primary key is weighted f=g+(1+ε)h using
#     primary shortest-path heuristic; tie-breakers prefer low uncertainty
#     and non-idle supports
#   - BEST-INCUMBENT PRUNING: aggressively skip states where
#     f > best_feasible_dist
#   - SUPPORT CAP: supports cannot exceed primary's arc-distance
#   - EARLY STOPPING: return once feasible solution found
#
# How it works:
#   1. Use optimistic lower-bound distances for f-value computation
#   2. Skip expensive evaluation if f_optimistic >= best_feasible
#   3. Only evaluate full B-spline paths when necessary
#   4. Track best feasible solution globally
#   5. Continue exploring for optimality proof
#
function joint_astar(graph::LandmarkGraph,
                     lms::Vector{Landmark},
                     unc_threshold::Float64,
                     num_agents::Int;
                     seed_paths::Union{Nothing, Vector{Vector{Int}}}=nothing,
                     seed_dists::Union{Nothing, Vector{Float64}}=nothing,
                     debug_animate::Bool=false,
                     debug_animate_start_iter::Int=1,
                     debug_animate_iters::Int=1000,
                     debug_animate_sample_period::Int=1,
                     debug_stop_after_animate::Bool=true,
                     debug_gif_path::String="fig_astar_partial.gif")
    n         = graph.n
    goal      = n                     # last node is goal
    na        = num_agents
    primary   = na                    # index of primary agent (last)

    # Fast path: single-agent weighted A* with incremental covariance updates.
    # Avoids rebuilding/evaluating full trajectories on every expansion.
    if na == 1
        w_astar = 1.0 + PRIMARY_EPSILON
        init_visited = falses(n)
        init_visited[1] = true

        states_sa = State[]
        push!(states_sa, State(1, 0.0, copy(lms[1].cov), -1, init_visited))

        pq_sa = PriorityQueue{Int, Tuple{Float64, Float64}}()
        h0 = graph.shortest_paths[1, goal]
        enqueue!(pq_sa, 1, (w_astar * h0, unc_radius(states_sa[1].cov)))

        best_goal_si = 0
        best_goal_dist = Inf
        iter_count_sa = 0

        # Node-level Pareto frontier with SOUND covariance dominance.
        # Keep labels (distance, covariance) and only prune when an existing
        # label is better in distance and covariance PSD order.
        frontier_at_node = Dict{Int, Vector{Tuple{Float64, Matrix{Float64}}}}()
        frontier_at_node[1] = [(0.0, copy(states_sa[1].cov))]

        while !isempty(pq_sa)
            si = dequeue!(pq_sa)
            S = states_sa[si]
            iter_count_sa += 1

            h = graph.shortest_paths[S.node, goal]
            f = S.dist + w_astar * h
            if isfinite(best_goal_dist) && f >= best_goal_dist
                continue
            end

            if S.node == goal
                gu = unc_radius(S.cov)
                if gu <= unc_threshold && S.dist < best_goal_dist
                    best_goal_dist = S.dist
                    best_goal_si = si
                end
                continue
            end

            for u in graph.neighbors[S.node]
                S.visited[u] && continue

                nd = S.dist + graph.distance[S.node, u]
                ncov = edge_cov_continuous(S.node, u, graph, lms, S.cov)
                nunc = unc_radius(ncov)

                if !haskey(frontier_at_node, u)
                    frontier_at_node[u] = Tuple{Float64, Matrix{Float64}}[]
                end

                # Dominated if an existing label has <= distance and covariance
                # no larger in PSD order.
                dominated = false
                for (od, ocov) in frontier_at_node[u]
                    if od <= nd + 1e-9 && cov_dominates(ocov, ncov)
                        dominated = true
                        break
                    end
                end
                dominated && continue

                # Remove labels dominated by the new label.
                kept = Tuple{Float64, Matrix{Float64}}[]
                for (od, ocov) in frontier_at_node[u]
                    if nd <= od + 1e-9 && cov_dominates(ncov, ocov)
                        continue
                    end
                    push!(kept, (od, ocov))
                end
                push!(kept, (nd, copy(ncov)))
                frontier_at_node[u] = kept

                nvis = copy(S.visited)
                nvis[u] = true
                push!(states_sa, State(u, nd, ncov, si, nvis))
                nsi = length(states_sa)
                nh = graph.shortest_paths[u, goal]
                nf = nd + w_astar * nh
                if isfinite(best_goal_dist) && nf >= best_goal_dist
                    continue
                end
                enqueue!(pq_sa, nsi, (nf, nunc))
            end
        end

        if best_goal_si == 0
            println("  [Constraint A*] No feasible single-agent solution found")
            return [Int[]], [0.0], Inf
        end

        path = Int[]
        si = best_goal_si
        while si != -1
            pushfirst!(path, states_sa[si].node)
            si = states_sa[si].parent
        end

        println("  [Constraint A*] Single-agent complete: $(iter_count_sa) iterations, final_dist=$(round(best_goal_dist, digits=3))")
        return [path], [best_goal_dist], unc_radius(states_sa[best_goal_si].cov)
    end

    # ── Initial state: all agents at node 1 (start) ──────────────────────────
    init_paths   = [fill(1, 1) for _ in 1:na]
    init_visited = [falses(n) for _ in 1:na]
    for a in 1:na; init_visited[a][1] = true; end

    # Evaluate initial state (all agents at node 1)
    init_covs, init_dists = evaluate_full_paths(init_paths, graph, lms, na)

    states = JointState[]
    push!(states, JointState(init_paths, init_covs, init_dists, 0.0, -1, init_visited))

    # Exact state cache keyed by immutable joint path tuples.
    exact_state_best = Dict{Tuple{Vararg{Tuple{Vararg{Int}}}}, Int}()
    exact_state_best[joint_path_key(init_paths)] = 1

    w_astar = 1.0 + PRIMARY_EPSILON
    pq = PriorityQueue{Int, Tuple{Float64, Float64, Float64}}()
    init_h = joint_heuristic(init_paths, goal, graph)
    init_f = init_dists[primary] + w_astar * init_h
    init_unc = unc_radius(init_covs[primary])
    enqueue!(pq, 1, (init_f, init_unc, support_idle_score(init_dists)))

    best_feasible_dist = Inf
    best_feasible_si   = 0
    iter_count         = 0

    # Safe incumbent seeding: only sets an upper bound (never removes optimality).
    if seed_paths !== nothing
        if length(seed_paths) == na && !isempty(seed_paths[primary]) && seed_paths[primary][end] == goal
            seed_covs, seed_eval_dists = evaluate_full_paths(seed_paths, graph, lms, na)
            seed_unc = unc_radius(seed_covs[primary])
            if seed_unc <= unc_threshold
                seed_use_dists = seed_dists === nothing ? seed_eval_dists : seed_dists
                push!(states, JointState([copy(seed_paths[a]) for a in 1:na],
                                          seed_covs,
                                          copy(seed_use_dists),
                                          seed_use_dists[primary],
                                          -1,
                                          [falses(n) for _ in 1:na]))
                best_feasible_si = length(states)
                best_feasible_dist = seed_use_dists[primary]
                exact_state_best[joint_path_key(seed_paths)] = best_feasible_si
                println("  ✓ Seed incumbent: dist=$(round(best_feasible_dist, digits=3)), unc=$(round(seed_unc, digits=4))")
            else
                println("  [Seed] Infeasible seed rejected: unc=$(round(seed_unc, digits=4))")
            end
        else
            println("  [Seed] Invalid seed shape or primary goal not reached; ignored.")
        end
    end

    # Safe dominance frontier keyed by (joint nodes, exact visited signature).
    # This only compares states with identical future action sets.
    frontier_by_signature = Dict{Any, Vector{Tuple{Float64, Matrix{Float64}}}}()
    init_sig = (joint_node_key(init_paths), joint_visited_key(init_visited))
    frontier_by_signature[init_sig] = [(0.0, copy(init_covs[primary]))]
    
    # Diagnostic: check if goal is reachable from start
    goal_h = graph.shortest_paths[1, goal]
    if isinf(goal_h)
        println("  [WARNING] Goal node $goal is unreachable from start node 1!")
    else
        println("  ✓ Goal reachable from node 1 with distance $goal_h")
    end

    # ── Main loop ────────────────────────────────────────────────────────────

    # --- Animation/Debugging additions ---
    animate_enabled = debug_animate
    animate_start_iter = max(1, debug_animate_start_iter)
    animate_limit = max(1, debug_animate_iters)
    animate_sample_period = max(1, debug_animate_sample_period)
    anim_frames = animate_enabled ? Plots.Animation() : nothing
    animation_saved = false

    function debug_joint_astar_plot(state::JointState, best_si::Int, iter_no::Int)
        plt = plot(legend=:outerright, aspect_ratio=:equal,
                   xlabel="x (m)", ylabel="y (m)",
                   title="Constraint A* debug frame $(iter_no)")

        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        scatter!(plt, xs, ys, label="Nodes", color=:gray, markersize=3, markerstrokewidth=0)
        scatter!(plt, [graph.landmarks[1].x], [graph.landmarks[1].y],
                 label="Start", color=:green, markersize=7, markerstrokewidth=0)
        scatter!(plt, [graph.landmarks[goal].x], [graph.landmarks[goal].y],
                 label="Goal", color=:orange, marker=:star5, markersize=9, markerstrokewidth=0)

        for (ai, path) in enumerate(state.paths)
            isempty(path) && continue
            px = [graph.landmarks[j].x for j in path]
            py = [graph.landmarks[j].y for j in path]
            clr = ai == primary ? :blue : :purple
            lbl = ai == primary ? "Primary (expanding)" : "Support $ai (expanding)"
            plot!(plt, px, py, label=lbl, color=clr, linewidth=2.0,
                  linestyle=(ai == primary ? :solid : :dash))
        end

        if best_si > 0
            best = states[best_si]
            for (ai, path) in enumerate(best.paths)
                isempty(path) && continue
                px = [graph.landmarks[j].x for j in path]
                py = [graph.landmarks[j].y for j in path]
                clr = ai == primary ? :darkgreen : :darkgray
                lbl = ai == primary ? "Primary (best)" : "Support $ai (best)"
                plot!(plt, px, py, label=lbl, color=clr, linewidth=1.5, alpha=0.8,
                      linestyle=:dot)
            end
        end

        prim_node = isempty(state.paths[primary]) ? 0 : state.paths[primary][end]
        prim_h = isinf(graph.shortest_paths[prim_node, goal]) ? "∞" : "$(round(graph.shortest_paths[prim_node, goal], digits=1))"
        ann = "Iter $iter_no, prim_node=$prim_node, h=$prim_h, best_dist=$(round(best_feasible_dist, digits=3))"
        annotate!(plt, (graph.landmarks[1].x, graph.landmarks[1].y + 100), text(ann, :black, 10))
        return plt
    end

    while !isempty(pq)
        si  = dequeue!(pq)
        S   = states[si]
        iter_count += 1

        # --- Animation: plot only the requested iteration window, sampled sparsely ---
        if animate_enabled && iter_count >= animate_start_iter && iter_count <= animate_start_iter + animate_limit - 1 &&
           mod(iter_count - animate_start_iter, animate_sample_period) == 0
            plt = debug_joint_astar_plot(S, best_feasible_si, iter_count)
            frame(anim_frames, plt)
        end

        if animate_enabled && !animation_saved && iter_count >= animate_start_iter + animate_limit - 1
            println("  [Constraint A*] Debug cutoff reached at iter $iter_count; saving partial animation to $debug_gif_path.")
            gif(anim_frames, debug_gif_path, fps=20)
            animation_saved = true
            if debug_stop_after_animate
                println("  [Constraint A*] Stopping early after animation cutoff.")
                break
            else
                println("  [Constraint A*] Continuing search after animation cutoff.")
            end
        end

        # Progress update every 500 iterations
        if mod(iter_count, 500) == 0
            prim_node = isempty(S.paths[primary]) ? 0 : S.paths[primary][end]
            prim_h = isinf(graph.shortest_paths[prim_node, goal]) ? "∞" : "$(round(graph.shortest_paths[prim_node, goal], digits=1))"
            println("  [Constraint A*] Iter $iter_count, prim_node=$prim_node, h=$prim_h, feasible=$(isfinite(best_feasible_dist) ? "✓" : "✗"), best_dist=$(round(best_feasible_dist, digits=3)), queue_size=$(length(pq))")
        end

        # Basic pruning: if this state's f-value exceeds best feasible, skip it
        h = joint_heuristic(S.paths, goal, graph)
        f = S.g + w_astar * h
        if isfinite(best_feasible_dist) && f >= best_feasible_dist
            continue
        end

        # Check if primary reached goal
        if S.paths[primary][end] == goal
            prim_unc = unc_radius(S.covs[primary])
            if prim_unc <= unc_threshold && S.g < best_feasible_dist
                best_feasible_dist = S.g
                best_feasible_si = si
                println("  ✓ FEASIBLE SOLUTION at iter $iter_count: dist=$(round(S.g, digits=3)), unc=$(round(prim_unc, digits=4))")
            end
            continue
        end

        # ── Expansion: move all agents synchronously one edge at a time ──────
        candidate_nodes = Vector{Int}(undef, na)
        move_order = Vector{Int}(undef, na)
        move_order[1] = primary
        order_pos = 2
        for a in 1:na
            a == primary && continue
            move_order[order_pos] = a
            order_pos += 1
        end

        function expand_joint_moves(agent_idx::Int, primary_dist::Float64)
            if agent_idx > na
                new_paths = [copy(S.paths[a]) for a in 1:na]
                new_covs = Vector{Matrix{Float64}}(undef, na)
                new_dists = Vector{Float64}(undef, na)
                new_visited = [copy(S.visited[a]) for a in 1:na]

                for a in 1:na
                    u = candidate_nodes[a]
                    prev_node = S.paths[a][end]
                    push!(new_paths[a], u)
                    new_dists[a] = S.dists[a] + graph.distance[prev_node, u]
                    new_covs[a] = edge_cov_continuous(prev_node, u, graph, lms, S.covs[a])
                    new_visited[a][u] = true
                end

                # Support agents must not travel further than the primary.
                for a in 1:(primary - 1)
                    new_dists[a] > new_dists[primary] && return
                end

                new_covs = apply_joint_step_comms(new_covs, candidate_nodes, new_dists, graph)
                new_g = new_dists[primary]
                new_h = joint_heuristic(new_paths, goal, graph)
                f_exact = new_g + w_astar * new_h

                if isfinite(best_feasible_dist) && f_exact >= best_feasible_dist
                    return
                end

                sig_key = (joint_node_key(new_paths), joint_visited_key(new_visited))
                labels = get(frontier_by_signature, sig_key, Tuple{Float64, Matrix{Float64}}[])
                for (od, ocov) in labels
                    if od <= new_g + 1e-9 && cov_dominates(ocov, new_covs[primary])
                        return
                    end
                end
                kept_labels = Tuple{Float64, Matrix{Float64}}[]
                for (od, ocov) in labels
                    if new_g <= od + 1e-9 && cov_dominates(new_covs[primary], ocov)
                        continue
                    end
                    push!(kept_labels, (od, ocov))
                end
                push!(kept_labels, (new_g, copy(new_covs[primary])))
                frontier_by_signature[sig_key] = kept_labels

                new_path_key = joint_path_key(new_paths)
                if haskey(exact_state_best, new_path_key)
                    return
                end

                if new_paths[primary][end] == goal
                    prim_unc = unc_radius(new_covs[primary])
                    if prim_unc <= unc_threshold && new_g < best_feasible_dist
                        push!(states, JointState(copy(new_paths), new_covs, new_dists,
                                                  new_g, si, new_visited))
                        best_feasible_si = length(states)
                        best_feasible_dist = new_g
                        exact_state_best[new_path_key] = best_feasible_si
                        println("  ✓ FEASIBLE at iter $iter_count: dist=$(round(new_g, digits=3)), unc=$(round(prim_unc, digits=4))")
                    end
                    return
                end

                prim_unc_key = unc_radius(new_covs[primary])
                push!(states, JointState(copy(new_paths), new_covs, new_dists,
                                          new_g, si, new_visited))
                new_si = length(states)
                exact_state_best[new_path_key] = new_si
                enqueue!(pq, new_si, (f_exact, prim_unc_key, support_idle_score(new_dists)))
                return
            end

            agent = move_order[agent_idx]
            curr_node = S.paths[agent][end]
            for u in graph.neighbors[curr_node]
                S.visited[agent][u] && continue

                candidate_nodes[agent] = u
                if agent == primary
                    primary_dist_next = S.dists[primary] + graph.distance[curr_node, u]
                    primary_h_next = graph.shortest_paths[u, goal]
                    if isfinite(best_feasible_dist) && primary_dist_next + w_astar * primary_h_next >= best_feasible_dist
                        continue
                    end
                    expand_joint_moves(agent_idx + 1, primary_dist_next)
                else
                    support_dist_next = S.dists[agent] + graph.distance[curr_node, u]
                    support_dist_next <= primary_dist && expand_joint_moves(agent_idx + 1, primary_dist)
                end
            end
        end

        expand_joint_moves(1, S.dists[primary])
    end

    # --- Save animation if enabled ---
    if animate_enabled && iter_count > 1 && !animation_saved
        gif(anim_frames, debug_gif_path, fps=20)
        println("  [Constraint A*] Animation saved to $debug_gif_path (iters $(animate_start_iter)-$(min(iter_count, animate_start_iter + animate_limit - 1)))")
    end

    # ── Extract final solution ────────────────────────────────────────────────
    if best_feasible_si == 0
        println("  [Constraint A*] No feasible solution found")
        return [Int[] for _ in 1:na], zeros(na), Inf
    end

    final = states[best_feasible_si]
    agent_paths = [copy(final.paths[a]) for a in 1:na]
    path_dists = copy(final.dists)

    println("  [Constraint A*] Complete: $(iter_count) iterations, final_dist=$(round(final.g, digits=3))")
    return agent_paths, path_dists, unc_radius(final.covs[primary])
end

# ==========================================================================
# Top-level planner: Comprehensive A* Search + Support Optimization
# ==========================================================================
# Search for shortest feasible path using A*, optimizing supports for each candidate path.
# Returns (paths, dists, uncs, base_gcov, shortest_path, shortest_dist,
#          goal_unc, expansions)

struct PrimaryPathState
    node::Int
    path::Vector{Int}
    dist::Float64
    parent::Int  # index in states vector
    cov::Matrix{Float64}
end

function multi_agent_usp(graph::LandmarkGraph,
                          unc_threshold::Float64,
                          num_ag::Int = 1;
                          binary_search_tol::Float64 = 0.5)  # unused, kept for compat
    n        = graph.n
    base_gcov = [copy(graph.landmarks[i].cov) for i in 1:n]
    lms_base  = graph.landmarks  # Use full landmark list from graph

    # ── Connectivity / distance-only reference ─────────────────────────────
    shortest_path, shortest_dist = search_shortest_path(graph)
    if isempty(shortest_path)
        println("No path exists between start and goal in graph.")
        return Vector{Int}[], Float64[], Float64[], base_gcov, Int[], Inf, Inf, 0
    end
    println("Shortest (distance-only) path: ", shortest_path,
            "  dist=", round(shortest_dist, digits=3))

    # ── Two-Phase Planning: single-agent A* incumbent + support refinement ─
    # Phase 1: Solve a single-agent uncertainty-constrained primary path.
    # Phase 2: Refine supports against that primary, then replan the primary
    #          with support effect modeled until the primary path stabilizes.
    
    println("\n── Lexicographic ε-A* Search: distance-first, information-aware tie-break ──")
    println("Using ε=$(PRIMARY_EPSILON): primary key is f=g+(1+ε)h; secondary key prioritizes higher information states.")
    
    goal = graph.n
    states = PrimaryPathState[]
    
    # Initial state: start at node 1
    init_state = PrimaryPathState(1, [1], 0.0, -1, copy(lms_base[1].cov))
    push!(states, init_state)
    
    # Priority queue: (state_index, f_value) where f = g + (1+ε)×h
    # h = Euclidean distance to goal
    h(node::Int) = hypot(graph.landmarks[node].x - graph.landmarks[goal].x,
                         graph.landmarks[node].y - graph.landmarks[goal].y)
    
    goal_x = graph.landmarks[goal].x
    goal_y = graph.landmarks[goal].y

    function posterior_unc_from_info(prior_cov::Matrix{Float64}, I11::Float64, I12::Float64, I22::Float64)
        det_c = prior_cov[1,1]*prior_cov[2,2] - prior_cov[1,2]*prior_cov[2,1]
        abs(det_c) < 1e-20 && return unc_radius(prior_cov)
        inv_det = 1.0 / det_c
        J11 = I11 + prior_cov[2,2]*inv_det
        J12 = I12 - prior_cov[1,2]*inv_det
        J22 = I22 + prior_cov[1,1]*inv_det
        det_j = J11*J22 - J12*J12
        abs(det_j) < 1e-20 && return unc_radius(prior_cov)
        inv_dj = 1.0 / det_j
        post_cov = [J22*inv_dj  -J12*inv_dj; -J12*inv_dj  J11*inv_dj]
        return unc_radius(post_cov)
    end

    # Precompute static information score at each node for fast lexicographic tie-breaks.
    node_info_score = zeros(Float64, n)
    for node in 1:n
        x = graph.landmarks[node].x
        y = graph.landmarks[node].y
        I11 = 0.0; I12 = 0.0; I22 = 0.0
        for lm in lms_base
            info = landmark_info(x, y, lm)
            info === nothing && continue
            I11 += info[1]; I12 += info[2]; I22 += info[3]
        end
        post_unc = posterior_unc_from_info(graph.landmarks[node].cov, I11, I12, I22)
        node_info_score[node] = 1.0 / (post_unc + 1e-9)
    end

    # Conservative lower bound on achievable goal uncertainty from current belief:
    # assume zero process noise from now and immediate goal-landmark fusion.
    goal_I11 = 0.0; goal_I12 = 0.0; goal_I22 = 0.0
    for lm in lms_base
        info = landmark_info(goal_x, goal_y, lm)
        info === nothing && continue
        goal_I11 += info[1]; goal_I12 += info[2]; goal_I22 += info[3]
    end

    # Lexicographic key: (primary f-cost, -information_score)
    pq = PriorityQueue{Int, Tuple{Float64, Float64}}()
    init_f = init_state.dist + (1.0 + PRIMARY_EPSILON) * h(1)
    init_info = node_info_score[1]
    enqueue!(pq, 1, (init_f, -init_info))
    
    # Track Pareto frontier at each node: (distance, uncertainty) pairs.
    # A new arrival is dominated if existing state has <= distance AND <= uncertainty.
    # This prunes multi-objective dominated states while preserving Pareto optimality.
    frontier_at_node = Dict{Int, Vector{Tuple{Float64, Float64}}}
    frontier_at_node = Dict()
    frontier_at_node[1] = [(0.0, unc_radius(init_state.cov))]
    
    visited_goal = 0  # index of first state reaching goal with feasible uncertainty
    feasible_paths = []  # kept for reporting compatibility
    expansion_count = 0
    prune_count = 0
    
    while !isempty(pq)
        state_idx = dequeue!(pq)
        S = states[state_idx]
        expansion_count += 1
        
        # AGGRESSIVE INTERMEDIATE PRUNING: skip states where primary uncertainty 
        # already exceeds threshold. This eliminates entire branches early.
        if unc_radius(S.cov) > unc_threshold
            prune_count += 1
            continue
        end
        
        # Progress update
        if mod(expansion_count, 1000) == 0
            println("  [A* Primary] Expanded $expansion_count states, queue_size=$(length(pq)), best_feasible=$(length(feasible_paths) > 0 ? "✓ ($(length(feasible_paths)) found)" : "✗")")
        end
        
        # Check if we reached goal
        if S.node == goal
            # Optimize supports to minimize uncertainty for this primary path
            support_paths = Vector{Vector{Int}}(undef, num_ag - 1)
            for sup_idx in 1:(num_ag - 1)
                other_paths = vcat(support_paths[1:sup_idx-1])
                sup_path, _ = single_support_astar(
                    graph, lms_base, S.path, S.dist, other_paths, sup_idx
                )
                support_paths[sup_idx] = sup_path
            end
            
            # Evaluate full joint paths
            full_paths = vcat(support_paths, [S.path])
            full_covs, full_dists = evaluate_full_paths(full_paths, graph, lms_base, num_ag)
            actual_unc = unc_radius(full_covs[end])
            
            if actual_unc <= unc_threshold
                println("    ✓ FEASIBLE #$(length(feasible_paths)+1) at expansion[$expansion_count]: dist=$(round(S.dist, digits=3)), unc=$(round(actual_unc, digits=4))")
                push!(feasible_paths, (S.path, S.dist, full_paths, full_dists, full_covs, actual_unc))
                if length(feasible_paths) == 1
                    visited_goal = state_idx
                end
                # Weighted A*: return on first feasible goal popped from OPEN.
                # This avoids post-hoc ranking among multiple goal states.
                uncs = [unc_radius(full_covs[a]) for a in 1:num_ag]
                println("A* search complete: explored $expansion_count states, pruned $prune_count states, first feasible goal popped.")
                return full_paths, full_dists, uncs, base_gcov,
                       shortest_path, shortest_dist, actual_unc, expansion_count
            else
                if mod(expansion_count, 500) == 0
                    println("    ✗ Goal reached but infeasible: unc=$(round(actual_unc, digits=4)) > $(round(unc_threshold, digits=4))")
                end
            end
            # Continue searching for more feasible paths
            continue
        end
        
        # Expand neighbors
        for next_node in graph.neighbors[S.node]
            # Skip if already in current path (cycle detection)
            if next_node in S.path
                continue
            end
            
            edge_dist = graph.distance[S.node, next_node]
            new_dist = S.dist + edge_dist
            
            new_path = vcat(S.path, next_node)
            new_cov = edge_cov_continuous(S.node, next_node, graph, lms_base, S.cov)
            new_unc = unc_radius(new_cov)
            
            # AGGRESSIVE INTERMEDIATE PRUNING: reject if already violates threshold.
            # This eliminates entire subtrees of infeasible paths early.
            if new_unc > unc_threshold
                prune_count += 1
                continue
            end
            
            # MULTI-OBJECTIVE DOMINANCE PRUNING: skip if dominated by existing frontier at this node.
            # A state is dominated if another state has distance <= new_dist AND uncertainty <= new_unc.
            is_dominated = false
            if haskey(frontier_at_node, next_node)
                for (d, u) in frontier_at_node[next_node]
                    if d <= new_dist + 1e-6 && u <= new_unc + 1e-6
                        is_dominated = true
                        prune_count += 1
                        break
                    end
                end
                if !is_dominated
                    # Remove any frontier states now dominated by this new state
                    new_frontier = filter(p -> !(new_dist <= p[1] + 1e-6 && new_unc <= p[2] + 1e-6), frontier_at_node[next_node])
                    push!(new_frontier, (new_dist, new_unc))
                    frontier_at_node[next_node] = new_frontier
                end
            else
                frontier_at_node[next_node] = [(new_dist, new_unc)]
            end
            
            if is_dominated
                continue
            end
            
            new_state = PrimaryPathState(next_node, new_path, new_dist, state_idx, new_cov)
            push!(states, new_state)
            
            f_primary = new_dist + (1.0 + PRIMARY_EPSILON) * h(next_node)
            info_score = node_info_score[next_node]
            enqueue!(pq, length(states), (f_primary, -info_score))
        end
    end
    
    # Return best feasible solution found
    if length(feasible_paths) > 0
        best_path, best_dist, full_paths, full_dists, full_covs, actual_unc = feasible_paths[1]
        println("A* search complete: explored $expansion_count states, pruned $prune_count states, found $(length(feasible_paths)) feasible path(s).")
        println("  Feasible paths found:")
        for (i, (ppath, pdist, _, _, _, punc)) in enumerate(feasible_paths)
            println("    Path $i: dist=$(round(pdist, digits=3)), unc=$(round(punc, digits=4))")
        end
        uncs = [unc_radius(full_covs[a]) for a in 1:num_ag]
        return full_paths, full_dists, uncs, base_gcov,
               shortest_path, shortest_dist, actual_unc, expansion_count
    else
        println("IMPOSSIBLE: A* exhausted search space (expanded $expansion_count states, pruned $prune_count states) without finding feasible solution.")
        return Vector{Int}[], Float64[], Float64[], base_gcov,
               shortest_path, shortest_dist, Inf, 0
    end
end


# ==========================================================================
# Hex-grid world (pointy-top) with heading-aware transitions
# ==========================================================================
# Motion primitives mirror sim.py semantics:
#   action 0: forward
#   action 1: forward-left  (turn left, then move)
#   action 2: forward-right (turn right, then move)
#
# Each graph node is (hex_cell, heading), so support/primary trajectories are
# constrained by vehicle heading, not just geometric adjacency.

const HEX_PADDING  = 2      # extra cells around bounding box
const HEX_HEADINGS = 6

const HEX_EVEN_ROW_DELTAS = [
    (1, 0), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1)
]

const HEX_ODD_ROW_DELTAS = [
    (1, 0), (1, 1), (0, 1),
    (-1, 0), (0, -1), (1, -1)
]

@inline function hex_center(gx::Int, gy::Int, x0::Float64, y0::Float64, hex_r::Float64)
    hex_w = sqrt(3.0) * hex_r
    x = x0 + gx * hex_w + (isodd(gy) ? hex_w / 2 : 0.0)
    y = y0 + gy * (1.5 * hex_r)
    return x, y
end

@inline function nearest_heading_to_goal(start_xy::Tuple{Float64,Float64}, goal_xy::Tuple{Float64,Float64})
    dx = goal_xy[1] - start_xy[1]
    dy = goal_xy[2] - start_xy[2]
    θ = atan(dy, dx)
    # 6 headings equally spaced at 60°
    best_h = 0
    best_err = Inf
    for h in 0:5
        ϕ = h * (π / 3)
        err = abs(atan(sin(θ - ϕ), cos(θ - ϕ)))
        if err < best_err
            best_err = err
            best_h = h
        end
    end
    return best_h
end

function build_hex_graph(sensor_landmarks::Vector{Landmark},
                         start_pos::Tuple{Float64,Float64},
                         goal_pos::Tuple{Float64,Float64};
                         hex_r::Float64 = HEX_RADIUS_M,
                         padding::Int = HEX_PADDING)
    # Keep the routing graph focused on the start→goal corridor but with ample y-extent.
    xmin = min(start_pos[1], goal_pos[1])
    xmax = max(start_pos[1], goal_pos[1])
    ymin = min(start_pos[2], goal_pos[2])
    ymax = max(start_pos[2], goal_pos[2])

    # Artificially expand y-extent to allow lateral planning and detours.
    y_expansion = 120.0  # ±120m total, allowing landmark-seeking routes without O(n³) blowup
    ymin -= y_expansion
    ymax += y_expansion

    hex_w = sqrt(3.0) * hex_r
    y_step = 1.5 * hex_r

    grid_w = max(3, Int(ceil((xmax - xmin) / hex_w)) + 1 + 2 * padding)
    grid_h = max(3, Int(ceil((ymax - ymin) / y_step)) + 1 + 2 * padding)

    x0 = xmin - padding * hex_w
    y0 = ymin - padding * y_step - 0.5 * y_step  # Shift down by half hex height to center at y=0

    cells = Tuple{Int,Int}[]
    centers = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    for gy in 0:grid_h-1
        # Filter: skip rows with y > 300 and skip lowest y row (gy == 0)
        cy = y0 + gy * y_step
        if cy > 300.0 || gy == 0
            continue
        end
        
        for gx in 1:(grid_w-2)  # Skip first and last columns (left and right edges)
            cell = (gx, gy)
            centers[cell] = hex_center(gx, gy, x0, y0, hex_r)
            push!(cells, cell)
        end
    end

    function nearest_cell(pos::Tuple{Float64,Float64})
        best_cell = cells[1]
        best_d2 = Inf
        for c in cells
            cc = centers[c]
            d2 = (pos[1] - cc[1])^2 + (pos[2] - cc[2])^2
            if d2 < best_d2
                best_d2 = d2
                best_cell = c
            end
        end
        return best_cell
    end

    start_cell = nearest_cell(start_pos)
    goal_cell = nearest_cell(goal_pos)
    start_heading = nearest_heading_to_goal(centers[start_cell], centers[goal_cell])

    # Enumerate heading-expanded routing states; force start state to index 1.
    route_states = Tuple{Tuple{Int,Int}, Int}[]  # ((gx,gy), heading0..5)
    start_state = (start_cell, start_heading)
    push!(route_states, start_state)
    for c in cells
        for h in 0:5
            st = (c, h)
            st == start_state && continue
            push!(route_states, st)
        end
    end

    route_idx = Dict{Tuple{Tuple{Int,Int}, Int}, Int}()
    for (i, st) in enumerate(route_states)
        route_idx[st] = i
    end

    n_route = length(route_states)
    n_sensor = length(sensor_landmarks)
    goal_idx = n_route + n_sensor + 1  # force goal to last node (graph.n)
    n_total = goal_idx

    null_cov = 1e-9 * Matrix{Float64}(I, 2, 2)
    all_lms = Vector{Landmark}(undef, n_total)

    # Routing nodes
    for (i, st) in enumerate(route_states)
        c, _ = st
        cx, cy = centers[c]
        all_lms[i] = Landmark(cx, cy, copy(null_cov))
    end
    # Start covariance should match the original start prior.
    all_lms[1] = Landmark(all_lms[1].x, all_lms[1].y, copy(sensor_landmarks[1].cov))

    # Sensor landmarks are appended as static observation sources (not routing nodes).
    sensor_offset = n_route
    for i in 1:n_sensor
        all_lms[sensor_offset + i] = sensor_landmarks[i]
    end

    # Terminal goal node (routing only)
    gx, gy = centers[goal_cell]
    all_lms[goal_idx] = Landmark(gx, gy, copy(null_cov))

    neighbors = [Int[] for _ in 1:n_total]

    # Heading-constrained transitions:
    # forward (0), forward-left (-1), forward-right (+1)
    turn_options = (0, -1, +1)
    for st in route_states
        c, h = st
        from_idx = route_idx[st]
        cx, cy = c
        deltas = iseven(cy) ? HEX_EVEN_ROW_DELTAS : HEX_ODD_ROW_DELTAS
        for turn in turn_options
            nh = mod(h + turn, 6)
            dx, dy = deltas[nh + 1]
            nc = (cx + dx, cy + dy)
            if haskey(centers, nc)
                to_idx = route_idx[(nc, nh)]
                push!(neighbors[from_idx], to_idx)
            end
        end
    end

    # Any heading at goal cell can transition to terminal goal node.
    for h in 0:5
        gstate = (goal_cell, h)
        if haskey(route_idx, gstate)
            push!(neighbors[route_idx[gstate]], goal_idx)
        end
    end

    # Pairwise geometric matrices used by existing covariance propagation code.
    dist = zeros(n_total, n_total)
    orient = zeros(n_total, n_total)
    for i in 1:n_total
        xi = all_lms[i].x; yi = all_lms[i].y
        for j in 1:n_total
            dx = all_lms[j].x - xi
            dy = all_lms[j].y - yi
            dist[i, j] = hypot(dx, dy)
            orient[i, j] = atan(dy, dx)
        end
    end

    sp = floyd_warshall(dist, neighbors)
    n_edges = sum(length.(neighbors))
    println("Hex graph: $(n_total) nodes ($(n_route) heading-states + $(n_sensor) sensors + goal), ",
            "radius=$(hex_r)m, directed_edges=$(n_edges), grid=$(grid_w)x$(grid_h), ",
            "start_cell=$(start_cell), goal_cell=$(goal_cell), start_heading=$(start_heading)")

    return LandmarkGraph(n_total, all_lms, dist, orient, neighbors, sp)
end

function node_role_masks(graph::LandmarkGraph)
    n = graph.n
    indeg = zeros(Int, n)
    for i in 1:n
        for j in graph.neighbors[i]
            indeg[j] += 1
        end
    end
    route_mask = falses(n)
    sensor_mask = falses(n)
    for i in 1:n
        if !isempty(graph.neighbors[i]) || indeg[i] > 0
            route_mask[i] = true
        else
            sensor_mask[i] = true
        end
    end
    return route_mask, sensor_mask
end

function route_tile_centers(graph::LandmarkGraph)
    route_mask, _ = node_role_masks(graph)
    s = Set{Tuple{Float64,Float64}}()
    for i in 1:graph.n
        route_mask[i] || continue
        push!(s, (graph.landmarks[i].x, graph.landmarks[i].y))
    end
    return collect(s)
end

function set_hex_world_limits!(plt, graph::LandmarkGraph)
    centers = route_tile_centers(graph)
    _, sensor_mask = node_role_masks(graph)
    sensor_idx = findall(sensor_mask)

    xs = Float64[]
    ys = Float64[]

    for c in centers
        push!(xs, c[1]); push!(ys, c[2])
    end
    for i in sensor_idx
        push!(xs, graph.landmarks[i].x)
        push!(ys, graph.landmarks[i].y)
    end

    isempty(xs) && return
    m = max(HEX_RADIUS_M, 30.0)
    xlims!(plt, minimum(xs) - m, maximum(xs) + m)
    ylims!(plt, minimum(ys) - m, maximum(ys) + m)
end

function draw_hex_tiles!(plt, graph::LandmarkGraph;
                         fill_color=:lightgrey,
                         line_color=:white,
                         fill_alpha::Float64=0.98,
                         line_alpha::Float64=1.0)
    centers = route_tile_centers(graph)
    # Match sim.py pointy-top orientation: vertices at 30° + k*60°
    θ = [π/6 + k*(π/3) for k in 0:6]
    for (cx, cy) in centers
        xs = [cx + HEX_RADIUS_M * cos(t) for t in θ]
        ys = [cy + HEX_RADIUS_M * sin(t) for t in θ]
        plot!(plt, xs, ys, seriestype=:shape, color=fill_color,
              alpha=fill_alpha, linecolor=line_color, linealpha=line_alpha,
              linewidth=1.0, label=false)
    end
end

function random_landmark_cov()
    # SPD covariance with randomized anisotropy/correlation.
    sx = 0.95 + 0.55 * rand()
    sy = 0.55 + 0.45 * rand()
    ρ = 0.35 * (2 * rand() - 1)
    cxy = ρ * sx * sy
    return [sx^2 cxy; cxy sy^2]
end

function make_scattered_landmarks(start_pos::Tuple{Float64,Float64},
                                  goal_pos::Tuple{Float64,Float64};
                                  n_scatter::Int = 8)
    sx, _ = start_pos
    gx, _ = goal_pos
    # Evenly dispersed landmarks: stratify in x and place in balanced top/bottom bands.
    # This keeps landmarks away from the start-goal row (y=0) while covering the map.
    x_min = sx + 55.0
    x_max = gx - 55.0
    y_band_min = 70.0
    y_band_max = 220.0

    cols = max(1, Int(ceil(n_scatter / 2)))
    x_bins = range(x_min, x_max, length=cols + 1)

    lms = Landmark[]
    order = collect(1:cols)
    shuffle!(order)

    placed = 0
    for c in order
        x = x_bins[c] + rand() * (x_bins[c + 1] - x_bins[c])

        if placed < n_scatter
            y_top = y_band_min + rand() * (y_band_max - y_band_min)
            push!(lms, Landmark(x, y_top, random_landmark_cov()))
            placed += 1
        end

        if placed < n_scatter
            y_bot = -(y_band_min + rand() * (y_band_max - y_band_min))
            push!(lms, Landmark(x, y_bot, random_landmark_cov()))
            placed += 1
        end

        placed >= n_scatter && break
    end

    return lms
end

# Start and goal are plain routing waypoints — not landmarks, no covariance meaning.
# Start is node 1 (first entry in graph), goal is appended after all landmarks+samples.
const START_POS = (0.0, 0.0)
const GOAL_POS  = (1000.0, 0.0)

# Randomized (seeded) sensor placement near the shortest path to induce detours.
landmarks = make_scattered_landmarks(START_POS, GOAL_POS)

graph = build_hex_graph(landmarks, START_POS, GOAL_POS; hex_r=HEX_RADIUS_M)

# Conditional multiplicative epsilon bound summary.
# Let L* be the unknown optimal continuous feasible primary distance.
# Use L_lb = straight-line(start, goal) as a conservative lower bound on L*.
δ_bound = HEX_RADIUS_M
L_lb = max(1e-9, hypot(GOAL_POS[1] - START_POS[1], GOAL_POS[2] - START_POS[2]))
ε_sample = (2 * δ_bound / max(1e-9, 2 * HEX_RADIUS_M)) + (2 * δ_bound / L_lb)
w_astar = 1.0 + PRIMARY_EPSILON
ε_total_conditional = w_astar * (1.0 + ε_sample) - 1.0

println("Conditional ε-bound: ε_sample=$(round(ε_sample, digits=4)), w=$(round(w_astar, digits=4)), ε_total=$(round(ε_total_conditional, digits=4))")
println("  Bound form (conditional): L_returned ≤ (1+ε_total) L* with assumptions documented in comments.")

n_landmarks  = length(landmarks)   # number of true landmarks (graph nodes 1..n_landmarks)
x_coords     = [lm.x for lm in landmarks]
y_coords     = [lm.y for lm in landmarks]
marker_sizes = [sqrt(max_eigenvalue(lm.cov)) * MARKER_PROPORTION for lm in landmarks]

function draw_covariance_ellipse!(plt, x, y, cov; npts=50, nstd=2, color=:red, alpha=0.3, display_scale=1.0)
    # display_scale is visualization-only (does not change estimation math)
    cov_vis = display_scale .* cov
    vals, vecs = eigen(Symmetric((cov_vis + cov_vis') / 2))
    a = nstd*sqrt(max(vals[1],0.0)); b = nstd*sqrt(max(vals[2],0.0))
    angle = atan(vecs[2,1], vecs[1,1]); θ = range(0, 2π, length=npts)
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    pts = R * vcat((a.*cos.(θ))', (b.*sin.(θ))')
    plot!(plt, x.+pts[1,:], y.+pts[2,:], seriestype=:shape, color=color, alpha=alpha, label=false)
end

# ------------------------------------------------------------------
# Two-stage lexicographic planning helpers
# Stage 1: choose information-gathering landmarks to meet uncertainty target.
# Stage 2: compute shortest primary route constrained to visit selected nodes.
# ------------------------------------------------------------------

@inline function posterior_cov_from_info(prior_cov::Matrix{Float64}, I11::Float64, I12::Float64, I22::Float64)
    det_c = prior_cov[1,1]*prior_cov[2,2] - prior_cov[1,2]*prior_cov[2,1]
    abs(det_c) < 1e-20 && return copy(prior_cov)
    inv_det = 1.0 / det_c
    J11 = I11 + prior_cov[2,2]*inv_det
    J12 = I12 - prior_cov[1,2]*inv_det
    J22 = I22 + prior_cov[1,1]*inv_det
    det_j = J11*J22 - J12*J12
    abs(det_j) < 1e-20 && return copy(prior_cov)
    inv_dj = 1.0 / det_j
    return [J22*inv_dj  -J12*inv_dj; -J12*inv_dj  J11*inv_dj]
end

function select_info_landmarks(goal_x::Float64,
                               goal_y::Float64,
                               lms::Vector{Landmark},
                               unc_threshold::Float64;
                               max_visits::Int = 6,
                               min_gain::Float64 = 1e-4)
    n = length(lms)
    selected = Int[]
    chosen = falses(n)
    prior_cov = copy(lms[1].cov)

    I11 = 0.0; I12 = 0.0; I22 = 0.0
    current_cov = posterior_cov_from_info(prior_cov, I11, I12, I22)
    current_unc = unc_radius(current_cov)

    while current_unc > unc_threshold && length(selected) < max_visits
        best_idx = 0
        best_unc = current_unc
        best_info = (I11, I12, I22)

        for i in 1:n
            chosen[i] && continue
            info = landmark_info(goal_x, goal_y, lms[i])
            info === nothing && continue

            tI11 = I11 + info[1]
            tI12 = I12 + info[2]
            tI22 = I22 + info[3]
            test_cov = posterior_cov_from_info(prior_cov, tI11, tI12, tI22)
            test_unc = unc_radius(test_cov)

            if test_unc < best_unc
                best_unc = test_unc
                best_idx = i
                best_info = (tI11, tI12, tI22)
            end
        end

        if best_idx == 0 || (current_unc - best_unc) < min_gain
            break
        end

        chosen[best_idx] = true
        push!(selected, best_idx)
        I11, I12, I22 = best_info
        current_cov = posterior_cov_from_info(prior_cov, I11, I12, I22)
        current_unc = best_unc
    end

    return selected, current_cov
end

function search_shortest_path_between(graph::LandmarkGraph, start::Int, goal::Int)
    n = graph.n
    dist = fill(Inf, n)
    parent = fill(-1, n)
    dist[start] = 0.0

    gx = graph.landmarks[goal].x
    gy = graph.landmarks[goal].y
    h(v::Int) = hypot(graph.landmarks[v].x - gx, graph.landmarks[v].y - gy)

    pq = PriorityQueue{Int,Float64}()
    enqueue!(pq, start, h(start))
    while !isempty(pq)
        v = dequeue!(pq)
        v == goal && break
        for u in graph.neighbors[v]
            nd = dist[v] + graph.distance[v, u]
            if nd < dist[u]
                dist[u] = nd
                parent[u] = v
                pq[u] = nd + h(u)
            end
        end
    end

    isinf(dist[goal]) && return Int[], Inf
    path = Int[]
    v = goal
    while v != -1
        push!(path, v)
        v = parent[v]
    end
    return reverse!(path), dist[goal]
end

function shortest_path_with_visits(graph::LandmarkGraph, visits::Vector{Int})
    length(visits) < 2 && return Int[], Inf
    full_path = Int[]
    total_dist = 0.0

    for i in 1:(length(visits)-1)
        seg, seg_dist = search_shortest_path_between(graph, visits[i], visits[i+1])
        isempty(seg) && return Int[], Inf
        if !isempty(full_path)
            append!(full_path, seg[2:end])
        else
            append!(full_path, seg)
        end
        total_dist += seg_dist
    end
    return full_path, total_dist
end

function greedy_visit_order(graph::LandmarkGraph, start_node::Int, visit_nodes::Vector{Int})
    remaining = copy(unique(visit_nodes))
    order = Int[]
    curr = start_node

    while !isempty(remaining)
        dists = [graph.shortest_paths[curr, v] for v in remaining]
        j = argmin(dists)
        next_node = remaining[j]
        push!(order, next_node)
        deleteat!(remaining, j)
        curr = next_node
    end
    return order
end


# ----------------------
# Two-stage lexicographic planner: info-gathering then optimal path
# ----------------------
goal_node = graph.n
goal_x = graph.landmarks[goal_node].x
goal_y = graph.landmarks[goal_node].y

# Stage 1: select info-gathering landmarks
selected_lms, info_cov = select_info_landmarks(goal_x, goal_y, graph.landmarks[1:n_landmarks], UNC_RADIUS_THRESHOLD)
println("Selected info-gathering landmarks: ", selected_lms)

# ────────────────────────────────────────────────────────────────────────────
# MULTI-AGENT PLANNING: Sequential with Synchronized Kalman Fusion
# ────────────────────────────────────────────────────────────────────────────
# NOTE: This uses sequential planning (primary then supports) but benefits from:
# 1. All agents using synchronized Kalman filtering at fixed intervals
# 2. Synchronized inter-agent communication with Gaussian tapering
# 3. Global uncertainty computed via joint covariance propagation
# 4. All agents' trajectories optimized together in continuous phase
#
# Future enhancement: Implement full joint A* search for global optimality guarantee
# across all agents simultaneously (state space = joint positions of all agents)

println("\n── MULTI-AGENT PLANNING (Synchronized Kalman) ──")
seed = nothing

if PIPELINE_MODE == :straight_continuous
    println("\n  Mode=:straight_continuous — skipping discrete search and using direct start→goal seed")
    primary_path = [1, goal_node]
    primary_dist = graph.distance[1, goal_node]
    support_paths = [Int[1, 1] for _ in 1:(NUM_AGENTS - 1)]
elseif PIPELINE_MODE == :discrete_then_continuous
    # Stage 1 + 2: Primary A* candidate enumeration + sequential support Dijkstra feasibility
    println("\n  Enumerating primary A* paths until first support-feasible solution...")
    joint_paths, joint_dists, joint_unc, candidate_count = first_feasible_primary_with_sequential_supports(
        graph, graph.landmarks, UNC_RADIUS_THRESHOLD, NUM_AGENTS;
        max_primary_expansions=1200000,
        max_support_expansions=150000,
        max_primary_candidates=5000,
        debug_animate=DEBUG_SEQ_SEARCH_ANIMATE,
        debug_gif_path=DEBUG_SEQ_SEARCH_GIF,
        debug_primary_progress_every=DEBUG_SEQ_PRIMARY_PROGRESS_EVERY,
        debug_support_progress_every=DEBUG_SEQ_SUPPORT_PROGRESS_EVERY,
        debug_anim_primary_sample_every=DEBUG_SEQ_ANIM_PRIMARY_SAMPLE_EVERY,
        debug_anim_support_sample_every=DEBUG_SEQ_ANIM_SUPPORT_SAMPLE_EVERY
    )

    if !isempty(joint_paths)
        println("  ✓ Found feasible solution after $candidate_count primary candidates")
        support_paths = pad_support_paths_to_primary(joint_paths[1:NUM_AGENTS-1], joint_paths[NUM_AGENTS])
        primary_path = joint_paths[NUM_AGENTS]
        primary_dist = joint_dists[NUM_AGENTS]
    else
        println("  ✗ No feasible solution found in configured candidate/expansion budget")
        support_paths = [Int[] for _ in 1:(NUM_AGENTS - 1)]
        primary_path = Int[]
        primary_dist = Inf
    end
else
    error("Unsupported PIPELINE_MODE=$(PIPELINE_MODE). Use :discrete_then_continuous or :straight_continuous")
end

if isempty(primary_path)
    paths = Vector{Vector{Int}}()
    dists = Float64[]
    uncs = Float64[]
    final_global_cov = Matrix{Float64}[]
    solo_unc = Inf
else
    # All agents' paths assembled with support agents first, primary last
    paths = vcat(support_paths, [primary_path])
    
    # Evaluate joint covariance WITH synchronized Kalman fusion
    # This ensures all agents benefit from inter-agent communication at fixed intervals
    final_global_cov, dists = evaluate_full_paths(paths, graph, landmarks, NUM_AGENTS)
    uncs = [unc_radius(final_global_cov[a]) for a in 1:NUM_AGENTS]
    solo_unc = uncs[end]
    
    println("\n✓ Multi-agent solution with synchronized Kalman fusion:")
    for a in 1:(NUM_AGENTS-1)
        println("  Support agent $a: dist=$(round(dists[a], digits=1))m, goal_unc=$(round(uncs[a], digits=4))m")
    end
    println("  Primary agent: dist=$(round(dists[end], digits=1))m, goal_unc=$(round(uncs[end], digits=4))m")
    println("  ├─ Communication: every $(COMM_INTERVAL)m, Gaussian width σ=$(COMM_SIGMA)m")
    println("  └─ All agents' uncertainties benefit from synchronized Kalman fusion at checkpoints")
end

shortest_path = primary_path
shortest_dist = primary_dist
converged_iter = 0

println("\n--- Final Results ---")
if isempty(paths)
    println("IMPOSSIBLE: uncertainty threshold cannot be met. No continuous optimization will be run.")
else
    # Report all agents with their uncertainties and Kalman filtering status
    for (i, path) in enumerate(paths[1:end-1])
        unc_status = ""
        if !isempty(path)
            unc_threshold_ok = uncs[i] <= UNC_RADIUS_THRESHOLD ? "✓" : "✗"
            unc_status = "  goal_unc=$(round(uncs[i], digits=4))  $unc_threshold_ok"
        end
        isempty(path) ? println("Support agent $i : no path found") :
                        println("Support agent $i : ", path, "  dist=", round(dists[i], digits=3), unc_status, "  (Kalman-fused)")
    end
    threshold_status = uncs[end] <= UNC_RADIUS_THRESHOLD ? "✓ met" : "✗ not met"
    println("Primary agent : ", paths[end],
            "  dist=",     round(dists[end], digits=3),
            "  goal_unc=", round(uncs[end],  digits=4),
            "  threshold=", UNC_RADIUS_THRESHOLD, " ", threshold_status, "  (Kalman-fused)")
    
    # Summary of Kalman filtering across all agents
    println("\n=== Kalman Filtering Summary ===")
    println("All $(length(paths)) agents performing synchronized Kalman filtering:")
    println("  • Dead-reckoning propagation: DIR=$(DIR_UNCERTAINTY_PER_METER)/m, PERP=$(PERP_UNCERTAINTY_PER_METER)/m")
    println("  • Landmark fusion: Information filter (Joseph form) at every waypoint")
    println("  • SYNCHRONIZED inter-agent communication: every $(COMM_INTERVAL)m of travel")
    println("  • Tapered Gaussian weighting: σ=$(COMM_SIGMA)m (100m ≈ 2σ transmission width)")
    println("  • Bidirectional Kalman fusion: all agent pairs within range at each checkpoint")
    println("  • Measurement model: bearing-angle sensor with noise=$(SENSOR_NOISE), ratio=$(BEARING_NOISE_RATIO)")
    println("  └─ Support agents: goal_unc=$(round(uncs[1], digits=4))m (improved by inter-agent fusion)")
    println("  └─ Primary agent: goal_unc=$(round(uncs[end], digits=4))m (direct path with landmark updates)")
end
println("Converged at iteration : ", converged_iter)

# ==========================================================================
# Plotting helpers (shared across all three figures)
# ==========================================================================

agent_colors = [:purple, :teal, :darkorange, :crimson, :magenta,
                :brown, :lime, :navy, :coral, :olive]

function make_base_plot(landmarks, graph)
    _, sensor_mask = node_role_masks(graph)
    p = plot(legend=:outerright, aspect_ratio=:equal)
    draw_hex_tiles!(p, graph; fill_color=:aliceblue, line_color=:cadetblue,
                    fill_alpha=0.98, line_alpha=1.0)

    sensor_idx = findall(sensor_mask)
    if !isempty(sensor_idx)
        lx = [graph.landmarks[i].x for i in sensor_idx]
        ly = [graph.landmarks[i].y for i in sensor_idx]
        scatter!(p, lx, ly, label="Landmarks", color=:black, markersize=4,
                 markerstrokewidth=0)
        for i in sensor_idx
            draw_covariance_ellipse!(p, graph.landmarks[i].x, graph.landmarks[i].y,
                                     graph.landmarks[i].cov, color=:red, alpha=0.25,
                                     display_scale=400.0)
        end
    end

    scatter!(p, [graph.landmarks[1].x], [graph.landmarks[1].y],
             color=:green, markersize=8, markerstrokewidth=0, label="Start")
    scatter!(p, [graph.landmarks[graph.n].x], [graph.landmarks[graph.n].y],
             color=:orange, marker=:star5, markersize=10, markerstrokewidth=0,
             label="Goal")
    set_hex_world_limits!(p, graph)
    return p
end

# ==========================================================================
# Figure 0 — graph node connectivity
# ==========================================================================
let
    route_mask, sensor_mask = node_role_masks(graph)
    fig0_title = "Fig 0: Graph"
    if seed !== nothing
        fig0_title = "Fig 0: Graph [warm len=$(round(seed_dists[end], digits=2)), unc=$(round(seed_unc, digits=3))]"
    end
    plt0 = plot(legend=:outerright, aspect_ratio=:equal,
                xlabel="x (m)", ylabel="y (m)",
               title=fig0_title)

    draw_hex_tiles!(plt0, graph; fill_color=:aliceblue, line_color=:cadetblue,
                    fill_alpha=0.98, line_alpha=1.0)

    # Draw edges (deduplicated: only draw i→j where i < j)
    drawn = Set{Tuple{Int,Int}}()
    goal_node = graph.n
    for i in 1:graph.n
        route_mask[i] || continue
        xi = graph.landmarks[i].x; yi = graph.landmarks[i].y
        for j in graph.neighbors[i]
            route_mask[j] || continue
            edge = (min(i,j), max(i,j))
            edge in drawn && continue
            push!(drawn, edge)
            xj = graph.landmarks[j].x; yj = graph.landmarks[j].y
            if i == goal_node || j == goal_node
                clr = :orange; lw = 0.9; alpha = 0.6
            else
                clr = :steelblue; lw = 0.5; alpha = 0.30
            end
            plot!(plt0, [xi, xj], [yi, yj], color=clr, linewidth=lw, alpha=alpha, label=false)
        end
    end

    # Sensor landmarks with covariance ellipses
    sensor_idx = findall(sensor_mask)
    for i in sensor_idx
        draw_covariance_ellipse!(plt0, graph.landmarks[i].x, graph.landmarks[i].y, graph.landmarks[i].cov;
                                  nstd=2, color=:red, alpha=0.18, display_scale=400.0)
    end
    if !isempty(sensor_idx)
        lx = [graph.landmarks[i].x for i in sensor_idx]
        ly = [graph.landmarks[i].y for i in sensor_idx]
        scatter!(plt0, lx, ly, color=:black, markersize=5, markerstrokewidth=0, label="Landmark nodes")
    end
    scatter!(plt0, [graph.landmarks[1].x],     [graph.landmarks[1].y],     color=:green,  markersize=8,
             markerstrokewidth=0, label="Start")
    scatter!(plt0, [graph.landmarks[goal_node].x], [graph.landmarks[goal_node].y], color=:orange, marker=:star5,
             markersize=9, markerstrokewidth=0, label="Goal (routing only)")

    plot!(plt0, [NaN],[NaN], color=:lightsteelblue, linewidth=5, alpha=0.3, label="Hex tiles")
    plot!(plt0, [NaN],[NaN], color=:steelblue,    linewidth=1.2, label="Heading edges")
    plot!(plt0, [NaN],[NaN], color=:orange,       linewidth=0.9, label="goal edge")

    set_hex_world_limits!(plt0, graph)

    savefig(plt0, "fig0_graph.png")
    println("Fig 0 saved  ($(length(drawn)) edges, $(graph.n) nodes)")
end

# ==========================================================================
# Figure 1 - discrete graph solution
# ==========================================================================
let
    plt1 = make_base_plot(landmarks, graph)
    if !isempty(paths)
        # Plot waypoints and paths
        for (vi, path) in enumerate(paths)
            if !isempty(path)
                px_  = [graph.landmarks[j].x for j in path]
                py_  = [graph.landmarks[j].y for j in path]
                clr  = vi==length(paths) ? :blue : get(agent_colors, vi, :gray)
                lbl  = vi==length(paths) ? "Primary" : "Support $vi"
                if vi != length(paths)
                    py_ = py_ .+ (SUPPORT_PLOT_OFFSET_M * vi)
                    lbl *= " (offset)"
                end
                # Vary linestyle by support agent index: dash, dashdot, etc.
                ls   = vi==length(paths) ? :solid : (vi == 1 ? :dash : (vi == 2 ? :dashdot : :dot))
                plot!(plt1, px_, py_, label=lbl, color=clr,
                      linewidth=vi==length(paths) ? 2.0 : 1.2,
                      linestyle=ls)
                scatter!(plt1, px_, py_, label=false, color=clr, markersize=3, markerstrokewidth=0)
            end
        end

        # Evaluate using discrete waypoint evaluation
        agent_positions_fig1 = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(paths))
        for (ai, path) in enumerate(paths)
            if isempty(path)
                agent_positions_fig1[ai] = []
            else
                agent_positions_fig1[ai] = [(graph.landmarks[i].x, graph.landmarks[i].y) for i in path]
            end
        end
        
        covs_all_fig1, arcs_all_fig1 = evaluate_joint_discrete(agent_positions_fig1, landmarks, length(paths); debug_goal_pos=GOAL_POS)

        # Primary is last agent
        prim_covs_fig1 = covs_all_fig1[end]
        prim_arcs_fig1 = arcs_all_fig1[end]
        dijk_prim_len_fig1 = prim_arcs_fig1[end]
        dijk_goal_unc_fig1 = unc_radius(prim_covs_fig1[end])

        println("Discrete evaluation: prim_len=$(round(dijk_prim_len_fig1, digits=3)), unc=$(round(dijk_goal_unc_fig1, digits=4))")

        # Draw primary covariance ellipses
        px_fig1 = [graph.landmarks[j].x for j in paths[end]]
        py_fig1 = [graph.landmarks[j].y for j in paths[end]]
        for k in 1:length(prim_covs_fig1)
            draw_covariance_ellipse!(plt1, px_fig1[k], py_fig1[k], prim_covs_fig1[k];
                                      nstd=2, color=:blue, alpha=0.10)
        end
        title!(plt1,"Fig 1: Discrete [len=$(round(dijk_prim_len_fig1,digits=2)), unc=$(round(dijk_goal_unc_fig1,digits=3))]")
    else
        title!(plt1,"Fig 1: No feasible path")
    end
    xlabel!(plt1,"x (m)"); ylabel!(plt1,"y (m)")
    savefig(plt1,"fig1_joint_discrete_astar.png"); println("Fig 1 saved.")
end

# ==========================================================================
# Figure 2 - agent visibility view (always generated)
# ==========================================================================
let
    plt2v = make_base_plot(landmarks, graph)
    if !isempty(paths)
        for (vi, path) in enumerate(paths)
            isempty(path) && continue
            px_ = [graph.landmarks[j].x for j in path]
            py_ = [graph.landmarks[j].y for j in path]
            clr = vi==length(paths) ? :blue : get(agent_colors, vi, :gray)
            lbl = vi==length(paths) ? "Primary" : "Support $vi"
            if vi != length(paths)
                py_ = py_ .+ (SUPPORT_PLOT_OFFSET_M * vi)
                lbl *= " (offset)"
            end
            ls = vi==length(paths) ? :solid : (vi == 1 ? :dash : (vi == 2 ? :dashdot : :dot))
            plot!(plt2v, px_, py_, label=lbl, color=clr, linewidth=2.0, linestyle=ls)
            scatter!(plt2v, px_, py_, label=false, color=clr, markersize=4, markerstrokewidth=0)
        end
        title!(plt2v, "Fig 2: Agents")
    else
        title!(plt2v, "Fig 2: No feasible path")
    end
    xlabel!(plt2v, "x (m)"); ylabel!(plt2v, "y (m)")
    savefig(plt2v, "fig2_agent_visibility.png"); println("Fig 2 saved.")
end

# ==========================================================================
# Continuous spline optimizer — barrier-based search in control-point space
# ==========================================================================
# Optimize waypoint positions (x, y) in continuous space to minimize primary
# path length while maintaining uncertainty constraint.
#
# Free variables: (x, y) coordinates of all intermediate waypoints across all
# agents. Start and goal positions are fixed.
#
# Objective  : minimize primary agent's total path length (Euclidean distances)
# Constraint : joint unc_radius(Σ_goal) ≤ UNC_RADIUS_THRESHOLD
#              enforced by gradient projection after each step

const CONT_OPT_ITERS  = 1000        # Adam iteration budget (safety limit)
const CONT_OPT_LR     = 5e-1       # Adam learning rate for waypoint positions
const CONT_OPT_H      = 1e-4       # Finite-difference step for gradient
const CONT_ADAM_B1    = 0.9        # Adam exponential decay rates
const CONT_ADAM_B2    = 0.999
const CONT_ADAM_EPS   = 1e-8
const CONT_CONV_TOL   = 1e-3       # Convergence: change in length < this
const CONT_UNC_USE_WAYPOINTS = true # keeps straight-line continuous uncertainty consistent with discrete evaluation

if !isempty(paths) && all(length.(paths) .> 0)  # Continuous optimization only if all agents have paths
    println("\n=== Continuous Spline Optimization ===")

    # Initialize continuous waypoints either from discrete paths or direct-line seed.
    all_agent_wpts = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(paths))
    is_primary_mask = Vector{Bool}(undef, length(paths))

    if PIPELINE_MODE == :straight_continuous
        n_agents = length(paths)
        sx = graph.landmarks[1].x
        sy = graph.landmarks[1].y
        gx = graph.landmarks[graph.n].x
        gy = graph.landmarks[graph.n].y
        n_seed = max(3, STRAIGHT_CONT_PRIMARY_WPTS)

        # Supports remain anchored (two points), primary gets direct-line seed with intermediates.
        for ai in 1:(n_agents - 1)
            all_agent_wpts[ai] = [(sx, sy), (sx, sy)]
            is_primary_mask[ai] = false
        end

        prim_wpts = Vector{Tuple{Float64,Float64}}(undef, n_seed)
        for i in 1:n_seed
            t = (i - 1) / (n_seed - 1)
            prim_wpts[i] = ((1 - t) * sx + t * gx, (1 - t) * sy + t * gy)
        end
        all_agent_wpts[n_agents] = prim_wpts
        is_primary_mask[n_agents] = true
    else
        for (ai, path) in enumerate(paths)
            all_agent_wpts[ai] = [(graph.landmarks[i].x, graph.landmarks[i].y) for i in path]
            is_primary_mask[ai] = (ai == length(paths))
        end
    end
    
    # Count free variables: all intermediate waypoints (keep start/goal fixed)
    num_agents = length(paths)
    free_counts = Int[]
    for ai in 1:num_agents
        n_wpts = length(all_agent_wpts[ai])
        if n_wpts <= 2
            push!(free_counts, 0)  # Start and goal only, no free points
        else
            push!(free_counts, (n_wpts - 2) * 2)  # (n_wpts - 2) intermediate points × 2 coordinates
        end
    end
    total_free_cont = sum(free_counts)
    
    if total_free_cont > 0
        # Function to pack waypoints into flat vector (start/goal fixed, intermediate free)
        function pack_waypoints(wpts_list::Vector{Vector{Tuple{Float64,Float64}}})
            flat = Float64[]
            for ai in 1:num_agents
                wpts = wpts_list[ai]
                if length(wpts) > 2
                    for i in 2:length(wpts)-1  # intermediate waypoints only
                        push!(flat, wpts[i][1])  # x
                        push!(flat, wpts[i][2])  # y
                    end
                end
            end
            return flat
        end
        
        # Function to unpack flat vector back to waypoints
        function unpack_waypoints(flat::Vector{Float64})
            wpts_list = Vector{Vector{Tuple{Float64,Float64}}}(undef, num_agents)
            idx = 1
            for ai in 1:num_agents
                orig_wpts = all_agent_wpts[ai]
                n_wpts = length(orig_wpts)
                wpts = Vector{Tuple{Float64,Float64}}(undef, n_wpts)
                
                # Keep start point fixed
                wpts[1] = orig_wpts[1]
                
                # Unpack intermediate waypoints
                if n_wpts > 2
                    for i in 2:n_wpts-1
                        x = flat[idx]
                        y = flat[idx+1]
                        wpts[i] = (x, y)
                        idx += 2
                    end
                end
                
                # Keep goal point fixed
                wpts[n_wpts] = orig_wpts[n_wpts]
                wpts_list[ai] = wpts
            end
            return wpts_list
        end
        
        # Objective helper: evaluate spline-sampled paths, uncertainty, and curvature.
        function eval_continuous(flat::Vector{Float64})
            ctrl_list = unpack_waypoints(flat)

            sampled_paths = Vector{Vector{Tuple{Float64,Float64}}}(undef, num_agents)
            curvatures = Vector{Vector{Float64}}(undef, num_agents)
            max_curvatures = zeros(Float64, num_agents)
            for ai in 1:num_agents
                sampled_paths[ai], curvatures[ai] = bspline_sample_path(ctrl_list[ai])
                max_curvatures[ai] = isempty(curvatures[ai]) ? 0.0 : maximum(curvatures[ai])
            end

            if any(pt -> !isfinite(pt[1]) || !isfinite(pt[2]), Iterators.flatten(sampled_paths))
                empty_covs = [Matrix{Float64}[] for _ in 1:num_agents]
                return Inf, Inf, sampled_paths, empty_covs, curvatures, max_curvatures, ctrl_list
            end

            unc_eval_paths = CONT_UNC_USE_WAYPOINTS ? ctrl_list : sampled_paths
            covs_all, _ = evaluate_joint_discrete(unc_eval_paths, landmarks, num_agents)
            if isempty(covs_all) || isempty(covs_all[end]) || any(!isfinite, covs_all[end][end])
                return Inf, Inf, sampled_paths, covs_all, curvatures, max_curvatures, ctrl_list
            end

            goal_unc = unc_radius(covs_all[end][end])
            if !isfinite(goal_unc)
                return Inf, Inf, sampled_paths, covs_all, curvatures, max_curvatures, ctrl_list
            end
            prim_len = bspline_path_length(sampled_paths[end])

            return prim_len, goal_unc, sampled_paths, covs_all, curvatures, max_curvatures, ctrl_list
        end
        
        # Initialize
        flat = pack_waypoints(all_agent_wpts)
        init_len, init_unc, init_wpts, init_covs, init_curvs, init_max_curv, init_ctrls = eval_continuous(flat)
        
        init_max_curv_peak = isempty(init_max_curv) ? 0.0 : maximum(init_max_curv)
        init_min_turn = init_max_curv_peak > 1e-12 ? 1.0 / init_max_curv_peak : Inf
        println("  Initial (discrete→spline): prim_len=$(round(init_len, digits=3)), unc=$(round(init_unc, digits=4)), " *
            "min_turn_radius=$(isfinite(init_min_turn) ? round(init_min_turn, digits=3) : Inf)")
        
        # ── PHASE 0: FEASIBILITY SMOOTHING ──
        # Check if initial point meets both constraints. If not, smooth into feasible region first.
        # Feasibility: uncertainty <= threshold AND max_curvature <= MAX_CURVATURE
        # (straight paths with infinite turn radius are valid; Inf curvature at stationary points is fixed by smoothing)
        init_curvature_ok = all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in init_curvs)
        init_feasible = init_unc <= UNC_RADIUS_THRESHOLD && init_curvature_ok
        
        if !init_feasible
            println("  [Smoothing] Initial point is infeasible; smoothing into feasible region...")
            
            # Simple quadratic penalty (not barrier) to push into feasibility
            function smoothing_objective(f::Vector{Float64})
                plen, unc, _, _, curvatures, max_curvs, _ = eval_continuous(f)
                
                # Penalties for constraint violations
                unc_penalty = 0.0
                if unc > UNC_RADIUS_THRESHOLD
                    unc_penalty = 1e3 * (unc - UNC_RADIUS_THRESHOLD)^2
                end
                
                curv_penalty = 0.0
                for curvset in curvatures
                    for κ in curvset
                        if κ > MAX_CURVATURE
                            curv_penalty += 1e3 * (κ - MAX_CURVATURE)^2
                        end
                    end
                end
                
                # Slight path-length penalty to avoid bloating
                return plen + unc_penalty + curv_penalty
            end
            
            smooth_adam_m = zeros(total_free_cont)
            smooth_adam_v = zeros(total_free_cont)
            local smooth_iter = 0
            local smooth_max_iters = 300
            h_smooth = CONT_OPT_H
            lr_smooth = CONT_OPT_LR * 0.5  # Slightly reduced learning rate for stability
            
            local smooth_flat = copy(flat)
            while smooth_iter < smooth_max_iters
                smooth_iter += 1
                
                obj0 = smoothing_objective(smooth_flat)
                
                # Gradient via finite differences
                grad = zeros(total_free_cont)
                for k in 1:total_free_cont
                    smooth_flat[k] += h_smooth
                    obj1 = smoothing_objective(smooth_flat)
                    grad[k] = (obj1 - obj0) / h_smooth
                    smooth_flat[k] -= h_smooth
                end
                
                # Adam update
                b1t = CONT_ADAM_B1^smooth_iter
                b2t = CONT_ADAM_B2^smooth_iter
                step = zeros(total_free_cont)
                for k in 1:total_free_cont
                    g = grad[k]
                    smooth_adam_m[k] = CONT_ADAM_B1 * smooth_adam_m[k] + (1 - CONT_ADAM_B1) * g
                    smooth_adam_v[k] = CONT_ADAM_B2 * smooth_adam_v[k] + (1 - CONT_ADAM_B2) * g^2
                    m̂ = smooth_adam_m[k] / (1 - b1t)
                    v̂ = smooth_adam_v[k] / (1 - b2t)
                    step[k] = lr_smooth * m̂ / (sqrt(v̂) + CONT_ADAM_EPS)
                end
                
                # Line search
                trial = smooth_flat .- step
                trial_obj = smoothing_objective(trial)
                backtracks = 0
                while (!isfinite(trial_obj) || trial_obj > obj0 - CONT_MIN_IMPROVEMENT) && backtracks < 12
                    step .*= CONT_LINESEARCH_SHRINK
                    trial = smooth_flat .- step
                    trial_obj = smoothing_objective(trial)
                    backtracks += 1
                end
                
                if !isfinite(trial_obj) || trial_obj > obj0 - CONT_MIN_IMPROVEMENT
                    break
                end
                
                smooth_flat .= trial
                
                # Check feasibility (ignore Inf curvatures from stationary points)
                slen, sunc, swpts, _, scurvs, smax_curv, _ = eval_continuous(smooth_flat)
                sfeas = sunc <= UNC_RADIUS_THRESHOLD && all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in scurvs)
                
                if mod(smooth_iter, 50) == 0
                    smin_turn = isempty(smax_curv) || maximum(smax_curv) < 1e-12 ? Inf : 1.0 / maximum(smax_curv)
                    sfeas_str = sfeas ? "✓" : "✗"
                    println("    Smooth iter $(lpad(smooth_iter, 3)): len=$(round(slen, digits=2)), unc=$(round(sunc, digits=4)), Rmin=$(round(smin_turn, digits=1)) $sfeas_str")
                end
                
                if sfeas
                    println("    → Feasible at smoothing iter $smooth_iter")
                    break
                end
            end
            
            if smooth_iter >= smooth_max_iters
                println("    [Smoothing] Max iterations reached; using best feasible or closest point")
            end
            
            flat = smooth_flat
            init_len, init_unc, init_wpts, init_covs, init_curvs, init_max_curv, init_ctrls = eval_continuous(flat)
            init_max_curv_peak = isempty(init_max_curv) ? 0.0 : maximum(init_max_curv)
            init_min_turn = init_max_curv_peak > 1e-12 ? 1.0 / init_max_curv_peak : Inf
            println("  After smoothing: prim_len=$(round(init_len, digits=3)), unc=$(round(init_unc, digits=4)), " *
                "min_turn_radius=$(isfinite(init_min_turn) ? round(init_min_turn, digits=3) : Inf)")
        else
            println("  [Smoothing] Initial point is feasible; proceeding to barrier optimization")
        end
        
        # Interior-point style barrier optimizer state
        adam_m = zeros(total_free_cont)
        adam_v = zeros(total_free_cont)

        opt_iter_log = Int[0]
        opt_len_log = Float64[init_len]
        opt_unc_log = Float64[init_unc]
        init_support_lens = [bspline_path_length(init_wpts[a]) for a in 1:(num_agents - 1)]
        opt_support_len_logs = [Float64[init_support_lens[a]] for a in 1:(num_agents - 1)]

        local best_flat = copy(flat)
        local best_len = init_len
        local best_unc = init_unc
        local best_feasible_flat::Union{Nothing,Vector{Float64}} = nothing
        local best_feasible_len = Inf
        local best_feasible_unc = Inf
        local prev_len = init_len

        function spline_barrier_objective(flat::Vector{Float64}, barrier_mu::Float64)
            prim_len, goal_unc, _, _, curvatures, _, _ = eval_continuous(flat)
            unc_slack = UNC_RADIUS_THRESHOLD - goal_unc
            penalty = 0.0
            if unc_slack <= 1e-9
                penalty += 1e4 * (1e-9 - unc_slack)^2
            else
                penalty -= barrier_mu * log(unc_slack)
            end
            for curvset in curvatures
                for κ in curvset
                    slack = MAX_CURVATURE - κ
                    if slack <= 1e-9
                        penalty += 1e4 * (1e-9 - slack)^2
                    else
                        penalty -= barrier_mu * log(slack)
                    end
                end
            end
            return prim_len + penalty
        end

        local stage_iters = max(25, cld(CONT_OPT_ITERS, CONT_BARRIER_STAGES))
        local barrier_mu = CONT_BARRIER_START
        local total_iter = 0
        local turn_txt = ""

        for stage in 1:CONT_BARRIER_STAGES
            println("  Barrier stage $(stage)/$(CONT_BARRIER_STAGES): μ=$(round(barrier_mu, digits=4))")
            for iter in 1:stage_iters
                total_iter += 1
                obj0 = spline_barrier_objective(flat, barrier_mu)

                # Finite-difference gradient of the barrier objective.
                grad = zeros(total_free_cont)
                for k in 1:total_free_cont
                    flat[k] += CONT_OPT_H
                    obj1 = spline_barrier_objective(flat, barrier_mu)
                    grad[k] = (obj1 - obj0) / CONT_OPT_H
                    flat[k] -= CONT_OPT_H
                end

                b1t = CONT_ADAM_B1^total_iter
                b2t = CONT_ADAM_B2^total_iter
                step = zeros(total_free_cont)
                for k in 1:total_free_cont
                    g = grad[k]
                    adam_m[k] = CONT_ADAM_B1 * adam_m[k] + (1 - CONT_ADAM_B1) * g
                    adam_v[k] = CONT_ADAM_B2 * adam_v[k] + (1 - CONT_ADAM_B2) * g^2
                    m̂ = adam_m[k] / (1 - b1t)
                    v̂ = adam_v[k] / (1 - b2t)
                    step[k] = CONT_OPT_LR * m̂ / (sqrt(v̂) + CONT_ADAM_EPS)
                end

                trial = flat .- step
                trial_obj = spline_barrier_objective(trial, barrier_mu)
                backtracks = 0
                while (!isfinite(trial_obj) || trial_obj > obj0 - CONT_MIN_IMPROVEMENT) && backtracks < 12
                    step .*= CONT_LINESEARCH_SHRINK
                    trial = flat .- step
                    trial_obj = spline_barrier_objective(trial, barrier_mu)
                    backtracks += 1
                end

                if !isfinite(trial_obj) || trial_obj > obj0 - CONT_MIN_IMPROVEMENT
                    println("  [Spline barrier] line search stalled; stopping at stage $stage")
                    break
                end

                flat .= trial

                len2, unc2, wpts2, covs2, curv2, max_curv2, _ = eval_continuous(flat)
                feasible = unc2 <= UNC_RADIUS_THRESHOLD && all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in curv2)

                push!(opt_iter_log, total_iter)
                push!(opt_len_log, len2)
                push!(opt_unc_log, unc2)
                for a in 1:(num_agents - 1)
                    push!(opt_support_len_logs[a], bspline_path_length(wpts2[a]))
                end

                if mod(total_iter, 20) == 0
                    status = feasible ? "✓" : "✗"
                    turn_r = isempty(max_curv2) || maximum(max_curv2) < 1e-12 ? Inf : 1.0 / maximum(max_curv2)
                    turn_txt = isfinite(turn_r) ? string(round(turn_r, digits=3)) : "Inf"
                    println("  Iter $(lpad(total_iter,3))  len=$(round(len2,digits=3))  unc=$(round(unc2,digits=4))  turnR=$(turn_txt)  $status")
                end

                if feasible && len2 < best_feasible_len
                    best_feasible_len = len2
                    best_feasible_unc = unc2
                    best_feasible_flat = copy(flat)
                end

                if len2 < best_len
                    best_len = len2
                    best_unc = unc2
                    best_flat = copy(flat)
                end

                if feasible && abs(len2 - prev_len) < CONT_CONV_TOL
                    println("  → Converged at iter $total_iter (Δlen=$(round(abs(len2-prev_len),digits=6)))")
                    break
                end
                prev_len = len2
            end

            barrier_mu *= CONT_BARRIER_DECAY
        end
        
        # Evaluate best solution: prefer best feasible iterate if one exists.
        selected_flat = isnothing(best_feasible_flat) ? best_flat : best_feasible_flat
        opt_len, opt_unc, opt_wpts, opt_covs, opt_curvs, opt_max_curv, opt_ctrls = eval_continuous(selected_flat)
        opt_max_curv_peak = isempty(opt_max_curv) ? 0.0 : maximum(opt_max_curv)
        opt_turn_r = opt_max_curv_peak < 1e-12 ? Inf : 1.0 / opt_max_curv_peak
        curvature_ok = all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in opt_curvs)
        turn_txt = isfinite(opt_turn_r) ? round(opt_turn_r, digits=3) : Inf
        println("  Final: prim_len=$(round(opt_len, digits=3)), unc=$(round(opt_unc, digits=4)), " *
            "min_turn_radius=$(turn_txt), threshold=$(UNC_RADIUS_THRESHOLD)")
        
        if opt_unc <= UNC_RADIUS_THRESHOLD && curvature_ok
            println("  ✓ Optimization successful — uncertainty constraint met")
            
            # Figure 2: optimized continuous paths
            plt2 = make_base_plot(landmarks, graph)
            for ai in 1:num_agents
                wpts = opt_wpts[ai]
                xs = [w[1] for w in wpts]
                ys = [w[2] for w in wpts]
                is_prim = is_primary_mask[ai]
                clr = is_prim ? :blue : get(agent_colors, ai, :gray)
                lbl = is_prim ? "Primary" : "Support $ai"
                if !is_prim
                    ys = ys .+ (SUPPORT_PLOT_OFFSET_M * ai)
                    lbl *= " (offset)"
                end
                lw = is_prim ? 2.2 : 1.3
                # Vary linestyle by support agent index: dash, dashdot, etc.
                ls = is_prim ? :solid : (ai == 1 ? :dash : (ai == 2 ? :dashdot : :dot))
                plot!(plt2, xs, ys, label=lbl, color=clr, linewidth=lw, linestyle=ls)
                scatter!(plt2, xs, ys, label=false, color=clr, markersize=3, markerstrokewidth=0)
            end
            
            # Primary uncertainty ellipses
            prim_covs = opt_covs[end]
            prim_wpts = opt_wpts[end]
            prim_xs = [w[1] for w in prim_wpts]
            prim_ys = [w[2] for w in prim_wpts]
            
            for k in 1:length(prim_covs)
                draw_covariance_ellipse!(plt2, prim_xs[k], prim_ys[k], prim_covs[k];
                                          nstd=2, color=:blue, alpha=0.10)
            end
            
            title!(plt2,"Fig 3: Continuous [len=$(round(opt_len,digits=2)), unc=$(round(opt_unc,digits=3))]")
            xlabel!(plt2,"x (m)"); ylabel!(plt2,"y (m)")
            savefig(plt2,"fig3_continuous_opt.png"); println("Fig 3 saved.")
            
            # Figure 3: convergence plot
            plt3 = plot(opt_iter_log, opt_len_log,
                       label="Path length", color=:blue, linewidth=2,
                       xlabel="Iteration", ylabel="Path length",
                       title="Fig 4: Length vs Iter",
                       legend=:topright)
                 for a in 1:(num_agents - 1)
                  support_clr = get(agent_colors, a, :gray)
                  plot!(plt3, opt_iter_log, opt_support_len_logs[a],
                     label="Support $a length", color=support_clr, linewidth=1.5, linestyle=:dash)
                 end
            hline!(plt3, [init_len], label="Initial ($(round(init_len,digits=2)))",
                   color=:gray, linestyle=:dash, linewidth=1.2)
            hline!(plt3, [opt_len], label="Final ($(round(opt_len,digits=2)))",
                   color=:red, linestyle=:dot, linewidth=1.2)
            savefig(plt3,"fig4_convergence.png"); println("Fig 4 saved.")
        else
            # Fallback: if discrete was feasible but continuous failed, accept discrete solution
            discrete_feasible = init_unc <= UNC_RADIUS_THRESHOLD
            if discrete_feasible
                println("  ⚠ Continuous optimization failed turn-radius constraint, but discrete solution is feasible")
                println("  → Accepting discrete solution: unc=$(round(init_unc,digits=4)) ≤ $(UNC_RADIUS_THRESHOLD)")
            else
                println("  ✗ Optimization did not meet feasibility constraints (unc=$(round(opt_unc,digits=4)), Rmin=$(turn_txt))")
            end
        end
    else
        println("  (No intermediate waypoints to optimize)")
    end
end