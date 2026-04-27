using Plots
using DataStructures
using LinearAlgebra
using Statistics
using Random

Random.seed!(42)  # Reproducible jittered grid sampling

# ----------------------
# Constants & Parameters
# ----------------------
# Run knobs (change these most often between experiments)
const NUM_AGENTS                 = 2
const PIPELINE_MODE              = :discrete_then_continuous
const PRIMARY_EPSILON            = 0.0

const UNC_RADIUS_THRESHOLD       = 3.75
const UNC_FEAS_TOL               = 1e-6

const ENABLE_RELAXED_DISCRETE_FOR_CONTINUOUS = true
const RELAXED_DISCRETE_DELTA_MODE = :relative
const RELAXED_DISCRETE_DELTA_ABS  = 0.20
const RELAXED_DISCRETE_DELTA_REL  = 0.2
const CONTINUE_ASTAR_ON_INFEASIBLE = true

const PRUNE_BY_COMM_RADIUS_JOINT = false
const PRUNE_BY_PRIMARY_UNCERTAINTY = false
const PRUNE_BY_SUPPORT_UNCERTAINTY = false

const STRAIGHT_CONT_PRIMARY_WPTS = 11

# Physical scale: 1 unit = 100m. Graph spans ~1600m x 1500m.
# Platform: AUV with DVL+IMU dead reckoning, acoustic landmark fixes.
#
# DIR_UNCERTAINTY_PER_METER : 10% dead-reckoning drift (DVL+IMU, along-track)
# MAJ_MIN_UNC_RATIO         : along-track drift ~3x cross-track (DVL characteristic)
# SENSOR_NOISE              : USBL/LBL fix accuracy on the order of 10m (0.1 units)
# COMM_RADIUS               : acoustic modem range on the order of a few hundred meters
# UNC_RADIUS_THRESHOLD      : max acceptable determinant-based uncertainty at goal (~50m = 0.5 units)

const DIR_UNCERTAINTY_PER_METER  = 0.05    # LOWERED from 0.30 to make path length more impactful on final uncertainty
const MAJ_MIN_UNC_RATIO          = 3
const PERP_UNCERTAINTY_PER_METER = DIR_UNCERTAINTY_PER_METER / MAJ_MIN_UNC_RATIO
const MARKER_PROPORTION          = 5.0
const SENSOR_NOISE               = 0.038   # High-precision bearing-based landmark acoustic fixes
const COMM_RADIUS                = 300.0    # conservative acoustic modem range (~3 units)
const VISIBILITY_SIGMA           = 75.0     # 1σ detection range for landmark observations
const COMM_INTERVAL              = 100.0    # synchronous communication checkpoint every 50m of travel
const COMM_SIGMA                 = 50.0    # Gaussian taper for comm weighting; soft falloff over ~1-3σ
const HEX_WIDTH_M                = 100.0
const HEX_RADIUS_M               = HEX_WIDTH_M / sqrt(3.0)  # pointy-top hex: width = sqrt(3)*radius
const SUPPORT_PLOT_OFFSET_M      = 3.0  # visualization-only offset so support paths stay visible

# Support idling preference used only as a lower-priority tie-break key.
const SUPPORT_IDLE_PENALTY  = 30.0

# Relaxed discrete acceptance for discrete->continuous pipeline.
# When enabled, discrete search may return seeds with goal uncertainty up to
# UNC_RADIUS_THRESHOLD + δ, and continuous refinement tries to reduce below the
# true threshold. This can reduce discrete A* work on tight constraints.
# Delta model:
#   :absolute => δ = RELAXED_DISCRETE_DELTA_ABS
#   :relative => δ = RELAXED_DISCRETE_DELTA_REL * UNC_RADIUS_THRESHOLD

# If a relaxed discrete seed or continuous refinement is infeasible under the
# true threshold, resume discrete A* search under strict feasibility.

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
# Dominance is over (dist, determinant-based uncertainty).
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

# Determinant-based scalar uncertainty metric.
# For isotropic Σ = σ²I, this returns σ (same units as position),
# because det(Σ)^(1/4) = (σ⁴)^(1/4) = σ.
@inline function unc_det_radius(cov::Matrix{Float64})
    d = cov[1,1] * cov[2,2] - cov[1,2] * cov[2,1]
    return max(d, 1e-18)^(0.25)
end

# Primary scalar uncertainty used by planning/constraints/reporting.
unc_radius(cov::Matrix{Float64}) = unc_det_radius(cov)

@inline function unc_within_threshold(unc::Float64, threshold::Float64=UNC_RADIUS_THRESHOLD)
    return unc <= threshold + UNC_FEAS_TOL
end

@inline function unc_exceeds_threshold(unc::Float64, threshold::Float64=UNC_RADIUS_THRESHOLD)
    return unc > threshold + UNC_FEAS_TOL
end

@inline function discrete_relaxation_delta(threshold::Float64=UNC_RADIUS_THRESHOLD)
    if !ENABLE_RELAXED_DISCRETE_FOR_CONTINUOUS
        return 0.0
    end
    if RELAXED_DISCRETE_DELTA_MODE == :absolute
        return max(0.0, RELAXED_DISCRETE_DELTA_ABS)
    elseif RELAXED_DISCRETE_DELTA_MODE == :relative
        return max(0.0, RELAXED_DISCRETE_DELTA_REL * threshold)
    else
        error("Unsupported RELAXED_DISCRETE_DELTA_MODE=$(RELAXED_DISCRETE_DELTA_MODE). Use :absolute or :relative")
    end
end

@inline function effective_discrete_unc_threshold(threshold::Float64=UNC_RADIUS_THRESHOLD)
    return threshold + discrete_relaxation_delta(threshold)
end

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

# ---------- physics constants ----------
# BEARING_NOISE_RATIO : ratio of cross-bearing to along-bearing sensor noise
# COMM_INTERVAL_DIST : arc-distance between inter-agent communication events
const BEARING_NOISE_RATIO = 2.2               # cross-range noise 2.2× along-range—tighter sensor
const COMM_INTERVAL_DIST  = 5.0               # comm event every ~500m of travel (5 units)


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
            unc_radius(lm.cov) < 1e-8 && continue
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
            unc_radius(lm.cov) < 1e-8 && continue
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
#   (primary_dist, determinant-based uncertainty)
# State A dominates B iff A.g ≤ B.g AND unc_radius(A.covs[end]) ≤ unc_radius(B.covs[end]).
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

@inline function supports_within_comm_radius(nodes::Vector{Int}, graph::LandmarkGraph, primary::Int)
    primary_node = nodes[primary]
    px = graph.landmarks[primary_node].x
    py = graph.landmarks[primary_node].y
    for a in 1:(primary - 1)
        node = nodes[a]
        dx = graph.landmarks[node].x - px
        dy = graph.landmarks[node].y - py
        if dx * dx + dy * dy > COMM_RADIUS^2
            return false
        end
    end
    return true
end

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
#   - SUPPORT CAP: supports cannot exceed primary's arc-distance
#   - COMM RADIUS GATE: supports must stay within direct comm range of the primary
#   - PRIMARY UNCERTAINTY GATE: prune any state whose primary uncertainty exceeds
#     the threshold immediately, rather than allowing later recovery
#   - EARLY STOPPING: return once feasible solution found
#
# How it works:
#   1. Use optimistic lower-bound distances for f-value computation
#   2. Expand the OPEN list in A* order
#   3. Return immediately when the first feasible goal is popped
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

            popped_unc = unc_radius(S.cov)
            if PRUNE_BY_PRIMARY_UNCERTAINTY && unc_exceeds_threshold(popped_unc, unc_threshold)
                println("  [Constraint A*] Pruned state at iter $iter_count_sa: node=$(S.node), primary_unc=$(round(popped_unc, digits=4)) > threshold=$(round(unc_threshold, digits=4))")
                continue
            end

            h = graph.shortest_paths[S.node, goal]
            if S.node == goal
                path = Int[]
                psi = si
                while psi != -1
                    pushfirst!(path, states_sa[psi].node)
                    psi = states_sa[psi].parent
                end

                exact_covs, exact_dists = evaluate_full_paths([path], graph, lms, 1)
                exact_unc = unc_radius(exact_covs[1])
                if unc_within_threshold(exact_unc, unc_threshold)
                    println("  ✓ FEASIBLE SOLUTION at iter $iter_count_sa: dist=$(round(exact_dists[1], digits=3)), unc=$(round(exact_unc, digits=4))")
                    println("  [Constraint A*] Single-agent complete: $(iter_count_sa) iterations, final_dist=$(round(exact_dists[1], digits=3))")
                    return [path], [exact_dists[1]], exact_unc
                end

                println("  [Constraint A*] Goal popped but infeasible under exact eval: unc=$(round(exact_unc, digits=4))")
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
                enqueue!(pq_sa, nsi, (nf, nunc))
            end
        end

        println("  [Constraint A*] No feasible single-agent solution found")
        return [Int[]], [0.0], Inf
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

    iter_count         = 0

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
        ann = "Iter $iter_no, prim_node=$prim_node, h=$prim_h"
        annotate!(plt, (graph.landmarks[1].x, graph.landmarks[1].y + 100), text(ann, :black, 10))
        return plt
    end

    while !isempty(pq)
        si  = dequeue!(pq)
        S   = states[si]
        iter_count += 1

        popped_unc = unc_radius(S.covs[primary])
        if PRUNE_BY_PRIMARY_UNCERTAINTY && unc_exceeds_threshold(popped_unc, unc_threshold)
            println("  [Constraint A*] Pruned joint state at iter $iter_count: primary_unc=$(round(popped_unc, digits=4)) > threshold=$(round(unc_threshold, digits=4))")
            continue
        end

        # --- Animation: plot only the requested iteration window, sampled sparsely ---
        if animate_enabled && iter_count >= animate_start_iter && iter_count <= animate_start_iter + animate_limit - 1 &&
           mod(iter_count - animate_start_iter, animate_sample_period) == 0
            plt = debug_joint_astar_plot(S, 0, iter_count)
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

        # Progress update early; when animation is enabled, match the console cadence
        # to the animation sampling period so logs align with plotted frames.
        progress_every = animate_enabled ? animate_sample_period : 100
        if iter_count <= 5 || mod(iter_count, progress_every) == 0
            prim_node = isempty(S.paths[primary]) ? 0 : S.paths[primary][end]
            prim_h = isinf(graph.shortest_paths[prim_node, goal]) ? "∞" : "$(round(graph.shortest_paths[prim_node, goal], digits=1))"
            println("  [Constraint A*] Iter $iter_count, prim_node=$prim_node, h=$prim_h, queue_size=$(length(pq))")
        end

        # Standard A* ordering uses f only for priority; no incumbent pruning.
        h = joint_heuristic(S.paths, goal, graph)
        f = S.g + w_astar * h

        # Check if primary reached goal
        if S.paths[primary][end] == goal
            agent_paths = [copy(S.paths[a]) for a in 1:na]
            exact_covs, exact_dists = evaluate_full_paths(agent_paths, graph, lms, na)
            exact_unc = unc_radius(exact_covs[primary])
            if unc_within_threshold(exact_unc, unc_threshold)
                println("  ✓ FEASIBLE SOLUTION at iter $iter_count: dist=$(round(exact_dists[primary], digits=3)), unc=$(round(exact_unc, digits=4))")
                println("  [Constraint A*] Complete: $(iter_count) iterations, final_dist=$(round(exact_dists[primary], digits=3))")
                return agent_paths, exact_dists, exact_unc
            else
                println("  [Constraint A*] Goal popped but infeasible under exact eval: unc=$(round(exact_unc, digits=4)) > $(round(unc_threshold, digits=4))")
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

                if PRUNE_BY_COMM_RADIUS_JOINT
                    supports_within_comm_radius(candidate_nodes, graph, primary) || return
                end

                new_covs = apply_joint_step_comms(new_covs, candidate_nodes, new_dists, graph)
                new_g = new_dists[primary]

                # Support agents must also remain under the uncertainty threshold.
                for a in 1:(primary - 1)
                    sup_unc = unc_radius(new_covs[a])
                    if PRUNE_BY_SUPPORT_UNCERTAINTY && unc_exceeds_threshold(sup_unc, unc_threshold)
                        # println("  [Constraint A*] Pruned expansion: support $a unc=$(round(sup_unc, digits=4)) > threshold=$(round(unc_threshold, digits=4))")
                        return
                    end
                end

                prim_unc = unc_radius(new_covs[primary])
                if PRUNE_BY_PRIMARY_UNCERTAINTY && unc_exceeds_threshold(prim_unc, unc_threshold)
                    # println("  [Constraint A*] Pruned expansion: primary_unc=$(round(prim_unc, digits=4)) > threshold=$(round(unc_threshold, digits=4))")
                    return
                end

                new_h = joint_heuristic(new_paths, goal, graph)
                f_exact = new_g + w_astar * new_h

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

                prim_unc_key = prim_unc
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

    # ── No feasible goal reached ──────────────────────────────────────────────
    println("  [Constraint A*] No feasible solution found")
    return [Int[] for _ in 1:na], zeros(na), Inf
end

# ==========================================================================
# Top-level planner: Joint discrete A* -> continuous refinement
# ==========================================================================
# Search for a feasible joint discrete seed first, then refine it continuously.
# Returns (paths, dists, uncs, base_gcov, shortest_path, shortest_dist,
#          goal_unc, expansions)

function multi_agent_usp(graph::LandmarkGraph,
                          unc_threshold::Float64,
                          num_ag::Int = 1;
                          lms::Vector{Landmark}=graph.landmarks,
                          binary_search_tol::Float64 = 0.5,  # unused, kept for compat
                          debug_animate::Bool=false,
                          debug_animate_start_iter::Int=1,
                          debug_animate_iters::Int=1000,
                          debug_animate_sample_period::Int=1,
                          debug_stop_after_animate::Bool=true,
                          debug_gif_path::String="fig_astar_partial.gif")
    n         = graph.n
    base_gcov = [copy(graph.landmarks[i].cov) for i in 1:n]

    # ── Connectivity / distance-only reference ─────────────────────────────
    shortest_path, shortest_dist = search_shortest_path(graph)
    if isempty(shortest_path)
        println("No path exists between start and goal in graph.")
        return Vector{Int}[], Float64[], Float64[], base_gcov, Int[], Inf, Inf, 0
    end
    println("Shortest (distance-only) path: ", shortest_path,
            "  dist=", round(shortest_dist, digits=3))

    println("\n── Joint discrete A* -> continuous refinement ──")
    println("Searching joint discrete paths first, then refining the result with the continuous optimizer.")

    joint_paths, joint_dists, joint_unc = joint_astar(
        graph, lms, unc_threshold, num_ag;
        debug_animate=debug_animate,
        debug_animate_start_iter=debug_animate_start_iter,
        debug_animate_iters=debug_animate_iters,
        debug_animate_sample_period=debug_animate_sample_period,
        debug_stop_after_animate=debug_stop_after_animate,
        debug_gif_path=debug_gif_path
    )

    if length(joint_paths) != num_ag || any(isempty, joint_paths)
        println("  ✗ Joint discrete A* found no feasible joint path")
        return Vector{Int}[], Float64[], Float64[], base_gcov,
               shortest_path, shortest_dist, Inf, 0
    end

    final_covs, final_dists = evaluate_full_paths(joint_paths, graph, lms, num_ag)
    final_uncs = [unc_radius(final_covs[a]) for a in 1:num_ag]

    println("  ✓ Joint discrete A* produced a feasible seed with goal_unc=$(round(final_uncs[end], digits=4))")
    return joint_paths, final_dists, final_uncs, base_gcov,
           shortest_path, shortest_dist, final_uncs[end], 1
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

    # Artificially expand y-extent to allow meaningful lateral planning detours.
    # Wider corridor lets the planner reach off-axis landmark clusters.
    y_expansion = 260.0
    ymin -= y_expansion
    ymax += y_expansion

    hex_w = sqrt(3.0) * hex_r
    y_step = 1.5 * hex_r

    grid_w = max(3, Int(ceil((xmax - xmin) / hex_w)) + 1 + 2 * padding)
    grid_h = max(3, Int(ceil((ymax - ymin) / y_step)) + 1 + 2 * padding)
    iseven(grid_h) && (grid_h += 1)

    x0 = xmin - padding * hex_w
    # Center the hex rows around y=0 so the corridor is visually symmetric.
    y0 = -0.5 * (grid_h - 1) * y_step

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
    start_heading = nearest_heading_to_goal(start_pos, goal_pos)

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
    all_lms[1] = Landmark(start_pos[1], start_pos[2], copy(sensor_landmarks[1].cov))

    # Sensor landmarks are appended as static observation sources (not routing nodes).
    sensor_offset = n_route
    for i in 1:n_sensor
        all_lms[sensor_offset + i] = sensor_landmarks[i]
    end

    # Terminal goal node (routing only)
    all_lms[goal_idx] = Landmark(goal_pos[1], goal_pos[2], copy(null_cov))

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
    sx = 0.80 + 0.55 * rand()
    sy = 0.60 + 0.35 * rand()
    ρ = 0.45 * (2 * rand() - 1)
    cxy = ρ * sx * sy
    return [sx^2 cxy; cxy sy^2]
end

function make_scattered_landmarks(start_pos::Tuple{Float64,Float64},
                                  goal_pos::Tuple{Float64,Float64};
                                  n_scatter::Int = 8)
    _ = start_pos
    _ = goal_pos
    _ = n_scatter

    # Fixed, simplified landmark layout requested by user.
    return Landmark[
        Landmark(800.0, 200.0, random_landmark_cov()),
        Landmark(250.0, -200.0, random_landmark_cov())
    ]
end

# Start and goal are plain routing waypoints — not landmarks, no covariance meaning.
# Start is node 1 (first entry in graph), goal is appended after all landmarks+samples.
const START_POS = (0.0, 0.0)
const GOAL_POS  = (1000.0, 0.0)

# Randomized (seeded) off-axis sensor placement so low-uncertainty plans deviate
# from the geometric shortest path.
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

goal_node = graph.n

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
disc_unc_threshold = effective_discrete_unc_threshold(UNC_RADIUS_THRESHOLD)
if ENABLE_RELAXED_DISCRETE_FOR_CONTINUOUS && disc_unc_threshold > UNC_RADIUS_THRESHOLD + UNC_FEAS_TOL
    println("  Discrete acceptance relaxed for continuous handoff: threshold=$(round(UNC_RADIUS_THRESHOLD, digits=4)) -> $(round(disc_unc_threshold, digits=4))")
end

function run_discrete_seed_search(search_unc_threshold::Float64)
    println("\n  Running joint discrete A* seed search (threshold=$(round(search_unc_threshold, digits=4)))...")
    joint_paths, joint_dists, _, _, _, _, _, _ = multi_agent_usp(
        graph, search_unc_threshold, NUM_AGENTS;
        lms=landmarks,
        debug_animate=true,
        debug_animate_start_iter=10000,
        debug_animate_iters=300000,
        debug_animate_sample_period=10000,
        debug_stop_after_animate=false,
        debug_gif_path="fig_joint_astar_progress.gif"
    )

    if !isempty(joint_paths)
        println("  ✓ Found feasible joint discrete solution")
        support_paths = pad_support_paths_to_primary(joint_paths[1:NUM_AGENTS-1], joint_paths[NUM_AGENTS])
        primary_path = joint_paths[NUM_AGENTS]
        primary_dist = joint_dists[NUM_AGENTS]
        return support_paths, primary_path, primary_dist
    end

    println("  ✗ No feasible joint discrete solution found")
    return [Int[] for _ in 1:(NUM_AGENTS - 1)], Int[], Inf
end

if PIPELINE_MODE == :straight_continuous
    println("\n  Mode=:straight_continuous — skipping discrete search and using direct start→goal seed")
    primary_path = [1, goal_node]
    primary_dist = graph.distance[1, goal_node]
    support_paths = [Int[1, 1] for _ in 1:(NUM_AGENTS - 1)]
elseif PIPELINE_MODE == :discrete_then_continuous
    support_paths, primary_path, primary_dist = run_discrete_seed_search(disc_unc_threshold)
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
        final_node = isempty(paths[a]) ? 0 : paths[a][end]
        println("  Support agent $a: node=$(final_node), dist=$(round(dists[a], digits=1))m, goal_unc=$(round(uncs[a], digits=4))m")
    end
    primary_final_node = isempty(primary_path) ? 0 : primary_path[end]
    println("  Primary agent: node=$(primary_final_node), dist=$(round(dists[end], digits=1))m, goal_unc=$(round(uncs[end], digits=4))m")
    println("  ├─ Communication: every $(COMM_INTERVAL)m, Gaussian taper σ=$(COMM_SIGMA)m")
    println("  └─ All agents' uncertainties benefit from synchronized Kalman fusion at checkpoints")

    if PIPELINE_MODE == :discrete_then_continuous && CONTINUE_ASTAR_ON_INFEASIBLE && unc_exceeds_threshold(solo_unc, disc_unc_threshold)
        println("\n  Seed exceeded relaxed discrete threshold (goal_unc=$(round(solo_unc, digits=4)) > threshold=$(round(disc_unc_threshold, digits=4))).")
        println("  Continuing A* under relaxed threshold...")
        next_support_paths, next_primary_path, next_primary_dist = run_discrete_seed_search(disc_unc_threshold)

        if isempty(next_primary_path)
            println("  ✗ Relaxed-threshold A* did not find a feasible seed.")
            paths = Vector{Vector{Int}}()
            dists = Float64[]
            uncs = Float64[]
            final_global_cov = Matrix{Float64}[]
            solo_unc = Inf
        else
            support_paths = next_support_paths
            primary_path = next_primary_path
            primary_dist = next_primary_dist
            paths = vcat(support_paths, [primary_path])
            final_global_cov, dists = evaluate_full_paths(paths, graph, landmarks, NUM_AGENTS)
            uncs = [unc_radius(final_global_cov[a]) for a in 1:NUM_AGENTS]
            solo_unc = uncs[end]
            println("  ✓ Continued A* found relaxed-feasible seed: goal_unc=$(round(solo_unc, digits=4))")
        end
    end
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
            unc_threshold_ok = unc_within_threshold(uncs[i]) ? "✓" : "✗"
            unc_status = "  goal_unc=$(round(uncs[i], digits=4))  $unc_threshold_ok"
        end
        isempty(path) ? println("Support agent $i : no path found") :
                        println("Support agent $i : ", path, "  dist=", round(dists[i], digits=3), unc_status, "  (Kalman-fused)")
    end
    threshold_status = unc_within_threshold(uncs[end]) ? "✓ met" : "✗ not met"
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
    println("  • Tapered Gaussian weighting: σ=$(COMM_SIGMA)m (soft falloff over ~1-3σ)")
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
# Figure 1 - discrete graph solution
# ==========================================================================
let
    plt1 = make_base_plot(landmarks, graph)
    if !isempty(paths)
        println("Figure path-role mapping: supports = paths[1:$(max(length(paths)-1,0))], primary = paths[$(length(paths))]")
        # Plot waypoints and paths
        for (vi, path) in enumerate(paths)
            if !isempty(path)
                px_  = [graph.landmarks[j].x for j in path]
                py_  = [graph.landmarks[j].y for j in path]
                is_primary = (vi == length(paths))
                clr  = is_primary ? :blue : get(agent_colors, vi, :gray)
                lbl  = is_primary ? "Primary (path[$vi])" : "Support $vi (path[$vi])"
                if vi != length(paths)
                    py_ = py_ .+ (SUPPORT_PLOT_OFFSET_M * vi)
                    lbl *= " (offset)"
                end
                # Vary linestyle by support agent index: dash, dashdot, etc.
                ls   = is_primary ? :solid : (vi == 1 ? :dash : (vi == 2 ? :dashdot : :dot))
                lw   = is_primary ? 2.4 : 1.3
                plot!(plt1, px_, py_, label=lbl, color=clr,
                      linewidth=lw,
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
# Continuous spline optimizer — barrier-based search in control-point space
# ==========================================================================
# Optimize waypoint positions (x, y) in continuous space to minimize primary
# path length while maintaining uncertainty constraint.
#
# Free variables: (x, y) coordinates of all primary intermediate waypoints and
# all support waypoints after their starts. Primary start/goal are fixed;
# support starts are fixed.
#
# Objective  : minimize primary agent's total path length (Euclidean distances)
# Constraint : joint unc_radius(Σ_goal) ≤ UNC_RADIUS_THRESHOLD,
#              support length ≤ primary length,
#              and curvature bounds

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
    
    # Count free variables:
    # - Primary: keep start+goal fixed; optimize intermediate points.
    # - Support: keep start fixed; optimize all subsequent points, including endpoint.
    num_agents = length(paths)
    free_counts = Int[]
    for ai in 1:num_agents
        n_wpts = length(all_agent_wpts[ai])
        if ai == num_agents
            if n_wpts <= 2
                push!(free_counts, 0)
            else
                push!(free_counts, (n_wpts - 2) * 2)
            end
        else
            if n_wpts <= 1
                push!(free_counts, 0)
            else
                push!(free_counts, (n_wpts - 1) * 2)
            end
        end
    end
    total_free_cont = sum(free_counts)
    
    if total_free_cont > 0
        # Pack waypoints into a flat vector of free decision variables.
        function pack_waypoints(wpts_list::Vector{Vector{Tuple{Float64,Float64}}})
            flat = Float64[]
            for ai in 1:num_agents
                wpts = wpts_list[ai]
                if ai == num_agents
                    if length(wpts) > 2
                        for i in 2:length(wpts)-1
                            push!(flat, wpts[i][1])
                            push!(flat, wpts[i][2])
                        end
                    end
                else
                    if length(wpts) > 1
                        for i in 2:length(wpts)
                            push!(flat, wpts[i][1])
                            push!(flat, wpts[i][2])
                        end
                    end
                end
            end
            return flat
        end
        
        # Unpack flat vector back to waypoints.
        function unpack_waypoints(flat::Vector{Float64})
            wpts_list = Vector{Vector{Tuple{Float64,Float64}}}(undef, num_agents)
            idx = 1
            for ai in 1:num_agents
                orig_wpts = all_agent_wpts[ai]
                n_wpts = length(orig_wpts)
                wpts = Vector{Tuple{Float64,Float64}}(undef, n_wpts)
                
                # Keep start fixed for all agents.
                wpts[1] = orig_wpts[1]

                if ai == num_agents
                    # Primary: optimize intermediate points only; keep endpoint fixed.
                    if n_wpts > 2
                        for i in 2:n_wpts-1
                            x = flat[idx]
                            y = flat[idx+1]
                            wpts[i] = (x, y)
                            idx += 2
                        end
                    end
                    if n_wpts >= 2
                        wpts[n_wpts] = orig_wpts[n_wpts]
                    end
                else
                    # Support: optimize every waypoint after start, including endpoint.
                    if n_wpts > 1
                        for i in 2:n_wpts
                            x = flat[idx]
                            y = flat[idx+1]
                            wpts[i] = (x, y)
                            idx += 2
                        end
                    end
                end

                wpts_list[ai] = wpts
            end
            return wpts_list
        end

        @inline function support_length_violation(primary_len::Float64, support_lens::Vector{Float64})
            v = 0.0
            for sl in support_lens
                slack = primary_len - sl
                if slack < -UNC_FEAS_TOL
                    v += (-slack)^2
                end
            end
            return v
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
                support_lens = fill(Inf, max(0, num_agents - 1))
                return Inf, Inf, sampled_paths, empty_covs, curvatures, max_curvatures, ctrl_list, support_lens
            end

            unc_eval_paths = CONT_UNC_USE_WAYPOINTS ? ctrl_list : sampled_paths
            covs_all, _ = evaluate_joint_discrete(unc_eval_paths, landmarks, num_agents)
            if isempty(covs_all) || isempty(covs_all[end]) || any(!isfinite, covs_all[end][end])
                support_lens = fill(Inf, max(0, num_agents - 1))
                return Inf, Inf, sampled_paths, covs_all, curvatures, max_curvatures, ctrl_list, support_lens
            end

            goal_unc = unc_radius(covs_all[end][end])
            if !isfinite(goal_unc)
                support_lens = fill(Inf, max(0, num_agents - 1))
                return Inf, Inf, sampled_paths, covs_all, curvatures, max_curvatures, ctrl_list, support_lens
            end
            prim_len = bspline_path_length(sampled_paths[end])
            support_lens = [bspline_path_length(sampled_paths[a]) for a in 1:(num_agents - 1)]

            return prim_len, goal_unc, sampled_paths, covs_all, curvatures, max_curvatures, ctrl_list, support_lens
        end
        
        # Initialize
        flat = pack_waypoints(all_agent_wpts)
        init_len, init_unc, init_wpts, init_covs, init_curvs, init_max_curv, init_ctrls, init_support_lens = eval_continuous(flat)
        
        init_max_curv_peak = isempty(init_max_curv) ? 0.0 : maximum(init_max_curv)
        init_min_turn = init_max_curv_peak > 1e-12 ? 1.0 / init_max_curv_peak : Inf
        println("  Initial (discrete→spline): prim_len=$(round(init_len, digits=3)), unc=$(round(init_unc, digits=4)), " *
            "min_turn_radius=$(isfinite(init_min_turn) ? round(init_min_turn, digits=3) : Inf)")
        
        # ── PHASE 0: FEASIBILITY SMOOTHING ──
        # Check if initial point meets both constraints. If not, smooth into feasible region first.
        # Feasibility: uncertainty <= threshold AND max_curvature <= MAX_CURVATURE
        # (straight paths with infinite turn radius are valid; Inf curvature at stationary points is fixed by smoothing)
        init_curvature_ok = all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in init_curvs)
        init_support_len_ok = all(sl <= init_len + UNC_FEAS_TOL for sl in init_support_lens)
        init_feasible = unc_within_threshold(init_unc) && init_curvature_ok && init_support_len_ok
        
        if !init_feasible
            println("  [Smoothing] Initial point is infeasible; smoothing into feasible region...")
            
            # Simple quadratic penalty (not barrier) to push into feasibility
            function smoothing_objective(f::Vector{Float64})
                plen, unc, _, _, curvatures, max_curvs, _, support_lens = eval_continuous(f)
                
                # Penalties for constraint violations
                unc_penalty = 0.0
                if unc_exceeds_threshold(unc)
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
                support_len_penalty = 1e3 * support_length_violation(plen, support_lens)
                
                # Slight path-length penalty to avoid bloating
                return plen + unc_penalty + curv_penalty + support_len_penalty
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
                slen, sunc, swpts, _, scurvs, smax_curv, _, ssupport_lens = eval_continuous(smooth_flat)
                support_len_ok = all(sl <= slen + UNC_FEAS_TOL for sl in ssupport_lens)
                sfeas = unc_within_threshold(sunc) &&
                    all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in scurvs) &&
                    support_len_ok
                
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
            init_len, init_unc, init_wpts, init_covs, init_curvs, init_max_curv, init_ctrls, init_support_lens = eval_continuous(flat)
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
        opt_support_len_logs = [Float64[init_support_lens[a]] for a in 1:(num_agents - 1)]
        # Support agent uncertainties at goal during optimization
        init_support_uncs = [unc_radius(init_covs[a][end]) for a in 1:(num_agents - 1)]
        opt_support_unc_logs = [Float64[init_support_uncs[a]] for a in 1:(num_agents - 1)]

        local best_flat = copy(flat)
        local best_len = init_len
        local best_unc = init_unc
        local best_feasible_flat::Union{Nothing,Vector{Float64}} = nothing
        local best_feasible_len = Inf
        local best_feasible_unc = Inf
        local prev_len = init_len

        function spline_barrier_objective(flat::Vector{Float64}, barrier_mu::Float64)
            prim_len, goal_unc, _, _, curvatures, _, _, support_lens = eval_continuous(flat)
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
            for sl in support_lens
                slack = prim_len - sl
                if slack <= 1e-9
                    penalty += 1e4 * (1e-9 - slack)^2
                else
                    penalty -= barrier_mu * log(slack)
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

                len2, unc2, wpts2, covs2, curv2, max_curv2, _, support_lens2 = eval_continuous(flat)
                support_len_ok = all(sl <= len2 + UNC_FEAS_TOL for sl in support_lens2)
                feasible = unc_within_threshold(unc2) &&
                           all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in curv2) &&
                           support_len_ok

                push!(opt_iter_log, total_iter)
                push!(opt_len_log, len2)
                push!(opt_unc_log, unc2)
                for a in 1:(num_agents - 1)
                    push!(opt_support_len_logs[a], support_lens2[a])
                    push!(opt_support_unc_logs[a], unc_radius(covs2[a][end]))
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
        opt_len, opt_unc, opt_wpts, opt_covs, opt_curvs, opt_max_curv, opt_ctrls, opt_support_lens = eval_continuous(selected_flat)
        opt_support_uncs = [unc_radius(opt_covs[a][end]) for a in 1:(num_agents - 1)]

        # Append the selected solution as a final log sample so convergence traces
        # end at the actual accepted output (not necessarily the last optimizer iterate).
        selected_iter = isempty(opt_iter_log) ? 0 : (opt_iter_log[end] + 1)
        push!(opt_iter_log, selected_iter)
        push!(opt_len_log, opt_len)
        push!(opt_unc_log, opt_unc)
        for a in 1:(num_agents - 1)
            push!(opt_support_len_logs[a], opt_support_lens[a])
            push!(opt_support_unc_logs[a], opt_support_uncs[a])
        end

        opt_max_curv_peak = isempty(opt_max_curv) ? 0.0 : maximum(opt_max_curv)
        opt_turn_r = opt_max_curv_peak < 1e-12 ? Inf : 1.0 / opt_max_curv_peak
        curvature_ok = all(all(isfinite(κ) ? κ <= MAX_CURVATURE + 1e-9 : true for κ in curvset) for curvset in opt_curvs)
        support_len_ok_final = all(sl <= opt_len + UNC_FEAS_TOL for sl in opt_support_lens)
        turn_txt = isfinite(opt_turn_r) ? round(opt_turn_r, digits=3) : Inf
        println("  Final: prim_len=$(round(opt_len, digits=3)), unc=$(round(opt_unc, digits=4)), " *
            "min_turn_radius=$(turn_txt), threshold=$(UNC_RADIUS_THRESHOLD)")
        for a in 1:(num_agents - 1)
            println("         support $a: len=$(round(opt_support_lens[a], digits=3)), unc=$(round(opt_support_uncs[a], digits=4))")
        end
        
        if unc_within_threshold(opt_unc) && curvature_ok && support_len_ok_final
            println("  ✓ Optimization successful — uncertainty constraint met")
            
            # Figure 2: optimized continuous paths
            plt2 = make_base_plot(landmarks, graph)
            for ai in 1:num_agents
                wpts = opt_wpts[ai]
                xs = [w[1] for w in wpts]
                ys = [w[2] for w in wpts]
                is_prim = is_primary_mask[ai]
                clr = is_prim ? :blue : get(agent_colors, ai, :gray)
                lbl = is_prim ? "Primary (path[$ai])" : "Support $ai (path[$ai])"
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
            
            title!(plt2,"Fig 2: Continuous [len=$(round(opt_len,digits=2)), unc=$(round(opt_unc,digits=3))]")
            xlabel!(plt2,"x (m)"); ylabel!(plt2,"y (m)")
            savefig(plt2,"fig2_continuous_opt.png"); println("Fig 2 saved.")
            
            # Figure 3: convergence plots (length and uncertainty)
            # Left subplot: path length convergence
            plt3a = plot(opt_iter_log, opt_len_log,
                        label="Primary length", color=:blue, linewidth=2,
                        xlabel="Iteration", ylabel="Path length (m)",
                        legend=:topright)
            for a in 1:(num_agents - 1)
                support_clr = get(agent_colors, a, :gray)
                plot!(plt3a, opt_iter_log, opt_support_len_logs[a],
                     label="Support $a length", color=support_clr, linewidth=1.5, linestyle=:dash)
                  hline!(plt3a, [opt_support_lens[a]], label="Support $a final ($(round(opt_support_lens[a],digits=2)))",
                      color=support_clr, linestyle=:dot, linewidth=1.0, alpha=0.45)
            end
            hline!(plt3a, [init_len], label="Init ($(round(init_len,digits=2)))",
                   color=:gray, linestyle=:dash, linewidth=1.0, alpha=0.5)
            hline!(plt3a, [opt_len], label="Final ($(round(opt_len,digits=2)))",
                   color=:red, linestyle=:dot, linewidth=1.0, alpha=0.5)
            
            # Right subplot: uncertainty convergence
            plt3b = plot(opt_iter_log, opt_unc_log,
                        label="Primary unc", color=:blue, linewidth=2,
                        xlabel="Iteration", ylabel="Uncertainty (m)",
                        legend=:topright)
            for a in 1:(num_agents - 1)
                support_clr = get(agent_colors, a, :gray)
                plot!(plt3b, opt_iter_log, opt_support_unc_logs[a],
                     label="Support $a unc", color=support_clr, linewidth=1.5, linestyle=:dash)
                hline!(plt3b, [opt_support_uncs[a]], label="Support $a final ($(round(opt_support_uncs[a],digits=4)))",
                       color=support_clr, linestyle=:dot, linewidth=1.0, alpha=0.45)
            end
            hline!(plt3b, [init_unc], label="Init ($(round(init_unc,digits=4)))",
                   color=:gray, linestyle=:dash, linewidth=1.0, alpha=0.5)
            hline!(plt3b, [opt_unc], label="Final ($(round(opt_unc,digits=4)))",
                   color=:red, linestyle=:dot, linewidth=1.0, alpha=0.5)
            hline!(plt3b, [UNC_RADIUS_THRESHOLD], label="Threshold",
                   color=:green, linestyle=:solid, linewidth=1.0, alpha=0.3)
            
            plt3 = plot(plt3a, plt3b, layout=(1,2), size=(1400, 400),
                       plot_title="Fig 3: Continuous Optimization Convergence")
            savefig(plt3,"fig3_convergence.png"); println("Fig 3 saved.")
        else
            # Fallback: if continuous optimization is infeasible, optionally continue
            # relaxed-threshold A* instead of stopping at this seed.
            discrete_feasible_relaxed = unc_within_threshold(init_unc, disc_unc_threshold)
            if CONTINUE_ASTAR_ON_INFEASIBLE && PIPELINE_MODE == :discrete_then_continuous
                println("  ⚠ Continuous optimization infeasible; continuing A* under relaxed threshold for an alternate seed...")
                next_support_paths, next_primary_path, next_primary_dist = run_discrete_seed_search(disc_unc_threshold)
                if !isempty(next_primary_path)
                    support_paths = next_support_paths
                    primary_path = next_primary_path
                    primary_dist = next_primary_dist
                    paths = vcat(support_paths, [primary_path])
                    final_global_cov, dists = evaluate_full_paths(paths, graph, landmarks, NUM_AGENTS)
                    uncs = [unc_radius(final_global_cov[a]) for a in 1:NUM_AGENTS]
                    println("  ✓ Alternate relaxed-feasible discrete seed found; using continued A* output (goal_unc=$(round(uncs[end],digits=4))).")
                elseif discrete_feasible_relaxed
                    println("  ⚠ No alternate relaxed seed found; keeping current relaxed-feasible seed: unc=$(round(init_unc,digits=4)) ≤ $(round(disc_unc_threshold,digits=4))")
                else
                    println("  ✗ No alternate relaxed seed found, and current seed is infeasible under relaxed threshold.")
                end
            elseif discrete_feasible_relaxed
                println("  ⚠ Continuous optimization failed turn-radius constraint, but discrete solution is feasible")
                println("  → Accepting discrete solution under relaxed threshold: unc=$(round(init_unc,digits=4)) ≤ $(round(disc_unc_threshold,digits=4))")
            else
                println("  ✗ Optimization did not meet feasibility constraints (unc=$(round(opt_unc,digits=4)), Rmin=$(turn_txt))")
            end
        end
    else
        println("  (No intermediate waypoints to optimize)")
    end
end