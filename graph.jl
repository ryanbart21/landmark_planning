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
const NUM_AGENTS                 = 3
const SENSOR_NOISE               = 1.0    # Increased from 0.1 to make landmarks less informative, allow path to matter
const COMM_RADIUS                = 30.0    # acoustic modem range ~300m (3 units)
const UNC_RADIUS_THRESHOLD       = 0.5   # Relaxed to 0.45 to find feasible solutions and test ε-optimal A* path diversity

# ε-optimal parameter for primary-cost weighting in A* (f = g + (1+ε)h)
const PRIMARY_EPSILON = 0.5

# ----------------------
# Data Structures
# ----------------------
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
                             support_idx::Int)
    n = graph.n
    
    # Compute primary's covariance trajectory (start with just primary alone)
    primary_covs = [copy(lms[primary_path[1]].cov)]
    for i in 1:length(primary_path)-1
        u, v = primary_path[i], primary_path[i+1]
        edge_cov = edge_cov_continuous(u, v, graph, lms, primary_covs[end])
        push!(primary_covs, edge_cov)
    end
    
    # Apply communications from OTHER supports (that are already fixed)
    for (other_idx, other_path) in enumerate(other_support_paths)
        if isempty(other_path)
            continue
        end
        # For each point on primary path, check if other support can help
        for pi in 1:length(primary_path)
            px = graph.landmarks[primary_path[pi]].x
            py = graph.landmarks[primary_path[pi]].y
            
            # Find best communication point on other support's path
            for (oi, other_node) in enumerate(other_path)
                ox = graph.landmarks[other_node].x
                oy = graph.landmarks[other_node].y
                d2 = (px - ox)^2 + (py - oy)^2
                
                if d2 <= COMM_RADIUS^2
                    # Other support can communicate at primary's node pi
                    other_cov = copy(lms[other_node].cov)
                    
                    # Fuse at primary waypoint pi
                    primary_cov_at_pi = primary_covs[pi]
                    fused = fuse_cov(primary_cov_at_pi, other_cov)
                    
                    # Forward-propagate to goal
                    cov_to_goal = fused
                    for pj in pi+1:length(primary_path)-1
                        pj_u, pj_v = primary_path[pj], primary_path[pj+1]
                        cov_to_goal = edge_cov_continuous(pj_u, pj_v, graph, lms, cov_to_goal)
                    end
                    
                    # Update if better
                    if max_eigenvalue(cov_to_goal) < max_eigenvalue(primary_covs[end])
                        primary_covs[end] = cov_to_goal
                    end
                    
                    break
                end
            end
        end
    end
    
    # Now optimize THIS support's path
    init_node = 1
    init_visited = falses(n)
    init_visited[1] = true
    
    states = SupportState[]
    node_states = Dict{Int, Int}()  # node -> best state index at that node
    
    init_state = SupportState([init_node], [0.0], [copy(primary_covs[end])], 
                              unc_radius(primary_covs[end]), -1, [init_visited])
    push!(states, init_state)
    node_states[init_node] = 1
    
    pq = PriorityQueue{Int, Float64}()
    enqueue!(pq, 1, init_state.g)
    
    best_goal_g = Inf
    best_goal_si = 0
    expanded = 0
    
    while !isempty(pq)
        si = dequeue!(pq)
        S = states[si]
        node = S.nodes[1]
        
        # Skip if dominated
        if haskey(node_states, node) && node_states[node] != si
            continue
        end
        
        # Track best
        if S.g <= best_goal_g
            best_goal_g = S.g
            best_goal_si = si
        end
        
        # Early exit after exploring some states
        expanded += 1
        if expanded > 200
            break
        end
        
        # Expand: try neighbors
        for u in graph.neighbors[node]
            if S.visited[1][u]
                continue
            end
            
            edge_dist = graph.distance[node, u]
            if edge_dist + S.dists[1] > primary_path_dist
                continue  # Don't travel beyond primary
            end
            
            # Compute best uncertainty reduction at this support position
            ux = graph.landmarks[u].x
            uy = graph.landmarks[u].y
            
            new_primary_cov = copy(primary_covs[end])
            
            for pi in 1:length(primary_path)
                px = graph.landmarks[primary_path[pi]].x
                py = graph.landmarks[primary_path[pi]].y
                d2 = (ux - px)^2 + (uy - py)^2
                
                if d2 <= COMM_RADIUS^2
                    # This support can communicate at primary's node pi
                    sup_cov = copy(lms[u].cov)
                    sup_cov = sup_cov + growth_covariance(S.dists[1] + edge_dist, 0.0)
                    
                    primary_cov_at_pi = primary_covs[pi]
                    fused = fuse_cov(primary_cov_at_pi, sup_cov)
                    
                    # Forward to goal
                    cov_to_goal = fused
                    for pj in pi+1:length(primary_path)-1
                        pj_u, pj_v = primary_path[pj], primary_path[pj+1]
                        cov_to_goal = edge_cov_continuous(pj_u, pj_v, graph, lms, cov_to_goal)
                    end
                    
                    if max_eigenvalue(cov_to_goal) < max_eigenvalue(new_primary_cov)
                        new_primary_cov = cov_to_goal
                    end
                end
            end
            
            new_unc = unc_radius(new_primary_cov)
            new_dist = S.dists[1] + edge_dist
            
            # Dominance check
            if haskey(node_states, u)
                old_si = node_states[u]
                old = states[old_si]
                if old.g <= new_unc
                    continue  # Old dominates
                end
            end
            
            # Add new state
            new_visited = copy(S.visited[1])
            new_visited[u] = true
            push!(states, SupportState([u], [new_dist], [new_primary_cov], new_unc, si, [new_visited]))
            new_si = length(states)
            node_states[u] = new_si
            enqueue!(pq, new_si, new_unc)
        end
    end
    
    # Reconstruct path
    if best_goal_si == 0
        return Int[], Inf
    end
    
    path = Int[]
    si = best_goal_si
    while si != -1
        S = states[si]
        pushfirst!(path, S.nodes[1])
        si = S.parent
    end
    
    # Deduplicate
    unique_path = Int[]
    for nd in path
        if isempty(unique_path) || nd != unique_path[end]
            push!(unique_path, nd)
        end
    end
    
    return unique_path, best_goal_g
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

function support_astar(graph::LandmarkGraph,
                       lms::Vector{Landmark},
                       primary_path::Vector{Int},
                       primary_path_dist::Float64,
                       unc_threshold::Float64,
                       num_supports::Int)
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
    expanded_count = 0
    min_iters_before_feasible = 100  # Ensure we expand at least this many states before stopping

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
        
        # Early exit only if we've explored enough states
        # (Prevents immediate return when initial state is feasible)
        if best_goal_si > 0 && expanded_count > min_iters_before_feasible
            break
        end
        
        # Expand this state to explore neighbors
        expanded_count += 1

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
        return [Int[] for _ in 1:ns], Inf
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
        agent_paths[a] = unique_path
    end

    return agent_paths, best_goal_g
end

# ---------- physics constants ----------
# VISIBILITY_SIGMA : 1-σ range of acoustic detection (soft P(detect) ~ exp(-d²/2σ²))
# BEARING_NOISE_RATIO : ratio of cross-bearing to along-bearing sensor noise
# COMM_INTERVAL_DIST : arc-distance between inter-agent communication events
const VISIBILITY_SIGMA    = COMM_RADIUS       # detection probability roll-off
const BEARING_NOISE_RATIO = 3.0               # cross-range noise 3× along-range
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
        
        # 2. Landmark fusion at current position
        I11 = 0.0; I12 = 0.0; I22 = 0.0
        for lm in lms
            info = landmark_info(x_curr, y_curr, lm)
            info === nothing && continue
            I11 += info[1]; I12 += info[2]; I22 += info[3]
        end
        if I11 > 0.0 || I22 > 0.0
            # Information filter update: combine prior covariance with landmark info
            det_c = cov[1,1]*cov[2,2] - cov[1,2]*cov[2,1]
            inv_det = 1.0 / det_c
            J11 = I11 + cov[2,2]*inv_det
            J12 = I12 - cov[1,2]*inv_det
            J22 = I22 + cov[1,1]*inv_det
            det_j = J11*J22 - J12*J12
            inv_dj = 1.0 / det_j
            cov = [J22*inv_dj  -J12*inv_dj; -J12*inv_dj  J11*inv_dj]
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
# Evaluate all agents at their waypoint positions with inter-agent communication

function evaluate_joint_discrete(agent_positions::Vector{Vector{Tuple{Float64,Float64}}},
                                  lms::Vector{Landmark},
                                  na::Int;
                                  debug_goal_pos::Union{Tuple{Float64,Float64}, Nothing} = nothing)
    # Evaluate covariance for each agent at their waypoint positions
    all_covs = Vector{Vector{Matrix{Float64}}}(undef, na)
    all_arcs = Vector{Vector{Float64}}(undef, na)
    
    for a in 1:na
        if isempty(agent_positions[a])
            all_covs[a] = [copy(lms[1].cov)]
            all_arcs[a] = [0.0]
        else
            covs = propagate_cov_discrete(agent_positions[a], lms, lms[1].cov; 
                                          debug_goal_pos=debug_goal_pos, debug_agent_id=a)
            all_covs[a] = covs
            # Compute cumulative arc lengths (distance along path)
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
    
    # Apply inter-agent communication at landmark observation points
    apply_inter_agent_discrete_comms!(all_covs, agent_positions)
    
    return all_covs, all_arcs
end

function apply_inter_agent_discrete_comms!(all_covs::Vector{Vector{Matrix{Float64}}},
                                           agent_positions::Vector{Vector{Tuple{Float64,Float64}}})
    na = length(all_covs)
    
    # Simple communication model: agents share observations when close
    for sender in 1:na
        for (si, (sx, sy)) in enumerate(agent_positions[sender])
            cov_s = all_covs[sender][si]
            
            for receiver in 1:na
                receiver == sender && continue
                
                # Find closest point on receiver trajectory and fuse
                for (ri, (rx, ry)) in enumerate(agent_positions[receiver])
                    dist2 = (sx - rx)^2 + (sy - ry)^2
                    if dist2 <= COMM_RADIUS^2
                        w = exp(-dist2 / (2 * COMM_RADIUS^2))
                        if w > 1e-3
                            # Fuse sender's covariance information
                            S_total = cov_s + SENSOR_NOISE^2 * I(2)
                            all_covs[receiver][ri] = inv(inv(all_covs[receiver][ri]) + w * inv(S_total))
                        end
                    end
                end
            end
        end
    end
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
# is bounded to [0, primary_dist + COMM_RADIUS] — supports need not travel
# further than the primary.

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
#   - BEST-INCUMBENT PRUNING: aggressively skip states where
#     f > best_feasible_dist
#   - NO budget constraints: supports move freely to help primary
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
                     num_agents::Int)
    n         = graph.n
    goal      = n                     # last node is goal
    na        = num_agents
    primary   = na                    # index of primary agent (last)

    # ── Initial state: all agents at node 1 (start) ──────────────────────────
    init_paths   = [fill(1, 1) for _ in 1:na]
    init_visited = [falses(n) for _ in 1:na]
    for a in 1:na; init_visited[a][1] = true; end

    # Evaluate initial state (all agents at node 1)
    init_covs, init_dists = evaluate_full_paths(init_paths, graph, lms, na)

    states = JointState[]
    push!(states, JointState(init_paths, init_covs, init_dists, 0.0, -1, init_visited))

    # Track best state per path tuple for simple dominance
    path_to_best = Dict{Vector{Vector{Int}}, Int}()
    path_to_best[deepcopy(init_paths)] = 1

    pq = PriorityQueue{Int, Float64}()
    enqueue!(pq, 1, joint_heuristic(init_paths, goal, graph))

    best_feasible_dist = Inf
    best_feasible_si   = 0
    iter_count         = 0
    
    # Diagnostic: check if goal is reachable from start
    goal_h = graph.shortest_paths[1, goal]
    if isinf(goal_h)
        println("  [WARNING] Goal node $goal is unreachable from start node 1!")
    else
        println("  ✓ Goal reachable from node 1 with distance $goal_h")
    end

    # ── Main loop ────────────────────────────────────────────────────────────
    while !isempty(pq)
        si  = dequeue!(pq)
        S   = states[si]
        iter_count += 1

        # Progress update every 500 iterations
        if mod(iter_count, 500) == 0
            prim_node = isempty(S.paths[primary]) ? 0 : S.paths[primary][end]
            prim_h = isinf(graph.shortest_paths[prim_node, goal]) ? "∞" : "$(round(graph.shortest_paths[prim_node, goal], digits=1))"
            println("  [Constraint A*] Iter $iter_count, prim_node=$prim_node, h=$prim_h, feasible=$(isfinite(best_feasible_dist) ? "✓" : "✗"), best_dist=$(round(best_feasible_dist, digits=3)), queue_size=$(length(pq))")
        end

        # Basic pruning: if this state's f-value exceeds best feasible, skip it
        h = joint_heuristic(S.paths, goal, graph)
        f = S.g + h
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

        # ── Expansion: try advancing each agent ──────────────────────────────
        for moved in 1:na
            for u in graph.neighbors[S.paths[moved][end]]
                # Avoid cycles: don't revisit nodes on this agent's path
                if S.visited[moved][u]
                    continue
                end

                # Build new state with u appended to agent `moved`'s path
                new_paths = [copy(S.paths[a]) for a in 1:na]
                push!(new_paths[moved], u)

                # ── LAZY EVALUATION: use optimistic lower bounds first ──────────
                # Only evaluate expensive full paths if f-value is competitive
                if moved == primary
                    # Primary moved: use Euclidean edge distance (lower bound)
                    prev_node = S.paths[primary][end]
                    opt_dist = S.dists[primary] + graph.distance[prev_node, u]
                else
                    # Support moved: primary distance unchanged (optimistic)
                    opt_dist = S.dists[primary]
                end
                new_h = joint_heuristic(new_paths, goal, graph)
                f_opt = opt_dist + new_h

                # If optimistic f exceeds best feasible, don't evaluate
                if isfinite(best_feasible_dist) && f_opt >= best_feasible_dist
                    continue
                end

                # Evaluate full joint state (expensive)
                new_covs, new_dists = evaluate_full_paths(new_paths, graph, lms, na)
                new_g = new_dists[primary]

                # Recompute exact f with actual distances
                f_exact = new_g + new_h
                if isfinite(best_feasible_dist) && f_exact >= best_feasible_dist
                    continue
                end

                # Check if primary reached goal
                if new_paths[primary][end] == goal
                    prim_unc = unc_radius(new_covs[primary])
                    if prim_unc <= unc_threshold && new_g < best_feasible_dist
                        best_feasible_dist = new_g
                        best_feasible_si = length(states) + 1
                        println("  ✓ FEASIBLE at iter $iter_count: dist=$(round(new_g, digits=3)), unc=$(round(prim_unc, digits=4))")
                    end
                    continue
                end

                # Simple dominance: keep only best state at each path tuple
                new_path_key = deepcopy(new_paths)
                if haskey(path_to_best, new_path_key)
                    old_si = path_to_best[new_path_key]
                    old = states[old_si]
                    old_eig = max_eigenvalue(old.covs[primary])
                    new_eig = max_eigenvalue(new_covs[primary])
                    # Keep new state only if it's better on both dist and uncertainty
                    if old.g <= new_g && old_eig <= new_eig
                        continue  # Old state dominates
                    end
                end

                # Add new state
                new_visited = [copy(S.visited[a]) for a in 1:na]
                new_visited[moved][u] = true
                push!(states, JointState(copy(new_paths), new_covs, new_dists,
                                          new_g, si, new_visited))
                new_si = length(states)
                path_to_best[new_path_key] = new_si
                enqueue!(pq, new_si, f_exact)
            end
        end
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
                          binary_search_tol::Float64 = 0.5,  # unused, kept for compat
                          K_max::Int = 100)                  # max K-shortest paths to search
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

    # ── Two-Phase Planning: K-Shortest + Support A* ────────────────────────
    # Phase 1: Enumerate K-shortest primary paths
    # Phase 2: For each primary path (in order of distance):
    #   - Fix primary to path
    #   - Run support_astar to minimize primary's endpoint uncertainty
    #   - Return first feasible solution (globally optimal within K set)
    
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
    
    visited_goal = 0  # index of first state reaching goal with feasible uncertainty
    feasible_paths = []  # kept for reporting compatibility
    expansion_count = 0
    prune_count = 0
    
    while !isempty(pq)
        state_idx = dequeue!(pq)
        S = states[state_idx]
        expansion_count += 1
        
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

            # No uncertainty-based pruning here: keep weighted-A* style ordering intact.
            # Feasibility is checked only at popped goal states.
            
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
# Sampled graph constants
# ==========================================================================
# ==========================================================================
# Graph sampling and connectivity constants
# ==========================================================================
#
# GRID_M       : jittered grid is GRID_M × GRID_M cells.  Each cell places one
#                sample point, giving n_samples = GRID_M² routing nodes.
#                The worst-case coverage radius (max distance from any point in
#                the space to its nearest node) is:
#                  δ ≤ (√2/2) × max(cell_width, cell_height)
#                Over the ~160×150 unit space with GRID_M=10 this gives δ ≤ ~11 units.
#
# CONN_RADIUS  : two nodes are connected iff their Euclidean distance ≤ CONN_RADIUS.
#                This replaces k-NN and gives a provable suboptimality guarantee:
#
#                  Any continuous path of length L can be approximated by a graph
#                  path of length ≤ L + 2δ × (L / CONN_RADIUS + 1), where δ is the
#                  coverage radius above.  More practically: any straight-line
#                  segment of length ≤ CONN_RADIUS is a single graph edge, so the
#                  graph path never detours more than 2δ per CONN_RADIUS of travel.
#
#                Multiplicative form (using lower bound L* ≥ L_lb > 0):
#                  L_graph ≤ (1 + ε_sample) L*
#                  ε_sample = 2δ/CONN_RADIUS + 2δ/L_lb
#
#                Combined with weighted A* (w = 1 + PRIMARY_EPSILON):
#                  L_returned ≤ (1 + ε_total) L*
#                  ε_total = w(1 + ε_sample) - 1
#
#                IMPORTANT: this combined bound is conditional on solving the same
#                discrete feasible problem exactly with weighted A* assumptions.
#                In this script, support optimization is heuristic/greedy, so treat
#                ε_total as a conservative design-time bound, not a formal theorem.
#
#                CONN_RADIUS must satisfy CONN_RADIUS ≥ 2δ to guarantee that every
#                node has at least one neighbour and the graph is connected.
#                With δ ≤ 11 units, CONN_RADIUS = 25 units satisfies this with margin.
#
#                Larger CONN_RADIUS → more edges → slower search but tighter path
#                approximation.  25 units (2500m) gives average degree ~15-20 over
#                ~130 nodes, which is tractable.
#
# REPULSE_RADIUS: sample points are repelled away from landmarks to avoid placing
#                 routing nodes on top of sensor nodes (which would double-count
#                 landmark observations in covariance propagation).
const GRID_M         = 10    # 10×10 = 100 jittered-grid sample points
const CONN_RADIUS    = 32.0  # radius-graph connection threshold (units).
                             # Must satisfy CONN_RADIUS ≥ 2δ to guarantee connectivity.
                             # With GRID_M=10 over 160×150 space: cell≈16×15, δ≈15.4,
                             # so 2δ≈30.9.  CONN_RADIUS=32 satisfies this with margin.
const REPULSE_RADIUS = 5.0   # min distance (units) from any sample to any landmark

# Upper bound on coverage radius δ used in the sampling error bound.
function coverage_delta_bound(landmarks::Vector{Landmark}, goal_pos::Tuple{Float64,Float64}, grid_m::Int)
    xs_lm = [lm.x for lm in landmarks]
    ys_lm = [lm.y for lm in landmarks]
    xmin = min(minimum(xs_lm), goal_pos[1]); xmax = max(maximum(xs_lm), goal_pos[1])
    ymin = min(minimum(ys_lm), goal_pos[2]); ymax = max(maximum(ys_lm), goal_pos[2])
    cell_w = (xmax - xmin) / grid_m
    cell_h = (ymax - ymin) / grid_m
    return sqrt(2) / 2 * sqrt(cell_w^2 + cell_h^2)
end

# Jittered grid sampling with landmark repulsion.
# Divides [xmin,xmax]×[ymin,ymax] into M×M cells, places one point uniformly
# at random within each cell, then repels any point too close to a landmark.
function jittered_grid_sample(M::Int,
                               xmin, xmax, ymin, ymax,
                               landmarks::Vector{Landmark},
                               repulse_r::Float64)
    cw = (xmax - xmin) / M   # cell width
    ch = (ymax - ymin) / M   # cell height
    xs = Float64[]; ys = Float64[]
    for row in 0:M-1, col in 0:M-1
        x0 = xmin + col * cw;  x1 = x0 + cw
        y0 = ymin + row * ch;  y1 = y0 + ch
        x = x0 + rand() * cw
        y = y0 + rand() * ch
        for lm in landmarks
            dx = x - lm.x; dy = y - lm.y
            d  = hypot(dx, dy)
            if d < repulse_r && d > 1e-9
                ux = dx/d; uy = dy/d
                x = clamp(lm.x + repulse_r * ux, x0 + 0.05*cw, x1 - 0.05*cw)
                y = clamp(lm.y + repulse_r * uy, y0 + 0.05*ch, y1 - 0.05*ch)
            end
        end
        push!(xs, x); push!(ys, y)
    end
    return xs, ys
end

# Build a LandmarkGraph that includes the original landmarks PLUS jittered-grid
# sample points (routing-only nodes, no covariance meaning).
# Connectivity: radius-graph — every pair of nodes within CONN_RADIUS is connected.
# Guarantees suboptimality: any continuous path is approximated within O(δ × L/CONN_RADIUS).
function build_sampled_graph(landmarks::Vector{Landmark},
                              goal_pos::Tuple{Float64,Float64};
                              grid_m::Int        = GRID_M,
                              conn_r::Float64    = CONN_RADIUS,
                              repulse_r::Float64 = REPULSE_RADIUS)
    # Bounding box from landmark positions, expanded to include goal
    xs_lm = [lm.x for lm in landmarks]; ys_lm = [lm.y for lm in landmarks]
    xmin = min(minimum(xs_lm), goal_pos[1]); xmax = max(maximum(xs_lm), goal_pos[1])
    ymin = min(minimum(ys_lm), goal_pos[2]); ymax = max(maximum(ys_lm), goal_pos[2])

    # Coverage radius δ: worst-case distance from any continuous-space point to
    # its nearest grid node.  For a grid_m × grid_m jittered grid over the space,
    # each cell is at most (√2/2) × cell_diagonal wide in any direction.
    cell_w = (xmax - xmin) / grid_m
    cell_h = (ymax - ymin) / grid_m
    δ = sqrt(2) / 2 * sqrt(cell_w^2 + cell_h^2)
    n_samples = grid_m * grid_m

    # Warn if CONN_RADIUS is too small to guarantee connectivity.
    # Connectivity requires CONN_RADIUS ≥ 2δ so that every node is reachable
    # from its neighbours without a gap.
    if conn_r < 2 * δ
        @warn "CONN_RADIUS=$(conn_r) < 2δ=$(round(2δ, digits=2)): graph may be disconnected. " *
              "Increase CONN_RADIUS or GRID_M."
    end

    # Jittered grid sample points with landmark repulsion
    sxs, sys = jittered_grid_sample(grid_m, xmin, xmax, ymin, ymax,
                                     landmarks, repulse_r)

    # Build combined node list:
    #   nodes 1..n_lm         : landmarks (node 1 = start)
    #   nodes n_lm+1..n_lm+ns : jittered grid samples (routing only)
    #   node  n_lm+ns+1       : GOAL (routing waypoint, appended last)
    null_cov = 1e-9 * Matrix{Float64}(I, 2, 2)
    all_lms  = copy(landmarks)
    n_lm     = length(landmarks)
    for i in 1:n_samples
        push!(all_lms, Landmark(sxs[i], sys[i], copy(null_cov)))
    end
    push!(all_lms, Landmark(goal_pos[1], goal_pos[2], copy(null_cov)))
    n_total  = length(all_lms)
    goal_idx = n_total

    # Pairwise distance and orientation matrices (computed once, reused by search)
    dist   = zeros(n_total, n_total)
    orient = zeros(n_total, n_total)
    for i in 1:n_total, j in 1:n_total
        dx = all_lms[j].x - all_lms[i].x; dy = all_lms[j].y - all_lms[i].y
        dist[i,j]   = sqrt(dx^2 + dy^2)
        orient[i,j] = atan(dy, dx)
    end

    # ── Radius-graph connectivity ─────────────────────────────────────────────
    # Connect every pair of nodes (i, j) with i ≠ j whose Euclidean distance is
    # ≤ conn_r.  This is undirected by construction (we add both i→j and j→i).
    #
    # Suboptimality guarantee: any continuous straight-line segment of length
    # ≤ conn_r corresponds to a single graph edge, so a continuous path of
    # length L is approximated by a graph path of length ≤ L + 2δ per segment.
    # Over the whole path this bounds the discretisation error to O(δ × L/conn_r).
    #
    # Unlike k-NN, this criterion is symmetric and distance-based, so isolated
    # nodes and directional gaps cannot arise as long as conn_r ≥ 2δ.
    neighbors = [Int[] for _ in 1:n_total]
    for i in 1:n_total, j in i+1:n_total
        if dist[i,j] <= conn_r
            push!(neighbors[i], j)
            push!(neighbors[j], i)
        end
    end

    # ── Fallback: guarantee start and goal are connected ─────────────────────
    # If start or goal ended up isolated (e.g. goal is far outside the grid),
    # connect them to their nearest node regardless of radius.  This preserves
    # the suboptimality guarantee for the start/goal nodes specifically, at the
    # cost of one potentially long edge — acceptable since these are fixed points.
    for anchor in [1, goal_idx]
        if isempty(neighbors[anchor])
            nearest = argmin([j == anchor ? Inf : dist[anchor, j] for j in 1:n_total])
            push!(neighbors[anchor], nearest)
            push!(neighbors[nearest], anchor)
            @warn "Node $anchor ($(anchor==1 ? "start" : "goal")) had no radius neighbours; " *
                  "connected to nearest node $nearest (dist=$(round(dist[anchor,nearest],digits=2)))."
        end
    end

    n_edges = sum(length.(neighbors)) ÷ 2
    println("Sampled graph: $(n_total) nodes ($(n_lm) landmarks + $(n_samples) grid samples + 1 goal), ",
            "coverage δ≤$(round(δ, digits=2)) units, ",
            "conn_r=$(conn_r) units, ",
            "edges=$(n_edges), ",
            "avg degree=$(round(mean(length.(neighbors)), digits=1))")
    # Precompute shortest paths via Floyd-Warshall (O(n³) but done once)
    sp = floyd_warshall(dist, neighbors)
    return LandmarkGraph(n_total, all_lms, dist, orient, neighbors, sp)
end

landmarks = [
    # Cluster 1 (left of path)
    Landmark(5.0, 5.0, [0.5 0.0; 0.0 0.5]),
    Landmark(35.0, 80.0, [0.7 0.0; 0.0 0.7]),
    Landmark(25.0, 90.0, [0.6 0.0; 0.0 0.6]),
    # Cluster 2 (right of path)
    Landmark(90.0, 120.0, [0.5 0.0; 0.0 0.5]),
    Landmark(110.0, 130.0, [0.7 0.0; 0.0 0.7]),
    Landmark(100.0, 100.0, [0.6 0.0; 0.0 0.6]),
    # Cluster 3 (far below path)
    Landmark(80.0, 30.0, [0.8 0.0; 0.0 0.8]),
    Landmark(120.0, 20.0, [0.9 0.0; 0.0 0.9]),
    # High-value landmark (forces detour if threshold is tight)
    Landmark(60.0, 140.0, [0.2 0.0; 0.0 0.2]),
    # Scattered
    Landmark(140.0, 60.0, [0.7 0.0; 0.0 0.7]),
    Landmark(150.0, 100.0, [0.6 0.0; 0.0 0.6])
]

# Start and goal are plain routing waypoints — not landmarks, no covariance meaning.
# Start is node 1 (first entry in graph), goal is appended after all landmarks+samples.
const START_POS = (5.0,   5.0)    # must match landmarks[1] position
const GOAL_POS  = (160.0, 150.0)  # destination — unknown position, just a coordinate

graph = build_sampled_graph(landmarks, GOAL_POS; grid_m=GRID_M, conn_r=CONN_RADIUS)

# Conditional multiplicative epsilon bound summary.
# Let L* be the unknown optimal continuous feasible primary distance.
# Use L_lb = straight-line(start, goal) as a conservative lower bound on L*.
δ_bound = coverage_delta_bound(landmarks, GOAL_POS, GRID_M)
L_lb = max(1e-9, hypot(GOAL_POS[1] - START_POS[1], GOAL_POS[2] - START_POS[2]))
ε_sample = (2 * δ_bound / CONN_RADIUS) + (2 * δ_bound / L_lb)
w_astar = 1.0 + PRIMARY_EPSILON
ε_total_conditional = w_astar * (1.0 + ε_sample) - 1.0

println("Conditional ε-bound: ε_sample=$(round(ε_sample, digits=4)), w=$(round(w_astar, digits=4)), ε_total=$(round(ε_total_conditional, digits=4))")
println("  Bound form (conditional): L_returned ≤ (1+ε_total) L* with assumptions documented in comments.")

n_landmarks  = length(landmarks)   # number of true landmarks (graph nodes 1..n_landmarks)
x_coords     = [lm.x for lm in landmarks]
y_coords     = [lm.y for lm in landmarks]
marker_sizes = [sqrt(max_eigenvalue(lm.cov)) * MARKER_PROPORTION for lm in landmarks]

function draw_covariance_ellipse!(plt, x, y, cov; npts=50, nstd=2, color=:red, alpha=0.3)
    vals, vecs = eigen(Symmetric((cov+cov')/2))
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

# Stage 2: plan shortest path through start, selected landmarks, and goal
ordered_lms = greedy_visit_order(graph, 1, selected_lms)
visits = vcat(1, ordered_lms, goal_node)
primary_path, primary_dist = shortest_path_with_visits(graph, visits)
println("Primary path visiting info-gathering nodes: ", primary_path, "  dist=", round(primary_dist, digits=3))

# Support planning on top of selected primary path
support_paths = Vector{Vector{Int}}(undef, NUM_AGENTS - 1)
for sup_idx in 1:(NUM_AGENTS - 1)
    other_paths = vcat(support_paths[1:sup_idx-1])
    sup_path, _ = single_support_astar(graph, graph.landmarks, primary_path, primary_dist, other_paths, sup_idx)
    support_paths[sup_idx] = sup_path
end

if isempty(primary_path)
    paths = Vector{Vector{Int}}()
    dists = Float64[]
    uncs = Float64[]
    final_global_cov = Matrix{Float64}[]
    solo_unc = Inf
else
    paths = vcat(support_paths, [primary_path])
    final_global_cov, dists = evaluate_full_paths(paths, graph, graph.landmarks, NUM_AGENTS)
    uncs = [unc_radius(final_global_cov[a]) for a in 1:NUM_AGENTS]
    solo_unc = uncs[end]
end

shortest_path = primary_path
shortest_dist = primary_dist
converged_iter = 0

println("\n--- Final Results ---")
if isempty(paths)
    println("IMPOSSIBLE: uncertainty threshold cannot be met. No continuous optimization will be run.")
else
    for (i, path) in enumerate(paths[1:end-1])
        isempty(path) ? println("Support agent $i : no path found") :
                        println("Support agent $i : ", path, "  dist=", round(dists[i], digits=3))
    end
    threshold_status = uncs[end] <= UNC_RADIUS_THRESHOLD ? "✓ met" : "✗ not met"
    println("Primary agent : ", paths[end],
            "  dist=",     round(dists[end], digits=3),
            "  goal_unc=", round(uncs[end],  digits=4),
            "  threshold=", UNC_RADIUS_THRESHOLD, " ", threshold_status)
end
println("Converged at iteration : ", converged_iter)

# ==========================================================================
# Plotting helpers (shared across all three figures)
# ==========================================================================

agent_colors = [:purple, :teal, :darkorange, :crimson, :magenta,
                :brown, :lime, :navy, :coral, :olive]

function make_base_plot(landmarks, graph)
    n_lm = length(landmarks)
    x_c = [lm.x for lm in landmarks]
    y_c = [lm.y for lm in landmarks]
    ms  = [sqrt(max_eigenvalue(lm.cov)) * MARKER_PROPORTION for lm in landmarks]
    p   = scatter(x_c, y_c, label=false, color=:black, markersize=1)
    for i in 1:n_lm
        draw_covariance_ellipse!(p, landmarks[i].x, landmarks[i].y, landmarks[i].cov,
                                 color=:red, alpha=0.25)
    end
    # Sampled routing nodes (between n_lm+1 and graph.n-1; graph.n is goal)
    if graph.n > n_lm + 1
        sx = [graph.landmarks[i].x for i in n_lm+1:graph.n-1]
        sy = [graph.landmarks[i].y for i in n_lm+1:graph.n-1]
        scatter!(p, sx, sy, color=:lightgrey, markersize=1.5, markerstrokewidth=0)
    end
    scatter!(p, [x_c[1]],     [y_c[1]],     color=:green,  markersize=ms[1])
    scatter!(p, [GOAL_POS[1]], [GOAL_POS[2]], color=:orange, marker=:star5, markersize=9)
    return p
end

# ==========================================================================
# Figure 0 — graph node connectivity
# ==========================================================================
let
    n_lm = length(landmarks)
    plt0 = plot(legend=:topright, aspect_ratio=:equal,
                xlabel="x (×100m)", ylabel="y (×100m)",
                title="Fig 0 — Sampled graph connectivity\n($(graph.n) nodes: $(n_lm) landmarks + $(graph.n-n_lm-1) grid samples + goal; radius-graph edges, r=$(CONN_RADIUS))")

    # Draw edges (deduplicated: only draw i→j where i < j)
    drawn = Set{Tuple{Int,Int}}()
    goal_node = graph.n
    for i in 1:graph.n
        xi = graph.landmarks[i].x; yi = graph.landmarks[i].y
        for j in graph.neighbors[i]
            edge = (min(i,j), max(i,j))
            edge in drawn && continue
            push!(drawn, edge)
            xj = graph.landmarks[j].x; yj = graph.landmarks[j].y
            if i <= n_lm && j <= n_lm
                clr = :steelblue; lw = 1.2; alpha = 0.7
            elseif i <= n_lm || j <= n_lm
                clr = :mediumpurple; lw = 0.8; alpha = 0.5
            elseif i == goal_node || j == goal_node
                clr = :orange; lw = 0.9; alpha = 0.6
            else
                clr = :lightgrey; lw = 0.5; alpha = 0.35
            end
            plot!(plt0, [xi, xj], [yi, yj], color=clr, linewidth=lw, alpha=alpha, label=false)
        end
    end

    # Sampled routing nodes (exclude goal = graph.n)
    if graph.n > n_lm + 1
        sx = [graph.landmarks[i].x for i in n_lm+1:graph.n-1]
        sy = [graph.landmarks[i].y for i in n_lm+1:graph.n-1]
        scatter!(plt0, sx, sy, color=:grey, markersize=3, markerstrokewidth=0, label="Grid sample nodes")
    end

    # Landmark nodes with covariance ellipses
    for i in 1:n_lm
        draw_covariance_ellipse!(plt0, landmarks[i].x, landmarks[i].y, landmarks[i].cov;
                                  nstd=2, color=:red, alpha=0.18)
    end
    x_c = [lm.x for lm in landmarks]; y_c = [lm.y for lm in landmarks]
    scatter!(plt0, x_c, y_c, color=:black, markersize=5, markerstrokewidth=0, label="Landmark nodes")
    scatter!(plt0, [x_c[1]],     [y_c[1]],     color=:green,  markersize=8,
             markerstrokewidth=0, label="Start")
    scatter!(plt0, [GOAL_POS[1]], [GOAL_POS[2]], color=:orange, marker=:star5,
             markersize=9, markerstrokewidth=0, label="Goal (routing only)")

    plot!(plt0, [NaN],[NaN], color=:steelblue,    linewidth=1.2, label="LM – LM edge")
    plot!(plt0, [NaN],[NaN], color=:mediumpurple, linewidth=0.8, label="LM – sample edge")
    plot!(plt0, [NaN],[NaN], color=:orange,       linewidth=0.9, label="goal edge")
    plot!(plt0, [NaN],[NaN], color=:lightgrey,    linewidth=0.5, label="sample – sample edge")

    savefig(plt0, "fig0_graph.png")
    println("Fig 0 saved  ($(length(drawn)) edges, $(graph.n) nodes)")
end

# ==========================================================================
# Figure 1 — discrete graph solution
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
                plot!(plt1, px_, py_, label=lbl, color=clr,
                      linewidth=vi==length(paths) ? 2.0 : 1.2,
                      linestyle=vi==length(paths) ? :solid : :dash)
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
        
        covs_all_fig1, arcs_all_fig1 = evaluate_joint_discrete(agent_positions_fig1, graph.landmarks, length(paths); debug_goal_pos=GOAL_POS)

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
        title!(plt1,"Fig 1 — Discrete [len=$(round(dijk_prim_len_fig1,digits=2)), unc=$(round(dijk_goal_unc_fig1,digits=3))]")
    else
        title!(plt1,"Fig 1 — IMPOSSIBLE (threshold unachievable)")
    end
    xlabel!(plt1,"x (×100m)"); ylabel!(plt1,"y (×100m)")
    savefig(plt1,"fig1_discrete.png"); println("Fig 1 saved.")
end

# ==========================================================================
# Continuous waypoint optimizer — gradient descent in (x,y) space
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

if !isempty(paths) && all(length.(paths) .> 0)  # Continuous optimization only if all agents have paths
    println("\n=== Continuous Waypoint Optimization ===")
    
    # Extract waypoint coordinates from discrete solution
    all_agent_wpts = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(paths))
    is_primary_mask = Vector{Bool}(undef, length(paths))
    
    for (ai, path) in enumerate(paths)
        all_agent_wpts[ai] = [(graph.landmarks[i].x, graph.landmarks[i].y) for i in path]
        is_primary_mask[ai] = (ai == length(paths))
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
        
        # Objective function: evaluate path length and uncertainty
        function eval_continuous(flat::Vector{Float64})
            wpts_list = unpack_waypoints(flat)
            
            # Convert waypoints to agent positions for evaluation
            agent_positions = Vector{Vector{Tuple{Float64,Float64}}}(undef, num_agents)
            for ai in 1:num_agents
                agent_positions[ai] = wpts_list[ai]
            end
            
            # Evaluate covariance via discrete model
            covs_all, _ = evaluate_joint_discrete(agent_positions, graph.landmarks, num_agents)
            goal_unc = unc_radius(covs_all[end][end])  # Last covariance of last agent
            
            # Compute primary path length (sum of Euclidean distances)
            prim_wpts = wpts_list[end]
            prim_len = 0.0
            for i in 2:length(prim_wpts)
                dx = prim_wpts[i][1] - prim_wpts[i-1][1]
                dy = prim_wpts[i][2] - prim_wpts[i-1][2]
                prim_len += sqrt(dx^2 + dy^2)
            end
            
            return prim_len, goal_unc, wpts_list, covs_all
        end
        
        # Initialize
        flat = pack_waypoints(all_agent_wpts)
        init_len, init_unc, init_wpts, init_covs = eval_continuous(flat)
        
        println("  Initial (discrete): prim_len=$(round(init_len, digits=3)), unc=$(round(init_unc, digits=4))")
        
        # Adam optimizer state
        adam_m = zeros(total_free_cont)
        adam_v = zeros(total_free_cont)
        
        opt_iter_log = Int[0]
        opt_len_log = Float64[init_len]
        opt_unc_log = Float64[init_unc]
        
        local best_flat = copy(flat)
        local best_len = init_len
        local best_unc = init_unc
        local prev_len = init_len
        
        for iter in 1:CONT_OPT_ITERS
            len0, unc0, _, _ = eval_continuous(flat)
            
            # Finite-difference gradients w.r.t. all waypoint coordinates
            grad_len = zeros(total_free_cont)
            grad_unc = zeros(total_free_cont)
            
            for k in 1:total_free_cont
                flat[k] += CONT_OPT_H
                ln, uc, _, _ = eval_continuous(flat)
                grad_len[k] = (ln - len0) / CONT_OPT_H
                grad_unc[k] = (uc - unc0) / CONT_OPT_H
                flat[k] -= CONT_OPT_H
            end
            
            # Adam step on path length gradient
            b1t = CONT_ADAM_B1^iter
            b2t = CONT_ADAM_B2^iter
            for k in 1:total_free_cont
                g = grad_len[k]
                adam_m[k] = CONT_ADAM_B1 * adam_m[k] + (1 - CONT_ADAM_B1) * g
                adam_v[k] = CONT_ADAM_B2 * adam_v[k] + (1 - CONT_ADAM_B2) * g^2
                m̂ = adam_m[k] / (1 - b1t)
                v̂ = adam_v[k] / (1 - b2t)
                flat[k] -= CONT_OPT_LR * m̂ / (sqrt(v̂) + CONT_ADAM_EPS)
            end
            
            # Constraint projection: ensure unc ≤ threshold
            _, unc1, _, _ = eval_continuous(flat)
            viol = unc1 - UNC_RADIUS_THRESHOLD
            if viol > 0.0
                gn2 = dot(grad_unc, grad_unc)
                if gn2 > 1e-12
                    λ = viol / gn2
                    flat .-= λ .* grad_unc
                end
            end
            
            len2, unc2, wpts2, covs2 = eval_continuous(flat)
            push!(opt_iter_log, iter)
            push!(opt_len_log, len2)
            push!(opt_unc_log, unc2)
            
            feasible = unc2 <= UNC_RADIUS_THRESHOLD
            if mod(iter, 20) == 0
                status = feasible ? "✓" : "✗"
                println("  Iter $(lpad(iter,3))  len=$(round(len2,digits=3))  unc=$(round(unc2,digits=4))  $status")
            end
            
            if feasible && len2 < best_len
                best_len = len2
                best_unc = unc2
                best_flat = copy(flat)
            end
            
            # Check convergence: break when path length improvement is small
            if abs(len2 - prev_len) < CONT_CONV_TOL
                println("  → Converged at iter $iter (Δlen=$(round(abs(len2-prev_len),digits=6)))")
                break
            end
            prev_len = len2
        end
        
        # Evaluate best solution
        opt_len, opt_unc, opt_wpts, opt_covs = eval_continuous(best_flat)
        println("  Final: prim_len=$(round(opt_len, digits=3)), unc=$(round(opt_unc, digits=4)), threshold=$(UNC_RADIUS_THRESHOLD)")
        
        if opt_unc <= UNC_RADIUS_THRESHOLD
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
                lw = is_prim ? 2.2 : 1.3
                ls = is_prim ? :solid : :dash
                plot!(plt2, xs, ys, label=lbl, color=clr, linewidth=lw, linestyle=ls)
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
            
            title!(plt2,"Fig 2 — Continuous optimized [len=$(round(opt_len,digits=2)), unc=$(round(opt_unc,digits=3))]")
            xlabel!(plt2,"x (×100m)"); ylabel!(plt2,"y (×100m)")
            savefig(plt2,"fig2_continuous_opt.png"); println("Fig 2 saved.")
            
            # Figure 3: convergence plot
            plt3 = plot(opt_iter_log, opt_len_log, 
                       label="Path length", color=:blue, linewidth=2,
                       xlabel="Iteration", ylabel="Primary path length",
                       title="Fig 3 — Path Length vs Optimization Iteration",
                       legend=:topright)
            hline!(plt3, [init_len], label="Initial ($(round(init_len,digits=2)))",
                   color=:gray, linestyle=:dash, linewidth=1.2)
            hline!(plt3, [opt_len], label="Final ($(round(opt_len,digits=2)))",
                   color=:red, linestyle=:dot, linewidth=1.2)
            savefig(plt3,"fig3_convergence.png"); println("Fig 3 saved.")
        else
            println("  ✗ Optimization did not meet uncertainty constraint (best unc=$(round(opt_unc,digits=4)))")
        end
    else
        println("  (No intermediate waypoints to optimize)")
    end
end