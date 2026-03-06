using Plots
using DataStructures
using LinearAlgebra

# ----------------------
# Constants & Parameters
# ----------------------
# Physical scale: 1 unit = 10m. Graph spans ~200m x 200m.
# Platform: AUV with DVL+IMU dead reckoning, acoustic landmark fixes.
#
# DIR_UNCERTAINTY_PER_METER : 10% dead-reckoning drift (DVL+IMU, along-track)
# MAJ_MIN_UNC_RATIO         : along-track drift ~3x cross-track (DVL characteristic)
# SENSOR_NOISE              : USBL/LBL fix accuracy ~1m (0.1 units)
# COMM_RADIUS               : acoustic modem range ~80m (8 units)
# UNC_RADIUS_THRESHOLD      : max acceptable sqrt(λ_max(Σ_goal)) at the goal.
#                             Primary agent minimises path length subject to this.
#                             Support agents position to minimise this value so
#                             the constraint becomes satisfiable / tighter.

const DIR_UNCERTAINTY_PER_METER  = 0.10
const MAJ_MIN_UNC_RATIO          = 3
const PERP_UNCERTAINTY_PER_METER = DIR_UNCERTAINTY_PER_METER / MAJ_MIN_UNC_RATIO
const MARKER_PROPORTION          = 50.0
const NUM_AGENTS                 = 3
const SENSOR_NOISE               = 0.1
const COMM_RADIUS                = 8.0
const UNC_RADIUS_THRESHOLD       = 0.3   # units (= 6m); tighten to force more support routing

# ----------------------
# Data Structures
# ----------------------
# Primary agent state: (node, cumulative distance, covariance at node).
# Dominance is over (dist, max_eigenvalue(cov)) — no risk field.
struct State
    node::Int
    dist::Float64
    cov::Matrix{Float64}
    parent::Int
end

# Support agent state: tracks predicted primary goal uncertainty if support
# stops here. Objective: minimise goal_unc.
struct SupportState
    node::Int
    dist::Float64
    cov::Matrix{Float64}
    goal_unc::Float64   # sqrt(λ_max(Σ_goal)) primary achieves with this support path
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
            dx = lj.x - li.x; dy = lj.y - li.y
            dist[i,j]   = sqrt(dx^2 + dy^2)
            orient[i,j] = atan(dy, dx)
        end
    end
    return LandmarkGraph(n, landmarks, dist, orient)
end

function max_eigenvalue(cov::Matrix{Float64})
    return maximum(real(eigvals(Symmetric((cov + cov') / 2))))
end

unc_radius(cov::Matrix{Float64}) = sqrt(max_eigenvalue(cov))

function growth_covariance(distance::Float64, angle::Float64)
    sd = DIR_UNCERTAINTY_PER_METER  * distance
    sp = PERP_UNCERTAINTY_PER_METER * distance
    R  = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    return R * Diagonal([sd^2, sp^2]) * R'
end

function fuse_cov(path_cov::Matrix{Float64}, landmark_cov::Matrix{Float64})
    R_s = SENSOR_NOISE^2 * I(2)
    return inv(inv(path_cov) + inv(landmark_cov + R_s))
end

# ----------------------
# Covariance propagation
# ----------------------
# Propagate covariance along a path; fuse landmark at each non-goal node.
function trace_path_covs(graph::LandmarkGraph,
                         path::Vector{Int},
                         global_cov::Vector{Matrix{Float64}})
    isempty(path) && return Matrix{Float64}[]
    goal_node   = path[end]
    covs        = Matrix{Float64}[]
    current_cov = copy(global_cov[path[1]])
    push!(covs, copy(current_cov))
    for k in 2:length(path)
        v = path[k-1]; u = path[k]
        current_cov = current_cov + growth_covariance(graph.distance[v,u], graph.orientation[v,u])
        u != goal_node && (current_cov = fuse_cov(current_cov, global_cov[u]))
        push!(covs, copy(current_cov))
    end
    return covs
end

# Predict goal uncertainty for the primary on primary_path given global_cov.
function predict_goal_unc(graph::LandmarkGraph,
                          primary_path::Vector{Int},
                          global_cov::Vector{Matrix{Float64}})
    isempty(primary_path) && return Inf
    covs = trace_path_covs(graph, primary_path, global_cov)
    return unc_radius(covs[end])
end

# ----------------------
# Unconstrained shortest path (Dijkstra)
# ----------------------
function search_shortest_path(graph::LandmarkGraph)
    n = graph.n; dist = fill(Inf, n); parent = fill(-1, n)
    dist[1] = 0.0
    pq = PriorityQueue{Int,Float64}(); enqueue!(pq, 1, 0.0)
    while !isempty(pq)
        v = dequeue!(pq)
        for u in 1:n
            u == v && continue
            nd = dist[v] + graph.distance[v, u]
            if nd < dist[u]; dist[u] = nd; parent[u] = v; pq[u] = nd; end
        end
    end
    isinf(dist[n]) && return Int[], Inf
    path = Int[]; v = n
    while v != -1; push!(path, v); v = parent[v]; end
    return reverse!(path), dist[n]
end

# ----------------------
# Primary agent search
# ----------------------
# Objective  : minimise cumulative path distance
# Constraint : unc_radius(Σ_goal) ≤ UNC_RADIUS_THRESHOLD
#
# Dominance  : state A dominates B at node u iff
#              A.dist ≤ B.dist  AND  max_eigenvalue(A.cov) ≤ max_eigenvalue(B.cov)
#
# Pruning    : if unc_radius(cov) already exceeds threshold at an intermediate
#              node, the covariance can only grow, so the branch is pruned.
function search_main_agent!(graph::LandmarkGraph,
                            global_cov::Vector{Matrix{Float64}},
                            unc_threshold::Float64;
                            update_global::Bool = true,
                            dist_cap::Float64   = Inf)
    n           = graph.n
    states      = State[]
    node_states = [Int[] for _ in 1:n]
    pq          = PriorityQueue{Int, Float64}()

    push!(states, State(1, 0.0, copy(global_cov[1]), -1))
    push!(node_states[1], 1); enqueue!(pq, 1, 0.0)

    best_goal_dist = Inf; goal_state = 0

    while !isempty(pq)
        si = dequeue!(pq)
        S  = states[si]; v, d, cov = S.node, S.dist, S.cov
        si ∉ node_states[v] && continue

        if v == n
            if d < best_goal_dist; best_goal_dist = d; goal_state = si; end
            continue
        end

        for u in 1:n
            u == v && continue
            anc = si; cycle = false
            while anc != -1
                states[anc].node == u && (cycle = true; break); anc = states[anc].parent
            end
            cycle && continue

            edge_dist = graph.distance[v, u]; angle = graph.orientation[v, u]
            new_dist  = d + edge_dist
            (new_dist > best_goal_dist || new_dist > dist_cap) && continue

            new_cov = cov + growth_covariance(edge_dist, angle)
            u != n && (new_cov = fuse_cov(new_cov, global_cov[u]))

            # Prune: covariance only grows, so if already over threshold, skip
            unc_radius(new_cov) > unc_threshold && continue

            new_eig    = max_eigenvalue(new_cov)
            dominated  = false; to_remove = Int[]
            for old_si in node_states[u]
                old = states[old_si]; old_eig = max_eigenvalue(old.cov)
                if old.dist <= new_dist && old_eig <= new_eig; dominated = true; break; end
                new_dist <= old.dist && new_eig <= old_eig && push!(to_remove, old_si)
            end
            dominated && continue
            for rem in to_remove
                deleteat!(node_states[u], findfirst(==(rem), node_states[u]))
            end

            push!(states, State(u, new_dist, new_cov, si))
            new_si = length(states); push!(node_states[u], new_si)
            enqueue!(pq, new_si, new_dist)
        end

        !isempty(pq) && peek(pq)[2] >= best_goal_dist && break
    end

    goal_state == 0 && return Int[], Inf, Inf

    path = Int[]; si = goal_state
    while si != -1; push!(path, states[si].node); si = states[si].parent; end
    reverse!(path); final = states[goal_state]

    if update_global
        path_covs = trace_path_covs(graph, path, global_cov)
        for (k, node) in enumerate(path)
            node == n && continue
            global_cov[node] = fuse_cov(global_cov[node], path_covs[k])
        end
    end

    return path, final.dist, unc_radius(final.cov)
end

# ----------------------
# Support agent search
# ----------------------
# Support agents traverse the graph within dist_cap. At each visited landmark
# they broadcast measurements to temporally-reachable primary-path nodes
# (support must arrive before the primary). The objective is to minimise the
# primary's goal uncertainty. The search keeps the stopping point that gives
# the smallest predicted goal_unc.
#
# Dominance: A dominates B at u if A.dist ≤ B.dist AND A.goal_unc ≤ B.goal_unc.
function search_support_agent!(graph::LandmarkGraph,
                                global_cov::Vector{Matrix{Float64}},
                                dist_cap::Float64,
                                primary_path::Vector{Int})
    n  = graph.n; np = length(primary_path)

    d_prim = zeros(np)
    for k in 2:np
        d_prim[k] = d_prim[k-1] + graph.distance[primary_path[k-1], primary_path[k]]
    end

    # Apply a support path's measurements to a fresh copy of global_cov and
    # return the resulting primary goal uncertainty.
    function eval_goal_unc(sup_nodes::Vector{Int}, sup_cum_dists::Vector{Float64})
        gcov = [copy(c) for c in global_cov]
        for idx in 1:length(sup_nodes)
            node    = sup_nodes[idx]; d_sup = sup_cum_dists[idx]
            lm_cov  = graph.landmarks[node].cov + SENSOR_NOISE^2 * I(2)
            for k in 1:np
                d_sup > d_prim[k] && continue
                pnode = primary_path[k]
                w     = exp(-graph.distance[node, pnode]^2 / (2*COMM_RADIUS^2))
                w < 1e-4 && continue
                gcov[pnode] = fuse_cov(gcov[pnode], lm_cov / w)
            end
        end
        return predict_goal_unc(graph, primary_path, gcov)
    end

    # Helper: reconstruct node sequence + cumulative distances up to state si
    function reconstruct(si::Int)
        nodes = Int[]; dists = Float64[]; anc = si
        while anc != -1
            push!(nodes, states[anc].node); push!(dists, states[anc].dist)
            anc = states[anc].parent
        end
        return reverse!(nodes), reverse!(dists)
    end

    states      = SupportState[]
    node_states = [Int[] for _ in 1:n]
    pq          = PriorityQueue{Int, Float64}()

    init_gu = eval_goal_unc(Int[], Float64[])
    push!(states, SupportState(1, 0.0, copy(global_cov[1]), init_gu, -1))
    push!(node_states[1], 1); enqueue!(pq, 1, init_gu)

    best_goal_unc = init_gu; goal_state = 0

    while !isempty(pq)
        si  = dequeue!(pq)
        S   = states[si]; v, d, cov, gu = S.node, S.dist, S.cov, S.goal_unc
        si ∉ node_states[v] && continue

        if d > 0 && d <= dist_cap && gu < best_goal_unc
            best_goal_unc = gu; goal_state = si
        end

        for u in 1:n
            u == v && continue
            anc = si; cycle = false
            while anc != -1
                states[anc].node == u && (cycle = true; break); anc = states[anc].parent
            end
            cycle && continue

            edge_dist = graph.distance[v, u]; new_dist = d + edge_dist
            new_dist > dist_cap && continue

            angle   = graph.orientation[v, u]
            new_cov = (u == n) ? cov + growth_covariance(edge_dist, angle) :
                                  fuse_cov(cov + growth_covariance(edge_dist, angle), global_cov[u])

            prefix_nodes, prefix_dists = reconstruct(si)
            push!(prefix_nodes, u); push!(prefix_dists, new_dist)
            new_gu = eval_goal_unc(prefix_nodes, prefix_dists)

            dominated = false; to_remove = Int[]
            for old_si in node_states[u]
                old = states[old_si]
                if old.dist <= new_dist && old.goal_unc <= new_gu; dominated = true; break; end
                new_dist <= old.dist && new_gu <= old.goal_unc && push!(to_remove, old_si)
            end
            dominated && continue
            for rem in to_remove
                deleteat!(node_states[u], findfirst(==(rem), node_states[u]))
            end

            push!(states, SupportState(u, new_dist, new_cov, new_gu, si))
            new_si = length(states); push!(node_states[u], new_si)
            enqueue!(pq, new_si, new_gu)
        end
    end

    goal_state == 0 && return Int[], Inf, Inf

    path = Int[]; si = goal_state
    while si != -1; push!(path, states[si].node); si = states[si].parent; end
    reverse!(path); final = states[goal_state]

    # Commit this support agent's measurements into global_cov
    acc = 0.0; sup_cum = Float64[]
    for idx in 1:length(path)
        idx > 1 && (acc += graph.distance[path[idx-1], path[idx]])
        push!(sup_cum, acc)
    end
    for idx in 1:length(path)
        node   = path[idx]; d_sup = sup_cum[idx]
        lm_cov = graph.landmarks[node].cov + SENSOR_NOISE^2 * I(2)
        for k in 1:np
            d_sup > d_prim[k] && continue
            pnode = primary_path[k]
            w = exp(-graph.distance[node, pnode]^2 / (2*COMM_RADIUS^2))
            w < 1e-4 && continue
            global_cov[pnode] = fuse_cov(global_cov[pnode], lm_cov / w)
        end
    end

    return path, final.dist, final.goal_unc
end

# ----------------------
# Iteration & outer loop
# ----------------------
function run_iteration(graph::LandmarkGraph,
                       provisional_path::Vector{Int},
                       provisional_dist::Float64,
                       unc_threshold::Float64,
                       num_ag::Int)
    n          = graph.n
    global_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    sup_paths  = Vector{Int}[]; sup_dists = Float64[]; sup_uncs = Float64[]

    for _ in 2:num_ag
        path, d, gu = search_support_agent!(graph, global_cov,
                                            provisional_dist, provisional_path)
        push!(sup_paths, path)
        push!(sup_dists, isempty(path) ? Inf : d)
        push!(sup_uncs,  isempty(path) ? Inf : gu)
    end

    main_path, main_d, main_gu =
        search_main_agent!(graph, global_cov, unc_threshold; update_global=true)

    return main_path, main_d, main_gu, sup_paths, sup_dists, sup_uncs, global_cov
end

function multi_agent_usp(graph::LandmarkGraph,
                         unc_threshold::Float64,
                         num_ag::Int = 1;
                         max_iter::Int = 5)
    n = graph.n

    shortest_path, shortest_dist = search_shortest_path(graph)
    if isempty(shortest_path)
        println("No path exists in graph.")
        return Vector{Int}[], Float64[], Float64[],
               [copy(graph.landmarks[i].cov) for i in 1:n], Int[], Inf, Inf, 0
    end

    init_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    solo_path, solo_dist, solo_unc =
        search_main_agent!(graph, init_cov, unc_threshold; update_global=false)

    println("Unconstrained shortest : ", shortest_path,
            "  dist=", round(shortest_dist, digits=3))
    println("Solo (no support)      : path=", solo_path,
            "  dist=", round(solo_dist, digits=3),
            "  unc=",  round(solo_unc,  digits=4))

    provisional_path = isinf(solo_dist) ? shortest_path : solo_path
    provisional_dist = isinf(solo_dist) ? shortest_dist : solo_dist

    isinf(solo_dist) && println("No solo feasible path — support agents needed.")

    best_paths = Vector{Int}[]; best_dists = Float64[]; best_uncs = Float64[]
    last_global_cov = [copy(graph.landmarks[i].cov) for i in 1:n]
    converged_iter  = 0

    for iter in 1:max_iter
        main_path, main_d, main_gu, sup_paths, sup_dists, sup_uncs, global_cov =
            run_iteration(graph, provisional_path, provisional_dist, unc_threshold, num_ag)

        if isempty(main_path)
            println("  Iter $iter: primary found no feasible path — stopping.")
            break
        end

        println("  Iter $iter: primary path=", main_path,
                "  dist=", round(main_d,  digits=3),
                "  unc=",  round(main_gu, digits=4),
                "  (threshold=", unc_threshold, ")")

        best_paths = vcat(sup_paths, [main_path])
        best_dists = vcat(sup_dists, [main_d])
        best_uncs  = vcat(sup_uncs,  [main_gu])
        last_global_cov = global_cov; converged_iter = iter

        if main_path == provisional_path
            println("  Converged at iteration $iter."); break
        end
        provisional_path = main_path; provisional_dist = main_d
    end

    return best_paths, best_dists, best_uncs,
           last_global_cov, shortest_path, shortest_dist, solo_unc, converged_iter
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
    a = nstd*sqrt(max(vals[1],0.0)); b = nstd*sqrt(max(vals[2],0.0))
    angle = atan(vecs[2,1], vecs[1,1]); θ = range(0, 2π, length=npts)
    R = [cos(angle) -sin(angle); sin(angle) cos(angle)]
    pts = R * vcat((a.*cos.(θ))', (b.*sin.(θ))')
    plot!(plt, x.+pts[1,:], y.+pts[2,:], seriestype=:shape, color=color, alpha=alpha, label=false)
end

plt = scatter(x_coords[2:end-1], y_coords[2:end-1], label=false, color=:black, markersize=1)
for i in 2:length(landmarks)-1
    draw_covariance_ellipse!(plt, landmarks[i].x, landmarks[i].y, landmarks[i].cov,
                             color=:red, alpha=0.3)
end
scatter!(plt, [x_coords[1]],   [y_coords[1]],   label="Start", color=:green, markersize=marker_sizes[1])
scatter!(plt, [x_coords[end]], [y_coords[end]],  label="Goal",  marker=:star5, color=:orange, markersize=7)

# ----------------------
# Run
# ----------------------
paths, dists, uncs, final_global_cov, shortest_path, shortest_dist, solo_unc, converged_iter =
    multi_agent_usp(graph, UNC_RADIUS_THRESHOLD, NUM_AGENTS; max_iter=5)

println("\n--- Final Results ---")
for (i, path) in enumerate(paths[1:end-1])
    isempty(path) ? println("Support agent $i : no path found") :
                    println("Support agent $i : ", path, "  dist=", round(dists[i], digits=3))
end
if isempty(paths[end])
    println("Primary agent : no feasible path found")
else
    println("Primary agent : ", paths[end],
            "  dist=",     round(dists[end], digits=3),
            "  goal_unc=", round(uncs[end],  digits=4),
            "  (threshold=", UNC_RADIUS_THRESHOLD, ")")
end
println("Converged at iteration : ", converged_iter)

# ----------------------
# B-Spline Interpolation
# ----------------------
const BSPLINE_NPTS    = 20
const FLANK_RATIO     = 0.18
const AUV_TURN_RADIUS = 4.0  # 40m in graph units (1 unit = 10m)

function min_flank_for_radius(dx_in, dy_in, dx_out, dy_out, R_min::Float64)::Float64
    d_in = hypot(dx_in,dy_in); d_out = hypot(dx_out,dy_out)
    (d_in < 1e-12 || d_out < 1e-12) && return 0.0
    cos_int = clamp((dx_in*dx_out+dy_in*dy_out)/(d_in*d_out), -1.0, 1.0)
    return (4.0/3.0) * R_min * sin((π - acos(cos_int))/2)^2
end

function bspline_basis(t::Float64)
    return (1-t)^3/6, (3t^3-6t^2+4)/6, (-3t^3+3t^2+3t+1)/6, t^3/6
end

function bspline_eval(px, py, s::Float64)
    m = length(px); ns = m-3
    i = clamp(floor(Int, s*ns)+1, 1, ns); t = s*ns-(i-1)
    b0,b1,b2,b3 = bspline_basis(t)
    return b0*px[i]+b1*px[i+1]+b2*px[i+2]+b3*px[i+3],
           b0*py[i]+b1*py[i+1]+b2*py[i+2]+b3*py[i+3]
end

function expand_control_points(xs, ys; R_min=AUV_TURN_RADIUS)
    n = length(xs); px=Float64[]; py=Float64[]
    for k in 1:n
        if k==1 && n>1
            dx=xs[2]-xs[1]; dy=ys[2]-ys[1]; f=clamp(FLANK_RATIO,0.,0.45)
            push!(px,xs[1]);push!(py,ys[1]); push!(px,xs[1]+f*dx);push!(py,ys[1]+f*dy)
        elseif k==1
            push!(px,xs[1]); push!(py,ys[1])
        elseif k==n
            dx=xs[n]-xs[n-1]; dy=ys[n]-ys[n-1]; f=clamp(FLANK_RATIO,0.,0.45)
            push!(px,xs[n]-f*dx);push!(py,ys[n]-f*dy); push!(px,xs[n]);push!(py,ys[n])
        else
            dxi=xs[k]-xs[k-1]; dyi=ys[k]-ys[k-1]
            dxo=xs[k+1]-xs[k]; dyo=ys[k+1]-ys[k]
            di=hypot(dxi,dyi); do_=hypot(dxo,dyo)
            h = clamp(max(FLANK_RATIO*min(di,do_), min_flank_for_radius(dxi,dyi,dxo,dyo,R_min)),
                      0., 0.45*min(di,do_))
            push!(px,xs[k]-h*dxi/di); push!(py,ys[k]-h*dyi/di)
            push!(px,xs[k]);          push!(py,ys[k])
            push!(px,xs[k]+h*dxo/do_);push!(py,ys[k]+h*dyo/do_)
        end
    end
    return px, py
end

function bspline_curve(xs, ys; npts=BSPLINE_NPTS)
    n=length(xs)
    n==1 && return xs, ys
    n==2 && return collect(range(xs[1],xs[2],length=npts)), collect(range(ys[1],ys[2],length=npts))
    epx,epy = expand_control_points(xs,ys)
    px=[epx[1];epx[1];epx;epx[end];epx[end]]; py=[epy[1];epy[1];epy;epy[end];epy[end]]
    ox=Vector{Float64}(undef,npts); oy=Vector{Float64}(undef,npts)
    for j in 1:npts; ox[j],oy[j]=bspline_eval(px,py,(j-1)/(npts-1)); end
    return ox, oy
end

# ----------------------
# Plot
# ----------------------
agent_colors = [:purple, :teal, :darkorange, :crimson, :magenta, :brown, :lime, :navy, :coral, :olive]

for (i, path) in enumerate(paths)
    isempty(path) && continue
    path_x = [graph.landmarks[idx].x for idx in path]
    path_y = [graph.landmarks[idx].y for idx in path]
    clr    = i == length(paths) ? :blue : (i <= length(agent_colors) ? agent_colors[i] : :gray)
    lbl_s  = i == length(paths) ? "Primary (B-spline)" : "Support $i (B-spline)"

    plot!(plt, path_x, path_y, label=false, color=clr,
          linewidth=(i==length(paths) ? 0.6 : 0.4), linestyle=:dot,
          alpha=(i==length(paths) ? 0.35 : 0.25))
    sx, sy = bspline_curve(path_x, path_y; npts=BSPLINE_NPTS)
    plot!(plt, sx, sy, label=lbl_s, color=clr,
          linewidth=(i==length(paths) ? 2.5 : 1.4),
          linestyle=(i==length(paths) ? :solid : :dash))
end

scatter!(plt, x_coords[2:end-1], y_coords[2:end-1], label=false, color=:black, markersize=2.5)

primary_path = isempty(paths) ? Int[] : paths[end]
if !isempty(primary_path)
    path_covs = trace_path_covs(graph, primary_path, final_global_cov)
    for (k, node) in enumerate(primary_path)
        draw_covariance_ellipse!(plt, graph.landmarks[node].x, graph.landmarks[node].y,
                                 path_covs[k]; nstd=2, color=:blue, alpha=0.18)
    end
    plot!(plt, [NaN], [NaN], seriestype=:shape, color=:blue, alpha=0.18, label="Primary 2σ cov")
end

xlabel!(plt, "x (×10m)"); ylabel!(plt, "y (×10m)")
title!(plt, "AUV Swarm — Min-Distance s.t. Goal Unc ≤ $(UNC_RADIUS_THRESHOLD) units")
savefig(plt, "multi_agent_paths_bspline.png")
println("\nPlot saved to multi_agent_paths_bspline.png")