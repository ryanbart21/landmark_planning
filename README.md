# landmark_planning

Uncertainty-aware multi-agent planning for a GPS-denied AUV team.

## Pipeline (short version)

1. Build heading-aware hex graph from `START_POS`, `GOAL_POS`, and landmarks.
2. Run joint discrete A* (`joint_astar`) for support + primary paths.
3. Run continuous B-spline refinement to reduce primary length while enforcing uncertainty and curvature constraints.
4. Save figures:
   - `fig1_discrete_*.png` — A* waypoint solution
   - `fig2_continuous_opt_*.png` — Fully refined B-spline solution

## Most important knobs (threshold behavior)

This is the main section to tune for feasibility and runtime.

### Strict uncertainty limit

- `UNC_RADIUS_THRESHOLD`
  - The true feasibility bound for primary goal uncertainty, measured with the determinant-based scalar uncertainty metric.
  - Lower value: harder problem, usually slower search, more failures.
  - Higher value: easier feasibility, typically faster.

- `UNC_FEAS_TOL`
  - Boundary tolerance for feasibility comparisons.
  - Keeps discrete and continuous checks consistent near the threshold.

### Relaxed discrete handoff (speed helper)

- `ENABLE_RELAXED_DISCRETE_FOR_CONTINUOUS`
  - `true`: discrete stage may accept a seed above strict threshold, then continuous stage tries to recover.
  - `false`: discrete stage must satisfy strict threshold directly.

- `RELAXED_DISCRETE_DELTA_MODE`
  - `:absolute` uses `RELAXED_DISCRETE_DELTA_ABS`
  - `:relative` uses `RELAXED_DISCRETE_DELTA_REL * UNC_RADIUS_THRESHOLD`

- Effective relaxed bound used by discrete stage:

$$
U_{disc} = UNC\_RADIUS\_THRESHOLD + \delta
$$

- `CONTINUE_ASTAR_ON_INFEASIBLE`
  - If a relaxed seed fails continuous feasibility, continue A* for another seed.
  - Note: Only used with `ASTAR_MODE = :threshold`; skipped when collecting Pareto set.

Practical guidance:

1. Start strict: relaxed disabled.
2. If search is too slow or frequently fails, enable relaxed handoff and use small `delta`.
3. Increase `delta` gradually if needed; keep it as small as possible.

## Performance knobs

- `PRIMARY_EPSILON`
  - Higher: faster, potentially worse primary length.
  - Lower: slower, closer to shortest primary path.

- Pruning:
  - `PRUNE_BY_COMM_RADIUS_JOINT`
  - `PRUNE_BY_PRIMARY_UNCERTAINTY`
  - `PRUNE_BY_SUPPORT_UNCERTAINTY`

- Continuous optimization budget:
  - `CONT_OPT_ITERS`, `CONT_OPT_LR`, `CONT_OPT_H`
  - `CONT_BARRIER_START`, `CONT_BARRIER_DECAY`, `CONT_BARRIER_STAGES`

## Scenario knobs

- Start/goal: `START_POS`, `GOAL_POS`
- Grid scale: `HEX_WIDTH_M`, `HEX_RADIUS_M`
- Landmark uncertainty: `random_landmark_cov()` uses the same determinant-based covariance metric as the planner.

### Landmark configurations

Toggle `LANDMARK_SCENARIO` (line 35) to switch between three predefined landmark layouts:

- **`:shoreline`** (default)
  - 3 landmarks at y=-300m: (200, -300), (500, -300), (800, -300)
  - Simulates a shoreline the agents travel alongside
  - Provides cross-track observability but limited along-track control

- **`:single`**
  - 1 landmark at (600, -250)
  - Minimal observation geometry
  - Tests planner robustness with limited landmark coverage

- **`:dual`**
  - 2 landmarks at (250, 200) and (750, -250)
  - Symmetric configuration for balanced observability
  - Off-axis placement creates incentive to deviate from Euclidean shortest path

- **`:random`**
  - 8 landmarks randomly spaced across the area (reproducible by the script seed)
  - Each landmark covariance is scaled by `8 * rand()` to create diverse landmark quality
  - Useful for stress-testing planner behavior under heterogeneous landmark configurations

Each scenario uses `random_landmark_cov()` for consistent initialization across runs.

### A* collection mode

Toggle `ASTAR_MODE` (line 34) to select discrete search strategy:

- **`:limit`** (default)
  - Runs A* collector until `ASTAR_ITERATION_LIMIT` (line 33) is reached
  - Collects **all** Pareto-optimal solutions (non-dominated on distance vs. uncertainty)
  - Returns all seeds to `PARETO_COLLECTED` for continuous refinement
  - Best for exploring trade-off frontier; finds multiple distinct solutions
  - Longer runtime but discovers full solution space
  - Generates plots for each Pareto seed: `pareto_1_*`, `pareto_2_*`, etc.

- **`:threshold`**
  - Stops A* on first feasible path below uncertainty threshold
  - Returns single representative seed
  - Faster but may miss alternative solutions
  - Useful for quick feasibility checks or when solution diversity is not needed
  - Generates single main plot only

**Interaction with `CONTINUE_ASTAR_ON_INFEASIBLE`:**
- When `ASTAR_MODE = :limit`, the flag is ignored (no re-run since we intentionally collect all Pareto seeds)
- When `ASTAR_MODE = :threshold`, re-runs A* if first seed exceeds threshold

## Output plots

Each continuous optimization run generates two plots:

1. **`fig1_discrete_*.png`** — A* discrete path solution (waypoints connected linearly)
   - Shows nodes visited by A* planning
   - Displays initial discrete uncertainty at goal
   - **Length** = path length through waypoint sequence
   - **Unc** = goal uncertainty from Kalman fusion

2. **`fig2_continuous_opt_*.png`** — Fully refined continuous B-spline solution
   - Shows optimized waypoint positions after barrier method + Adam optimizer
   - Displays refined uncertainty at goal (typically lower than discrete)
   - **Length** = smooth B-spline path length
   - **Unc** = goal uncertainty evaluated on refined path

Additional plots:
- **`fig_pareto_discrete.png`** — Pareto front of (distance, uncertainty) pairs from A* (discrete stage only)
- **`fig_pareto_continuous_overlay.png`** — All refined Pareto solutions overlaid (when `ASTAR_MODE = :limit`)

## Guarantees (concise)

- A* ordering uses weighted key `f = g + (1 + PRIMARY_EPSILON) * h`.
- With `PRIMARY_EPSILON = 0`, this is standard A* ordering for primary-cost search.
- Strict feasibility is always evaluated against `UNC_RADIUS_THRESHOLD` (with `UNC_FEAS_TOL`).
- Relaxed discrete mode is a speed mechanism for seed generation, not a replacement for strict final feasibility.

## Run

```bash
julia planner.jl
```

Outputs are saved to the same directory with PNG figures and CSV control-point files.
