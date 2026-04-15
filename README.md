# landmark_planning

Uncertainty-aware multi-agent planning for a GPS-denied AUV team.

## Pipeline (short version)

1. Build heading-aware hex graph from `START_POS`, `GOAL_POS`, and landmarks.
2. Run joint discrete A* (`joint_astar`) for support + primary paths.
3. Run continuous B-spline refinement to reduce primary length while enforcing uncertainty and curvature constraints.
4. Save figures:
   - `fig1_joint_discrete_astar.png`
   - `fig2_continuous_opt.png`
   - `fig3_convergence.png`

## Most important knobs (threshold behavior)

This is the main section to tune for feasibility and runtime.

### Strict uncertainty limit

- `UNC_RADIUS_THRESHOLD`
  - The true feasibility bound for primary goal uncertainty.
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
- Landmark placement: `make_scattered_landmarks(...)`
- Landmark uncertainty: `random_landmark_cov()`
- Grid scale: `HEX_WIDTH_M`, `HEX_RADIUS_M`

## Guarantees (concise)

- A* ordering uses weighted key `f = g + (1 + PRIMARY_EPSILON) * h`.
- With `PRIMARY_EPSILON = 0`, this is standard A* ordering for primary-cost search.
- Strict feasibility is always evaluated against `UNC_RADIUS_THRESHOLD` (with `UNC_FEAS_TOL`).
- Relaxed discrete mode is a speed mechanism for seed generation, not a replacement for strict final feasibility.

## Run

```bash
julia planner.jl
```
