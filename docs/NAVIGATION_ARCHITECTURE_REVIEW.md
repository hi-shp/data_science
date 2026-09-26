# Navigation architecture experiment — 2026-09-26

## Frozen starting point

See `KNOWN_GOOD_DYNAMICS.md`. Baseline source and local leaderboard changes are
backed up under `data/architecture_rethink/baseline-sgl_aqsn/`. No physics, playback,
commit or push is part of this experiment. New candidates are opt-in modules under
`experiments/`; the production GUI remains on its existing algorithm until a
candidate earns promotion. No 200-episode benchmark.

## Existing pipeline and hypotheses

LiDAR returns feed `NavigationMap` and decaying hit-grid clustering. Gap state
(`current_wp`, queued `next_wp`) can bias the A* corridor, but the endpoint is the
global goal. A* routes are smoothed and validated against perception; continuous
lookahead feeds the final predictive speed/yaw command selector. Its candidates
hold one command over the horizon rather than optimizing a command sequence.

Confirmed preceding evidence: legitimate route replacement plus smoothing is a
CPU hotspot, seed 2189 times out, and playback requests beyond CPU capacity fall
behind. Hypotheses to test: redundant passage state can be removed; a coarse
topological guide plus dynamically feasible local rollout may remove expensive
geometric smoothing and allow turn-then-straight actions. It is not established
that removing Gap improves reliability, or that MPPI is faster.

Keep LiDAR, perception-only map, surveyed boundary prior, full hull checks,
unchanged allocator/integrator, deterministic evaluation. Candidate removals are
Gap passage state, queued waypoints and geometric smoothing from the control
pipeline. Their removal from production requires comparative evidence.

## Research and applicability

Primary sources checked on 2026-09-26. Published performance on other robots is
not a timing estimate for this Python USV simulation.

| Family | Relevance to this vessel | Decision for prototype shortlist |
|---|---|---|
| A* + smooth tracking | Existing reliable reference, but geometric paths do not encode lag/sway and repeated smoothing is costly | Keep current pipeline as reference A |
| Hybrid A* / state lattice | Heading-aware search can improve feasibility; marine primitives need velocity, yaw and actuator state, increasing dimensionality | Defer; car primitives are not a marine model |
| Regulated Pure Pursuit | Low-cost curvature/clearance-based speed regulation; needs an additional dynamic safety rollout for this vessel | Defer; does not alone address inertia/lag or local minima |
| DWA | Reachable command window and admissible stopping trajectories; existing constant-command rollout is related, but original synchro-drive arcs cannot be copied | Baseline already covers a related alternative |
| nonlinear MPC | Can directly encode physical dynamics and constraints, but nonconvex obstacle avoidance and solver warm starts add complexity | Defer solver/model rewrite |
| MPPI / sampling MPC | Receding-horizon sequences naturally use the existing batch vessel model; multi-modal sampling and collision checks suit this representation | Prototype B with coarse guide, C without guide |
| lattice / motion primitives | Explicitly feasible vessel maneuvers are explainable, but a precomputed library must cover speed/lag states | Use generic turn/stop/reverse exploration sequences inside B/C, not a lattice planner |
| RRT* / closed-loop sampling | Useful for larger topology searches, less compelling for this small rapidly updated local map; asymptotic optimality is not a fixed-time guarantee | Defer |

Sources:

- [Nav2 Smac A*, Hybrid A*, lattice](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/planners_plugins/smac/).
- [Regulated Pure Pursuit paper](https://arxiv.org/abs/2305.20026).
- [Fox, Burgard, Thrun: Dynamic Window Approach](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance).
- [CasADi optimal-control framework](https://web.casadi.org/docs/).
- [Williams et al.: MPPI with importance sampling](https://arxiv.org/abs/1509.01149).
- [Nav2 MPPI implementation and configuration](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/controller_plugins/mppi_controller/configuring_mppic/).
- [Karaman/Frazzoli: sampling-based optimal motion planning](https://arxiv.org/abs/1105.1186).

## Shortlist and experimental contract

A: current baseline. B: unchanged perception map → coarse perception-only A*
without Gap/smoothing → sampling MPC. C: same sampling MPC directly to goal.
B is the initial recommendation because coarse topology should reduce the local
minimum risk of C. CPU complexity is approximately samples × horizon × perceived
obstacles plus A* for B; actual latency must be measured, not borrowed from Nav2.

The experiment optimizes bounded speed/yaw-rate command sequences through the
unchanged thrust allocator, rather than bypassing its independently frozen yaw
gain with arbitrary motor commands. Every rollout calls the exact existing
`integrate` at 0.04 s and refreshes `allocate` each step. Commands last three steps,
matching execution. The selected sequence maps to left/right thrust through the
same physical actuator model. This preserves the user-approved response contract.

The optimizer is MPPI-inspired sampling MPC, not a mathematically exact MPPI
implementation. Correlated Gaussian perturbations around a shifted warm start
are augmented by generic forward/turn/brake/reverse sequences. Costs combine
route/goal progress, cross-track distance, command changes, turn rate, clearance
and effort. Collision-free candidates are ranked first. A soft-weighted proposal
is separately rolled out and adopted only when feasible and cheaper than the
best sample; averaging safe opposite-side paths does not guarantee safety.
No safe horizon means best predicted clearance, not a claim of safety guarantee.

No online function receives ground truth. `SamplingNavigator.plan` accepts only
state, observation map, goal and frame. Ground truth is accessed only by the scan
simulator and evaluator. Unknown A* cells retain the explicit high cost; they are
not marked observed free. Occupied/free memory lifetimes are unchanged.

Prototype trials: tuning maps 2000, 2081, 2189. Comparative validation maps 2069,
2168, 2037. Final holdout maps must be separate and tested only after selection.
Timing runs are sequential to avoid competing prototype processes.

## Small prototype results and decision gate

All six representative maps use identical frozen dynamics, map geometry, scan
range, observation memory, dt and three-step control cadence. A/B/C below are the
initial (command-change weight 3) prototypes, not final promoted settings.

| Seed | Existing A | Coarse-guide B | Direct C |
|---|---:|---:|---:|
| 2000 | success 48.28 s | success 31.72 s | success 25.80 s |
| 2081 | success 71.68 s | success 39.08 s | success 42.76 s |
| 2189 | timeout 140 s | success 41.68 s | success 55.04 s |
| 2069 | success 34.88 s | success 26.40 s | success 26.96 s |
| 2168 | success 55.64 s | success 40.68 s | success 35.84 s |
| 2037 | success 82.52 s | success 31.68 s | success 32.80 s |

No collisions in this set. The untuned B/C commands were more variable; B reached
72.11 deg/s² episode peak yaw acceleration. This is not the isolated frozen yaw
step response: repeated command reversals produce different acceleration peaks.

One-parameter coarse search used command-change weights 10 and 30 for B/C, plus
the initial weight 3; B also tested an intermediate weight 6. The tuning score
was 10000 per collision/boundary, 1000 per non-success, plus mean completion time,
2 × reversals, 20 × normalized command TV/s, 0.2 × peak yaw acceleration and
0.1 × planning p99 in ms. Safety is lexicographic in the observed trial range;
the scalar score does not by itself authorize promotion.

| Variant, tuning seeds 2000/2081/2189 | Outcome | Mean time, s | Mean peak yaw acceleration, deg/s² | Mean command TV/s |
|---|---|---:|---:|---:|
| B, weight 10 | 3 success | 47.76 | 57.22 | 0.733 |
| B, weight 30 | 3 success | 74.97 | 57.80 | 0.341 |
| C, weight 10 | 2 success, 1 timeout | 67.87 including timeout | 58.80 | 0.474 |
| C, weight 30 | 2 success, 1 timeout | 77.87 including timeout | 49.85 | 0.315 |

Weight 6 retained large acceleration peaks and slowed seed 2189 to 83.24 s.
Weight 10 was rejected at the independent validation gate: seed 2069 took 77.08 s
and 2168 took 65.48 s. Increasing a soft command-change penalty suppresses useful
maneuvers as well as switching. It is not an adequate solution by itself.

An explicit U-shaped cul-de-sac, sensed through the same LiDAR pipeline, further
distinguished topology: A succeeded in 58.80 s, B with weight 10 in 52.76 s, and C
with weight 3 timed out at 140 s. This is a general geometry stress case, not an
environment recognized or special-cased by control. Direct local optimization
was excluded because its goal-only objective can remain trapped.

The final candidate retains B with weight 3 and adds a convex command-sequence
constraint: |yaw_command[k] - yaw_command[k-1]| <= 0.1 rad/s per 0.12 s knot,
including the first command relative to the last executed command. This is a
navigation command bound (equivalent slope 0.8333 rad/s²), not a change to physical
inertia, actuator response, allocator gain or allowed maximum yaw-rate command.
All candidates and the weighted proposal obey it. This limits sudden command
reversals before they excite the fixed vessel dynamics.

| Seed | Final candidate outcome/time | Reversals | Peak yaw acceleration, deg/s² |
|---|---|---:|---:|
| 2000 | success 32.24 s | 6 | 43.47 |
| 2081 | success 46.48 s | 14 | 47.21 |
| 2189 | success 88.76 s | 29 | 47.66 |
| 2069 | success 29.68 s | 8 | 45.92 |
| 2168 | success 38.92 s | 12 | 45.24 |
| 2037 | success 33.00 s | 8 | 41.69 |

No collision or timeout in these six, but seed 2189 remains inefficient and some
successful maps have more reversals than A. Do not describe all motion metrics as
improved. This frozen candidate proceeds to one disjoint paired 24-map holdout,
2200–2223, before any default GUI replacement. Do not tune on those results.

## Completed validation and decision — 2026-09-27

The saved paired holdout at `data/architecture_rethink/holdout24/results.json`
is complete: all 48 rows for 24 matching maps (seeds 2200–2223), 0.04 s dt,
140 s timeout. Both A and slew-limited B succeeded 24/24 with no collision or
timeout. No already completed episode was rerun. B was faster on all 24 maps,
with smallest improvement 3.24 s at seed 2219 and largest 72.44 s at 2213.

| Metric | Existing A | Slew-limited B |
|---|---:|---:|
| Mean / median completion, simulation s | 58.237 / 47.58 | 34.113 / 31.12 |
| Mean path length, m | 35.996 | 33.041 |
| Mean minimum actual hull clearance, m | 0.143 | 0.174 |
| Mean episode-peak / observed worst yaw acceleration, deg/s² | 44.963 / 62.730 | 42.476 / 47.642 |
| Mean steering reversals | 10.417 | 9.292 |
| Mean normalized yaw-command variation/s | 0.448 | 0.747 |
| Mean headless compute per episode, wall s | 5.615 | 6.966 |

This holdout predates the subsequent terminal-goal quality change. B's lower
completion time does not prove that final variant has the same reliability.
B's command variation and CPU cost regress despite its favorable route and
yaw-acceleration metrics. The controller samples 128 sequences over 40 knots
of three fixed physics steps each (4.8 s horizon) and may physically re-roll
a weighted proposal. This is an MPPI-inspired sampling-based predictive
controller, not a canonical MPPI implementation. Coarse A* is topological
guidance, not the executed trajectory. Online obstacle inputs remain LiDAR
observations and their perception map; truth is used only by the simulator
scan and offline evaluator.

The production goal check ends at center distance <1.4 m irrespective of
heading or yaw. In the holdout A's first-radius-entry mean absolute heading
error / yaw rate were 14.12 deg / 8.44 deg/s; pre-terminal B's were 11.32 deg
/ 3.87 deg/s. B seed 2205 nevertheless entered stern-first, about 177 deg
off its final tangent. The opt-in B variant now freezes the last 2 m coarse
route tangent on entering a 12 m terminal zone and adds normalized endpoint
position, heading, yaw-rate, cross-track and reverse-motion costs to the
exact lagged physical rollout. Its quality completion requires center distance
≤1.0 m, heading error ≤35 deg, |yaw rate| ≤0.12 rad/s, cross-track ≤0.8 m,
|lateral speed| ≤0.35 m/s and forward speed ≥0.05 m/s. No physics parameter,
forced turn, or production A termination was changed.

Post-terminal targeted seeds 2000/2081/2189/2205/2069/2168/2037 all
succeeded without collision at 36.00/43.92/59.04/40.56/27.64/40.00/35.72 s.
Seed 2205's first-entry heading error fell to 19 deg and yaw rate to
0.76 deg/s; its final qualified entry had 15.4 deg error, 4.8 deg/s yaw,
and 0.996 m minimum center miss. Seed 2189 still first entered with 87 deg
error and 13.1 deg/s yaw, needing another 11.64 s to qualify; final error
was 7.9 deg, yaw 1.9 deg/s, center miss 0.135 m. This prevents the most
obvious premature termination, but does not prove universally smooth first
arrival. No disjoint holdout was run after this terminal change.

The generic U-shaped cul-de-sac is the decisive blocker. Final slew-limited
B with terminal quality timed out at 140 s, collision 0, closest center
approach 27.539 m (`data/architecture_rethink/terminal_culdesac/`). Existing
A had succeeded at 58.80 s and earlier B without the slew bound at 52.76 s;
direct C timed out. The perception-only A* route still goes west to the U
opening, then around the upper wall. Around 15 s B's vessel was near
[4.05, 6.87] m and heading southeast while its guide pointed northwest
to [2.65, 9.32] m; it selected forward motion and returned to the interior.
Over 140 s it oscillated near x≈4.7–6.3 m. The terminal cost is inactive
until within 12 m of the goal, so it did not trigger this distant failure.
The short-horizon objective and repeated route-progress reset are plausible
contributors, not a verified root fix. No seed-specific recovery or further
parameter sweep was added.

The opt-in B GUI now publishes its selected physical future prediction every
three steps, separate from the coarse A* route. The renderer clips only the
display copy at the current boat position on every rendered frame, hiding
elapsed prediction segments; the controller retains the full prediction.
The selected prediction immediately replaces the prior completed one.
`data/realtime_perf/final_gui.png` verifies that the displayed future curve
starts at the boat in a captured X11/3D frame. A single still does not prove
subjective smoothness throughout an episode. Production A still displays a
smoothed route rather than a physical prediction; that semantic difference
remains a UI synchronization issue.

Sequential five-second actual X11/3D 1x probes after this display change:

| Metric | B 2081 | B 2000 | B 2006 | A 2081 |
|---|---:|---:|---:|---:|
| Average FPS | 81.26 | 80.09 | 84.61 | 101.13 |
| Median frame, ms | 8.27 | 8.30 | 8.29 | 8.21 |
| p90 / p95, ms | 26.79 / 28.59 | 27.92 / 31.43 | 24.99 / 28.18 | 10.72 / 11.72 |
| p99 / max, ms | 42.12 / 43.51 | 43.18 / 45.87 | 39.20 / 42.48 | 51.25 / 84.00 |
| Frame std dev, ms | 8.79 | 9.09 | 7.96 | 8.10 |
| Frames ≥33 / ≥50 / ≥100 / ≥250 ms | 17 / 0 / 0 / 0 | 16 / 0 / 0 / 0 | 18 / 0 / 0 / 0 | 17 / 7 / 0 / 0 |

On B seed 2081, 249 physics steps, 83 prediction replacements and 17 A*
searches completed in 5.13 wall seconds: approximately 48.5 Hz physics,
16.2 Hz prediction, 3.3 Hz route rebuilding and 81.3 Hz render plus visual
projection. Mean full sampling call was 22.72 ms; rollout call 9.56 ms
(164 calls), A* 14.07 ms (17 calls), LiDAR 0.09 ms/step, occupancy
0.87 ms/control tick and rendering 4.64 ms/frame. The largest 43.5 ms
frame contained 13.4 ms A*, 21.1 ms rollout and 5.7 ms render; 3D wait was
0.1 ms. These are nested stages; do not add rollout/A* to the full plan time.
No GUI-frame CSV writing or intentional display throttling occurred. B
improves short-run p99/max relative to A, but lowers mean FPS and still has
39–46 ms planning frames. The 120 FPS target and stable frame pacing are not
met. The 37 relevant unit/navigation/dynamics tests pass.

B is therefore not promoted. Production Gap state, queued waypoints, smoothed
route and existing controller remain; no legacy control component was removed
from A. A new ticket should establish and fix the general dead-end progress
mechanism, then test hard seeds and a fresh disjoint paired holdout for the
post-terminal source. A separate performance step must remove or schedule the
measured planning-frame CPU burst without reducing visual update frequency,
physics steps or safety checks. No 200-episode run, dynamics/playback change,
commit or push was performed.

Targeted reproduction: `python3 -m unittest test_sampling_navigation
test_goal_guidance test_trajectory_display test_navigation_pipeline
test_vessel_dynamics test_dynamics_tuning test_yaw_control -q`; opt-in GUI:
`python3 -m experiments.gui_compare --mode corridor --config
data/architecture_rethink/slew_candidate.json --speed 1 --seed 2081
--label local_b_2081`. Normal `python3 main.py` still runs A.
