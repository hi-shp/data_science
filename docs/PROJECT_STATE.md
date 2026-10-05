# Main-heavy experiment state

Status: EXPERIMENTAL. This branch is based on `origin/main` at
`757243cc68f25f5d6fbbac4dc955dab02d5f4010`; its physics reference is
`origin/codex` at `60f2cf46500354a3e0690b9b62888e0dd166a9cb`.
The `main` and `codex` branches are not changed by this experiment.

## Objective and implementation

The MAIN LiDAR, GAP selection, waypoint, Bezier, Pure Pursuit, and predictive
safety-guard pipeline remains. The old MAIN force/yaw equations were replaced
by the exact CODEX `vessel_dynamics.py` and `vessel_config.json` source, with a
GAP steer/desired-speed adapter to the CODEX thrust allocator. The guard's
shadow prediction now integrates this same heavy vessel state. The only
retained navigation adaptation is `yaw_command_gain = 5.0`; a 30-seed
development comparison favored it over gains 1.0, 2.5, and 10.0. Waypoint,
lookahead, GAP scoring, Bezier, emergency thresholds, and cruise capability
remain at their MAIN settings. Episode reset clears steering and emergency
state so each seed starts independently.

Physics timestep remains 0.04 s; displayed 1x playback is 2.4 simulation
seconds per wall second. The branch's leaderboard namespace is `main_heavy`
and its benchmarks are explicitly pending. MAIN/CODEX benchmark values have
not been reused or modified.

## Physics parity

`vessel_dynamics.py` and `vessel_config.json` are byte-identical to the CODEX
reference (SHA-256 `25e45a32...` and `22c870be...`). Environment-step tests
match the authoritative integrator for combined forward, turning, coast, and
opposite-turn commands to 1e-11 state tolerance. Identical physical commands
therefore give identical CODEX/MAIN_HEAVY dynamics under the same state.

Full-thrust terminal speed is 2.26247 m/s in both, with 0% source/model
error. The same step response gives t50 0.76 s, t90 1.76 s, coast distance to
10% speed 3.573 m, steady yaw 0.5 rad/s, yaw t90 1.04 s, and steady turning
radius 2.687 m. These are simulation-model values, not vessel measurements.

## Navigation evaluation

Deterministic development seeds 2000-2199 through the actual `main.run()`
physics/navigation loop, replacing rendering and wall pacing only:

| Candidate | Seed subset | Success | Collision | Timeout |
| --- | --- | ---: | ---: | ---: |
| Yaw adapter gain 1.0 | 2030-2059 | 5 | 25 | 0 |
| Gain 2.5 | 2030-2059 | 7 | 23 | 0 |
| Gain 5.0 | 2030-2059 | 11 | 19 | 0 |
| Gain 10.0 | 2030-2059 | 9 | 21 | 0 |
| Gain 5.0, 100 px lookahead | 2030-2059 | 11 | 19 | 0 |
| Retained gain 5.0 | 2000-2199 | 69 | 131 | 0 |

The 100 px lookahead changed which maps succeeded and raised mean steering
reversals among successful runs (59.0 to 63.3); it was not retained. An early
waypoint release and two-segment pursuit prototype also failed on targeted
seeds and were removed. A longer/more eager shadow guard delayed seed 2000's
first collision but produced an unsafe later turn, so it was removed.

For retained gain 5.0, development success is 34.5%. Successful completion
times: mean 50.52 s, median 50.28 s, p95 56.81 s, best 42.68 s, all in
simulation time. Per-seed outcomes, times, path lengths, and map hashes are
preserved in `experiments/results/main_heavy_dev200.jsonl`. Map seed 3000's
hash matched the existing MAIN benchmark map hash `842bd3a267a03294`.

Observed root cause: the old controller retains the first gap until very
close, then the downstream gap/heading can change abruptly. On seed 2000,
the target changed near x=293 px at t=4.84 s while the hull was already
moving at about 1.25 m/s; collision followed near x=344 px at t=5.76 s.
Other collision seeds require individual traces before assigning the same
cause. Representative failures include 2000, 2063, 2075, 2094, 2111, 2174,
and 2189. The complete failure list is in the JSONL file. The disjoint
2200-2299 holdout was not opened because the 200-seed development gate of
192 successes was missed by a wide margin.

## Real X11 fullscreen-3D 4x

Thirty wall seconds, autonomous MAIN loop, real renderer and display flip,
seed 2000 start, visible X11 display `:0`, 4x requested 240 steps/s:

| Metric | Result |
| --- | ---: |
| Actual physics steps/s | 235.94 |
| Simulation seconds/wall second | 9.438 |
| Average GUI FPS | 42.46 |
| Minimum rolling 1-second FPS | 35 |
| Median/p95/p99/max frame ms | 23.89 / 29.88 / 36.69 / 52.64 |
| Frames at least 100 ms | 0 |
| Maximum/final backlog, steps | 9.20 / 0.15 |

The backlog did not keep growing, but the 75-FPS gate failed. Renderer work
alone averaged 17.14 ms/frame in this measurement, above the 13.33 ms total
frame budget for 75 FPS. Perception/GAP work adds further cost. The renderer
and visual quality have not been changed. Full stage timings are saved in
`experiments/results/main_heavy_gui4x.json`.

## Phase 2 GAP handoff experiment — not accepted

All 131 baseline collision seeds were replayed: map hash, outcome, and contact
time matched 131/131. The five seconds before contact are saved in
`data/main_heavy/phase2_collision_diagnosis.jsonl`. A disjoint, rule-based
classification of observed contact contexts is saved in
`data/main_heavy/phase2_collision_classification.json`:

| Context | Seeds |
| --- | ---: |
| Recent waypoint handoff and heading jump/error | 10 |
| At least 90 px from gate with heading error >=0.5 rad | 41 |
| Within 90 px of gate with next gap present | 33 |
| Within 90 px of gate without next gap | 8 |
| Near obstacle without guard rescue | 22 |
| Direct-target mode | 4 |
| Other/ambiguous | 13 |

These are observable signatures, not exclusive causal diagnoses. Wrong-side
steering was not independently confirmed. The next-gap identity changed at
least three times in the five-second window for 79/131 contacts. In seed
2000, the active target jumped at about 4.76 s, only 0.96 s before contact;
the vessel was moving near 1.26 m/s, while measured yaw t90 is 1.04 s.

A stable-next-gap heading blend was tried and removed: it reduced same-seed
2000-2029 success from 12/30 to 9/30 because a temporarily stable but wrong
second gap could steer the vessel the wrong way. Speed-dependent Pure Pursuit
lookahead with the longer guard scored 11/30 and was removed. Extending the
guard's maximum turn hold or appending the next Bezier in its shadow preview
did not improve representative results and was removed. First/second gap
choice and Bezier handoff therefore remain unchanged.

The best collision-count candidate left as uncommitted experimental work
extends the existing safety preview from 0.8 to 2.0 simulation seconds and
gates it by 60 px plus speed times preview duration. It scored 18/30 on
2000-2029. On the complete 2000-2199 development set, with identical map
hashes 200/200:

| Version | Success | Collision | Timeout | Mean successful time | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 69 | 131 | 0 | 50.52 s | 56.81 s |
| Longer guard | 100 | 100 | 0 | 52.65 s | 64.27 s |

It rescued 35 previous collisions but caused four new collisions (2041,
2054, 2071, 2146). It misses the required 192/200 development gate; the
2200-2299 holdout was not opened. No commit or push was made.

Matched real X11 fullscreen-3D 4x profiles (seed 2000, 12 wall seconds)
measured baseline versus longer guard: 235.02 vs 236.65 physics steps/s,
40.60 vs 33.72 average FPS, 33 vs 27 minimum rolling FPS, and zero >=100 ms
frames in both. Safety-guard CPU time increased from 0.192 to 0.839 ms/call.
See `data/main_heavy/phase2_baseline_gui4x12.json` and
`data/main_heavy/phase2_guard2s_gui4x12.json`. This candidate is therefore
not a performance-safe production replacement.

Physics and playback remain untouched: `vessel_dynamics.py` and
`vessel_config.json` retain CODEX SHA-256 prefixes `25e45a32`/`22c870be`,
`dt=0.04`, displayed 1x=2.4, and the previously measured terminal speed
2.26247 m/s. The same 16 dynamics, safety, playback, and leaderboard tests
pass. The next navigation experiment must stabilize second-gap topology and
evaluate yaw/turn feasibility before committing to the first gap. Avoid
longer guard horizons or lookahead changes alone. The 4x renderer issue is
separate and still pending.

## Validation and next action

`test_main_heavy_dynamics`, `test_main_safety_guard`,
`test_main_playback_timing`, `test_leaderboard`, and the portal geometry tests
pass (23 tests). The branch does not meet navigation or 4x GUI acceptance and
must not replace MAIN or CODEX. Do not run holdout until a candidate reaches
192/200 on development.

## Phase 3 portal continuation — experimental, not accepted

The Phase 2 uncommitted diff was saved at
`data/main_heavy/phase2_uncommitted.patch`. The working safety guard is back
to the production 20-step (0.8 s), 60 px configuration. `MAIN_HEAVY_PORTAL=1`
selects an experimental portal prototype; the default path retains the
original GAP midpoint navigation. Default-mode seeds 2000 and 2003 exactly
matched the recorded map hash, outcome, contact/completion time, path length,
and steering reversals.

The prototype keeps GAP topology and computes an orientation-dependent safe
interval using observed gap endpoints, simulator obstacle radii, and the
frozen hull geometry. It chooses a crossing pose and concatenates the
current and downstream Beziers for Pure
Pursuit. A short exit continuation is used when no next gap exists. It now
checks each approach curve against the actual oriented hull with a 10 px
margin, tries other crossing positions and then other detected gaps, and
retains the prior route only while it remains geometrically clear. These
checks are experimental and do not establish dynamic turn reachability.
The latest prototype also marks a portal visited only when the boat center
actually crosses its line inside the safe interval, using a consistent
approach-side normal rather than the old 60 px midpoint proximity rule.
The simulator-radius lookup is not a LiDAR-only perception proof and must be
addressed before any production adoption.

Lifecycle traces for seeds 2000, 2069, 2081, 2094, and 2189 are in
`data/main_heavy/phase3_lifecycle_*.trace.json`. Initial prototype results:

| Seed | Baseline collision time | Initial portal collision time | Path check/fallback | Actual-line handoff |
| --- | ---: | ---: | ---: | ---: |
| 2000 | 5.72 s | 54.92 s | 6.56 s | 6.48 s |
| 2069 | 14.04 s | 10.08 s | 53.08 s | 8.20 s |
| 2081 | 9.40 s | 5.96 s | 6.68 s | 6.68 s |
| 2094 | 44.56 s | 6.48 s | 6.84 s | 6.44 s |
| 2189 | 19.52 s | 6.04 s | 6.16 s | 6.16 s |

All five still collide. The ten paired baseline-success seeds
2003, 2004, 2005, 2008, 2009, 2012, 2016, 2018, 2021, and 2022 retained
two successes under the path-check prototype (eight new collisions), then
zero successes under the actual-line handoff prototype (ten new collisions).
Full rows are in `data/main_heavy/phase3_alternate_targeted5.jsonl`,
`data/main_heavy/phase3_safe_regression10.jsonl`,
`data/main_heavy/phase3_crossing_targeted5.jsonl`, and
`data/main_heavy/phase3_crossing_safe10.jsonl`. These results rule out
promoting the candidate or running development 2000-2199 now.

The primary observed failure is not a single missing GAP candidate. In seed
2000, three 1-to-0 candidate-count transitions still left the active portal
present. Several active portals were instead discarded while both endpoint
IDs remained visible and unvisited because the prototype's heading/interval
recheck returned `narrow_interval`, `no_heading_safe_sample`, or
`smoothed_interval_empty`. Seed 2094 repeatedly lost and reacquired the same
portal between 3.12 and 4.80 s. Seed 2081 kept its portal until contact;
the geometric approach path and the physically delayed vessel response were
not equivalent. The approach path was regenerated on 148-857 physics steps
per initial five-seed run. Maximum pursuit-target jumps were 7.7-45.9 px at
portal-loss events, reaching 111 px elsewhere in seed 2000. The later
oriented-hull path veto often engaged only after inertia made avoidance
unrecoverable; stopping then did not prevent contact. This is a measured
architecture/feasibility gap, not evidence for changing physics.

Do not tune the crossing-point cost or extend the guard horizon as the next
step. First make portal selection and path validity depend on dynamic turn
reachability early enough to reject an infeasible approach before the vessel
commits, and verify a retained route can be followed without stale Bezier
vertices. Keep this behind the experimental switch and repeat the same five
collision seeds plus the ten success seeds. Only if those improve should the
200-seed development set or real 4x GUI performance test be run. The
2200-2299 holdout remains unopened. No commit or push was made.

## Phase 4 dynamics-feasible path experiment — not accepted

The Phase 3 dirty state was preserved in
`data/main_heavy/phase3_before_phase4.patch` and matching portal/test copies.
The original GAP/Bezier/Pure Pursuit remains the default;
`MAIN_HEAVY_DYN_PATH=1` alone enables the experimental selector and disables
the Phase 3 portal switch. Physics, dt=0.04 s, 1x playback=2.4, and the 0.8 s
safety guard were not changed. Default-mode seeds 2000 and 2003 still match
the baseline map hashes and outcomes: collision at 5.72 s and success at
55.96 s, respectively.

`dynamic_path_feasibility.py` previews the actual MAIN follower, thrust
allocation, lagged vessel dynamics, moving buoys, and oriented-hull collision
for candidate Beziers. It checks up to three alternate GAP paths after a
rejection, retains a feasible incumbent, attempts a tangent-continuous splice,
or brakes if none passes. The evaluator records prediction lead and selector
counters. Unit tests cover dynamics parity, buoy motion, and splice tangent.

On the unchanged baseline, the 40-step (1.6 s) moving-buoy shadow detected
all five eventual collisions while the active geometric path was still clear.
Detection leads for seeds 2000, 2069, 2081, 2094, and 2189 were 38, 39, 37,
40, and 38 steps. A frozen-buoy shadow missed seed 2094; modeling its actual
buoy motion removed that diagnostic false negative. After JIT warmup the
diagnostic probe cost was approximately 1.5-1.9 ms/call.

| Seed | Baseline | Phase 4 moving-buoy selector |
| --- | --- | --- |
| 2000 | collision 5.72 s | success 53.28 s |
| 2069 | collision 14.04 s | success 61.92 s |
| 2081 | collision 9.40 s | collision 6.72 s |
| 2094 | collision 44.56 s | collision 44.32 s |
| 2189 | collision 19.52 s | collision 14.68 s |

The initial gate is only 2/5 successes against the required 3/5, and three
contacts occur earlier. Of ten recorded baseline-success seeds (2003, 2004,
2005, 2008, 2009, 2012, 2016, 2018, 2021, 2022), only four remained
successful; 2003, 2004, 2008, 2009, 2016, and 2021 newly collided. This
candidate is not adopted. No development 200, holdout, or GUI performance
acceptance test was run. Multi-candidate online probe cost also needs a
performance gate before any future adoption. Detailed rows are in
`data/main_heavy/phase4_shadow5_motion_h40.jsonl`,
`phase4_initial5_motion_final.jsonl`, and `phase4_safe10_motion.jsonl`.
Selector counters are reset at episode boundaries in the final diagnostic
record; this bookkeeping change left the five outcomes unchanged.

The shadow identifies unsafe incumbent paths more than a second early, but
the current fallback cannot reliably create a recoverable forward route.
Braking after every candidate fails can still leave the hull inside a moving
buoy's future sweep. Next, diagnose candidate generation at the first
rejection using current momentum and moving-obstacle geometry, then repeat
the same five collision plus ten safe seeds before a wider benchmark.
Keep Phase 4 experimental. No commit or push was made.
