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

## Validation and next action

`test_main_heavy_dynamics`, `test_main_safety_guard`,
`test_main_playback_timing`, and `test_leaderboard` pass (16 tests). The branch
does not meet either navigation or 4x GUI acceptance; it must not replace
MAIN or CODEX. Next, redesign the GAP handoff so the downstream gap and heavy
turning feasibility are considered before the first waypoint is reached,
then validate targeted collisions before any new 200-seed run. Separately
profile the unchanged fullscreen-3D render path against its 75-FPS budget.
Only after the development and GUI gates pass should a disjoint holdout and
MAIN_HEAVY fullscreen-3D leaderboard benchmark be run.
