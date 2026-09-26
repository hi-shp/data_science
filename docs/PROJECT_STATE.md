# PROJECT_STATE.md

Last verified: 2026-09-27. This verified state supersedes the provisional numbers in `docs/RESUME_PROMPT.md` and older presentation/report text.

## Displayed-4x throughput and prediction freshness — verified 2026-09-27

Scope: performance and live predicted trajectory only. Production remains
architecture A; the experimental sampling controller B remains opt-in because
its known U-shaped deadlock is not resolved. Frozen yaw inertia 3.8 kg m²,
thrust/drag/damping/lag, physics dt 0.04 s, 3-step control period, LiDAR cadence,
candidate count, prediction horizons and displayed 1x/2x/4x semantics were not
changed. Displayed 4x requests 200 physics steps and 8 simulated seconds per
wall second. No navigation tuning, 200-episode benchmark, commit or push.

Initial real-X11/3D seed-2081 profile: A advanced 953 steps in 5.117 wall s
(186.24 steps/s), drew 24.63 FPS (median/p95/p99/max frame 30.90/93.96/
111.65/114.12 ms; 4 frames >=100 ms). Opt-in B advanced 521 steps in 5.184 s
(100.50 steps/s), drew 13.04 FPS (76.01/105.08/119.75/126.40 ms; 7 frames
>=100 ms). The original probe did not save frame timestamps or backlog, so
initial 1-second rolling FPS and backlog are unmeasured. B's controller averaged
24.16 ms/call; its two physical rollouts averaged 10.32 ms/call, swept-hull
checks accounted for 1495 ms in 5.184 s, and coarse A* averaged 13.16 ms/search.
A's A*/smoothing/whole route rebuild averaged 11.24/17.81/30.03 ms. Render
itself averaged 6.64 ms (A) and 6.14 ms (B), confirming synchronous planning
and rollout work, rather than path drawing, caused the worst frame stalls.

The same fixed-step physical rollout, hull envelope and cost candidates now run
in optional Numba-compiled CPU kernels (`fast_constant_rollout.py` for A and
`experiments/fast_rollout.py` for B); Python reference paths remain for tests
and fallback. Exact corridor-distance prefilter, observed-map clearance and the
same weighted 8-neighbor A* search use compiled kernels in `fast_corridor.py`,
`fast_clearance.py`, `fast_astar.py`. The final smoothing and perception-limited
safety validation remain in force. B's hull check skips expensive geometry only
when a center-distance lower bound proves that obstacle cannot reduce already
known clearance. The B rollout still propagates every candidate and substep.
Repeated random reference/compiled checks match trajectory, clearance,
blocked flags and A* route; seed-2081 physics hashes match the saved
pre-optimization runs for A's first 900 and B's first 500 steps. Reset obstacle sampling now compares
squared scalar distances; 12 saved map hashes match before/after. PNG encoding
at goal was moved to a prestarted lossless worker, removing the 38 ms in-frame
save; reset fell from about 106 ms to about 25 ms. `main.py` uses precise
monotonic wall time, retains unspent fixed-step budget at episode reset, and
does not discard elapsed wall time. Its eight-step/frame catch-up bound is
unchanged. No physics step, observation, candidate, collision check, render or
display update is deliberately omitted.

The prior `control_path` is the safety-validated smoothed route, not a physical
future prediction. A now publishes the selected candidate's physical rollout;
B publishes its selected rollout. The actual travel history remains a separate
white dotted layer; optional raw A* remains a separate debug layer. Every 2D/3D
render uses the newest completed prediction and projects the current vessel
position onto its time-consistent future segments; the source prediction and
control path are never clipped in place. A new controller result replaces the
display source immediately. On the final measured 4x A/2081 run, replacement
was 66.72 Hz, render clipping 99.51 Hz, trajectory age mean/max 8.47/49.98 ms,
and the displayed first-point error was 0 m across 1988 frames with a
prediction. The 3D completed view was mean/p95/max 3.39/3.94/10.82 ms old;
at most two physics steps behind.
The current GUI screenshot is `data/realtime_perf/final_gui.png`.

Final actual X11/3D seed-2081 stress run, displayed 4x, 20 wall seconds:

| Metric | Production A | Opt-in B |
|---|---:|---:|
| Steps / wall time | 4001 / 20.009 s | 4001 / 20.009 s |
| Whole-window actual steps/s | 199.96 | 199.96 |
| Simulation seconds / wall second | 8.00 within one-step sampling error | 8.00 within one-step sampling error |
| Maximum / final backlog, simulated s | 0.069 / 0.016 | 0.066 / 0.004 |
| Average / minimum 1-second rolling FPS | 99.51 / 78 | 112.45 / 102 |
| Median / p90 / p95 / p99 / max frame, ms | 9.48 / 13.20 / 15.43 / 18.89 / 47.57 | 8.63 / 10.21 / 10.79 / 11.96 / 43.18 |
| Frames >=16.7 / 33 / 50 / 100 / 250 ms | 57 / 5 / 0 / 0 / 0 | 4 / 4 / 0 / 0 / 0 |
| Controller / prediction replacement Hz | 66.72 / 66.72 | 66.77 / 66.77 |
| Prediction age mean / max, ms | 8.47 / 49.98 | 8.97 / 42.11 |

Finite-window step counts can fall fractionally below 200/s by less than one
step at a stopping boundary. The backlog returns close to zero instead of
growing; the scheduler is not persistently throughput-limited on these runs.
For the final A run, the interval between first and last completed physics
steps gives 200.04 steps/s. An independent A/2081 repeat gave 100.42 FPS and
rolling minimum 83. A separate first-episode probe reconfirmed the identical
pre-optimization A physics hash at steps 200, 500 and 900; the B probe matched
its saved pre-optimization hashes at steps 200 and 500. Production A/2081 now
averages 0.29 ms A*, 3.17 ms
smoothing, 4.36 ms total route regeneration and 5.41 ms rendering/frame.
B/2081 now averages 0.27 ms
A*, 1.88 ms plan and 0.47 ms/rollout, with 5.31 ms rendering/frame. B/2000
gave 2001 steps/10.005 s, 111.34 FPS, rolling minimum 103, p99/max
12.02/40.73 ms, no >=50 ms frame; B/2006 gave 2001/10.007 s, 111.62 FPS,
rolling minimum 82, p99/max 13.26/75.80 ms, one >=50 ms frame and no >=100 ms
frame. The 75.8 ms frame combined five physics steps, two controller ticks,
18.4 ms observed-map update and 23.0 ms rendering; its backlog cleared by
run end. A/2000 and A/2006 previously recorded rolling minima 80/81 FPS with
no >=50 ms frames; a final-source A/2000 screenshot run measured 1201 steps/
6.005 s, 109.75 FPS and rolling minimum 92.

All 43 selected dynamics, perception, route, safety, predictive-equivalence and
display tests pass. `requirements-realtime.txt` records the measured optional
Numba/Pillow versions; without Numba the reference fallback is correct but the
4x FPS result is not established. Neither planner nor controller has been moved
to another thread or process: the bounded compiled work still precedes 2D
rendering on the main loop. The PNG writer and existing 3D worker are separate.
This machine's three tested seeds meet the sustained-FPS and no-100-ms-hitch
targets; it is not a guarantee against OS contention or untested maps. Next
navigation ticket may address B's U-shaped progress/replanning continuity
without conflating that failure with this performance result.

## Navigation architecture rethink — promotion withheld, 2026-09-27

The approved 3.8 kg m² dynamics, 0.04 s physics step, three-step controller cadence,
and displayed playback mapping are frozen. Production still runs architecture A:
LiDAR/perception → Gap-assisted A* → validated smoothing → predictive command
selection. Candidate B remains opt-in under `experiments/`; no navigation
replacement, commit, push, or 200-episode benchmark was made. The dirty
`leaderboard.json` and all existing navigation work were preserved. See
`NAVIGATION_ARCHITECTURE_REVIEW.md` for the complete comparison and gate decision.

The saved disjoint paired holdout `data/architecture_rethink/holdout24/results.json`
is complete: 24 maps × A/B, identical map hashes and dt, success 24/24 for each,
collision 0, timeout 0. B's mean/median completion times are 34.11/31.12 s,
versus A's 58.24/47.58 s; B is faster on all 24 (smallest gain 3.24 s at seed
2219). Mean path length 33.04 versus 36.00 m, mean minimum ground-truth hull
clearance 0.174 versus 0.143 m, mean episode-peak yaw acceleration 42.48 versus
44.96 deg/s², observed worst 47.64 versus 62.73 deg/s², mean reversal 9.29
versus 10.42. B regresses normalized yaw-command variation (0.747 versus 0.448
per second) and headless compute (6.97 versus 5.62 wall seconds per episode).
This holdout was run before the subsequent terminal-goal quality change; it is
not validation of that changed candidate.

The old episode completion check in `main.py` is position-only (`distance <
1.4 m`). Baseline 24-map first-entry means were 14.12 deg absolute terminal
heading error and 8.44 deg/s absolute yaw rate; pre-terminal B had 11.32 deg
and 3.87 deg/s, but seed 2205 entered stern-first (about 177 deg error). The
opt-in B prototype now chooses a stable final route tangent within 12 m, adds
position/heading/yaw-rate/cross-track/reverse-speed cost on the exact physical
rollout endpoint, and requires a centered forward entry within 1.0 m, heading
35 deg, yaw rate 0.12 rad/s, lateral speed 0.35 m/s. This changes no physics or
default GUI completion rule. Targeted seeds 2000, 2081, 2189, 2205, 2069,
2168, 2037 all succeeded without collision; 2205's first-entry error changed
from about 177 deg stern-first to 19 deg. Seed 2189 still first entered at
87 deg error and required another 11.64 s to achieve quality arrival; this
remains a weak terminal approach. No disjoint holdout has been run for this
terminal variant.

The decisive topology gate fails for final slew-limited B: the same general
U-shaped cul-de-sac solved by production A in 58.80 s and by an earlier,
non-slew B in 52.76 s now times out at 140 s, with minimum goal-center distance
27.54 m and no collision. A* repeatedly finds a valid exit around the upper
wall. From about 5 s onward B moves between x≈4.7–6.3 m near y≈6 m,
alternating forward/reverse without reaching the westward exit. At 15 s its
guide points northwest (about [2.65, 9.32] m), while the vessel is heading
southeast and commanded forward; at 25 s it is back near [5.06, 6.54] m.
The failure begins far outside the 12 m terminal zone, so terminal cost did
not cause it. A short-horizon sampling objective/resetting coarse-route
progress appears insufficient to commit to the exit; this is a diagnosis,
not a verified fix. Do not promote B or change its frozen parameters to mask
the failure.

Candidate visualization now separates the full control prediction from a
per-render visual projection: each actual X11/3D render clips elapsed points
behind the latest vessel state; the selected physical prediction is replaced
every three physics steps, and the coarse route remains a distinct debug
overlay. At displayed 1x the measured rates on seed 2081 were about 49 physics
steps/s, 16.2 prediction replacements/s, 3.3 coarse A* rebuilds/s, and 81.3
renders/s (per-frame display projection at the render rate). The stored X11
screenshot `data/realtime_perf/final_gui.png` shows the future path beginning
at the vessel rather than the passed segment. This opt-in display fix does not
make the default A path a physical prediction; default A still shows its
smoothed route. Do not describe the two as equivalent.

Sequential five-second real-X11/3D candidate runs at displayed 1x: seeds
2081/2000/2006 rendered 81.3/80.1/84.6 FPS; median frame 8.27/8.30/8.29 ms;
p95 28.59/31.43/28.18 ms; p99 42.12/43.18/39.20 ms; maxima
43.51/45.87/42.48 ms. Frames ≥33 ms: 17/16/18; ≥50/100/250 ms: zero in all
three. Same-session A on 2081 rendered 101.1 FPS, p99 51.25 ms, max 84.0 ms,
17 frames ≥33 ms and seven ≥50 ms. Candidate B reduces the measured tail but
lowers average FPS and retains visible planning-frame spikes. On B seed 2081,
mean sampling-controller call 22.72 ms, mean rollout 9.56 ms (typically twice
per call), mean A* search 14.07 ms when invoked, LiDAR 0.09 ms/step, rendering
4.64 ms/frame. The largest 43.5 ms frame includes 13.4 ms A* plus 21.1 ms
rollouts (nested within 36.2 ms planning) and 5.7 ms rendering; 3D wait was
0.1 ms. No display throttling or physics skips were introduced. The main-thread
planner still blocks that render frame and the 120 FPS target is unmet.

The relevant 37 unit/navigation/dynamics tests pass, including perception,
physical rollout equivalence, collision envelope, terminal condition and
visual-only clipping. Next ticket: resolve the measured U-shaped coarse-route
deadlock with a general progress/turn commitment mechanism, then repeat only
targeted topology and hard-seed checks before a fresh disjoint holdout. Separately
reduce planning-frame CPU time or decouple rendering without changing controller
decisions; retain A as the default until both gates pass. The default A visual
route-versus-physical-prediction distinction remains a UI synchronization issue.

## Navigation architecture review — prior milestone, 2026-09-26

The user's latest scope supersedes the previous strict navigation-equivalence FPS
ticket: navigation may change, but approved dynamics and playback stay frozen.
Read `KNOWN_GOOD_DYNAMICS.md` and `NAVIGATION_ARCHITECTURE_REVIEW.md` first.
Starting HEAD is the existing `45e15e0`; only `leaderboard.json` was dirty.
Source/working-tree backup: `data/architecture_rethink/baseline-sgl_aqsn/`.
Production navigation has not yet been replaced. All candidates are isolated in
`experiments/`. No commit/push or 200-episode benchmark.

Fresh dynamics measurement: yaw step t90 1.04 s, peak acceleration 39.7068 deg/s²,
terminal forward speed 2.26247 m/s. All six frozen source/config hashes still match.
The existing 26 tests and five new prototype contracts pass (perception boundary,
fixed-step rollout equivalence, hull envelope, command bounds and command slew).

Three architectures compared on 2000/2081/2189/2069/2168/2037: baseline 5 success,
0 collision, 1 timeout; coarse route + sampling MPC 6 success, 0 collision;
direct sampling MPC 6 success, 0 collision. Untuned MPC commands varied more and
episode peak yaw acceleration reached approximately 72 deg/s². Direct MPC was
rejected after a U-shaped cul-de-sac timeout and a tuning variant's seed-2081
timeout. Coarse guidance and baseline solved the cul-de-sac at 52.76 and 58.80 s.

Command-change cost search used only 2000/2081/2189: weights 3, 10, 30 and a
coarse-to-fine weight 6. Weight 10 passed tuning but regressed seed 2069 to 77.08 s
versus baseline 34.88 s, so it was not promoted. A final bounded-command candidate
uses weight 3 and max yaw-command change 0.1 rad/s per 0.12 s command knot;
physical inertia/allocator gains remain unchanged. Tuning results: 2000 success
32.24 s, 2081 success 46.48 s, 2189 success 88.76 s; peaks 43.47/47.21/47.66 deg/s².
The separate hard-map validation, paired holdout, and GUI measurement were
subsequently completed as documented in the leading 2026-09-27 entry. The
candidate was not promoted. Traces, maps, parameters and timing JSONs are
under `data/architecture_rethink/`; no generated evidence is committed.

## Real-display FPS ticket — measured partial improvement, 2026-09-26

Status: the X11 `main.run()` GUI was measured with the active 3D worker on seed 2081 at displayed 1x/2x/4x/8x/16x. The 120-render-FPS acceptance target is not met, so this is a measured partial performance candidate, not a completed real-time release. Dynamics, dt = 0.04 s, navigation/Gap decisions, LiDAR-only perception boundary and the displayed playback multiplier remain unchanged. No 200-episode run, seed-2189 fix, commit or push was done. The user's other uncommitted work is retained.

`BASE_PLAYBACK_RATE = 2.0`: displayed 1x/2x/4x/8x/16x requests 50/100/200/400/800 physics steps per wall second, with one planning tick every three steps. This differs from the 25–400 step/s explanatory figures in the ticket; the user explicitly chose to retain the existing displayed-speed definition. The main loop accumulates wall-time debt, performs fixed-dt steps before drawing, and previously allowed 64 steps (up to 21 planning ticks) in one frame. On the first 16x X11 run it displayed 2.9 FPS, had two frames above 500 ms, and advanced only about 170 physics steps/s. A single slow frame took 616 ms and included nine route regenerations. At this speed, the machine cannot finish the requested 800 steps/s while rendering at 120 FPS with the current Python planner.

Changes in this ticket:

- `control_path.py`: reject impossible chord candidates using a conservative spatial-cell corridor check before allocating dense line samples; retain the earlier eight-sample check, exact full corridor test, perception-obstacle safety callback and final path safety validation. Build cumulative path geometry once on route replacement and reuse it in continuous lookahead.
- `route_planner.py`: calculate the full clearance grid only on actual route regeneration; precompute invariant A* neighbor distances and goal heuristic; use dense Python cost/parent/visited grids instead of repeated tuple-key dictionary/set access. The A* search now has a separate profiler boundary. All dirty causes remain active: absent path, newly unsafe route, changed accepted Gap, or the existing due/retention rule. No planning tick or same-frame regeneration was skipped. In 900 seed-2081 steps, 58 regenerations comprised 35 unsafe routes, 13 absent paths, seven Gap changes and three expirations. Only two of the 35 unsafe cases lay entirely behind path progress; broad coalescing would change safety decisions.
- `engine_3d.py`: poll the worker without waiting; draw a local copy of the latest completed shared-memory frame, dropping only stale visual requests when the worker is busy. Do not drop simulation state. The default IPC payload omits legacy `current_wp`/`next_wp` beacons; accepted Gap is shown only with the raw-route debug overlay.
- `ui_renderer.py`, `environment.py`, `simulation.py`: stop generating display-only combinations of every Gap candidate; remove 1st/2nd/candidate waypoint markers and their controls from the world view, minimap and camera strip. Show the actual control path and clipped controller target by default; raw A* is optional debug. Cache the path's world-coordinate render geometry until route replacement. Internal `current_wp` and `next_wp` remain because passage selection and queued promotion use them. The former weight-breakdown panel is no longer drawn; layout consolidation remains a UI follow-up.
- `main.py`: cap catch-up work at eight fixed physics steps per rendered frame instead of 64. Pending accumulator debt is retained, so no physics step is discarded or merged; when the requested speed exceeds CPU capacity, real playback falls behind the selected multiplier. This is a responsiveness/throughput tradeoff, not a claim that 16x now runs at 16x wall speed.

Five-second focused X11/3D measurements on the same machine and seed (speed-specific runs progress to different simulation positions as throughput changes):

| Display speed | Before FPS | Current FPS | Current p95/p99/max frame, ms | Current frames >=100/250/500 ms | Current simulation seconds / wall second (requested) |
|---|---:|---:|---:|---:|---:|
| 1x | 68.6 | 101.9 | 11.8 / 51.6 / 92.9 | 0 / 0 / 0 | 1.96 (2) |
| 2x | 33.9 | 69.1 | 51.9 / 78.7 / 91.5 | 0 / 0 / 0 | 3.90 (4) |
| 4x | 4.8 | 24.6 | 95.1 / 117.5 / 122.1 | 5 / 0 / 0 | 7.40 (8) |
| 8x | 2.9 | 22.8 | 98.0 / 123.9 / 136.9 | 6 / 0 / 0 | 7.11 (16) |
| 16x | 2.9 | 24.0 | 86.1 / 123.1 / 134.6 | 5 / 0 / 0 | 7.49 (32) |

The initial 16x run had p95/p99/max 579.4/609.0/616.4 ms, 15/11/2 frames >=100/250/500 ms, and 902 physics steps in 5.29 s. The current 16x run completed 958 steps in 5.11 s, about 187 step/s against the required 800. It processed 320 planning ticks and 60 full A* + smoothing rebuilds. Per rebuild, measured A* search 11.48 ms, smoothing 17.59 ms, whole regeneration 30.05 ms. At 1x the same means were 14.15/28.14/42.98 ms, down from the initial whole-route/smoothing 58.03/36.13 ms. Main render time fell from about 10.23 to 4.65 ms at 1x; 3D response wait fell from 4.28 to 0.035 ms per frame. At 16x, route regeneration totaled 1.80 s of the 5.11 s sample, non-route predictive rollout about 1.12 s, occupancy/clustering about 0.74 s, and rendering about 0.82 s. These nested timer totals must not be added without subtracting included child calls. CPU work, especially repeated legitimate replan + smoothing, remains the limit; rendering alone is not the remaining bottleneck.

Correctness: the 26 navigation/dynamics tests pass. The 35-route smoothing audit compared every resulting path with the pre-prefilter implementation: 10,415 shortcut candidates, 9,685 early rejections, 730 full-validation candidates, 169 selected chords, zero false rejections. Seed 2081 and 2000 first 900-step hashes match the pre-ticket state exactly for vessel physics, raw route, control path, selected Gap, lookahead, command and target heading. Seed 2081's full-episode summary and trajectory hash also match the saved pre-ticket run: success at 71.68 simulation seconds. Actual GUI first-200-step hashes for physics, raw route, control path, selected Gap, pursuit and command match across all five display speeds. The paired 24-map smoke (seeds 2000–2023) remains 24/24 success, collision 0, timeout 0, mean completion 55.5317 simulation seconds; its maps and trajectory CSVs are byte-identical to the preceding integrated candidate (episode CSV differs only in `compute_s`). Seed 2189 remains a 140.0 s timeout with the same saved motion metrics. The selected GUI screenshot confirms the legacy markers are absent and the controller path/target remain visible; the old dashboard layout was not redesigned. The old angle-view panel still has cramped/overlapping top text, and the former weight panel is blank; handle those only in a separate UI layout ticket.

Evidence (ignored local files): `data/realtime_perf/gui_probe.py`, `equiv_probe.py`, `baseline_*x.json`, `cap8_*x.json`, `eq_*x.json`, `final_gui.png`, `smoke24/`, and `data/fps_ticket/prefilter_audit.json`.

Remaining blocker and next ticket: 120 FPS is not achieved even at 1x, and requested 8x/16x wall playback cannot be sustained. The eight-step limit removes 250/500 ms stalls in the sampled runs but leaves 100–140 ms replan frames and slows actual playback under load. Do not claim this as a completed 120-FPS ticket or move directly to seed-2189 navigation tuning on the assumption that GUI performance is solved. Next, assess a portable compiled or incremental A*/smoothing implementation and a render/physics scheduling design against strict route/command/trajectory equivalence, with actual achieved step/s recorded. If that cost or semantic risk is unacceptable, explicitly accept a lower render/speed target before resuming the seed-2189 and replanning-continuity ticket.

## Residual FPS and navigation integration ticket — verified, 2026-09-26

Status: one automatic navigation pipeline is retained. The accepted 3.8 kg·m² yaw inertia, all thrust/drag/damping/speed settings, 0.04 s physics timestep, 1x/2x/4x multipliers and three-step control interval remain fixed. No 200-episode run, seed-specific recovery, commit or push. Existing uncommitted user and architecture work was preserved. The navigation candidate still times out on seed 2189 and is not a final holdout release.

Residual stall cause: actual `main.run()` used a wall-time accumulator. Before this ticket, its initial 250-step diagnostic had a 444 ms slow frame with 12 physics steps and three route rebuilds. A focused replay on the current pre-integration code measured a 467 ms frame: 12 steps, four perception-map updates, seven Gap calls, three A* route regenerations, three smoothing calls, 90 final obstacle-safety callback calls, 439 ms inside route regeneration (148 ms smoothing), 11 ms total rendering and 7 ms 3D receive wait. The expensive remainder was mostly Python A* edge scoring: it constructed small NumPy vectors and recomputed the same corridor and cell penalties for each explored edge. The GUI clock also counted environment/3D initialization time as playback debt, provoking a startup catch-up burst. Rendering and 3D blocking were minor. Each `build_route` call includes one A* search; its exact A* subroutine has no separate timer. No planning calls were coalesced or skipped, because doing so could change simulation-time decisions.

The minimal FPS changes are: precompute the destination-cell traversal penalty once per A* build in `route_planner.py` (same unknown-space and Gap-corridor terms); rebase `main.py`'s existing 120 FPS clock after environment initialization, before accumulating playback time. Playback scale and dt are unchanged. The previously validated smoothing prefilter and full safety checks remain in `control_path.py`. The display-only legacy Bezier preview work was then removed during navigation integration.

Final focused GUI timing uses seed 2081, displayed 1x `main.run()`, first 500 physics steps, SDL dummy with actual 2D rendering and active 3D worker. Both variants use the final clock rebase and integrated pipeline; the before variant substitutes only the saved pre-optimization A* builder at runtime. Thus these are paired measurements of the A* change, not measurements of a physical display:

| Frame metric | Previous A* builder | Final A* builder |
|---|---:|---:|
| Rendered frames | 673 | 725 |
| p95, ms | 60.76 | 51.09 |
| p99, ms | 87.73 | 78.13 |
| Maximum, ms | 350.79 | 138.91 |
| Frames >=100 / >=250 / >=500 ms | 5 / 1 / 0 | 2 / 0 / 0 |

The slowest paired frame contained six physics steps and two route regenerations: route time 332.4→120.4 ms, while smoothing stayed 69.4→70.3 ms. Render time was about 10 ms, and 3D wait about 6 ms. The first 500 physics-step traces match exactly for state, raw/smoothed route, selected Gap and commands. A separate pre/post startup-clock comparison also matched its first 250 physics-step trace; its worst frame fell from 254.9 to approximately 139 ms in the final paired run, although absolute tail measurements vary by replay. Short replays do not guarantee that no later GUI stall exists. Two 100–250 ms frames remain in this 500-step sample; no threading, planner-frequency change, safety relaxation or resolution reduction was introduced.

Current automatic pipeline:
`LiDAR returns → observed NavigationMap + hit-grid clustering → persistent passage candidate and optional accepted Gap corridor → perception-map A* raw_route → validated tangent-continuous control_path → arc-length lookahead → hull-checked predictive controller`.

| Element | Input → output; command role | Final status / duplication / UI role |
|---|---|---|
| Hit-grid clustering (historically called DBSCAN) | Scan hit memory → tracked cluster centers/IDs → Gap search | Retained; it is connected-component clustering, not sklearn DBSCAN |
| Gap candidates and `current_wp` (old 1st waypoint) | Clusters, perceived obstacles, visited pairs → best persistent passage candidate | Retained; `route_target` accepts it only when perceived-safe and goal-progressing; not itself a controller target |
| `selected_gap` | Candidate safety gate → optional A* corridor penalty and replan key | Retained, actual accepted passage; 2D/3D active marker now reads this state |
| `next_wp` (old 2nd waypoint) | Downstream Gap search → queued candidate | Retained because it can be promoted to `current_wp`; never a direct controller target or A* endpoint |
| Old ranked candidate waypoints | Sorted Gap list → visual-only two alternates | Removed calculation, state, and overlay; the existing checkbox now shows actual raw A* route |
| Legacy current/next Bezier preview | Gap/goal → visual paths subsequently overwritten by route planning | Removed from simulation step, environment state and UI; not the active smoothing algorithm |
| Direct-target Bezier feasibility | Perceived obstacles and destination → yes/no direct-clear decision | Retained because it changes Gap state; it is a safety/decision check, not a control path |
| Legacy Pure Pursuit | Preview Bezier samples → discrete point overwritten by route planner | Removed; no controller/fallback role |
| A* and `raw_route` | Observed map, optional Gap cost → grid route | Retained as one raw planning output and optional debug overlay; old duplicate `safe_route` removed |
| Current smoothing Bezier fillets and `control_path` | Raw route + perception-only safety validator → continuous curve | Retained as the sole path followed and drawn; full validation unchanged |
| Continuous lookahead / `pursuit_target` | Control path + monotonic arc progress → interpolated target | Retained; this is a preview target in pixels for diagnostics |
| Predictive controller / `controller_target` | Lookahead plus wall clip, dynamics and perceived hull clearance → speed/yaw commands | Final command authority; 2D target marker reads the actual clipped target |
| `all_gaps`, queued marker, perception overlay | Scan/cluster and queued passage state → UI/debug only | Retained as explicitly non-authoritative overlays; no ground truth re-entered planning |

`current_wp` and `next_wp` have observed state-transition roles: on the fixed maps 2000 / 2081 / 2189, the queued Gap was promoted once in each episode. Accepted `selected_gap` was active for 162 / 108 / 1008 physics steps respectively. A unit test confirms that changing a valid Gap changes the actual A* route, smoothed curve and lookahead target. This supports retaining Gap as a corridor preference while allowing A* to proceed when no passage is safe. The line-tracing and manual controls remain explicit separate user modes, not competing automatic navigation branches.

Validation: syntax checks and all 26 navigation/dynamics unit tests pass. Seed 2081 full episode succeeds at 71.68 simulation seconds. Seeds 2000 / 2069 / 2037 / 2083 / 2114 succeed at 48.28 / 34.88 / 82.52 / 27.80 / 36.72 s. Seed 2189 still times out at 140 s. The six hard-set trajectory CSVs match the pre-integration candidate byte-for-byte. Final 24-seed smoke (2000–2023): 24 success, 0 collision, 0 boundary failure, 0 timeout; mean success time 55.53167 s, mean yaw reversals 9.375, mean episode peak yaw acceleration 42.98286 deg/s². Its trajectory CSV matches the prior navigation candidate byte-for-byte. A later `main.py` change rebases only the GUI clock and was verified with a 500-step GUI physics trace; the headless smoke evaluator does not call `main.run()`. This is selected evidence, not a 200-map validation.

Remaining UI synchronization: rename remaining WP1/WP2 and weight-breakdown labels to distinguish accepted passage from queued/candidate scoring; make the perception map's known/unknown status explicit; optionally show raw route and actual clipped controller target together. The basic existing layout and colors were not redesigned. Next ticket: diagnose seed 2189 timeout and route-replacement heading continuity. If rare 100–250 ms frame stalls remain perceptible on the user's machine, investigate them in a separate measured GUI ticket; do not silently change planning frequency or physics to mask them.

Changed production files in this ticket: `route_planner.py`, `main.py`, `simulation.py`, `navigation.py`, `utils.py`, `environment.py`, `ui_renderer.py`, `engine_3d.py`, `boat_control.py`. The previous FPS prefilter in `control_path.py` is retained, not newly altered. Evidence (ignored local artifacts): `data/fps_ticket/residual_probe.py`, `route_planner_before_residual.py`, `residual_probe_before_2081.json`, `residual_probe_after_2081.json`, `integrated_hard_results.json`, `integrated_smoke24_final/`, `passage_audit.json`.

## FPS stall optimization ticket — validated candidate, 2026-09-26

The current working-tree candidate retains the navigation architecture and all user changes. The only production change made for this ticket is in the untracked `control_path.py`: it precomputes raw-route segment geometry and rejects a shortcut early when up to eight of its existing samples already violate the 0.40 m route corridor. Candidates that pass still undergo the original full obstacle/perception safety callback and full sampled-corridor check. The final smoothed-path safety check is retained. No dynamics, playback, Gap/A* architecture, UI, or seed-specific behavior changed. The exact prior version is saved as `data/fps_ticket/control_path_before.py` (ignored local artifact).

Verification: `python3 -m unittest test_navigation_pipeline test_vessel_dynamics test_dynamics_tuning test_yaw_control -v` passed 26/26 on the current tree. This covers smoothing geometry and safety, observed-map/perception boundary, A* routing, continuous lookahead, collision prediction, and dynamics checks. Full seed 2081 episode before/after: success at 71.68 simulation seconds, 0 collision, 0 timeout, 1,792 physics steps; all-step hashes for physics state, selected Gap, raw route, smoothed path, continuous target, target heading and control command match exactly. The recorded full trajectory also matches. Seed 2000 succeeds at 48.28 s; seeds 2069, 2037, 2083, 2114 succeed at 34.88, 82.52, 27.80, 36.72 s respectively. Seed 2189 remains the known navigation-candidate timeout at 140 s; it is not addressed here. For all seven seeds (2081, 2000, 2189 and four hard seeds), before/after trajectory CSV files match byte-for-byte. This is a selected regression set, not a 200-map validation.

Same seed 2081, 500 rendered frames / 20 simulation seconds, one physics step per frame, SDL dummy with real 2D drawing and active 3D worker:

| Metric | Before | Prefilter candidate |
|---|---:|---:|
| Mean frame, ms | 24.661 | 17.355 |
| Frame p95 / p99, ms | 162.709 / 255.942 | 71.007 / 101.924 |
| Maximum frame, ms | 320.036 | 191.819 |
| Frames >=100 ms / >=500 ms | 32 / 0 | 6 / 0 |
| Route regeneration, mean per call, ms | 159.916 | 69.586 |
| Smoothing, mean per call, ms | 123.792 | 33.154 |
| Route regeneration minus smoothing, ms | 36.124 | 36.432 |
| Obstacle safety callback calls / total, ms | 11,647 / 1,838 | 1,276 / 131 |
| Rendering, mean per frame, ms | 9.281 | 8.996 |

The route-minus-smoothing row includes A* and other route-generation work; A* was not isolated by its own timer, so it must not be presented as a pure A* measurement. Disabling path drawing left the >=100 ms count at 32; identical-input smoothed-path caching also left it at 32 because almost every raw route differed. Those prior diagnostics rule out drawing and simple repeated-input caching as the main fix.

The actual `main.run()` displayed-1x paced replay, same seed and first 250 identical physics steps, improved frame p95 34.456→18.069 ms, p99 369.425→86.249 ms, maximum 635.735→444.410 ms, >=100 ms stalls 9→3, and >=500 ms stalls 2→0. The before runner completed seven extra physics steps in its final render iteration (257 versus 250), so frame-level distributions are diagnostic rather than perfectly paired. Remaining 100–500 ms stalls occur when the playback accumulator groups multiple physics steps and route rebuilds in one rendered frame; the worst after frame grouped 12 steps and three rebuilds. The user's playback definition was not modified. SDL dummy profiling does not establish physical-display FPS.

Prefilter audit on 35 recorded representative routes: 10,415 shortcut candidates; 9,685 (92.99%) rejected before full validation; 730 proceeded to full validation; 169 passed and were selected. Replaying the original full validator on every candidate found zero false rejections; all 35 resulting control paths match exactly. More generally, an early rejected probe is an unchanged member of the final candidate sample set, so it cannot pass the original full-corridor predicate. The safety callback and perception boundary are unchanged.

The candidate meets the requested equivalence and >500 ms stall reduction checks, though isolated 100–500 ms GUI stalls remain. No second optimization is justified within this ticket because the residual combines route rebuilds and fixed playback catch-up; monitor it in a separate measured performance ticket if it remains disruptive. Next navigation ticket: diagnose seed 2189 timeout and route-replacement heading continuity with this validated performance change retained. Do not infer that seed 2189 was fixed or that a 200-episode holdout was run. No commit or push.

Evidence: `data/fps_ticket/unit.log`, `before_A_2081_summary.json`, `A_2081_summary.json`, `paced_compare_2081.json`, `full_equivalence_2081.json`, `prefilter_audit.json`, plus `data/yaw_playback/nav_corridor/`, `fps_after/`, and `fps_hard/` (ignored local artifacts).

## Navigation architecture ticket — implemented candidate, 2026-09-26

Status: perception boundary, Gap corridor integration, safe corner fillets and continuous lookahead implemented and tested. This is not a claim of performance equivalence to the prior navigation: seed 2189 regressed to timeout, and route-replacement heading discontinuities remain. Current code contains the candidate described here, not the old navigation. The next ticket should address this regression before final UI synchronization or adoption as a fully validated navigation release. User-frozen 3.8 dynamics, all thrust/drag/damping settings and playback are byte-for-byte unchanged. No 200-episode run, commit, push, UI layout or presentation changes.

### Audit: information boundary before this ticket

A = actual scan returns; B = processed or remembered observations; C = simulator ground truth.

| Stage / function | Previous input and actual role | Current candidate input / role |
|---|---|---|
| `perception.lidar_hits_np` | C is used to simulate 180 nearest ray returns, range 320 px = 6.4 m; this sensor use is allowed | Same sensor; distances are explicitly clipped to max range |
| `perception.update_grid`, `extract_clusters_from_grid` | A hit cells accumulate, decay by 0.945 per 0.04 s; weighted connected components feed Gap. Despite DBSCAN terminology, this is grid connected-component clustering, not a sklearn DBSCAN call | Unchanged clustering parameters and hit-grid semantics; still feeds Gap |
| Existing occupancy grid | B occupied-hit evidence only, no ray-cleared free/unknown distinction; not an A* input | Retained for clustering/Gap, not misrepresented as free-space evidence |
| `route_planner.route_target` | Reads C centers/radii for objects whose centers are within 6.4 m, ignoring ray occlusion. Remembers those objects indefinitely. Previously unseen centers outside range do not directly enter A*, but hidden in-range objects do. Unknown cells have no extra cost | Reads only `NavigationMap`; estimated geometry and finite observation memory; unknown cells explicitly cost more |
| `boat_control.select_command` | C filtered to range + 20 px = 6.8 m; can use never-observed geometry outside LiDAR range and behind occluders | Scan-derived perceived obstacles and memory only; full existing predictive hull collision envelope retained |
| `simulation.advance` → `find_gap`, `is_direct_target_safe`, `make_bezier_path` | Entire C list passed to Gap scoring, direct-path/Bezier checks and curve deformation | Shared `perceived_obstacles`, derived only from scans |
| `environment.validate_wp_obstacle_5x5` | Entire C list can invalidate waypoint | Same perceived obstacle snapshot as planner |
| New smoothing validation | Did not exist for A* path | Same perception map and surveyed course boundary, no future world objects |
| `environment.collide`, evaluator | C used to evaluate actual collisions / clearance | Preserved; explicitly separate from navigation inputs |

A direct reproduction of the legacy leak placed B's nearest surface at 6.9 m, beyond the 6.4 m LiDAR range. Scan arrays were identical with/without B, but the legacy direct-path safety answer changed true → false and Bezier points changed by up to 1.01635 m when handed the whole world list. Evidence: `data/navigation_architecture/legacy_leak.json`.

The current map accepts only position, heading, angles, measured ranges and simulation time. It fits circles to visible surface points when fit residual/radius checks pass; otherwise it uses small hit discs. These are geometric estimates, not object IDs or hidden simulator radii. Occupied memory TTL is 2.2 s, comparable to the existing hit-grid decay from 20 below 1 in about 2.12 s. Inferred free-cell evidence expires after 5 s. Unknown cells remain traversable with explicit +1.5 cost, rather than becoming ground-truth free cells. Map dimensions/course boundary and destination are assumed surveyed prior information; unknown buoys are not. Plans through unknown space remain provisional, not proven collision-free.

### Roles and architecture decision

| Element | Input → output | Control role / overwrite / necessity |
|---|---|---|
| DBSCAN-named clustering | Accumulated hit grid → tracked cluster centers/IDs | Necessary to current Gap selector; algorithm is weighted connected components |
| Gap detection | Clusters, perceived obstacles, hit grid, goal alignment → scored candidates | Retained; width/clearance/alignment select a candidate corridor |
| 1st waypoint | Selected cluster pair → moving gap midpoint | Was mostly bypassed by successful A*. Now a candidate corridor anchor; A* only accepts it when perceived-safe, goal-progressing and within sensor range |
| 2nd waypoint | Downstream cluster-pair search → candidate next gap | Does not directly command controller or set A* endpoint. Can be promoted to 1st after passage; predictive/queued role |
| Legacy Bezier | Current/next gap and perceived obstacles → curves | Previously fallback control when A* failed. New control no longer falls back to these curves. Direct-route feasibility still uses Bezier; next-gap curve is preview only |
| Legacy `pure_pursuit` | Legacy Bezier samples → discrete point | Intermediate/preview result overwritten; not the steering law. Candidate uses `control_path.lookahead` instead |
| Occupancy | Old hit grid plus new observed free/occupied/unknown map | Old grid supports clustering; new map supports A*, smoothing and common obstacle estimates |
| A* | Perception map, global destination, optional gap corridor cost → raw path | Retained search algorithm. Gap center is not its endpoint; it influences its preferred corridor cost |
| Route planner | Raw path → validated smooth path + continuous target | Single authoritative path interface to predictive control |
| Predictive controller | Continuous target, current physical state, perceived obstacles → speed/yaw commands | Final authority with unchanged lagged dynamics/hull rollout; missing safe curve uses existing checked reverse recovery |

| Option | Change / stability / deadlock | Perception, smoothness, cost, explanation and project identity |
|---|---|---|
| A: A*-centric, Gap debug-only | Smallest role cleanup; retains dependence on global-route deadlocks | Can be LiDAR-only and smooth; lowest redundant computation, clearest single planner, but removes the project's Gap decision idea |
| B: Gap + A* | Moderate interface changes; compulsory gap waypoints risk trapping, so use a safe corridor preference | LiDAR-only is achievable; smoothness added after search; extra map/gap computation; clear division between corridor selection, search and motion; preserves meaningful Gap contribution |
| C: Gap/Bezier restoration | Largest control-behavior change relative to the frozen baseline; loses current graph-search escape capability | LiDAR-only is possible but does not itself guarantee safe curves or low deadlock rate; less search cost, familiar project explanation; weakest evidence for restored avoidance stability |

Option comparison is structural engineering judgment; A and C were not independently benchmarked in this ticket.

Implemented Option B corridor variant:
`LiDAR → hit-grid clustering + NavigationMap → Gap candidates → optional safe selected corridor → perception-map A* → validated corner fillets → continuous arc-length lookahead → predictive controller`.

Gap is not forced when infeasible; a unit test verifies that changing a valid gap changes raw route, smoothed control path and pursuit target. The initial simplifier could erase corridor preference with a clear straight shortcut; the final candidate constrains curve/shortcut deviation to 0.40 m from the chosen raw route.

### Path geometry and continuity

- `raw_route` is retained separately from `control_path`. The old `bezier_path` field is only a compatibility adapter for the actual control curve display after planning. `selected_gap` is the accepted corridor; `local_goal` denotes that corridor anchor (or destination), not a mandatory A* endpoint.
- Greedy validated chord simplification, followed by local quadratic Bezier fillets. Fillet endpoints are tangent to adjoining straight segments: G1 / arc-length tangent continuity. Shrink only an unsafe corner, up to 12 halvings; reject an unvalidated curve. Dense samples use an extra 0.025 m clearance reserve. This is not a curvature/acceleration guarantee.
- Smoothing uses the same inherited center-path inflation (0.54 m + 0.20 m margin), not a full-orientation hull proof. The independent predictive controller retains its full two-capsule hull envelope and actual pose dynamics. Do not equate center-path clearance with actual hull clearance.
- Projection onto path segments → monotonic path progress → +3 m arc length → interpolated point. No discrete vertex target selection. Existing controller wall-margin clipping remains a separate safety guard and can move the final target off the displayed curve near a wall; record for UI synchronization rather than silently remove it.
- Replanning checks every 0.96 s, validates against current perception each control plan, and retains an unchanged safe corridor path up to 2.88 s. Newly unsafe paths or changed gap choice can replace it immediately. This reduces unnecessary replacement but is not a guaranteed continuous splice between routes. Large heading jumps remain and must be measured separately from within-path lookahead continuity.
- Actual vessel position replaces the quantized first raw node. Without this correction, snapping could invent an initial clearance violation. Missing curves must not return before the existing hull-checked recovery is evaluated; an explicit regression test covers this.

### Final candidate validation — no 200 run

`OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest test_navigation_pipeline test_vessel_dynamics test_dynamics_tuning test_yaw_control -v`: 26 passed. Tests cover unseen and occluded obstacles, route change after perception, controller independence from a hidden world list, memory expiration/unknown cells, corner clearance/tangent changes, continuous lookahead, real Gap influence, reset, recovery and all prior dynamics/playback tests.

Synthetic A-in-range/B-out-of-range test: identical scans, raw path, smooth path, target and controller commands before B is visible. B changes the route after entering measured range. A fully occluded B within nominal range also has no effect. These are deterministic ideal-sensor tests, not real-LiDAR robustness evidence.

| Representative seed | Frozen-navigation baseline | Candidate |
|---|---:|---:|
| 2081 success time, s | 98.44 | 71.68 |
| 2081 yaw reversals | 17 | 12 |
| 2081 max target-heading jump, deg | 177.868 | 151.361 |
| 2081 max normalized command jump | 1.40 | 1.30 |
| 2081 actual sampled hull clearance min, m | 0.12882 | 0.21369 |
| 2081 travelled path length, m | 37.68382 | 40.48419 |
| 2000 success time, s | 73.40 | 48.28 |
| 2000 yaw reversals | 18 | 7 |
| 2000 max target-heading jump, deg | 100.295 | 125.163 |
| 2000 max normalized command jump | 1.10 | 1.10 |
| 2000 actual sampled hull clearance min, m | 0.18523 | 0.19448 |
| 2000 travelled path length, m | 40.04763 | 36.49508 |

Candidate minimum perceived center-inflation clearance at control planning instants: 2081 = 0.20584 m, 2000 = 0.17346 m. This is a different metric from actual polygon hull clearance; the latter is evaluated offline against ground truth at 0.04 s samples. Sampled minima are not continuous-time guarantees. Largest heading jumps occurred across route replacement, even with a valid path before/after, not merely crossing a sample index.

Additional hard seeds: 2069 success 34.88 s; 2037 success 82.52 s; 2083 success 27.80 s; 2006 success 54.16 s; 2114 success 36.72 s; 2189 timeout 140 s (previously success 89.16 s). Total representative/hard group: 7 success, 0 collision/boundary, 1 timeout out of 8. Seed 2037 also slowed from 66.48 s. Do not hide these regressions behind average improvements. The exact final 2189 stall mechanism has not been isolated; candidate-1/2 diagnostic logs are not final-candidate causal proof.

| Paired smoke, seeds 2000–2023 | Previous frozen navigation | Current candidate |
|---|---:|---:|
| Success / collision / timeout | 24 / 0 / 0 | 24 / 0 / 0 |
| Mean success simulation time, s | 65.09000 | 55.53167 |
| Mean episode peak yaw acceleration, deg/s² | 47.41584 | 42.98286 |
| Mean yaw reversals | 12.250 | 9.375 |
| Mean normalized command variation / s | 0.43446 | 0.43777 |

Smoke time improved 14.68%; command variation did not improve. Per-seed success/time improvements do not establish universal stability or real-vessel performance. The final candidate was run on one 24-episode smoke, not 200 episodes. Combined unique tested maps = 30, successes = 29, timeout = 1, collisions/boundary failures = 0; the eight-seed set overlaps smoke on 2000 and 2006.

### UI / legacy follow-up and evidence

Do not present current `current_wp`/`next_wp` as guaranteed controller targets: display must distinguish candidate gap, actually accepted corridor, queued downstream gap, raw route, actual control curve, arc-length target and controller wall-clipped target. Label unknown/free/occupied memory and visibility honestly; simulator obstacles may remain a clearly marked world/debug layer. Legacy next-Bezier/next-Pure-Pursuit previews, discrete Pure Pursuit output, and the unused `observed_buoys` dictionary are cleanup/debug-only candidates. First/second gap state is not wholly dead because corridor selection and promotion now affect planning. Direct-target Bezier feasibility still influences Gap state and must not be removed as allegedly unused.

Remaining issues: seed 2189 regression; large route-replacement heading jumps (including worse maximum on 2000); finite memory/circle-fit perception assumptions; redundant legacy curve computation; unknown-space exploration without a formal visibility-based stopping guarantee; center-path versus full-hull safety distinction; computation cost above the old planner. Current code validates the whole retained path, including already traversed portions, so stale evidence behind the vessel can cause avoidable replans; this is a follow-up hypothesis, not a measured root cause of 2189. No physics or blind perception relaxation should be used to conceal these issues.

Next recommended ticket: diagnose 2189 and route-replacement continuity using the frozen perception boundary/dynamics, then repeat only affected small tests before any UI synchronization. Do not start another dynamics sweep or 200-map run without a new request.

Changed this ticket: `navigation_map.py`, `control_path.py`, `route_planner.py`, `boat_control.py`, `simulation.py`, `perception.py`, `environment.py`, `test_navigation_pipeline.py`, `test_vessel_dynamics.py`, this document. Existing unrelated local edits were preserved.

Evidence: `data/navigation_architecture/baseline/` (pre-ticket snapshots), `legacy_leak.json`, `final_unit.log`, `final_diagnostics/`, `paired_diagnostics.json`, `paired_trajectories.png`, `smoke24/`, `smoke_comparison.json`; final eight-seed results in `data/yaw_playback/nav_corridor/`. Candidate-1/2 artifacts are historical rejected experiments. Source/config manifests preserve the exact candidate; all data artifacts are ignored local files. Trajectory chart was rendered and visually checked. No commit/push.

## Current frozen dynamics — user acceptance, 2026-09-25

This section supersedes all historical adoption statements below. The user directly ran yaw inertia 3.8 kg·m² and accepted the turning response, forward speed and default 1x playback. Freeze this configuration. Do not perform further inertia sweeps, forward-model changes, playback changes or a 200-episode validation under the current request.

Runtime settings: yaw inertia 3.8 kg·m²; explicit yaw-rate torque feedback gain 11.76923076923077 N·m/(rad/s); yaw_response_s 0.65 remains the legacy/reference field, but the explicit gain determines active feedback. Mass 20 kg, yaw drag 8 / 5, lateral drag 55 / 45, motor lag 0.25 s, motor maximum 25 N each, arm 0.22 m, maximum differential moment 11 N·m, forward drag 6 / 7.116009950310329, cruise reference 1.5 m/s are unchanged from the previous accepted configuration. Full-thrust terminal speed remains approximately 2.262469 m/s.

Completed evidence before the freeze:

- Inertia-only tests exposed controller coupling: reducing inertia also reduced feedback torque via I / response_time. Explicit torque gain separates physical inertia from controller authority. No integrator, damping, thrust or forward-speed changes were needed for this yaw ticket.
- At dt 0.04 s, fixed-gain inertia candidates 7.65 / 5.0 / 4.0 / 3.8 had yaw-rate t90 of 1.76 / 1.28 / 1.08 / 1.04 s; 90-degree heading settling within ±2 degrees of 7.08 / 4.44 / 4.64 / 4.68 s; overshoot 4.203 / 0.844 / 0.033 / 0 degrees; heading-step peak angular acceleration 25.485 / 33.665 / 38.511 / 39.707 deg/s². Exact 90-degree crossing is not the settling metric: 3.8 approaches without overshoot.
- Current 3.8 passed eight representative seeds: 2000, 2081, 2069, 2189, 2037, 2083, 2006, 2114. This is a small selected set, not full-suite evidence.
- Current 3.8 smoke, seeds 2000–2023: 24 success, 0 collision, 0 boundary failure, 0 timeout; mean successful simulation time 65.09 s; mean episode peak yaw acceleration 47.41584 deg/s²; mean yaw-direction reversals 12.25. Previous 7.65 on the same smoke set: 24 success, 57.59833 s, 29.41794 deg/s², 9.95833 reversals. Thus agility improved, but smoke completion time and reversal count worsened. User acceptance must not be reported as aggregate navigation improvement.
- Dynamics/controller/playback unit suite: 17 passed (`test_vessel_dynamics`, `test_dynamics_tuning`, `test_yaw_control`).
- Playback uses fixed physics dt 0.04 s and planning every three physics steps. BASE_PLAYBACK_RATE = 2.0; displayed 1x / 2x / 4x targets simulation-time / wall-time ratios 2 / 4 / 8. Render cap 120 FPS; actual ratio depends on compute/render capacity. Measured short headless clock runs were approximately 1.997 / 3.992 / 7.648. This is playback acceleration, not improved simulation completion time.
- Full-episode actual-main-loop tests on seeds 2000, 2081 and 2189 gave identical physics-state/command hashes, planner/route/gap hashes and outcomes at displayed 1x / 2x / 4x. A forced-collision fixture also matched across all three rates. Respective successful simulation times were 73.40 / 98.44 / 89.16 s, independent of playback rate.

No final 200-episode validation has been performed at yaw inertia 3.8. The 200/200, 58.2142 s, 29.80515 deg/s² and 10.16 reversals below belong exclusively to the historical 7.65 configuration. Do not attach those numbers to the current frozen setting.

Evidence: `data/yaw_playback/step_candidates.json`, `inertia38/`, `inertia38_more/`, `smoke24/summary.json`, `before_clock.json`, `after_clock.json`, `playback_equivalence.json`. These are ignored local artifacts. Previously completed yaw-ticket source changes: `vessel_config.json`, `vessel_dynamics.py`, `main.py`, `test_vessel_dynamics.py`, `test_dynamics_tuning.py`, `test_yaw_control.py`. Current diagnosis does not change these files.

## Historical angular-path investigation — structure confirmed, no smoothing implemented

Classification: B, with an important distinction between discontinuous reference/commands and continuous physical vessel motion. The current primary displayed path is also the raw route used to select the controller's target; it is not merely a display-only angular approximation of a smooth control curve.

- `simulation.py` generates a Bezier curve and calls `utils.pure_pursuit` to select an auxiliary point. Subsequently `boat_control.select_command` calls `route_planner.route_target`.
- When A* succeeds, `route_target` overwrites `env.bezier_path` with the unsmoothed 0.25 m, eight-neighbour grid polyline. It selects a whole vertex at or beyond 3 m cumulative path distance, without arc-length interpolation. The resulting target feeds a sampled predictive controller; this is not direct classical Pure Pursuit curvature control.
- The 2D current-path line and 3D path strip consume that same `bezier_path` field. Its name and some comments are misleading. The next-gap Bezier overlay is auxiliary. Bezier also participates in gap/path feasibility and can feed control as a fallback when A* returns no route; it is not universally UI-only.
- Heading reference is atan2 from vessel position to the chosen target (after wall-margin clipping). Vertex selection and route replacement can change that reference discretely. The predictive controller selects discrete yaw commands every 0.12 simulation seconds; torque allocation, motor lag and physical integration still filter the actual hull motion.
- Read-only diagnostic replay of seed 2000: all 612 planning calls used the A* route and displayed the exact trimmed route. At 66.16 s, within the same stored route, a target-vertex transition coincided with a 6.987-degree reference jump and normalized steering command -0.30 → -0.70 over 0.12 s. This establishes co-occurrence, not proof that the vertex change alone explains all controller switching: obstacle rollout costs also select the command.
- Larger changes were associated with replanning: seed 2000 at 18.28 s had a 100.295-degree reference jump and steering delta +0.30, while physical heading changed only -0.209 degrees over the interval. At 25.96 s the reference changed -75.005 degrees and steering delta was -0.50. Thus smoothing vertices alone cannot be assumed to eliminate replanning discontinuities. Not every target switch changes the command: the largest same-route reference jump, 10.954 degrees at 68.80 s, retained the prior command.

Seed 2081 replay independently confirmed the structure: 805 of 821 planning calls used A*, with exact route/display equality; the remaining 16 calls used the fallback branch. At 87.64 s, a same-route target transition changed reference heading -9.985 degrees and normalized steering by -0.50, while the hull heading changed +1.134 degrees over 0.12 s. Both diagnostic episodes succeeded (2000: 73.40 s; 2081: 98.44 s), matching the prior frozen-setting results. These two replays diagnose path structure, not full-suite performance.

Recommended separate path-quality ticket (design only): retain raw A* route separately; construct a collision-checked, tangent-continuous control path within its safe corridor; use continuous arc-length lookahead and preserve progress across replans; retain the safe raw route if a proposed curve fails hull-clearance checks. Interpolating the lookahead point alone reduces vertex quantization but does not make polyline geometry smooth. Replanning handover continuity needs its own explicit check. Render the actual chosen control path from the same source; do not conceal the issue with display-only curves. Keep frozen dynamics and existing obstacle safety margins. No implementation, UI synchronization change or new benchmark sweep is authorized in this investigation.

Diagnostic evidence: `data/path_diagnosis/probe.py`, per-seed plan logs and summaries, and `source_manifest.json`. The probe observes function returns without replacing control functions. Production Python/config hashes are checked before and after replay. Existing user changes remain intact. Only this state document and ignored diagnostic artifacts are updated in this investigation; no commit or push.

## Historical 7.65 dynamics ticket — retained comparison evidence

## Historical objective and milestone

Dynamics tuning ticket complete. Adopted a simulation-validated 1.7x nominal yaw inertia, faster bounded yaw response, and 1.5x full-thrust terminal speed. Final paired 200-map comparison: 200 success, 0 collision, 0 timeout; successful mean time 58.2142 s. No navigation rewrite, UI/report changes, commit, or push.

Comparison baseline: mass 20 kg and yaw inertia 9 kg·m² (twice prior nominal 10 / 4.5). The new user request supersedes the previous fixed-inertia constraint for this ticket only. This is an engineering approximation, not an identified real-vessel model.

## Adopted dynamics and measured response

Actual runtime settings are loaded from `vessel_config.json`; `best_learned_params.json` currently has no recognized physics overrides.

| Setting | Pre-tuning baseline | Adopted |
|---|---:|---:|
| Yaw inertia, kg·m² | 9.0 (2.0x nominal) | 7.65 (1.7x nominal) |
| Yaw feedback response, s | 0.85 | 0.65 |
| Quadratic surge drag, N/(m/s)² | 18.0 | 7.116009950310329 |
| Cruise speed reference, m/s | 1.0 | 1.5 |

Unchanged: mass 20 kg; linear surge drag 6; sway drag 55 linear / 45 quadratic; yaw drag 8 linear / 5 quadratic; thruster arm 0.22 m; maximum thrust ±25 N per motor; actuator lag 0.25 s; maximum requested yaw rate 0.5 rad/s. Maximum differential moment is 11 N·m. Navigation parameters, hull envelope and all navigation source are unchanged.

There is no hard velocity clamp. Full-thrust equilibrium obeys `50 = 6*u + quadratic_drag*u²`. The new coefficient is calculated for `u_new = 1.5*u_old`; no velocity multiplication or thrust increase was introduced. Reduced drag represents a different assumed resistance curve, not measured improvement of an actual hull.

Cause classification: F, dominated by E (yaw feedback/operating-speed commands), with physical inertia and drag also contributing. Inertia-only tests at 2.0x / 1.7x / 1.5x sped up open-loop turning, but yaw-rate 90% response became 2.27 / 2.39 / 2.49 s at 0.01 s timestep: allocator feedback scales its moment with inertia. Increasing available moment alone did not improve the closed-loop response. Damping/lag/arm diagnostic changes were isolated experiments and were not adopted.

Final dynamics experiments at the actual dt = 0.04 s:

| Measurement | Baseline | Adopted |
|---|---:|---:|
| Full-thrust terminal speed, m/s | 1.508313 | 2.262469 |
| Peak forward acceleration, m/s² | 1.665916 | 1.814163 |
| Full-thrust forward speed t90, s | 1.24 | 1.76 |
| Full differential-thrust yaw t90, s | 1.68 | 1.48 |
| Commanded yaw-rate step t90, s | 2.28 | 1.76 |
| 90-degree heading settling within 2 degrees, s | 8.04 | 7.08 |
| Heading overshoot, degrees | 7.515 | 4.203 |
| Heading-step peak angular acceleration, deg/s² | 20.824 | 25.485 |
| Sinusoidal yaw tracking RMSE at 1 m/s, rad/s | 0.043007 | 0.034372 |

The higher terminal speed takes longer to reach 90% because startup force was retained. The initial acceleration is essentially unchanged and peak forward acceleration rises only about 9%. This is a bounded engineering model, not a real-vessel identification result.

## Dynamics ticket validation

- Baseline preserved before testing. Ticket 9 rejected recovery logic was confirmed absent. Existing user edits in `vessel_dynamics.py` constructor defaults were preserved; these defaults differ from runtime JSON (mass 10, inertia 6, linear surge drag 1, linear yaw drag 3), but `BoatEnv.configure_dynamics()` explicitly supplies all JSON fields. Do not use constructor defaults to describe the adopted runtime model.
- Tested inertia ratios 2.0 / 1.7 / 1.5 independently. Inertia 1.7 alone had a timeout on difficult seed 2069; adding the separately measured 0.65 s yaw response passed the six difficult seeds.
- Integrated candidate passed 8 difficult seeds: 2154, 2065, 2042, 2189, 2069, 2168, 2073, 2194. Candidate physical speed remains 1.5x; a 1.25 m/s cruise alternative was slower and had more reversals across 32 test maps and was rejected.
- Selected smoke: seeds 2000–2023, 24/24 success, collision 0, timeout 0, mean time 57.5983 s, mean peak angular acceleration 29.4179 deg/s², reversals 9.9583. Paired smoke baseline: 59.6117 s, 22.1251 deg/s², 7.375 reversals. Smoke alone has a reversal increase; across all 32 targeted/smoke maps, reversals were 10.59375 → 10.65625 and matched prior-success mean time was 71.2258 → 61.5032 s.
- `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest test_vessel_dynamics test_dynamics_tuning -v`: 13 passed. New tests verify force equilibrium, startup acceleration, absence of a hard speed clamp, heading overshoot, bounded acceleration and timestep convergence.
- Exactly one final 200-episode run for this ticket, seeds 2000–2199, 1800 × 630 px, 50 px/m, dt 0.04 s, planning every 3 steps, timeout 140 s. All initial map hashes match baseline; saved final source/config hashes match the adopted state. This is a previously used paired comparison set, not untouched holdout evidence.

| Final 200-map metric | 199/200 baseline | Adopted dynamics |
|---|---:|---:|
| Success | 199/200 (99.5%) | 200/200 (100%) |
| Obstacle collision / boundary / timeout | 0 / 0 / 1 | 0 / 0 / 0 |
| Successful mean time, s | 64.431156 | 58.214200 |
| Mean episode peak angular acceleration, deg/s² | 23.646594 | 29.805153 |
| Mean yaw-direction reversals (>3 deg/s) | 9.320 | 10.160 |
| Mean normalized command variation / s | 0.380964 | 0.464208 |
| Mean vessel speed, m/s | 0.586793 | 0.686292 |
| Maximum observed navigation speed, m/s | 0.999879 | 1.498499 |
| Mean sampled path length, m | 35.286531 | 36.277408 |
| Mean episode minimum hull clearance, m | 0.153455 | 0.159217 |
| Minimum hull clearance across episodes, m | 0.047980 | 0.080849 |

Speed/path/hull-clearance metrics above use matching 0.2 s trajectory samples plus endpoints, with moving buoy positions reconstructed from the unchanged motion equation and exact hull polygons. These sampled minima are not continuous-time lower bounds. Do not confuse them with the negative conservative circle-clearance proxy in the evaluator CSV.

The final candidate met the pre-recorded acceptance criteria (>=199 successes, no collision/boundary failures, >5% mean-time reduction, <20% mean-reversal increase, mean peak acceleration <40 deg/s²). Actual mean-time reduction: 9.65%; maximum individual episode peak angular acceleration: 44.512807 deg/s². Historical original snap-turn reference was about 229 deg/s² mean peak.

Adoption follows the NEW dynamics-ticket priorities. It does not satisfy Ticket 9's stricter no-individual-map-regression criterion: 131 prior-success maps got faster, 68 slower. Worst slowdown: seed 2081, +77.32 s (still success at 135.04 s). Seed 2189 now succeeds at 65.08 s. No seed-specific logic was added.

Files changed by this ticket: `vessel_config.json` (four settings), new `test_dynamics_tuning.py`, and this state file. `boat_control.py`, `route_planner.py`, the dynamics integrator, existing UI, report files and prior user changes were preserved.

Evidence and reproduction:
- `data/dynamics_tuning/final200/{summary.json,episodes.csv,trajectories.csv,maps.json,paired_comparison.json}`
- `data/dynamics_tuning/final_step_summary.json`, `baseline_extra/`, `final200_extra/`, `final_manifest.json`
- `data/dynamics_tuning/diagnostic_trajectories.png`: actual baseline/selected trajectories, heading and speed for 2189/2069/2168; gray circles show initial buoy positions.
- Run: `OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 benchmark_navigation.py --episodes 200 --seed 2000 --workers 8 --timeout 140 --width 1800 --output <new-directory>`.
- Data artifacts are ignored local files. Do not rerun the completed final comparison merely to regenerate identical evidence.

## Ticket 9 closure — rejected candidate, baseline retained

- Confirmed the original timeout enters a local restart cycle around 67.24 s. After that point: 33 target-direction jumps over 90 degrees, 38.20 s below 0.1 m/s, and only 0.499 m net movement by timeout despite 7.80 m travelled. Fixed 3 s reverse recoveries move only about 0.26–0.41 m.
- Diagnostic classification: H (fixed-time recovery ends without enough room for a behind-target restart), producing C/G-like restart/local-planning cycles. No DBSCAN or A* search bug was established.
- Final experimental candidate extended recovery to 5 s only for a nearby repeated stall with its target behind the vessel. For seed 2189, motion remained identical through 83.76 s; the extended recovery moved 0.834 m instead of 0.368 m, then succeeded at 104.88 s. Three targeted repeats matched.
- Candidate passed 14 unit tests, 10 difficult seeds and 24 smoke seeds. Exactly one final 200-map run: 200 success, 0 collision, 0 timeout; mean peak angular acceleration 23.617826 deg/s², reversals 9.255, successful time 64.4618 s.
- Rejected under the user's no-regression constraint: seed 2069 took 27.48 s longer and added 9 reversals; seed 2168 took 8.72 s longer. Of 200 episodes, 195 were unchanged, 2065/2124 improved, and 2189 was resolved.
- Removed only this ticket's experimental navigation changes and new tests. `boat_control.py` and `route_planner.py` match the saved baseline byte-for-byte. Restored baseline unit tests: 9 passed. Existing user changes were preserved.
- Evidence: `data/ticket9/decision.json`, `diagnosis.json`, `before/`, `candidate3/`, `final200/paired_comparison.json`; rejected source and patch: `data/ticket9/final_candidate_boat_control.py`, `final_candidate.patch`. These are ignored local experiment artifacts, not committed files.
- The result retained at Ticket 9 closure was 199/200, collision 0, timeout 1 (2189), mean peak acceleration 23.646594 deg/s², reversals 9.32, successful time 64.431156 s. That became the comparison baseline for the adopted dynamics tuning above. Do not confuse the rejected Ticket 9 200/200 experiment with the newly adopted dynamics result.

## Git and guideline integration (Ticket 0 history)

- Branch remains `main`; HEAD and fetched `origin/main` are `e9418fb7bcb621b42f6b6d7f53de97ee4c54bf41`.
- Fetched `origin/codex-guidelines`: `9d437d083b3c4b16646f5b9daaab4c93d7eefb05`.
- Compared both local guideline files before merging. Integrated scope control, efficient validation, physics constraints, project memory, and explicit Git authorization rules.
- Preserved project-specific engineering tone, no emoji/asterisk emphasis, vessel-motion evidence, overlap/clipping checks, 10/11/13 pt minimum text sizes, and Conventional Commits when committing is authorized.
- Removed only the `AGENTS.md` ignore entry; other ignore patterns remain intact.
- Imported missing `docs/CODEX_ROADMAP.md`, `docs/RESUME_PROMPT.md`, and this state file from the guideline branch. Updated this file after verification.
- No checkout, reset, code replacement, commit, or push. Existing `leaderboard.json` changes were also preserved.

## Working tree to preserve

Existing tracked modifications: `README.md`, `engine_3d.py`, `environment.py`, `leaderboard.json`, `main.py`, `test_success_rate.py`, `ui_renderer.py`.

Existing untracked work: `benchmark_navigation.py`, `boat_control.py`, `route_planner.py`, `simulation.py`, `test_vessel_dynamics.py`, `vessel_config.json`, `vessel_dynamics.py`, and `report5/` (data, summary, figures, plotting script, report).

This resumed task changed only `AGENTS.md`, `.agents/rules/user_guidelines.md`, `.gitignore`, and the three `docs/` files. Prior work was fingerprinted before applying guidelines; backup/manifest: `/tmp/kaboat-guidelines-resume-sxsk8gfx/working-files-sha256.json` (machine-local, not durable repository evidence).

## Baseline stern envelope (unchanged)

`boat_control.py` predicts clearance using the union of two hull capsules. Segment limits are -0.56 to +0.52 m along the vessel; lateral offsets ±0.22 m; radius 0.32 m. This replaces an earlier envelope that under-covered the stern. The wall envelope is 0.93 m. The matching existing unit test checks collision-hull vertices against the capsules.

The actual model integrates surge/sway/yaw with bounded lagged thrust. Prior per-step angular multiplier and artificial rotation-related displacement were removed in the earlier session. `simulation.advance()` is shared by GUI and evaluator; current planning is every 3 physics steps (0.12 s), physics dt is 0.04 s. The GUI accumulator has a wall-time regression test.

## Historical pre-tuning evidence

Sources:
- `report5/original_navigation/summary.json` and `episodes.csv`
- `report5/inertia_navigation/summary.json` and `episodes.csv`

At Ticket 0 closure, hashes matched the final current-run snapshot for `boat_control.py`, `environment.py`, `route_planner.py`, `simulation.py`, `vessel_dynamics.py`, `navigation.py`, `perception.py`, `utils.py`, and `config.py`. `vessel_config.json` also matches exactly. GUI/report/test/export-only changes elsewhere do not invalidate this dynamics evidence.

Both runs: seeds 2000–2199, 200 episodes, 1800 × 630 px, 50 px/m, dt 0.04 s, timeout 140 simulation seconds, common outcome/boundary conditions. All 200 initial map hashes match. The baseline preserves its original GUI planning pattern (first/last substep in a 4-step display batch); current planning is every 3 physics steps. Thus this is a paired whole-system comparison, not an isolated controller experiment with identical planning frequency.

The initially reported baseline 189/200 used a harness that did not retain `new_wp` inside a display batch. The previous session fixed this in `legacy_step()` using `_legacy_new_wp` and completed the corrected baseline run: 190/200. Use 190/200 for the final comparison. The pre-stern-correction 200/200 current result is historical, not the final result.

| Metric | Corrected original baseline | Final stern-envelope version |
|---|---:|---:|
| Success | 190/200 (95.0%) | 199/200 (99.5%) |
| Obstacle collision | 10 | 0 |
| Boundary failure | 0 | 0 |
| Timeout | 0 | 1 |
| Mean successful completion time | 49.652 s | 64.431 s |
| Mean episode peak yaw rate | 48.744 deg/s | 18.986 deg/s |
| Mean episode peak yaw acceleration | 229.082 deg/s² | 23.647 deg/s² |
| Mean yaw-direction reversals (>3 deg/s threshold) | 13.42 | 9.32 |
| Mean cumulative absolute yaw | 423.663 deg | 370.522 deg |
| Mean normalized-command variation / s | 0.214794 | 0.380964 |

Final success Wilson 95% interval: 97.22–99.91%. These seeds were reused for a final regression after the envelope correction; do not describe them as a fresh untouched holdout or claim universal collision freedom.

## Historical Ticket 0 verification

- `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest test_vessel_dynamics -v`: 9 tests passed.
- AST syntax validation: 11 relevant Python files passed.
- Current seeds rerun directly with `benchmark_navigation.episode((seed, root, False, 140., 1800, {}, 5, False))`: 2000 success at 44.88 s; 2028 success at 88.60 s; 2189 timeout at 140.00 s. Outcomes, map hashes and six saved motion metrics match (numeric tolerance 1e-8).
- Corrected baseline spot checks: seed 2040 success at 51.64 s and seed 2081 collision at 47.96 s, both matching saved outcomes and times.
- Preservation audit: all 28 fingerprinted pre-existing work files remain byte-for-byte unchanged after the resumed task.
- No 20–30 episode smoke or 100–200 episode validation repeated: completed exact-core-code evidence already exists.
- Prior 2D/3D/manual/line-trace smoke tests were recorded in the earlier session; not repeated here.

## Remaining issues and interpretation

- Mean reversal count rises about 9%; normalized command variation rises about 22%. The increased response and lower mean travel time do not establish improved command smoothness.
- Some maps take substantially longer even though all 200 succeed. Keep seeds 2081, 2037, 2083, 2006, 2114, 2069 and 2168 in follow-up regression sets.
- Physical coefficients remain engineering assumptions. Range-limited ideal obstacle geometry, sensor noise/occlusion, real-vessel resistance and actuator identification remain unverified.
- Constructor defaults and runtime JSON differ; the pre-existing defaults were intentionally preserved. A later parameter-consolidation ticket should make standalone construction unambiguous.
- Existing README/report figures remain historical and include obsolete 189/200 comparisons. No report or UI files were updated in this ticket.

## Next exact recommended ticket

Ticket 3 — characterize and reduce steering/differential-thrust command variation under the newly adopted, frozen dynamics. Separate avoidable command oscillation from necessary collision avoidance; use the listed slow seeds and fixed heading/curve tests. Do not simultaneously retune inertia/drag, rewrite navigation, or modify the waypoint display. Preserve this final 200-map evidence as the new comparison baseline.
