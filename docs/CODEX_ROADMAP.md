# CODEX_ROADMAP.md

This roadmap breaks USV development into small Codex tickets. Each ticket should normally have one primary objective and explicit acceptance criteria.

## Current continuation order, verified 2026-09-27

Displayed-4x GUI performance and current-state prediction display were the
priority after the navigation review. With optional Numba installed, real
X11/3D seed-2081 20-second runs now show production A at 99.51 FPS,
minimum 1-second rolling 78 FPS, 4001 physics steps/20.009 s and no >=50 ms
frame; opt-in B at 112.45 FPS, rolling minimum 102, 4001 steps/20.009 s and
no >=50 ms frame. Backlog ended below 0.005 simulated s on both. Finite-window
step rates differ from the 200/s target by less than one step at the stopping
boundary. Prediction is replaced at about 66.7 Hz wall time and clipped from
the current vessel pose every render, with no display throttle. See the leading
`PROJECT_STATE.md` entry for per-stage data, other seeds, tests and caveats.
The GUI loop still executes compiled planning before 2D rendering; no separate
physics/render process was introduced because measured work stays within the
requested 75-FPS target on the tested scenarios. Numba is required to reproduce
those performance numbers. No dynamics or navigation semantics were tuned.

Next ticket: return to the opt-in B U-shaped progress/replanning-continuity
failure only if navigation work is resumed. Keep production A as default and
preserve the 4x performance/freshness regression probes. Do not change 8x/16x
or start a 200-episode benchmark merely because the 4x ticket finished.

## Earlier navigation continuation order, recorded 2026-09-27

The frozen 3.8 kg m² dynamics, 0.04 s physics step and displayed playback
mapping remain unchanged. The production GUI still uses the existing
Gap-assisted, smoothed A* architecture A. The opt-in coarse-route plus
sampling-based predictive controller B passed a disjoint 24-map paired holdout
with 24/24 success, but failed the U-shaped dead-end stress case at 140 s after
the final yaw-command slew constraint; it also runs at about 81 FPS rather than
the existing A's 101 FPS on the same X11/3D seed-2081 probe. Its terminal-goal
quality variant passed seven targeted normal/hard seeds but has no disjoint
holdout. See the leading `PROJECT_STATE.md` entry and
`NAVIGATION_ARCHITECTURE_REVIEW.md` for exact numbers and limitations.

Next navigation ticket: diagnose the U-shaped exit-progress failure from the
stored trace and route audit. Change one general coarse-route/controller
commitment rule only if its mechanism is confirmed; test that geometry and
hard seeds before running another disjoint paired holdout. Do not tune the
frozen physical response or benchmark the unaltered rejected variant again.

Next performance ticket, separable from navigation: eliminate or schedule the
measured A* plus rollout planning-frame work without reducing display freshness,
physics cadence, collision checks or navigation equivalence. Confirm real-X11
p99/max and visible prediction freshness before any default B promotion.
The existing A display is a smoothed route, not a dynamically predicted
trajectory; label or expose its physical prediction in a later UI synchronization
ticket. No 200-episode run or commit/push is authorized by this roadmap.

## Previous continuation order, verified 2026-09-26

The frozen dynamics remain yaw inertia 3.8 kg·m² with unchanged thrust, drag, damping, speed, and displayed playback multipliers. The automatic chain remains LiDAR observations → map/clustering → optional accepted Gap corridor → A* raw route → safety-validated smoothed control path → continuous lookahead → predictive controller. No 200-episode validation of this integrated candidate has been run.

The real-X11 GUI FPS ticket is a measured partial improvement, not a 120-FPS completion. See the leading entry in `PROJECT_STATE.md`: 1x/2x/4x/8x/16x measured about 102/69/25/23/24 render FPS, with zero sampled >=250 ms frames but remaining 100–140 ms replan frames at higher speeds. Fixed-step physics and navigation hashes match across speeds; 24 smoke trajectories remain identical. The current eight-step catch-up bound trades high-speed wall throughput for visible responsiveness. Displayed 16x requests 800 physics steps/s but reached about 187 on this machine. Seed 2189 still times out; this ticket did not address it. The visible legacy waypoint/candidate overlays were removed, while the internally used passage state remains.

Next performance ticket: determine whether a portable compiled or incremental A*/smoothing path and a separated render/physics scheduler can close the measured compute gap without changing perception, route, command or trajectory results. Compare against the saved X11 profiles and stop if the semantic or maintenance cost is too high. If a lower display/speed target is explicitly accepted, proceed next to seed 2189 timeout and route-replacement continuity. A later UI ticket can consolidate the now-empty former weight panel and clarify known/unknown map display. Historical Tickets 0–12 below describe broader phases, not a request to restart them.

## Ticket 0 - Resume and validate interrupted work

Goal:
Finish verification of the current uncommitted dynamics/control/planner changes without adding new features.

Scope:
- inspect current diff;
- verify the stern collision-envelope change;
- run minimal regression and final holdout if required;
- produce verified baseline-vs-current metrics.

Done when:
- current code is internally consistent;
- success/collision/timeout are separated;
- representative dynamics/control metrics are available;
- no unverified performance claim remains.

## Ticket 1 - Establish reproducible benchmark protocol

Goal:
Make evaluation conditions identical and repeatable.

Scope:
- fixed seed handling;
- fixed map size;
- explicit physics timestep;
- explicit planning frequency;
- clear realtime/evaluation timebase;
- separate collision and timeout;
- machine-readable per-episode output.

Done when:
The same commit and seed set reproduce materially identical aggregate results.

## Ticket 2 - Validate vessel dynamics model

Goal:
Check physical consistency independently of obstacle avoidance.

Scope:
- mass/inertia units;
- thrust-to-force model;
- thrust lag;
- yaw moment;
- rotational damping;
- lateral damping;
- straight-line acceleration/deceleration;
- step-steer/yaw response.

Do not:
Tune navigation in this ticket.

Done when:
Standalone tests characterize the model and no frame-rate-dependent dynamics remain in the validated path.

## Ticket 3 - Stabilize low-level control

Goal:
Reduce oscillation and overshoot with the chosen dynamics fixed.

Scope:
- heading/yaw control;
- steering/differential thrust;
- command smoothing/rate limits;
- anti-windup or damping logic if applicable.

Do not:
Change obstacle perception or global gap scoring unless required by a demonstrated interface bug.

Done when:
Heading step/curve tracking is smoother than baseline with quantified reversal/yaw metrics and no major loss of response.

## Ticket 4 - Improve inertia-aware local avoidance

Goal:
Make obstacle avoidance work with the realistic/high-inertia dynamics rather than relying on snap turns.

Scope:
- braking/turn anticipation;
- predicted vessel pose/hull footprint;
- minimum turning feasibility;
- restart behavior after safe stop;
- local route feasibility.

Done when:
Known failure seeds are resolved without weakening dynamics and holdout safety improves.

## Ticket 5 - Gap selection robustness

Goal:
Reduce bad gap choices, dead ends, wall hugging, and repeated direction switching.

Scope:
- candidate gap feasibility;
- width/clearance;
- goal progress;
- heading consistency;
- hysteresis/state persistence;
- dead-end handling.

Done when:
Gap-switch count and avoidable deadlock cases decrease on fixed scenarios.

## Ticket 6 - Bezier + Pure Pursuit path quality

Goal:
Improve path curvature and tracking smoothness.

Scope:
- control point placement;
- curvature limits;
- lookahead;
- speed-dependent lookahead;
- obstacle clearance of the path itself.

Done when:
Tracking error and yaw/steering oscillation improve without reducing safety.

## Ticket 7 - Parameter consolidation

Goal:
Remove scattered or contradictory tuning constants.

Scope:
- identify duplicated parameters;
- define units;
- move stable parameters to one configuration source;
- document physically meaningful vs algorithmic parameters.

Do not:
Use this ticket as a broad rewrite.

Done when:
Each important parameter has one authoritative definition and a clear meaning.

## Ticket 8 - Holdout evaluation

Goal:
Generate final algorithm evidence.

Scope:
- untouched holdout seed set;
- 200+ episodes if computationally reasonable;
- success/collision/timeout;
- completion time;
- path length;
- clearance;
- oscillation/control metrics;
- failure-seed list.

Done when:
Results are reproducible and saved in compact CSV/JSON plus a summary.

## Ticket 9 - Failure analysis

Goal:
Explain remaining failures rather than blindly tune.

Scope:
Cluster failures into categories such as:
- narrow gap;
- stern clipping;
- late turn;
- deadlock;
- excessive overshoot;
- timeout;
- perception/gap error.

Done when:
Each remaining failure category has representative seeds and a plausible measured cause.

## Ticket 10 - Presentation dataset

Goal:
Freeze trustworthy results for presentation use.

Scope:
- choose baseline/current runs;
- export trajectories and time series;
- create a small clean dataset;
- record exact commit/config/seed metadata.

Do not:
Modify the algorithm in this ticket.

Done when:
Every plot can be traced back to a real run and exact configuration.

## Ticket 11 - Visualization and PPT figures

Goal:
Create presentation-ready engineering figures.

Use:
`.agents/rules/user_guidelines.md`

Preferred figures:
- same-seed trajectory before/after;
- heading/yaw comparison;
- steering reversal comparison;
- speed and clearance;
- success/collision/timeout summary;
- algorithm pipeline diagram.

Done when:
Figures are readable, accurate, uncluttered, and based on real results.

## Ticket 12 - Final code cleanup

Goal:
Clean only after behavior is frozen.

Scope:
- remove dead code;
- improve names/comments;
- dependency/reproducibility file;
- README update;
- final regression.

Done when:
Cleanup causes no behavior regression and the project can be reproduced from documented instructions.

## Operating rule

Do not combine Tickets 2-7 into one autonomous request unless there is a strong reason.

Default sequence:

`0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12`

A ticket may be skipped if evidence shows it is unnecessary.
