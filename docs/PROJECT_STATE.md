# Main-heavy experiment state

## Current inspection candidate: V2.1_GAP_PERSISTENCE (2026-10-06)

Ready for user GUI inspection. Run
`MAIN_HEAVY_MOMENTUM_GAP=1 python3 main.py`. No commit/push, large regression,
controller tuning or MAIN/CODEX branch changes in this ticket.

- Valid first/second pair identities are retained before any new ranking.
  Intersections update on the actual current displayed route; a moving point on
  the same pair is not a switch. Ranking still uses local crossing-region
  midpoint proximity, then compactness, only when acquiring a replacement.
- Invalid markers disappear immediately. A replacement first pair must remain
  valid for two distinct prediction generations, and a replacement second for
  three. Repeated display refreshes in one generation do not advance this
  confirmation. A valid pending pair is checked before reranking. Initial
  acquisition and actual first passage/valid second promotion are immediate.
  Remembered identities can resume only after validation against CURRENT
  candidates, route, front hemisphere and exact safety; no stale ghost drawing.
- Second acquisition scans later actual future crossing groups for a segment
  that does not intersect the first finite segment (including shared endpoints
  or collinear overlap). If none is eligible, second stays empty. Existing
  route separation/bounded distance band remains; unrelated pairs are not added.
- Corrected the display safe interval's observed-circle support projection.
  GUI cluster centroids are surface points, not circle centers. Previously adding
  their center displacement to the circle radius double-counted displacement and
  falsely invalidated some still-safe intersections. Current observed geometry,
  actual hull, exact polygon clearance gate and 0.20 m margin are unchanged.
- Internal diagnostics record reasons, completed/current visible lifetimes and
  identity switches INCLUDING changes across hidden frames. Empty confirmation
  frames cannot artificially reduce the reported pair-change count.

Same-input complete-route comparison with the previous checkpoint's candidate
geometry and state logic, seeds 2000/2069:

| Seed | First pair switches before/after | Second pair switches before/after | First visible lifetime before/after sim s | Second visible lifetime before/after sim s |
| --- | ---: | ---: | ---: | ---: |
| 2000 | 18 / 12 | 13 / 9 | 1.002 / 1.712 | 0.688 / 0.589 |
| 2069 | 18 / 13 | 19 / 8 | 1.377 / 1.598 | 0.616 / 0.992 |

Combined true first changes 36 -> 25; second 32 -> 17. Second display fractions
fell from 33.72%/38.05% to 28.74%/27.95% because crossed segments are no longer
filled and replacements need confirmation. Second lifetime did not improve on
seed 2000; do not claim universal improvement or fill an unrelated second GAP.
Artifacts: `v21_gap_persistence_annotation_comparison.json`, annotation ticks,
parity summary and seed trace logs in `data/main_heavy/`. Earlier confirmation-only
results are preserved under `v21_gap_confirmation_only_*`.

127 targeted tests and syntax checks pass. Seeds 2000/2069 succeed in 776/888
steps. Every physics state, applied command and hull clearance match the previous
checkpoint exactly (maximum differences zero), with identical map hashes. Every
visible waypoint was checked against current candidate endpoints, current bow
and exact displayed route intersection. Raw controller arrays are unchanged.

Actual real-X11 MAIN-style 2D 4x, 22.008 wall s: 239.963/240 requested steps/s,
115.97 average FPS, 107 minimum one-second rolling FPS. Median/p95/p99/max frame
time 8.699/10.677/11.303/31.256 ms; >=33/50/100 ms frames all zero. Backlog
max/final 0.41401/0.03216 simulation seconds, nonaccumulating. Annotation
mean/p95 1.688/2.307 ms. Screenshot inspected:
`v21_gap_persistence_gui_2069.png`; profile/frame files use
`v21_gap_persistence_4x_gui` prefix. Stop for user GUI feedback; no further tuning.

### Previous inspection checkpoint: V2.1_MIDPOINT_PRESENTATION

Awaiting user GUI inspection; no commit/push or large regression in this ticket.
Run `MAIN_HEAVY_MOMENTUM_GAP=1 python3 main.py`. Existing V1 control remains
available with `MAIN_HEAVY_MOTION_VERSION=V1`; the default experimental flag
selects V2.1_MIDPOINT_PRESENTATION. MAIN is unchanged. CODEX motion source and branch HEAD are
unchanged; the UI-only wall-hit correction is isolated in a CODEX worktree.

Latest local midpoint-proximity presentation update (2026-10-06):
- Actual completed CODEX prediction and the interpolating Bezier construction
  are unchanged. First select a route crossing region using the existing MAIN
  distance band; then compare ONLY that region's current front/safe candidate
  segments. Width/orientation no longer choose the first crossing region.
  The second region is chosen by its route-order anchor and existing bounded
  future distance band, before its representative segment is considered.
- `select_presentation_gap()` prioritizes absolute distance from each candidate's
  existing midpoint to its actual route intersection. Candidates within one GUI
  coordinate pixel of the best distance use shorter segment length as tie-break;
  orientation/clearance/deterministic geometry only resolve remaining ties.
  Midpoints never become waypoints: the returned point remains the exact
  selected segment/unchanged displayed route intersection. All candidates have
  already passed the same exact hull safety and current front-180-degree gate.
- First/second valid tracked identities remain pinned despite improved midpoint
  scores or newly farther passages. Second acquisition clears the whole first
  local crossing group. For retention, only the actual waypoint separation
  footprint is used: newly appearing presentation aliases extending the first
  group's end cannot evict an otherwise valid second. Unsafe/off-route/absent
  current candidates and rear crossings still release immediately. First passage
  promotes the valid second as before. No timer, legacy-weight ranking, virtual
  pair, off-route point smoothing or changed physical safety threshold.
- Goal/waypoint endpoint clipping, MAIN-style renderer, candidate population,
  legacy post-selection diagnostics and episode reset are unchanged.

Same-input presentation comparisons on two complete existing routes:

| Seed | First midpoint error median before/after px | Second midpoint error median before/after px | First switches before/after | Second switches before/after |
| --- | ---: | ---: | ---: | ---: |
| 2000 | 31.01 / 26.81 | 37.18 / 25.06 | 13 / 12 | 6 / 7 |
| 2069 | 30.41 / 25.35 | 48.24 / 28.21 | 22 / 17 | 11 / 10 |

Combined first switching 35 -> 29; second switching 17 -> 17 (no claimed
measured reduction). Combined first/second midpoint-error medians
30.87 -> 25.51 px / 41.47 -> 27.14 px. Combined segment-length medians
285.97 -> 291.01 px / 265.71 -> 263.13 px: this prototype does NOT demonstrate
an overall first-segment shortening. Compactness is intentionally secondary to
midpoint proximity; persistence can retain wider valid incumbents. Do not tune
further before user inspection. Pairwise logs:
`data/main_heavy/v21_midpoint_annotation_comparison.json` and annotation ticks.

121 targeted tests pass, including midpoint-vs-width priority, one-pixel compact
near-ties, remote-region exclusion, actual non-midpoint waypoints, first-group
alias growth retention, stronger second identity hold and prior front/candidate/
safety/goal/reset tests. Syntax checks pass. Full seeds 2000/2069 succeed in
776/888 steps; map hashes and every physics state/applied command/clearance match
V2.1_FORWARD_CANDIDATE_GAPS exactly. Control arrays/commands were checked around
every annotation and all visible waypoints checked against current candidates,
current heading and exact displayed intersections. Results:
`v21_midpoint_parity_summary.json` and paired trace logs.

Real-X11 MAIN-style 2D 4x, 22.001 s: 239.994/240 requested steps/s;
116.73 average FPS, rolling minimum 108. p95/p99/max
10.505/11.204/45.623 ms; >=33/50/100 ms frames 1/0/0. Backlog max/final
0.40550/0.02503 simulation seconds, nonaccumulating. Annotation mean/p95
1.530/2.170 ms. GUI screenshot inspected: `v21_midpoint_gui_2069.png`;
performance summary/frame files use `v21_midpoint_4x_gui` prefix.
MAIN/CODEX branch HEADs unchanged. No commit/push, large regression or control
changes. Stop here for user GUI inspection.

### Previous inspection checkpoint: V2.1_FORWARD_CANDIDATE_GAPS

Latest forward/current-candidate annotation update (2026-10-06):
- Found two separate populations: MAIN-style displayed GAPs used front buoy
  pairs, while annotation independently enumerated all-around buoy pairs.
  Retention also rechecked stale endpoints even after their pair vanished from
  the current candidate set. These permitted rear/non-displayed waypoints.
- Removed the independent annotation population. Route crossings now use the
  exact current MAIN GUI candidate list (also generated when GAPS is hidden).
  Retention uses tracked pair identity to retrieve CURRENT candidate endpoints;
  absent pairs release immediately. No virtual pair or hidden rear fallback.
- `crossing_in_front()` uses current bow dot actual route intersection, inclusive
  +/-90 degrees with floating-point rounding tolerance only. All members of a
  passage group are filtered before regrouping; a rear canonical representative
  cannot hide a valid front alternative. First/second retention and promotion
  also enforce the current-bow condition. After every physics tick, a cached
  waypoint that moves behind the bow triggers display-only reannotation without
  waiting for the next prediction. Controller prediction cadence is unchanged.
- Newly acquired first crossings inside the hull bounding footprint are not
  selected merely to fill a marker. MAIN distance/shape preferences remain
  route-local. Existing first identities can approach normally until passage,
  candidate disappearance or route/safety/front invalidity. Exact updated
  intersections remain on the displayed Bezier, with no off-route coordinate
  smoothing. Candidate markers exclude retained selected identities as well.
- Existing bounded second distance range/persistence, partial/X overlap
  allowance, exact crossing safety and goal clipping are preserved. Legacy
  metrics are still computed only AFTER selection. No controller/planner,
  dynamics, raw rollout, GUI renderer, MAIN or CODEX branch changes.

115 targeted tests passed; 52 affected tests passed again after marker-identity
exclusion. Syntax checks pass. New cases cover recovery-route rear absence,
missing current pairs, current candidate endpoint updates, bow-turn release,
+/-90 degree crossing boundaries, mixed front/rear local groups and immediate
between-prediction invalidation. Seed 2000/2069 complete in 776/888 steps with
zero differences in every physics state/applied command/clearance and identical
map hashes versus V2.1_SECOND_GAP_BAND. Every physics-tick display was checked
for front crossings; every annotated waypoint belonged to the current candidate
set and used its current endpoints, lying on the exact drawn route. Stored:
`data/main_heavy/v21_forward_candidates_parity_summary.json`, paired traces and
`v21_forward_candidates_annotation_ticks.json`. Second separation medians were
132.44 / 116.54 px; invalid/absent front passages deliberately leave markers empty.

Actual real-X11 MAIN-style 2D 4x continuous 22.000 s: 239.995/240 requested
steps/s, 117.23 average FPS, 104 rolling minimum. p95/p99/max
10.545/11.351/29.353 ms; >=33/50/100 ms frames all zero. Backlog max/final
0.40811/0.02834 simulation seconds, nonaccumulating. Annotation mean/p95
1.530/2.155 ms. Actual screenshot inspected:
`data/main_heavy/v21_forward_candidates_gui_2069.png`; performance summary/frames
use the `v21_forward_candidates_4x_gui` prefix. MAIN/CODEX HEADs unchanged.
No commit/push or large regression. Stop for user GUI inspection.

### Previous inspection checkpoint: V2.1_SECOND_GAP_BAND

Latest bounded/persistent second-GAP update (2026-10-06):
- The previous acquisition ordered nonintersecting/close gates ahead of X gates,
  then selected the earliest route crossing. Retention accepted any positive
  first-to-second arc difference. Same-input logs contained a 0.35 px second
  separation and a combined 77.91 px median; these were different identities
  describing nearly the same place, not different control routes.
- `second_gap_band()` derives a minimum distinct passage distance from the
  actual hull bounding extent, safety margin, first passage footprint and local
  group end. The preferred range uses unchanged MAIN's measured second-gap
  median/IQR (84.15 px median, 72.78--141.22 px IQR at 50 px/m), augmented by
  vessel/actuator travel and local yaw-sweep geometry. No fixed pixel lookahead
  or horizon-fraction target is introduced. Only the available completed route
  caps the range; the nominal far edge does not chase a longer horizon.
- Acquisition prefers the far portion of that bounded range. If a sparse route
  has no crossing in-range, it chooses the closest distinct safe passage to the
  range, never an almost coincident second just to fill the display. Partial
  overlaps and X segments are allowed; route separation precedes those visual
  tie-breaks. Existing local passage grouping and exact crossing safety remain.
- A retained second is revalidated on the current route and retained regardless
  of a newly visible farther candidate or a shifted preferred band. It releases
  only on crossing, route/safety invalidity, duplicate identity or insufficient
  route separation. Actual crossing of the first still promotes the valid old
  second. Waypoints remain exact segment/Bezier intersections; goal clipping
  and controller rollout arrays are unchanged. Band values are diagnostics only.

Paired presentation comparison on the same completed routes/observations:

| Seed | Second arc separation median before/after (px) | Second identity switches before/after | Second displayed before/after |
| --- | ---: | ---: | ---: |
| 2000 | 59.21 / 127.49 | 12 / 4 | 60.23% / 33.20% |
| 2069 | 109.45 / 117.57 | 20 / 18 | 67.23% / 57.43% |

Combined median separation 77.91 -> 124.34 px; second switches 32 -> 22.
No valid retained identity switched due to a newly farther candidate. Fourteen
first-GAP promotions were recorded. The lower display fraction is deliberate:
no second is fabricated when only nearly coincident crossings remain. Full
seed 2000/2069 episodes succeed in 776/888 physics steps with exactly zero
changes in position, heading, yaw, speed, surge, commands/PWM and hull clearance
relative to the preceding checkpoint; map hashes match. Raw logs and paired
summary: `data/main_heavy/v21_second_band_annotation_ticks.json`,
`v21_second_band_annotation_comparison.json`, `v21_second_band_parity_summary.json`.

106 targeted tests pass, including bounded horizon growth, incumbent persistence,
close-crossing release, short-horizon absence, X-gate distance precedence,
first/second promotion, exact marker intersections, goal clipping and reset.
Syntax checks pass. Real-X11 MAIN-style 2D 4x continuous 22.000 s:
239.999/240 requested physics steps/s; 113.56 average FPS; rolling minimum 100.
p95/p99/max 11.074/12.164/29.613 ms; >=33/50/100 ms frames all zero. Backlog
max/final 0.39788/0.000113 simulation seconds, nonaccumulating. Annotation
mean/p95 1.930/2.849 ms. GUI screenshot inspected:
`data/main_heavy/v21_second_band_gui_2069.png`. Measurement logs:
`v21_second_band_4x_gui_profile.json` and `v21_second_band_4x_gui_frames.json`.
No controller/planner/dynamics/rendering-architecture changes; MAIN/CODEX branch
HEADs remain unchanged. Stop here for user GUI inspection; no large regression,
commit/push or further tuning.

### Previous inspection checkpoint: V2.1_STABLE_GAPS_GOAL

Latest goal/persistent-annotation update (2026-10-06):
- Display clipping now applies even without GAP1. It inserts the actual segment
  projection before the exact goal-center endpoint and removes every subsequent
  knot. With two waypoints, goal before GAP2 ends at goal (retaining GAP1), rather
  than incorrectly falling back to GAP1. Waypoint-before-goal remains a valid
  earlier endpoint. Control rollout arrays are untouched.
- `heavy_gap_state.py` retains tracked first/second obstacle-pair identities and
  revalidates their fixed presentation segments against each new predicted
  Bezier and observed exact hull safety. Valid identities are not replaced for
  small legacy-width/distance/verticality differences. Actual finite boat/portal
  crossing releases GAP1 and promotes a still-valid GAP2. A just-crossed passage
  is not reacquired until the vessel leaves its physical crossing footprint;
  this uses passage geometry, not a timer. Unsafe/off-route annotations release
  immediately. No off-route crossing-coordinate EMA is used: the waypoint stays
  the exact new Bezier/retained-segment intersection. Reset clears all persistent
  identities, crossings, ages, counters, previous position and passed regions.
- Independent route-forward GAP2 candidates now use staged presentation
  preferences: ideal separated/nonintersecting, closer nonintersecting, partial
  overlap/X fallback. Near-total duplicate overlays and the same passage remain
  excluded; exact crossing safety remains hard. No control horizon expansion or
  speculative unsafe extrapolation was added: the available future prediction
  is scanned in full. Legacy panel metrics remain post-selection only.
- Internal diagnostics expose first/second switch counts, selected/last-valid
  frames, ages and release reasons; nothing new is drawn in the main HUD.

Same-input paired annotation comparison over full seeds 2000/2069:

| Seed | First switches before/after | Second displayed before/after | Relaxed stateless second switches / persistent |
| --- | ---: | ---: | ---: |
| 2000 | 54 / 15 | 7.72% / 60.23% | 40 / 12 |
| 2069 | 63 / 18 | 9.80% / 67.23% | 58 / 20 |

Combined first switches: 117 -> 33 (71.8% fewer). Combined second display
fraction: 8.83% -> 63.96%. Old strict policy had only three second identity
switches because it almost never displayed a second GAP; after broadening
eligibility, persistence reduces switches from 98 stateless choices to 32.
These counts are adjacent planning-tick changes between two non-null identities;
acquisition/removal are recorded separately. All inputs are the same completed
CODEX routes/observations, not different navigation runs. The release audit
found no immediate re-selection of an actually passed first GAP in final logs.
Comparison/tick logs: `data/main_heavy/v21_stable_gaps_goal_annotation_comparison.json`
and `v21_stable_gaps_goal_annotation_ticks.json`.

Both full episodes still match every state/applied command/clearance and map
hash of the preceding checkpoint (776/888 steps, both success). Parity file:
`v21_stable_gaps_goal_exact_parity.json`. 100 targeted tests pass, including
no-GAP goal clipping, intermediate-goal truncation, persistent first/second
identity, exact moving intersection, finite crossing/promotion, immediate
unsafe release, reset, passed-region release, and softened independent GAP2.

Actual real-X11 MAIN-style 2D 4x, continuous 22.006 s, no pauses/manual/speed
changes: 239.980/240 requested steps/s, 115.247 average FPS, 104 minimum
1-second rolling FPS. p95/p99/max 10.960/11.706/30.029 ms; >=33/50/100 ms
frames 0/0/0. Backlog max/final 0.42712/0.01271 simulation seconds,
nonaccumulating. Annotation mean/p95 1.934/2.769 ms. Actual screenshot inspected;
MAIN styles, sky-blue path, single pursuit and legacy diagnostic panel remain.
Profile/frame/screenshot: `v21_stable_gaps_goal_4x_gui_*` and
`v21_stable_gaps_goal_gui_2069.png`. No navigation/dynamics/physics/timing,
main/codex ref, renderer, benchmark or leaderboard changes in this update.
No commit/push or large evaluation. Stop for user GUI inspection. Descriptions
below are historical candidates and may contain superseded display rules.

Latest measured presentation update (2026-10-06): MAIN was measured from an
isolated archive of unchanged main 757243cc68f25f5d6fbbac4dc955dab02d5f4010.
Three representative seeds 2000/2069/2081 completed at 1274/1296/1254 MAIN steps.
Only geometry selection statistics were collected using deterministic 1x
scheduling and a disabled renderer; this is NOT a GUI timing benchmark.
3230 first-GAP and 2335 second-GAP per-step snapshots provide these p25/median/p75
lengths in MAIN render pixels:

| Quantity | p25 | median | p75 |
| --- | ---: | ---: | ---: |
| First route distance | 99.34 | 136.30 | 184.47 |
| First forward projection | 94.03 | 130.42 | 176.10 |
| First segment length | 124.28 | 139.71 | 187.97 |
| First verticality | 0.856 | 0.960 | 0.990 |
| First forward alignment | 0.968 | 0.987 | 0.996 |
| Second route distance | 199.72 | 236.37 | 267.01 |
| First-to-second route separation | 72.78 | 84.15 | 141.22 |
| Second segment length | 130.85 | 152.24 | 190.05 |
| Second verticality | 0.738 | 0.882 | 0.972 |
| Relative segment angle (rad) | 0.448 | 0.688 | 1.113 |

MAIN samples include 1479 segment crossings and zero collinear overlaps; these
visual conflicts are NOT inherited by the new presentation selector. Full
samples/metrics: `data/main_heavy/v21_legacy_gap_presentation_samples.json` and
`v21_legacy_gap_presentation_stats.json`. The measured profile is retained in
`heavy_gap_presentation_profile.json` as scale-independent lengths, read once
at adapter initialization by `heavy_gap_profile.py`.

All eligible candidates must still be true finite, exact-safe crossings of the
unchanged physical-prediction Bezier. For GAP1, candidates before measured p25
are skipped if a farther crossing exists; if only a near passage exists, it
remains available. Among route-local passages, bounded lexicographic distance
and compact-width distribution preferences choose the presentation region.
Only inside that region does verticality resolve remaining ties. There is no
global legacy score and no approximation that invents off-route waypoints.
GAP2 still follows route order after the selected cluster, with physical
separation plus measured second-separation p25, nonoverlap/nonintersection,
then local compact-width/distance and verticality preferences. Missing GAP2 is
allowed. Sole wide valid passages remain fallback rather than being hidden.
Original control routes and the untrimmed source Bezier are unchanged; existing
waypoint/goal display clipping and post-selection diagnostics are retained.

88 targeted tests passed. Two full physical episodes 2000/2069 match every
state/applied command/clearance and map hash in the immediately preceding
checkpoint (776/888 steps, both success). Parity file:
`data/main_heavy/v21_legacy_presentation_exact_parity.json`.
Real X11 MAIN-style 2D 4x continuous 22.005 s, no pause/manual/speed-change frames:
239.995/240 requested steps/s, 117.983 average FPS, 103 rolling minimum.
p95/p99/max 10.414/11.218/30.037 ms; >=33/50/100 ms frames 0/0/0.
Backlog max/final 0.40546/0.00983 simulation seconds, nonaccumulating.
Annotation mean/p95 1.462/2.143 ms. GUI screenshot inspected: former near tall
GAP is skipped; sky-blue route and single current pursuit remain, legacy panel
is populated. Profile/frame/screenshot files: `v21_legacy_presentation_4x_gui_*`
and `v21_legacy_presentation_gui_2069.png`.

Descriptive current render-frame presentation medians were route distance
133.70 px, segment length 270.07 px, verticality 0.693; second separation
126.27 px. First-distance distribution is near the MAIN reference. Segment
widths can remain larger than MAIN because the exact physical hull/crossing
safety gate is unchanged and compact valid alternatives may not exist. This
is a display preference, not a promise to reproduce unsafe MAIN geometry.
Comparison: `v21_legacy_presentation_distribution_comparison.json`; MAIN
per-step versus current per-render sampling is explicitly labeled. No large
regression or commit/push; main/codex source refs unchanged. Stop for user GUI
inspection. Prior sections below are historical checkpoint descriptions.

Latest UI-only addition (2026-10-06): safe route crossings are grouped into
anchored local passage regions with a shared observed boundary and overlap in
arc distance and physical crossing position. Alternatives previously discarded
by deduplication are retained for presentation. Within that same group only,
map verticality |dy|/length is the first tie-break, then compact segment length
and deterministic geometry. It cannot move selection to a different passage,
modify the control route, or regenerate the untrimmed interpolating Bezier.
GAP2 eligibility is checked after the first cluster ends, including the existing
speed/hull/curvature separation and nonintersection/overlay rules, before its
local verticality tie-break. If none qualifies, GAP2 is absent. Waypoints remain
exact route/segment intersections; legacy diagnostics remain post-selection.
Selected groups do not reappear as extra candidate-waypoint annotations.

Two full small sanity episodes (2000/2069) exactly match every state, applied
command and clearance in the previous GUI-semantics logs (776/888 steps,
same map hashes, both success). No dynamics/controller/planner changes.
80 targeted tests pass; includes local-only ranking, unrelated pairs remaining
separate, nontransitive grouping, cluster-end separation, nonintersection before
verticality, candidate-order determinism, actual exact-safe crossing eligibility
and original clipping/reset/control-isolation checks. Two-GAP test fixtures now
use a farther second passage; close crossings need not produce a second marker.

Actual X11 MAIN-style 2D 4x, continuous 22.007 s (2463 frames, no pauses/manual
frames/speed changes): 239.970/240 requested steps/s, 111.926 average FPS,
98 minimum 1-second rolling FPS. p95/p99/max 10.854/11.575/31.450 ms;
>=33/50/100 ms frames 0/0/0. Backlog max/final 0.43505/0.02559 simulation s,
nonaccumulating. Annotation mean/p95 1.628/2.431 ms. Real screenshot inspected:
MAIN panel/sky-blue path/current pursuit and legacy weight panel retained.
Results: `data/main_heavy/v21_local_presentation_exact_parity.json`,
`v21_local_presentation_4x_gui_profile.json`, `v21_local_presentation_4x_gui_frames.json`,
`v21_local_presentation_gui_2069.png`. No large evaluation or commit/push;
main/codex refs unchanged. Stop here for user GUI inspection. Measurements below
remain prior checkpoint history.

Prior UI-only handoff (2026-10-06):
- MAIN_HEAVY's 53 targeted tests passed again; its already measured X11 2D
  4x result remains 117.4 average FPS, 99 minimum rolling FPS, 239.98 steps/s,
  nonaccumulating backlog and zero >=100 ms frames. No further motion tuning.
- `/home/soonhong/kaboat_codex_ui` is a separate worktree on `codex` with
  uncommitted UI-only changes. Do not remove it or overwrite its edits.
  `lidar_hit_display.py` filters arena-boundary marker positions only;
  `ui_renderer.py` uses it for yellow world/POV points; `engine_3d.py` uses it
  for amber markers after emitting the original rays. Worker payload adds
  arena bounds only for that display classification. Buoy hits, scan ranges,
  wall perception, grid, planner, safety and dynamics are untouched.
- CODEX display/RC tests: 22 passed. Sequential real-X11 before/after runs
  with the production 3D worker matched every state, command and scan/hit
  value for seed 2000's first 180 physics steps. Both retained 8210 sensor
  wall hits: only rendering is suppressed. Results and screenshot are in
  `data/main_heavy/v21_codex_wall_ui_sanity.json` and
  `data/main_heavy/v21_codex_wall_hits_hidden.png`.
- No commit/push. Next action: user GUI inspection; no additional tuning.

Actual control is the pinned CODEX `eta_continuity_forward` source at
`60f2cf46500354a3e0690b9b62888e0dd166a9cb`, namespaced in
`heavy_motion_core/`. GAP/Bezier/Pure Pursuit fields are display annotations,
never controller inputs. Dynamics/config remain byte-identical to CODEX;
dt=0.04 and displayed 1x=2.4 simulation seconds/wall second are unchanged.

The MAIN `EnvRenderer` class retains MAIN source except the V2-only second
path stroke uses the same sky-blue color/width as the first. `engine_3d.py`
is byte-identical to MAIN. Default overlays, colors, widths, buttons, HUD,
GAPS toggle, Bezier helper and Pure Pursuit display calculation use MAIN.
Extra predicted-path/safe-portal/A* overlays are hidden. Following the user's
latest correction, wall-buoy GAP annotations and yellow wall LiDAR hits are
removed. CODEX's internal wall safety is retained. Display-only LiDAR/grid
copies match MAIN's buoy-only sensor path; originals are restored after draw.

GUI GAP candidates retain MAIN's buoy-only clustering and forward pair
population/style. Selection is now entirely route-first: the completed CODEX
physical prediction is converted to an interpolating, piecewise cubic Bezier
representation. Controls deviate at most 0.25 px from the linear rollout
segments; controller states and rollouts are untouched. `heavy_gap_annotation.py`
finds finite segment crossings in displayed route arc order and validates
orientation-dependent projected hull intervals plus the unchanged exact polygon
clearance against observed obstacles and walls. Only true opposite-side
traversals qualify; tangent touches, extended-line hits and gates spanning an
intervening observed obstacle are rejected. Duplicate observed pairs and nearby
shared-boundary passage events are consolidated using obstacle/hull geometry.

No `find_gap`, MAIN weight/goal/width score, or raw A* ranking fallback is used
for GAP1/GAP2. MAIN's front-pair population is kept only for candidate glyphs;
selection considers all local buoy-pair geometry overlapping the completed
route's bounding box. During reverse recovery, route-forward arc progress is
used rather than the bow-facing direction. Ordering is deterministic: route arc, crossing clearance, tangent
change, then identity. Width is feasibility only. Missing GAPs remain `None`;
this prototype does not extrapolate beyond the completed prediction horizon.
Selected intersections are inserted into the actual displayed arrays. With one
GAP, the display stops at GAP1; with two, it splits at GAP1 and stops at GAP2.
A goal encountered earlier clips the display and removes post-goal annotations.
If no GAP is selected, the existing full prediction display is retained. The
control rollout is never clipped. Both strokes use MAIN sky-blue (50,210,255),
width 4; both derive from one interpolating route without waypoint regeneration.
Only the current MAIN-style Pure Pursuit marker is drawn on the joined displayed
route; the future pursuit marker is always None.

GAP2 skips intersecting/overlapping GAP segments and insufficiently separated
passage events. Arc separation uses hull extent, free width, speed times actuator
plus controller response time, and accumulated route curvature; no fixed pixel
threshold or score ranking is used. If no suitable next passage exists, GAP2
remains absent. After selection, `heavy_gap_diagnostics.py` computes the legacy
MAIN Align/Heading/Forward/Width/Clear/Perpend factors at the dynamic intersection.
The existing bottom-right panel layout is unchanged. Diagnostics are output
only; tests verify changing weights cannot affect selected GAPs or paths.

GUI-semantics verification: 72 targeted tests passed, including clipping at
GAP1/GAP2/goal, no-GAP preservation, single pursuit, geometry-based separation,
X/overlap rejection, diagnostic agreement with MAIN, score independence and
control/display isolation. Seeds 2000/2069/2081 completed at 776/888/922 steps,
with every recorded state, applied command and clearance identical to the prior
V2.1 checkpoint; map hashes matched. Results:
`data/main_heavy/v21_gui_semantics_exact_parity.json`.

Real X11 MAIN-style 2D 4x: the initial uninterrupted 22.199 s of the fresh run
measured 115.876 average FPS, 104 minimum 1-second rolling FPS, 240.013 steps/s
(requested 240), and 9.60053 simulation seconds/wall second. Median/p95/p99/max
frame times were 8.629/10.700/11.508/48.825 ms; >=16.7/33/50/100 ms counts
were 10/3/0/0. Backlog max/final was 0.41955/0.01021 simulation seconds and
was nonaccumulating. Later user pause toggles are retained in the full frame
log and excluded from the continuous 4x throughput interval; the entire 30 s
run is not represented as uninterrupted playback. Stage means over the full
run were controller 6.377 ms, annotation 1.521 ms, renderer 5.790 ms.
Profile/frame logs: `data/main_heavy/v21_gui_semantics_4x_gui_profile.json`
and `v21_gui_semantics_4x_gui_frames.json`. The paused full-run summary and
an earlier interrupted run are preserved separately. Actual GUI screenshot:
`data/main_heavy/v21_gui_semantics_gui_2069.png`; inspected sky-blue continuous
path, waypoint intersections, single current pursuit and populated diagnostics.
No large regression, controller tuning, commit or push. Next action: user GUI
inspection; no additional tuning before feedback. Earlier measurements below
are historical checkpoints, not the latest GUI-semantics measurement.

The post-checkpoint 3D buffer/nogil/wake-birth/lifetime/forecast-only environment
changes were removed. MAIN wake aging is restored; episode reset additionally
clears reflected wakes, wake/path/trail surfaces, GAPs, Beziers, pursuit markers,
predictions, sensor display caches and control generation state.

**2D scheduling decision:** synchronous control plus unchanged MAIN drawing
measured only ~30 FPS and ~232 steps/s, with accumulated debt. Consequently a
bounded, separate control process is necessary for the 2D acceptance too.
`heavy_motion_worker.py` now uses a clean `spawn` process, with no SDL/3D/wake
work. It computes the unmodified CODEX commands; the main process applies every
command through authoritative `BoatEnv.step` exactly once and verifies exact
state equality. Episode epochs invalidate queued old commands without blocking
the renderer. No physics/controller cadence or quality is reduced. An inline
comparison remains available through `MAIN_HEAVY_SYNC_MOTION=1`.

Representative motion sanity against saved CODEX logs:

| Seed | Result | Arrival steps | Simulation completion | Per-step position/heading/speed/yaw difference |
| --- | --- | ---: | ---: | ---: |
| 2000 | success | 776 | 31.04 s | 0 |
| 2069 | success | 888 | 35.52 s | 0 |
| 2081 | success | 922 | 36.88 s | 0 |

All map hashes, surge, desired speed, steering and actual hull-clearance traces
also match. Applied left/right PWM matches the earlier motion checkpoint
exactly; original CODEX diagnostic PWM was sampled after integration, whereas
applied commands are sampled before integration. Raw traces and the sampling
note remain in `data/main_heavy/v21_codex_parity_summary.json`.

Real X11 MAIN-style **2D 4x**, production 320x220 3D dashboard panel retained,
30.002 s run cycling representative maps (fullscreen 3D is not this gate):

- Requested/actual physics: 240 / 239.980 steps/s.
- Simulation progression: 9.59921 s/wall s.
- Average / minimum 1-second rolling FPS: 117.40 / 99.
- Median / p95 / p99 / max frame: 8.371 / 10.685 / 11.637 / 55.797 ms.
- Frames >=16.7/33/50/100 ms: 12 / 2 / 1 / 0.
- Maximum/final debt: 0.42312 / 0.02469 simulation s; no persistent accumulation.
- Controller / cached annotation / render mean: 6.247 / 1.276 / 6.143 ms.

53 relevant tests pass, including source pinning, MAIN renderer identity,
MAIN-compatible candidate population, buoy-only GUI LiDAR, display/control
isolation, dynamic GAP1/GAP2 endpoints, exact GUI-cluster/reference equality,
numerical worker/environment equality and full episode visual reset.
Screenshot: `data/main_heavy/v21_gui_2069.png`; profile/frame logs:
`v21_4x_gui_profile.json` / `v21_4x_gui_frames.json`.
Pre-cleanup dirty sources/patch are preserved under
`data/main_heavy/v21_pre_cleanup/`. Historical experimental results below are
not the current default candidate. No next tuning until user inspection.

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

## Phase 5 momentum GAP fast iteration — experimental

Phase 5 remains opt-in with `MAIN_HEAVY_MOMENTUM_GAP=1`; the default MAIN
GAP/Bezier/Pure Pursuit path and `main`/`codex` branches are unchanged.
The controller uses LiDAR-fitted moving buoys, two-stage dynamics rollouts,
the oriented three-part hull, and the existing 0.20 m hard margin. Known
arena walls now enter both the Phase 5 collision outcome and every preview
midpose/endpose. A terminal hull-vertex velocity check compares wall room
over actuator lag plus yaw response; it only filters candidates when at least
one safer continuation exists. Vessel parameters, dt, and playback are intact.

The Phase 5 GUI draws the selected physical rollout from the current boat
position, the selected portal, its buoy/wall-safe interval, and a predicted
crossing marker in the 2D world and minimap. The legacy Bezier and midpoint
markers are not used as the Phase 5 control line. Portal candidates are
generated from the observed cluster map and ordered by broken-line route
length after geometric width, direction, and wall checks. The original GAP
weight product remains for other modes and as a diagnostic fallback only.
The default Phase 5 online route does not use `env.dynamic_obstacles` as a
planner input. The inexpensive 16-step route-ordering calculation has an
equivalent compiled implementation; seed 2126's outcome and recorded motion
metrics matched before/after exactly.

The earlier Phase 5 partial development run was intentionally interrupted at
97/200 (97 successes) before the wall and portal changes. It is not a final
200-map result. A first geometric portal variant timed out on seed 2094;
terminal wall turning-room screening rescued it. Rejecting an entire wide
portal when a third observed buoy intersected part of it caused large
steering regressions on 2000 and 2069, so that filter was removed.
The buoy/wall-safe portal interval remained without changing the 25-seed
gate outcomes. The initial geometric candidate passed the fixed 25-seed
small gate, including the five difficult seeds, five regression seeds, ten
normal seeds, and five straight seeds. It then timed out on the medium gate's
seed 2126 after 27/50 rows. Its active portal had negative predicted progress
for several seconds while the hull moved toward the lower wall. Waiting for
the vessel to nearly stop before switching corridors was too late.

The current candidate changes portal when the selected safe rollout has no
progress for one physical yaw-response interval and a different observed
portal is available. Seed 2126 changed from timeout at 140 s and 1194
steering reversals to success at 38.76 s and 51 reversals. The current
25-seed quick gate passed 25/25, collision 0, timeout 0; neighboring seeds
2124-2128 passed 5/5. On five straight seeds, straight-segment reversals
fell from 43 to 11, straight command variation from 23.09 to 8.49, and
mean straight yaw RMS from 0.212 to 0.151 rad/s against the saved Phase 5
baseline. Results are under `data/main_heavy/phase5_fast_early_handoff_*`.

The new 2100-2149 medium gate is the next validation step and must complete
before any 200-seed development run. The evaluator now supports
`--stop-on-failure` and JSONL prefix resume, and the category quick gate is
`experiments/evaluate_phase5_quick.py`. The real 2D GUI line should be
visually checked after the medium gate; final 4x GUI performance is not yet
validated. Do not treat Phase 5 as production or open the 2200-2299 holdout
until the development gate is satisfied. No commit or push was made.

## Phase 5 V1 — user GUI inspection checkpoint (2026-10-06)

The latest user instruction supersedes the fast-iteration gate ladder above.
Do not run medium/development/holdout/1000-map evaluations before the user
approves the motion. V1 is an experimental GUI candidate, not an adopted
zero-collision navigation result. Stop here and await GUI feedback.

Reference source: `codex` at `60f2cf46500354a3e0690b9b62888e0dd166a9cb`.
Its `vessel_config.json` and `vessel_dynamics.py` match MAIN_HEAVY exactly:
20 kg mass, 3.8 kg m2 yaw inertia, 0.65 s yaw response, 0.25 s actuator lag,
dt 0.04 s, playback 2.4. These files and parameters were not changed.
The reference was exported into an isolated temporary directory for five
small motion comparisons; neither `main` nor `codex` was edited.

CODEX's relevant differences are continuous yaw hypotheses, a 0.1 rad/s
command increment per 0.12 s knot, near-ETA-tie command-change quality,
forward feasibility before reverse, exact oriented hull/wall safety, and a
longer 4.8 s preview. MAIN_HEAVY's coarse yaw commands and family-only tie
preference could switch from left to right abruptly. V1 adds trim/release
motions, command-change quality inside the existing progress tie band, and
the equivalent 0.0333 rad/s yaw-command increment per 0.04 s. The same ramp
is simulated in the safety preview and applied to the real first command.
The preview remains 1.6 s; CODEX-equivalent anticipation is not established.

Normal candidates contain forward, reduced-speed, release, pivot, and brake
motions. Reverse is a separate space-making library and is considered only
when no verified forward/turn motion or margin-improving forward escape
exists. A reverse rollout must improve terminal surface clearance by the
existing margin and create corresponding space. It ends with braking; a
previous negative speed command is not inherited as normal cruise. A safe
forward motion immediately excludes recovery reverse.

All four known walls produce actual ray intersections and yellow hit points.
Wall hits are kept out of buoy DBSCAN/fitting; known wall geometry is handled
separately. Observed buoy-to-wall portals use a zero-radius wall endpoint,
actual buoy radius, hull width, margin, and oriented wall clipping. Crossing
is measured from a selected dynamics rollout, never forced to midpoint.
The wall gate matches CODEX's oriented vertices plus its existing independent
axis-aligned center guard. Both controllers use a 0.20 m predicted margin.
This is a model-based eligibility threshold, not proof of 0.20 m separation
from the true moving buoy in every run.

GUI: selected portal and safe interval are green, predicted crossing yellow,
physical selected trajectory cyan, fitted piecewise Bezier reference violet,
and display-only Pure Pursuit reference point pink. The Bezier is fitted
AFTER physical selection; it is not a second safety authority. The gauge
uses `PP REF` rather than an ordering point as a steering target. The portal
geometry and crossing come from the same selected result snapshot. No A*
was added. Actual X11 rendering was exercised; screenshot:
`data/main_heavy/v1_gui_2069.png` (path, portal, Bezier panel, pursuit marker,
and wall hits present).

| Seed | MAIN_HEAVY V1 sanity | CODEX reference | V1/CODEX straight command reversals | V1/CODEX reverse duration |
| --- | --- | --- | --- | --- |
| 2000 | success 43.88 s | success 31.04 s | 0 / 4 | 0 / 0 s |
| 2069 | success 47.48 s | success 35.52 s | 1 / 2 | 0 / 0 s |
| 2081 | success 54.08 s | success 36.88 s | 1 / 3 | 0 / 0 s |
| 2003 | collision 31.36 s | success 27.00 s | 0 / 5 | 0 / 0 s |
| 2004 | no completion at 100 s sanity cutoff | success 37.68 s | 0 / 3 | 0 / 0.36 s |

All five paired map hashes match. Straight metrics use the same offline
heading/front-clear/speed mask, not arbitrary hand-selected intervals.
V1 has fewer straight reversals and smaller yaw-command TV, but the slower
completion, seed 2003 contact, and seed 2004 stagnation remain explicit
limitations. In successful V1 runs, recorded true minimum hull clearance was
0.165 / 0.170 / 0.136 m. No claim of zero collisions or full CODEX motion
parity is made. As requested, the candidate is retained for user inspection
rather than discarded or tuned further after this small sanity.

Raw state/command/clearance traces: `data/main_heavy/v1_inspection_*` and
`v1_codex_*.trace.json`. Shared metric summary and trajectory/heading/yaw
command/speed plot: `v1_codex_motion_comparison.json` and `.png`.
`experiments/summarize_phase5_v1.py` reads these files without running seeds.
Earlier `v1_motion_*` and `v1_ready_*` files are intermediate pre-ramp
observations, not this final V1 result. Relevant unit tests: 32 passed;
syntax and diff whitespace checks passed. No performance acceptance claim.

Next action: user runs `MAIN_HEAVY_MOMENTUM_GAP=1 python3 main.py` and assesses
straight stability, turn timing, reverse use, wall clearance, and free portal
crossing. Wait for feedback before V2 or larger validation. Dirty Phase 5
work and unrelated untracked leaderboard scripts are preserved. No commit,
push, branch switch, or production branch modification.
