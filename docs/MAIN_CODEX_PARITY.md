# MAIN / CODEX control and goal display investigation

Reference: origin/codex `1688a6ba11b88dbfe626a253d7aa846ca4f22217`.
MAIN baseline: `6a90177b55cb56fca0d884b30797ed7ad4743b36`.
Both production defaults use `eta_continuity_forward`, dt 0.04 s,
planning every three physics ticks and 2.4 simulation s/wall s at displayed 1x.
Physics configuration and dynamics source are byte-identical.

## Control evidence

The frozen MAIN motion core and current CODEX differ in vectorized noise,
passage sort preparation, compiled yaw slew, unused ETA objective elimination,
and perception array dispatch. No controller synchronization was necessary.
With identical initial maps/states/observations/RNG, all generated rollout
command matrices, rollout states, clearances and selected indices were exact
matches. Every applied speed/yaw/PWM and eight-component physical state matched.

| Seed | Steps in both branches | Outcome in both |
| --- | ---: | --- |
| 2000 | 776 | success |
| 2069 | 888 | success |
| 2081 | 922 | success |
| 2003 | 675 | success |
| 2004 | 942 | success |
| 2026 | 701 | success |
| 3000 | 675 | success |

Direct compiled forward/recovery probe calls were additionally intercepted
for seeds 2000/2069; their candidate arrays and results also match exactly.

No first divergent physics frame; maximum state/command difference zero.
Candidate quality costs differed by at most 2.665e-15, below the 1e-12
absolute/relative numeric gate. Frame 1's complete quality difference was
reproduced exactly by changing only the identical sequence array's strides:
MAIN's C-contiguous noise buffer versus CODEX's vectorized allocation layout
changes NumPy reduction order. ETA, feasibility and selected commands did not
change. This is numeric cost roundoff, not a measured motion divergence.

One boundary discrepancy was corrected: MAIN V2 accepted distance <=70 px;
CODEX accepts distance <1.4 m, i.e. <70 px. The sampled episodes did not hit
exactly 70 px. No arrival threshold or physical parameter was retuned.

## Display cause and correction

A* endpoints remain at the goal. The GUI was instead clipping a finite rollout
at retained GAPs, or retaining its finite endpoint when the prediction did not
enter the 1.4 m display circle. Thus the visible endpoint could be unrelated to
the stable A* goal. For example seed 2069, frame 768: goal (1740,322), boat
(1512.761,86.826), finite endpoint (1733.270,244.389), no selected GAP.
The physical prediction is identical in CODEX. MAIN's old display also used
1.4 m for center connection whereas CODEX's prediction/display capture uses
0.55 m. The mission's separate 1.4 m completion boundary remains intact.

A separate terminal guidance array now retains a safe physical prefix, removes
an overshooting/retreating terminal branch, and connects exactly to goal. It is
used only in the existing 12 m terminal zone, when the full connector hull
corridor lies within the current LiDAR range and is clear of perceived circles
and all four walls. A continuous enclosing-hull-circle segment check includes
the existing safety margin and Bezier's shared subpixel deviation reserve.
It is conservative; an uncertain/blocked connection retains the finite future.
Even the short 0.55 m center connector is independently safety-gated.

No physical state, prediction array, controller sequence, cost, collision check,
or navigation map is changed. GAP identity/visit/persistence, MAIN candidate
population, exact displayed crossings, Bezier styling and pursuit interpolation
remain separate. Goal guidance can extend beyond a finite prediction; it must
not be interpreted as an additional dynamics rollout. Existing GAP clipping
still takes precedence while an earlier valid selected passage remains.

## Regression and performance

186 repository tests passed, plus syntax and whitespace checks. New cases cover
finite/overshooting/oblique/short/degenerate predictions, blocked connectors,
wall envelopes, sensor range, regeneration, earlier GAPs, and unsafe center
connections. Existing GAP completion/persistence/reset and renderer tests pass.
All seven recorded approaches end at exact goal throughout the final 5 m zone.
A real X11 seed 2069 goal screenshot was inspected: direct guide ends at center,
no stale selected GAP, existing pursuit and LiDAR/Bezier panels remain visible.

Actual production X11 workers, seed 2000, 1x/2x/4x: all 776 physical records and
prediction hashes match CODEX and each other. Packets apply once and in order;
worker-authoritative state assertions remain enabled. These are complete first
episodes, not synthetic render-rate checks or headless performance results.

Real X11 2D, seed 2069, displayed 4x, 12 s per run:

| Metric | Before | After |
| --- | ---: | ---: |
| Average FPS | 99.90 | 105.51 |
| Minimum rolling 1 s FPS | 71 | 93 |
| p95 frame ms | 12.78 | 11.38 |
| p99 frame ms | 25.10 | 13.26 |
| Frames >=100 ms | 0 | 0 |
| Physics steps/s (240 requested) | 239.72 | 240.08 |
| Maximum backlog, physics steps | 26.69 | 11.96 |
| Final backlog, physics steps | 0.81 | 0.29 |

No persistent debt or observed performance regression. These short runs have
OS/run-to-run variability; FPS differences are not a claimed speed improvement.
Final worker median planning time was 6.83 ms; deterministic seed 2069 median
planning was 6.35 ms before and 6.46 ms after (minor timing variability).
One invalid instrumented run saved a PNG after every episode reset and is
explicitly excluded (`gui_after4.json`); the instrumentation was corrected.

## Reproduction and artifacts

Numerical regression (SDL dummy for numerical evidence only):

```bash
mkdir -p scratch/codex_reference
git archive origin/codex | tar -x -C scratch/codex_reference
python3 experiments/regression/compare_motion_branches.py \
  --reference scratch/codex_reference --output scratch/motion_parity \
  --seeds 2000 2069 2081 2003 2004 2026 3000
```

Actual GUI worker/cadence regression:

```bash
python3 experiments/regression/measure_motion_gui.py \
  --branch main --rate 4 --seed 2000 --episode --output scratch/main_gui.json
python3 experiments/regression/measure_motion_gui.py \
  --branch codex --root scratch/codex_reference --rate 4 --seed 2000 \
  --episode --output scratch/codex_gui.json
```

The comparison script fails on initial/control/state/candidate/safety mismatch,
reports the first differing frame and accepts only stated numeric cost tolerance.
`--detail-frame N` saves the complete candidate arrays for diagnosis.
Local evidence: `scratch/parity_goal/{before,extra_before,final}/`, per-step NPY,
per-plan observation/RNG/candidate hashes/costs and pixel endpoint logs;
`gui_comparison.json`, `gui_before.json`, `gui_final4.json`, and
`goal_inspection_goal.png`. Raw traces remain generated artifacts, not runtime
inputs or leaderboard records. CODEX source and branch are unchanged.

Limits: seven fixed initial maps and one GUI seed at three playback rates;
not exhaustive equivalence over all possible observations/maps. No 200/1000
benchmark was run. Next action: inspect goal approaches with `python3 main.py`;
report any remaining specific seed/scene before further controller work.


## Render-frame follow-up: transient terminal jumps (2026-10-12)

This follow-up supersedes the earlier endpoint-completion claim above.
Resumed the dirty changes and existing `scratch/goal_frames/` diagnostics;
no baseline reconstruction, tuning or large benchmark was required.
MAIN reference before this follow-up: `7107fbc`; CODEX: `1688a6b`.

### Confirmed failures

Seed 2081, MAIN physics frame 822: display endpoint at (1740,322).
At 825 a newly confirmed last GAP became the display endpoint at
(1509.954,416.458), a roughly 249 px retreat. The physical future still came
near the goal; GAP clipping, not control, produced this large presentation
switch. Seed 2026 repeated the pattern at frames 601 then 605. Seed 2069
showed large finite-prediction/LiDAR endpoint changes on both branches.

The former short .55 m connector could bypass the terminal guide's sensor/hull
coverage gate. Conversely, finite prediction display remained active when the
safe guide was unavailable. Those modes could exchange ownership of the
visible endpoint with a pending GAP. CODEX additionally converted its float32
goal into SI before connecting: reconstructed pixel endpoint error about
3.825e-5 px. This tiny precision error is distinct from the large GAP jump.

### Final display behavior

Both branches now use the same pure `goal_display_policy.py` geometry/state
principle. MAIN keeps its GAP/Bézier/pursuit structure; CODEX keeps its prediction
UI. No GAP hierarchy or renderer style was copied across branches.

A finite predicted mission arrival arms terminal presentation. This state
controls only clipping ownership: reappearing valid GAP markers remain on the
route, but do not cut a terminal path back to their waypoint. Each proposed
goal connection is independently revalidated against current perceived circles,
all walls, the enclosing actual-hull sweep, safety margin and sensor coverage.
Bezier's existing subpixel reserve is included. A curved selected prefix may
be retained only when its full current displayed corridor and final connector
pass, with its physical predicted clearance also valid. No old safe connection
is treated as safe after a new obstruction or observation uncertainty.

When coverage is insufficient before connection, the finite future is capped
at the first mission approach and sensor-supported portion; its post-goal tail
is removed. After a certified connection, a failed recheck suppresses guidance
instead of flashing a corner-directed finite tail. Safety failure always wins;
there is no safety hysteresis. Subsequent safe updates may reconnect. Physical
rollout, candidate arrays, safety gates, control path and commands remain exact.

MAIN refreshes terminal guidance at render cadence after selected passages
clear, without clustering or planning at render cadence. CODEX already refreshed
its prediction at render cadence; the independent guide uses the same hook.
Both reset terminal state between episodes. Goal assignment uses the original
world coordinate in float64 at the display boundary only. Mission completion,
0.55 m prediction capture, and 1.4 m display-terminal arming remain distinct.

### Focused continuous GUI evidence

Real X11 2D, seed 2069 at 1x/2x/4x on both branches, plus seeds 2081 (1x),
2026 (4x), and 2000 (4x): 12 complete first-episode runs. From within 12 m
until the final pre-reset render, every frame's world path and endpoint were
logged and its screen captured, totaling 2,792 consecutive render records.
MP4 frame index maps one-to-one to the corresponding trace render index.
The capture video's nominal 120 fps is an inspection index, not a claim that
the actual GUI ran at 120 fps; timestamps/performance are recorded separately.

549 certified-connected renders pass goal tolerance atol=1e-6, rtol=0;
maximum reconstructed pixel error is 2.274e-13 (MAIN zero; CODEX float64 unit
conversion roundoff). Unexpected goal-to-finite switches: zero. Both branches
connect only after current safety/support certification and then retain center
through the remaining approach in these runs. Boundary/uncertain transitions
are additionally tested by deterministic multi-update fixtures.

At MAIN seed 2081 frames 822 and 825 the final endpoints are now
(1690.161,354.542) and (1691.603,347.197): 7.485 px change instead of roughly
249 px. GAP pair (49,53) remains visible at 825. These two frames are outside
full hull-supported LiDAR coverage and have a blocked conservative direct
corridor; they are deliberately NOT called safe goal guidance. Exact goal
connection occurs at frame 879 after certification. This is removal of the
false connection/cut-back transition, not forced goal painting at frame 822.
Seed 2026 likewise has a single certified transition, with no GAP-induced
return; seed 2069 is stable after connection at all three playback rates.

The continuous captures and extracted before/after/goal screenshots were
inspected. Existing GAP markers and MAIN Bézier/pursuit/LiDAR panels and CODEX
prediction panels remain present. Legitimate obstacle-avoidance prediction
changes before terminal arming remain visible; this ticket does not hide all
finite-horizon replanning changes throughout the mission.

### Control, tests and performance

Seven seeds (2000,2069,2081,2003,2004,2026,3000), 5,579 physics steps:
exact selected commands, PWM, eight-component physical state and candidate
rollouts still match. Quality cost roundoff remains <=2.665e-15; selected
indices/safety results agree. Actual X11 step traces before/after and between
branches also match at 1x/2x/4x, including the two GAP-reappearance seeds.

MAIN: 196 tests pass. CODEX: 96 tests and 18 subtests pass. Existing display,
GAP, persistence, latch, reset and control-isolation tests are preserved.
New cases cover repeated connection, one-update range uncertainty, actual new
blockers, stale-safe rejection, curved approaches, finite overshoot, terminal
GAP reappearance, exact world goal precision and invalid prediction suppression.

Real X11 2D 4x, seed 2069, 12 s, capture/trace overhead disabled:

| Metric | MAIN before | MAIN after | CODEX before | CODEX after |
| --- | ---: | ---: | ---: | ---: |
| Average FPS | 105.33 | 105.18 | 121.76 | 121.26 |
| p95 frame ms | 11.40 | 11.54 | 9.21 | 9.37 |
| p99 frame ms | 12.72 | 12.89 | 10.42 | 11.01 |
| Physics steps/s (240 requested) | 240.13 | 240.01 | 240.35 | 240.28 |
| Worker planning median ms | 6.74 | 6.74 | 5.30 | 5.51 |
| Maximum scheduling backlog, steps | 49.01 | 11.26 | 99.31 | 14.11 |
| Final scheduling backlog, steps | 0.22 | 0.76 | 0.70 | 0.64 |
| Frames >=100 ms | 0 | 0 | 0 | 0 |

No significant pacing regression or persistent physics debt is observed.
Short-run planning/backlog variation includes episode resets/worker startup;
maximum debt is transient and is not reported as sustained throughput loss.
Capture runs have higher frame costs and are not substituted for these timings.
Physics parameters, controller, candidate evaluation and collision safety are
unchanged. No 200/1000-episode benchmark was run.

### Reproduction and preserved artifacts

Use the existing branch comparison command above. For per-render evidence:

```bash
python3 experiments/regression/measure_motion_gui.py --branch main \
  --seed 2069 --rate 1 --episode --trace-display --capture \
  --output scratch/render_check/main_1.json
python3 experiments/regression/measure_motion_gui.py --branch codex \
  --root scratch/codex_reference --seed 2069 --rate 1 --episode \
  --trace-display --capture --output scratch/render_check/codex_1.json
python3 experiments/regression/analyze_goal_frames.py \
  scratch/render_check/*.display.json --output scratch/render_check/check.json
```

Generated artifacts remain under `scratch/goal_frames/`: `before_*` and
`final_*` display JSON/MP4, per-step state/PWM arrays and physical prediction
hashes; `final_render_analysis.json`, `validation_summary.json`,
`resumed_parity/comparison.json`, `perf_baseline_*`, `perf_final_*`,
`inspect_after_822.png`, `inspect_after_825.png`, and the two branch goal PNGs.
Numeric traces include full paths for reproducing individual frame transitions;
these are diagnostics, never runtime inputs. CODEX validation used an exported
reference tree before the minimal display changes were applied to its branch.

Limits: fixed representative seeds, finite X11 observation windows and synthetic
unsafe-transition fixtures, not proof for every map. If the GUI still shows an
unexpected branch, supply that seed/render sequence for the next focused ticket;
do not retune the controller from a presentation symptom.
