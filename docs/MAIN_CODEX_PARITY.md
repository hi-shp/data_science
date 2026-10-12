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
