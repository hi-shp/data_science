# PROJECT_STATE.md

Last updated: 2026-09-25

## Current objective

Continue the interrupted USV dynamics/navigation optimization without restarting repository-wide analysis.

Target behavior:

- keep vessel rotational inertia at roughly 2x the prior nominal level;
- avoid recovering success rate by weakening inertia;
- reduce unrealistic snap turns and left-right steering oscillation;
- retain high obstacle-avoidance success rate;
- keep the vessel model physically consistent, while clearly treating it as an engineering approximation rather than a validated real-vessel model.

## Confirmed repository baseline

At the start of the previous Codex handoff:

- branch: `main`;
- local and remote commit reported as `e9418fb`;
- working tree was initially clean;
- recent work focused on vectorization, asynchronous 3D processing, rendering cache, and display defaults;
- main execution: `python3 main.py`;
- success evaluation: `python3 test_success_rate.py 100` (or a larger episode count);
- planning/perception stack includes LiDAR, DBSCAN, gap selection, Bezier path generation, and Pure Pursuit.

## Previous Codex optimization session

The previous session edited or added these files in the local working tree:

- `benchmark_navigation.py`
- `boat_control.py`
- `environment.py`
- `main.py`
- `report5/plot_validation.py`
- `route_planner.py`
- `test_vessel_dynamics.py`
- `ui_renderer.py`
- `vessel_dynamics.py`

Reported diff size at interruption:

- 9 files;
- +624 / -17 lines.

These changes may exist only in the local working tree and may not yet be in GitHub. Verify with `git status` and `git diff`; do not assume the remote branch contains them.

## Previous session findings

Reported engineering findings:

1. the old implementation applied strong per-physics-step angular-velocity damping, approximately multiplying angular velocity by 0.84 each step;
2. there was also a rotation-related positional correction;
3. simply increasing inertia was therefore insufficient to produce natural turning;
4. the session reorganized damping/thrust/inertia toward a time-based model;
5. the controller/planner was adjusted to anticipate inertia and slow/turn earlier rather than rely on last-moment snap steering;
6. a deadlock/stall mode was found where the vessel could stop safely but lacked a good restart-direction decision;
7. realtime and evaluation loops appeared to use different planning/physics timing;
8. at one point the 1x realtime mode reportedly advanced physics by about 0.04 s per render frame, making simulated time depend on render FPS;
9. collision and timeout needed separate accounting;
10. the predicted hull envelope was later found to under-cover part of the stern and was modified near the end of the session.

## Reported intermediate results

Treat these as previous-session reports that must be verified from the local outputs/diff before final publication.

- baseline on one 200-episode set: 189/200 success, 11 collisions;
- tuning set of 24 cases: 24/24 complete, no collision;
- first separate 200-case validation after changes: 200/200 complete;
- average maximum angular acceleration reportedly decreased from about 230 deg/s^2 to about 24 deg/s^2;
- average left/right turning-direction changes reportedly decreased substantially;
- the final stern-envelope correction was made after the first 200/200 result;
- the final post-correction validation was still running when the usage limit was reached.

Do not claim the final result until the last stern-envelope version is validated.

## Current highest-priority task

Do not add features first.

1. inspect `git status` and the current uncommitted diff;
2. confirm that the interrupted-session files are present and internally consistent;
3. inspect the last hull/stern collision-envelope change;
4. run the cheapest relevant regression checks;
5. complete only the minimum validation needed after that last change;
6. compare final metrics to the recorded baseline under identical conditions;
7. record any failures by category;
8. update this file with verified results.

## Validation policy for the resumed task

Do not restart all expensive experiments automatically.

Recommended order:

1. syntax/unit tests;
2. known failure or representative seeds;
3. 20-30 episode smoke test if needed;
4. final 100-200 episode holdout only if the prior evidence is not already valid for the current exact code.

If a completed benchmark output already corresponds exactly to the current code, reuse it rather than re-running it.

## Explicit non-goals for the resumed task

Until the interrupted optimization is validated, do not:

- redesign the UI;
- create PPT material;
- generate a broad new report;
- perform unrelated refactors;
- change perception algorithms without evidence they cause the current failures;
- reduce inertia merely to regain maneuverability;
- run large benchmarks repeatedly after trivial edits;
- commit or push unless explicitly requested.

## Next exact action

Open the local project and continue from the existing working tree:

`git status -> git diff -> inspect relevant modified files -> verify last change -> minimal regression -> final holdout if needed -> concise result summary`
