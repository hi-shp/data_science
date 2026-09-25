# RESUME_PROMPT.md

Copy the text below into Codex when resuming the interrupted USV optimization.

---

Continue the existing USV optimization from the current local working tree.

Do not restart the project analysis from scratch and do not scan the whole repository.

First read:
1. `AGENTS.md`
2. `docs/PROJECT_STATE.md`
3. `git status`
4. the current uncommitted diff for the files already changed in the interrupted session

The previous Codex session ended because the usage limit was reached during final validation.

The existing local changes reportedly include:
- `benchmark_navigation.py`
- `boat_control.py`
- `environment.py`
- `main.py`
- `report5/plot_validation.py`
- `route_planner.py`
- `test_vessel_dynamics.py`
- `ui_renderer.py`
- `vessel_dynamics.py`

Primary objective:
Finish validating and, only if necessary, minimally correcting the existing dynamics/control/navigation changes.

Preserve these design goals:
- rotational inertia remains roughly 2x the old nominal value;
- do not lower inertia merely to make avoidance easier;
- reduce unrealistic snap rotation and left-right steering oscillation;
- maintain high obstacle-avoidance success;
- use a physically consistent engineering model, but do not claim measured real-vessel fidelity without real measurements.

Previous reported results that must be treated as provisional until verified:
- baseline: 189/200 success and 11 collisions on one 200-case set;
- modified version: first 200/200 validation;
- average maximum angular acceleration: roughly 230 deg/s^2 -> 24 deg/s^2;
- steering direction reversals substantially reduced;
- the predicted hull envelope was then modified to cover the stern better;
- final validation after that last change was interrupted.

Do not immediately rerun every large benchmark.

Work in this order:
1. inspect the current diff and identify the exact final state left by the previous session;
2. verify the last stern/hull-envelope change;
3. run syntax/unit tests and selected representative or known-failure seeds;
4. if those pass, run only the minimum holdout validation required to verify the exact current code;
5. separate success, collision, and timeout;
6. compare current metrics to baseline under identical conditions;
7. inspect detailed traces only for failures;
8. update `docs/PROJECT_STATE.md` with verified results.

Do not do the following in this ticket:
- UI redesign;
- PPT/report generation;
- unrelated refactoring;
- broad perception changes;
- new features;
- parameter sweeps without a measured cause;
- repeated 200+ episode runs after trivial edits;
- commit or push unless I explicitly ask.

If you discover an unrelated improvement, record it as a follow-up rather than implementing it.

Completion report:
1. root cause / state found;
2. files actually changed in this resumed session;
3. validation commands and conditions;
4. baseline vs current numeric results;
5. any remaining failures or unverified items;
6. exact next recommended ticket.

---
