# KABOAT / USV Codex Instructions

## Project purpose

This repository is a USV autonomous-navigation simulation based on LiDAR perception, DBSCAN obstacle clustering, gap selection, Bezier path generation, and Pure Pursuit tracking.

Primary engineering goals:

- improve obstacle-avoidance success rate without making the dynamics unrealistically easy;
- reduce unnecessary left-right steering oscillation;
- preserve physically plausible inertia, yaw response, thrust lag, and damping;
- keep simulation/evaluation conditions reproducible;
- generate trustworthy experimental data and presentation-ready visualizations from real runs.

Respect existing code, configuration, and uncommitted user work.

## Start-of-task rule

Do not re-read the whole repository on every task.

At the start of a task, inspect only:

1. `git status`;
2. `docs/PROJECT_STATE.md` if present;
3. the files directly relevant to the requested task;
4. recent diff/history only when needed to understand those files.

Do not scan `report*`, images, large generated outputs, or unrelated modules unless the task requires them.

Separate confirmed facts, experimental results, and assumptions.

## Relevant file map

Read only the subset required by the task.

- simulation loop / environment: `main.py`, `environment.py`
- perception / LiDAR / DBSCAN: `perception.py`
- navigation / gap selection / path following: `navigation.py`, `route_planner.py`, `utils.py`
- vessel dynamics: `vessel_dynamics.py`, `config.py`
- low-level control: `boat_control.py`
- evaluation: `benchmark_navigation.py`, `test_success_rate.py`, `test_vessel_dynamics.py`
- 2D visualization: `ui_renderer.py`
- 3D visualization: `engine_3d.py`
- presentation/report work: relevant `report*` directory and reporting rule only

Some files may exist only in the local working tree. Verify before assuming they are tracked.

## Scope control

Prefer the smallest change that can solve the measured problem.

Do not perform unrelated refactors, broad architecture rewrites, UI redesigns, or report work during an algorithm/dynamics ticket.

If a useful issue is discovered outside the current scope, record it as a follow-up instead of fixing it immediately.

For large requests, split the work into milestones with one main engineering objective per milestone.

## Physics and dynamics

Do not raise success rate by weakening the physical model.

In particular, do not casually reduce or bypass:

- yaw inertia;
- mass;
- thrust response lag;
- rotational damping;
- lateral damping;
- realistic actuator limits.

Treat dynamics parameters and navigation/controller parameters as separate categories.

When a dynamics parameter changes:

- state the physical meaning and unit;
- explain why it changes;
- compare the effect under identical evaluation conditions.

Do not claim that the model matches a real vessel unless supported by measurement data. Without measured vessel data, describe it as a physically consistent engineering approximation.

Avoid frame-rate-dependent physics constants when a time-based formulation is practical.

## Optimization procedure

Do not randomly tune many parameters at once.

Use this order:

1. reproduce the failure;
2. measure the likely cause;
3. change the smallest relevant component;
4. run a cheap targeted check;
5. inspect failure cases;
6. only then expand validation.

Prefer algorithmic or control fixes over hiding a navigation weakness by reducing inertia.

## Evaluation metrics

Success rate alone is not enough.

When relevant, track:

- success count/rate;
- collision count/rate;
- timeout count/rate;
- completion time;
- path length;
- minimum obstacle clearance;
- steering-direction reversals;
- heading oscillation;
- yaw rate;
- yaw acceleration;
- linear speed;
- control smoothness.

Collision and timeout must be reported separately.

## Benchmark efficiency

Do not run expensive large benchmarks after every small edit.

Default validation ladder:

1. syntax/unit tests;
2. known failure seeds or a few representative seeds;
3. 20-30 episode smoke benchmark;
4. 100-200 episode holdout validation only after improvement is established;
5. larger runs only for final evidence when needed.

Use fixed seeds and identical map/timestep/planning-frequency conditions for before/after comparisons whenever possible.

Do not repeatedly load huge raw logs into context. Prefer summary statistics and inspect detailed traces only for selected failures.

Keep tuning seeds separate from final holdout seeds.

## Timebase consistency

Be alert to differences among:

- physics timestep;
- render FPS;
- simulation speed multiplier;
- planning frequency;
- evaluation-loop frequency.

A controller that works only because evaluation and realtime execution advance physics differently is not considered validated.

## Long-running work and project memory

Use `docs/PROJECT_STATE.md` as the durable handoff record.

Keep it concise and update it at meaningful milestones with:

- current objective;
- completed milestone;
- changed files;
- important decisions and reasons;
- validation results;
- known issues;
- exact next action.

Do not re-scan the whole repository merely to rediscover information already recorded there. Verify only facts that may have changed.

For multi-stage work, follow `docs/CODEX_ROADMAP.md` when relevant.

## Git safety

Never discard or overwrite existing uncommitted user changes.

Do not use destructive commands such as `git reset --hard`, broad file deletion, or forced checkout without explicit user instruction.

Do not commit every experimental edit.

Commit only when:

- the user explicitly asks; or
- a coherent milestone is complete and validated, and the current task explicitly includes committing.

Push only when the user explicitly asks.

## Completion rule

Before declaring a ticket complete:

- verify the requested behavior;
- run the cheapest sufficient tests;
- report before/after metrics under comparable conditions;
- note anything not verified.

Final report order:

1. root cause;
2. changes;
3. validation and numbers;
4. remaining limitations;
5. exact next recommended ticket.

## Preserved KABOAT project conventions

The existing project's reporting conventions remain in force without expanding the scope of algorithm tickets:

- No emoji or Markdown asterisk emphasis in reports or Markdown documents; use headings, tables, or `<b>...</b>` for emphasis.
- Use a factual engineering tone without decorative or exaggerated claims.
- Navigation performance reports must include recorded vessel trajectory, heading, steering or differential-thrust command, and speed when those quantities support the analysis. Do not substitute conceptual diagrams for motion evidence or label a normalized command as a measured rudder angle.
- Verify text overlap, clipping, font rendering, axis labels, and contrast before delivering figures. Minimum sizes are 10 pt for annotations/body text, 11 pt for labels, and 13 pt for subplot titles.
- Follow `.agents/rules/user_guidelines.md` for report/visualization details.
- When a commit is explicitly authorized, retain the project's Conventional Commits convention (`feat`, `fix`, `docs`, `refactor`, etc.). The former automatic commit/push requirement is superseded by the Git safety rules above and the user's current no-commit/no-push instruction.

Guideline integration source: `origin/codex-guidelines` at `9d437d0`. Existing local project conventions were compared and preserved; simulation code was not replaced from that branch.
