# Frozen vessel dynamics — 2026-09-26

Navigation experiments must not change these effective runtime values. Read from
`BoatEnv(headless=True).dynamics` after `configure_dynamics`, including any
`best_learned_params.json` overrides. Runtime values match `vessel_config.json`.
Dataclass constructor defaults alone are not the active configuration.

| Parameter | Value | Unit |
|---|---:|---|
| pixels_per_m | 50 | px/m |
| mass_kg | 20 | kg |
| yaw_inertia_kg_m2 | 3.8 | kg m² |
| surge_linear_drag | 6 | N/(m/s) |
| surge_quadratic_drag | 7.116009950310329 | N/(m/s)² |
| sway_linear_drag / quadratic_drag | 55 / 45 | N/(m/s), N/(m/s)² |
| yaw_linear_drag / quadratic_drag | 8 / 5 | N m/(rad/s), N m/(rad/s)² |
| thruster_arm_m | 0.22 | m |
| max_thrust_N | 25 per thruster, either direction | N |
| actuator_tau_s | 0.25 | s |
| cruise_speed_m_s | 1.5 | m/s |
| max_yaw_rate_rad_s | 0.5 | rad/s command bound |
| yaw_response_s | 0.65 | s, legacy fallback when independent gain absent |
| yaw_rate_gain_Nm_s | 11.76923076923077 | N m s, independent of inertia |
| speed_response_s | 1.5 | s |
| physics timestep | 0.04 | s |
| planning period | 3 physics steps | 0.12 s |
| displayed playback base | 2 | simulation seconds / wall second at displayed 1x |

No explicit physical speed clamp: full forward thrust balances linear plus
quadratic surge drag. Maximum differential torque is 11 N m. `allocate` preserves
yaw authority on thrust saturation; `integrate` includes actuator lag, body-frame
Coriolis terms and midpoint integration. This is an engineering approximation,
not a model identified from physical vessel measurements.

Fresh isolated measurements at dt=0.04 s, initially at rest:

- A commanded yaw-rate step 0 → 0.5 rad/s at zero commanded surge reaches 90% in
  1.04 s; peak yaw acceleration 39.7068345 deg/s².
- Both thrusters continuously at +25 N for 20 s: terminal speed
  2.2624689053 m/s; peak startup surge acceleration 1.8141634 m/s².
- These are isolated dynamics tests, not a 200-episode navigation validation.

The baseline navigation controller has horizon 5 s, prediction interval 0.2 s,
lookahead 3 m, safety margin 0.2 m, target wall margin 1.8 m, 21 yaw samples,
heading gain 0.65, cross-track weight 0.18, turning weight 0.22,
command-change weight 3, terminal-heading weight 0.4, goal-distance weight 0.2.
Navigation costs/architecture may change; the physical model, allocator gain,
command bounds and actual integration timestep above remain fixed.

## Reproduction and source identity

Baseline HEAD: `45e15e0` (pre-existing commit, not created by this ticket).
The pre-existing `leaderboard.json` working change is preserved.
Backup: `data/architecture_rethink/baseline-sgl_aqsn/`, containing source copies,
full source SHA256 manifest, Git status, HEAD and binary working-tree diff.
Runtime values and measurements: `data/architecture_rethink/frozen_dynamics.json`.

| Source | SHA256 before navigation work |
|---|---|
| vessel_config.json | 22c870bef4866ba41ed6425bec5fe5eace6f0a6518ab4c977a36d8f1b9b1eff9 |
| vessel_dynamics.py | 25e45a3287b30413858749bd4e2efb62fb2c5054bc683e3ed7f607b184256454 |
| environment.py | 02fb7dc181bd85e7dded351a8d8076785e59cc2055f2723eab3710f32251c8a9 |
| boat_control.py | 2ef58bbdd27b9d5690664f9b42c2f34d271a668a5015c13fa1f8e585c06ddd5c |
| main.py | b13fef6b09c9c6427e7cf06453e8cf14f4713d27901644f6507492e811eed481 |
| best_learned_params.json | 770fdecd11748feea60cca196715f121edb821f7eda0df6d58aa70b47e5b15a1 |

The table is a pre-navigation source manifest, not a claim that every file
still has that SHA256. The displayed-4x performance ticket changed `main.py`,
`environment.py` and `boat_control.py` implementation for scheduling, reset
sampling and equivalent predictive computation/display. It did not change the
runtime physical parameters above or `vessel_config.json`,
`vessel_dynamics.py`, or `best_learned_params.json`.
