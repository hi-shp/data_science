"""Short, actual-command collision preview for the MAIN gap follower.

This does not choose a gap or alter the selected Bezier. Its only possible
intervention is a brief differential-thrust command while the existing route
and perception/planning loop continue to run.
"""

import math

import numpy as np

from main_safety_kernels import packed_hulls, preview_rollout, warm_preview_kernels


HORIZON_STEPS = 20  # 0.8 s at MAIN's unchanged 0.04 s physics step.
CHECK_PERIOD_STEPS = 3
RISK_DISTANCE = 60.0  # All six baseline contacts crossed this sensor range.
EMPTY_PATH = np.empty((0, 2), dtype=np.float32)


class MainSafetyGuard:
    def __init__(self):
        warm_preview_kernels()
        self.hold_steps = 0
        self.turn = 0.0
        self.last_frame = -1
        self.predictions = 0
        self.alarms = 0
        self.interventions = 0
        self._packed_hulls = None

    def _preview(self, env, left, right, override=None):
        x, y = float(env.boat_pos[0]), float(env.boat_pos[1])
        vx, vy = float(env.boat_vel[0]), float(env.boat_vel[1])
        heading = float(env.boat_heading)
        yaw = float(env.boat_ang_vel)
        forward = float(getattr(env, 'current_fwd', 0.0))
        obstacles = env.dynamic_obstacles
        path = env.bezier_path if env.bezier_path is not None else EMPTY_PATH
        params = env.params
        polygons, lengths = self._packed_hulls

        # A speed upper bound plus hull/obstacle extents makes this conservative
        # for the whole preview. Only the small near subset enters exact checks.
        max_radius = float(np.max(obstacles[:, 2])) if len(obstacles) else 0.0
        near_bound = max(100.0, math.hypot(vx, vy)) * HORIZON_STEPS * env.dt + 45.0 + max_radius + 5.0
        delta = obstacles[:, :2] - env.boat_pos
        close = np.sum(delta * delta, axis=1) <= near_bound * near_bound
        nearby = obstacles[close]
        lidar_bound = env.lidar_range + max_radius + near_bound
        lidar_obstacles = obstacles[np.sum(delta * delta, axis=1) <= lidar_bound * lidar_bound]

        hit, end_x, end_y = preview_rollout(
            x, y, vx, vy, heading, yaw, forward, left, right,
            float(env.min_wide_dist), float(env.prev_steer),
            bool(env.emergency_mode), int(getattr(env, 'emergency_cooldown', 0)),
            path, env.current_wp is not None, nearby, lidar_obstacles,
            polygons, lengths, env.rel_angles, float(env.dt),
            float(env.mass), float(env.drag), float(env.rot_drag),
            float(env.inertia), float(params['mom_coeff']),
            float(env.lidar_range), float(params['steer_gain']),
            float(params['steer_alpha']), float(params['avoid_normal']),
            float(params['avoid_em']), float(params['em_enter']),
            float(params['em_exit']), int(params['em_hold_frames']),
            float(params['pwm_rng']),
            0.0 if override is None else float(override[0]),
            0 if override is None else int(override[1]), HORIZON_STEPS,
        )
        return (hit if hit else None), (end_x, end_y)

    def command(self, env, left, right, steer):
        if self._packed_hulls is None:
            self._packed_hulls = packed_hulls(
                (env.left_hull_local, env.right_hull_local, env.deck_local)
            )
        if env.frame < self.last_frame:
            self.hold_steps = 0
        self.last_frame = env.frame
        if env.manual_mode or env.linetrace_mode:
            return left, right

        if self.hold_steps == 0 and env.frame % CHECK_PERIOD_STEPS == 0 \
                and float(getattr(env, 'min_wide_dist', 999.0)) < RISK_DISTANCE:
            self.predictions += 1
            hit, _ = self._preview(env, left, right)
            if hit is not None:
                self.alarms += 1
                options = []
                destination = env.current_wp['pos'] if env.current_wp is not None else env.target
                for turn in (-1.0, -0.5, 0.5, 1.0):
                    for hold in (5, 10, 20):
                        alternative_hit, endpoint = self._preview(env, left, right, (turn, hold))
                        if alternative_hit is None:
                            remaining = math.hypot(endpoint[0] - destination[0],
                                                   endpoint[1] - destination[1])
                            options.append((remaining + 3.0 * abs(turn - steer) + hold * 0.03,
                                            turn, hold))
                if options:
                    _, self.turn, self.hold_steps = min(options)
                    self.interventions += 1
        if self.hold_steps:
            self.hold_steps -= 1
            return env.get_pwm(self.turn)
        return left, right
