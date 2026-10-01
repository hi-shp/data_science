"""Short, actual-command collision preview for the MAIN gap follower.

This does not choose a gap or alter the selected Bezier. Its only possible
intervention is a brief differential-thrust command while the existing route
and perception/planning loop continue to run.
"""

import math

import numpy as np

from main_safety_kernels import (packed_hulls, preview_hull_collides,
                                 preview_lidar_distances, preview_follow_steering,
                                 warm_preview_kernels)
from vessel_dynamics import allocate, integrate


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
        state = env.physics_state()
        p = env.dynamics
        scale = p.pixels_per_m
        vx, vy = float(env.boat_vel[0]), float(env.boat_vel[1])
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

        min_wide = float(env.min_wide_dist)
        previous_steer = float(env.prev_steer)
        emergency = bool(env.emergency_mode)
        cooldown = int(getattr(env, 'emergency_cooldown', 0))
        for index in range(HORIZON_STEPS):
            if index:
                distances = preview_lidar_distances(
                    state[0]*scale, state[1]*scale, state[2],
                    lidar_obstacles, float(env.lidar_range))
                steer, previous_steer, emergency, cooldown, min_wide = \
                    preview_follow_steering(
                        state[0]*scale, state[1]*scale, state[2], state[5],
                        previous_steer, emergency, cooldown, path, distances,
                        env.rel_angles, float(params['steer_gain']),
                        float(params['steer_alpha']), float(params['avoid_normal']),
                        float(params['avoid_em']), float(params['em_enter']),
                        float(params['em_exit']), int(params['em_hold_frames']),
                        env.current_wp is not None)
            if override is not None and index < override[1]:
                steer = override[0]
            elif not index:
                steer = 0.0
            if index or override is not None:
                speed_factor = math.tanh(max(0.0, min_wide)/100.0) ** 1.35
                thrust_left, thrust_right = allocate(
                    state, p.cruise_speed_m_s*speed_factor,
                    float(np.clip(steer*params['yaw_command_gain'], -1.0, 1.0))*p.max_yaw_rate_rad_s, p)
            else:
                thrust_left = env.pwm_to_thrust(left)
                thrust_right = env.pwm_to_thrust(right)
            state = integrate(state, thrust_left, thrust_right, env.dt, p)
            x, y = state[0]*scale, state[1]*scale
            if preview_hull_collides(x, y, state[2], nearby, polygons, lengths):
                return index + 1, (x, y)
        return None, (state[0]*scale, state[1]*scale)

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
