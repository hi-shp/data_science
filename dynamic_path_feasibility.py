"""Experimental short preview of MAIN_HEAVY's actual GAP path follower.

The live vessel, controller state, and physics parameters are never modified.
This is a compiled scalar copy of vessel_dynamics.integrate/allocate, with
parity tests against those authoritative functions.
"""

import math

import numpy as np
from numba import njit

from main_safety_kernels import (packed_hulls, preview_follow_steering,
                                 preview_hull_collides, preview_lidar_distances)
from portal_navigation import path_has_hull_clearance, remaining_path
from utils import make_bezier_path, pure_pursuit, wrap


def packed_parameters(dynamics):
    return np.array((dynamics.pixels_per_m, dynamics.mass_kg,
                     dynamics.yaw_inertia_kg_m2, dynamics.surge_linear_drag,
                     dynamics.surge_quadratic_drag, dynamics.sway_linear_drag,
                     dynamics.sway_quadratic_drag, dynamics.yaw_linear_drag,
                     dynamics.yaw_quadratic_drag, dynamics.thruster_arm_m,
                     dynamics.max_thrust_N, dynamics.actuator_tau_s,
                     dynamics.cruise_speed_m_s, dynamics.max_yaw_rate_rad_s,
                     dynamics.speed_response_s,
                     dynamics.yaw_rate_gain_Nm_s if dynamics.yaw_rate_gain_Nm_s is not None
                     else dynamics.yaw_inertia_kg_m2 / dynamics.yaw_response_s),
                    dtype=np.float64)


@njit(cache=True)
def shadow_allocate(state, speed, yaw_rate, p):
    u, r = state[3], state[5]
    force = p[3]*u + p[4]*u*abs(u) + p[1]*(speed-u)/p[14]
    moment = p[7]*r + p[8]*r*abs(r) + p[15]*(yaw_rate-r)
    diff = min(p[10], max(-p[10], moment/(2.0*p[9])))
    bound = p[10]-abs(diff)
    common = min(bound, max(-bound, force/2.0))
    return common-diff, common+diff


@njit(cache=True)
def _derivatives(h, u, v, r, fl, fr, p):
    ch, sh = math.cos(h), math.sin(h)
    dx = u*ch-v*sh
    dy = u*sh+v*ch
    du = (fl+fr-p[3]*u-p[4]*u*abs(u))/p[1]+v*r
    dv = (-p[5]*v-p[6]*v*abs(v))/p[1]-u*r
    dr = ((fr-fl)*p[9]-p[7]*r-p[8]*r*abs(r))/p[2]
    return dx, dy, r, du, dv, dr


@njit(cache=True)
def shadow_integrate(state, left, right, dt, p):
    left = min(p[10], max(-p[10], left))
    right = min(p[10], max(-p[10], right))
    alpha = -math.expm1(-dt/p[11])
    fl = state[6]+alpha*(left-state[6])
    fr = state[7]+alpha*(right-state[7])
    d1 = _derivatives(state[2], state[3], state[4], state[5], fl, fr, p)
    d2 = _derivatives(state[2]+0.5*dt*d1[2],
                      state[3]+0.5*dt*d1[3],
                      state[4]+0.5*dt*d1[4],
                      state[5]+0.5*dt*d1[5], fl, fr, p)
    result = np.empty(8, dtype=np.float64)
    for i in range(6):
        result[i] = state[i]+dt*d2[i]
    result[6], result[7] = fl, fr
    return result


@njit(cache=True)
def shadow_move_obstacles(base, output, frame, dt):
    phase_time = frame * dt
    for j in range(base.shape[0]):
        ox, oy, radius = base[j, 0], base[j, 1], base[j, 2]
        phase = phase_time + 0.05 * (ox + oy)
        output[j, 0] = ox + math.sin(phase) * (radius * 0.2)
        output[j, 1] = oy + math.cos(phase * 1.2) * (radius * 0.2)


@njit(cache=True)
def _rollout(state, path, has_waypoint, nearby_base, lidar_base,
             polygons, lengths, rel_angles, p, dt, horizon,
             previous_steer, emergency, cooldown, min_wide,
             steer_gain, alpha, avoid_normal, avoid_em,
             em_enter, em_exit, em_hold, yaw_command_gain,
             lidar_range, first_left, first_right, use_first_command,
             first_frame):
    z = state.copy()
    nearby = nearby_base.copy()
    lidar_obstacles = lidar_base.copy()
    max_cross_track = 0.0
    minimum_center_clearance = 1e6
    steering_saturation_steps = 0
    collision_step = 0
    for step in range(horizon):
        # The live simulation moves every buoy before the next physics step.
        # Keep the shadow safety and LiDAR input on the same future frame.
        shadow_move_obstacles(nearby_base, nearby, first_frame + step + 1, dt)
        shadow_move_obstacles(lidar_base, lidar_obstacles,
                              first_frame + step + 1, dt)
        if step == 0 and use_first_command:
            left, right = first_left, first_right
        else:
            distances = preview_lidar_distances(z[0]*p[0], z[1]*p[0],
                                                z[2], lidar_obstacles,
                                                lidar_range)
            steer, previous_steer, emergency, cooldown, min_wide = \
                preview_follow_steering(
                    z[0]*p[0], z[1]*p[0], z[2], z[5], previous_steer,
                    emergency, cooldown, path, distances, rel_angles,
                    steer_gain, alpha, avoid_normal, avoid_em,
                    em_enter, em_exit, em_hold, has_waypoint)
            if abs(steer) >= 0.999:
                steering_saturation_steps += 1
            speed_factor = math.tanh(max(0.0, min_wide)/100.0)**1.35
            speed = p[12]*speed_factor
            desired_yaw = min(1.0, max(-1.0, steer*yaw_command_gain))*p[13]
            left, right = shadow_allocate(z, speed, desired_yaw, p)
        z = shadow_integrate(z, left, right, dt, p)
        x, y = z[0]*p[0], z[1]*p[0]
        if path.shape[0]:
            nearest2 = 1e12
            for j in range(path.shape[0]):
                dx = path[j, 0]-x
                dy = path[j, 1]-y
                nearest2 = min(nearest2, dx*dx+dy*dy)
            max_cross_track = max(max_cross_track, math.sqrt(nearest2))
        for obstacle in nearby:
            clearance = math.hypot(obstacle[0]-x, obstacle[1]-y)-obstacle[2]
            minimum_center_clearance = min(minimum_center_clearance, clearance)
        if preview_hull_collides(x, y, z[2], nearby, polygons, lengths):
            collision_step = step+1
            break
    return (collision_step, minimum_center_clearance, max_cross_track,
            steering_saturation_steps, z)


def evaluate_candidate(env, path, horizon_steps=40, first_pwm=None,
                       has_waypoint=None):
    """Preview the supplied path with unchanged physics dt and hull geometry."""
    p = packed_parameters(env.dynamics)
    state = np.ascontiguousarray(env.physics_state(), dtype=np.float64)
    path = (np.empty((0, 2), dtype=np.float32) if path is None else
            np.ascontiguousarray(path, dtype=np.float32))
    obstacles = np.ascontiguousarray(env.dynamic_obstacles, dtype=np.float32)
    x, y = env.boat_pos
    speed_px_s = math.hypot(float(env.boat_vel[0]), float(env.boat_vel[1]))
    maximum_radius = float(np.max(obstacles[:, 2])) if len(obstacles) else 0.0
    reach = (max(speed_px_s, env.dynamics.cruise_speed_m_s*p[0]) *
             horizon_steps*env.dt + 45.0 + maximum_radius + 10.0)
    delta = obstacles[:, :2]-env.boat_pos
    distance2 = np.sum(delta*delta, axis=1)
    nearby = np.ascontiguousarray(env.obstacles[distance2 <= reach*reach], dtype=np.float32)
    lidar_radius = reach+env.lidar_range+maximum_radius
    lidar_obstacles = np.ascontiguousarray(
        env.obstacles[distance2 <= lidar_radius*lidar_radius], dtype=np.float32)
    polygons, lengths = packed_hulls((env.left_hull_local,
                                       env.right_hull_local,
                                       env.deck_local))
    left = right = 0.0
    if first_pwm is not None:
        left, right = (env.pwm_to_thrust(first_pwm[0]),
                       env.pwm_to_thrust(first_pwm[1]))
    collision_step, clearance, cross_track, saturation, terminal = _rollout(
        state, path, (env.current_wp is not None if has_waypoint is None
                      else has_waypoint), nearby, lidar_obstacles,
        polygons, lengths, env.rel_angles, p, env.dt, int(horizon_steps),
        float(env.prev_steer), bool(env.emergency_mode),
        int(env.emergency_cooldown), float(env.min_wide_dist),
        float(env.params['steer_gain']), float(env.params['steer_alpha']),
        float(env.params['avoid_normal']), float(env.params['avoid_em']),
        float(env.params['em_enter']), float(env.params['em_exit']),
        int(env.params['em_hold_frames']), float(env.params['yaw_command_gain']),
        float(env.lidar_range), float(left), float(right),
        first_pwm is not None, int(env.frame))
    return {'collision_step': int(collision_step),
            'minimum_center_clearance_px': float(clearance),
            'maximum_cross_track_px': float(cross_track),
            'steering_saturation_steps': int(saturation),
            'terminal_state': terminal}


def splice_path(previous_path, position, new_goal, obstacles, boat_radius,
                boat_speed, anchor_distance_px):
    """Retain a short forward part of the old path and join with C1 tangent."""
    forward = remaining_path(previous_path, position)
    if forward is None or len(forward) < 2:
        return None
    accumulated = 0.0
    anchor_index = 1
    for index in range(1, len(forward)):
        accumulated += math.dist(forward[index-1], forward[index])
        anchor_index = index
        if accumulated >= anchor_distance_px:
            break
    tangent = forward[anchor_index]-forward[anchor_index-1]
    if math.hypot(float(tangent[0]), float(tangent[1])) < 1e-8:
        return None
    heading = math.atan2(float(tangent[1]), float(tangent[0]))
    suffix = make_bezier_path(forward[anchor_index], heading, new_goal,
                              obstacles=obstacles, boat_radius=boat_radius,
                              boat_speed=boat_speed, start_tangent_fixed=True)
    if suffix is None:
        return None
    return np.ascontiguousarray(np.vstack((forward[:anchor_index+1], suffix[1:])),
                                 dtype=np.float32)


def replacement_metrics(old_path, new_path, position, heading, lookahead=70.0):
    old_target = pure_pursuit(remaining_path(old_path, position), position,
                              lookahead=lookahead)
    new_target = pure_pursuit(new_path, position, lookahead=lookahead)
    if old_target is None or new_target is None:
        return 0.0, 0.0
    old_angle = math.atan2(float(old_target[1]-position[1]),
                           float(old_target[0]-position[0]))
    new_angle = math.atan2(float(new_target[1]-position[1]),
                           float(new_target[0]-position[0]))
    return (math.dist(old_target, new_target),
            abs(wrap(new_angle-old_angle)))


def retain_feasible_previous(old_length, new_length, old_clearance,
                             new_clearance, required_gain, remaining_distance):
    """Prevent an immaterial route-score change from replacing a valid path."""
    return (old_length-new_length < required_gain and
            new_clearance-old_clearance < 10.0 and
            remaining_distance > 70.0)


class DynamicsPathSelector:
    """Small top-K feasibility gate; the baseline path remains the default."""

    def __init__(self, horizon_steps=40):
        self.horizon_steps = horizon_steps
        self.active_path = None
        self.active_wp = None
        self.last_frame = -1
        self.probes = 0
        self.rejections = 0
        self.alternates = 0
        self.retained = 0
        self.brakes = 0
        self.splices = 0
        self.last_reason = 'uninitialized'
        self.last_jump_px = 0.0
        self.last_heading_jump = 0.0

    def _feasible(self, env, path, waypoint=None):
        if path is None or len(path) < 2:
            return False, 'empty', None
        hull = (env.left_hull_local, env.right_hull_local, env.deck_local)
        if not path_has_hull_clearance(path, env.dynamic_obstacles, hull,
                                       margin_px=0.0):
            return False, 'geometric', None
        prediction = evaluate_candidate(env, path, self.horizon_steps,
                                        has_waypoint=waypoint is not None)
        self.probes += 1
        if prediction['collision_step']:
            return False, 'dynamic_collision', prediction
        # The actual beam width supplies a meaningful tracking tolerance.
        if prediction['maximum_cross_track_px'] > 54.0:
            return False, 'untrackable', prediction
        return True, 'feasible', prediction

    @staticmethod
    def _length(path):
        if path is None or len(path) < 2:
            return math.inf
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

    def select(self, env, proposed_path, proposed_wp, ranked_gaps):
        if env.frame < self.last_frame:
            self.active_path = self.active_wp = None
            self.probes = self.rejections = self.alternates = 0
            self.retained = self.brakes = self.splices = 0
        self.last_frame = env.frame
        previous = remaining_path(self.active_path, env.boat_pos)
        old_feasible, _, old_prediction = self._feasible(env, previous,
                                                        self.active_wp)

        proposed_feasible, reason, prediction = self._feasible(env, proposed_path,
                                                               proposed_wp)
        selected_path, selected_wp = proposed_path, proposed_wp
        if not proposed_feasible:
            self.rejections += 1
            # GAP score remains ordering only; no new global planner.
            for gap in ranked_gaps[:3]:
                if proposed_wp is not None and set(gap['pair']) == set(proposed_wp['pair']):
                    continue
                speed = math.hypot(float(env.boat_vel[0]), float(env.boat_vel[1]))
                alternate = make_bezier_path(
                    env.boat_pos, env.boat_heading, gap['pos'],
                    obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius,
                    boat_speed=speed)
                okay, _, alternate_prediction = self._feasible(env, alternate,
                                                                gap)
                if okay:
                    selected_path, selected_wp = alternate, gap
                    prediction = alternate_prediction
                    proposed_feasible = True
                    self.alternates += 1
                    reason = 'alternate'
                    break

        if old_feasible and proposed_feasible:
            jump, heading_jump = replacement_metrics(
                previous, selected_path, env.boat_pos, env.boat_heading)
            self.last_jump_px, self.last_heading_jump = jump, heading_jump
            half_beam = 27.0
            if jump > 2.0*half_beam or heading_jump > math.atan2(half_beam, 70.0):
                speed = math.hypot(float(env.boat_vel[0]), float(env.boat_vel[1]))
                anchor = max(half_beam, speed*env.dynamics.yaw_response_s*0.5)
                spliced = splice_path(previous, env.boat_pos,
                                      selected_wp['pos'] if selected_wp is not None else env.target,
                                      env.dynamic_obstacles, env.boat_radius,
                                      speed, anchor)
                splice_feasible, _, splice_prediction = self._feasible(
                    env, spliced, selected_wp)
                if splice_feasible:
                    selected_path = spliced
                    prediction = splice_prediction
                    self.splices += 1
                    reason = 'splice'
            old_length = self._length(previous)
            new_length = self._length(selected_path)
            required_gain = math.hypot(*env.boat_vel)*env.dynamics.yaw_response_s
            old_clearance = old_prediction['minimum_center_clearance_px']
            new_clearance = prediction['minimum_center_clearance_px']
            if retain_feasible_previous(
                    old_length, new_length, old_clearance, new_clearance,
                    required_gain, math.dist(env.boat_pos, previous[-1])):
                selected_path, selected_wp = previous, self.active_wp
                reason = 'retain'
                self.retained += 1
        elif old_feasible and not proposed_feasible:
            selected_path, selected_wp = previous, self.active_wp
            reason = 'retain_after_reject'
            self.retained += 1
        elif not proposed_feasible:
            selected_path, selected_wp = None, proposed_wp
            reason = 'brake_no_feasible_path'
            self.brakes += 1

        self.last_reason = reason
        if selected_path is not None:
            self.active_path = np.ascontiguousarray(selected_path,
                                                     dtype=np.float32)
            self.active_wp = selected_wp
        return selected_path, selected_wp, reason
