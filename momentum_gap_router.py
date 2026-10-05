"""Experimental dynamics-first GAP local motion selection for MAIN_HEAVY.

Only LiDAR hit geometry enters the online obstacle model. The simulator's
ground-truth obstacle list is used by collision physics and offline checks,
never by this selector. This module does not change the vessel model.
"""

import math

import numpy as np
from numba import njit

from dynamic_path_feasibility import (packed_parameters, shadow_allocate,
                                      shadow_integrate, shadow_move_obstacles)
from main_safety_kernels import packed_hulls, preview_hull_collides


HARD_MARGIN_M = 0.20
PREVIEW_STEPS = 40


def perceived_circles(hits_x, hits_y, clusters, boat_position, known_radius):
    """Fit known-radius circular buoys to currently observed LiDAR arcs.

    The small residual allowance covers hit/grid quantization. Clusters with
    no current rays are not promoted to confirmed free-space knowledge.
    """
    valid = np.isfinite(hits_x) & np.isfinite(hits_y)
    points = np.column_stack((hits_x[valid], hits_y[valid])).astype(np.float64)
    if not len(points) or not len(clusters):
        return np.empty((0, 3), dtype=np.float32)
    centers = np.asarray(clusters, dtype=np.float64).reshape(-1, 2)
    assigned = np.argmin(np.sum((points[:, None] - centers[None]) ** 2, axis=2), axis=1)
    result = []
    boat = np.asarray(boat_position, dtype=np.float64)
    for index, center in enumerate(centers):
        arc = points[assigned == index]
        if len(arc) < 2:
            continue
        mean = np.mean(arc, axis=0)
        direction = mean - boat
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        estimate = mean + known_radius * direction / norm
        for _ in range(5):
            delta = estimate - arc
            distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
            residual = distance - known_radius
            jacobian = delta / distance[:, None]
            update, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
            estimate -= np.clip(update, -known_radius, known_radius)
        residual = np.abs(np.linalg.norm(estimate - arc, axis=1) - known_radius)
        if np.median(residual) > known_radius * 0.35:
            continue
        result.append((estimate[0], estimate[1], known_radius +
                       max(2.0, float(np.percentile(residual, 90)))))
    return np.ascontiguousarray(result, dtype=np.float32).reshape(-1, 3)


def portal_interval(gap, obstacle_radius, hull_half_width, margin_px):
    """Conservative geometric interval; no midpoint target is returned."""
    first = np.asarray(gap['c1'], dtype=np.float64)
    second = np.asarray(gap['c2'], dtype=np.float64)
    length = float(np.linalg.norm(second - first))
    if length <= 1e-8:
        return None
    pad = obstacle_radius + hull_half_width + margin_px
    low, high = pad / length, 1.0 - pad / length
    return (low, high) if low < high else None


def portal_crossing(previous, current, gap, interval, goal):
    """Actual trajectory segment crossing, independent of a fixed waypoint."""
    a = np.asarray(gap['c1'], dtype=np.float64)
    b = np.asarray(gap['c2'], dtype=np.float64)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length <= 1e-8 or interval is None:
        return False, None
    axis /= length
    normal = np.array([-axis[1], axis[0]])
    if np.dot(normal, np.asarray(goal) - a) < 0:
        normal = -normal
    before = float(np.dot(np.asarray(previous) - a, normal))
    after = float(np.dot(np.asarray(current) - a, normal))
    if before >= 0 or after <= 0 or after == before:
        return False, None
    fraction = -before / (after - before)
    crossing = np.asarray(previous) + fraction * (np.asarray(current) - np.asarray(previous))
    s = float(np.dot(crossing - a, axis) / length)
    return interval[0] <= s <= interval[1], s


def infer_buoy_origins(observed, frame, dt):
    """Invert the known buoy oscillation from fitted current LiDAR centers."""
    base = np.ascontiguousarray(observed.copy(), dtype=np.float32)
    for _ in range(8):
        phase = frame*dt + 0.05*(base[:, 0] + base[:, 1])
        base[:, 0] = observed[:, 0] - np.sin(phase)*observed[:, 2]*0.2
        base[:, 1] = observed[:, 1] - np.cos(phase*1.2)*observed[:, 2]*0.2
    return base


@njit(cache=True)
def _simulate(state, commands, obstacle_origins, polygons, lengths, parameters,
              dt, steps, margin_px, first_frame):
    vessel = state.copy()
    last_x = vessel[0] * parameters[0]
    last_y = vessel[1] * parameters[0]
    last_heading = vessel[2]
    obstacles = obstacle_origins.copy()
    safe_obstacles = obstacle_origins.copy()
    for j in range(len(safe_obstacles)):
        safe_obstacles[j, 2] += margin_px
    trajectory = np.empty((steps + 1, 3), dtype=np.float64)
    trajectory[0] = (last_x, last_y, last_heading)
    min_center_bound = 1e9
    margin_safe = True
    endpoint_center_bound = 1e9
    for step in range(steps):
        shadow_move_obstacles(obstacle_origins, obstacles,
                              first_frame + step + 1, dt)
        shadow_move_obstacles(obstacle_origins, safe_obstacles,
                              first_frame + step + 1, dt)
        phase = 0 if step < steps // 2 else 1
        left, right = shadow_allocate(vessel, commands[phase, 0],
                                      commands[phase, 1], parameters)
        vessel = shadow_integrate(vessel, left, right, dt, parameters)
        x, y, heading = vessel[0] * parameters[0], vessel[1] * parameters[0], vessel[2]
        trajectory[step + 1] = (x, y, heading)
        # Midpose checks the swept hull between adjacent physics samples.
        middle_heading = last_heading + 0.5 * ((heading - last_heading + math.pi) % (2*math.pi) - math.pi)
        for px, py, ph in ((0.5*(x+last_x), 0.5*(y+last_y), middle_heading),
                           (x, y, heading)):
            if preview_hull_collides(px, py, ph, obstacles, polygons, lengths):
                return False, False, step + 1, trajectory, vessel, min_center_bound, endpoint_center_bound
            if preview_hull_collides(px, py, ph, safe_obstacles, polygons, lengths):
                margin_safe = False
        endpoint_center_bound = 1e9
        for j in range(len(obstacles)):
            bound = math.hypot(x - obstacles[j, 0], y - obstacles[j, 1]) - obstacles[j, 2]
            min_center_bound = min(min_center_bound, bound)
            endpoint_center_bound = min(endpoint_center_bound, bound)
        last_x, last_y, last_heading = x, y, heading
    return True, margin_safe, 0, trajectory, vessel, min_center_bound, endpoint_center_bound


def candidate_commands(env):
    """Deterministic two-stage surge/yaw family; current command is first."""
    speed = env.dynamics.cruise_speed_m_s
    yaw = env.dynamics.max_yaw_rate_rad_s
    current_speed = float(getattr(env, 'command_speed', speed))
    current_yaw = float(getattr(env, 'command_yaw_rate', 0.0))
    families = [('continue', current_speed, current_yaw, current_speed, current_yaw),
                ('straight', speed, 0., speed, 0.)]
    for sign, side in ((-1., 'left'), (1., 'right')):
        for strength, name in ((0.3, 'gentle'), (0.65, 'medium'), (1., 'strong')):
            families.append((f'{side}_{name}', speed, sign*strength*yaw,
                             speed, sign*strength*yaw))
        families.append((f'{side}_release', speed, sign*yaw, speed, 0.))
        families.append((f'{side}_slow', 0.55*speed, sign*yaw,
                         0.55*speed, sign*yaw))
        families.append((f'{side}_pivot_exit', 0., sign*yaw,
                         0.55*speed, sign*yaw))
        families.append((f'{side}_pivot', 0., sign*yaw, 0., sign*yaw))
        families.append((f'{side}_reverse_escape', -0.35*speed, sign*0.7*yaw,
                         0., sign*0.7*yaw))
    families.append(('brake', 0., 0., 0., 0.))
    return families


def select_near_tie(candidates, previous_family, open_straight, tolerance_px=5.0):
    best_progress = max(item[0] for item in candidates)
    near = [item for item in candidates if item[0] >= best_progress-tolerance_px]
    if open_straight:
        straight = [item for item in near if item[3] == 'straight']
        if straight:
            return straight[0]
    return max(near, key=lambda item: (item[3] == previous_family,
                                        item[3] == 'straight', item[0]))


class MomentumGapRouter:
    def __init__(self, horizon_steps=PREVIEW_STEPS, margin_m=HARD_MARGIN_M):
        self.horizon_steps = horizon_steps
        self.margin_m = margin_m
        self.previous_family = None
        self.last_result = None
        self.last_observations = np.empty((0, 3), dtype=np.float32)
        self.last_scores = []
        self.last_frame = -1
        self.last_position = None

    def reset_episode(self):
        self.previous_family = None
        self.last_result = None
        self.last_observations = np.empty((0, 3), dtype=np.float32)
        self.last_scores = []
        self.last_position = None
        self.last_frame = -1

    def gap_interval(self, env, gap):
        if gap is None:
            return None
        # At an aligned crossing the portal axis is the hull's lateral axis.
        half_beam = max(abs(y) for polygon in
                        (env.left_hull_local, env.right_hull_local, env.deck_local)
                        for _, y in polygon)
        margin_px = self.margin_m * env.dynamics.pixels_per_m
        return portal_interval(gap, env.obs_r, half_beam,
                               margin_px + 2.0)

    def gap_ahead(self, env, gap):
        interval = self.gap_interval(env, gap)
        if interval is None:
            return False
        first = np.asarray(gap['c1'], dtype=np.float64)
        vector = np.asarray(gap['c2'], dtype=np.float64) - first
        direction = np.asarray(env.target, dtype=np.float64) - env.boat_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return True
        direction /= norm
        return max(float(np.dot(first + s*vector - env.boat_pos, direction))
                   for s in interval) > 0.0

    def choose(self, env, hits_x, hits_y, gap):
        self.last_frame = env.frame
        observed = perceived_circles(hits_x, hits_y, env.clusters,
                                     env.boat_pos, float(env.obs_r))
        self.last_observations = observed
        state = np.ascontiguousarray(env.physics_state(), dtype=np.float64)
        parameters = packed_parameters(env.dynamics)
        polygons, lengths = packed_hulls((env.left_hull_local,
                                          env.right_hull_local, env.deck_local))
        hard_margin_px = self.margin_m * env.dynamics.pixels_per_m
        # The observed buoy can move up to twice its 0.2r oscillation from
        # the present center; this conservative bound needs no future truth.
        obstacle_origins = infer_buoy_origins(observed, env.frame, env.dt)
        axis = normal = None
        interval = None
        if gap is not None:
            a = np.asarray(gap['c1'], dtype=np.float64)
            b = np.asarray(gap['c2'], dtype=np.float64)
            length = np.linalg.norm(b-a)
            if length > 1e-8:
                axis = (b-a)/length
                normal = np.array([-axis[1], axis[0]])
                if np.dot(normal, env.target-a) < 0:
                    normal = -normal
                interval = self.gap_interval(env, gap)
        candidates = []
        recovery_candidates = []
        for family, speed1, yaw1, speed2, yaw2 in candidate_commands(env):
            commands = np.array(((speed1, yaw1), (speed2, yaw2)), dtype=np.float64)
            physical_safe, margin_safe, collision_step, trajectory, terminal, center_bound, end_bound = _simulate(
                state, commands, obstacle_origins, polygons, lengths, parameters,
                env.dt, self.horizon_steps, hard_margin_px, int(env.frame))
            if not physical_safe:
                continue
            endpoint = trajectory[-1, :2]
            progress = float(np.dot(endpoint - env.boat_pos,
                                    (env.target-env.boat_pos) /
                                    max(float(np.linalg.norm(env.target-env.boat_pos)), 1e-9)))
            if axis is not None and interval is not None:
                a = np.asarray(gap['c1'], dtype=np.float64)
                crossing = False
                crossing_s = None
                for index in range(1, len(trajectory)):
                    crossing, crossing_s = portal_crossing(
                        trajectory[index-1, :2], trajectory[index, :2],
                        gap, interval, env.target)
                    if crossing:
                        break
                s_end = float(np.dot(endpoint-a, axis) / length)
                off_interval = max(interval[0]-s_end, s_end-interval[1], 0.0)*length
                normal_progress = float(np.dot(endpoint-env.boat_pos, normal))
                progress = normal_progress - off_interval
                if crossing:
                    progress += 30.0
            else:
                crossing_s = None
            if np.linalg.norm(endpoint - env.boat_pos) < 8.0:
                desired = (math.atan2(float(normal[1]), float(normal[0]))
                           if normal is not None and interval is not None else
                           math.atan2(float(env.target[1]-env.boat_pos[1]),
                                      float(env.target[0]-env.boat_pos[0])))
                old_align = math.cos(float(state[2]) - desired)
                new_align = math.cos(float(terminal[2]) - desired)
                progress += 0.5 * 84.0 * (new_align - old_align)
            continuity = 1.0 if family == self.previous_family else 0.0
            straight = 1.0 if family == 'straight' else 0.0
            # Safety has already been enforced as a hard gate. A small
            # progress tie band preserves the current command and straight.
            item = (progress, continuity, straight, family, commands,
                    trajectory, terminal, center_bound, crossing_s)
            if margin_safe:
                candidates.append(item)
            else:
                recovery_candidates.append((end_bound, item))
        if candidates:
            if abs(float(state[3])) < 0.05:
                escape = [item for item in candidates
                          if item[3] not in ('brake', 'continue') and
                          (abs(item[4][0, 0]) > 1e-6 or
                           abs(item[4][0, 1]) > 1e-6)]
                if escape:
                    # A stopped incumbent is not useful continuity. Safety
                    # has already been checked for the escape primitives.
                    candidates = escape
            forward_useful = any(item[0] > 5.0 and
                                 item[4][0, 0] >= 0.0 and
                                 'reverse' not in item[3]
                                 for item in candidates)
            if forward_useful:
                candidates = [item for item in candidates
                              if 'reverse' not in item[3]]
            self.last_scores = [(item[3], round(item[0], 2), True)
                                for item in candidates]
            goal_angle = math.atan2(float(env.target[1]-env.boat_pos[1]),
                                    float(env.target[0]-env.boat_pos[0]))
            heading_error = (goal_angle-env.boat_heading+math.pi)%(2*math.pi)-math.pi
            front = np.abs(env.rel_angles) <= math.pi/6
            open_straight = (abs(heading_error) < 0.15 and
                             np.min(env.lidar_dists[front]) > 150.0)
            selected = select_near_tie(candidates, self.previous_family,
                                       open_straight)
            self.previous_family = selected[3]
            commands = selected[4]
            left, right = shadow_allocate(state, commands[0, 0], commands[0, 1], parameters)
            reason = 'safe_candidate'
            result = {'family': selected[3], 'safe': True,
                      'candidate_count': len(candidates),
                      'progress_px': selected[0],
                      'minimum_center_bound_px': selected[7],
                      'crossing_s': selected[8], 'prediction': selected[5],
                      'left_pwm': 1500.0 + 400.0*left/env.dynamics.max_thrust_N,
                      'right_pwm': 1500.0 + 400.0*right/env.dynamics.max_thrust_N,
                      'speed_command': float(commands[0, 0]),
                      'yaw_rate_command': float(commands[0, 1]),
                      'reason': reason}
        elif recovery_candidates:
            self.last_scores = [(item[3], round(item[0], 2), False)
                                for _, item in recovery_candidates]
            # Already within the comfort envelope: only physically
            # collision-free trajectories can escape it. Prefer the greatest
            # final separation and avoid normal full-speed progress rewards.
            _, selected = max(recovery_candidates,
                              key=lambda pair: (pair[0], pair[1][1], pair[1][2]))
            commands = selected[4]
            left, right = shadow_allocate(state, commands[0, 0], commands[0, 1], parameters)
            self.previous_family = selected[3]
            result = {'family': selected[3], 'safe': False,
                      'candidate_count': len(recovery_candidates),
                      'progress_px': selected[0],
                      'minimum_center_bound_px': selected[7],
                      'crossing_s': selected[8], 'prediction': selected[5],
                      'left_pwm': 1500.0 + 400.0*left/env.dynamics.max_thrust_N,
                      'right_pwm': 1500.0 + 400.0*right/env.dynamics.max_thrust_N,
                      'speed_command': float(commands[0, 0]),
                      'yaw_rate_command': float(commands[0, 1]),
                      'reason': 'physical_safe_margin_recovery'}
        else:
            self.last_scores = []
            left, right = shadow_allocate(state, 0., 0., parameters)
            result = {'family': 'brake_no_safe_candidate', 'safe': False,
                      'candidate_count': 0, 'progress_px': 0.,
                      'minimum_center_bound_px': None, 'crossing_s': None,
                      'prediction': None,
                      'left_pwm': 1500.0 + 400.0*left/env.dynamics.max_thrust_N,
                      'right_pwm': 1500.0 + 400.0*right/env.dynamics.max_thrust_N,
                      'speed_command': 0.0, 'yaw_rate_command': 0.0,
                      'reason': 'no_safe_candidate'}
        self.last_result = result
        return result
