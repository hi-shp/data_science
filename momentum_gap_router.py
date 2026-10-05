"""Experimental dynamics-first GAP local motion selection for MAIN_HEAVY.

Only LiDAR hit geometry enters the online obstacle model. The simulator's
ground-truth obstacle list is used by collision physics and offline checks,
never by this selector. This module does not change the vessel model.
"""

import math
import os

import numpy as np
from numba import njit

from dynamic_path_feasibility import (packed_parameters, shadow_allocate,
                                      shadow_integrate, shadow_move_obstacles)
from main_safety_kernels import (packed_hulls, preview_hull_collides,
                                 preview_hull_wall_clearance,
                                 preview_hull_surface_clearance,
                                 preview_hull_wall_turning_room)
from portal_navigation import remaining_path


HARD_MARGIN_M = 0.20
PREVIEW_STEPS = 40
MOTION_VERSION = 'V1'


def wall_lidar_hits(env, distances, hits_x, hits_y):
    """Merge known arena ray intersections for display; buoy mapping stays separate."""
    angles = env.rel_angles + env.boat_heading
    vx, vy = np.cos(angles), np.sin(angles)
    x, y = env.boat_pos
    wall = np.full(len(angles), float(env.lidar_range))
    for component, origin, edge in ((vx, x, 0.), (vx, x, env.map_w),
                                    (vy, y, 0.), (vy, y, env.sim_h)):
        t = np.divide(edge-origin, component, out=np.full(len(angles), np.inf),
                      where=np.abs(component) > 1e-8)
        wall = np.minimum(wall, np.where(t >= 0., t, np.inf))
    use = (wall < distances) & (wall < env.lidar_range)
    return (np.where(use, wall, distances).astype(np.float32),
            np.where(use, x+vx*wall, hits_x).astype(np.float32),
            np.where(use, y+vy*wall, hits_y).astype(np.float32))


def trajectory_bezier_reference(prediction):
    """Piecewise cubic Bezier reference fitted AFTER selecting physical motion.

    No crossing point is used to steer. The physical rollout remains the safety
    authority; the Bezier and pursuit point are separate display references.
    """
    if prediction is None or len(prediction) < 2:
        return None
    points = np.asarray(prediction[:, :2], dtype=float)
    indices = list(range(0, len(points)-1, 4)) + [len(points)-1]
    tangents = np.gradient(points, axis=0)
    t = np.linspace(0., 1., 5)[:, None]
    pieces = []
    for start, end in zip(indices[:-1], indices[1:]):
        span = end-start
        a, d = points[start], points[end]
        b, c = a+tangents[start]*span/3., d-tangents[end]*span/3.
        curve = (1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*d
        pieces.append(curve if not pieces else curve[1:])
    return np.concatenate(pieces)


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
    r1, r2 = gap.get('endpoint_radii', (obstacle_radius, obstacle_radius))
    low = (r1 + hull_half_width + margin_px) / length
    high = 1.0 - (r2 + hull_half_width + margin_px) / length
    return (low, high) if low < high else None


def portal_wall_interval(gap, interval, heading, hull_polygons,
                         margin_px, map_width, map_height):
    """Clip a portal to positions where its oriented hull clears all walls."""
    if interval is None:
        return None
    first = gap['c1']
    second = gap['c2']
    axis_x = float(second[0]-first[0])
    axis_y = float(second[1]-first[1])
    ch, sh = math.cos(heading), math.sin(heading)
    low, high = interval
    min_x = min_y = float('inf')
    max_x = max_y = -float('inf')
    for polygon in hull_polygons:
        for local_x, local_y in polygon:
            offset_x = ch*local_x-sh*local_y
            offset_y = sh*local_x+ch*local_y
            min_x, max_x = min(min_x, offset_x), max(max_x, offset_x)
            min_y, max_y = min(min_y, offset_y), max(max_y, offset_y)
    for origin, delta, minimum, maximum in (
            (float(first[0]), axis_x, margin_px-min_x,
             map_width-margin_px-max_x),
            (float(first[1]), axis_y, margin_px-min_y,
             map_height-margin_px-max_y)):
        if abs(delta) < 1e-12:
            if not minimum <= origin <= maximum:
                return None
            continue
        start = (minimum-origin)/delta
        end = (maximum-origin)/delta
        low = max(low, min(start, end))
        high = min(high, max(start, end))
        if low >= high:
            return None
    return low, high


@njit(cache=True)
def _portal_route_order(ax, ay, dx, dy, px, py, gx, gy, low, high):
    """The existing 16-step ternary ordering without Python array churn."""
    left, right = low, high
    for _ in range(16):
        one = (2.0*left+right)/3.0
        two = (left+2.0*right)/3.0
        x_one, y_one = ax+one*dx, ay+one*dy
        x_two, y_two = ax+two*dx, ay+two*dy
        d_one = (math.sqrt((x_one-px)**2+(y_one-py)**2)+
                 math.sqrt((gx-x_one)**2+(gy-y_one)**2))
        d_two = (math.sqrt((x_two-px)**2+(y_two-py)**2)+
                 math.sqrt((gx-x_two)**2+(gy-y_two)**2))
        if d_one <= d_two:
            right = two
        else:
            left = one
    s = 0.5*(left+right)
    x, y = ax+s*dx, ay+s*dy
    length = (math.sqrt((x-px)**2+(y-py)**2)+
              math.sqrt((gx-x)**2+(gy-y)**2))
    distance = math.sqrt((x-px)**2+(y-py)**2)
    return x, y, length, distance


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
              dt, steps, margin_px, first_frame, map_width, map_height,
              previous_yaw=0., yaw_step=1e9):
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
    commanded_yaw = previous_yaw
    for step in range(steps):
        shadow_move_obstacles(obstacle_origins, obstacles,
                              first_frame + step + 1, dt)
        shadow_move_obstacles(obstacle_origins, safe_obstacles,
                              first_frame + step + 1, dt)
        phase = 0 if step < steps // 2 else 1
        commanded_yaw = max(commanded_yaw-yaw_step,
                            min(commanded_yaw+yaw_step, commands[phase, 1]))
        left, right = shadow_allocate(vessel, commands[phase, 0],
                                      commanded_yaw, parameters)
        vessel = shadow_integrate(vessel, left, right, dt, parameters)
        x, y, heading = vessel[0] * parameters[0], vessel[1] * parameters[0], vessel[2]
        trajectory[step + 1] = (x, y, heading)
        # Midpose checks the swept hull between adjacent physics samples.
        middle_heading = last_heading + 0.5 * ((heading - last_heading + math.pi) % (2*math.pi) - math.pi)
        for px, py, ph in ((0.5*(x+last_x), 0.5*(y+last_y), middle_heading),
                           (x, y, heading)):
            wall_clearance = preview_hull_wall_clearance(
                px, py, ph, polygons, lengths, map_width, map_height)
            if wall_clearance <= 0.0:
                return False, False, step + 1, trajectory, vessel, min_center_bound, endpoint_center_bound
            if wall_clearance < margin_px:
                margin_safe = False
            if preview_hull_collides(px, py, ph, obstacles, polygons, lengths):
                return False, False, step + 1, trajectory, vessel, min_center_bound, endpoint_center_bound
            if preview_hull_collides(px, py, ph, safe_obstacles, polygons, lengths):
                margin_safe = False
        endpoint_center_bound = 1e9
        endpoint_center_bound = preview_hull_wall_clearance(
            x, y, heading, polygons, lengths, map_width, map_height)
        min_center_bound = min(min_center_bound, endpoint_center_bound)
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
    current_speed = max(0., float(getattr(env, 'command_speed', speed)))
    current_yaw = float(getattr(env, 'command_yaw_rate', 0.0))
    families = [('continue', current_speed, current_yaw, current_speed, current_yaw),
                ('straight', speed, 0., speed, 0.)]
    # Finer corrections and gradual release avoid quantized left/right hunting.
    families.append(('yaw_release', current_speed, .5*current_yaw,
                     current_speed, 0.))
    families.append(('reduced_straight', .65*speed, 0., .65*speed, 0.))
    for sign, side in ((-1., 'left'), (1., 'right')):
        for strength, name in ((.1, 'trim'), (0.3, 'gentle'), (0.65, 'medium'), (1., 'strong')):
            families.append((f'{side}_{name}', speed, sign*strength*yaw,
                             speed, sign*strength*yaw))
        families.append((f'{side}_release', speed, sign*yaw, speed, 0.))
        families.append((f'{side}_slow', 0.55*speed, sign*yaw,
                         0.55*speed, sign*yaw))
        families.append((f'{side}_pivot_exit', 0., sign*yaw,
                         0.55*speed, sign*yaw))
        families.append((f'{side}_pivot', 0., sign*yaw, 0., sign*yaw))
    families.append(('brake', 0., 0., 0., 0.))
    return families


def recovery_commands(env):
    """Reverse is a separate short space-making library, followed by braking."""
    return [(f'{side}_reverse_escape', -.35*env.dynamics.cruise_speed_m_s,
             sign*.7*env.dynamics.max_yaw_rate_rad_s, 0.,
             sign*.7*env.dynamics.max_yaw_rate_rad_s)
            for sign, side in ((-1., 'left'), (1., 'right'))]


def adaptive_pursuit_command(env, path):
    """Optional A/B reference command; final safety still uses _simulate.

    Distance is traveled along the displayed Bezier, starting at the current
    segment projection. The response time and actual hull length set the
    straight-run scale; curvature, yaw momentum, and observed clearance
    shorten it when a turn is already required.
    """
    if path is None or len(path) < 2:
        return None
    route = remaining_path(path, env.boat_pos)
    if len(route) < 2:
        return None
    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1e-8
    if not np.any(valid):
        return None
    half_length = max(abs(x) for polygon in
                      (env.left_hull_local, env.right_hull_local, env.deck_local)
                      for x, _ in polygon)
    speed_px_s = max(0.0, float(env.physics_state()[3])) * env.dynamics.pixels_per_m
    straight_lookahead = half_length + speed_px_s * env.dynamics.yaw_response_s
    # Estimate the heading change over the near part of the existing route.
    first = int(np.flatnonzero(valid)[0])
    last = min(len(segments) - 1, first + 12)
    bend = abs((math.atan2(segments[last, 1], segments[last, 0]) -
                math.atan2(segments[first, 1], segments[first, 0]) + math.pi)
               % (2 * math.pi) - math.pi)
    turn_factor = 1.0 / (1.0 + bend)
    yaw_factor = 1.0 / (1.0 + abs(float(env.boat_ang_vel)) *
                        env.dynamics.yaw_response_s)
    observed_clearance = float(np.min(env.lidar_dists)) if len(env.lidar_dists) else 200.0
    clearance_factor = min(1.0, max(0.65, observed_clearance / (2.0 * half_length)))
    lookahead = float(np.clip(straight_lookahead * turn_factor * yaw_factor *
                              clearance_factor, half_length, 3.0 * half_length))
    left = lookahead
    target = route[-1]
    for index, length in enumerate(lengths):
        if left <= length:
            target = route[index] + (left / length) * segments[index]
            break
        left -= length
    heading = math.atan2(float(target[1] - env.boat_pos[1]),
                         float(target[0] - env.boat_pos[0]))
    error = (heading - env.boat_heading + math.pi) % (2 * math.pi) - math.pi
    desired_yaw = float(np.clip(error / env.dynamics.yaw_response_s -
                                env.boat_ang_vel, -env.dynamics.max_yaw_rate_rad_s,
                                env.dynamics.max_yaw_rate_rad_s))
    return env.dynamics.cruise_speed_m_s, desired_yaw, lookahead


def select_near_tie(candidates, previous_family, tolerance_px=5.0,
                    previous_command=None, yaw_rate=0., response_s=.85):
    best_progress = max(item[0] for item in candidates)
    near = [item for item in candidates if item[0] >= best_progress-tolerance_px]
    if previous_command is None:
        return max(near, key=lambda item: (item[3] == previous_family,
                                            item[3] == 'straight', item[0]))
    previous = np.asarray(previous_command)
    def quality(item):
        command = item[4]
        delta = command[0]-previous
        change = float(delta[1]**2 + .2*delta[0]**2)
        change += float((command[1, 1]-command[0, 1])**2)
        change += .02*float(np.abs(command[:, 1]).sum())*response_s
        return (change, item[3] != previous_family,
                abs(float(command[0, 1]-yaw_rate)), -item[0])
    return min(near, key=quality)


class MomentumGapRouter:
    def __init__(self, horizon_steps=PREVIEW_STEPS, margin_m=HARD_MARGIN_M):
        self.horizon_steps = horizon_steps
        self.margin_m = margin_m
        self.wall_clip = os.environ.get('MAIN_HEAVY_PORTAL_WALL_CLIP', '1') == '1'
        self.previous_family = None
        self.last_result = None
        self.last_observations = np.empty((0, 3), dtype=np.float32)
        self.last_scores = []
        self.last_candidate_states = []
        self.last_detour_active = False
        self.last_frame = -1
        self.last_position = None
        self.no_progress_pair = None
        self.no_progress_start_frame = None

    def reset_episode(self):
        self.previous_family = None
        self.last_result = None
        self.last_observations = np.empty((0, 3), dtype=np.float32)
        self.last_scores = []
        self.last_candidate_states = []
        self.last_detour_active = False
        self.last_position = None
        self.last_frame = -1
        self.no_progress_pair = None
        self.no_progress_start_frame = None

    def gap_interval(self, env, gap):
        if gap is None:
            return None
        # At an aligned crossing the portal axis is the hull's lateral axis.
        half_beam = max(abs(y) for polygon in
                        (env.left_hull_local, env.right_hull_local, env.deck_local)
                        for _, y in polygon)
        margin_px = self.margin_m * env.dynamics.pixels_per_m
        interval = portal_interval(gap, env.obs_r, half_beam,
                                   margin_px + 2.0)
        if interval is None or not self.wall_clip:
            return interval
        axis = np.asarray(gap['c2'])-np.asarray(gap['c1'])
        normal = np.array([-axis[1], axis[0]])
        if np.dot(normal, np.asarray(env.target)-np.asarray(gap['c1'])) < 0.0:
            normal = -normal
        heading = math.atan2(float(normal[1]), float(normal[0]))
        return portal_wall_interval(
            gap, interval, heading,
            (env.left_hull_local, env.right_hull_local, env.deck_local),
            margin_px + 2.0, float(env.map_w), float(env.sim_h))

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

    def portal_options(self, env, position=None, visited=None):
        """LiDAR-map GAP topology without legacy midpoint scores or truth buoys."""
        position = np.asarray(env.boat_pos if position is None else position,
                              dtype=np.float64)
        visited = env.visited if visited is None else visited
        centers = np.asarray(env.clusters, dtype=np.float64).reshape(-1, 2)
        if not len(centers):
            return []
        goal = np.asarray(env.target, dtype=np.float64)
        direction = goal-position
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        half_beam = max(abs(y) for polygon in
                        (env.left_hull_local, env.right_hull_local, env.deck_local)
                        for _, y in polygon)
        pad = float(env.obs_r) + half_beam + self.margin_m * env.dynamics.pixels_per_m + 2.0
        polygons, lengths = packed_hulls((env.left_hull_local,
                                          env.right_hull_local, env.deck_local))
        options = []
        pairs = []
        for i, first in enumerate(centers):
            for j in range(i+1, len(centers)):
                pairs.append((first, centers[j],
                              (env.cluster_ids[i], env.cluster_ids[j]),
                              (env.obs_r, env.obs_r), None))
            # Known wall geometry is not an unseen obstacle observation.
            for wall_id, anchor in enumerate(((0., first[1]),
                    (env.map_w, first[1]), (first[0], 0.),
                    (first[0], env.sim_h))):
                if np.linalg.norm(np.asarray(anchor)-position) <= env.lidar_range:
                    pairs.append((np.asarray(anchor), first,
                                  (-1001-wall_id, env.cluster_ids[i]),
                                  (0., env.obs_r), wall_id))
        for first, second, pair, radii, wall_id in pairs:
            if pair in visited or pair[::-1] in visited:
                continue
            axis = second-first
            width = float(np.linalg.norm(axis))
            if width <= radii[0]+radii[1]+2.0*(pad-env.obs_r) or width > 2.0*env.lidar_range:
                continue
            normal = np.array([-axis[1], axis[0]])/width
            if np.dot(normal, direction) < 0.0:
                normal = -normal
            # A portal crossed nearly sideways to the mission cannot
            # provide useful forward progress, even when its segment
            # happens to intersect a short geometric route to the goal.
            if float(np.dot(normal, direction)) < 0.5:
                continue
            gap = {'c1': first, 'c2': second, 'endpoint_radii': radii,
                   'wall_id': wall_id}
            interval = self.gap_interval(env, gap)
            if interval is None:
                continue
            low, high = interval
            # A geometric ordering point, never a fixed controller
            # target. The physical rollout may cross anywhere in the
            # entire safe interval.
            cross_x, cross_y, route_length, approach_distance = _portal_route_order(
                float(first[0]), float(first[1]), float(axis[0]),
                float(axis[1]), float(position[0]), float(position[1]),
                float(goal[0]), float(goal[1]), low, high)
            crossing = np.array((cross_x, cross_y))
            if float(np.dot(crossing-position, direction)) <= 0.0:
                continue
            if approach_distance > 1.5*env.lidar_range:
                continue
            crossing_heading = math.atan2(float(normal[1]), float(normal[0]))
            if preview_hull_wall_clearance(
                    crossing[0], crossing[1], crossing_heading,
                    polygons, lengths, float(env.map_w), float(env.sim_h)) < self.margin_m*env.dynamics.pixels_per_m:
                continue
            options.append((route_length, approach_distance,
                            {'pos': crossing, 'c1': first.copy(),
                             'c2': second.copy(), 'pair': pair, 'score': 0.0,
                             'endpoint_radii': radii, 'wall_id': wall_id}))
        options.sort(key=lambda item: (item[0], item[1], item[2]['pair']))
        return [item[2] for item in options]

    def choose(self, env, hits_x, hits_y, gap, adaptive_path=None,
               observed=None):
        self.last_frame = env.frame
        if observed is None:
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
        initial_clearance = preview_hull_surface_clearance(
            env.boat_pos[0], env.boat_pos[1], state[2], observed, polygons,
            lengths, float(env.map_w), float(env.sim_h))
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
        brake_prediction = None
        debug_candidates = os.environ.get('MAIN_HEAVY_MOMENTUM_TRACE') == '1'
        self.last_candidate_states = []
        families = candidate_commands(env)
        adaptive = adaptive_pursuit_command(env, adaptive_path)
        if adaptive is not None:
            speed, yaw, lookahead = adaptive
            families.append(('adaptive_pp', speed, yaw, speed, yaw))
        # Recovery motions are evaluated only after the normal library fails.
        families += recovery_commands(env)
        previous_yaw = float(getattr(env, 'command_yaw_rate', 0.))
        # CODEX allows .1 rad/s per .12 s prediction knot. Apply the same
        # rate at our .04 s steps, including inside the safety rollout.
        yaw_step = .1 * env.dt / .12
        def motion_available(item):
            moved = np.linalg.norm(item[5][-1, :2]-env.boat_pos)
            turned = abs((float(item[6][2])-float(state[2])+math.pi) % (2*math.pi)-math.pi)
            return moved >= hard_margin_px or turned >= env.dynamics.max_yaw_rate_rad_s*env.dt
        for family, speed1, yaw1, speed2, yaw2 in families:
            reverse = min(speed1, speed2) < 0.
            if reverse and (any(motion_available(item) and np.all(item[4][:, 0] >= 0.)
                                   for item in candidates) or any(
                    bound >= initial_clearance + hard_margin_px
                    and np.all(item[4][:, 0] >= 0.)
                    for bound, item in recovery_candidates)):
                continue
            commands = np.array(((speed1, yaw1), (speed2, yaw2)), dtype=np.float64)
            physical_safe, margin_safe, collision_step, trajectory, terminal, center_bound, end_bound = _simulate(
                state, commands, obstacle_origins, polygons, lengths, parameters,
                env.dt, self.horizon_steps, hard_margin_px, int(env.frame),
                float(env.map_w), float(env.sim_h), previous_yaw, yaw_step)
            if family == 'brake' and not physical_safe:
                brake_prediction = trajectory[:collision_step + 1].copy()
            if not physical_safe:
                continue
            endpoint = trajectory[-1, :2]
            end_obstacles = obstacle_origins.copy()
            shadow_move_obstacles(obstacle_origins, end_obstacles,
                                  int(env.frame)+self.horizon_steps, env.dt)
            end_clearance = preview_hull_surface_clearance(
                endpoint[0], endpoint[1], terminal[2], end_obstacles,
                polygons, lengths, float(env.map_w), float(env.sim_h))
            if reverse and (end_clearance < initial_clearance + hard_margin_px or
                    np.linalg.norm(endpoint-env.boat_pos) < hard_margin_px):
                continue
            if reverse:
                # A stationary brake is safe but is not a forward escape.
                # Keep it as fallback if no validated recovery can make room.
                candidates = [item for item in candidates if motion_available(item)]
            wall_turning_room = preview_hull_wall_turning_room(
                endpoint[0], endpoint[1], terminal[2], terminal[3],
                terminal[4], terminal[5], polygons, lengths,
                float(env.map_w), float(env.sim_h),
                env.dynamics.pixels_per_m, hard_margin_px,
                env.dynamics.actuator_tau_s + env.dynamics.yaw_response_s)
            progress = float(np.dot(endpoint - env.boat_pos,
                                    (env.target-env.boat_pos) /
                                    max(float(np.linalg.norm(env.target-env.boat_pos)), 1e-9)))
            alternative_progress = None
            normal_progress = None
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
                normal_progress = float(np.dot(endpoint-env.boat_pos, normal))
                end_s = float(np.dot(endpoint-a, axis) / length)
                off_interval = max(interval[0]-end_s, end_s-interval[1], 0.0)*length
                progress = normal_progress - off_interval
                # A blocked approach may first need lateral travel around a
                # nearer buoy. Keep the established normal score whenever a
                # safe forward candidate can make normal progress; otherwise
                # assess distance to the entire safe portal interval.
                start_s = float(np.dot(env.boat_pos-a, axis) / length)
                start_on_interval = a + np.clip(start_s, *interval) * length * axis
                end_on_interval = a + np.clip(end_s, *interval) * length * axis
                alternative_progress = (
                    float(np.linalg.norm(env.boat_pos-start_on_interval)) -
                    float(np.linalg.norm(endpoint-end_on_interval)))
                if crossing:
                    progress += 30.0
                    alternative_progress += 30.0
            else:
                crossing_s = None
                # A short horizon can reward orbiting a nearby goal forever:
                # tangential motion looks locally productive even when the
                # vessel cannot turn into the goal radius before passing it.
                # Express the missing turning room as distance, using the
                # frozen yaw response and the terminal surge/yaw state.
                goal_delta = np.asarray(env.target) - endpoint
                goal_distance = float(np.linalg.norm(goal_delta))
                goal_heading = math.atan2(float(goal_delta[1]),
                                          float(goal_delta[0]))
                heading_error = (goal_heading - float(terminal[2]) + math.pi) % (2 * math.pi) - math.pi
                max_yaw = env.dynamics.max_yaw_rate_rad_s
                yaw_toward = (float(terminal[5]) * math.copysign(1.0, heading_error)
                              if heading_error else 0.0)
                turn_time = (abs(heading_error) / max_yaw +
                             0.5 * env.dynamics.yaw_response_s *
                             max(0.0, 1.0 - yaw_toward / max_yaw))
                travel_before_alignment = (max(0.0, float(terminal[3])) *
                                           env.dynamics.pixels_per_m * turn_time)
                remaining_to_entry = max(0.0, goal_distance - 70.0)
                progress -= max(0.0, travel_before_alignment - remaining_to_entry)
            if np.linalg.norm(endpoint - env.boat_pos) < 8.0:
                desired = (math.atan2(float(normal[1]), float(normal[0]))
                           if normal is not None and interval is not None else
                           math.atan2(float(env.target[1]-env.boat_pos[1]),
                                      float(env.target[0]-env.boat_pos[0])))
                old_align = math.cos(float(state[2]) - desired)
                new_align = math.cos(float(terminal[2]) - desired)
                alignment_bonus = 0.5 * 84.0 * (new_align - old_align)
                progress += alignment_bonus
                if alternative_progress is not None:
                    alternative_progress += alignment_bonus
            continuity = 1.0 if family == self.previous_family else 0.0
            straight = 1.0 if family == 'straight' else 0.0
            # Safety has already been enforced as a hard gate. A small
            # progress tie band preserves the current command and straight.
            commands = commands.copy()
            commands[0, 1] = np.clip(commands[0, 1], previous_yaw-yaw_step,
                                      previous_yaw+yaw_step)
            item = (progress, continuity, straight, family, commands,
                    trajectory, terminal, center_bound, crossing_s,
                    alternative_progress, normal_progress, wall_turning_room)
            if debug_candidates:
                self.last_candidate_states.append((family, round(progress, 2),
                    round(float(endpoint[0]), 1), round(float(endpoint[1]), 1),
                    round(float(terminal[2]), 3), round(float(terminal[3]), 3),
                    round(float(terminal[5]), 3), bool(margin_safe),
                    None if alternative_progress is None else round(alternative_progress, 2),
                    None if normal_progress is None else round(normal_progress, 2)))
            if margin_safe:
                candidates.append(item)
            else:
                recovery_candidates.append((end_clearance, item))
        if candidates:
            self.last_detour_active = False
            turning_room_candidates = [item for item in candidates
                                       if item[11] >= 0.0]
            if turning_room_candidates:
                candidates = turning_room_candidates
            if interval is not None:
                forward_normal = [item[10] for item in candidates
                                  if item[4][0, 0] >= 0.0 and
                                  'reverse' not in item[3]]
                if (forward_normal and max(forward_normal) <= 5.0 and
                        any(item[9] is not None and item[9] > 0.0
                            for item in candidates)):
                    candidates = [(item[9],) + item[1:] for item in candidates]
                    self.last_detour_active = True
            if abs(float(state[3])) < 0.05:
                escape = [item for item in candidates
                          if item[3] not in ('brake', 'continue') and
                          (abs(item[4][0, 0]) > 1e-6 or
                           abs(item[4][0, 1]) > 1e-6)]
                if escape:
                    # A stopped incumbent is not useful continuity. Safety
                    # has already been checked for the escape primitives.
                    candidates = escape
            if any(np.all(item[4][:, 0] >= 0.) for item in candidates):
                candidates = [item for item in candidates
                              if np.all(item[4][:, 0] >= 0.)]
            self.last_scores = [(item[3], round(item[0], 2), True)
                                for item in candidates]
            adaptive_safe = [item for item in candidates if item[3] == 'adaptive_pp']
            selected = (adaptive_safe[0] if adaptive_safe else
                        select_near_tie(candidates, self.previous_family,
                            previous_command=(getattr(env, 'command_speed', 0.),
                                              getattr(env, 'command_yaw_rate', 0.)),
                            yaw_rate=float(state[5]),
                            response_s=env.dynamics.yaw_response_s))
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
                      'prediction': brake_prediction,
                      'left_pwm': 1500.0 + 400.0*left/env.dynamics.max_thrust_N,
                      'right_pwm': 1500.0 + 400.0*right/env.dynamics.max_thrust_N,
                      'speed_command': 0.0, 'yaw_rate_command': 0.0,
                      'reason': 'no_safe_candidate'}
        pair = None if gap is None else tuple(gap['pair'])
        if pair is not None and result['progress_px'] <= 0.0:
            if pair != self.no_progress_pair or self.no_progress_start_frame is None:
                self.no_progress_start_frame = env.frame
        else:
            self.no_progress_start_frame = None
        self.no_progress_pair = pair
        result['version'] = MOTION_VERSION
        result['recovery'] = result['speed_command'] < 0.
        result['portal'] = None if gap is None else {
            'c1': np.asarray(gap['c1']).copy(), 'c2': np.asarray(gap['c2']).copy(),
            'interval': interval, 'pair': pair, 'wall_id': gap.get('wall_id')}
        result['bezier_reference'] = trajectory_bezier_reference(result['prediction'])
        self.last_result = result
        return result
