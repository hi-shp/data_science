"""Select a feasible crossing state on an observed two-obstacle gap.

The gap pair supplies topology. This module chooses a point and heading on
the portal; it does not change vessel physics or execute a control rollout.
"""

import math

import numpy as np

from utils import wrap
from hull_collision import hull_collides


SAFETY_MARGIN_PX = 10.0  # 0.20 m at the unchanged 50 px/m display scale.
YAW_T90_S = 1.04  # Measured with the frozen MAIN_HEAVY vessel model.


def _unit(vector):
    length = math.hypot(float(vector[0]), float(vector[1]))
    return np.asarray(vector, dtype=float) / length if length > 1e-9 else None


def _endpoint_radius(center, obstacles):
    """Conservatively transfer a detected cluster to its observed obstacle."""
    if len(obstacles) == 0:
        return None
    d2 = np.sum((obstacles[:, :2] - center) ** 2, axis=1)
    nearest = int(np.argmin(d2))
    offset = math.sqrt(float(d2[nearest]))
    radius = float(obstacles[nearest, 2])
    if offset > radius + 20.0:
        return None
    return radius + offset


def _hull_support(axis, heading, hull_polygons):
    ch, sh = math.cos(heading), math.sin(heading)
    local_axis_x = axis[0] * ch + axis[1] * sh
    local_axis_y = -axis[0] * sh + axis[1] * ch
    return max(abs(x * local_axis_x + y * local_axis_y)
               for polygon in hull_polygons for x, y in polygon)


def _crossing_heading(position, crossing, downstream, normal):
    incoming = _unit(crossing - position)
    outgoing = _unit(downstream - crossing)
    if incoming is None:
        incoming = normal
    if outgoing is None:
        outgoing = normal
    tangent = _unit(incoming + outgoing)
    if tangent is None or float(np.dot(tangent, normal)) < 0.65:
        tangent = normal
    return math.atan2(float(tangent[1]), float(tangent[0]))


def _line_portal_projection(position, downstream, start, axis, normal, length):
    direction = downstream - position
    denominator = float(np.dot(direction, normal))
    if abs(denominator) > 1e-6:
        travel = float(np.dot(start - position, normal)) / denominator
        crossing = position + travel * direction
    else:
        crossing = 0.5 * (position + downstream)
    return float(np.dot(crossing - start, axis)) / length


def choose_portal_crossing(gap, position, heading, velocity_px_s, yaw_rate,
                           downstream, obstacles, hull_polygons, dynamics,
                           previous_s=None, plan_dt=0.04, diagnostic=None,
                           forced_s=None):
    """Return a gap copy with a dynamic crossing point, or None if no safe span.

    Point samples only refine a one-dimensional portal coordinate. Endpoint
    clearance is a hard constraint; the ranking has units of estimated seconds.
    """
    start = np.asarray(gap['c1'], dtype=float)
    end = np.asarray(gap['c2'], dtype=float)
    position = np.asarray(position, dtype=float)
    downstream = np.asarray(downstream, dtype=float)
    portal = end - start
    length = math.hypot(float(portal[0]), float(portal[1]))
    if length < 1e-6:
        if diagnostic is not None:
            diagnostic['reason'] = 'degenerate'
        return None
    axis = portal / length
    normal = np.array([-axis[1], axis[0]], dtype=float)
    if float(np.dot(normal, downstream - position)) < 0:
        normal = -normal

    radius_a = _endpoint_radius(start, obstacles)
    radius_b = _endpoint_radius(end, obstacles)
    if radius_a is None or radius_b is None:
        if diagnostic is not None:
            diagnostic['reason'] = 'endpoint_unmatched'
        return None
    normal_heading = math.atan2(float(normal[1]), float(normal[0]))
    nominal_width = _hull_support(axis, normal_heading, hull_polygons)
    base_min = (radius_a + nominal_width + SAFETY_MARGIN_PX) / length
    base_max = 1.0 - (radius_b + nominal_width + SAFETY_MARGIN_PX) / length
    if base_min > base_max:
        if diagnostic is not None:
            diagnostic['reason'] = 'narrow_interval'
        return None

    projected = _line_portal_projection(position, downstream, start, axis,
                                        normal, length)
    if forced_s is not None:
        samples = [float(forced_s)]
    else:
        samples = list(np.linspace(base_min, base_max, 9))
        samples.append(float(np.clip(projected, base_min, base_max)))
        if previous_s is not None:
            samples.append(float(np.clip(previous_s, base_min, base_max)))

    px_per_m = dynamics.pixels_per_m
    speed_m_s = math.hypot(float(velocity_px_s[0]),
                          float(velocity_px_s[1])) / px_per_m
    cruise_m_s = dynamics.cruise_speed_m_s
    best = None
    for s in samples:
        crossing = start + s * portal
        crossing_heading = _crossing_heading(position, crossing, downstream,
                                             normal)
        lateral_half = _hull_support(axis, crossing_heading, hull_polygons)
        safe_min = (radius_a + lateral_half + SAFETY_MARGIN_PX) / length
        safe_max = 1.0 - (radius_b + lateral_half + SAFETY_MARGIN_PX) / length
        if not safe_min - 1e-8 <= s <= safe_max + 1e-8:
            continue

        incoming_m = math.dist(position, crossing) / px_per_m
        outgoing_m = math.dist(crossing, downstream) / px_per_m
        turn = wrap(crossing_heading - heading)
        aligned_yaw = max(0.0, math.copysign(yaw_rate, turn))
        remaining_turn = max(0.0, abs(turn) - aligned_yaw * YAW_T90_S)
        response_fraction = min(1.0, abs(turn) / 0.35)
        turn_time = ((dynamics.actuator_tau_s + YAW_T90_S) * response_fraction
                     + remaining_turn / dynamics.max_yaw_rate_rad_s)
        approach_time = incoming_m / max(speed_m_s, 0.35)
        turn_shortfall = max(0.0, turn_time - approach_time)
        travel_time = ((incoming_m + outgoing_m) /
                       max(cruise_m_s, speed_m_s, 0.35))
        # A small deterministic continuity tie-break only; safety and useful
        # physical travel time dominate it.
        continuity = 0.0 if previous_s is None else abs(s - previous_s) * 0.02
        cost = travel_time + turn_shortfall + continuity
        if best is None or cost < best[0]:
            best = (cost, s, crossing, crossing_heading, safe_min, safe_max,
                    turn_shortfall)
    if best is None:
        if diagnostic is not None:
            diagnostic['reason'] = 'no_heading_safe_sample'
        return None

    _, optimal_s, _, _, _, _, _ = best
    if previous_s is not None and forced_s is None:
        # Keep the selected portal identity while the crossing point moves at
        # the measured yaw-response time scale. Project immediately into the
        # newly safe interval if the observed obstacle geometry changes.
        factor = -math.expm1(-plan_dt / YAW_T90_S)
        optimal_s = previous_s + factor * (optimal_s - previous_s)
        optimal_s = float(np.clip(optimal_s, base_min, base_max))
    crossing = start + optimal_s * portal
    crossing_heading = _crossing_heading(position, crossing, downstream,
                                         normal)
    lateral_half = _hull_support(axis, crossing_heading, hull_polygons)
    safe_min = (radius_a + lateral_half + SAFETY_MARGIN_PX) / length
    safe_max = 1.0 - (radius_b + lateral_half + SAFETY_MARGIN_PX) / length
    if safe_min > safe_max:
        if diagnostic is not None:
            diagnostic['reason'] = 'smoothed_interval_empty'
        return None
    optimal_s = float(np.clip(optimal_s, safe_min, safe_max))
    crossing = start + optimal_s * portal
    crossing_heading = _crossing_heading(position, crossing, downstream,
                                         normal)
    result = gap.copy()
    result['portal_midpoint'] = 0.5 * (start + end)
    result['portal_safe_interval'] = (safe_min, safe_max)
    result['portal_s'] = float(optimal_s)
    result['portal_heading'] = crossing_heading
    result['portal_turn_shortfall_s'] = best[6]
    result['portal_cost_s'] = best[0]
    result['pos'] = crossing.astype(np.float32)
    if diagnostic is not None:
        diagnostic['reason'] = 'accepted'
    return result


def path_has_hull_clearance(path, obstacles, hull_polygons,
                            margin_px=SAFETY_MARGIN_PX):
    """Check the planned centerline with the actual oriented hull footprint.

    This is a geometric path check, not a substitute for the online dynamic
    safety guard. Broad-phase center distances avoid polygon work on distant
    obstacles. At most 5 px between checked points limits swept gaps.
    """
    if path is None or len(path) < 2 or len(obstacles) == 0:
        return True
    maximum_extent = max(math.hypot(x, y) for poly in hull_polygons for x, y in poly)
    inflated = np.asarray(obstacles, dtype=float).copy()
    inflated[:, 2] += margin_px
    last = np.asarray(path[0], dtype=float)
    for endpoint in path[1:]:
        endpoint = np.asarray(endpoint, dtype=float)
        vector = endpoint - last
        distance = math.hypot(float(vector[0]), float(vector[1]))
        if distance < 1e-9:
            continue
        heading = math.atan2(float(vector[1]), float(vector[0]))
        divisions = max(1, math.ceil(distance / 5.0))
        for index in range(divisions + 1):
            point = last + vector * (index / divisions)
            delta = inflated[:, :2] - point
            near = np.sum(delta * delta, axis=1) < (inflated[:, 2] + maximum_extent) ** 2
            if np.any(near) and hull_collides(point, heading, inflated[near], hull_polygons):
                return False
        last = endpoint
    return True


def remaining_path(path, position):
    """Project onto a polyline segment and discard already traversed geometry."""
    if path is None or len(path) < 2:
        return path
    points = np.asarray(path)
    segment = points[1:] - points[:-1]
    lengths2 = np.sum(segment * segment, axis=1)
    along = np.clip(np.sum((position - points[:-1]) * segment, axis=1) /
                    np.maximum(lengths2, 1e-9), 0.0, 1.0)
    projection = points[:-1] + along[:, None] * segment
    index = int(np.argmin(np.sum((projection - position) ** 2, axis=1)))
    return np.vstack((projection[index], points[index + 1:]))


def portal_crossing_status(gap, position, previous_signed=None):
    """Return (crossed, signed distance) for a fixed approach-side normal."""
    first = np.asarray(gap['c1'], dtype=float)
    second = np.asarray(gap['c2'], dtype=float)
    axis = second - first
    width = math.hypot(float(axis[0]), float(axis[1]))
    if width < 1e-6:
        return False, 0.0
    axis /= width
    normal = np.array([-axis[1], axis[0]])
    direction = np.array([math.cos(gap['portal_heading']),
                          math.sin(gap['portal_heading'])])
    if np.dot(normal, direction) < 0:
        normal = -normal
    offset = np.asarray(position, dtype=float) - first
    signed = float(np.dot(offset, normal))
    lateral_s = float(np.dot(offset, axis)) / width
    low, high = gap['portal_safe_interval']
    crossed = (previous_signed is not None and previous_signed <= 0.0 < signed
               and low <= lateral_s <= high)
    return crossed, signed
