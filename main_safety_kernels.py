"""Compiled preview-only LiDAR and hull kernels; reference code remains authoritative."""

import math

import numpy as np
from numba import njit

from perception import _COS_REL, _SIN_REL

COS_REL = _COS_REL[:, 0].copy()
SIN_REL = _SIN_REL[:, 0].copy()


@njit(cache=True)
def preview_lidar_distances(x, y, heading, obstacles, lidar_range):
    out = np.empty(COS_REL.shape[0], dtype=np.float32)
    ch = math.cos(heading)
    sh = math.sin(heading)
    for i in range(out.shape[0]):
        direction_x = np.float32(np.float32(ch * COS_REL[i]) - np.float32(sh * SIN_REL[i]))
        direction_y = np.float32(np.float32(sh * COS_REL[i]) + np.float32(ch * SIN_REL[i]))
        nearest = lidar_range
        for j in range(obstacles.shape[0]):
            px = np.float32(obstacles[j, 0] - x)
            py = np.float32(obstacles[j, 1] - y)
            radius = obstacles[j, 2]
            distance_sq = np.float32(np.float32(px * px) + np.float32(py * py))
            if distance_sq >= (lidar_range + radius) ** 2:
                continue
            b = np.float32(np.float32(px * direction_x) + np.float32(py * direction_y))
            disc = np.float32(np.float32(radius * radius - distance_sq) + np.float32(b * b))
            if b > 0.0 and disc >= 0.0:
                hit = np.float32(b - np.float32(math.sqrt(disc)))
                if hit < nearest:
                    nearest = hit
        out[i] = nearest
    return out


@njit(cache=True)
def preview_hull_collides(x, y, heading, obstacles, polygons, lengths):
    ch = math.cos(heading)
    sh = math.sin(heading)
    for obstacle in obstacles:
        dx = obstacle[0] - x
        dy = obstacle[1] - y
        px = dx * ch + dy * sh
        py = -dx * sh + dy * ch
        radius = obstacle[2]
        if abs(px) > 42.0 + radius or abs(py) > 27.0 + radius:
            continue
        radius_sq = radius * radius
        for part in range(3):
            count = lengths[part]
            inside = False
            for i in range(count):
                j = (i - 1) % count
                xi, yi = polygons[part, i]
                xj, yj = polygons[part, j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
            if inside:
                return True
            for i in range(count):
                x1, y1 = polygons[part, i]
                x2, y2 = polygons[part, (i + 1) % count]
                vx = x2 - x1
                vy = y2 - y1
                segment_sq = vx * vx + vy * vy
                if segment_sq < 1e-8:
                    distance_sq = (px - x1) ** 2 + (py - y1) ** 2
                else:
                    t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / segment_sq))
                    cx = x1 + t * vx
                    cy = y1 + t * vy
                    distance_sq = (px - cx) ** 2 + (py - cy) ** 2
                if distance_sq <= radius_sq:
                    return True
    return False


def packed_hulls(polygons):
    packed = np.zeros((3, 8, 2), dtype=np.float64)
    lengths = np.array([len(polygon) for polygon in polygons], dtype=np.int64)
    for part, polygon in enumerate(polygons):
        packed[part, :lengths[part]] = polygon
    return packed, lengths


def warm_preview_kernels():
    empty = np.empty((0, 3), dtype=np.float32)
    polygons = np.zeros((3, 8, 2), dtype=np.float64)
    lengths = np.array([8, 8, 4], dtype=np.int64)
    preview_lidar_distances(0.0, 0.0, 0.0, empty, 320.0)
    preview_hull_collides(0.0, 0.0, 0.0, empty, polygons, lengths)
    preview_follow_steering(
        0.0, 0.0, 0.0, 0.0, 0.0, False, 0,
        np.empty((0, 2), dtype=np.float32),
        np.full(180, 320.0, dtype=np.float32),
        np.linspace(-math.pi, math.pi, 180, endpoint=False),
        1.1, 0.3515, 0.05, 0.7, 125.0, 160.0, 18, True,
    )
    preview_follow_steering(
        0.0, 0.0, 0.0, 0.0, 0.0, False, 0,
        np.zeros((2, 2), dtype=np.float64),
        np.full(180, 320.0, dtype=np.float32),
        np.linspace(-math.pi, math.pi, 180, endpoint=False),
        1.1, 0.3515, 0.05, 0.7, 125.0, 160.0, 18, True,
    )
    preview_pwm(0.0, 270.36)
    angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
    for path in (np.empty((0, 2), dtype=np.float32),
                 np.zeros((2, 2), dtype=np.float64)):
        preview_rollout(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500, 1500,
            320.0, 0.0, False, 0, path, True, empty, empty,
            polygons, lengths, angles, 0.04, 10.0, 0.2, 0.8, 4.5,
            0.00665, 320.0, 1.1, 0.3515, 0.05, 0.7,
            125.0, 160.0, 18, 270.36, 0.0, 0, 1,
        )


@njit(cache=True)
def _clip(value, minimum, maximum):
    return max(minimum, min(maximum, value))


@njit(cache=True)
def preview_follow_steering(x, y, heading, yaw_rate, previous_steer,
                            emergency_mode, cooldown, path, distances,
                            rel_angles, steer_gain_base, alpha,
                            avoid_normal, avoid_em, em_enter, em_exit, em_hold,
                            has_waypoint):
    """MAIN update_steering + Pure Pursuit, evaluated on the shadow state."""
    min_front = 1e9
    span = int(180 * 220 / 360 / 2)
    for i in range(90 - span, 90 + span):
        min_front = min(min_front, distances[i])
    if min_front < em_enter:
        emergency_mode = True
        cooldown = em_hold
    elif emergency_mode:
        cooldown -= 1
        if min_front > em_exit and cooldown <= 0:
            emergency_mode = False
    if path.shape[0] == 0:
        return 0.0, previous_steer, emergency_mode, cooldown, min_front

    target_x = path[-1, 0]
    target_y = path[-1, 1]
    for i in range(path.shape[0]):
        dx = path[i, 0] - x
        dy = path[i, 1] - y
        if dx * dx + dy * dy > 70.0 * 70.0:
            target_x = path[i, 0]
            target_y = path[i, 1]
            break
    heading_target = math.atan2(target_y - y, target_x - x)
    heading_error = (heading_target - heading + math.pi) % (2.0 * math.pi) - math.pi
    clear_ratio = _clip((min_front - 150.0) / 50.0, 0.0, 1.0)
    gain = steer_gain_base + (1.0 - clear_ratio) * 0.4
    avoid_multiplier = avoid_normal + (1.0 - clear_ratio) * avoid_em * 0.40
    steer_raw = heading_error * gain - 0.12 * yaw_rate
    steer_f = alpha * steer_raw + (1.0 - alpha) * previous_steer
    previous_steer = steer_f

    if not has_waypoint:
        fov = 1.134464
        closest_index = -1
        closest_distance = 999.0
        d_left = 999.0
        d_right = 999.0
        for i in range(distances.shape[0]):
            angle = rel_angles[i]
            if abs(angle) <= fov:
                if distances[i] < closest_distance:
                    closest_distance = distances[i]
                    closest_index = i
                if angle < -0.05:
                    d_left = min(d_left, distances[i])
                elif angle > 0.05:
                    d_right = min(d_right, distances[i])
        if closest_index >= 0 and closest_distance < 100.0:
            closest_angle = rel_angles[closest_index]
            push_right = max(0.0, (100.0 - d_left) / 100.0) ** 1.5
            push_left = max(0.0, (100.0 - d_right) / 100.0) ** 1.5
            net_direction = push_right - push_left
            urgency = _clip((100.0 - closest_distance) / 40.0, 0.0, 1.0)
            front_factor = max(0.0, math.cos(closest_angle * (math.pi / 2.0 / fov)))
            if closest_distance < 60.0:
                if abs(closest_angle) > 0.04:
                    avoid_direction = -1.0 if closest_angle > 0.0 else 1.0
                else:
                    avoid_direction = -1.0 if d_left >= d_right else 1.0
                avoid_steer = avoid_direction * (0.75 + 0.25 * urgency)
                if closest_distance < 55.0:
                    steer_command = avoid_direction * 0.3
                else:
                    weight = min(0.50, urgency * front_factor)
                    steer_command = (1.0 - weight) * steer_f + weight * avoid_steer
            else:
                steer_command = steer_f + _clip(net_direction * 0.35, -0.45, 0.45)
            flank_distance = 999.0
            flank_angle = 0.0
            for i in range(distances.shape[0]):
                angle = rel_angles[i]
                if fov < abs(angle) <= 1.658 and distances[i] < flank_distance:
                    flank_distance = distances[i]
                    flank_angle = angle
            if flank_distance < 42.0:
                push_direction = -1.0 if flank_angle > 0.0 else 1.0
                steer_command = _clip(steer_command + push_direction *
                                      (42.0 - flank_distance) / 42.0 * 0.40,
                                      -1.0, 1.0)
            return _clip(steer_command, -1.0, 1.0), previous_steer, emergency_mode, cooldown, min_front

    avoidance = 0.0
    for i in range(distances.shape[0]):
        distance = distances[i]
        if distance < 450.0:
            angle = rel_angles[i]
            weight = math.exp(-((distance / 150.0) ** 2))
            front = max(1.2 - abs(angle) / (math.pi / 2.0), 0.3)
            avoidance += -weight * front * math.sin(angle)
    if steer_f * avoidance < 0.0 and abs(steer_f) > 0.15:
        avoidance *= 0.25
    for i in range(distances.shape[0]):
        if abs(rel_angles[i]) >= math.pi / 2.0 - 1e-5 and distances[i] <= 45.3:
            return 0.0, previous_steer, emergency_mode, cooldown, min_front
    return _clip(steer_f + avoid_multiplier * avoidance, -1.0, 1.0), previous_steer, emergency_mode, cooldown, min_front


@njit(cache=True)
def preview_pwm(steer, pwm_range):
    if abs(steer) < 0.02:
        steer = 0.0
    displacement = abs(steer) ** 1.15 * pwm_range
    if steer >= 0.0:
        left = 1500.0 - displacement
        right = 1500.0 + displacement
    else:
        left = 1500.0 + displacement
        right = 1500.0 - displacement
    return int(_clip(left, 1230.0, 1770.0)), int(_clip(right, 1230.0, 1770.0))


@njit(cache=True)
def preview_rollout(x, y, vx, vy, heading, yaw, forward, left, right,
                    min_wide, previous_steer, emergency_mode, cooldown,
                    path, has_waypoint, nearby, lidar_obstacles, polygons, lengths,
                    rel_angles, dt, mass, drag, rotational_drag, inertia,
                    moment_coefficient, lidar_range, steer_gain, alpha,
                    avoid_normal, avoid_em, em_enter, em_exit, em_hold, pwm_range,
                    override_turn, override_steps, horizon):
    """The MAIN control/dynamics shadow in one compiled, allocation-light loop."""
    for index in range(horizon):
        if index:
            distances = preview_lidar_distances(x, y, heading, lidar_obstacles, lidar_range)
            steer, previous_steer, emergency_mode, cooldown, min_wide = \
                preview_follow_steering(
                    x, y, heading, yaw, previous_steer,
                    emergency_mode, cooldown, path, distances,
                    rel_angles, steer_gain, alpha, avoid_normal, avoid_em,
                    em_enter, em_exit, em_hold, has_waypoint,
                )
            left, right = preview_pwm(steer, pwm_range)
        if index < override_steps:
            left, right = preview_pwm(override_turn, pwm_range)
        target_forward = ((left + right) * 10.0 / 6.0) * \
            (math.tanh(max(0.0, min_wide) / 100.0) ** 1.35)
        forward = forward * 0.90 + target_forward * 0.10
        cosine = math.cos(heading)
        sine = math.sin(heading)
        speed = math.hypot(vx, vy)
        lateral = -vx * sine + vy * cosine
        drag_x = -drag * vx * speed + sine * lateral * 18.0
        drag_y = -drag * vy * speed - cosine * lateral * 18.0
        vx += (forward / mass * cosine + drag_x) * dt
        vy += (forward / mass * sine + drag_y) * dt
        x = float(np.float32(x + vx * dt))
        y = float(np.float32(y + vy * dt))
        moment = (right - left) * 10.0 * moment_coefficient
        yaw += (moment - rotational_drag * yaw) / inertia * dt
        yaw *= 0.84
        heading += yaw * dt
        x = float(np.float32(x - math.sin(heading) * yaw * 4.0 * dt))
        y = float(np.float32(y + math.cos(heading) * yaw * 4.0 * dt))
        if preview_hull_collides(x, y, heading, nearby, polygons, lengths):
            return index + 1, x, y
    return 0, x, y
