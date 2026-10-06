"""Exact circular-obstacle contact with the existing three-part MAIN hull."""

import math

import numpy as np


def hull_collides(position, heading, obstacles, polygons):
    """Return the same hull/obstacle contact result as BoatEnv.collide()."""
    if len(obstacles) == 0:
        return False

    bx, by = position
    ch = math.cos(heading)
    sh = math.sin(heading)
    ox = obstacles[:, 0]
    oy = obstacles[:, 1]
    radii = obstacles[:, 2]
    dx = ox - bx
    dy = oy - by
    x_local = dx * ch + dy * sh
    y_local = -dx * sh + dy * ch
    candidate = (abs(x_local) <= 42.0 + radii) & (abs(y_local) <= 27.0 + radii)
    if not np.any(candidate):
        return False

    for index in np.where(candidate)[0]:
        px = x_local[index]
        py = y_local[index]
        radius_sq = radii[index] * radii[index]
        for polygon in polygons:
            inside = False
            count = len(polygon)
            for i in range(count):
                j = (i - 1) % count
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
            if inside:
                return True
            for i in range(count):
                x1, y1 = polygon[i]
                x2, y2 = polygon[(i + 1) % count]
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
