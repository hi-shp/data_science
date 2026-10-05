"""Display-only hit classification; scan arrays and ranges stay untouched."""
import numpy as np


def obstacle_hit_mask(hits_x, hits_y, map_width, map_height):
    """Keep finite non-wall hits, including float32 boundary roundoff."""
    x, y = np.asarray(hits_x), np.asarray(hits_y)
    epsilon = 4 * np.finfo(np.float32).eps * max(map_width, map_height, 1.)
    wall = ((np.abs(x) <= epsilon) | (np.abs(x-map_width) <= epsilon)
            | (np.abs(y) <= epsilon) | (np.abs(y-map_height) <= epsilon))
    return np.isfinite(x) & np.isfinite(y) & ~wall
