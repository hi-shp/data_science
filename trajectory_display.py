"""Visual clipping of a completed prediction; never alters control state."""
import numpy as np


def future_trajectory(path, position, elapsed_steps, steps_per_knot=3):
    """Return vessel-now → future, discarding the elapsed prediction prefix.

    The physics/controller retains `path` exactly. The display recomputes this
    short array for every rendered frame from the latest vessel position. Time
    bounds the projection search so a crossing path cannot snap to an old branch.
    """
    points = np.asarray(path, dtype=float)
    boat = np.asarray(position, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return boat[None, :].copy()
    first = max(0, int(elapsed_steps)//steps_per_knot)
    if first >= len(points)-1:
        return boat[None, :].copy()
    last = min(len(points)-1, first+3)
    starts, ends = points[first:last], points[first+1:last+1]
    delta = ends-starts
    t = np.clip(np.sum((boat-starts)*delta, axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-12),0.,1.)
    projections = starts+t[:,None]*delta
    idx = int(np.argmin(np.sum((projections-boat)**2,axis=1)))+first
    # Nearest space alone can point behind a vessel that overtook its earlier
    # prediction. Advance through those segments; never draw a backward stub.
    while idx < len(points)-1:
        tangent = points[idx+1]-points[idx]
        fraction = np.clip(np.dot(boat-points[idx],tangent)/max(np.dot(tangent,tangent),1e-12),0.,1.)
        projection = points[idx]+fraction*tangent
        if np.dot(projection-boat,tangent) >= -1e-9:
            break
        idx += 1
    if idx >= len(points)-1:
        return boat[None,:].copy()
    future = points[idx+1:]
    result = [boat]
    if np.linalg.norm(projection-boat) > 1e-4:
        result.append(projection)
    result.extend(future)
    return np.asarray(result)
