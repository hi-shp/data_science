"""Visual clipping of a completed prediction; never alters control state."""
import numpy as np


def end_prediction_at_goal(path, goal, radius):
    """Display a goal-reaching prediction through the center, never beyond it.

    The short final connector is visual guidance; the physical rollout remains
    in the original control path. Before the prediction reaches the goal
    region, show every available future point.
    """
    points = np.asarray(path, dtype=float)
    if len(points) < 2:
        return points.copy()
    goal = np.asarray(goal, dtype=float)
    delta = np.diff(points, axis=0)
    fraction = np.clip(np.sum((goal-points[:-1])*delta, axis=1)/
                       np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0., 1.)
    projected = points[:-1]+fraction[:, None]*delta
    distance2 = np.sum((projected-goal)**2, axis=1)
    nearby = np.flatnonzero(distance2 <= radius*radius)
    if not len(nearby):
        return points.copy()
    first = int(nearby[0])
    last = first
    while last+1 < len(distance2) and distance2[last+1] <= radius*radius:
        last += 1
    segment = first+int(np.argmin(distance2[first:last+1]))
    result = np.vstack((points[:segment+1], projected[segment], goal))
    # Avoid zero-length final line when the physical prediction hits center.
    keep = np.r_[True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-8]
    return result[keep]


def predicted_state_marker(display_path, point_index=9):
    """Pick a display-only future point, clamped to the visible path end."""
    points = np.asarray(display_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        return None
    return points[min(point_index, len(points)-1)].copy()


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
