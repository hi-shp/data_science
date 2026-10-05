"""Visual clipping of a completed prediction; never alters control state."""
import math
import numpy as np


def clip_trajectory_to_radius(path, center, radius):
    """Keep the visible trajectory through its first exit from a moving circle.

    The result is a new display array beginning at the current vessel pose;
    the completed physical rollout is never shortened or modified.
    """
    boat = np.asarray(center, dtype=float)
    points = np.asarray(path, dtype=float) if path is not None else np.empty((0, 2))
    result = [boat.copy()]
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
            not np.isfinite(boat).all() or not np.isfinite(radius) or radius <= 0):
        return np.asarray(result)
    radius_sq = radius * radius
    for point in points[1:]:
        if not np.isfinite(point).all():
            break
        offset = point - boat
        if float(np.dot(offset, offset)) <= radius_sq:
            result.append(point.copy())
            continue
        inside = result[-1]
        direction = point - inside
        a = float(np.dot(direction, direction))
        if a > 0:
            relative = inside - boat
            b = 2.0 * float(np.dot(relative, direction))
            c = float(np.dot(relative, relative)) - radius_sq
            discriminant = max(0.0, b * b - 4.0 * a * c)
            fraction = min(1.0, max(0.0, (-b + math.sqrt(discriminant)) / (2.0 * a)))
            intersection = inside + fraction * direction
            # Guard against a roundoff point just outside the visible circle.
            distance = float(np.linalg.norm(intersection - boat))
            if distance > radius:
                intersection = boat + (intersection - boat) * (radius / distance)
            if np.linalg.norm(intersection - inside) > 1e-12:
                result.append(intersection)
        break
    return np.asarray(result)


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


def _project_onto_path(points, point):
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    fraction = np.clip(np.sum((point-points[:-1])*segments, axis=1) /
                       np.maximum(lengths*lengths, 1e-12), 0., 1.)
    projections = points[:-1]+fraction[:, None]*segments
    index = int(np.argmin(np.sum((projections-point)**2, axis=1)))
    arc = np.r_[0., np.cumsum(lengths)]
    return projections[index], arc[index]+fraction[index]*lengths[index], float(
        np.linalg.norm(projections[index]-point)), arc


def _point_at_arc(points, arc, distance):
    distance = np.clip(distance, 0., arc[-1])
    index = min(int(np.searchsorted(arc, distance, side='right'))-1, len(points)-2)
    index = max(0, index)
    fraction = (distance-arc[index])/max(arc[index+1]-arc[index], 1e-12)
    return points[index]+fraction*(points[index+1]-points[index])


def predicted_state_marker(display_path, point_index=9, *, prediction_path=None,
                           progress=0., anchor=None):
    """Interpolate a display-only future state on the visible polyline.

    `prediction_path` is the unchanged physical rollout. `progress` advances
    between its knots; the projected result stays on the clipped GUI path.
    On a nearby replanning replacement, `anchor` releases the previous marker
    smoothly over one prediction stride. A genuinely different path is used
    immediately.
    """
    points = np.asarray(display_path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        return None
    if prediction_path is None:
        return points[min(point_index, len(points)-1)].copy()
    if len(points) < 2:
        return points[0].copy()
    prediction = np.asarray(prediction_path, dtype=float)
    if prediction.ndim != 2 or prediction.shape[1] != 2 or not len(prediction):
        return points[-1].copy()
    progress = float(np.clip(progress, 0., 1.))
    base_index = min(point_index, len(prediction)-1)
    next_index = min(base_index+1, len(prediction)-1)
    candidate = prediction[base_index]*(1.-progress)+prediction[next_index]*progress
    projected, desired_arc, _, arc = _project_onto_path(points, candidate)
    if anchor is None:
        return projected.copy()
    anchor = np.asarray(anchor, dtype=float)
    prior, prior_arc, prior_distance, _ = _project_onto_path(points, anchor)
    _, base_arc, _, _ = _project_onto_path(points, prediction[base_index])
    knot_length = np.linalg.norm(prediction[next_index]-prediction[base_index])
    nearby = 2.*max(0.1, knot_length)
    if prior_distance > nearby or abs(prior_arc-base_arc) > 2.*nearby:
        return projected.copy()
    marker_arc = desired_arc-(1.-progress)*(base_arc-prior_arc)
    return _point_at_arc(points, arc, marker_arc)


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
