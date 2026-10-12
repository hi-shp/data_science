"""Safe mission-terminal guidance, independent of the finite physical rollout."""
import numpy as np
from heavy.motion_core.goal_guidance import GOAL_ZONE_M
from heavy.gap_annotation import BEZIER_CONTROL_DEVIATION_PX


def corridor_clear(path, obstacles, hull, margin, bounds):
    """Certify continuous segments using the hull's enclosing-circle sweep.

    Conservative by construction: no sparse pose sampling or relaxed safety.
    All inputs are pixel coordinates. This is a display gate, not a planner.
    """
    points = np.asarray(path, dtype=float)
    if not len(points) or not np.isfinite(points).all():
        return False
    extent = float(np.linalg.norm(np.asarray(hull).reshape(-1, 2), axis=1).max()) + margin + BEZIER_CONTROL_DEVIATION_PX
    if (np.any(points < extent) or np.any(points[:, 0] > bounds[0]-extent) or
            np.any(points[:, 1] > bounds[1]-extent)):
        return False
    obs = np.asarray(obstacles).reshape(-1, 3)
    if not len(obs):
        return True
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    delta = np.diff(points, axis=0)
    offset = obs[:, None, :2]-points[None, :-1]
    t = np.clip(np.sum(offset*delta[None], axis=2)/
                np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0., 1.)
    distance2 = np.sum((offset-t[..., None]*delta[None])**2, axis=2)
    return bool(np.all(distance2 >= (obs[:, None, 2]+extent)**2))


def goal_guidance_path(path, headings, goal, obstacles, hull, margin, bounds, scale, visible_radius=None):
    """Keep a safe prediction prefix, then connect exactly to the goal center.

    Only the existing 12 m terminal zone and a continuously clear boat-to-goal
    hull corridor within the current sensor range enable this independent guide. Otherwise the physical future
    stays unchanged. A finite rollout may overshoot or turn away: that tail is
    not mission guidance and is never written back into the prediction.
    """
    points = np.asarray(path, dtype=float)
    angles = np.asarray(headings, dtype=float)
    goal = np.asarray(goal, dtype=float)
    extent = float(np.linalg.norm(np.asarray(hull).reshape(-1,2),axis=1).max())+margin+BEZIER_CONTROL_DEVIATION_PX
    if (len(points) and visible_radius is not None and
            np.linalg.norm(points[0]-goal)+extent > visible_radius):
        return points.copy(), angles.copy(), False
    if (not len(points) or np.linalg.norm(points[0]-goal) > GOAL_ZONE_M*scale or
            not corridor_clear(np.vstack((points[0], goal)), obstacles, hull, margin, bounds)):
        return points.copy(), angles.copy(), False
    if len(points) < 2:
        guide = np.vstack((points[0], goal))
        guide_angles = np.array([angles[0],np.arctan2(*(goal-points[0])[::-1])])
    else:
        delta = np.diff(points, axis=0)
        t = np.clip(np.sum((goal-points[:-1])*delta, axis=1)/
                    np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0., 1.)
        projections = points[:-1]+t[:, None]*delta
        closest = int(np.argmin(np.sum((projections-goal)**2, axis=1)))
        # Do not preserve a finite prediction branch that is already turning
        # away from the terminal corridor, even if a later knot returns.
        retreat = np.flatnonzero(np.sum(delta*(goal-points[:-1]), axis=1) < 0.)
        if len(retreat) and retreat[0] <= closest:
            closest = int(retreat[0]); endpoint = points[closest]
        else:
            endpoint = projections[closest]
        guide = np.vstack((points[:closest+1], endpoint, goal))
        endpoint_angle = angles[closest] if len(retreat) and retreat[0] <= closest else (
            angles[closest]+t[closest]*(angles[closest+1]-angles[closest]))
        guide_angles = np.r_[angles[:closest+1],endpoint_angle,
            np.arctan2(*(goal-endpoint)[::-1])]
        if not corridor_clear(guide, obstacles, hull, margin, bounds):
            # The direct corridor above is certified; a wandering physical
            # prefix need not be retained in independent terminal guidance.
            guide = np.vstack((points[0], goal))
            guide_angles = np.array([angles[0],np.arctan2(*(goal-points[0])[::-1])])
    keep = np.r_[True, np.linalg.norm(np.diff(guide, axis=0), axis=1) > 1e-8]
    guide = guide[keep]; guide_angles = guide_angles[keep]
    if len(guide) < 2:
        return guide, np.array([angles[0]]), True
    return guide, np.unwrap(guide_angles), True
