"""Presentation-only terminal state. No remembered path is a safety authority."""
import numpy as np


def goal_location(path, goal, radius):
    points=np.asarray(path,dtype=float)
    if len(points)<2:return None
    delta=np.diff(points,axis=0)
    fraction=np.clip(np.sum((goal-points[:-1])*delta,axis=1)/
                     np.maximum(np.sum(delta*delta,axis=1),1e-12),0.,1.)
    projected=points[:-1]+fraction[:,None]*delta
    d2=np.sum((projected-goal)**2,axis=1)
    near=np.flatnonzero(d2<=radius*radius)
    if not len(near):return None
    first=last=int(near[0])
    while last+1<len(d2) and d2[last+1]<=radius*radius:last+=1
    i=first+int(np.argmin(d2[first:last+1]))
    return i, fraction[i], projected[i]


def goal_prefix(path, headings, goal, radius):
    event=goal_location(path,goal,radius)
    if event is None:return None
    i,t,point=event
    points=np.vstack((path[:i+1],point,goal))
    connector=goal-point
    angle=np.arctan2(connector[1],connector[0]) if np.linalg.norm(connector)>1e-8 else headings[i]
    angles=np.r_[headings[:i+1],headings[i]+t*(headings[i+1]-headings[i]),angle]
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-8]
    return points[keep],np.unwrap(angles[keep])


class TerminalDisplayState:
    """Latch terminal PRESENTATION, never clearance or a former safe connector.

    A future mission arrival arms the state. Later GAP confirmation cannot
    switch the display endpoint back to a nearby waypoint. Each update must
    certify its own goal connection. On failure after connection, suppress the
    guide instead of flashing an unrelated finite-prediction tail.
    """
    def __init__(self):self.reset()
    def reset(self):
        self.terminal=False;self.ever_connected=False;self.frame=-1;self.goal=None
        self.reason='FINITE_ROLLOUT_ONLY'

    def update(self,path,headings,goal,frame,zone,mission_radius,connect,clip):
        points=np.asarray(path,dtype=float);angles=np.asarray(headings,dtype=float)
        goal=np.asarray(goal,dtype=float)
        if frame<self.frame or self.goal is None or not np.array_equal(goal,self.goal):self.reset()
        self.frame=frame;self.goal=goal.copy()
        if not len(points) or not np.isfinite(points).all():
            self.reason='INVALID_PREDICTION'
            return np.empty((0,2)),np.empty(0),False
        if np.linalg.norm(points[0]-goal)>zone:
            self.terminal=False;self.ever_connected=False
        elif goal_location(points,goal,mission_radius) is not None:
            self.terminal=True
        guide,guide_angles,connected,reason=connect(points,angles)
        if connected:
            self.terminal=True;self.ever_connected=True;self.reason='GOAL_CONNECTED'
            # Assign from the exact world goal, not a lower-precision SI conversion.
            guide=guide.copy();guide[-1]=goal
            return guide,guide_angles,True
        self.reason=reason
        if self.terminal:
            if self.ever_connected:
                self.reason='GUIDANCE_SUPPRESSED_'+reason
                return points[:1].copy(),angles[:1].copy(),False
            # A finite physical prediction remains finite. Hide its post-mission
            # tail, and restrict it to currently observable hull support.
            event=goal_location(points,goal,mission_radius)
            if event is not None:
                i,t,point=event
                points=np.vstack((points[:i+1],point))
                angles=np.r_[angles[:i+1],angles[i]+t*(angles[i+1]-angles[i])]
            limited=clip(points)
            # Safety clipping inserts at most one new terminal point.
            lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            new_lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(limited,axis=0),axis=1))]
            angles=np.interp(new_lengths,lengths,angles)
            return limited,angles,False
        return points.copy(),angles.copy(),False


def corridor_clear(path, obstacles, hull, margin, bounds, reserve=.25):
    """Certify continuous segments using the hull's enclosing-circle sweep.

    Conservative by construction: no sparse pose sampling or relaxed safety.
    All inputs are pixel coordinates. This is a display gate, not a planner.
    """
    points = np.asarray(path, dtype=float)
    if not len(points) or not np.isfinite(points).all():
        return False
    extent = float(np.linalg.norm(np.asarray(hull).reshape(-1, 2), axis=1).max()) + margin + reserve
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


def safe_goal_path(path, headings, goal, obstacles, hull, margin, bounds, scale, visible_radius=None, prediction_clearance=None, *, goal_zone_m=12., reserve=.25):
    """Keep a safe prediction prefix, then connect exactly to the goal center.

    Only the existing 12 m terminal zone and a continuously clear boat-to-goal
    hull corridor within the current sensor range enable this independent guide. Otherwise the physical future
    stays unchanged. A finite rollout may overshoot or turn away: that tail is
    not mission guidance and is never written back into the prediction.
    """
    points = np.asarray(path, dtype=float)
    angles = np.asarray(headings, dtype=float)
    goal = np.asarray(goal, dtype=float)
    extent = float(np.linalg.norm(np.asarray(hull).reshape(-1,2),axis=1).max())+margin+reserve
    if (len(points) and visible_radius is not None and
            np.linalg.norm(points[0]-goal)+extent > visible_radius):
        return points.copy(), angles.copy(), False
    if not len(points) or np.linalg.norm(points[0]-goal) > goal_zone_m*scale:
        return points.copy(), angles.copy(), False
    if not corridor_clear(np.vstack((points[0], goal)), obstacles, hull, margin, bounds, reserve):
        # The selected physical prefix already passed exact swept-hull safety.
        # A curved approach can be safe while its direct chord is blocked.
        # Recheck the full displayed prefix and connector against CURRENT
        # observations too; a previous rollout certificate cannot bypass a
        # new blocker. The enclosing-circle sweep is conservative.
        prefix=goal_prefix(points,angles,goal,.55*scale)
        if prefix is not None and prediction_clearance is not None:
            guide,guide_angles=prefix
            covered=(visible_radius is None or np.all(
                np.linalg.norm(guide-points[0],axis=1)+extent<=visible_radius))
            if (prediction_clearance*scale>=margin+reserve and covered and
                    corridor_clear(guide,obstacles,hull,margin,bounds,reserve)):
                return guide,guide_angles,True
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
        if not corridor_clear(guide, obstacles, hull, margin, bounds, reserve):
            # The direct corridor above is certified; a wandering physical
            # prefix need not be retained in independent terminal guidance.
            guide = np.vstack((points[0], goal))
            guide_angles = np.array([angles[0],np.arctan2(*(goal-points[0])[::-1])])
    if visible_radius is not None and np.any(np.linalg.norm(guide-points[0],axis=1)+extent>visible_radius):
        # The certified direct chord stays in the convex sensor-support disk.
        guide=np.vstack((points[0],goal))
        guide_angles=np.array([angles[0],np.arctan2(*(goal-points[0])[::-1])])
    keep = np.r_[True, np.linalg.norm(np.diff(guide, axis=0), axis=1) > 1e-8]
    guide = guide[keep]; guide_angles = guide_angles[keep]
    if len(guide) < 2:
        return guide, np.array([angles[0]]), True
    return guide, np.unwrap(guide_angles), True
