"""Predictive local avoidance using the same lagged dynamics as the vessel."""
import math
from dataclasses import dataclass
import numpy as np
from vessel_dynamics import allocate, integrate
from utils import wrap
from route_planner import route_target
from fast_constant_rollout import compiled_constant_rollout, parameter_vector


@dataclass(frozen=True)
class ControllerParameters:
    planning_period_steps: int = 3  # 0.12 s, independent of display frame rate
    horizon_s: float = 5.0
    prediction_step_s: float = 0.2
    lookahead_m: float = 3.0
    safety_margin_m: float = 0.20
    target_wall_margin_m: float = 1.8
    yaw_samples: int = 21
    heading_gain: float = 0.65
    cross_track_weight: float = 0.18
    turning_weight: float = 0.22
    command_change_weight: float = 3.0
    terminal_heading_weight: float = 0.40
    goal_distance_weight: float = 0.20

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self,name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.prediction_step_s > self.horizon_s:
            raise ValueError('prediction_step_s must not exceed horizon_s')
        if self.planning_period_steps != int(self.planning_period_steps) or self.yaw_samples != int(self.yaw_samples):
            raise ValueError('planning_period_steps and yaw_samples must be integers')


def select_command(env):
    p = env.dynamics
    cfg = env.control
    if (env.frame-1) % cfg.planning_period_steps and hasattr(env, 'command_yaw_rate'):
        return env.command_yaw_rate / p.max_yaw_rate_rad_s
    state = env.physics_state()
    if math.hypot(state[3],state[4]) < .10:
        env.stalled_s = getattr(env,'stalled_s',0.) + env.dt*cfg.planning_period_steps
    else:
        env.stalled_s = 0.
    if env.stalled_s > 2.5:
        env.recovery_until = env.frame*env.dt + 3.0
        env.stalled_s = 0.
    # One authoritative, perception-validated control path and continuous target.
    # The rollout independently checks the full swept hull, not a point.
    repaired_target = route_target(env)
    path_unavailable = repaired_target is None
    # No raw/legacy route fallback. A missing safe curve enters the existing
    # hull-checked reverse recovery, rather than disabling recovery by returning.
    target = (state[:2]+np.array([math.cos(state[2]),math.sin(state[2])])*cfg.lookahead_m
              if path_unavailable else repaired_target)
    target = target.copy()
    target[1] = np.clip(target[1], cfg.target_wall_margin_m, env.sim_h/p.pixels_per_m-cfg.target_wall_margin_m)
    env.controller_target = target * p.pixels_per_m
    angle = math.atan2(target[1]-state[1], target[0]-state[0])
    env.heading_target = angle
    desired = np.clip(wrap(angle-state[2])*cfg.heading_gain, -p.max_yaw_rate_rad_s, p.max_yaw_rate_rad_s)
    rates = np.unique(np.r_[np.linspace(-p.max_yaw_rate_rad_s,p.max_yaw_rate_rad_s,cfg.yaw_samples), desired,
                             getattr(env,'command_yaw_rate',0.)])
    rate_cmd = np.tile(rates, 5)
    speeds = np.repeat(np.array([1., .65, .3, 0., -.25])*p.cruise_speed_m_s, len(rates))
    z = np.repeat(state[None,:],len(speeds),axis=0)
    closest = np.full(len(speeds), 10.)
    blocked = np.zeros(len(speeds), dtype=bool)
    trajectory = np.empty((len(speeds),round(cfg.horizon_s/cfg.prediction_step_s),8))
    local_obs = env.perceived_obstacles
    if len(local_obs):
        # Scan-derived geometry and time-limited observation memory only.
        local_obs = local_obs[np.linalg.norm(local_obs[:,:2]-env.boat_pos,axis=1) < env.lidar_range+20]
    obs = local_obs/p.pixels_per_m
    turn_cost = np.zeros(len(speeds))
    horizon_steps = trajectory.shape[1]
    if compiled_constant_rollout is not None:
        if not hasattr(env,'_rollout_params'):
            env._rollout_params = parameter_vector(p)
        z,closest,blocked,turn_cost,trajectory = compiled_constant_rollout(
            state,speeds,rate_cmd,obs,env.map_w/p.pixels_per_m,
            env.sim_h/p.pixels_per_m,cfg.prediction_step_s,horizon_steps,
            cfg.safety_margin_m,env._rollout_params)
    else:
        for step in range(horizon_steps):
            left,right = allocate(z,speeds,rate_cmd,p)
            z = integrate(z,left,right,cfg.prediction_step_s,p)
            trajectory[:,step,:] = z
            if len(obs):
                dx = obs[None,:,0]-z[:,None,0]
                dy = obs[None,:,1]-z[:,None,1]
                c,s = np.cos(z[:,None,2]),np.sin(z[:,None,2])
                longitudinal = dx*c+dy*s
                lateral = -dx*s+dy*c
                # Union of two hull capsules, including the wider stern corners.
                longitudinal_gap = np.maximum.reduce([-0.56-longitudinal, longitudinal-0.52, np.zeros_like(longitudinal)])
                lateral_gap = np.minimum(np.abs(lateral-.22),np.abs(lateral+.22))
                separation = np.hypot(longitudinal_gap,lateral_gap)-obs[None,:,2]-.32
                margin = separation.min(axis=1)
                closest = np.minimum(closest, margin)
                blocked |= margin < cfg.safety_margin_m
            wall = np.minimum.reduce([z[:,1]-.93, env.sim_h/p.pixels_per_m-.93-z[:,1],
                                      z[:,0]-.93, env.map_w/p.pixels_per_m-.93-z[:,0]])
            closest = np.minimum(closest,wall)
            blocked |= wall < .05
            turn_cost += z[:,5]**2*cfg.prediction_step_s
    # Penalize reversals and target changes; speed is reduced when a turn would
    # violate the swept safety envelope. No instantaneous rotation escape.
    previous = getattr(env,'command_yaw_rate',0.)
    direction = (target-state[:2])/max(np.linalg.norm(target-state[:2]),.01)
    displacement = z[:,:2]-state[:2]
    progress = displacement @ direction
    cross_track = displacement[:,0]*direction[1]-displacement[:,1]*direction[0]
    cost = -progress + cfg.cross_track_weight*cross_track**2 + cfg.turning_weight*turn_cost + cfg.command_change_weight*(rate_cmd-previous)**2
    cost += cfg.terminal_heading_weight*np.abs(wrap(angle-z[:,2]))
    cost += cfg.goal_distance_weight*np.linalg.norm(z[:,:2]-env.target/p.pixels_per_m, axis=1)
    cost += .10/np.maximum(closest+.15,.03)
    cost += .30*(p.cruise_speed_m_s-speeds)
    cost += (speeds < 0)*.6
    cost += blocked*1000.
    recovering = path_unavailable or env.frame*env.dt < getattr(env,'recovery_until',0.)
    if recovering:
        cost = np.where(speeds < 0, -closest + .5*rate_cmd**2 + blocked*1000., np.inf)
    best = int(np.argmin(cost))
    if np.all(blocked):
        # The candidate set cannot promise a safe passage: brake and retain a
        # continuous yaw command rather than selecting a colliding fast turn.
        recovery = speeds <= 0
        best = int(np.argmin(np.where(recovery,-closest+.03*(rate_cmd-previous)**2,np.inf)))
    env.command_speed = float(speeds[best])
    env.command_yaw_rate = float(rate_cmd[best])
    env.predicted_trajectory = np.vstack([state[:2],trajectory[best,:,:2]])
    env.prediction_frame = env.frame
    env.prediction_stride_steps = round(cfg.prediction_step_s/env.dt)
    env.predicted_clearance = float(closest[best])
    env.emergency_mode = bool(np.all(blocked))
    env.min_wide_dist = float(np.min(env.lidar_dists))
    env.closest_avoid_hit = None
    steer = env.command_yaw_rate/p.max_yaw_rate_rad_s
    env.prev_steer = steer
    return steer
