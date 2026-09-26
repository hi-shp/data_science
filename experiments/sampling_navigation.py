"""Experimental vessel-model sampling MPC, with optional coarse A* guidance.

This is an MPPI-inspired bounded shooting optimizer, not a claim of exact
path-integral optimal control: deterministic exploration primitives augment
correlated Gaussian perturbations, and the weighted proposal is safety checked.
Inputs deliberately exclude the simulation environment and ground-truth objects.
Physical integration and thrust allocation are imported unchanged.
"""
from dataclasses import dataclass
import time
import numpy as np
from vessel_dynamics import allocate, integrate
from route_planner import a_star
from control_path import lookahead, path_geometry
from goal_guidance import approach_heading, terminal_cost
from experiments.fast_rollout import compiled_rollout, parameter_vector


@dataclass(frozen=True)
class SamplingConfig:
    samples: int = 128
    horizon_steps: int = 40
    temperature: float = .3
    noise_speed: float = .35
    noise_yaw: float = .18
    smooth_weight: float = 3.
    clearance_weight: float = .25
    margin: float = .20
    yaw_command_step: float | None = None  # rad/s command change per .12 s knot


def hull_clearance(states, obstacles, width, height):
    """Same conservative twin-capsule envelope as the existing controller."""
    z = np.atleast_2d(states)
    margin = np.minimum.reduce([z[:, 0]-.93, width-.93-z[:, 0],
                                z[:, 1]-.93, height-.93-z[:, 1]])
    if len(obstacles):
        dx = obstacles[None, :, 0]-z[:, None, 0]
        dy = obstacles[None, :, 1]-z[:, None, 1]
        c, s = np.cos(z[:, None, 2]), np.sin(z[:, None, 2])
        along, lateral = dx*c+dy*s, -dx*s+dy*c
        gap = np.maximum(np.maximum(-.56-along, along-.52), 0.)
        side = np.minimum(abs(lateral-.22), abs(lateral+.22))
        margin = np.minimum(margin, (np.hypot(gap, side)-obstacles[None, :, 2]-.32).min(axis=1))
    return margin


class SamplingNavigator:
    def __init__(self, physics, dt=.04, mode='corridor', config=None):
        if mode not in ('corridor', 'direct'):
            raise ValueError(mode)
        self.p, self.dt, self.mode = physics, dt, mode
        self.cfg = config or SamplingConfig()
        self.rng = np.random.default_rng(731)  # same stream for every map; not a map seed
        self.sequence = None
        self.path = None
        self.path_frame = -1000
        self.progress = 0.
        self.previous = np.zeros(2)
        self.goal_heading = None
        self.timings = {'route': [], 'rollout': [], 'plan': []}
        self.fast_params = parameter_vector(physics)

    def guidance(self, state, observation, goal, frame):
        if self.mode == 'direct':
            return np.array([state[:2], goal])
        m = observation
        unsafe = self.path is not None and np.any(m.clearance(self.path) < self.cfg.margin+.04)
        if self.path is None or unsafe or frame-self.path_frame >= 24:
            started = time.perf_counter()
            clearance = np.minimum.reduce([m.gx-.54, m.width-.54-m.gx,
                                            m.gy-.54, m.height-.54-m.gy])
            for x, y, r in m.obstacles:
                clearance = np.minimum(clearance, np.hypot(m.gx-x, m.gy-y)-r-.54)
            free = clearance >= self.cfg.margin+.04
            weights = 1.+np.where(m.known_free, 0., 1.5)+.15/np.maximum(clearance, .1)
            start = (int(np.argmin(abs(m.ys-state[1]))), int(np.argmin(abs(m.xs-state[0]))))
            end = (int(np.argmin(abs(m.ys-goal[1]))), int(np.argmin(abs(m.xs-goal[0]))))
            self.path = a_star(free, weights, m.xs, m.ys, start, end, m.resolution)
            if self.path is not None:
                self.path[0] = state[:2]
            self.path_frame, self.progress = frame, 0.
            self.timings['route'].append(time.perf_counter()-started)
        return self.path

    def rollout(self, initial, sequences, obstacles, width, height):
        """Batch candidates, fixed .04 s steps, allocator refreshed every step."""
        if compiled_rollout is not None:
            return compiled_rollout(initial, sequences, obstacles, width, height,
                                    self.dt, self.fast_params)
        return self.rollout_reference(initial, sequences, obstacles, width, height)

    def rollout_reference(self, initial, sequences, obstacles, width, height):
        """NumPy reference retained as an exact-physics equivalence oracle."""
        n, horizon, _ = sequences.shape
        z = np.repeat(initial[None, :], n, axis=0)
        states = np.empty((n, horizon, 8))
        closest = np.full(n, np.inf)
        # Check every physical integration step between command knots.
        for t in range(horizon):
            for _ in range(3):
                forces = allocate(z, sequences[:, t, 0], sequences[:, t, 1], self.p)
                z = integrate(z, *forces, self.dt, self.p)
                closest = np.minimum(closest, hull_clearance(z, obstacles, width, height))
            states[:, t] = z
        return states, closest

    def plan(self, state, observation, goal, frame):
        started = time.perf_counter()
        cfg, p = self.cfg, self.p
        path = self.guidance(state, observation, goal, frame)
        if path is None or len(path) < 2:
            # Topology unavailable: local goal still supplies direction, while
            # every rollout retains the same perceived hull collision check.
            path = np.array([state[:2], goal])
        self.goal_heading = approach_heading(path, state[:2], goal, self.goal_heading)
        geometry = path_geometry(path)
        target, self.progress = lookahead(path, state[:2], 3., self.progress, geometry)
        # Coarse route is guidance only. It is never presented as executed motion.
        delta = target-state[:2]
        error = (np.arctan2(delta[1], delta[0])-state[2]+np.pi)%(2*np.pi)-np.pi
        desired = np.clip(.65*error, -.5, .5)
        horizon = cfg.horizon_steps
        if self.sequence is None:
            self.sequence = np.tile([p.cruise_speed_m_s, desired], (horizon, 1))
        else:
            self.sequence = np.vstack([self.sequence[1:], self.sequence[-1:]])
        # Correlated perturbations avoid using hundreds of independent switches.
        knots = self.rng.normal(size=(cfg.samples, (horizon+4)//5+1, 2))
        noise = np.empty((cfg.samples, horizon, 2))
        for t in range(horizon):
            f = (t%5)/5.
            noise[:, t] = (1-f)*knots[:, t//5]+f*knots[:, t//5+1]
        sequences = self.sequence[None, :, :]+noise*np.array([cfg.noise_speed, cfg.noise_yaw])
        sequences[0] = self.sequence
        # Broad turn-then-straight primitives provide multi-modal exploration,
        # including braking/reverse, without map-specific recovery rules.
        index = 1
        for speed in [p.cruise_speed_m_s, .65*p.cruise_speed_m_s, .3*p.cruise_speed_m_s, 0., -.25*p.cruise_speed_m_s]:
            for rate in np.linspace(-.5, .5, 9):
                if index >= cfg.samples:
                    break
                sequences[index, :, 0] = speed
                sequences[index, :, 1] = rate
                if index % 2:
                    sequences[index, horizon//2:, 1] = 0.
                index += 1
        sequences[:, :, 0] = np.clip(sequences[:, :, 0], -.25*p.cruise_speed_m_s, p.cruise_speed_m_s)
        sequences[:, :, 1] = np.clip(sequences[:, :, 1], -p.max_yaw_rate_rad_s, p.max_yaw_rate_rad_s)
        if cfg.yaw_command_step is not None:
            prior = np.full(cfg.samples, self.previous[1])
            for t in range(horizon):
                sequences[:, t, 1] = np.clip(sequences[:, t, 1],
                                             prior-cfg.yaw_command_step,
                                             prior+cfg.yaw_command_step)
                prior = sequences[:, t, 1]
        # Same perception map, bounded local subset; no simulator object access.
        obs = observation.obstacles
        obs = obs[np.linalg.norm(obs[:, :2]-state[:2], axis=1) < 6.8]
        tick = time.perf_counter()
        states, closest = self.rollout(state, sequences, obs, observation.width, observation.height)
        self.timings['rollout'].append(time.perf_counter()-tick)
        # Arc-length route progress and distance, evaluated against route samples.
        arc = geometry[2]
        local = (arc >= max(0., self.progress-.5)) & (arc <= self.progress+9.)
        guide, guide_arc = path[local], arc[local]
        if len(guide) < 2:
            guide, guide_arc = path, arc
        if self.mode == 'direct':
            # Resample the straight goal reference to avoid endpoint-only costs.
            length = np.linalg.norm(goal-state[:2])
            guide_arc = np.linspace(0., length, max(2, int(length/.25)+1))
            guide = state[:2]+guide_arc[:, None]*(goal-state[:2])/max(length, .001)
        def objective(u, z, clearance):
            distance2 = np.sum((z[:, -1, None, :2]-guide[None, :, :])**2, axis=-1)
            nearest = distance2.argmin(axis=1)
            progress = guide_arc[nearest]-self.progress
            change = np.diff(np.concatenate([np.tile(self.previous, (len(u), 1, 1)), u], axis=1), axis=1)
            cost = -2.*progress + .8*distance2[np.arange(len(u)), nearest]
            cost += .2*np.linalg.norm(z[:, -1, :2]-goal, axis=1)
            cost += cfg.smooth_weight*np.sum(change[:, :, 1]**2, axis=1)
            cost += .2*np.sum(change[:, :, 0]**2, axis=1)
            cost += .05*np.mean(z[:, :, 5]**2, axis=1)
            cost += cfg.clearance_weight/np.maximum(clearance+.15, .03)
            cost += .05*np.mean(u[:, :, 0]**2, axis=1)
            cost += terminal_cost(z[:, -1], state[:2], goal, self.goal_heading)
            return cost
        costs = objective(sequences, states, closest)
        feasible = closest >= cfg.margin
        best = int(np.argmin(np.where(feasible, costs, np.inf))) if np.any(feasible) else int(np.argmax(closest))
        selected, prediction = sequences[best].copy(), states[best].copy()
        selected_clearance = closest[best]
        if np.any(feasible):
            weights = np.where(feasible, np.exp(np.clip(-(costs-costs[best])/cfg.temperature, -80., 0.)), 0.)
            proposal = np.sum(weights[:, None, None]*sequences, axis=0)/weights.sum()
            proposal_states, proposal_clearance = self.rollout(state, proposal[None, :], obs, observation.width, observation.height)
            proposal_cost = objective(proposal[None, :], proposal_states, proposal_clearance)[0]
            # Averaging safe left/right paths is not necessarily safe. Recheck it.
            if proposal_clearance[0] >= cfg.margin and proposal_cost <= costs[best]:
                selected, prediction = proposal, proposal_states[0]
                selected_clearance = proposal_clearance[0]
        self.sequence, self.previous = selected, selected[0].copy()
        self.timings['plan'].append(time.perf_counter()-started)
        return selected[0], prediction, target, float(selected_clearance)
