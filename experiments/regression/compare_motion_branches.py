"""Isolated, stepwise MAIN V2 / remote CODEX comparison (no branch switching).

Export the reference with git archive, then run this script with --reference.
Numerical runs use SDL dummy; this is not a GUI performance benchmark.
All observations, candidates, rollouts, ETA terms and selections are hashed
before display annotation. Detailed arrays can be saved at a divergent frame.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np


def digest(value):
    if isinstance(value, dict):
        return {k: digest(v) for k, v in sorted(value.items())}
    a = np.asarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def record(root, branch, seed, output, steps=2000, rate=1, detail_frame=-1):
    os.environ.update(SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
                      PYGAME_HIDE_SUPPORT_PROMPT='1')
    sys.path.insert(0, str(root))
    os.chdir(root)
    from unittest.mock import patch
    from environment import BoatEnv
    if branch == 'main':
        from heavy.motion_v2 import HeavyMotionV2
        from heavy.motion_core.experiments.sampling_navigation import SamplingNavigator
        from heavy.motion_core.trajectory_runtime import advance_trajectory
    else:
        from experiments.sampling_navigation import SamplingNavigator
        from trajectory_runtime import advance_trajectory
    class Renderer:
        def __init__(self, env): self.engine_3d = None
        def render(self, *args): pass
    random.seed(seed); np.random.seed(seed)
    with patch('environment.EnvRenderer', Renderer):
        env = BoatEnv()
    visual = HeavyMotionV2(env) if branch == 'main' else None
    env.navigation_mode = 'eta_continuity_forward'
    plan_orig, rollout_orig = SamplingNavigator.plan, SamplingNavigator.rollout
    plans, rows, endpoints = [], [], []
    active = {}
    # Forward-policy probes also call the compiled kernel directly, outside
    # SamplingNavigator.rollout. Capture those candidate arrays as well.
    import importlib
    prefix = 'heavy.motion_core.' if branch == 'main' else ''
    kernel_module = importlib.import_module(prefix+'experiments.fast_rollout')
    sampling_module = importlib.import_module(prefix+'experiments.sampling_navigation')
    compiled_orig = kernel_module.compiled_rollout
    if compiled_orig is not None:
        def compiled(*args, **kwargs):
            z, clearance = compiled_orig(*args, **kwargs)
            active.setdefault('compiled_calls', []).append(dict(
                initial=digest(args[0]), sequences=digest(args[1]),
                states=digest(z), clearance=digest(clearance)))
            return z, clearance
        kernel_module.compiled_rollout = sampling_module.compiled_rollout = compiled
    def rollout(nav, state, seq, obs, w, h, goal=None):
        z, clearance = rollout_orig(nav, state, seq, obs, w, h, goal)
        active.setdefault('rollouts', []).append(dict(sequences=digest(seq),
            states=digest(z), clearance=digest(clearance), obstacles=digest(obs)))
        if env.frame == detail_frame:
            n = len(active['rollouts'])
            np.savez_compressed(output.parent/f'{branch}_{seed}_frame{env.frame}_rollout{n}.npz',
                                sequences=seq, states=z, clearance=clearance, obstacles=obs)
        return z, clearance
    def plan(nav, state, obs, goal, frame):
        nav.record_diagnostics = True
        active.clear()
        active.update(frame=frame, state=digest(state), observation=digest(obs.obstacles),
                      rng=digest(json.dumps(nav.rng.bit_generator.state, sort_keys=True)))
        began = time.perf_counter()
        result = plan_orig(nav, state, obs, goal, frame)
        active.update(cost_values={k:np.asarray(v).tolist() for k,v in nav.last_diagnostics.items()
            if isinstance(v,np.ndarray) and v.dtype.kind in 'fbiu'},
            command=digest(result[0]), prediction=digest(result[1]),
            selected=nav.last_diagnostics['selected'], diagnostics=digest(nav.last_diagnostics),
            route=digest(nav.path), clearance=result[3], plan_ms=1000*(time.perf_counter()-began))
        if frame == detail_frame:
            np.savez_compressed(output.parent/f'{branch}_{seed}_frame{frame}_diagnostics.npz', **nav.last_diagnostics)
        plans.append(active.copy())
        return result
    SamplingNavigator.plan, SamplingNavigator.rollout = plan, rollout
    initial = dict(state=env.physics_state().tolist(), map_hash=digest(env.obstacles),
        target=env.target.tolist(), dynamics=vars(env.dynamics), control=vars(env.control),
        dt=env.dt, mode=env.navigation_mode)
    old_step = env.step
    pwm = [0.,0.]
    def step(l, r, **kw):
        pwm[:] = l, r
        return old_step(l, r, **kw)
    env.step = step
    for _ in range(steps):
        before = env.physics_state().copy()
        hits = visual.advance(env) if visual else advance_trajectory(env)
        collision = bool(env.collide())
        distance = float(np.linalg.norm(env.boat_pos-env.target))
        # Keep the production difference visible, not hidden by common criteria.
        success = distance/env.dynamics.pixels_per_m < 1.4
        rows.append([env.frame, env.frame*env.dt, *env.physics_state(),
                     env.command_speed, env.command_yaw_rate, *pwm, distance,
                     env.prediction_frame, plans[-1]['selected'], int(collision), int(success)])
        if visual:
            visible = visual.reference
            endpoints.append(dict(frame=env.frame, goal_distance_px=distance,
                boat=env.boat_pos.tolist(), goal=env.target.tolist(),
                raw=(env.raw_route[-1]*env.dynamics.pixels_per_m).tolist(),
                physical=(env.control_path[-1]*env.dynamics.pixels_per_m).tolist(),
                future=visual.last_result['prediction'][-1].tolist(),
                visible=visible[-1].tolist(),
                first=None if env.current_wp is None else env.current_wp['pos'].tolist(),
                second=None if env.next_wp is None else env.next_wp['pos'].tolist()))
            # Render cadence changes only the presentation state.
            for _ in range(max(1, round(2/rate))): visual.render(env,hits)
        if collision or success: break
    np.save(output.with_suffix('.npy'), rows)
    output.write_text(json.dumps(dict(seed=seed, branch=branch, initial=initial, plans=plans,
        steps=len(rows), outcome='collision' if collision else 'success' if success else 'timeout',
        endpoints=endpoints), indent=2))
    import pygame
    pygame.quit()


def compare(a, b):
    x, y = np.load(a.with_suffix('.npy')), np.load(b.with_suffix('.npy'))
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    n = min(len(x), len(y))
    bad = np.flatnonzero(np.any(x[:n] != y[:n], axis=1))
    plan_diff = []; max_cost_delta=0.; numeric_equal=True; candidate_equal=True
    for p, q in zip(left['plans'], right['plans']):
        candidate_equal &= (p['rollouts']==q['rollouts'] and
                            p.get('compiled_calls')==q.get('compiled_calls'))
        keys = [k for k in p if k not in ('plan_ms','diagnostics','cost_values') and p[k] != q[k]]
        for key, values in p.get('cost_values',{}).items():
            v,w=np.asarray(values),np.asarray(q['cost_values'][key])
            numeric_equal &= np.allclose(v,w,rtol=1e-12,atol=1e-12,equal_nan=True)
            if v.size and v.dtype.kind=='f':
                finite=np.isfinite(v)&np.isfinite(w)
                if finite.any():max_cost_delta=max(max_cost_delta,float(np.max(np.abs(v[finite]-w[finite]))))
        if not numeric_equal:keys.append('cost_values')
        if keys: plan_diff.append(dict(frame=p['frame'], keys=keys))
    return dict(initial_equal=left['initial']==right['initial'], steps=[len(x),len(y)],
        outcome=[left['outcome'],right['outcome']], exact_state_command=np.array_equal(x,y),
        max_abs_difference=np.max(np.abs(x[:n]-y[:n]),axis=0).tolist(),
        first_divergent_frame=None if not len(bad) else int(x[bad[0],0]),
        first_plan_difference=None if not plan_diff else plan_diff[0],
        differing_plans=len(plan_diff), candidate_rollouts_exact=candidate_equal,
        candidate_costs_tolerance_equal=bool(numeric_equal),max_cost_delta=max_cost_delta,
        plan_ms_median=[float(np.median([p['plan_ms'] for p in data['plans']])) for data in (left,right)])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.cwd())
    p.add_argument('--reference', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[2000,2069,2081])
    p.add_argument('--record', choices=['main','codex'])
    p.add_argument('--rate', type=int, default=1)
    p.add_argument('--detail-frame', type=int, default=-1)
    a=p.parse_args(); a.output=a.output.resolve(); a.output.mkdir(parents=True, exist_ok=True)
    if a.record:
        record(a.root.resolve(),a.record,a.seeds[0],a.output/f'{a.record}_{a.seeds[0]}.json',rate=a.rate,detail_frame=a.detail_frame)
    else:
        if a.reference is None:p.error('--reference is required for comparison')
        results={}
        for seed in a.seeds:
            for label,root in [('main',a.root),('codex',a.reference)]:
                subprocess.run([sys.executable,__file__,'--record',label,'--root',str(root.resolve()),
                    '--output',str(a.output),'--seeds',str(seed),'--rate',str(a.rate),
                    '--detail-frame',str(a.detail_frame)],check=True)
            results[seed]=compare(a.output/f'main_{seed}.json',a.output/f'codex_{seed}.json')
        (a.output/'comparison.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2))
        if not all(r['initial_equal'] and r['exact_state_command'] and
                   r['candidate_rollouts_exact'] and r['candidate_costs_tolerance_equal']
                   and r['first_plan_difference'] is None for r in results.values()):
            raise SystemExit('Motion parity gate failed; inspect comparison.json')
