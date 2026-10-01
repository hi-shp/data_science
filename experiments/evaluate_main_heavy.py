"""Fast, deterministic evaluation through MAIN's actual navigation loop.

Only rendering, screenshot output, and wall-clock pacing are replaced. Every
physics/control/planning step still comes from main.run().
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame


class Finished(Exception):
    pass


class _Renderer:
    def __init__(self, env):
        self.engine_3d = None

    def render(self, hits_x, hits_y):
        pass


def run(seeds, output, timeout_s, trace_seed=None):
    import environment
    import main

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if [row['seed'] for row in rows] != seeds[:len(rows)]:
        raise ValueError('Existing evaluation rows are not this seed prefix')
    if len(rows) == len(seeds):
        return rows

    original_renderer = environment.EnvRenderer
    original_factory = main.BoatEnv
    original_capture = main.save_episode_frame
    original_start_capture = main.start_capture_worker
    original_time = main.time
    original_guard = main.MainSafetyGuard
    original_find_gap = main.find_gap
    clock = {'wall': 0.0}
    state = {'seed': seeds[len(rows)], 'active': False, 'guard': None,
             'last_pos': None, 'last_heading': None, 'distance': 0.0,
             'turn_deg': 0.0, 'last_sign': 0, 'reversals': 0,
             'speed_sum': 0.0, 'steps': 0, 'guard_interventions': 0}
    trace = []

    class FakeClock:
        def tick(self, fps=120):
            clock['wall'] += .04 / main.BASE_PLAYBACK_RATE
            return 1000*.04 / main.BASE_PLAYBACK_RATE

    class FakeTime:
        @staticmethod
        def perf_counter():
            return clock['wall']

    def guard_factory():
        guard = original_guard()
        state['guard'] = guard
        return guard

    def find_gap(*args, **kwargs):
        result = original_find_gap(*args, **kwargs)
        if not kwargs.get('is_next_wp'):
            state['new_wp'] = None if result is None else [round(float(v), 1) for v in result['pos']]
        return result

    def factory():
        random.seed(state['seed'])
        np.random.seed(state['seed'])
        env = original_factory()
        env.clock = FakeClock()
        original_step, original_reset = env.step, env.reset

        def step(*args, **kwargs):
            if not state['active']:
                state['active'] = True
                state['last_pos'] = env.boat_pos.copy()
                state['last_heading'] = env.boat_heading
                state['guard_interventions'] = state['guard'].interventions
            left, right = float(args[0]), float(args[1])
            sign = 1 if right-left > 2 else -1 if left-right > 2 else 0
            if sign and state['last_sign'] and sign != state['last_sign']:
                state['reversals'] += 1
            if sign:
                state['last_sign'] = sign
            result = original_step(*args, **kwargs)
            if state['seed'] == trace_seed and state['steps'] % 5 == 0:
                trace.append({
                    't': round(env.frame*env.dt, 2),
                    'x': round(float(env.boat_pos[0]), 1),
                    'y': round(float(env.boat_pos[1]), 1),
                    'heading': round(float(env.boat_heading), 3),
                    'yaw': round(float(env.boat_ang_vel), 3),
                    'speed': round(float(np.linalg.norm(env.boat_vel))/env.dynamics.pixels_per_m, 3),
                    'target_heading': round(float(env.heading_target), 3),
                    'steer': round(float(env.prev_steer), 3),
                    'desired_speed': round(float(env.command_speed), 3),
                    'min_wide': round(float(env.min_wide_dist), 1),
                    'wp': None if env.current_wp is None else [round(float(v), 1) for v in env.current_wp['pos']],
                    'next_wp': None if env.next_wp is None else [round(float(v), 1) for v in env.next_wp['pos']],
                    'new_wp': state.get('new_wp'),
                    'pursuit': None if env.pursuit_target is None else [round(float(v), 1) for v in env.pursuit_target],
                    'guard': state['guard'].interventions,
                })
            state['distance'] += float(np.linalg.norm(env.boat_pos-state['last_pos']))/env.dynamics.pixels_per_m
            state['turn_deg'] += math.degrees(abs(env.boat_heading-state['last_heading']))
            state['last_pos'] = env.boat_pos.copy()
            state['last_heading'] = env.boat_heading
            state['speed_sum'] += float(np.linalg.norm(env.boat_vel))/env.dynamics.pixels_per_m
            state['steps'] += 1
            return result

        def reset():
            if state['active']:
                distance = float(np.linalg.norm(env.target-env.boat_pos))
                collision = bool(env.collide())
                outcome = ('timeout' if env.frame*env.dt >= timeout_s and
                           not collision and distance >= 70 else
                           'collision' if collision else
                           'success' if distance < 70 else None)
                if outcome is None:
                    raise RuntimeError('Unexpected episode reset')
                row = {
                    'seed': state['seed'],
                    'map_hash': hashlib.sha256(env.obstacles.tobytes()).hexdigest()[:16],
                    'outcome': outcome,
                    'completion_sim_s': round(env.frame*env.dt, 4),
                    'path_length_m': round(state['distance'], 4),
                    'average_speed_m_s': round(state['speed_sum']/state['steps'], 4),
                    'cumulative_turn_deg': round(state['turn_deg'], 2),
                    'steering_reversals': state['reversals'],
                    'safety_interventions': state['guard'].interventions-state['guard_interventions'],
                }
                with output.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
                rows.append(row)
                if state['seed'] == trace_seed:
                    output.with_suffix('.trace.json').write_text(json.dumps({
                        'steps': trace, 'obstacles': env.obstacles.tolist(),
                    }, indent=2))
                if len(rows) == len(seeds):
                    raise Finished()
                state['seed'] = seeds[len(rows)]
                random.seed(state['seed'])
                np.random.seed(state['seed'])
            result = original_reset()
            state.update(active=False, last_pos=None, last_heading=None,
                         distance=0.0, turn_deg=0.0, last_sign=0,
                         reversals=0, speed_sum=0.0, steps=0)
            return result

        def render(*args, **kwargs):
            if state['active'] and env.frame*env.dt >= timeout_s:
                if not env.collide() and np.linalg.norm(env.target-env.boat_pos) >= 70:
                    env.reset()

        env.step, env.reset, env.render = step, reset, render
        return env

    environment.EnvRenderer = _Renderer
    main.BoatEnv = factory
    main.MainSafetyGuard = guard_factory
    main.find_gap = find_gap
    main.save_episode_frame = lambda *a, **kw: None
    main.start_capture_worker = lambda: None
    main.time = FakeTime
    try:
        main.run()
    except Finished:
        pass
    finally:
        environment.EnvRenderer = original_renderer
        main.BoatEnv = original_factory
        main.MainSafetyGuard = original_guard
        main.find_gap = original_find_gap
        main.save_episode_frame = original_capture
        main.start_capture_worker = original_start_capture
        main.time = original_time
        pygame.quit()
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', required=True,
                        help='Comma-separated integers or inclusive ranges, e.g. 2000-2199')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-sim', type=float, default=140.0)
    parser.add_argument('--trace-seed', type=int)
    args = parser.parse_args()
    seeds = []
    for item in args.seeds.split(','):
        if '-' in item:
            start, end = map(int, item.split('-', 1))
            seeds.extend(range(start, end+1))
        else:
            seeds.append(int(item))
    result = run(seeds, args.output, args.timeout_sim, args.trace_seed)
    print(json.dumps({kind: sum(row['outcome']==kind for row in result)
                      for kind in ('success', 'collision', 'timeout')}))
