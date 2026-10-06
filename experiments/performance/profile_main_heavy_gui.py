"""Measure the real X11 fullscreen-3D MAIN loop without changing its pacing."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pygame


def run(seed, seconds, output):
    if not os.environ.get('DISPLAY') or os.environ.get('SDL_VIDEODRIVER') == 'dummy':
        raise RuntimeError('A visible X11 display is required')
    import main

    original_factory = main.BoatEnv
    original_events = pygame.event.get
    original_budget = main.playback_budget
    original_capture = main.save_episode_frame
    original_start_capture = main.start_capture_worker
    original_guard = main.MainSafetyGuard
    timed_names = ('lidar_hits_np', 'update_grid', 'extract_clusters_from_grid',
                   'match_clusters', 'find_gap', 'make_bezier_path')
    original_functions = {name: getattr(main, name) for name in timed_names}
    durations = {}
    record = {'start': None, 'end': None, 'steps': 0, 'frames': [],
              'backlog': [], 'last_budget': 0., 'frame_steps': 0,
              'seed': seed, 'fullscreen_3d': False}

    def factory():
        random.seed(seed)
        np.random.seed(seed)
        env = original_factory()
        env.fullscreen_3d = True
        env.sim_speed = 4
        record['fullscreen_3d'] = True
        old_step, old_render = env.step, env.render

        def step(*args, **kwargs):
            began = time.perf_counter()
            result = old_step(*args, **kwargs)
            track('physics_step', time.perf_counter()-began)
            if record['start'] is not None:
                record['steps'] += 1
                record['frame_steps'] += 1
            return result

        def render(*args, **kwargs):
            began = time.perf_counter()
            result = old_render(*args, **kwargs)
            now = time.perf_counter()
            track('render', now-began)
            if record['start'] is None:
                record['start'] = now
            else:
                record['frames'].append(now)
                record['backlog'].append(max(0., record['last_budget'] -
                                              record['frame_steps'] * env.dt) / env.dt)
            record['frame_steps'] = 0
            record['end'] = now
            return result

        env.step, env.render = step, render
        return env

    def budget(*args):
        result = original_budget(*args)
        record['last_budget'] = result
        return result

    def track(name, duration):
        if record['start'] is not None:
            total, count, maximum = durations.get(name, (0., 0, 0.))
            durations[name] = (total+duration, count+1, max(maximum, duration))

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            began = time.perf_counter()
            result = fn(*args, **kwargs)
            track(name, time.perf_counter()-began)
            return result
        return wrapper

    def guard_factory():
        guard = original_guard()
        guard.command = timed('safety_guard', guard.command)
        return guard

    def events(*args, **kwargs):
        result = original_events(*args, **kwargs)
        if record['start'] is not None and time.perf_counter()-record['start'] >= seconds:
            result.append(pygame.event.Event(pygame.QUIT))
        return result

    main.BoatEnv = factory
    main.MainSafetyGuard = guard_factory
    for name, fn in original_functions.items():
        setattr(main, name, timed(name, fn))
    main.playback_budget = budget
    main.save_episode_frame = lambda *args, **kwargs: None
    main.start_capture_worker = lambda: None
    pygame.event.get = events
    try:
        main.run()
    finally:
        main.BoatEnv = original_factory
        main.MainSafetyGuard = original_guard
        for name, fn in original_functions.items():
            setattr(main, name, fn)
        main.playback_budget = original_budget
        main.save_episode_frame = original_capture
        main.start_capture_worker = original_start_capture
        pygame.event.get = original_events

    times = np.array(record['frames'])
    intervals = np.diff(times) if len(times) > 1 else np.array([])
    window = deque()
    rolling = []
    for t in times:
        window.append(t)
        while window and window[0] < t-1.:
            window.popleft()
        if t-times[0] >= 1.:
            rolling.append(len(window))
    wall = record['end']-record['start']
    result = {
        'seed': seed, 'display': os.environ['DISPLAY'],
        'fullscreen_3d': record['fullscreen_3d'],
        'seconds': wall, 'requested_steps_per_s': main.BASE_PLAYBACK_RATE*4/.04,
        'actual_steps_per_s': record['steps']/wall,
        'sim_seconds_per_wall_second': record['steps']*.04/wall,
        'average_fps': len(times)/wall,
        'minimum_rolling_fps': min(rolling) if rolling else None,
        'median_frame_ms': float(np.median(intervals)*1000),
        'p95_frame_ms': float(np.quantile(intervals, .95)*1000),
        'p99_frame_ms': float(np.quantile(intervals, .99)*1000),
        'max_frame_ms': float(max(intervals)*1000),
        'frames_ge_100_ms': int(np.sum(intervals >= .1)),
        'max_backlog_steps': max(record['backlog']),
        'final_backlog_steps': record['backlog'][-1],
        'stage_ms_per_call': {name: {'mean': total/count*1000,
                                    'max': maximum*1000, 'calls': count}
                              for name, (total, count, maximum) in durations.items()},
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.seed, args.seconds, args.output)
