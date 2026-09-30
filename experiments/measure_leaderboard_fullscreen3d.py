"""Measure autonomous main.run() with the RC fullscreen-3D display path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


class Done(Exception):
    pass


def summarize(directory, rows, count):
    if len(rows) != count:
        raise RuntimeError(f'Only {len(rows)}/{count} episodes completed')
    wins = [r for r in rows if r['outcome'] == 'success']
    if not wins:
        raise RuntimeError('No successful episode')
    summary = {
        'outcomes': {kind: sum(r['outcome'] == kind for r in rows)
                     for kind in ('success', 'collision', 'timeout')},
        'average_time_s_success': sum(r['time_s'] for r in wins) / len(wins),
        'average_turn_deg_success': sum(r['cumulative_turn_deg'] for r in wins) / len(wins),
        'average_collisions_all': sum(r['outcome'] == 'collision' for r in rows) / len(rows),
        'best_successful_run': min(wins, key=lambda r: r['time_s']),
    }
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def run(args):
    root, directory = args.root.resolve(), args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    os.chdir(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import pygame
    import main
    from config import WIDTH, HEIGHT

    records_file = directory / 'episodes.jsonl'
    rows = [json.loads(line) for line in records_file.read_text().splitlines()] \
        if records_file.exists() else []
    seeds = list(range(args.seed, args.seed + args.episodes))
    if [row['seed'] for row in rows] != seeds[:len(rows)]:
        raise ValueError('Existing episodes are not the requested seed prefix')
    if len(rows) == args.episodes:
        summarize(directory, rows, args.episodes)
        return
    seed = seeds[len(rows)]
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    metadata = dict(schema_version=2, version=args.version,
                    source_commit=commit, seed_start=args.seed,
                    episodes=args.episodes, timeout_wall_s=args.timeout,
                    display_speed='1x',
                    window_width_px=WIDTH, window_height_px=HEIGHT,
                    measurement='visible_X11_fullscreen_3D_main_run_perf_counter',
                    render_mode='fullscreen_3d', window='real X11',
                    headless=False, manual_mode=False, autonomous_control=True,
                    playback_rate=main.BASE_PLAYBACK_RATE, dt=0.04,
                    entrypoint='main.run',
                    nav_mode='eta_continuity_forward' if args.version == 'codex'
                    else 'main_default')
    metadata_file = directory / 'metadata.json'
    if metadata_file.exists():
        previous = json.loads(metadata_file.read_text())
        prior_count = previous.pop('episodes', None)
        current = dict(metadata)
        current.pop('episodes')
        if previous != current or not (0 < prior_count <= args.episodes):
            raise ValueError('Existing episodes have different measurement conditions')
    metadata_file.write_text(json.dumps(metadata, indent=2) + '\n')

    original_factory = main.BoatEnv
    state = dict(env=None, seed=seed, start=None, flip=None, heading=None,
                 turn=0., timeout=False, screenshot_saved=False,
                 render_count=0)

    def observed_factory(*factory_args, **factory_kwargs):
        random.seed(state['seed'])
        np.random.seed(state['seed'])
        env = original_factory(*factory_args, **factory_kwargs)
        # RC entry changes this display flag, not the X11 window size. Keep
        # autonomous control active while using its full-size 3D render path.
        env.fullscreen_3d = True
        state['env'] = env
        if (pygame.display.get_driver() != 'x11' or getattr(env, 'headless', False) or
                env.renderer is None or env.renderer.engine_3d is None or
                not env.fullscreen_3d or env.manual_mode or
                pygame.display.get_surface() is not env.screen):
            raise RuntimeError('Visible X11 autonomous fullscreen-3D game window required')
        window_id = pygame.display.get_wm_info().get('window')
        if window_id is None:
            raise RuntimeError('No desktop window ID')
        window_info = subprocess.check_output(
            ['xwininfo', '-id', hex(window_id)], text=True)
        if 'Map State: IsViewable' not in window_info:
            raise RuntimeError('The game window is not visible on the X11 desktop')
        state['heading'] = float(env.boat_heading)
        original_step, original_render, original_reset = env.step, env.render, env.reset

        def step(*a, **kw):
            if env.sim_speed != 1 or env.manual_mode or env.paused:
                raise RuntimeError('Displayed 1x autonomous play changed')
            if state['start'] is None:
                state['start'] = time.perf_counter()
            result = original_step(*a, **kw)
            heading = float(env.boat_heading)
            state['turn'] += abs(heading - state['heading'])
            state['heading'] = heading
            return result

        def render(*a, **kw):
            if not env.fullscreen_3d or env.manual_mode:
                raise RuntimeError('Fullscreen-3D autonomous display mode changed')
            result = original_render(*a, **kw)  # ends in pygame.display.flip()
            state['render_count'] += 1
            if (args.probe_screenshot and state['render_count'] >= 30 and
                    not state['screenshot_saved']):
                pygame.image.save(env.screen, str(args.probe_screenshot))
                state['screenshot_saved'] = True
            state['flip'] = time.perf_counter()
            if (state['start'] is not None and not state['timeout'] and
                    state['flip'] - state['start'] >= args.timeout):
                distance = float(np.linalg.norm(env.target - env.boat_pos))
                reached = (distance < 70 if args.version == 'main' else
                           main.flythrough_goal_reached(
                               distance / env.dynamics.pixels_per_m))
                if not reached and not env.collide():
                    state['timeout'] = True
                    env.reset()
            return result

        def reset():
            if state['start'] is not None:
                distance = float(np.linalg.norm(env.target - env.boat_pos))
                reached = (distance < 70 if args.version == 'main' else
                           main.flythrough_goal_reached(
                               distance / env.dynamics.pixels_per_m))
                collided = bool(env.collide())
                if state['timeout']:
                    outcome = 'timeout'
                elif collided:
                    outcome = 'collision'
                elif reached:
                    outcome = 'success'
                else:
                    raise RuntimeError('Reset outside goal, collision, or timeout')
                if state['flip'] is None or state['flip'] < state['start']:
                    raise RuntimeError('Result was not displayed')
                row = dict(seed=state['seed'],
                           map_hash=hashlib.sha256(env.obstacles.tobytes()).hexdigest()[:16],
                           outcome=outcome,
                           time_s=round(state['flip'] - state['start'], 4),
                           cumulative_turn_deg=round(float(np.degrees(state['turn'])), 1))
                with records_file.open('a', encoding='utf-8') as output:
                    output.write(json.dumps(row) + '\n')
                rows.append(row)
                if len(rows) == args.episodes:
                    raise Done()
                state['seed'] = seeds[len(rows)]
                random.seed(state['seed'])
                np.random.seed(state['seed'])
            result = original_reset()
            state.update(start=None, flip=None, heading=float(env.boat_heading),
                         turn=0., timeout=False)
            return result

        env.step, env.render, env.reset = step, render, reset
        return env

    main.BoatEnv = observed_factory
    try:
        if args.version == 'codex':
            main.run(nav_mode='eta_continuity_forward', seed=seed)
        else:
            main.run()
    except Done:
        pass
    finally:
        env = state['env']
        if env is not None:
            if hasattr(env, 'close'):
                env.close()
            else:
                if env.renderer is not None and env.renderer.engine_3d is not None:
                    env.renderer.engine_3d.close()
                pygame.quit()
        main.BoatEnv = original_factory
    summarize(directory, rows, args.episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('main', 'codex'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=3000)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--timeout', type=float, default=140.)
    parser.add_argument('--probe-screenshot', type=Path)
    run(parser.parse_args())
