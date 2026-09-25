"""Seeded navigation experiments; the GUI and evaluator use the same step.

Use --legacy-root with an archived e9418fb checkout for the original GUI loop.
Outputs are simulation measurements, not measurements of a physical vessel.
"""
import argparse
import ast
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import textwrap
import time


def legacy_step(root):
    """Extract only the old GUI's physics loop, without reset/render/file I/O."""
    import main
    source = (root / 'main.py').read_text()
    tree = ast.parse(source)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'step_idx')
    end = next(n for n in loop.body if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'dist_tgt_end')
    body = textwrap.dedent('\n'.join(source.splitlines()[loop.body[0].lineno-1:end.lineno-1]))
    preamble = ('step_idx = env.frame % 4\nsub_steps = 4\nplan_interval = 4\n'
                'new_wp = getattr(env, "_legacy_new_wp", None) if step_idx else None\n')
    code = 'def advance(env):\n' + textwrap.indent(preamble + body + '\nenv._legacy_new_wp = new_wp\n', '    ')
    namespace = vars(main).copy()
    exec(compile(code, str(root / 'main.py'), 'exec'), namespace)
    return namespace['advance']


def episode(job):
    seed, root, legacy, timeout, width, overrides, trace_stride, render = job
    os.environ.update(KABOAT_WIDTH=str(width), PYGAME_HIDE_SUPPORT_PROMPT='1',
                      OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if not render:
        os.environ.update(SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy')
    os.chdir(root)
    sys.path.insert(0, root)
    import numpy as np
    import environment
    if legacy:
        environment.EnvRenderer = lambda env: None
        step = legacy_step(Path(root))
        env = environment.BoatEnv()
    else:
        from simulation import advance
        step = advance
        env = environment.BoatEnv(headless=not render)
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    env.params.update(overrides)
    if not legacy:
        env.configure_dynamics()
    env.show_all_gaps = False
    initial_obstacles = env.obstacles.tolist()
    map_hash = hashlib.sha256(env.obstacles.tobytes()).hexdigest()[:16]
    trace, steer_tv, turn, min_clearance = [], 0., 0., float('inf')
    peak_yaw, peak_accel, prev_yaw, prev_steer = 0., 0., 0., 0.
    reversals, last_turn_sign = 0, 0
    started = time.perf_counter()
    for k in range(round(timeout / env.dt)):
        hits = step(env)
        if render:
            import pygame
            if any(e.type == pygame.QUIT for e in pygame.event.get()):
                env.close()
                raise KeyboardInterrupt('Evaluation window closed')
            env.render(*hits)
        yaw = float(env.boat_ang_vel)
        steer = float(env.prev_steer)
        turn += abs(yaw) * env.dt
        steer_tv += abs(steer - prev_steer)
        peak_yaw = max(peak_yaw, abs(yaw))
        peak_accel = max(peak_accel, abs(yaw - prev_yaw) / env.dt)
        sign = 1 if yaw > np.deg2rad(3) else (-1 if yaw < -np.deg2rad(3) else 0)
        if sign and last_turn_sign and sign != last_turn_sign:
            reversals += 1
        if sign:
            last_turn_sign = sign
        prev_yaw, prev_steer = yaw, steer
        speed = float(np.linalg.norm(env.boat_vel)) / 50.
        if len(env.dynamic_obstacles):
            # Conservative circumscribed hull circle; not the collision predicate.
            clearance = float(np.min(np.linalg.norm(env.dynamic_obstacles[:, :2] - env.boat_pos, axis=1)
                                     - env.dynamic_obstacles[:, 2] - 45.3)) / 50.
            min_clearance = min(min_clearance, clearance)
        # Common boundary condition for both versions (old gap mode omitted it).
        boundary = bool(env.boat_pos[1] < 27 or env.boat_pos[1] > env.sim_h-27
                        or env.boat_pos[0] < 42 or env.boat_pos[0] > env.map_w-42)
        collision = bool(env.collide()) and not boundary
        success = bool(np.linalg.norm(env.target-env.boat_pos) < 70 and not collision and not boundary)
        if k % trace_stride == 0 or collision or boundary or success:
            trace.append([seed, round((k+1)*env.dt, 4), float(env.boat_pos[0])/50,
                          float(env.boat_pos[1])/50, float(env.boat_heading), yaw, steer,
                          speed, float(getattr(env, 'thrust_left', float('nan'))),
                          float(getattr(env, 'thrust_right', float('nan')))])
        if collision or boundary or success:
            break
    result = dict(seed=seed, map_hash=map_hash,
                  outcome='success' if success else ('collision' if collision else ('boundary' if boundary else 'timeout')),
                  time_s=(k+1)*env.dt, cumulative_turn_deg=float(np.rad2deg(turn)),
                  steering_tv_per_s=steer_tv/((k+1)*env.dt), yaw_reversals=reversals,
                  peak_yaw_deg_s=float(np.rad2deg(peak_yaw)), peak_yaw_accel_deg_s2=float(np.rad2deg(peak_accel)),
                  min_circle_clearance_m=min_clearance, compute_s=time.perf_counter()-started)
    if not legacy:
        env.close()
    return result, trace, initial_obstacles


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--timeout', type=float, default=140.)
    p.add_argument('--width', type=int, default=1800)
    p.add_argument('--legacy-root', type=Path)
    p.add_argument('--params', type=Path)
    p.add_argument('--render', action='store_true', help='Show the evaluator (one worker)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.episodes < 1 or args.timeout < .04 or args.width < 1200 or args.workers < 1:
        p.error('episodes/timeout must be positive; width must be at least 1200')
    if args.render and args.legacy_root:
        p.error('--render cannot be used with an archived legacy root')
    root = (args.legacy_root or Path(__file__).parent).resolve()
    output = args.output.resolve()
    overrides = json.loads(args.params.read_text()) if args.params else {}
    jobs = [(seed, str(root), bool(args.legacy_root), args.timeout, args.width, overrides, 5, args.render)
            for seed in range(args.seed, args.seed+args.episodes)]
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        p.error('Output already contains an experiment; choose a new directory')
    source_hashes = {file.name:hashlib.sha256(file.read_bytes()).hexdigest()
                     for file in sorted(root.glob('*.py'))}
    configuration = {file.name:json.loads(file.read_text()) for file in
                     [root/'best_learned_params.json',root/'vessel_config.json'] if file.exists()}
    rows, maps = [], {}
    with (output/'trajectories.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed','time_s','x_m','y_m','heading_rad','yaw_rate_rad_s','steering_command',
                         'speed_m_s','left_thrust_N','right_thrust_N'])
        with concurrent.futures.ProcessPoolExecutor(max_workers=1 if args.render else args.workers) as pool:
            for result, trace, obstacles in pool.map(episode, jobs):
                rows.append(result)
                writer.writerows(trace)
                maps[str(result['seed'])] = obstacles
                print(f"{len(rows)}/{args.episodes}: {result['seed']} {result['outcome']} {result['time_s']:.1f}s", flush=True)
    with (output/'episodes.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'episodes':len(rows), 'seed_start':args.seed, 'width':args.width,
               'timeout_s':args.timeout, 'legacy':bool(args.legacy_root), 'overrides':overrides}
    summary['source_sha256'] = source_hashes
    summary['configuration'] = configuration
    summary['units'] = {'position':'m (50 px/m)', 'heading':'rad', 'time':'simulation seconds',
                        'steering':'normalized command; not a measured rudder angle'}
    for outcome in ['success','collision','boundary','timeout']:
        summary[outcome] = sum(r['outcome']==outcome for r in rows)
    for field in ['time_s','cumulative_turn_deg','steering_tv_per_s','yaw_reversals','peak_yaw_deg_s','peak_yaw_accel_deg_s2','compute_s']:
        summary['mean_'+field] = sum(r[field] for r in rows)/len(rows)
    success_times = [r['time_s'] for r in rows if r['outcome']=='success']
    summary['mean_success_time_s'] = sum(success_times)/len(success_times) if success_times else None
    # Wilson interval: avoid presenting a finite all-success sample as a guarantee.
    n = len(rows)
    fraction = summary['success']/n
    center = (fraction+1.96**2/(2*n))/(1+1.96**2/n)
    half = 1.96*((fraction*(1-fraction)/n+1.96**2/(4*n*n))**.5)/(1+1.96**2/n)
    summary['success_rate_wilson95'] = [center-half,center+half]
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'maps.json').write_text(json.dumps(maps)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','configuration']}, indent=2))


if __name__ == '__main__':
    main()
