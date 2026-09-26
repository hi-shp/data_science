"""Isolated paired architecture trials. Never changes the default GUI pipeline.

python3 -m experiments.compare_navigation --mode corridor --seeds 2000 2081 \
    --output data/architecture_rethink/corridor
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('KABOAT_WIDTH', '1800')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import numpy as np
from environment import BoatEnv
from simulation import advance
from perception import lidar_hits_np, update_grid
from navigation_map import NavigationMap
from experiments.sampling_navigation import SamplingNavigator, SamplingConfig, hull_clearance
from goal_guidance import approach_heading, goal_reached

SOURCE_HASHES = {str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in [*Path('.').glob('*.py'), *Path('experiments').glob('*.py'),
                           Path('vessel_config.json'), Path('best_learned_params.json')]}


def experimental_step(env, navigator):
    env.frame += 1
    env.update_dynamic_obstacles()
    scale = env.dynamics.pixels_per_m
    dists, hx, hy = lidar_hits_np(env.boat_pos, env.boat_heading, env.rel_angles,
                                 env.dynamic_obstacles, env.lidar_range,
                                 (0, 0, env.map_w, env.sim_h))
    env.lidar_dists = dists
    update_grid(env.grid, hx, hy)
    env.grid *= .945
    if (env.frame-1) % env.control.planning_period_steps == 0:
        if not hasattr(env, 'navigation_map'):
            env.navigation_map = NavigationMap(env.map_w/scale, env.sim_h/scale)
        env.navigation_map.observe(env.boat_pos/scale, env.boat_heading, env.rel_angles,
                                   dists/scale, env.lidar_range/scale, env.frame*env.dt)
        env.perceived_obstacles = env.navigation_map.obstacles*scale
        command, prediction, target, clearance = navigator.plan(
            env.physics_state(), env.navigation_map, env.target/scale, env.frame)
        env.command_speed, env.command_yaw_rate = map(float, command)
        env.raw_route = navigator.path
        env.control_path = np.vstack([env.physics_state()[:2], prediction[:, :2]])
        env.predicted_trajectory = env.control_path
        env.prediction_frame = env.frame
        env.prediction_stride_steps = env.control.planning_period_steps
        env.controller_target = prediction[min(8,len(prediction)-1), :2]*scale
        env.pursuit_target = target*scale
        env.heading_target = float(np.arctan2(target[1]-env.boat_pos[1]/scale, target[0]-env.boat_pos[0]/scale))
        env.predicted_clearance = clearance
        env.min_wide_dist = float(dists.min())
        env.selected_gap = None
    env.prev_steer = env.command_yaw_rate/env.dynamics.max_yaw_rate_rad_s
    env.step(*env.get_pwm(env.prev_steer))
    env.update_camera()
    return hx, hy


def run(seed, mode, cfg, output, timeout=140., scenario=None, goal_quality=False):
    env = BoatEnv(headless=True)
    random.seed(seed); np.random.seed(seed); env.reset()
    if scenario == 'culdesac':
        # Explicit general topology stress case, not a navigation seed exception.
        # U opens toward the start; only LiDAR can reveal its sides/end wall.
        points = [[x,y,.35] for x in np.arange(4.,10.,.6) for y in [3.,9.]]
        points += [[10.,y,.35] for y in np.arange(3.,9.1,.6)]
        env.obstacles = np.asarray(points)*env.dynamics.pixels_per_m
        env.dynamic_obstacles = env.obstacles.copy()
    frozen = json.loads(Path('data/architecture_rethink/frozen_dynamics.json').read_text())
    assert asdict(env.dynamics) == frozen['physics'] and env.dt == frozen['dt']
    navigator = None if mode == 'baseline' else SamplingNavigator(env.dynamics, env.dt, mode, cfg)
    initial_map = env.obstacles.copy()
    previous = env.physics_state()
    last_sign = 0
    reversals = 0
    distance = total_turn = total_variation = peak_accel = 0.
    min_clear = min_perceived = float('inf')
    trace = []
    tick_times = []
    previous_command = np.zeros(2)
    reference_goal_heading = None
    first_goal_entry = None
    center_miss = float('inf')
    start = time.perf_counter()
    for k in range(round(timeout/env.dt)):
        tick = time.perf_counter()
        if navigator is None:
            advance(env)
        else:
            experimental_step(env, navigator)
        if k % 3 == 0:
            tick_times.append(time.perf_counter()-tick)
        state = env.physics_state()
        if mode == 'baseline':
            reference_goal_heading = approach_heading(getattr(env, 'control_path', None),
                                                      state[:2], env.target/env.dynamics.pixels_per_m,
                                                      reference_goal_heading)
            goal_heading = reference_goal_heading
        else:
            goal_heading = navigator.goal_heading
        command = np.array([env.command_speed, env.command_yaw_rate])
        distance += np.linalg.norm(state[:2]-previous[:2])
        total_turn += abs(state[5])*env.dt
        total_variation += abs(command[1]-previous_command[1])/.5
        peak_accel = max(peak_accel, abs(state[5]-previous[5])/env.dt)
        sign = int(np.sign(state[5])) if abs(state[5]) > np.deg2rad(3) else 0
        if sign and last_sign and sign != last_sign:
            reversals += 1
        if sign:
            last_sign = sign
        scale = env.dynamics.pixels_per_m
        # Ground truth is used here only, strictly after navigation for scoring.
        gt = float(hull_clearance(state, env.dynamic_obstacles/scale, env.map_w/scale, env.sim_h/scale)[0])
        pc = float(hull_clearance(state, env.perceived_obstacles/scale, env.map_w/scale, env.sim_h/scale)[0])
        min_clear, min_perceived = min(min_clear, gt), min(min_perceived, pc)
        boundary = bool(env.boat_pos[1] < 27 or env.boat_pos[1] > env.sim_h-27 or env.boat_pos[0] < 42 or env.boat_pos[0] > env.map_w-42)
        collision = bool(env.collide()) and not boundary
        goal_center = env.target/scale
        distance_to_goal = float(np.linalg.norm(state[:2]-goal_center))
        center_miss = min(center_miss, distance_to_goal)
        if first_goal_entry is None and distance_to_goal < 1.4:
            heading_error = (float(state[2]-goal_heading+np.pi)%(2*np.pi)-np.pi
                             if goal_heading is not None else float('nan'))
            first_goal_entry = dict(time_s=env.frame*env.dt, distance_m=distance_to_goal,
                                    heading_error_deg=float(np.rad2deg(heading_error)),
                                    yaw_rate_deg_s=float(np.rad2deg(state[5])))
        arrived = (goal_reached(state, goal_center, goal_heading) if goal_quality
                   else distance_to_goal < 1.4)
        success = bool(arrived and not collision and not boundary)
        target = np.asarray(getattr(env, 'controller_target', env.target))/scale
        trace.append([env.frame*env.dt, *state.tolist(), *command.tolist(), pc, gt, *target.tolist(), goal_heading])
        previous, previous_command = state, command
        if collision or boundary or success:
            break
    elapsed = time.perf_counter()-start
    def stats(times):
        return {n: float(v)*1000 for n, v in zip(['mean_ms','p95_ms','p99_ms','max_ms'],
                    [np.mean(times), *np.percentile(times, [95,99]), max(times)])} if times else {}
    result = dict(seed=seed, mode=mode, scenario=scenario, goal_quality=goal_quality,
                  first_goal_entry=first_goal_entry, center_miss_distance_m=center_miss,
                  terminal_heading_rad=goal_heading,
                  outcome='success' if success else 'collision' if collision else 'boundary' if boundary else 'timeout',
                  time_s=env.frame*env.dt, path_length_m=float(distance), yaw_reversals=reversals,
                  peak_yaw_accel_deg_s2=float(np.rad2deg(peak_accel)), cumulative_turn_deg=float(np.rad2deg(total_turn)),
                  steering_tv_per_s=float(total_variation/(env.frame*env.dt)), min_hull_clearance_m=min_clear,
                  min_perceived_clearance_m=min_perceived, compute_s=elapsed, physics_steps_per_wall_s=env.frame/elapsed,
                  planning_tick=stats(tick_times), map_hash=hashlib.sha256(initial_map.tobytes()).hexdigest(),
                  source_sha256=SOURCE_HASHES, config=asdict(cfg))
    if navigator:
        result['stages'] = {name:stats(times) for name,times in navigator.timings.items()}
    output.mkdir(parents=True,exist_ok=True)
    with (output/f'{seed}_trace.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['time_s','x_m','y_m','heading_rad','u_m_s','v_m_s','yaw_rate_rad_s','left_thrust_N','right_thrust_N','command_speed','command_yaw_rate','perceived_clearance','gt_clearance','target_x','target_y','terminal_heading_rad']);writer.writerows(trace)
    (output/f'{seed}_map.json').write_text(json.dumps(initial_map.tolist()))
    (output/f'{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
    env.close()
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline','corridor','direct'],required=True)
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--timeout',type=float,default=140.)
    parser.add_argument('--scenario',choices=['culdesac'])
    parser.add_argument('--goal-quality',action='store_true')
    args=parser.parse_args()
    cfg=SamplingConfig(**json.loads(args.config.read_text())) if args.config else SamplingConfig()
    if args.output.exists():
        parser.error('Choose a new output directory to preserve earlier trials')
    for seed in args.seeds:
        run(seed,args.mode,cfg,args.output,args.timeout,args.scenario,args.goal_quality)


if __name__=='__main__':
    main()
