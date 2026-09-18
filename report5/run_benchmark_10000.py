#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Report 5 Benchmark Runner
================================
Runs 5,000 paired simulation episodes each for Line Tracing and GAP Navigation (total 10,000 runs)
using multiprocessing across CPU cores with the current tuned physics & algorithm parameters.
"""

import os
import sys
import time
import math
import random
import pickle
import json
import numpy as np
from multiprocessing import Pool, cpu_count

# Ensure repo root is on sys.path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.environ["SDL_VIDEODRIVER"] = "dummy"

from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, line_trace_steering, is_direct_target_safe
from utils import make_bezier_path, pure_pursuit, wrap

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_single_episode(args):
    seed, mode = args
    # Set seeds for exact paired obstacle generation
    random.seed(seed)
    np.random.seed(seed)
    
    env = BoatEnv(render_enabled=False)
    if mode == 'linetrace':
        env.linetrace_mode = True
    else:
        env.linetrace_mode = False
    
    env.reset()
    
    # Store initial target distance for detour calculation
    start_pos = env.boat_pos.copy()
    target_pos = env.target.copy()
    direct_dist = float(math.hypot(target_pos[0] - start_pos[0], target_pos[1] - start_pos[1]))
    
    steps = 0
    max_steps = 3500  # 140.0 seconds timeout limit
    cum_turn = 0.0
    steers = []
    speeds = []
    clearances = []
    
    # Trajectory subsampling (keep every 8 steps, and keep full trajectory for first 300 seeds of each mode)
    keep_traj = (seed < 1300)
    traj = []
    
    success = False
    collision = False
    collision_type = "none"
    timeout = False
    
    prev_heading = env.boat_heading
    path_len = 0.0
    prev_pos = env.boat_pos.copy()
    
    while steps < max_steps:
        steps += 1
        env.frame += 1
        env.update_dynamic_obstacles()
        
        # 1. LiDAR scan
        dists, hits = lidar_hits_np(
            env.boat_pos, env.boat_heading,
            env.rel_angles, env.dynamic_obstacles,
            env.lidar_range
        )
        
        # 2. Track clearance to buoys and walls
        d_obs = np.hypot(env.dynamic_obstacles[:, 0] - env.boat_pos[0], env.dynamic_obstacles[:, 1] - env.boat_pos[1])
        c_obs = float(np.min(d_obs - env.dynamic_obstacles[:, 2] - env.boat_radius))
        c_wall = float(min(env.boat_pos[1], env.sim_h - env.boat_pos[1]) - env.boat_radius)
        c_min = min(c_obs, c_wall)
        clearances.append(c_min)
        
        # 3. Steering and guidance
        steer = 0.0
        if mode == 'linetrace':
            steer, h_target, min_front, c_hit = line_trace_steering(
                env.boat_pos, env.boat_heading, env.target,
                dists, env.rel_angles,
                env.boat_ang_vel, env.prev_steer
            )
            env.heading_target = h_target
            env.min_wide_dist = min_front
            env.closest_avoid_hit = c_hit
            env.prev_steer = steer
        else:
            # GAP Navigation
            should_plan = (steps % 2 == 0)
            if should_plan:
                update_grid(env.grid, hits)
                env.grid *= 0.945
                new_c = extract_clusters_from_grid(env.grid)
                env.clusters, env.cluster_ids = match_clusters(env.clusters, env.cluster_ids, new_c)
            
            dist_to_tgt = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
            boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
            clear_to_target = (dist_to_tgt <= 400.0) and is_direct_target_safe(
                env.boat_pos, env.boat_heading, env.target, env.dynamic_obstacles,
                env.boat_radius, boat_spd, params=env.params
            )
            
            if clear_to_target:
                env.current_wp = None
                env.next_wp = None
            elif should_plan:
                new_wp = find_gap(
                    env.clusters, env.cluster_ids, env.boat_pos, env.boat_heading,
                    env.target, env.visited, env.grid, env.dynamic_obstacles,
                    params=env.params
                )
                if new_wp is not None and env.current_wp is None:
                    env.current_wp = new_wp
                    
                if env.current_wp is not None:
                    # Check WP clear condition
                    mid = env.current_wp["pos"]
                    d_wp = math.hypot(mid[0] - env.boat_pos[0], mid[1] - env.boat_pos[1])
                    if d_wp < 60.0:
                        p = env.current_wp["pair"]
                        env.visited.add(p); env.visited.add((p[1], p[0]))
                        env.current_wp = env.next_wp
                        env.next_wp = None
                    else:
                        new_next_wp = find_gap(
                            env.clusters, env.cluster_ids, env.current_wp["pos"], env.boat_heading,
                            env.target, env.visited, env.grid, env.dynamic_obstacles,
                            params=env.params,
                            is_next_wp=True
                        )
                        env.next_wp = new_next_wp
                        
            # Bezier path planning
            env.path_timer += env.dt
            if env.path_timer >= 0.01:
                env.path_timer = 0
                if env.current_wp is None:
                    goal = env.target
                    env.bezier_path = make_bezier_path(
                        env.boat_pos, env.boat_heading, goal,
                        obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius,
                        min_clearance=8.0, boat_speed=boat_spd
                    )
                else:
                    goal = env.current_wp["pos"]
                    env.bezier_path = make_bezier_path(
                        env.boat_pos, env.boat_heading, goal,
                        obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius,
                        boat_speed=boat_spd
                    )
                    
                if env.bezier_path is not None:
                    env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=95)
                    
                if env.current_wp is not None and env.next_wp is not None:
                    if env.bezier_path is not None and len(env.bezier_path) >= 2:
                        t1 = env.bezier_path[-1] - env.bezier_path[-2]
                        next_head = math.atan2(t1[1], t1[0]) if np.linalg.norm(t1) > 1e-6 else math.atan2(env.current_wp["pos"][1] - env.boat_pos[1], env.current_wp["pos"][0] - env.boat_pos[0])
                    else:
                        next_head = math.atan2(env.current_wp["pos"][1] - env.boat_pos[1], env.current_wp["pos"][0] - env.boat_pos[0])
                        
                    env.next_bezier_path = make_bezier_path(
                        env.current_wp["pos"], next_head, env.next_wp["pos"],
                        obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius,
                        boat_speed=boat_spd, start_tangent_fixed=True
                    )
                    if env.next_bezier_path is not None:
                        env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=95)
                else:
                    env.next_bezier_path = None
                    env.next_pursuit_target = None
                    
            steer = env.update_steering(dists)
            if steer is None:
                steer = 0.0
                
        steers.append(float(steer))
        L, R = env.get_pwm(steer)
        env.step(L, R)
        
        # Physics step metrics tracking
        d_head = abs(wrap(env.boat_heading - prev_heading))
        cum_turn += math.degrees(d_head)
        prev_heading = env.boat_heading
        
        step_disp = math.hypot(env.boat_pos[0] - prev_pos[0], env.boat_pos[1] - prev_pos[1])
        path_len += step_disp
        speeds.append(step_disp / env.dt)
        prev_pos = env.boat_pos.copy()
        
        if keep_traj and (steps % 8 == 0):
            traj.append((round(float(env.boat_pos[0]), 1), round(float(env.boat_pos[1]), 1)))
            
        # Collision check
        if env.collide():
            collision = True
            bx, by = env.boat_pos
            if by <= 18.0 or by >= (env.sim_h - 18.0) or bx <= 18.0 or bx >= env.map_w:
                collision_type = "wall"
            else:
                collision_type = "obstacle"
            break
            
        # Goal check
        dist_to_goal = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
        if dist_to_goal < 70.0:
            success = True
            break
            
    if not success and not collision:
        timeout = True
        
    time_sec = round(steps * env.dt, 2)
    cum_turn_deg = round(cum_turn, 1)
    min_clear = round(float(np.min(clearances)), 2) if clearances else 0.0
    mean_spd = round(float(np.mean(speeds)), 2) if speeds else 0.0
    steer_arr = np.array(steers)
    jitter = round(float(np.mean(np.abs(np.diff(steer_arr)))), 4) if len(steer_arr) > 1 else 0.0
    detour = round(path_len / max(1.0, direct_dist), 3)
    
    return {
        "mode": mode,
        "seed": seed,
        "success": success,
        "collision": collision,
        "collision_type": collision_type,
        "timeout": timeout,
        "steps": steps,
        "time_sec": time_sec,
        "cum_turn_deg": cum_turn_deg,
        "min_clearance": min_clear,
        "mean_speed": mean_spd,
        "steer_jitter": jitter,
        "path_len": round(path_len, 1),
        "detour_ratio": detour,
        "final_pos": (round(float(env.boat_pos[0]), 1), round(float(env.boat_pos[1]), 1)),
        "trajectory": traj
    }

def main():
    total_seeds = 5000
    start_seed = 1000
    seeds = list(range(start_seed, start_seed + total_seeds))
    
    print(f"============================================================")
    print(f"  KABOAT REPORT 5 BENCHMARK RUNNER (10,000 TOTAL EPISODES)  ")
    print(f"============================================================")
    print(f"Seeds: {start_seed} ~ {start_seed + total_seeds - 1} ({total_seeds:,} paired runs)")
    
    tasks = []
    for s in seeds:
        tasks.append((s, 'linetrace'))
        tasks.append((s, 'gapnav'))
        
    num_workers = min(24, max(1, cpu_count() - 2))
    print(f"Executing on {num_workers} parallel CPU processes...")
    t0 = time.time()
    
    results = []
    completed = 0
    chunk_size = 10
    
    with Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(run_single_episode, tasks, chunksize=chunk_size):
            results.append(res)
            completed += 1
            if completed % 250 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                pct = completed / len(tasks) * 100.0
                rate = completed / max(0.1, elapsed)
                eta = (len(tasks) - completed) / max(0.01, rate)
                print(f"Progress: {completed:5d}/{len(tasks)} ({pct:5.1f}%) | Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} eps/s | ETA: {eta:.1f}s", flush=True)
                
    total_time = time.time() - t0
    print(f"\nAll 10,000 simulations finished in {total_time:.2f} seconds ({total_time/60.0:.2f} minutes)!")
    
    # Save raw results pkl
    pkl_file = os.path.join(OUTPUT_DIR, "benchmark_5000_results.pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved raw results: {pkl_file} ({os.path.getsize(pkl_file)/(1024*1024):.2f} MB)")
    
    # Aggregate statistics
    modes = ['linetrace', 'gapnav']
    summary = {}
    
    for m in modes:
        m_runs = [r for r in results if r['mode'] == m]
        n_total = len(m_runs)
        n_succ = sum(1 for r in m_runs if r['success'])
        n_coll = sum(1 for r in m_runs if r['collision'])
        n_wall = sum(1 for r in m_runs if r['collision_type'] == 'wall')
        n_obs = sum(1 for r in m_runs if r['collision_type'] == 'obstacle')
        n_time = sum(1 for r in m_runs if r['timeout'])
        
        succ_runs = [r for r in m_runs if r['success']]
        times = [r['time_sec'] for r in succ_runs] if succ_runs else [0.0]
        turns = [r['cum_turn_deg'] for r in succ_runs] if succ_runs else [0.0]
        spds = [r['mean_speed'] for r in succ_runs] if succ_runs else [0.0]
        jitters = [r['steer_jitter'] for r in succ_runs] if succ_runs else [0.0]
        clears = [r['min_clearance'] for r in succ_runs] if succ_runs else [0.0]
        detours = [r['detour_ratio'] for r in succ_runs] if succ_runs else [0.0]
        
        summary[m] = {
            "total_episodes": n_total,
            "success_count": n_succ,
            "collision_count": n_coll,
            "timeout_count": n_time,
            "wall_collision_count": n_wall,
            "obstacle_collision_count": n_obs,
            "success_rate_pct": round(n_succ / n_total * 100.0, 2),
            "collision_rate_pct": round(n_coll / n_total * 100.0, 2),
            "timeout_rate_pct": round(n_time / n_total * 100.0, 2),
            "wall_collision_rate_pct": round(n_wall / max(1, n_coll) * 100.0, 2),
            "time_sec_mean": round(float(np.mean(times)), 2),
            "time_sec_std": round(float(np.std(times)), 2),
            "time_sec_median": round(float(np.median(times)), 2),
            "time_sec_q25": round(float(np.percentile(times, 25)), 2),
            "time_sec_q75": round(float(np.percentile(times, 75)), 2),
            "cum_turn_deg_mean": round(float(np.mean(turns)), 2),
            "cum_turn_deg_std": round(float(np.std(turns)), 2),
            "cum_turn_deg_median": round(float(np.median(turns)), 2),
            "mean_speed_mean": round(float(np.mean(spds)), 2),
            "steer_jitter_mean": round(float(np.mean(jitters)), 4),
            "min_clearance_mean": round(float(np.mean(clears)), 2),
            "min_clearance_median": round(float(np.median(clears)), 2),
            "detour_ratio_mean": round(float(np.mean(detours)), 3),
            "detour_ratio_median": round(float(np.median(detours)), 3)
        }
        
    summary_file = os.path.join(OUTPUT_DIR, "benchmark_5000_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary JSON: {summary_file}")
    
    print("\n=== BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
