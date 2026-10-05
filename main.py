import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import numpy as np
import math
import datetime
import time
import argparse
import random
import leaderboard
from environment import BoatEnv
from simulation import advance
from perception import lidar_hits_np
from fast_astar import warmup_astar
from fast_corridor import compiled_within_corridor
from fast_constant_rollout import warmup_constant_rollout
from frame_capture import save_episode_frame, start_capture_worker
from fast_clearance import warmup_clearance
from fast_command_arrays import warmup_command_arrays
from trajectory_modes import NAV_MODES
from trajectory_objective import flythrough_goal_reached
from playback_scheduler import playback_budget

BASE_PLAYBACK_RATE = 2.4  # simulation seconds per wall second at displayed 1x
MAX_PHYSICS_STEPS_PER_RENDER = 8  # bound catch-up latency; never skip a physics step
DEFAULT_NAV_MODE = 'eta_continuity_forward'


def run(nav_mode=None, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env = BoatEnv()
    # The original controller remains available for matched comparisons.
    env.navigation_mode = nav_mode or os.environ.get('KABOAT_NAVIGATION', DEFAULT_NAV_MODE)
    if env.navigation_mode not in (*NAV_MODES, 'trajectory_control', 'legacy'):
        raise ValueError(f'Unknown navigation mode: {env.navigation_mode}')
    # Raw A* stays available through the dashboard debug toggle.
    warmup_astar()
    warmup_constant_rollout()
    warmup_clearance()
    warmup_command_arrays()
    if env.navigation_mode not in ('legacy_a', 'legacy'):
        from experiments.fast_rollout import compiled_rollout, parameter_vector
        if compiled_rollout is not None:
            if env.navigation_mode in ('eta_continuity_passage_exact',
                                       'eta_continuity_forward'):
                from passage_geometry import (physical_hull_polygons,
                                              prepare_hull_edges, fast_surface_clearances)
                hull = physical_hull_polygons(env.dynamics.pixels_per_m)
                edges, bound, box = prepare_hull_edges(hull)
                compiled_rollout(np.zeros(8), np.zeros((1,1,2)), np.empty((0,3)),
                                 env.map_w/env.dynamics.pixels_per_m,
                                 env.sim_h/env.dynamics.pixels_per_m,
                                 env.dt, parameter_vector(env.dynamics), 0., 0., 1.4,
                                 hull, None, edges, bound, box)
                fast_surface_clearances(np.zeros((1, 8)), np.empty((0, 3)),
                                        env.map_w/env.dynamics.pixels_per_m,
                                        env.sim_h/env.dynamics.pixels_per_m,
                                        edges, bound, box)
                if env.navigation_mode == 'eta_continuity_forward':
                    compiled_rollout(
                        np.zeros(8), np.zeros((1, 1, 2)), np.empty((0, 3)),
                        env.map_w/env.dynamics.pixels_per_m,
                        env.sim_h/env.dynamics.pixels_per_m,
                        env.dt, parameter_vector(env.dynamics), 0., 0., 0.,
                        hull, None, edges, bound, box, np.zeros((1, 8)), True)
            compiled_rollout(np.zeros(8), np.zeros((1,1,2)), np.empty((0,3)),
                             env.map_w/env.dynamics.pixels_per_m,
                             env.sim_h/env.dynamics.pixels_per_m,
                             env.dt, parameter_vector(env.dynamics), 0., 0., 1.4)
    start_capture_worker()
    worker = None
    if (not env.headless and env.navigation_mode.startswith('eta_') and
            os.environ.get('KABOAT_SYNC_TRAJECTORY', '') != '1'):
        from trajectory_worker import TrajectoryWorker
        worker = TrajectoryWorker(env)
        env._trajectory_worker = worker
        worker.prime()
    worker_active = worker is not None
    if compiled_within_corridor is not None:
        # Compile before the clock starts; warmup never enters planning state.
        compiled_within_corridor(np.zeros((1,2)),np.zeros((1,2)),
                                 np.zeros((1,2)),np.ones(1))
    # Do not turn environment/renderer initialization time into catch-up physics.
    env.clock.tick(120)
    last_tick_time = time.perf_counter()
    accumulator = 0.0
    scheduled_speed = env.sim_speed
    hits_x = hits_y = np.empty(0)

    while True:
        env.clock.tick(120)
        now = time.perf_counter()
        elapsed = now-last_tick_time
        last_tick_time = now
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                if worker is not None:
                    worker.close()
                if hasattr(env, 'renderer') and hasattr(env.renderer, 'engine_3d') and env.renderer.engine_3d:
                    env.renderer.engine_3d.close()
                pygame.quit()
                return
            elif e.type == pygame.KEYDOWN:
                # 랭킹 모달이 열려 있을 때 키보드 단축키
                if getattr(env, 'show_leaderboard', False):
                    if e.key in [pygame.K_SPACE, pygame.K_RETURN, pygame.K_r]:
                        if not getattr(env, 'leaderboard_view_only', False):
                            env.reset_manual_episode()
                            continue
                    elif e.key in [pygame.K_ESCAPE, pygame.K_m]:
                        if getattr(env, 'leaderboard_view_only', False):
                            env.show_leaderboard = False
                            env.leaderboard_view_only = False
                        else:
                            env.toggle_manual_mode()
                        continue

                if e.key == pygame.K_SPACE:
                    env.paused = not env.paused
                elif e.key == pygame.K_c:
                    env.cam_3d_mode = (getattr(env, 'cam_3d_mode', 1) + 1) % 3
                elif e.key == pygame.K_v:
                    env.fullscreen_3d = not getattr(env, 'fullscreen_3d', False)
                elif e.key == pygame.K_m:
                    env.toggle_manual_mode()
                elif e.key == pygame.K_b:
                    if getattr(env, 'manual_mode', False):
                        env.toggle_blind_mode()
                elif e.key == pygame.K_r:
                    if getattr(env, 'manual_mode', False):
                        env.reset_manual_episode()
                elif e.key == pygame.K_F11:
                    env.toggle_fullscreen()
                elif e.key == pygame.K_ESCAPE:
                    if worker is not None:
                        worker.close()
                    if hasattr(env, 'renderer') and hasattr(env.renderer, 'engine_3d') and env.renderer.engine_3d:
                        env.renderer.engine_3d.close()
                    pygame.quit()
                    return
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)
                    if getattr(env, 'needs_break', False):
                        env.needs_break = False
                        break

        use_worker = (worker is not None and not env.manual_mode and
                      not env.linetrace_mode)
        if use_worker and not worker_active:
            worker.reset(env)
        worker_active = use_worker
        if getattr(env, 'paused', False):
            accumulator = 0.0
            env.display_step_fraction = 0.0
            map_bounds = (0, 0, env.map_w, env.sim_h) if getattr(env, 'linetrace_mode', False) else None
            dists, hits_x, hits_y = lidar_hits_np(
                env.boat_pos, env.boat_heading,
                env.rel_angles, env.dynamic_obstacles,
                env.lidar_range,
                map_bounds=map_bounds
            )
            env.lidar_dists = dists
            env.render(hits_x, hits_y)
            continue

        # Displayed 1x/2x/4x means 2.4/4.8/9.6 simulated seconds per wall second.
        # Only the step budget changes; dt and simulation-time planning stay fixed.
        speed_changed = scheduled_speed != env.sim_speed
        accumulator = playback_budget(accumulator, elapsed, scheduled_speed,
                                      env.sim_speed, BASE_PLAYBACK_RATE)
        scheduled_speed = env.sim_speed
        if speed_changed:
            last_tick_time = time.perf_counter()
        sub_steps = min(MAX_PHYSICS_STEPS_PER_RENDER, int(accumulator / env.dt))
        if use_worker:
            # Retain all unexecuted wall-clock budget while showing frames during
            # planning. Completed packets are applied once, in physics order.
            sub_steps = min(sub_steps, worker.available())
        accumulator -= sub_steps * env.dt
        
        for step_idx in range(sub_steps):
            hits_x, hits_y = (worker.advance(env, step_idx, sub_steps) if use_worker
                              else advance(env, step_idx, sub_steps))

            dist_tgt_end = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
            if getattr(env, 'manual_mode', False):
                if dist_tgt_end < 70 and not getattr(env, 'show_leaderboard', False):
                    # RC 수동 조종 모드 목적지 도달: 랭킹 기록 저장 및 리더보드 모달 표출
                    elapsed_time = round(time.time() - getattr(env, 'manual_start_time', time.time()), 2)
                    record = leaderboard.add_record(
                        collisions=env.manual_collisions,
                        arrival_time=elapsed_time,
                        cum_turn=round(env.manual_cum_turn, 1)
                    )
                    env.last_manual_result = record
                    env.show_leaderboard = True
                    env.leaderboard_view_only = False
                    env.boat_vel = np.zeros(2)
                    env.boat_ang_vel = 0.0
                # 수동 조종 모드에서는 충돌 발생 시 에피소드를 종료/리스폰하지 않고 계속 주행함
            else:
                reached = flythrough_goal_reached(dist_tgt_end/env.dynamics.pixels_per_m)
                if env.collide() or reached:
                    is_success = (reached and not env.collide())
                    tag = "SUCCESS" if is_success else "FAIL"
                    subfolder = "success" if is_success else "fail"
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    outdir = os.path.join("screenshot", subfolder)
                    if not os.path.exists(outdir):
                        try:
                            os.makedirs(outdir, exist_ok=True)
                        except:
                            pass
                    p = os.path.join(outdir, f"{ts}_{tag}.png")
                    try:
                        if hits_x is not None:
                            env.render(hits_x, hits_y)
                        save_episode_frame(env.screen, p)
                    except:
                        pass
                    env.reset()
                    if worker is not None:
                        worker.reset(env)
                    # A goal can end this frame's step batch early. Return its
                    # unexecuted fixed-step budget to the accumulator so 4x
                    # playback never silently loses physics time at a reset.
                    accumulator += (sub_steps-step_idx-1)*env.dt
                    break

        # Display interpolation only; no controller or physics state reads it.
        env.display_step_fraction = min(1.0, max(0.0, accumulator / env.dt))
        if hits_x is not None:
            env.render(hits_x, hits_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nav-mode', choices=NAV_MODES, default=None)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    run(args.nav_mode, args.seed)
