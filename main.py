import pygame
import numpy as np
import math
import datetime
import time
import os
import leaderboard
from environment import BoatEnv
from simulation import advance
from perception import lidar_hits_np

BASE_PLAYBACK_RATE = 2.0  # simulation seconds per wall second at displayed 1x
MAX_PHYSICS_STEPS_PER_RENDER = 8  # bound catch-up latency; never skip a physics step


def run():
    env = BoatEnv()
    # Do not turn environment/renderer initialization time into catch-up physics.
    env.clock.tick(120)
    accumulator = 0.0
    hits_x = hits_y = np.empty(0)

    while True:
        elapsed = env.clock.tick(120) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
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

        if getattr(env, 'paused', False):
            accumulator = 0.0
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

        # Displayed 1x/2x/4x means 2/4/8 simulation seconds per wall second.
        # Only the step budget changes; dt and simulation-time planning stay fixed.
        accumulator += min(elapsed, .25) * BASE_PLAYBACK_RATE * env.sim_speed
        sub_steps = min(MAX_PHYSICS_STEPS_PER_RENDER, int(accumulator / env.dt))
        accumulator -= sub_steps * env.dt
        
        for step_idx in range(sub_steps):
            hits_x, hits_y = advance(env, step_idx, sub_steps)

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
                if env.collide() or dist_tgt_end < 70:
                    is_success = (dist_tgt_end < 70 and not env.collide())
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
                        pygame.image.save(env.screen, p)
                    except:
                        pass
                    env.reset()
                    accumulator = 0.0
                    break

        if hits_x is not None:
            env.render(hits_x, hits_y)

if __name__ == "__main__":
    run()
