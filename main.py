import pygame
import numpy as np
import math
import datetime
import time
import os
import leaderboard
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear, is_direct_target_safe, is_waypoint_switch_safe, is_front_blocked, line_trace_steering
from utils import wrap, make_bezier_path, pure_pursuit

def run():
    env = BoatEnv()

    while True:
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
            map_bounds = (0, 0, env.map_w, env.sim_h) if getattr(env, 'linetrace_mode', False) else None
            dists, hits_x, hits_y = lidar_hits_np(
                env.boat_pos, env.boat_heading,
                env.rel_angles, env.dynamic_obstacles,
                env.lidar_range,
                map_bounds=map_bounds
            )
            env.lidar_dists = dists
            env.render(hits_x, hits_y)
            env.clock.tick(120)
            continue

        # 실시간 배속 설정에 따른 서브스텝 반복 실행 (120 FPS 타겟: 4배속까지 물리 연산 100% 보존 및 적응형 인지/탐색 주기)
        sub_steps = max(1, int(getattr(env, 'sim_speed', 1)))
        plan_interval = sub_steps
        hits_x = None
        hits_y = None
        new_wp = None
        
        for step_idx in range(sub_steps):
            env.frame += 1
            env.update_dynamic_obstacles()

            map_bounds = (0, 0, env.map_w, env.sim_h) if getattr(env, 'linetrace_mode', False) else None
            dists, hits_x, hits_y = lidar_hits_np(
                env.boat_pos, env.boat_heading,
                env.rel_angles, env.dynamic_obstacles,
                env.lidar_range,
                map_bounds=map_bounds
            )
            env.lidar_dists = dists

            update_grid(env.grid, hits_x, hits_y)
            env.grid *= 0.945

            # 연산 부하 절감을 위한 적응형 인지/탐색 주기 (4배속 이하는 매 스텝 100% 실행)
            should_plan = (step_idx % plan_interval == 0 or step_idx == sub_steps - 1)
            if should_plan:
                new_c = extract_clusters_from_grid(env.grid)
                env.clusters, env.cluster_ids = match_clusters(
                    env.clusters, env.cluster_ids, new_c
                )

                # [Gaps 버튼 전용 데이터] 평소에는 O(1)로 총 개수만 산출하고, Gaps 버튼이 활성화된 경우에만 렌더링용 객체를 생성하여 지연 평가(Lazy Evaluation) 최적화
                if step_idx == sub_steps - 1:
                    bx, by = env.boat_pos
                    ch = math.cos(env.boat_heading)
                    sh = math.sin(env.boat_heading)
                    front_clusters = [c for c in env.clusters if (c[0] - bx) * ch + (c[1] - by) * sh >= 0]
                    n_fc = len(front_clusters)
                    env.total_gaps_count = n_fc * (n_fc - 1) // 2 if n_fc >= 2 else 0

                    if getattr(env, 'show_all_gaps', False) and n_fc >= 2:
                        gui_all_gaps = []
                        for i in range(n_fc):
                            c1 = front_clusters[i]
                            for j in range(i + 1, n_fc):
                                c2 = front_clusters[j]
                                mid_pt = (c1 + c2) / 2.0
                                gui_all_gaps.append({
                                    "pos": mid_pt.copy(),
                                    "c1": c1.copy(),
                                    "c2": c2.copy()
                                })
                        env.all_gaps = gui_all_gaps
                    else:
                        env.all_gaps = []

            if getattr(env, 'manual_mode', False):
                if getattr(env, 'show_leaderboard', False):
                    target_thr = 0.0
                    target_str = 0.0
                    env.manual_throttle = 0.0
                    env.manual_steer = 0.0
                    L = 1500
                    R = 1500
                    steer = 0.0
                else:
                    keys = pygame.key.get_pressed()
                    target_thr = 0.0
                    target_str = 0.0
                    if keys[pygame.K_w] or keys[pygame.K_UP]:
                        target_thr += 1.0
                    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                        target_thr -= 0.6  # 후진 및 급제동
                    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                        target_str -= 1.0  # 좌현(Port) 선회
                    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                        target_str += 1.0  # 우현(Starboard) 선회

                    # 실시간 조종 응답 필터링
                    env.manual_throttle = getattr(env, 'manual_throttle', 0.0) * 0.75 + target_thr * 0.25
                    env.manual_steer = getattr(env, 'manual_steer', 0.0) * 0.70 + target_str * 0.30

                    base_pwm = 1500
                    diff = env.manual_steer * 270.0
                    L = int(np.clip(base_pwm - diff, 1100, 1900))
                    R = int(np.clip(base_pwm + diff, 1100, 1900))
                    steer = env.manual_steer
                env.prev_steer = steer
                env.heading_target = env.boat_heading + steer * 0.45
                env.current_wp = None
                env.next_wp = None
                env.candidate_wps = []
                env.bezier_path = None
                env.next_bezier_path = None
                env.pursuit_target = None
                env.all_gaps = []
                env.total_gaps_count = 0
            elif getattr(env, 'linetrace_mode', False):
                steer, h_target, min_front, c_hit = line_trace_steering(
                    env.boat_pos, env.boat_heading, env.target,
                    dists, env.rel_angles,
                    env.boat_ang_vel, env.prev_steer
                )
                env.heading_target = h_target
                env.min_wide_dist = min_front
                env.closest_avoid_hit = c_hit
                env.prev_steer = steer
                env.current_wp = None
                env.next_wp = None
                env.candidate_wps = []
                env.bezier_path = None
                env.next_bezier_path = None
                env.pursuit_target = None
                env.next_pursuit_target = None
                env.all_gaps = []
                env.total_gaps_count = 0
            else:
                if should_plan:
                    dist_to_target = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
                    boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
                    clear_to_target = (dist_to_target <= 400.0) and is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, env.dynamic_obstacles, env.boat_radius, boat_spd, params=env.params)

                    # 목적지와 400픽셀 이하로 가까워졌고, 목적지 방향 직선 경로에 장애물이 없으면 즉시 목적지 직행
                    if clear_to_target:
                        new_wp = None
                        env.current_wp = None
                        env.next_wp = None
                        env.candidate_wps = []
                    else:
                        # 경로 상에 장애물이 있으면 장애물 사이 갭(웨이포인트)을 찾아 안전하게 우회
                        new_wp = find_gap(
                            env.clusters, env.cluster_ids,
                            env.boat_pos, env.boat_heading,
                            env.target, env.visited,
                            env.grid, env.dynamic_obstacles,
                            params=env.params
                        )
                        if new_wp is not None:
                            env.candidate_wps = new_wp.get("candidates", [])
                        elif env.current_wp is not None:
                            env.candidate_wps = env.current_wp.get("candidates", [])
                        else:
                            env.candidate_wps = []
                else:
                    boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])

                if env.current_wp is not None:
                    should_clear = False
                    c1 = env.current_wp.get("c1")
                    c2 = env.current_wp.get("c2")
                    mid = env.current_wp["pos"]
                    vec_to_wp = mid - env.boat_pos
                    dnow = math.hypot(vec_to_wp[0], vec_to_wp[1])
                    
                    # 1. 웨이포인트 중심점 근접 시 즉시 해제 (60px 이내)
                    if dnow < 60:
                        should_clear = True
                        
                    # 2. 웨이포인트 게이트 선 통과 판정 (c1, c2 사이 게이트 선을 전방으로 통과 시 즉시 해제)
                    if not should_clear and c1 is not None and c2 is not None:
                        vgx = c2[0] - c1[0]; vgy = c2[1] - c1[1]
                        gate_len = math.hypot(vgx, vgy)
                        if gate_len > 1e-3:
                            ugx = vgx / gate_len; ugy = vgy / gate_len
                            ngx = -ugy; ngy = ugx
                            hx = math.cos(env.boat_heading); hy = math.sin(env.boat_heading)
                            if ngx * hx + ngy * hy < 0:
                                ngx = -ngx; ngy = -ngy
                            rbx = env.boat_pos[0] - mid[0]; rby = env.boat_pos[1] - mid[1]
                            d_normal = rbx * ngx + rby * ngy
                            d_lateral = abs(rbx * ugx + rby * ugy)
                            if 15.0 <= d_normal < 60.0 and d_lateral < (gate_len / 2.0 + 20.0):
                                should_clear = True
                                
                    # 3. 웨이포인트를 이미 지나쳐 측후방으로 넘어간 경우 (95도 이상 & 75px 이내)
                    if not should_clear:
                        wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
                        angle_diff = abs(wrap(wp_angle - env.boat_heading))
                        if angle_diff > 1.6580627893946132 and dnow < 75:  # np.deg2rad(95)
                            should_clear = True
                        
                    if should_clear:
                        p = env.current_wp["pair"]
                        env.visited.add(p)
                        env.visited.add((p[1], p[0]))
                        # 1차 웨이포인트 통과 시 2차 웨이포인트가 미리 감지되어 있으면 부드럽게 1차로 승격
                        if env.next_wp is not None:
                            env.current_wp = env.next_wp
                            env.next_wp = None
                        else:
                            env.current_wp = None
                            env.total_gaps_count = 0
                            env.all_gaps = []
                        env.candidate_wps = []

                # 1차 웨이포인트 양쪽 장애물의 실시간 위치 및 점수 갱신
                if env.current_wp is not None:
                    id1, id2 = env.current_wp["pair"]
                    matched = False
                    if id1 in env.cluster_ids and id2 in env.cluster_ids:
                        idx1 = env.cluster_ids.index(id1)
                        idx2 = env.cluster_ids.index(id2)
                        c1_now = env.clusters[idx1]
                        c2_now = env.clusters[idx2]
                        env.current_wp["c1"] = c1_now
                        env.current_wp["c2"] = c2_now
                        env.current_wp["pos"] = (c1_now + c2_now) / 2.0
                        matched = True
                        if new_wp is not None and (new_wp["pair"] == env.current_wp["pair"] or new_wp["pair"] == (id2, id1)):
                            env.current_wp["score"] = new_wp["score"]
                            if "factors" in new_wp:
                                env.current_wp["factors"] = new_wp["factors"]
                    
                    # ID가 변경되었더라도 기존 부표 물리 좌표(c1, c2)와 가까운 클러스터(35px 이내)로 안정적 추종
                    if not matched:
                        c1_old = env.current_wp.get("c1")
                        c2_old = env.current_wp.get("c2")
                        if c1_old is not None and c2_old is not None and len(env.clusters) >= 2:
                            cl_arr = np.array(env.clusters)  # (N, 2)
                            d1 = np.sqrt(np.sum((cl_arr - c1_old)**2, axis=1))
                            d2 = np.sqrt(np.sum((cl_arr - c2_old)**2, axis=1))
                            i1, i2 = int(np.argmin(d1)), int(np.argmin(d2))
                            if d1[i1] < 35.0 and d2[i2] < 35.0 and i1 != i2:
                                env.current_wp["c1"] = env.clusters[i1]
                                env.current_wp["c2"] = env.clusters[i2]
                                env.current_wp["pos"] = (env.clusters[i1] + env.clusters[i2]) / 2.0
                                env.current_wp["pair"] = (env.cluster_ids[i1], env.cluster_ids[i2])

                # 1차 웨이포인트가 비어있을 때만 새로운 웨이포인트 최초 지정 (접근 중인 1차 WP를 전방 2차 WP로 덮어쓰지 않음)
                if new_wp is not None and env.current_wp is None:
                    env.current_wp = new_wp
                                
                if should_plan:
                    if env.current_wp is not None and not clear_to_target:
                        temp_visited = env.visited.copy()
                        temp_visited.add(env.current_wp["pair"])
                        temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))
                        
                        vec = env.current_wp["pos"] - env.boat_pos
                        next_head = math.atan2(vec[1], vec[0])
                        
                        # 2차 갭 탐색 (1차 웨이포인트 이후 전방에 장애물 갭이 존재하면 주황색 2차 웨이포인트로 표출)
                        env.next_wp = find_gap(
                            env.clusters, env.cluster_ids,
                            env.current_wp["pos"], next_head,
                            env.target, temp_visited,
                            env.grid, env.dynamic_obstacles,
                            params=env.params,
                            is_next_wp=True
                        )
                    else:
                        env.next_wp = None

                if should_plan:
                    boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
                    if env.current_wp is None:
                        # 목적지 직행 상황: 과도한 160px 외측 대우회를 방지하고 틈새로 직진 진입하도록 클리어런스 완화 (min_clearance=8.0)
                        goal = env.target
                        env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, min_clearance=8.0, boat_speed=boat_spd)
                    else:
                        # 웨이포인트(갭) 우회 통과 구간: 속도 기반 선행 회전 및 장애물 외측 굴곡 곡률 부여
                        goal = env.current_wp["pos"]
                        env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)
                        
                    if env.bezier_path is not None:
                        env.pursuit_target = pure_pursuit(env.bezier_path, env.boat_pos, lookahead=70)
                        
                    if env.current_wp is not None and env.next_wp is not None:
                        if env.bezier_path is not None and len(env.bezier_path) >= 2:
                            t1 = env.bezier_path[-1] - env.bezier_path[-2]
                            if math.hypot(t1[0], t1[1]) > 1e-6:
                                next_start_head = math.atan2(t1[1], t1[0])
                            else:
                                vec = env.current_wp["pos"] - env.boat_pos
                                next_start_head = math.atan2(vec[1], vec[0])
                        else:
                            vec = env.current_wp["pos"] - env.boat_pos
                            next_start_head = math.atan2(vec[1], vec[0])

                        env.next_bezier_path = make_bezier_path(
                            env.current_wp["pos"], next_start_head, env.next_wp["pos"],
                            obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd,
                            start_tangent_fixed=True
                        )
                        if env.next_bezier_path is not None:
                            env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=75)
                    else:
                        env.next_bezier_path = None
                        env.next_pursuit_target = None

                steer = env.update_steering(dists)

            if not getattr(env, 'manual_mode', False):
                if steer is None:
                    steer = 0
                L, R = env.get_pwm(steer)

            env.step(L, R, sub_step_idx=step_idx, total_sub_steps=sub_steps)
            env.update_camera()

            if not getattr(env, 'linetrace_mode', False) and not getattr(env, 'manual_mode', False):
                env.validate_wp_grid()
                env.validate_wp_obstacle_5x5()

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
                    break

        if hits_x is not None:
            env.render(hits_x, hits_y)
        env.clock.tick(120)

if __name__ == "__main__":
    run()