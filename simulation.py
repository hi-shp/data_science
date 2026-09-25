"""One fixed simulation step shared by the GUI and evaluation tools."""
import math
import numpy as np
import pygame
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, is_direct_target_safe, line_trace_steering
from utils import wrap
from navigation_map import NavigationMap


def advance(env, step_idx=0, sub_steps=1):
    # Planning is tied to simulation time, never to display FPS or speed buttons.
    plan_interval = env.control.planning_period_steps
    new_wp = None
    env.frame += 1
    env.update_dynamic_obstacles()

    map_bounds = (0, 0, env.map_w, env.sim_h)
    dists, hits_x, hits_y = lidar_hits_np(
        env.boat_pos, env.boat_heading,
        env.rel_angles, env.dynamic_obstacles,
        env.lidar_range,
        map_bounds=map_bounds
    )
    env.lidar_dists = dists

    update_grid(env.grid, hits_x, hits_y)
    env.grid *= 0.945

    # 시뮬레이션 시간 기준으로 인지/경로 계획 주기를 고정한다.
    should_plan = ((env.frame - 1) % plan_interval == 0)
    if should_plan:
        scale = env.dynamics.pixels_per_m
        if not hasattr(env, 'navigation_map'):
            env.navigation_map = NavigationMap(env.map_w/scale, env.sim_h/scale)
        env.navigation_map.observe(env.boat_pos/scale, env.boat_heading,
                                   env.rel_angles, dists/scale,
                                   env.lidar_range/scale, env.frame*env.dt)
        env.perceived_obstacles = env.navigation_map.obstacles*scale
        new_c = extract_clusters_from_grid(env.grid)
        env.clusters, env.cluster_ids = match_clusters(
            env.clusters, env.cluster_ids, new_c
        )

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
        env.pursuit_target = None
        env.controller_target = None
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
        env.pursuit_target = None
        env.controller_target = None
        env.all_gaps = []
        env.total_gaps_count = 0
    else:
        if should_plan:
            dist_to_target = math.hypot(env.target[0] - env.boat_pos[0], env.target[1] - env.boat_pos[1])
            boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
            clear_to_target = (dist_to_target <= 400.0) and is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, env.perceived_obstacles, env.boat_radius, boat_spd, params=env.params)

            # 목적지와 400픽셀 이하로 가까워졌고, 목적지 방향 직선 경로에 장애물이 없으면 즉시 목적지 직행
            if clear_to_target:
                new_wp = None
                env.current_wp = None
                env.next_wp = None
            else:
                # 경로 상에 장애물이 있으면 장애물 사이 갭(웨이포인트)을 찾아 안전하게 우회
                new_wp = find_gap(
                    env.clusters, env.cluster_ids,
                    env.boat_pos, env.boat_heading,
                    env.target, env.visited,
                    env.grid, env.perceived_obstacles,
                    params=env.params
                )

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
                    env.grid, env.perceived_obstacles,
                    params=env.params,
                    is_next_wp=True
                )
            else:
                env.next_wp = None

        steer = env.update_steering(dists)

    if steer is None:
        steer = 0
    L, R = env.get_pwm(steer)

    env.step(L, R, sub_step_idx=step_idx, total_sub_steps=sub_steps)
    env.update_camera()

    if not getattr(env, 'linetrace_mode', False) and not getattr(env, 'manual_mode', False):
        env.validate_wp_grid()
        env.validate_wp_obstacle_5x5()

    return hits_x, hits_y
