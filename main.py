import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import numpy as np
import math
import datetime
import time
import leaderboard
from environment import BoatEnv
from perception import lidar_hits_np, update_grid, extract_clusters_from_grid, match_clusters
from navigation import find_gap, target_is_clear, is_direct_target_safe, is_waypoint_switch_safe, is_front_blocked, line_trace_steering
from utils import wrap, make_bezier_path, pure_pursuit
from playback_scheduler import playback_budget
from frame_capture import save_episode_frame, start_capture_worker
from main_safety_guard import MainSafetyGuard
from dynamic_path_feasibility import DynamicsPathSelector
from momentum_gap_router import (MomentumGapRouter, perceived_circles,
                                 portal_crossing, portal_interval, wall_lidar_hits)
from vessel_dynamics import allocate
import main_line_compat as main_line
from portal_navigation import (choose_portal_crossing, path_has_hull_clearance,
                               remaining_path, portal_crossing_status)

BASE_PLAYBACK_RATE = 2.4  # CODEX-equivalent display rate; dt remains 0.04 s.
MAX_PHYSICS_STEPS_PER_RENDER = 8  # retain unexecuted budget for later frames

def run():
    env = BoatEnv()
    # main_heavy runs V2.1 by default; explicit 0 retains the debug fallback.
    phase5_mode = os.environ.get('MAIN_HEAVY_MOMENTUM_GAP', '1') == '1'
    geometric_portals = (phase5_mode and
                         os.environ.get('MAIN_HEAVY_GEOMETRIC_PORTALS', '1') == '1')
    adaptive_pp_ab = (phase5_mode and
                      os.environ.get('MAIN_HEAVY_ADAPTIVE_PP_AB', '') == '1')
    dynamic_path_mode = not phase5_mode and os.environ.get('MAIN_HEAVY_DYN_PATH', '') == '1'
    portal_mode = not (dynamic_path_mode or phase5_mode) and os.environ.get('MAIN_HEAVY_PORTAL', '') == '1'
    path_selector = DynamicsPathSelector() if dynamic_path_mode else None
    v2_mode = phase5_mode and os.environ.get('MAIN_HEAVY_MOTION_VERSION', 'V2') != 'V1'
    motion_v2 = None
    if v2_mode:
        from heavy_motion_v2 import HeavyMotionV2, warmup
        motion_v2 = HeavyMotionV2(env)
        warmup(env)
        motion_v2.start(env)
    momentum_router = MomentumGapRouter() if phase5_mode and not v2_mode else None
    if momentum_router is not None:
        env.momentum_gap_router = momentum_router
    if dynamic_path_mode:
        env.dynamic_path_selector = path_selector
    portal_trace = portal_mode and os.environ.get('MAIN_HEAVY_PORTAL_TRACE', '') == '1'
    hull_polygons = ((env.left_hull_local, env.right_hull_local, env.deck_local)
                     if portal_mode else None)

    def portal(gap, position, heading, downstream, previous=None, forced_s=None):
        if gap is None:
            return None
        diagnostic = {} if portal_trace else None
        selected = choose_portal_crossing(
            gap, position, heading, env.boat_vel, env.boat_ang_vel,
            downstream, env.dynamic_obstacles, hull_polygons, env.dynamics,
            previous_s=previous, plan_dt=env.dt * max(1, int(env.sim_speed)),
            diagnostic=diagnostic, forced_s=forced_s)
        if portal_trace:
            env.portal_decisions.append((tuple(map(int, gap['pair'])), diagnostic['reason']))
        return selected

    def same_portal(first, second):
        return first is not None and second is not None and set(first['pair']) == set(second['pair'])
    safety_guard = MainSafetyGuard()
    start_capture_worker()

    # Do not charge initialization or renderer warmup to physics playback.
    env.clock.tick(120)
    last_tick_time = time.perf_counter()
    accumulator = 0.0
    scheduled_speed = env.sim_speed
    hits_x = hits_y = None

    while True:
        env.clock.tick(120)
        now = time.perf_counter()
        elapsed = now - last_tick_time
        last_tick_time = now
        mode_generation = getattr(env, 'line_mode_generation', 0)
        before_events = (env.sim_speed, env.manual_mode, env.paused,
                         env.show_leaderboard, env.fullscreen_3d,
                         id(env.obstacles))
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                if hasattr(env, 'renderer') and hasattr(env.renderer, 'engine_3d') and env.renderer.engine_3d:
                    env.renderer.engine_3d.close()
                if motion_v2 is not None:
                    motion_v2.close()
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
                    if motion_v2 is not None:
                        motion_v2.close()
                    pygame.quit()
                    return
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    env.handle_click(e.pos)
                    if getattr(env, 'needs_break', False):
                        env.needs_break = False
                        break

        after_events = (env.sim_speed, env.manual_mode, env.paused,
                        env.show_leaderboard, env.fullscreen_3d,
                        id(env.obstacles))
        if getattr(env, 'line_mode_generation', 0) != mode_generation:
            hits_x = hits_y = None
            safety_guard = MainSafetyGuard()
        timing_reset = after_events != before_events
        if timing_reset:
            if motion_v2 is not None and after_events[1] != before_events[1]:
                motion_v2.reset_episode(env)
            accumulator = 0.0
            elapsed = 0.0
            hits_x = hits_y = None
            last_tick_time = time.perf_counter()

        if getattr(env, 'paused', False) or getattr(env, 'show_leaderboard', False):
            accumulator = 0.0
            map_bounds = (0, 0, env.map_w, env.sim_h) if (v2_mode or getattr(env, 'linetrace_mode', False)) else None
            raycast = lidar_hits_np
            if main_line.active(env):
                raycast = main_line.lidar_hits_np
            elif v2_mode:
                from heavy_motion_core.perception import lidar_hits_np as raycast
            dists, hits_x, hits_y = raycast(
                env.boat_pos, env.boat_heading,
                env.rel_angles, env.dynamic_obstacles,
                env.lidar_range,
                map_bounds=map_bounds
            )
            if phase5_mode and not v2_mode and not env.manual_mode and not main_line.active(env):
                dists, hits_x, hits_y = wall_lidar_hits(env, dists, hits_x, hits_y)
            env.lidar_dists = dists
            env.render(hits_x, hits_y)
            last_tick_time = time.perf_counter()
            continue

        accumulator = playback_budget(accumulator, elapsed, scheduled_speed,
                                      env.sim_speed, BASE_PLAYBACK_RATE)
        scheduled_speed = env.sim_speed
        sub_steps = min(MAX_PHYSICS_STEPS_PER_RENDER, int(accumulator / env.dt))
        if v2_mode and not env.manual_mode and not env.linetrace_mode:
            available = motion_v2.available()
            if available is not None:
                sub_steps = min(sub_steps,available)
        accumulator -= sub_steps * env.dt
        # At 1x the original MAIN planned on every physics step, even if
        # several steps now share a render frame.
        plan_interval = max(1, int(env.sim_speed))
        keys = pygame.key.get_pressed() if env.manual_mode else None
        new_wp = None

        for step_idx in range(sub_steps):
            if v2_mode and not env.manual_mode and not env.linetrace_mode:
                hits_x, hits_y = motion_v2.advance(env, step_idx, sub_steps)
            else:
                env.frame += 1
                if momentum_router is not None and env.frame <= momentum_router.last_frame:
                    momentum_router.reset_episode()
                env.update_dynamic_obstacles()

                map_bounds = (0, 0, env.map_w, env.sim_h) if (v2_mode or getattr(env, 'linetrace_mode', False)) else None
                raycast = main_line.lidar_hits_np if main_line.active(env) else lidar_hits_np
                dists, hits_x, hits_y = raycast(
                    env.boat_pos, env.boat_heading,
                    env.rel_angles, env.dynamic_obstacles,
                    env.lidar_range,
                    map_bounds=map_bounds
                )
                buoy_hits_x, buoy_hits_y = hits_x, hits_y
                if phase5_mode and not v2_mode and not env.manual_mode and not main_line.active(env):
                    dists, hits_x, hits_y = wall_lidar_hits(env, dists, hits_x, hits_y)
                env.lidar_dists = dists

                update_grid(env.grid, buoy_hits_x, buoy_hits_y)
                env.grid *= 0.945

                # 연산 부하 절감을 위한 적응형 인지/탐색 주기 (4배속 이하는 매 스텝 100% 실행)
                should_plan = (step_idx % plan_interval == 0 or step_idx == sub_steps - 1)
                # Line Tracing consumes raw LiDAR directly and clears all
                # GAP annotations below. Clustering those same rays here is
                # unused work; keep the occupancy grid/rendering unchanged.
                if should_plan and not getattr(env, 'linetrace_mode', False):
                    if portal_trace:
                        env.portal_decisions = []
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

                        if getattr(env, 'show_all_gaps', True) and n_fc >= 2:
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

                        L, R = env.get_manual_pwm()
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
                    steer, h_target, nearest_distance, c_hit = line_trace_steering(
                        env.boat_pos, env.boat_heading, env.target,
                        dists, env.rel_angles,
                        env.boat_ang_vel, env.prev_steer
                    )
                    env.heading_target = h_target
                    # MAIN uses its original wide-angle speed input.
                    env.min_wide_dist = nearest_distance
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
                        planning_obstacles = (perceived_circles(buoy_hits_x, buoy_hits_y, env.clusters,
                                              env.boat_pos, float(env.obs_r))
                                              if phase5_mode else env.dynamic_obstacles)
                        clear_to_target = (dist_to_target <= 400.0) and is_direct_target_safe(env.boat_pos, env.boat_heading, env.target, planning_obstacles, env.boat_radius, boat_spd, params=env.params)

                        # 목적지와 400픽셀 이하로 가까워졌고, 목적지 방향 직선 경로에 장애물이 없으면 즉시 목적지 직행
                        if clear_to_target:
                            new_wp = None
                            env.current_wp = None
                            env.next_wp = None
                            env.candidate_wps = []
                        else:
                            # 경로 상에 장애물이 있으면 장애물 사이 갭(웨이포인트)을 찾아 안전하게 우회
                            if geometric_portals:
                                options = momentum_router.portal_options(env)
                                new_wp = options[0] if options else None
                                if new_wp is not None:
                                    new_wp['candidates'] = options[1:3]
                            else:
                                new_wp = find_gap(
                                    env.clusters, env.cluster_ids,
                                    env.boat_pos, env.boat_heading,
                                    env.target, env.visited,
                                    env.grid, planning_obstacles,
                                    params=env.params
                                )
                            if phase5_mode and new_wp is not None:
                                ranked = [new_wp] + new_wp.get('candidates', [])
                                new_wp = next((gap for gap in ranked
                                               if momentum_router.gap_ahead(env, gap)),
                                              None)
                            if new_wp is not None:
                                env.candidate_wps = new_wp.get("candidates", [])
                            elif env.current_wp is not None:
                                env.candidate_wps = env.current_wp.get("candidates", [])
                            else:
                                env.candidate_wps = []
                        if phase5_mode:
                            env.phase5_new_wp_pair = (None if new_wp is None else
                                                     tuple(new_wp['pair']))
                        if portal_trace:
                            env.portal_gap_count = 0 if new_wp is None else 1 + len(new_wp.get('candidates', []))
                            env.portal_pair_visible = (env.current_wp is not None and
                                all(endpoint in env.cluster_ids for endpoint in env.current_wp['pair']))
                            env.portal_pair_visited = (env.current_wp is not None and
                                env.current_wp['pair'] in env.visited)
                    else:
                        boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])

                    if env.current_wp is not None:
                        should_clear = False
                        phase5_invalid_portal = False
                        c1 = env.current_wp.get("c1")
                        c2 = env.current_wp.get("c2")
                        mid = env.current_wp.get('portal_midpoint', env.current_wp["pos"]) if portal_mode else env.current_wp["pos"]
                        vec_to_wp = mid - env.boat_pos
                        dnow = math.hypot(vec_to_wp[0], vec_to_wp[1])

                        if portal_mode:
                            should_clear, current_signed = portal_crossing_status(
                                env.current_wp, env.boat_pos,
                                env.current_wp.get('portal_last_signed'))
                            env.current_wp['portal_last_signed'] = current_signed

                        # 1. 웨이포인트 중심점 근접 시 즉시 해제 (60px 이내)
                        if dnow < 60 and not portal_mode and not phase5_mode:
                            should_clear = True

                        # 2. 웨이포인트 게이트 선 통과 판정 (c1, c2 사이 게이트 선을 전방으로 통과 시 즉시 해제)
                        if not portal_mode and not phase5_mode and not should_clear and c1 is not None and c2 is not None:
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
                        if not portal_mode and not phase5_mode and not should_clear:
                            wp_angle = math.atan2(vec_to_wp[1], vec_to_wp[0])
                            angle_diff = abs(wrap(wp_angle - env.boat_heading))
                            if angle_diff > 1.6580627893946132 and dnow < 75:  # np.deg2rad(95)
                                should_clear = True

                        if phase5_mode and c1 is not None and c2 is not None:
                            axis = np.asarray(c2) - np.asarray(c1)
                            axis_length = math.hypot(float(axis[0]), float(axis[1]))
                            if axis_length > 1e-8:
                                axis /= axis_length
                                safe_span = momentum_router.gap_interval(env, env.current_wp)
                                phase5_invalid_portal = (safe_span is None or
                                                         not momentum_router.gap_ahead(env, env.current_wp))
                                previous_position = getattr(momentum_router, 'last_position', None)
                                if previous_position is not None and safe_span is not None:
                                    should_clear, _ = portal_crossing(
                                        previous_position, env.boat_pos,
                                        env.current_wp, safe_span, env.target)
                                if not should_clear:
                                    normal = np.array([-axis[1], axis[0]])
                                    if np.dot(normal, env.target - c1) < 0:
                                        normal = -normal
                                    hull_extent = max(math.hypot(px, py)
                                                      for poly in (env.left_hull_local,
                                                                   env.right_hull_local,
                                                                   env.deck_local)
                                                      for px, py in poly)
                                    if np.dot(env.boat_pos - c1, normal) > 2.0*hull_extent:
                                        phase5_invalid_portal = True

                        if phase5_invalid_portal:
                            # A closed portal cannot remain an active topology
                            # target. Do not mark it visited: it was not crossed.
                            env.current_wp = None
                            env.next_wp = None
                            env.candidate_wps = []
                            should_clear = False

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
                            if not portal_mode:
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
                                    if not portal_mode:
                                        env.current_wp["pos"] = (env.clusters[i1] + env.clusters[i2]) / 2.0
                                    env.current_wp["pair"] = (env.cluster_ids[i1], env.cluster_ids[i2])

                    # 1차 웨이포인트가 비어있을 때만 새로운 웨이포인트 최초 지정 (접근 중인 1차 WP를 전방 2차 WP로 덮어쓰지 않음)
                    if new_wp is not None and env.current_wp is None:
                        if portal_mode:
                            options = [new_wp] + new_wp.get('candidates', [])
                            env.current_wp = next((choice for option in options
                                                   if (choice := portal(option, env.boat_pos,
                                                                        env.boat_heading, env.target)) is not None), None)
                        else:
                            env.current_wp = new_wp
                        if portal_trace:
                            env.portal_initial_selection_failed = env.current_wp is None
                    if (phase5_mode and should_plan and new_wp is not None and
                            env.current_wp is not None and
                            not same_portal(new_wp, env.current_wp) and
                            momentum_router.last_result is not None and
                            momentum_router.last_result['progress_px'] <= 0.0 and
                            momentum_router.no_progress_start_frame is not None and
                            (env.frame - momentum_router.no_progress_start_frame) * env.dt >=
                            env.dynamics.yaw_response_s):
                        # A geometrically open portal need not be reachable from
                        # this momentum state. If the current safe rollout makes
                        # no portal progress for one yaw-response interval, try
                        # perceived next GAP without marking the old one visited.
                        env.current_wp = new_wp
                        env.next_wp = None
                        env.candidate_wps = new_wp.get('candidates', [])
                        env.portal_abandon_reason = 'stopped_without_safe_progress'

                    if should_plan:
                        if env.current_wp is not None and not clear_to_target:
                            temp_visited = env.visited.copy()
                            temp_visited.add(env.current_wp["pair"])
                            temp_visited.add((env.current_wp["pair"][1], env.current_wp["pair"][0]))

                            vec = env.current_wp["pos"] - env.boat_pos
                            next_head = math.atan2(vec[1], vec[0])

                            # 2차 갭 탐색 (1차 웨이포인트 이후 전방에 장애물 갭이 존재하면 주황색 2차 웨이포인트로 표출)
                            if geometric_portals:
                                downstream = momentum_router.portal_options(
                                    env, position=env.current_wp['pos'],
                                    visited=temp_visited)
                                proposed_next = downstream[0] if downstream else None
                            else:
                                proposed_next = find_gap(
                                    env.clusters, env.cluster_ids,
                                    env.current_wp["pos"], next_head,
                                    env.target, temp_visited,
                                    env.grid, planning_obstacles,
                                    params=env.params,
                                    is_next_wp=True
                                )
                            if portal_mode and proposed_next is not None:
                                options = [proposed_next] + proposed_next.get('candidates', [])
                                env.next_wp = next((choice for option in options
                                                    if (choice := portal(option, env.current_wp['pos'],
                                                                         env.current_wp.get('portal_heading', next_head),
                                                                         env.target,
                                                                         env.next_wp.get('portal_s') if same_portal(env.next_wp, option) else None)) is not None), None)
                            else:
                                env.next_wp = proposed_next
                        else:
                            env.next_wp = None

                        if portal_mode and env.current_wp is not None:
                            downstream = env.target if env.next_wp is None else env.next_wp['pos']
                            updated = portal(env.current_wp, env.boat_pos, env.boat_heading,
                                             downstream, env.current_wp.get('portal_s'))
                            if updated is not None:
                                env.current_wp = updated
                            else:
                                old_route = remaining_path(env.bezier_path, env.boat_pos)
                                if not path_has_hull_clearance(old_route, env.dynamic_obstacles, hull_polygons):
                                    env.current_wp = None
                                    env.next_wp = None

                    if should_plan and (not phase5_mode or adaptive_pp_ab):
                        boat_spd = math.hypot(env.boat_vel[0], env.boat_vel[1])
                        old_path = env.bezier_path
                        old_pair = getattr(env, 'portal_path_pair', None)
                        env.portal_no_safe_route = False
                        if env.current_wp is None:
                            # 목적지 직행 상황: 과도한 160px 외측 대우회를 방지하고 틈새로 직진 진입하도록 클리어런스 완화 (min_clearance=8.0)
                            goal = env.target
                            env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, min_clearance=8.0, boat_speed=boat_spd)
                        else:
                            # 웨이포인트(갭) 우회 통과 구간: 속도 기반 선행 회전 및 장애물 외측 굴곡 곡률 부여
                            goal = env.current_wp["pos"]
                            env.bezier_path = make_bezier_path(env.boat_pos, env.boat_heading, goal, obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd)

                        if portal_mode and env.current_wp is not None:
                            safe = path_has_hull_clearance(env.bezier_path,
                                                           env.dynamic_obstacles, hull_polygons)
                            if not safe:
                                low, high = env.current_wp['portal_safe_interval']
                                preferred = env.current_wp['portal_s']
                                alternatives = sorted((low, (low + high) / 2.0, high),
                                                      key=lambda s: abs(s - preferred))
                                downstream = env.target if env.next_wp is None else env.next_wp['pos']
                                for s in alternatives:
                                    alternate = portal(env.current_wp, env.boat_pos,
                                                       env.boat_heading, downstream,
                                                       forced_s=s)
                                    if alternate is None:
                                        continue
                                    alternative_path = make_bezier_path(
                                        env.boat_pos, env.boat_heading, alternate['pos'],
                                        obstacles=env.dynamic_obstacles,
                                        boat_radius=env.boat_radius, boat_speed=boat_spd)
                                    if path_has_hull_clearance(alternative_path,
                                                               env.dynamic_obstacles, hull_polygons):
                                        env.current_wp = alternate
                                        env.bezier_path = alternative_path
                                        safe = True
                                        break
                            if not safe and old_pair == tuple(sorted(env.current_wp['pair'])):
                                continuation = remaining_path(old_path, env.boat_pos)
                                if path_has_hull_clearance(continuation,
                                                           env.dynamic_obstacles, hull_polygons):
                                    env.bezier_path = continuation
                                    safe = True
                            if not safe and new_wp is not None:
                                # The selected portal may be geometrically open
                                # while its approach curve is unsafe. Try another
                                # observed gap before falling back to direct mode.
                                for option in [new_wp] + new_wp.get('candidates', []):
                                    if same_portal(option, env.current_wp):
                                        continue
                                    alternate = portal(option, env.boat_pos,
                                                       env.boat_heading, env.target)
                                    if alternate is None:
                                        continue
                                    alternative_path = make_bezier_path(
                                        env.boat_pos, env.boat_heading, alternate['pos'],
                                        obstacles=env.dynamic_obstacles,
                                        boat_radius=env.boat_radius, boat_speed=boat_spd)
                                    if path_has_hull_clearance(alternative_path,
                                                               env.dynamic_obstacles, hull_polygons):
                                        env.current_wp = alternate
                                        env.next_wp = None
                                        env.bezier_path = alternative_path
                                        safe = True
                                        break
                            env.portal_no_safe_route = not safe
                            if safe:
                                env.portal_path_pair = tuple(sorted(env.current_wp['pair']))

                        if env.current_wp is not None and (env.next_wp is not None or portal_mode):
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

                            if env.next_wp is not None:
                                continuation_goal = env.next_wp['pos']
                            else:
                                portal_heading = env.current_wp['portal_heading']
                                continuation_goal = env.current_wp['pos'] + 110.0 * np.array(
                                    [math.cos(portal_heading), math.sin(portal_heading)])
                            env.next_bezier_path = make_bezier_path(
                                env.current_wp["pos"], next_start_head, continuation_goal,
                                obstacles=env.dynamic_obstacles, boat_radius=env.boat_radius, boat_speed=boat_spd,
                                start_tangent_fixed=True
                            )
                            if env.next_bezier_path is not None and env.next_wp is not None:
                                env.next_pursuit_target = pure_pursuit(env.next_bezier_path, env.current_wp["pos"], lookahead=75)
                        else:
                            env.next_bezier_path = None
                            env.next_pursuit_target = None

                        if dynamic_path_mode:
                            original_wp = env.current_wp
                            options = ([] if new_wp is None else
                                       [new_wp] + new_wp.get('candidates', []))
                            selected_path, selected_wp, _ = path_selector.select(
                                env, env.bezier_path, env.current_wp, options)
                            env.bezier_path = selected_path
                            env.current_wp = selected_wp
                            if selected_wp is not original_wp:
                                env.next_wp = None
                                env.next_bezier_path = None
                                env.next_pursuit_target = None

                        if env.bezier_path is not None:
                            if portal_mode and env.next_bezier_path is not None:
                                control_path = np.concatenate((env.bezier_path, env.next_bezier_path[1:]), axis=0)
                            else:
                                control_path = env.bezier_path
                            env.pursuit_target = pure_pursuit(control_path, env.boat_pos, lookahead=70)
                        else:
                            env.pursuit_target = None

                    if phase5_mode:
                        if should_plan or momentum_router.last_result is None:
                            phase5_result = momentum_router.choose(
                                env, hits_x, hits_y, env.current_wp,
                                adaptive_path=env.bezier_path if adaptive_pp_ab else None,
                                observed=planning_obstacles if should_plan else None)
                        else:
                            phase5_result = momentum_router.last_result
                        env.bezier_path = phase5_result['bezier_reference']
                        env.next_bezier_path = None
                        # Visualization/reference only; predictive commands are authoritative.
                        lookahead = (42.0 + max(0., env.physics_state()[3]) *
                                     env.dynamics.pixels_per_m * env.dynamics.yaw_response_s)
                        env.pursuit_target = (None if env.bezier_path is None else
                            pure_pursuit(env.bezier_path, env.boat_pos, lookahead=lookahead))
                        env.next_pursuit_target = None
                        env.command_speed = float(phase5_result['speed_command'])
                        env.command_yaw_rate = float(phase5_result['yaw_rate_command'])
                        env.prev_steer = float(phase5_result['yaw_rate_command'] /
                                               max(env.dynamics.max_yaw_rate_rad_s, 1e-9))
                        env.heading_target = env.boat_heading + env.command_yaw_rate * env.dt
                        steer = env.prev_steer
                        momentum_router.last_position = np.asarray(env.boat_pos).copy()
                    else:
                        steer = env.update_steering(dists)

                if not getattr(env, 'manual_mode', False):
                    if steer is None:
                        steer = 0
                    if phase5_mode and not getattr(env, 'linetrace_mode', False):
                        L, R = phase5_result['left_pwm'], phase5_result['right_pwm']
                    else:
                        L, R = env.get_pwm(steer)
                    L, R = safety_guard.command(env, L, R, steer)
                    if (dynamic_path_mode and not main_line.active(env) and
                            path_selector.last_reason == 'brake_no_feasible_path' and
                            safety_guard.hold_steps == 0):
                        left, right = allocate(env.physics_state(), 0.0, 0.0,
                                               env.dynamics)
                        L = 1500 + 400*float(left)/env.dynamics.max_thrust_N
                        R = 1500 + 400*float(right)/env.dynamics.max_thrust_N
                    if portal_mode and not main_line.active(env) and getattr(env, 'portal_no_safe_route', False):
                        L = R = 1500

                # One 1x step must retain the old 120-FPS per-step environment
                # behavior, including work formerly keyed to the render batch.
                if env.sim_speed == 1:
                    env.step(L, R, sub_step_idx=0, total_sub_steps=1)
                else:
                    env.step(L, R, sub_step_idx=step_idx, total_sub_steps=sub_steps)
                env.update_camera()

                if not getattr(env, 'linetrace_mode', False) and not getattr(env, 'manual_mode', False):
                    if not portal_mode:
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
                    timing_reset = True
                    break
                # 수동 조종 모드에서는 충돌 발생 시 에피소드를 종료/리스폰하지 않고 계속 주행함
            else:
                reached = (main_line.goal_reached(dist_tgt_end) if main_line.active(env)
                           else dist_tgt_end <= 70 if v2_mode else dist_tgt_end < 70)
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
                            env.capture_frame = not v2_mode
                            env.render(hits_x, hits_y)
                        save_episode_frame(env.screen, p)
                    except:
                        pass
                    finally:
                        env.capture_frame = False
                    env.reset()
                    if v2_mode:
                        # CODEX returns the unfinished batch; no physics debt
                        # is dropped at an autonomous episode boundary.
                        accumulator += (sub_steps-step_idx-1)*env.dt
                    else:
                        timing_reset = True
                    hits_x = hits_y = None
                    break

        if hits_x is None:
            map_bounds = (0, 0, env.map_w, env.sim_h) if (v2_mode or getattr(env, 'linetrace_mode', False)) else None
            raycast = lidar_hits_np
            if main_line.active(env):
                raycast = main_line.lidar_hits_np
            elif v2_mode:
                from heavy_motion_core.perception import lidar_hits_np as raycast
            dists, hits_x, hits_y = raycast(
                env.boat_pos, env.boat_heading, env.rel_angles,
                env.dynamic_obstacles, env.lidar_range, map_bounds=map_bounds
            )
            if phase5_mode and not v2_mode and not env.manual_mode and not main_line.active(env):
                dists, hits_x, hits_y = wall_lidar_hits(env, dists, hits_x, hits_y)
            env.lidar_dists = dists
        env.render(hits_x, hits_y)
        if timing_reset:
            accumulator = 0.0
            last_tick_time = time.perf_counter()

if __name__ == "__main__":
    run()
