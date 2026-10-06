"""Pinned MAIN Line Tracing baseline; never used by native navigation.

The baseline is MAIN's legacy pixel/world-velocity integrator, not the SI
VesselParameters model. Do not mix its PWM*10 force scale with SI thrusts.
Controller, sensor, integrator and contact methods below are copied verbatim
from the reference and checked against it by the compatibility tests.
"""
import math
import random
from dataclasses import dataclass
import numpy as np
import pygame
from utils import wrap
from hull_collision import hull_collides

# MAIN sensor's immutable 180-beam trig cache, with its original float32 angles.
_REL_ANGLES = np.linspace(-np.pi, np.pi, 180, endpoint=False, dtype=np.float32)
_COS_REL = np.cos(_REL_ANGLES)[:, None]
_SIN_REL = np.sin(_REL_ANGLES)[:, None]

MAIN_REFERENCE_COMMIT = '1b6d5ecaca38575f26bb0b01a9b98912dd402e92'

MAIN_LINE_PARAMETERS = {'steer_gain': 1.1, 'steer_alpha': 0.3515, 'mom_coeff': 0.00665, 'pwm_rng': 200.36, 'avoid_normal': 0.05, 'avoid_em': 0.3, 'clear_margin': 10, 'em_enter': 125.0, 'em_exit': 160.0, 'em_hold_frames': 18, 'align_exp': 6.0, 'heading_exp': 4.0, 'fwd_exp': 6.0, 'width_exp': 4.0, 'clear_exp': 4.0, 'wp_switch_thresh': 1.1, 'perp_exp': 2.0}

@dataclass(frozen=True)
class MainLinePhysics:
    mass: float = 10
    inertia: float = 4.5
    drag: float = 0.2
    rot_drag: float = 0.8
    dt: float = 0.04

MAIN_LINE_PHYSICS = MainLinePhysics()

def active(env):
    return bool(getattr(env, 'linetrace_mode', False) and not getattr(env, 'manual_mode', False))


def restore_native_parameters(env):
    saved = getattr(env, '_main_line_native_params', None)
    if saved is not None:
        env.params = saved
        del env._main_line_native_params
    physics = getattr(env, '_main_line_native_scalars', None)
    if physics is not None:
        for name, value in physics.items():
            if value is None:
                delattr(env, name)
            else:
                setattr(env, name, value)
        del env._main_line_native_scalars
    env.line_physics_profile = None


def activate_profile(env):
    if not active(env) or getattr(env, "line_physics_profile", None) is not None:
        return
    env._main_line_native_params = env.params.copy()
    env.params = dict(env.params, **MAIN_LINE_PARAMETERS)
    env.line_physics_profile = MAIN_LINE_PHYSICS
    env._main_line_native_scalars = {
        name: getattr(env, name, None) for name in ('mass', 'inertia', 'drag', 'rot_drag', 'dt')}
    for name in ('mass', 'inertia', 'drag', 'rot_drag', 'dt'):
        setattr(env, name, getattr(MAIN_LINE_PHYSICS, name))


def configure_episode(env):
    if not active(env):
        return
    activate_profile(env)
    # MAIN stores position in float32 and world velocity in float64.
    env.boat_pos = np.asarray(env.boat_pos, dtype=np.float32)
    env.current_fwd = 0.0
    env._main_line_applied_moment = 0.0
    env.min_wide_dist = 999.0
    env.prev_steer = 0.0
    env.thrust_left = env.thrust_right = 0.0


def invalidate_controller(env):
    """Discard mode-local references, never physical/map/episode state."""
    for name in ('navigation_map', 'trajectory_navigator', 'path_geometry',
                 'route_plan_frame', 'path_progress', '_rollout_params',
                 'visual_marker_state', 'motion_prediction_states'):
        if hasattr(env, name):
            delattr(env, name)
    for name in ('raw_route', 'control_path', 'predicted_trajectory',
                 'visual_trajectory', 'controller_target', 'visual_controller_target',
                 'current_wp', 'next_wp', 'selected_gap', 'bezier_path',
                 'next_bezier_path', 'pursuit_target', 'next_pursuit_target',
                 'visual_pursuit_target', 'closest_obstacle_hit', 'closest_avoid_hit'):
        setattr(env, name, None)
    env.all_gaps = []
    env.candidate_wps = []
    env.total_gaps_count = 0
    env.prev_steer = 0.0
    env.emergency_mode = False
    env.emergency_cooldown = 0
    env.stalled_s = env.recovery_until = 0.0
    env.prediction_frame = -1
    env.heading_target = env.boat_heading
    env.command_speed = env.dynamics.cruise_speed_m_s
    env.command_yaw_rate = 0.0
    env._line_resume_plan = not active(env)


def switch_mode(env, enabled):
    """Hot-swap profiles using propulsion-acceleration-equivalent outputs.

    MAIN's pixel force is NOT Newtons. Transfer common force through mass and
    pixels/m; transfer yaw moment through both inertias and MAIN's per-step
    .84 yaw multiplier. Native thrust limits bound the return conversion.
    Neither fluid drag nor current velocity is folded into actuator output.
    """
    enabled = bool(enabled)
    if enabled == bool(getattr(env, 'linetrace_mode', False)):
        return False
    was_active = active(env)
    env.linetrace_mode = enabled
    env.linetrace_queued = False
    now_active = active(env)
    native = env.dynamics
    if now_active and not was_active:
        left, right = env.thrust_left, env.thrust_right
        activate_profile(env)
        env.current_fwd = ((left + right) * env.mass * native.pixels_per_m
                           / native.mass_kg)
        env._main_line_applied_moment = ((right - left) * native.thruster_arm_m
                                       * env.inertia / native.yaw_inertia_kg_m2 / .84)
        # Initialize MAIN's controller memory from the applied differential,
        # not a stale normal-controller yaw request.
        difference = env._main_line_applied_moment / env.params['mom_coeff'] / 10.
        steer = math.copysign(min(1., abs(difference)/(2*env.params['pwm_rng']))
                             ** (1/1.15), difference)
    elif was_active and not now_active:
        common = env.current_fwd / env.mass / native.pixels_per_m * native.mass_kg
        moment = getattr(env, '_main_line_applied_moment', 0.)
        difference = moment / env.inertia * .84 * native.yaw_inertia_kg_m2 / native.thruster_arm_m
        differential = float(np.clip(difference/2, -native.max_thrust_N, native.max_thrust_N))
        common = float(np.clip(common/2, -native.max_thrust_N+abs(differential),
                               native.max_thrust_N-abs(differential)))
        env.thrust_left = common-differential
        env.thrust_right = common+differential
        env.current_fwd = env.thrust_left + env.thrust_right
        restore_native_parameters(env)
        steer = 0.0
    else:
        steer = 0.0
    invalidate_controller(env)
    if now_active:
        env.prev_steer = steer
        env.min_wide_dist = 999.0  # refreshed by MAIN sensor before its first command
    visuals = getattr(env, 'phase5_visuals', None)
    if visuals is not None:
        visuals.invalidate_controller(env)
        if now_active:
            env.prev_steer = steer
    env.line_mode_generation = getattr(env, 'line_mode_generation', 0) + 1
    return True


def apply_step(env, left, right, sub_step_idx=0, total_sub_steps=1):
    result = step(env, left, right, sub_step_idx, total_sub_steps)
    # MAIN has a smoothed common force and an algebraic differential moment.
    # Bookkeeping only: the copied MAIN integrator below remains unmodified.
    env._main_line_applied_moment = (pwm_to_thrust(env, right)-pwm_to_thrust(env, left))*env.params['mom_coeff']
    return result


def goal_reached(distance_px):
    return distance_px < 70


def line_trace_steering(boat_pos, boat_heading, target_pos, dists, rel_angles, boat_ang_vel=0.0, prev_steer=0.0):
    """
    [라인트레이싱 원리 반응형 항법 알고리즘]
    1순위: 목적지 방향 직행 조향 (Steer to Target)
    2순위: 전방 시야 내 장애물 조우 시 가장 가까운 히트점의 반대 방향으로 즉각적이고 민첩한 회피 조향
    - 급선회 제한(인위적 각도 캡)을 완전히 제거하여 위험 시 즉시 전방 장애물을 신속하게 회피.
    - 전방 시야각(65도) 밖으로 장애물이 벗어나면 코사인 감쇠 가중치에 의해 자연스럽게 1순위 목적지 직행으로 복귀하여 뒤로 도는 현상 원천 차단.
    - 광각(220도) 감지 거리를 반환하여 장애물 밀집 구간에서 선체 속도를 자연스럽게 감속 제어.
    """
    bx, by = boat_pos
    tx, ty = target_pos

    # 1. 목적지 방향 (1순위 기본 방향)
    goal_angle = math.atan2(ty - by, tx - bx)
    heading_err = wrap(goal_angle - boat_heading)
    steer_goal = float(np.clip(heading_err * 1.30, -1.0, 1.0))

    # 2. 전방 유효 시야각 (|rel_angle| <= 65도) 내 장애물 탐지
    fov_rad = 1.134464  # np.deg2rad(65)
    fwd_mask = np.abs(rel_angles) <= fov_rad
    fwd_indices = np.where(fwd_mask)[0]

    # 광각(220도) 전체 범위 내 최소 거리 (선박 물리 엔진의 연속 속도 제어 연동)
    wide_mask = np.abs(rel_angles) <= 1.91986  # np.deg2rad(110)
    min_wide = float(np.min(dists[wide_mask])) if np.any(wide_mask) else 999.0

    SAFE_DIST = 180.0       # 장애물 감지 및 회피 개시 거리 (px)
    CRIT_DIST = 55.0        # 긴급 완전 회피 기준 거리 (px)

    if len(fwd_indices) > 0:
        fwd_dists = dists[fwd_indices]
        min_i = int(np.argmin(fwd_dists))
        closest_idx = fwd_indices[min_i]
        min_dist = float(dists[closest_idx])
        closest_ang = float(rel_angles[closest_idx])
    else:
        min_dist = 999.0
        closest_ang = 0.0
        closest_idx = None

    closest_hit_world = None
    if min_dist < SAFE_DIST:
        # 가장 가까운 회피 대상 장애물 히트점 월드 좌표 계산 (빨간색 SHOW 표출용)
        closest_hit_world = (
            float(bx + math.cos(boat_heading + closest_ang) * min_dist),
            float(by + math.sin(boat_heading + closest_ang) * min_dist)
        )

        # 장애물 조우: 가장 가까운 히트점의 반대 방향으로 회피 조향
        # 우현(closest_ang > 0)에 장애물 -> 좌회전(avoid_dir < 0)
        # 좌현(closest_ang < 0)에 장애물 -> 우회전(avoid_dir > 0)
        if abs(closest_ang) > 0.04:
            avoid_dir = -float(np.sign(closest_ang))
        else:
            # 정면 정중앙 장애물: 좌/우 여유 공간 비교하여 더 넓게 트인 쪽으로 회피
            left_mask = (rel_angles < -0.05) & fwd_mask
            right_mask = (rel_angles > 0.05) & fwd_mask
            left_c = float(np.min(dists[left_mask])) if np.any(left_mask) else 0.0
            right_c = float(np.min(dists[right_mask])) if np.any(right_mask) else 0.0
            avoid_dir = -1.0 if left_c >= right_c else 1.0

        # 전방 각도 집중도(정면에 가까울수록 최대 회피력 발휘, 65도 경계로 벗어나면 부드럽게 0으로 수렴)
        front_f = max(0.0, math.cos(closest_ang * (np.pi / 2.0 / fov_rad)))
        urgency = float(np.clip((SAFE_DIST - min_dist) / (SAFE_DIST - CRIT_DIST), 0.0, 1.0))

        # 긴급도에 따른 적극적인 회피 조향
        avoid_steer = avoid_dir * (0.75 + 0.25 * urgency)
        avoid_weight = urgency * front_f

        # 근접 위험 시 급선회(100% 회피 조향) 허용
        if min_dist < CRIT_DIST + 15.0:
            steer_cmd = avoid_dir * 1.0
        else:
            steer_cmd = (1.0 - avoid_weight) * steer_goal + avoid_weight * avoid_steer
    else:
        steer_cmd = steer_goal

    # 측면 근접 보호(Flank Guard): 배 옆(65~95도)에 장애물이 45px 이내로 근접 시 외측 선체 찰과 방지
    flank_mask = (np.abs(rel_angles) > fov_rad) & (np.abs(rel_angles) <= 1.658)
    if np.any(flank_mask):
        f_dists = dists[flank_mask]
        f_min = float(np.min(f_dists))
        if f_min < 45.0:
            f_idx = np.where(flank_mask)[0][np.argmin(f_dists)]
            f_ang = float(rel_angles[f_idx])
            f_dir = -float(np.sign(f_ang))
            f_push = f_dir * float(np.clip((45.0 - f_min) / 20.0, 0.0, 0.5))
            if (steer_cmd * f_dir) <= 0:
                steer_cmd = steer_cmd * 0.5 + f_push

    # 3. 각속도 댐핑 및 지수 이동 평균 평활화 (오버슈트 및 횡방향 출렁임 방지)
    d_term = -0.25 * float(boat_ang_vel)
    steer_raw = float(np.clip(steer_cmd + d_term, -1.0, 1.0))
    steer_f = float(np.clip(0.55 * steer_raw + 0.45 * prev_steer, -1.0, 1.0))

    # HUD 표출용 지향 헤딩각
    cmd_heading = boat_heading + steer_f * 0.8

    return steer_f, cmd_heading, min_wide, closest_hit_world

def lidar_hits_np(boat_pos, boat_heading, rel_angles, obstacles, lidar_range, map_bounds=None):
    n = len(rel_angles)
    ch = math.cos(boat_heading)
    sh = math.sin(boat_heading)
    vx = ch * _COS_REL - sh * _SIN_REL
    vy = sh * _COS_REL + ch * _SIN_REL

    x0, y0 = boat_pos

    if len(obstacles) > 0:
        ox = obstacles[:, 0]
        oy = obstacles[:, 1]
        orad = obstacles[:, 2]
        px = ox - x0
        py = oy - y0
        p_sq = px * px + py * py
        max_reach = lidar_range + orad
        cand = p_sq < (max_reach * max_reach)
        if np.any(cand):
            px_c = px[cand][None, :]
            py_c = py[cand][None, :]
            orad_c = orad[cand][None, :]
            p_sq_c = p_sq[cand][None, :]
            base_c = orad_c * orad_c - p_sq_c
            b = px_c * vx + py_c * vy
            disc = base_c + b * b
            mask = (b > 0) & (disc >= 0)
            t = np.where(mask, b - np.sqrt(np.maximum(0.0, disc)), lidar_range)
            d_final = np.min(t, axis=1).astype(np.float32)
        else:
            d_final = np.full(n, lidar_range, dtype=np.float32)
    else:
        d_final = np.full(n, lidar_range, dtype=np.float32)

    # 맵 외곽 벽(Boundary Walls)을 장애물로 인식 (목적지 방향 정면 수직벽 xmax 제외)
    if map_bounds is not None:
        xmin, ymin, xmax, ymax = map_bounds
        t_left = np.where(vx < -1e-5, (xmin - x0) / np.minimum(vx, -1e-5), lidar_range)
        # 목적지 방향의 정면 수직벽(xmax)은 벽으로 인식하지 않음
        t_top = np.where(vy < -1e-5, (ymin - y0) / np.minimum(vy, -1e-5), lidar_range)
        t_bottom = np.where(vy > 1e-5, (ymax - y0) / np.maximum(vy, 1e-5), lidar_range)

        t_left = np.where(t_left > 0, t_left, lidar_range)
        t_top = np.where(t_top > 0, t_top, lidar_range)
        t_bottom = np.where(t_bottom > 0, t_bottom, lidar_range)

        t_wall = np.minimum(t_left, np.minimum(t_top, t_bottom))
        d_final = np.minimum(d_final, t_wall[:, 0].astype(np.float32))

    # 벡터화된 히트 좌표 연산 (Python for 루프 제거)
    vx_flat = vx[:, 0]
    vy_flat = vy[:, 0]
    hits_x = x0 + vx_flat * d_final
    hits_y = y0 + vy_flat * d_final
    valid = d_final < lidar_range
    # 유효하지 않은 히트점은 NaN으로 마스킹 (렌더러에서 None 대신 NaN 검사)
    hits_x_out = np.where(valid, hits_x, np.nan).astype(np.float32)
    hits_y_out = np.where(valid, hits_y, np.nan).astype(np.float32)

    return d_final, hits_x_out, hits_y_out

def get_pwm(self, steer):
        dead = 0.02
        if abs(steer) < dead: steer = 0
        mid = 1500; rng = self.params['pwm_rng']
        m = (abs(steer) ** 1.15)
        d = m * rng
        if steer >= 0: L = mid - d; R = mid + d
        else: L = mid + d; R = mid - d
        return int(np.clip(L, 1230, 1770)), int(np.clip(R, 1230, 1770))

def pwm_to_thrust(self, p):
        return p * 10

def step(self, L, R, sub_step_idx=0, total_sub_steps=1):
        tL = self.pwm_to_thrust(L)
        tR = self.pwm_to_thrust(R)

        if getattr(self, 'manual_mode', False):
            # 수동 조종 모드: W/S 키 입력에 따른 직접 추진력 제어
            m_thr = getattr(self, 'manual_throttle', 0.0)
            target_fwd = m_thr * 5500.0
            mom = (tR - tL) * self.params['mom_coeff']
        else:
            # 220도 범위 내 최소 장애물 거리에 따른 순수 연속 함수 속도 제어 (장애물 근접 시 최소 속도를 더욱 낮추어 서행)
            em_dist = float(getattr(self, 'min_wide_dist', 999.0))
            speed_factor = (math.tanh(max(0.0, em_dist) / 100.0)) ** 1.35
            # 라인트레이싱 모드에서는 갭 내비 대비 살짝 느린 속도 (85%)로 주행하여 반응형 회피에 여유 확보
            if getattr(self, 'linetrace_mode', False):
                speed_factor *= 0.85
            target_fwd = ((tL + tR) / 6.0) * speed_factor
            mom = (tR - tL) * self.params['mom_coeff']

        if not hasattr(self, 'current_fwd'):
            self.current_fwd = 0.0

        self.current_fwd = self.current_fwd * 0.90 + target_fwd * 0.10
        ch = math.cos(self.boat_heading)
        sh = math.sin(self.boat_heading)

        acc = self.current_fwd / self.mass
        vel0, vel1 = float(self.boat_vel[0]), float(self.boat_vel[1])
        vel_norm = math.hypot(vel0, vel1)

        # 유체 항력 및 횡방향 슬립 댐핑 고속 연산 (numpy 임시 배열 할당 제거)
        lat_speed = -vel0 * sh + vel1 * ch
        drag0 = -self.drag * vel0 * vel_norm + sh * lat_speed * 18.0
        drag1 = -self.drag * vel1 * vel_norm - ch * lat_speed * 18.0

        prev0, prev1 = float(self.boat_pos[0]), float(self.boat_pos[1])
        self.boat_vel[0] = vel0 + (acc * ch + drag0) * self.dt
        self.boat_vel[1] = vel1 + (acc * sh + drag1) * self.dt
        self.boat_pos[0] = prev0 + self.boat_vel[0] * self.dt
        self.boat_pos[1] = prev1 + self.boat_vel[1] * self.dt

        if getattr(self, 'manual_mode', False):
            self.boat_pos[0] = min(max(25.0, float(self.boat_pos[0])), float(self.map_w - 25.0))
            self.boat_pos[1] = min(max(25.0, float(self.boat_pos[1])), float(self.sim_h - 25.0))

        if self.frame % 7 == 0:
            p0x, p0y = int(prev0), int(prev1)
            p1x, p1y = int(self.boat_pos[0]), int(self.boat_pos[1])
            pygame.draw.line(self.trail, (255, 255, 255, 60), (p0x, p0y), (p1x, p1y), 2)
            min_lx = min(p0x, p1x) - 4
            max_lx = max(p0x, p1x) + 4
            min_ly = min(p0y, p1y) - 4
            max_ly = max(p0y, p1y) + 4
            if min_lx < self.trail_min_x: self.trail_min_x = float(min_lx)
            if max_lx > self.trail_max_x: self.trail_max_x = float(max_lx)
            if min_ly < self.trail_min_y: self.trail_min_y = float(min_ly)
            if max_ly > self.trail_max_y: self.trail_max_y = float(max_ly)

        ang_acc = (mom - self.rot_drag * self.boat_ang_vel) / self.inertia
        self.boat_ang_vel += ang_acc * self.dt
        self.boat_ang_vel *= 0.84

        d_head = self.boat_ang_vel * self.dt
        self.boat_heading += d_head

        # RC 수동 조종 모드 시 누적 회전 각도 및 비단절 충돌 카운트 추적
        if getattr(self, 'manual_mode', False):
            self.manual_cum_turn = getattr(self, 'manual_cum_turn', 0.0) + math.degrees(abs(d_head))
            if getattr(self, 'manual_collision_flash', 0) > 0:
                self.manual_collision_flash -= 1
            if getattr(self, 'manual_collision_cooldown', 0) > 0:
                self.manual_collision_cooldown -= 1
            if self.collide():
                if self.manual_collision_cooldown <= 0:
                    self.manual_collisions = getattr(self, 'manual_collisions', 0) + 1
                    self.manual_collision_cooldown = 45 # 0.75초간 중복 카운트 방지
                    self.manual_collision_flash = 35    # 화면 충돌 알림 플래시 지속 시간
                    self.boat_vel = -self.boat_vel * 0.35 # 부표 충돌 반발 감속

        # 선미 추진 선박의 후방 회전축(L_pivot = 4.0px)에 따른 자연스러운 선회 궤적
        L_pivot = 4.0
        lat_vec = np.array([-math.sin(self.boat_heading), math.cos(self.boat_heading)])
        self.boat_pos += lat_vec * (self.boat_ang_vel * L_pivot * self.dt)

        # 실제 선박 유체역학 파도 생성 (Realistic Hydrodynamic Wave System)
        if vel_norm > 2.0:
            h = self.boat_heading
            intensity = min(1.0, vel_norm / 11.0)
            sh = math.sin(h); ch = math.cos(h)
            GAP = 11; L = 84

            # 선미 듀얼 쓰러스터 추진 제트 기포 및 후방 횡단 웨이크 (Enlarged Stern Roostertail & Trailing Foam)
            if self.frame % 2 == 0:
                stern_lx = self.boat_pos[0] - sh * GAP - ch * (L * 0.50)
                stern_ly = self.boat_pos[1] + ch * GAP - sh * (L * 0.50)
                stern_rx = self.boat_pos[0] + sh * GAP - ch * (L * 0.50)
                stern_ry = self.boat_pos[1] - ch * GAP - sh * (L * 0.50)

                self.wakes.append([stern_lx + random.uniform(-1.5, 1.5), stern_ly + random.uniform(-1.5, 1.5), 3.0, 180 * intensity, -ch * 0.65, -sh * 0.65])
                self.wakes.append([stern_rx + random.uniform(-1.5, 1.5), stern_ry + random.uniform(-1.5, 1.5), 3.0, 180 * intensity, -ch * 0.65, -sh * 0.65])

            if self.frame % 3 == 0:
                cx = self.boat_pos[0] - ch * 42
                cy = self.boat_pos[1] - sh * 42
                self.wakes.append([cx + random.uniform(-2.5, 2.5), cy + random.uniform(-2.5, 2.5), 4.5, 130 * intensity, -ch * 0.85, -sh * 0.85])

            # 좌/우 회전 시 외측 선체 유체 저항에 의한 흰색 거품 (Outer Hull Resistance Foam)
            if abs(self.boat_ang_vel) > 0.06:
                turn_p = min(1.0, abs(self.boat_ang_vel) / 0.42) * intensity
                s = 1.0 if self.boat_ang_vel < 0 else -1.0

                rand_l = random.uniform(-L * 0.25, L * 0.15)
                bx_foam = self.boat_pos[0] + s * (-sh) * (GAP + random.uniform(1.5, 4.0)) + ch * rand_l
                by_foam = self.boat_pos[1] + s * ch * (GAP + random.uniform(1.5, 4.0)) + sh * rand_l

                drift_vx = s * (-sh) * random.uniform(0.3, 0.7) - ch * 0.35
                drift_vy = s * ch * random.uniform(0.3, 0.7) - sh * 0.35
                init_r = random.uniform(2.0, 3.5)
                alpha = random.uniform(140, 200) * turn_p

                # 7번째 원소=1: 순백색 거품 태그 (뷰쪽 파란색 링 없이 흰색만)
                self.wakes.append([bx_foam, by_foam, init_r, alpha, drift_vx, drift_vy, 1])

        # 파도-장애물 물리 상호작용 (Wave Absorption & Frothy Micro-Bubble Scattering) - 렌더링 직전 마지막 서브스텝에서만 연산
        is_last_substep = (sub_step_idx == total_sub_steps - 1) if total_sub_steps > 1 else True
        if is_last_substep and len(self.wakes) > 0 and len(self.dynamic_obstacles) > 0:
            bx, by = self.boat_pos
            dx_b = self.dynamic_obstacles[:, 0] - bx
            dy_b = self.dynamic_obstacles[:, 1] - by
            near_mask = dx_b * dx_b + dy_b * dy_b < 32400.0  # 180.0**2
            if np.any(near_mask):
                near_obs = self.dynamic_obstacles[near_mask]
                near_ox = near_obs[:, 0]
                near_oy = near_obs[:, 1]
                near_or = near_obs[:, 2]

                # 살아있는 웨이크만 필터링하여 일괄 2D 행렬 연산
                active_wake_indices = [idx for idx, w in enumerate(self.wakes) if w[3] > 0]
                if active_wake_indices:
                    w_arr = np.array([[self.wakes[idx][0], self.wakes[idx][1], self.wakes[idx][2], self.wakes[idx][3]] for idx in active_wake_indices], dtype=np.float32)
                    wx = w_arr[:, 0:1]
                    wy = w_arr[:, 1:2]
                    wr = w_arr[:, 2:3]
                    wa = w_arr[:, 3:4]

                    ddx = wx - near_ox[None, :]
                    ddy = wy - near_oy[None, :]
                    dd_sq = ddx * ddx + ddy * ddy

                    # 1. 장애물 내부로 들어간 파도 소멸 (Absorption) - 제곱 거리로 sqrt 연산 제거
                    absorb_thresh = (near_or + 2.0)**2
                    absorb_matrix = dd_sq < absorb_thresh[None, :]
                    absorbed_w_idx = np.any(absorb_matrix, axis=1)
                    if np.any(absorbed_w_idx):
                        for a_idx in np.where(absorbed_w_idx)[0]:
                            self.wakes[active_wake_indices[a_idx]][3] = 0

                    # 2. 장애물 둘레에 닿은 파도 반사 산란 (Scatter)
                    scatter_cand = (wa > 35) & (~absorbed_w_idx[:, None])
                    if np.any(scatter_cand):
                        dd = np.sqrt(dd_sq)
                        target_dist = wr + near_or[None, :]
                        scatter_mask = scatter_cand & (np.abs(dd - target_dist) < 5.0)
                        if np.any(scatter_mask):
                            w_hits, obs_hits = np.where(scatter_mask)
                            new_reflected = []
                            for wi, oi in zip(w_hits, obs_hits):
                                if random.random() < 0.35:
                                    ox_s = float(near_ox[oi])
                                    oy_s = float(near_oy[oi])
                                    orad_s = float(near_or[oi])
                                    base_angle = math.atan2(float(ddy[wi, oi]), float(ddx[wi, oi]))
                                    w_alpha = float(wa[wi, 0])
                                    for _ in range(random.randint(2, 4)):
                                        angle = base_angle + random.uniform(-0.8, 0.8)
                                        spd = random.uniform(0.8, 1.8)
                                        r_off = orad_s + random.uniform(0.8, 2.2)
                                        ca, sa = math.cos(angle), math.sin(angle)
                                        new_reflected.append([
                                            ox_s + ca * r_off, oy_s + sa * r_off,
                                            random.uniform(0.3, 0.65), w_alpha * 0.85,
                                            ca * spd, sa * spd
                                        ])
                            if new_reflected:
                                self.reflected_wakes.extend(new_reflected)

def collide(self):
        bx, by = self.boat_pos

        # 라인트레이싱 모드: 외곽 벽(Boundary Walls)을 장애물로 인식 및 충돌 판정 (목적지 방향 정면 수직벽 xmax 제외)
        if getattr(self, 'linetrace_mode', False):
            hull_margin = 18.0
            if bx <= hull_margin or \
               by <= hull_margin or by >= (self.sim_h - hull_margin) or \
               bx >= self.map_w:
                return True

        # Shared exact geometry; the shadow predictor receives only a nearby subset.
        return hull_collides(
            self.boat_pos, self.boat_heading, self.dynamic_obstacles,
            (self.left_hull_local, self.right_hull_local, self.deck_local),
        )
