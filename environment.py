import os
import json
import time
import pygame
import numpy as np
import math
import random
from config import WIDTH, HEIGHT, SIM_H, DASH_H, MAP_W, GRID, GRID_W, GRID_H, get_dashboard_layout
from utils import wrap
from perception import init_grid
from navigation import reactive_avoidance
from ui_renderer import EnvRenderer

class BoatEnv:
    def __init__(self):
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        self.w = WIDTH
        self.h = HEIGHT
        self.map_w = MAP_W  # 월드 맵 가로 폭 (1840px)
        self.cam_x = 0      # 카메라 X 오프셋 (보트 추종)
        self.sim_h = SIM_H
        self.is_fullscreen_window = False
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("kaboat simulation")
        self.clock = pygame.time.Clock()
        self.dt = 0.04

        # 학습된 최적 파라미터 자동 로드
        self.params = {
            'steer_gain': 1.1,
            'steer_alpha': 0.3515,
            'mom_coeff': 0.00665,
            'pwm_rng': 270.36,
            'avoid_normal': 0.05,
            'avoid_em': 0.7,
            'clear_margin': 10.0,
            'em_enter': 175.0,
            'em_exit': 220.0,
            'em_hold_frames': 18,
            'align_exp': 6.0,
            'heading_exp': 4.0,
            'fwd_exp': 6.0,
            'width_exp': 4.0,
            'clear_exp': 4.0,
            'wp_switch_thresh': 1.1,
            'perp_exp': 2.0
        }
        self.load_params()
         
        self.lidar_beams = 180
        self.lidar_range = 320
        self.rel_angles = np.linspace(-np.pi, np.pi, self.lidar_beams, endpoint=False)
        
        self.mass = 10
        self.inertia = 10
        self.drag = 0.2
        self.rot_drag = 15
        self.boat_radius = 25
        
        # 선체 표면 기하 형상 (ui_renderer의 선체 렌더링과 100% 일치하는 정밀 히트박스)
        GAP = 11.0; L = 84.0; W = 16.0
        hull_local = [
            (L*0.50, 0.0), (L*0.12, W),
            (-L*0.28, W*0.85), (-L*0.48, W*0.6),
            (-L*0.50, 0.0), (-L*0.48, -W*0.6),
            (-L*0.28, -W*0.85), (L*0.12, -W)
        ]
        self.left_hull_local = [(p[0], p[1] + GAP) for p in hull_local]
        self.right_hull_local = [(p[0], p[1] - GAP) for p in hull_local]
        self.deck_local = [
            (L * 0.25, -GAP * 0.85),
            (L * 0.25, GAP * 0.85),
            (-L * 0.35, GAP * 0.85),
            (-L * 0.35, -GAP * 0.85)
        ]
        
        self.trail = pygame.Surface((self.map_w, self.sim_h), pygame.SRCALPHA)
        self.path_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.wake_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.occ_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self.shadow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        
        self.obs_n = int(80 * (self.map_w / self.w))   # 맵 확장에 비례하는 장애물 수 (기본 80개)
        self.obs_r = 17
        self.min_obs = 130
        
        self.grid = init_grid()
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.next_wp = None
        self.visited = set()
        
        self.frame = 0
        self.prev_steer = 0
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        
        self.bezier_path = None
        self.next_bezier_path = None
        self.pursuit_target = None
        self.next_pursuit_target = None
        self.heading_target = 0.0
        self.wakes = [] # [x, y, radius, alpha]
        self.reflected_wakes = [] # [x, y, radius, alpha] (장애물 충돌 반사파)
        
        self.obstacles = np.array([])
        self.dynamic_obstacles = np.array([])
        
        self.show_1st_path = True
        self.show_2nd_path = True
        self.show_paths = True
        self.show_candidates = True
        self.show_lidar = True
        self.show_lidar_range = True
        self.candidate_wps = []
        self.total_gaps_count = 0
        self.show_all_gaps = False
        self.all_gaps = []
        self.gaps_btn_rect = None
        
        self.linetrace_mode = False
        self.linetrace_queued = False
        self.needs_break = False
        self.mode_btn_top_rect = pygame.Rect(25, 16, 165, 28)
        self.show_closest_obstacle = True
        self.closest_avoid_hit = None
        
        # 체크박스 및 버튼 좌표 (self.sim_h 기준 동적 오프셋 계산)
        base_y = self.sim_h + 33
        self.cb1_rect = pygame.Rect(40, base_y + 5, 20, 20)
        self.cb2_rect = pygame.Rect(40, base_y + 41, 20, 20)
        self.cb3_rect = pygame.Rect(40, base_y + 77, 20, 20)
        self.cb4_rect = pygame.Rect(40, base_y + 113, 20, 20)
        self.cb5_rect = pygame.Rect(40, base_y + 149, 20, 20)
        
        # 체크박스 및 텍스트 라벨 전체 클릭 영역 (가로 275px)
        self.cb1_row_rect = pygame.Rect(35, base_y, 275, 30)
        self.cb2_row_rect = pygame.Rect(35, base_y + 36, 275, 30)
        self.cb3_row_rect = pygame.Rect(35, base_y + 72, 275, 30)
        self.cb4_row_rect = pygame.Rect(35, base_y + 108, 275, 30)
        self.cb5_row_rect = pygame.Rect(35, base_y + 144, 275, 30)
        
        self.paused = False
        btn_y = base_y + 185
        self.pause_btn = pygame.Rect(38, btn_y, 52, 34)
        self.sim_speed = 1
        self.speed_btns = {
            1: pygame.Rect(96, btn_y, 38, 34),
            2: pygame.Rect(140, btn_y, 38, 34),
            4: pygame.Rect(184, btn_y, 38, 34),
            8: pygame.Rect(228, btn_y, 38, 34),
            16: pygame.Rect(272, btn_y, 46, 34)
        }
        
        # 실시간 3D 그래픽스 엔진 상태 변수
        self.cam_3d_mode = 1  # 0: 1인칭 조타석, 1: 3인칭 추종 체이스, 2: 전술 드론
        self.fullscreen_3d = False
        self.layout = get_dashboard_layout(self.w, self.sim_h)
        self.panel_3d_rect = pygame.Rect(self.layout['p3_x'], self.sim_h + 35, 320, 220)
        
        # RC 조종기 모드 및 블라인드 시연 모드
        self.manual_mode = False
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self.saved_manual_state = None
        self.rc_btn_rect = None
        self.blind_mode = False
        self.blind_btn_rect = None
        
        # RC 주행 메트릭스 및 랭킹 시스템
        self.manual_start_time = time.time()
        self.manual_collisions = 0
        self.manual_cum_turn = 0.0
        self.manual_collision_cooldown = 0
        self.manual_collision_flash = 0
        self.show_leaderboard = False
        self.last_manual_result = None
        self.leaderboard_retry_rect = None
        self.leaderboard_exit_rect = None
        
        self.renderer = EnvRenderer(self)
        self.reset()

    def load_params(self):
        json_path = "best_learned_params.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    self.params.update(json.load(f))
            except Exception:
                pass

    def reset(self):
        self.load_params()
        self.frame = 0
        self.boat_pos = np.array([65, self.sim_h/2], dtype=np.float32)
        self.boat_vel = np.zeros(2)
        self.boat_ang_vel = 0
        self.target = np.array([self.map_w - 100, self.sim_h/2], dtype=np.float32)
        self.cam_x = 0
        
        self.trail.fill((0, 0, 0, 0))
        self.path_surf.fill((0, 0, 0, 0))
        self.wake_surf.fill((0, 0, 0, 0))
        
        obs = []
        t = 0
        while len(obs) < self.obs_n and t < 5000:
            t += 1
            x = random.randint(300, self.map_w - 300)
            y = random.randint(30, self.sim_h - 30)
            p = np.array([x, y])
            if np.linalg.norm(p - self.target) < 180: continue
            if np.linalg.norm(p - self.boat_pos) < 180: continue
            
            ok = True
            for (ox, oy, r) in obs:
                if np.linalg.norm(p - np.array([ox, oy])) < self.min_obs:
                    ok = False
                    break
            if ok:
                obs.append((x, y, self.obs_r))
                
        self.obstacles = np.array(obs, dtype=np.float32)
        self.dynamic_obstacles = self.obstacles.copy()
        
        dx = self.target[0] - self.boat_pos[0]
        dy = self.target[1] - self.boat_pos[1]
        self.boat_heading = math.atan2(dy, dx)
        self.heading_target = self.boat_heading
        
        self.grid[:] = 0
        self.clusters = []
        self.cluster_ids = []
        self.current_wp = None
        self.next_wp = None
        self.visited = set()
        self.total_gaps_count = 0
        self.all_gaps = []
        
        self.wp_check_timer = 0
        self.steer_timer = 0
        self.path_timer = 0
        self.bezier_path = None
        self.next_bezier_path = None
        self.pursuit_target = None
        self.next_pursuit_target = None
        self.wakes = []
        self.emergency_mode = False
        if getattr(self, 'linetrace_queued', False):
            self.linetrace_mode = True
            self.linetrace_queued = False

        # RC 주행 메트릭스 리셋
        self.manual_start_time = time.time()
        self.manual_collisions = 0
        self.manual_cum_turn = 0.0
        self.manual_collision_cooldown = 0
        self.manual_collision_flash = 0
        self.show_leaderboard = False
        self.last_manual_result = None

    def handle_click(self, pos):
        # 0-0. 랭킹 모달 창이 열려 있을 때의 클릭 이벤트 처리
        if getattr(self, 'show_leaderboard', False):
            if getattr(self, 'leaderboard_retry_rect', None) and self.leaderboard_retry_rect.collidepoint(pos):
                self.reset_manual_episode()
                return
            if getattr(self, 'leaderboard_exit_rect', None) and self.leaderboard_exit_rect.collidepoint(pos):
                if getattr(self, 'leaderboard_view_only', False):
                    self.show_leaderboard = False
                    self.leaderboard_view_only = False
                else:
                    self.toggle_manual_mode()
                return
            return

        # 0. RC 조종기 모드 토글 버튼 클릭 (우측 상단 텔레메트리 HUD 하단)
        if getattr(self, 'rc_btn_rect', None) and self.rc_btn_rect.collidepoint(pos):
            self.toggle_manual_mode()
            return

        # 0-1. 눈 깜빡임(블라인드 시연 모드) 토글 버튼 클릭 (RC 모드 상태에서만 활성화)
        if getattr(self, 'manual_mode', False) and getattr(self, 'blind_btn_rect', None) and self.blind_btn_rect.collidepoint(pos):
            self.toggle_blind_mode()
            return

        # 0-2. 새 에피소드 재시작 버튼 클릭 (RC 모드 상태에서만 활성화, 눈 깜빡임 버튼 좌측)
        if getattr(self, 'manual_mode', False) and getattr(self, 'restart_btn_rect', None) and self.restart_btn_rect.collidepoint(pos):
            self.reset_manual_episode()
            return

        # 0-3. 랭킹 대시보드 열람 버튼 클릭 (RC 모드 상태에서만 활성화, 재시작 버튼 좌측)
        if getattr(self, 'manual_mode', False) and getattr(self, 'leaderboard_btn_rect', None) and self.leaderboard_btn_rect.collidepoint(pos):
            self.show_leaderboard = True
            self.leaderboard_view_only = True
            return

        # 1. 3D 전체화면/2D 화면 교체 버튼 클릭 (메인 화면 좌측 하단)
        view_rect = getattr(self, 'view_btn_rect', getattr(self, 'view_btn_top_rect', None))
        if view_rect and view_rect.collidepoint(pos):
            self.fullscreen_3d = not getattr(self, 'fullscreen_3d', False)
            return

        # 2. 카메라 모드 변경 버튼 클릭 (메인 화면 좌측 하단 버튼 또는 하단 패널 내부 버튼)
        cam_rect = getattr(self, 'cam_btn_rect', getattr(self, 'cam_btn_top_rect', None))
        if cam_rect and cam_rect.collidepoint(pos):
            self.cam_3d_mode = (getattr(self, 'cam_3d_mode', 1) + 1) % 3
            return

        if getattr(self, 'cam_panel_btn_rect', None) and self.cam_panel_btn_rect.collidepoint(pos):
            self.cam_3d_mode = (getattr(self, 'cam_3d_mode', 1) + 1) % 3
            return

        # 3. 하단 제3패널(3D 또는 2D 패널) 영역 클릭 시
        if getattr(self, 'panel_3d_rect', None) and self.panel_3d_rect.collidepoint(pos):
            if getattr(self, 'fullscreen_3d', False):
                # 풀화면 3D 상태에서 하단 2D 패널을 클릭하면 상/하 화면 스왑 복귀
                self.fullscreen_3d = False
            else:
                self.cam_3d_mode = (getattr(self, 'cam_3d_mode', 1) + 1) % 3
            return

        is_mode_click = getattr(self, 'mode_btn_top_rect', None) and self.mode_btn_top_rect.collidepoint(pos)
        if is_mode_click:
            # 버튼 클릭 시 현재 실행 중인 에피소드에서 실시간으로 알고리즘 즉시 변경 (GAP NAVIGATION <-> LINE TRACING)
            self.linetrace_mode = not getattr(self, 'linetrace_mode', False)
            self.linetrace_queued = False
        elif getattr(self, 'linetrace_mode', False):
            # 라인트레이싱 모드: 3개 버튼 (1. 가장 가까운 장애물 SHOW / 2. 라이다 히트 / 3. 라이다 레인지)
            if getattr(self, 'cb1_row_rect', self.cb1_rect).collidepoint(pos) or self.cb1_rect.collidepoint(pos):
                self.show_closest_obstacle = not getattr(self, 'show_closest_obstacle', True)
            elif getattr(self, 'cb2_row_rect', self.cb2_rect).collidepoint(pos) or self.cb2_rect.collidepoint(pos):
                self.show_lidar = not self.show_lidar
            elif getattr(self, 'cb3_row_rect', self.cb3_rect).collidepoint(pos) or self.cb3_rect.collidepoint(pos):
                self.show_lidar_range = not self.show_lidar_range
            elif self.pause_btn.collidepoint(pos):
                self.paused = not self.paused
            else:
                for spd, rect in self.speed_btns.items():
                    if rect.collidepoint(pos):
                        self.sim_speed = spd
                        self.paused = False
                        break
        else:
            # 갭 항법 모드: 5개 체크박스
            if getattr(self, 'cb1_row_rect', self.cb1_rect).collidepoint(pos) or self.cb1_rect.collidepoint(pos):
                self.show_1st_path = not getattr(self, 'show_1st_path', True)
                self.show_paths = self.show_1st_path or getattr(self, 'show_2nd_path', True)
            elif getattr(self, 'cb2_row_rect', self.cb2_rect).collidepoint(pos) or self.cb2_rect.collidepoint(pos):
                self.show_2nd_path = not getattr(self, 'show_2nd_path', True)
                self.show_paths = getattr(self, 'show_1st_path', True) or self.show_2nd_path
            elif getattr(self, 'cb3_row_rect', self.cb3_rect).collidepoint(pos) or self.cb3_rect.collidepoint(pos):
                self.show_candidates = not self.show_candidates
            elif getattr(self, 'cb4_row_rect', self.cb4_rect).collidepoint(pos) or self.cb4_rect.collidepoint(pos):
                self.show_lidar = not self.show_lidar
            elif getattr(self, 'cb5_row_rect', self.cb5_rect).collidepoint(pos) or self.cb5_rect.collidepoint(pos):
                self.show_lidar_range = not self.show_lidar_range
            elif self.pause_btn.collidepoint(pos):
                self.paused = not self.paused
            elif getattr(self, 'gaps_btn_rect', None) and self.gaps_btn_rect.collidepoint(pos):
                self.show_all_gaps = not getattr(self, 'show_all_gaps', False)
            else:
                for spd, rect in self.speed_btns.items():
                    if rect.collidepoint(pos):
                        self.sim_speed = spd
                        self.paused = False
                        break

    def toggle_fullscreen(self):
        self.is_fullscreen_window = not getattr(self, 'is_fullscreen_window', False)
        flags = pygame.FULLSCREEN if self.is_fullscreen_window else 0
        self.screen = pygame.display.set_mode((self.w, self.h), flags)

    def toggle_blind_mode(self):
        """블라인드 모드 토글: 라이다 탐지 반경(6.4m / 320px) 이외 시야 암전 처리 및 원 테두리 목표점 위치 표출"""
        self.blind_mode = not getattr(self, 'blind_mode', False)

    def toggle_manual_mode(self):
        """RC 조종기 모드 토글: 3D 전체화면 즉시 전환 및 WASD 수동 조종 활성화, 복귀 시 이전 세팅 복원 및 새 에피소드 시작"""
        if not getattr(self, 'manual_mode', False):
            # 1. 수동 조종 모드 진입: 현재 세팅 저장 후 3D 전체화면 전환 및 새 에피소드 시작
            self.saved_manual_state = {
                'fullscreen_3d': getattr(self, 'fullscreen_3d', False),
                'cam_3d_mode': getattr(self, 'cam_3d_mode', 1),
                'sim_speed': getattr(self, 'sim_speed', 1),
                'show_paths': getattr(self, 'show_paths', True),
                'show_1st_path': getattr(self, 'show_1st_path', True),
                'show_2nd_path': getattr(self, 'show_2nd_path', True),
                'show_candidates': getattr(self, 'show_candidates', True),
                'show_lidar': getattr(self, 'show_lidar', False),
                'show_lidar_range': getattr(self, 'show_lidar_range', True),
                'show_all_gaps': getattr(self, 'show_all_gaps', False),
                'linetrace_mode': getattr(self, 'linetrace_mode', False),
            }
            self.manual_mode = True
            self.blind_mode = False  # 진입 시 기본 블라인드 OFF
            self.fullscreen_3d = True  # 즉시 3D View 전체화면 전환 (2D는 하단 패널로 자동 스왑)
            self.sim_speed = 1
            self.manual_throttle = 0.0
            self.manual_steer = 0.0
            self.reset()  # 사용자 요청: RC 모드 진입할 때도 새 에피소드 생성
        else:
            # 2. 수동 조종 모드 종료: 조종 모드 이전 세팅 완벽 복원 후 새 에피소드 리셋
            self.manual_mode = False
            self.blind_mode = False
            self.blind_btn_rect = None
            self.restart_btn_rect = None
            self.show_leaderboard = False
            self.last_manual_result = None
            saved = getattr(self, 'saved_manual_state', None)
            if saved:
                self.fullscreen_3d = saved.get('fullscreen_3d', False)
                self.cam_3d_mode = saved.get('cam_3d_mode', 1)
                self.sim_speed = saved.get('sim_speed', 1)
                self.show_paths = saved.get('show_paths', True)
                self.show_1st_path = saved.get('show_1st_path', True)
                self.show_2nd_path = saved.get('show_2nd_path', True)
                self.show_candidates = saved.get('show_candidates', True)
                self.show_lidar = saved.get('show_lidar', False)
                self.show_lidar_range = saved.get('show_lidar_range', True)
                self.show_all_gaps = saved.get('show_all_gaps', False)
            self.reset()

    def reset_manual_episode(self):
        """RC 수동 조종 모드 상태를 유지하면서 새 에피소드로 리셋"""
        self.manual_mode = True
        self.show_leaderboard = False
        self.leaderboard_view_only = False
        self.last_manual_result = None
        self.manual_start_time = time.time()
        self.manual_collisions = 0
        self.manual_cum_turn = 0.0
        self.manual_collision_cooldown = 0
        self.manual_collision_flash = 0
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self.reset()

    def update_dynamic_obstacles(self):
        ox = self.obstacles[:, 0]
        oy = self.obstacles[:, 1]
        r = self.obstacles[:, 2]
        phase = self.frame * 0.04 + ox * 0.05 + oy * 0.05
        self.dynamic_obstacles[:, 0] = ox + np.sin(phase) * (r * 0.2)
        self.dynamic_obstacles[:, 1] = oy + np.cos(phase * 1.2) * (r * 0.2)
        
        # 부표 중앙을 기준으로 부드러운 백색 원형 구름 파도가 주기적으로 퍼져나감
        if self.frame % 36 == 0:
            for i in range(len(self.obstacles)):
                self.reflected_wakes.append([
                    self.dynamic_obstacles[i, 0], self.dynamic_obstacles[i, 1], r[i] + 1.0, 72
                ])

    def pwm_to_thrust(self, p):
        return p * 10

    def step(self, L, R):
        tL = self.pwm_to_thrust(L)
        tR = self.pwm_to_thrust(R)

        if getattr(self, 'manual_mode', False):
            # 수동 조종 모드: W/S 키 입력에 따른 직접 추진력 제어
            m_thr = getattr(self, 'manual_throttle', 0.0)
            target_fwd = m_thr * 5500.0
            mom = (tR - tL) * self.params['mom_coeff']
        else:
            # 220도 범위 내 최소 장애물 거리에 따른 순수 연속 함수 속도 제어
            em_dist = float(getattr(self, 'min_wide_dist', 999.0))
            dist_speed_factor = (math.tanh(em_dist / 50.0)) ** 1.35
            # 전방 85px 이내 초근접 시 선속 추가 안전 제한 (관성 슬립 충돌 차단)
            if em_dist < 85.0:
                dist_speed_factor = min(dist_speed_factor, 0.15 + 0.15 * (em_dist / 85.0))
            
            # 회전해야 하는 각도(헤딩 오차 및 조향 명령 강도)가 클수록 속도를 대폭 감속 (회전 관성 16, 질량 20 대응)
            turn_err = abs(wrap(self.heading_target - self.boat_heading))
            steer_angle_equiv = abs(getattr(self, 'prev_steer', 0.0)) * (math.pi * 0.5)
            effective_turn_angle = max(turn_err, steer_angle_equiv)
            
            # 각도가 0도일 때 1.0, 45도일 때 ~0.46, 75도 이상일 때 ~0.10으로 급격히 감속하여 제자리 선회력 확보
            turn_cos = max(0.0, math.cos(min(math.pi * 0.5, effective_turn_angle)))
            turn_speed_factor = max(0.10, turn_cos ** 1.2)
            
            speed_factor = dist_speed_factor * turn_speed_factor
            
            # 라인트레이싱 모드에서는 갭 내비 대비 살짝 느린 속도 (85%)로 주행하여 반응형 회피에 여유 확보
            if getattr(self, 'linetrace_mode', False):
                speed_factor *= 0.85
            target_fwd = ((tL + tR) / 6.0) * speed_factor
            mom = (tR - tL) * self.params['mom_coeff']
            
        if not hasattr(self, 'current_fwd'):
            self.current_fwd = 0.0
            
        self.current_fwd = self.current_fwd * 0.90 + target_fwd * 0.10
        hv = np.array([math.cos(self.boat_heading), math.sin(self.boat_heading)])
        
        acc = self.current_fwd / self.mass
        vel_norm = math.hypot(self.boat_vel[0], self.boat_vel[1])
        
        # 유체 항력
        drag = -self.drag * self.boat_vel * vel_norm
        
        # 횡방향 슬립 댐핑
        lat_v = np.array([-math.sin(self.boat_heading), math.cos(self.boat_heading)])
        lat_speed = np.dot(self.boat_vel, lat_v)
        drag += -lat_v * lat_speed * 18.0
            
        prev = self.boat_pos.copy()
        self.boat_vel += (acc * hv + drag) * self.dt
        self.boat_pos += self.boat_vel * self.dt
        
        if getattr(self, 'manual_mode', False):
            self.boat_pos[0] = np.clip(self.boat_pos[0], 25, self.map_w - 25)
            self.boat_pos[1] = np.clip(self.boat_pos[1], 25, self.sim_h - 25)
        
        if self.frame % 7 == 0:
            pygame.draw.line(self.trail, (255, 255, 255, 60),
                             (int(prev[0]), int(prev[1])),
                             (int(self.boat_pos[0]), int(self.boat_pos[1])), 2)
                             
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

        # 파도-장애물 물리 상호작용 (Wave Absorption & Frothy Micro-Bubble Scattering)
        if len(self.wakes) > 0 and len(self.dynamic_obstacles) > 0:
            bx, by = self.boat_pos
            dx_b = self.dynamic_obstacles[:, 0] - bx
            dy_b = self.dynamic_obstacles[:, 1] - by
            near_mask = dx_b * dx_b + dy_b * dy_b < 32400.0  # 180.0**2
            if np.any(near_mask):
                near_obs = self.dynamic_obstacles[near_mask]
                near_list = [(float(row[0]), float(row[1]), float(row[2])) for row in near_obs]
                for w in self.wakes:
                    if w[3] <= 0:
                        continue
                    wx, wy = w[0], w[1]
                    absorbed = False
                    for ox, oy, orad in near_list:
                        dx = wx - ox
                        dy = wy - oy
                        d = math.hypot(dx, dy)
                        
                        # 1. 장애물 내부로 들어간 파도는 완전히 소멸/흡수 (Absorption)
                        if d < orad + 2.0:
                            w[3] = 0
                            absorbed = True
                            break
                        
                        # 2. 장애물 둘레에 파도가 닿으면 나노 거품 반사 산란
                        if w[3] > 35 and abs(d - (w[2] + orad)) < 5.0:
                            if random.random() < 0.35:
                                for _ in range(random.randint(2, 4)):
                                    angle = math.atan2(dy, dx) + random.uniform(-0.8, 0.8)
                                    spd = random.uniform(0.8, 1.8)
                                    fx = ox + math.cos(angle) * (orad + random.uniform(0.8, 2.2))
                                    fy = oy + math.sin(angle) * (orad + random.uniform(0.8, 2.2))
                                    self.reflected_wakes.append([
                                        fx, fy, random.uniform(0.3, 0.65), w[3] * 0.85,
                                        math.cos(angle) * spd, math.sin(angle) * spd
                                    ])
                    if absorbed:
                        continue

    def collide(self):
        bx, by = self.boat_pos
        ch = math.cos(self.boat_heading)
        sh = math.sin(self.boat_heading)

        # 라인트레이싱 모드: 외곽 벽(Boundary Walls)을 장애물로 인식 및 충돌 판정 (목적지 방향 정면 수직벽 xmax 제외)
        if getattr(self, 'linetrace_mode', False):
            hull_margin = 18.0
            if bx <= hull_margin or \
               by <= hull_margin or by >= (self.sim_h - hull_margin) or \
               bx >= self.map_w:
                return True

        # 장애물 충돌: 선체 로컬 좌표계로 변환하여 3개 선체 폴리곤(좌/우 선체, 데크)과 원형 장애물 정밀 표면 충돌 검사
        if len(self.dynamic_obstacles) == 0:
            return False
            
        ox = self.dynamic_obstacles[:, 0]
        oy = self.dynamic_obstacles[:, 1]
        orr = self.dynamic_obstacles[:, 2]
        
        dx = ox - bx
        dy = oy - by
        
        x_loc = dx * ch + dy * sh
        y_loc = -dx * sh + dy * ch
        
        # 바운딩 박스 1차 고속 필터링 (선체 길이 L/2=42px, 선폭 W_tot/2=27px)
        cand_mask = (abs(x_loc) <= 42.0 + orr) & (abs(y_loc) <= 27.0 + orr)
        if not np.any(cand_mask):
            return False
            
        cand_indices = np.where(cand_mask)[0]
        polys = (self.left_hull_local, self.right_hull_local, self.deck_local)
        
        for idx in cand_indices:
            px = x_loc[idx]
            py = y_loc[idx]
            r = orr[idx]
            r2 = r * r
            
            for poly in polys:
                # 2-1. 장애물 중심이 선체 폴리곤 내부인지 검사 (Ray-casting)
                inside = False
                n = len(poly)
                for i in range(n):
                    j = (i - 1) % n
                    xi, yi = poly[i]
                    xj, yj = poly[j]
                    if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                        inside = not inside
                if inside:
                    return True
                    
                # 2-2. 장애물 중심과 선체 각 모서리 선분 사이의 최단 거리 검사
                for i in range(n):
                    x1, y1 = poly[i]
                    x2, y2 = poly[(i + 1) % n]
                    vx = x2 - x1
                    vy = y2 - y1
                    seg_len_sq = vx * vx + vy * vy
                    if seg_len_sq < 1e-8:
                        dist_sq = (px - x1)**2 + (py - y1)**2
                    else:
                        t = max(0.0, min(1.0, ((px - x1) * vx + (py - y1) * vy) / seg_len_sq))
                        cx = x1 + t * vx
                        cy = y1 + t * vy
                        dist_sq = (px - cx)**2 + (py - cy)**2
                    if dist_sq <= r2:
                        return True
                        
        return False

    def get_pwm(self, steer):
        dead = 0.02
        if abs(steer) < dead: steer = 0
        mid = 1500; rng = self.params['pwm_rng']
        m = (abs(steer) ** 1.15)
        d = m * rng
        if steer >= 0: L = mid - d; R = mid + d
        else: L = mid + d; R = mid - d
        return int(np.clip(L, 1230, 1770)), int(np.clip(R, 1230, 1770))

    def validate_wp_grid(self):
        if self.current_wp is None: return
        self.wp_check_timer += self.dt
        if self.wp_check_timer < 0.05: return
        self.wp_check_timer = 0
        wp = self.current_wp["pos"]; pair = self.current_wp["pair"]
        gx = int(wp[0] // GRID); gy = int(wp[1] // GRID); rad = int(35 // GRID)
        for yy in range(max(0, gy - rad), min(GRID_H, gy + rad + 1)):
            for xx in range(max(0, gx - rad), min(GRID_W, gx + rad + 1)):
                if self.grid[yy, xx] >= 3:
                    self.visited.add(pair); self.visited.add((pair[1], pair[0]))
                    self.current_wp = None; return

    def validate_wp_obstacle_5x5(self):
        if self.current_wp is None: return
        wp = self.current_wp["pos"]
        dx = self.dynamic_obstacles[:, 0] - wp[0]
        dy = self.dynamic_obstacles[:, 1] - wp[1]
        dist_sq = dx * dx + dy * dy
        r_thresh = self.dynamic_obstacles[:, 2] + 2.5 * GRID
        if np.any(dist_sq <= r_thresh * r_thresh):
            p = self.current_wp["pair"]
            self.visited.add(p); self.visited.add((p[1], p[0]))
            self.current_wp = None

    def update_steering(self, dists):
        self.steer_timer += self.dt
        center_idx = self.lidar_beams // 2
        # 정면 + 양옆 20도 = 총 220도 범위 감시
        span = int(self.lidar_beams * 220 / 360 / 2)
        front_dists = dists[center_idx - span : center_idx + span]
        min_front_dist = np.min(front_dists)
        self.min_wide_dist = min_front_dist
        
        if not hasattr(self, 'emergency_cooldown'):
            self.emergency_cooldown = 0
            
        if min_front_dist < self.params['em_enter']:
            self.emergency_mode = True
            self.emergency_cooldown = self.params['em_hold_frames']
        elif self.emergency_mode:
            self.emergency_cooldown -= 1
            if min_front_dist > self.params['em_exit'] and self.emergency_cooldown <= 0:
                self.emergency_mode = False

        if self.pursuit_target is None:
            if self.current_wp is not None:
                self.heading_target = math.atan2(self.current_wp["pos"][1] - self.boat_pos[1], self.current_wp["pos"][0] - self.boat_pos[0])
            else:
                self.heading_target = math.atan2(self.target[1] - self.boat_pos[1], self.target[0] - self.boat_pos[0])
            return 0
        px, py = self.pursuit_target
        heading_target = math.atan2(py - self.boat_pos[1], px - self.boat_pos[0])
        self.heading_target = heading_target
        heading_error = wrap(heading_target - self.boat_heading)

        # 거리에 따라 연속적으로 조향 및 회피력 스케일링
        clear_ratio = np.clip((min_front_dist - 170.0) / 60.0, 0.0, 1)
        steer_gain = self.params['steer_gain'] + (1.0 - clear_ratio) * 0.45
        avoid_multiplier = self.params['avoid_normal'] + (1.0 - clear_ratio) * (self.params['avoid_em'] * 0.45)
            
        # 각속도 댐핑을 강화하여 관성 오버슈트 및 휙휙 도는 회전 억제 (관성 16 대응)
        d_term = -0.22 * getattr(self, 'boat_ang_vel', 0.0)
        steer_raw = heading_error * steer_gain + d_term
        alpha = self.params['steer_alpha']
        steer_f = alpha * steer_raw + (1.0 - alpha) * self.prev_steer
        self.prev_steer = steer_f
        
        # [갭 내비게이션 다이렉트 모드 전용 회피]
        # 질량 20 / 관성 16에 맞춰 회피 개시 거리를 175px로 대폭 확장
        if self.current_wp is None:
            fov_rad = 1.134464  # np.deg2rad(65)
            fwd_mask = np.abs(self.rel_angles) <= fov_rad
            fwd_indices = np.where(fwd_mask)[0]

            SAFE_DIST = 175.0        # 회피 개시 거리 (원거리 조기 회피)
            CRIT_DIST = 85.0        # 근접 긴급 회피 기준 거리 (선체 반경 25px + 장애물 반경 17px = 42px 충돌선 대비 여유 확보)

            if len(fwd_indices) > 0:
                fwd_dists = dists[fwd_indices]
                min_i = int(np.argmin(fwd_dists))
                closest_idx = fwd_indices[min_i]
                min_dist = float(dists[closest_idx])
                closest_ang = float(self.rel_angles[closest_idx])
            else:
                min_dist = 999.0
                closest_ang = 0.0

            if min_dist < SAFE_DIST:
                # UI 렌더링용 최근접 회피 히트점
                bx, by = self.boat_pos
                self.closest_avoid_hit = (
                    float(bx + math.cos(self.boat_heading + closest_ang) * min_dist),
                    float(by + math.sin(self.boat_heading + closest_ang) * min_dist)
                )

                left_mask = (self.rel_angles < -0.05) & fwd_mask
                right_mask = (self.rel_angles > 0.05) & fwd_mask
                d_left = float(np.min(dists[left_mask])) if np.any(left_mask) else 999.0
                d_right = float(np.min(dists[right_mask])) if np.any(right_mask) else 999.0

                # 1. 좁은 갭 사이 중앙 통과 시 좌우 대칭 밸런싱으로 떨림 방지
                push_r = max(0.0, (SAFE_DIST - d_left) / SAFE_DIST) ** 1.5   # 좌측 장애물 -> 우측 반발
                push_l = max(0.0, (SAFE_DIST - d_right) / SAFE_DIST) ** 1.5  # 우측 장애물 -> 좌측 반발
                net_dir = push_r - push_l

                # 2. 근접 위험도(Urgency) 계산
                urgency = float(np.clip((SAFE_DIST - min_dist) / (SAFE_DIST - CRIT_DIST), 0.0, 1.0))
                front_f = max(0.0, math.cos(closest_ang * (np.pi / 2.0 / fov_rad)))

                if min_dist < CRIT_DIST:
                    # [근접 위험 구간] 기존의 강력한 회피력 완전 유지 및 회피 가중치 상향
                    avoid_dir = -float(np.sign(closest_ang)) if abs(closest_ang) > 0.04 else (-1.0 if d_left >= d_right else 1.0)
                    avoid_steer = avoid_dir * (0.85 + 0.15 * urgency)
                    if min_dist < CRIT_DIST - 10.0:  # 75px 이하 극근접 충돌 위험 시 100% 완전 회피
                        steer_cmd = avoid_dir * 1.0
                    else:
                        avoid_weight = max(0.65, urgency * front_f)
                        steer_cmd = (1.0 - avoid_weight) * steer_f + avoid_weight * avoid_steer
                else:
                    # [중거리(85px ~ 175px) 접근 구간] 양측 밸런싱을 적용하여 크게 돌지 않고 틈새 중앙으로 안정적 진입
                    avoid_steer = np.clip(net_dir * 0.40, -0.50, 0.50)
                    steer_cmd = steer_f + avoid_steer

                # 측면 근접 보호(Flank Guard): 배 옆(65~95도) 52px 이내 장애물 근접 시 측면 찰과 충돌 강력 방지
                flank_mask = (np.abs(self.rel_angles) > fov_rad) & (np.abs(self.rel_angles) <= 1.658)
                if np.any(flank_mask):
                    f_dists = dists[flank_mask]
                    f_min = float(np.min(f_dists))
                    if f_min < 52.0:
                        f_idx = np.where(flank_mask)[0][np.argmin(f_dists)]
                        f_ang = float(self.rel_angles[f_idx])
                        f_push = -float(np.sign(f_ang)) * (52.0 - f_min) / 52.0 * 0.45
                        steer_cmd = float(np.clip(steer_cmd + f_push, -1.0, 1.0))

                return float(np.clip(steer_cmd, -1.0, 1.0))
            else:
                self.closest_avoid_hit = None

        avoid = reactive_avoidance(dists, self.rel_angles)

        # 반발력과 조향이 반대로 충돌할 때 조향력 상쇄(직진 현상)를 방지하기 위해 반발력 소프트 감쇠(0.25) 적용
        if (steer_f * avoid < 0) and abs(steer_f) > 0.15:
            avoid *= 0.25

        # 후방 반원(|rel_angle| >= 90도) 내 선체 360도 회전 히트박스 반경(약 45.3px) 이내 장애물 감지 시 회전 억제 (조향 0)
        rear_mask = np.abs(self.rel_angles) >= (np.pi / 2.0 - 1e-5)
        if np.any(rear_mask) and np.min(dists[rear_mask]) <= 45.3:
            return 0.0

        final_steer = np.clip(steer_f + avoid_multiplier * avoid, -1, 1)

        # 측면 근접 보호(Flank Guard): 웨이포인트 주행 중에도 배 옆(65~95도) 52px 이내 장애물 근접 시 측면 찰과 충돌 강력 방지
        flank_mask = (np.abs(self.rel_angles) > 1.134464) & (np.abs(self.rel_angles) <= 1.658)
        if np.any(flank_mask):
            f_dists = dists[flank_mask]
            f_min = float(np.min(f_dists))
            if f_min < 52.0:
                f_idx = np.where(flank_mask)[0][np.argmin(f_dists)]
                f_ang = float(self.rel_angles[f_idx])
                f_push = -float(np.sign(f_ang)) * (52.0 - f_min) / 52.0 * 0.45
                final_steer = float(np.clip(final_steer + f_push, -1.0, 1.0))

        return final_steer

    def update_camera(self):
        """카메라 X 오프셋을 보트 위치에 맞춰 부드럽게 추종 (맵 경계 클램핑)"""
        target_cam_x = self.boat_pos[0] - self.w / 2
        target_cam_x = max(0, min(self.map_w - self.w, target_cam_x))
        # 부드러운 카메라 추종 (lerp)
        self.cam_x = self.cam_x * 0.85 + target_cam_x * 0.15

    def render(self, hits):
        self.renderer.render(hits)