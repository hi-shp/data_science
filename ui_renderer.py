import pygame
import numpy as np
import math
import time
import os
import leaderboard
from engine_3d import Engine3D
from config import get_dashboard_layout

class EnvRenderer:
    def __init__(self, env):
        self.env = env
        self.pov_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
        self.cam_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
        self.real_cam_surf = pygame.Surface((320, 220), pygame.SRCALPHA)
        try:
            self.engine_3d = Engine3D(320, 220)
        except Exception as e:
            print(f"[Warning] ModernGL Engine3D init failed: {e}")
            self.engine_3d = None
        self.safety_surf = pygame.Surface((120, 120), pygame.SRCALPHA)
        self.hud_surf = pygame.Surface((210, 110), pygame.SRCALPHA)
        self.bezier_surf = pygame.Surface((190, 220), pygame.SRCALPHA)
        self.weights_surf = pygame.Surface((190, 220), pygame.SRCALPHA)
        self._cand_surf = pygame.Surface((env.w, env.h), pygame.SRCALPHA)
        self.shadow_surf = pygame.Surface((180, 180), pygame.SRCALPHA)
        self.world_2d_surf = pygame.Surface((env.w, env.sim_h))
        self.font = pygame.font.SysFont(None, 24)
        self.bold_font = pygame.font.SysFont(None, 26, bold=True)
        self.small_font = pygame.font.SysFont(None, 18)
        self.micro_font = pygame.font.SysFont(None, 15)
        self.fps_font = pygame.font.SysFont("sans-serif", 18, bold=False)  # 슬림하면서도 가독성을 확보한 게임 오버레이 18px 폰트
        self.engine_info_font = pygame.font.SysFont("sans-serif", 14)     # 3D 엔진(ModernGL) 텍스트와 100% 동일한 14px 폰트

        # 한글 폰트 로드 (NotoSansCJK 시스템 폰트 연동, 부재 시 SysFont 대체)
        korean_regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        korean_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
        if os.path.exists(korean_regular):
            self.ko_title_font = pygame.font.Font(korean_bold, 22)
            self.ko_header_font = pygame.font.Font(korean_bold, 15)
            self.ko_font = pygame.font.Font(korean_regular, 14)
            self.ko_bold_font = pygame.font.Font(korean_bold, 14)
            self.ko_small_font = pygame.font.Font(korean_regular, 12)
            self.ko_alert_font = pygame.font.Font(korean_bold, 20)
        else:
            self.ko_title_font = pygame.font.SysFont(None, 24, bold=True)
            self.ko_header_font = pygame.font.SysFont(None, 16, bold=True)
            self.ko_font = pygame.font.SysFont(None, 15)
            self.ko_bold_font = pygame.font.SysFont(None, 15, bold=True)
            self.ko_small_font = pygame.font.SysFont(None, 13)
            self.ko_alert_font = pygame.font.SysFont(None, 22, bold=True)

        self.curv_buffer = None
        self.curv_y_max = 0.08
        self.smooth_path_m = 0.0
        self._text_cache = {}

    def get_text_surf(self, font, text, color):
        key = (id(font), text, color)
        surf = self._text_cache.get(key)
        if surf is None:
            surf = font.render(text, True, color)
            self._text_cache[key] = surf
        return surf

    def render(self, hits):
        env = self.env
        cam_x = env.cam_x  # 카메라 X 오프셋
        
        # 헬퍼: 월드좌표 → 스크린좌표 변환
        def sx(world_x):
            return world_x - cam_x

        is_full_3d = getattr(env, 'fullscreen_3d', False)
        
        if is_full_3d and getattr(self, 'engine_3d', None) is not None:
            # 전체화면 3D 모드 활성화 시:
            # 1. 상단 메인 화면(0, 0, 1800, 630)에 고해상도 3D 엔진 버퍼 렌더링
            try:
                main_3d = self.engine_3d.render(env, hits, env.w, env.sim_h)
                env.screen.blit(main_3d, (0, 0))
            except Exception as e:
                print(f"[Warning] Fullscreen 3D render failed: {e}")
            # 2. 하단 슬롯에 스왑 표출할 2D 월드를 전용 버퍼(world_2d_surf)에 사전 렌더링
            self._draw_2d_world(hits, sx, cam_x, target_surf=self.world_2d_surf)
        else:
            # 기본 2D 모드: 상단 메인 화면에 2D 월드 직접 렌더링
            self._draw_2d_world(hits, sx, cam_x, target_surf=env.screen)

        # 7. 하단 대시보드 UI (320x220 슬롯에 3D 또는 스왑된 2D 전술 맵 표출)
        self._draw_dashboard(hits, sx)

        # 8. 실시간 텔레메트리 HUD
        self._draw_telemetry()

        # 8-2. 메인 시뮬레이션 맵 좌측 상단 컨트롤러 (GAP NAVIGATION 버튼만 단독 배치)
        mpos = pygame.mouse.get_pos()

        # [버튼 1] 주행 알고리즘 모드 토글 (GAP NAVIGATION / LINE TRACING) - 좌측 상단 단독 유지
        top_btn = getattr(env, 'mode_btn_top_rect', pygame.Rect(25, 16, 165, 28))
        env.mode_btn_top_rect = top_btn
        top_hover = top_btn.collidepoint(mpos)
        is_lt_active = getattr(env, 'linetrace_mode', False)
        if is_lt_active:
            t_str = "LINE TRACING"
            t_bg = (45, 12, 36, 220) if top_hover else (32, 8, 25, 195)
            t_border = (255, 40, 195) if top_hover else (220, 20, 170)
            t_col = (255, 160, 230) if top_hover else (255, 50, 200)
        else:
            t_str = "GAP NAVIGATION"
            t_bg = (18, 36, 58, 220) if top_hover else (12, 26, 42, 195)
            t_border = (0, 190, 240) if top_hover else (0, 125, 175)
            t_col = (200, 235, 255) if top_hover else (150, 195, 225)
        top_surf = pygame.Surface((top_btn.w, top_btn.h), pygame.SRCALPHA)
        pygame.draw.rect(top_surf, t_bg, (0, 0, top_btn.w, top_btn.h), border_radius=4)
        pygame.draw.rect(top_surf, t_border, (0, 0, top_btn.w, top_btn.h), 1, border_radius=4)
        t_lbl = self.font.render(t_str, True, t_col)
        top_surf.blit(t_lbl, t_lbl.get_rect(center=(top_btn.w // 2, top_btn.h // 2)))
        env.screen.blit(top_surf, (top_btn.x, top_btn.y))

        # 8-3. 메인 시뮬레이션 맵 좌측 하단 제어 패널 (화면 스왑 버튼, 카메라 모드 버튼)
        # [버튼 2] 3D 전체화면 / 2D 맵 상하 화면 스왑 버튼
        view_btn = pygame.Rect(25, env.sim_h - 70, 165, 26)
        env.view_btn_top_rect = view_btn
        env.view_btn_rect = view_btn
        v_hover = view_btn.collidepoint(mpos)
        if is_full_3d:
            v_str = "VIEW: 2D MAP"
            v_bg = (24, 46, 32, 220) if v_hover else (14, 28, 20, 195)
            v_border = (40, 240, 140) if v_hover else (20, 190, 100)
            v_col = (180, 255, 210) if v_hover else (120, 240, 170)
        else:
            v_str = "VIEW: 3D FULL"
            v_bg = (16, 40, 68, 220) if v_hover else (10, 25, 45, 195)
            v_border = (0, 210, 255) if v_hover else (0, 150, 200)
            v_col = (210, 245, 255) if v_hover else (150, 215, 250)
        v_surf = pygame.Surface((view_btn.w, view_btn.h), pygame.SRCALPHA)
        pygame.draw.rect(v_surf, v_bg, (0, 0, view_btn.w, view_btn.h), border_radius=4)
        pygame.draw.rect(v_surf, v_border, (0, 0, view_btn.w, view_btn.h), 1, border_radius=4)
        v_lbl = self.font.render(v_str, True, v_col)
        v_surf.blit(v_lbl, v_lbl.get_rect(center=(view_btn.w // 2, view_btn.h // 2)))
        env.screen.blit(v_surf, (view_btn.x, view_btn.y))

        # [버튼 3] 3D 카메라 모드 변경 버튼 (Helm 1st / Chase 3rd / Drone Top)
        cam_btn = pygame.Rect(25, env.sim_h - 38, 165, 26)
        env.cam_btn_top_rect = cam_btn
        env.cam_btn_rect = cam_btn
        c_hover = cam_btn.collidepoint(mpos)
        cam_idx = getattr(env, 'cam_3d_mode', 1)
        cam_short = ["CAM: Helm (1st)", "CAM: Chase (3rd)", "CAM: Drone (Top)"][cam_idx % 3]
        c_bg = (32, 30, 52, 220) if c_hover else (20, 18, 34, 195)
        c_border = (190, 140, 255) if c_hover else (130, 90, 200)
        c_col = (235, 220, 255) if c_hover else (190, 175, 240)
        c_surf = pygame.Surface((cam_btn.w, cam_btn.h), pygame.SRCALPHA)
        pygame.draw.rect(c_surf, c_bg, (0, 0, cam_btn.w, cam_btn.h), border_radius=4)
        pygame.draw.rect(c_surf, c_border, (0, 0, cam_btn.w, cam_btn.h), 1, border_radius=4)
        c_lbl = self.small_font.render(cam_short, True, c_col)
        c_surf.blit(c_lbl, c_lbl.get_rect(center=(cam_btn.w // 2, cam_btn.h // 2)))
        env.screen.blit(c_surf, (cam_btn.x, cam_btn.y))

        # 8-4. 화면 중앙 최상단 상시 FPS 인디케이터 (슬림 게임 오버레이 스타일, 가독성 높은 18px 폰트 및 드롭 섀도우)
        fps_val = int(env.clock.get_fps()) if hasattr(env, 'clock') else 60
        fps_lbl = self.fps_font.render(f"{fps_val} FPS", True, (230, 235, 245))
        fps_sh = self.fps_font.render(f"{fps_val} FPS", True, (10, 15, 25))
        fps_rect = fps_lbl.get_rect(center=(env.w // 2, 18))
        env.screen.blit(fps_sh, (fps_rect.x + 1, fps_rect.y + 1))
        env.screen.blit(fps_lbl, fps_rect)

        # 8-5. 2D 메인 화면 우측 하단 엔진 인디케이터 (3D 화면의 ModernGL 텍스트와 100% 동일한 14px 폰트 및 위치)
        if not is_full_3d:
            pg_lbl = self.engine_info_font.render("Pygame 2D Engine", True, (225, 242, 255))
            pg_sh = self.engine_info_font.render("Pygame 2D Engine", True, (10, 15, 25))
            rx = env.w - pg_lbl.get_width() - 16
            ry = env.sim_h - 24
            env.screen.blit(pg_sh, (rx + 1, ry + 1))
            env.screen.blit(pg_lbl, (rx, ry))

        # 9. 미니맵 오버레이 (맵이 확장된 경우 주행화면 우측 하단에 표시, 3D 풀화면 모드에서는 가림)
        if env.map_w > env.w and not is_full_3d:
            self._draw_minimap()

        # 10. RC 수동 조종 모드 리더보드 모달 (목적지 도달 시 상위 10등 랭킹 및 알고리즘 벤치마크 표출)
        if getattr(env, 'show_leaderboard', False):
            self._draw_leaderboard_modal()

        pygame.display.flip()

    def _draw_2d_world(self, hits, sx, cam_x, target_surf=None):
        env = self.env
        if target_surf is None:
            target_surf = env.screen
        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)
        sbx = sx(bx)
        sby = by
        
        # 1. 밝고 맑은 마린 오션 수면 배경 (Brighter Clean Ocean)
        target_surf.fill((40, 118, 178))
        
        # 아주 은은하고 자연스러운 해양 잔물결 파도 (Gentle Natural Ocean Swell Waves)
        wave_t = env.frame * 0.016
        wave_start_j = int(cam_x // 80) * 80
        for i in range(25, env.sim_h, 60):
            for j in range(wave_start_j, int(cam_x + env.w + 80), 80):
                wj = j - cam_x
                wx = wj + math.cos(wave_t + i * 0.025 + j * 0.01) * 9
                wy = i + math.sin(wave_t * 0.7 + j * 0.025) * 5
                w_len = 16 + math.sin(wave_t + j * 0.02) * 6
                pygame.draw.line(target_surf, (55, 134, 196), (int(wx), int(wy)), (int(wx + w_len), int(wy)), 1)
                if (i + j) % 160 == 0:
                    pygame.draw.circle(target_surf, (210, 235, 255), (int(wx + w_len * 0.5), int(wy - 1)), 1)

        # 2. 360도 라이다 범위
        if env.show_lidar_range:
            pygame.draw.circle(target_surf, (80, 175, 140), (int(sbx), int(sby)), int(env.lidar_range), 1)
            for ang in env.rel_angles:
                ray_ang = h + ang
                rx = sbx + math.cos(ray_ang) * env.lidar_range
                ry = sby + math.sin(ray_ang) * env.lidar_range
                pygame.draw.line(target_surf, (55, 115, 90), (int(sbx), int(sby)), (int(rx), int(ry)), 1)

        # 3. 실제 선박 유체역학 항적 웨이크 + 장애물 반사/산란 미세 거품 (Realistic Wakes & Scattering Bubbles)
        env.wake_surf.fill((0, 0, 0, 0))

        for w in env.wakes:
            # w: [x, y, radius, alpha, vx, vy]
            if len(w) >= 6:
                w[0] += w[4] * 0.80
                w[1] += w[5] * 0.80
                w[4] *= 0.93
                w[5] *= 0.93
            w[2] += 1.15  # 웅장하고 풍성한 선미 거품 확장
            w[3] -= 2.4   # 긴 항적 지속성
            if w[3] > 0:
                wsx = int(sx(w[0]))
                wsy = int(w[1])
                if -50 < wsx < env.w + 50:
                    # 외곽 확장 파도 크레스트 (Wave Crest)
                    pygame.draw.circle(env.wake_surf, (220, 240, 255, int(w[3] * 0.42)), (wsx, wsy), int(w[2]))
                    # 내부 백색 기포 난류 (Aerated Foam Core)
                    if w[2] > 1.8:
                        pygame.draw.circle(env.wake_surf, (255, 255, 255, int(w[3] * 0.72)), (wsx, wsy), int(w[2] * 0.52))
        env.wakes = [w for w in env.wakes if w[3] > 0]
        
        # 장애물 충돌 반사 및 부표 들썩임 시 퍼지는 원형 백색 구름 파도
        if hasattr(env, 'reflected_wakes'):
            for rw in env.reflected_wakes:
                if len(rw) == 4:
                    rw[2] += 0.38
                    rw[3] -= 2.2
                    if rw[3] > 0:
                        rwsx = int(sx(rw[0]))
                        if -50 < rwsx < env.w + 50:
                            pygame.draw.circle(env.wake_surf, (225, 242, 255, int(rw[3] * 0.48)), (rwsx, int(rw[1])), int(rw[2]), 2)
                elif len(rw) >= 6:
                    rw[0] += rw[4]; rw[1] += rw[5]
                    rw[4] *= 0.88; rw[5] *= 0.88
                    rw[3] -= 6.0
                    if rw[3] > 0:
                        rwsx = int(sx(rw[0]))
                        if -50 < rwsx < env.w + 50:
                            pygame.draw.circle(env.wake_surf, (255, 255, 255, int(rw[3] * 0.8)), (rwsx, int(rw[1])), 1)
            env.reflected_wakes = [rw for rw in env.reflected_wakes if rw[3] > 0]
            
        # 장애물 부표 위치의 파도/구름을 완전히 지워 가림 처리 (Obstacle Clean Masking)
        for ox, oy, r in env.dynamic_obstacles:
            osx = int(sx(ox))
            if -50 < osx < env.w + 50:
                pygame.draw.circle(env.wake_surf, (0, 0, 0, 0), (osx, int(oy)), int(r + 0.5))
        
        target_surf.blit(env.wake_surf, (0, 0))
        target_surf.blit(env.trail, (0, 0), area=pygame.Rect(int(cam_x), 0, env.w, env.sim_h))
        
        # 4. 해상 장애물 - 뷰포트 내부만
        for ox, oy, r in env.dynamic_obstacles:
            osx = int(sx(ox))
            if osx < -30 or osx > env.w + 30:
                continue
            pygame.draw.circle(target_surf, (10, 42, 75, 140), (osx + 4, int(oy + 4)), int(r + 1))
            pygame.draw.circle(target_surf, (210, 45, 30), (osx, int(oy)), int(r))
            pygame.draw.circle(target_surf, (245, 75, 50), (osx - 1, int(oy - 1)), int(r * 0.76))
            pygame.draw.circle(target_surf, (255, 255, 255), (osx - 1, int(oy - 1)), int(r * 0.40))
            pygame.draw.circle(target_surf, (255, 255, 255), (osx, int(oy)), int(r * 0.20))
            
        from config import GRID
        occ_y, occ_x = np.where(env.grid >= 3)
        if len(occ_x) > 0:
            env.occ_surf.fill((0, 0, 0, 0))
            for gx_i, gy_i in zip(occ_x, occ_y):
                osx = int(gx_i * GRID - cam_x)
                if -10 < osx < env.w + 10:
                    pygame.draw.rect(env.occ_surf, (220, 50, 50, 60), (osx, gy_i * GRID, GRID, GRID))
            target_surf.blit(env.occ_surf, (0, 0))
            
        if env.show_lidar:
            for p in hits:
                if p is not None:
                    psx = int(sx(p[0]))
                    if -10 < psx < env.w + 10:
                        pygame.draw.circle(target_surf, (225, 220, 130), (psx, int(p[1])), 2)

        # 라인트레이싱 모드: 회피 장애물 히트지점
        if getattr(env, 'linetrace_mode', False) and getattr(env, 'show_closest_obstacle', True):
            c_hit = getattr(env, 'closest_avoid_hit', None)
            if c_hit is not None:
                csx, csy = int(sx(c_hit[0])), int(c_hit[1])
                bx_i, by_i = int(sbx), int(sby)
                pygame.draw.line(target_surf, (255, 20, 190), (bx_i, by_i), (csx, csy), 2)
                pygame.draw.circle(target_surf, (255, 20, 190, 80), (csx, csy), 10)
                pygame.draw.circle(target_surf, (255, 20, 190), (csx, csy), 6)
                pygame.draw.circle(target_surf, (255, 255, 255), (csx, csy), 2)

        # Safety Envelope
        self.safety_surf.fill((0, 0, 0, 0))
        safety_r = int(env.boat_radius + 18)
        em = getattr(env, 'emergency_mode', False)
        safety_color = (255, 60, 60, 40) if em else (0, 200, 120, 25)
        pygame.draw.circle(self.safety_surf, safety_color, (60, 60), safety_r)
        target_surf.blit(self.safety_surf, (int(sbx - 60), int(sby - 60)))
                
        # 목표점: 해양 항로 비콘
        tgx = sx(env.target[0]); tgy = env.target[1]
        if -30 < tgx < env.w + 30:
            pulse = math.sin(env.frame * 0.09) * 4.5
            pygame.draw.circle(target_surf, (0, 240, 100, 40), (int(tgx), int(tgy)), int(20 + pulse), 1)
            pygame.draw.circle(target_surf, (0, 230, 90, 75), (int(tgx), int(tgy)), int(14 + pulse * 0.5))
            pygame.draw.circle(target_surf, (20, 245, 80), (int(tgx), int(tgy)), 10)
            pygame.draw.circle(target_surf, (255, 255, 255), (int(tgx), int(tgy)), 5)
            itgx, itgy = int(tgx), int(tgy)
            pygame.draw.line(target_surf, (255, 255, 255, 180), (itgx - 16, itgy), (itgx + 16, itgy), 1)
            pygame.draw.line(target_surf, (255, 255, 255, 180), (itgx, itgy - 16), (itgx, itgy + 16), 1)
        
        # 5. 실시간 동적 추종 궤적 (베지어 곡선 및 웨이포인트)
        is_lt = getattr(env, 'linetrace_mode', False)
        show_1st = getattr(env, 'show_1st_path', True) and not is_lt
        show_2nd = getattr(env, 'show_2nd_path', True) and not is_lt

        if show_2nd:
            if env.next_wp is not None:
                nwp = env.next_wp
                if nwp.get("pair") != (-1, -1):
                    pygame.draw.line(target_surf, (255, 140, 0), (int(sx(nwp["c1"][0])), int(nwp["c1"][1])), (int(sx(nwp["c2"][0])), int(nwp["c2"][1])), 3)
                pygame.draw.circle(target_surf, (200, 100, 255, 100), (int(sx(nwp["pos"][0])), int(nwp["pos"][1])), 8)
                pygame.draw.circle(target_surf, (200, 100, 255), (int(sx(nwp["pos"][0])), int(nwp["pos"][1])), 3)

            if env.next_bezier_path is not None:
                pts = [(int(sx(x)), int(y)) for x, y in env.next_bezier_path]
                if len(pts) > 1:
                    pygame.draw.lines(target_surf, (255, 200, 50), False, pts, 3)

            if env.next_pursuit_target is not None:
                px_nt, py_nt = env.next_pursuit_target
                pygame.draw.circle(target_surf, (255, 255, 255), (int(sx(px_nt)), int(py_nt)), 8, 2)
                pygame.draw.circle(target_surf, (255, 150, 50), (int(sx(px_nt)), int(py_nt)), 4)

        if show_1st:
            if env.current_wp is not None:
                wp = env.current_wp
                pygame.draw.line(target_surf, (0, 255, 200), (int(sx(wp["c1"][0])), int(wp["c1"][1])), (int(sx(wp["c2"][0])), int(wp["c2"][1])), 4)
                pygame.draw.circle(target_surf, (0, 255, 255, 100), (int(sx(wp["pos"][0])), int(wp["pos"][1])), 10)
                pygame.draw.circle(target_surf, (0, 255, 255), (int(sx(wp["pos"][0])), int(wp["pos"][1])), 4)
                             
            if env.bezier_path is not None:
                pts = [(int(sx(x)), int(y)) for x, y in env.bezier_path]
                if len(pts) > 1:
                    pygame.draw.lines(target_surf, (50, 210, 255), False, pts, 4)

            if env.pursuit_target is not None:
                px_t, py_t = env.pursuit_target
                pygame.draw.circle(target_surf, (255, 255, 255), (int(sx(px_t)), int(py_t)), 10, 2)
                pygame.draw.circle(target_surf, (255, 50, 150), (int(sx(px_t)), int(py_t)), 5)

        # 6. 차순위 후보 웨이포인트 렌더링 (투명도 적용, 라인트레이싱 모드에서는 완전 제외)
        if not is_lt and getattr(env, 'show_candidates', True) and getattr(env, 'candidate_wps', None):
            cand_surf = self._cand_surf
            cand_surf.fill((0, 0, 0, 0))
            cand_colors = [(80, 210, 255, 130), (255, 180, 70, 120)]
            for rank_idx, cand in enumerate(env.candidate_wps[:2]):
                col = cand_colors[rank_idx % len(cand_colors)]
                c1, c2 = cand["c1"], cand["c2"]
                mid = cand["pos"]
                pygame.draw.line(cand_surf, col, (int(sx(c1[0])), int(c1[1])), (int(sx(c2[0])), int(c2[1])), 2)
                pygame.draw.circle(cand_surf, (col[0], col[1], col[2], 50), (int(sx(mid[0])), int(mid[1])), 9)
                pygame.draw.circle(cand_surf, col, (int(sx(mid[0])), int(mid[1])), 9, 2)
                pygame.draw.circle(cand_surf, (col[0], col[1], col[2], 210), (int(sx(mid[0])), int(mid[1])), 3)
                txt_rank = self.small_font.render(f"#{rank_idx + 2}", True, (col[0], col[1], col[2]))
                cand_surf.blit(txt_rank, (int(sx(mid[0])) + 10, int(mid[1]) - 8))
            target_surf.blit(cand_surf, (0, 0))

        # 6-2. 고려 중인 모든 갭의 중간점 위치 렌더링 (Gaps 버튼 클릭 시 ON/OFF 토글, 라인트레이싱 모드에서는 완전 제외)
        if not is_lt and getattr(env, 'show_all_gaps', False) and getattr(env, 'all_gaps', None):
            gaps_surf = getattr(self, '_all_gaps_surf', None)
            if gaps_surf is None:
                self._all_gaps_surf = pygame.Surface((env.w, env.h), pygame.SRCALPHA)
                gaps_surf = self._all_gaps_surf
            else:
                gaps_surf.fill((0, 0, 0, 0))
            visible_idx = 1
            for g in env.all_gaps:
                mid = g["pos"]
                # 전방 180도 (헤딩 기준 좌우 ±90도, 전방 성분 >= 0) 검사
                dx = mid[0] - bx
                dy = mid[1] - by
                if dx * ch + dy * sh < 0:
                    continue
                msx = int(sx(mid[0]))
                if msx < -20 or msx > env.w + 20:
                    continue
                c1, c2 = g.get("c1"), g.get("c2")
                # 갭 부표 사이 얇은 가이드 라인
                if c1 is not None and c2 is not None:
                    pygame.draw.line(gaps_surf, (0, 220, 255, 55), (int(sx(c1[0])), int(c1[1])), (int(sx(c2[0])), int(c2[1])), 1)
                # 갭 중간점 마커 (반투명 헤일로 + 링 + 중심 코어)
                pygame.draw.circle(gaps_surf, (0, 240, 255, 45), (msx, int(mid[1])), 8)
                pygame.draw.circle(gaps_surf, (0, 240, 255, 180), (msx, int(mid[1])), 6, 1)
                pygame.draw.circle(gaps_surf, (255, 255, 255, 230), (msx, int(mid[1])), 2)
                # 갭 인덱스 라벨 (G1, G2, ...)
                lbl_g = self.micro_font.render(f"G{visible_idx}", True, (0, 240, 255))
                gaps_surf.blit(lbl_g, (msx + 8, int(mid[1]) - 6))
                visible_idx += 1
            target_surf.blit(gaps_surf, (0, 0))

        # 선박 형상 정밀 렌더링 (스크린 좌표)
        self._draw_boat_hull(sbx, sby, ch, sh, target_surf=target_surf)

        # 7. 라이다 블라인드 시연 모드 (LiDAR Blind Vision Darkness Mask)
        # 사용자 요청: 라이다 범위 내에만 시야를 밝혀두고 나머진 검은색으로 다 뜨도록 하고, 검은색 원 테두리에 도착 지점이 어디인지만 표시
        if getattr(env, 'blind_mode', False):
            surf_w, surf_h = target_surf.get_size()
            if not hasattr(self, '_blind_mask_surf') or self._blind_mask_surf.get_size() != (surf_w, surf_h):
                self._blind_mask_surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)
            
            # (1) 암전 마스크 생성 및 원형 라이다 시야 천공
            mask = self._blind_mask_surf
            mask.fill((0, 0, 0, 255))
            r_lidar = int(env.lidar_range)
            pygame.draw.circle(mask, (0, 0, 0, 0), (int(sbx), int(sby)), r_lidar)
            target_surf.blit(mask, (0, 0))
            
            # (2) 라이다 시야 원 테두리 발광 링 렌더링
            pygame.draw.circle(target_surf, (0, 220, 255), (int(sbx), int(sby)), r_lidar, 2)
            pygame.draw.circle(target_surf, (0, 160, 255, 80), (int(sbx), int(sby)), r_lidar + 1, 1)
            
            # (3) 원 테두리 상에 도착 지점(Target) 방향 및 거리 지시 표식 렌더링
            dx_t = env.target[0] - env.boat_pos[0]
            dy_t = env.target[1] - env.boat_pos[1]
            dist_t = math.hypot(dx_t, dy_t)
            ang_t = math.atan2(dy_t, dx_t)
            
            # 원 테두리 상의 목표점 교차 좌표
            rim_x = sbx + math.cos(ang_t) * r_lidar
            rim_y = sby + math.sin(ang_t) * r_lidar
            
            # 펄스 헤일로 및 네온 그린 비콘 마커
            pulse = math.sin(env.frame * 0.15) * 3.0
            pygame.draw.circle(target_surf, (0, 255, 120, 90), (int(rim_x), int(rim_y)), int(14 + pulse), 2)
            
            # 다이아몬드 마커
            dm_r = 7
            irx, iry = int(rim_x), int(rim_y)
            diamond_pts = [
                (irx, iry - dm_r),
                (irx + dm_r, iry),
                (irx, iry + dm_r),
                (irx - dm_r, iry)
            ]
            pygame.draw.polygon(target_surf, (40, 255, 110), diamond_pts)
            pygame.draw.polygon(target_surf, (255, 255, 255), diamond_pts, 1)
            
            # 목표 방향 화살표 지시선 (원 테두리 바깥 방향)
            arr_len = 16
            ax = irx + int(math.cos(ang_t) * arr_len)
            ay = iry + int(math.sin(ang_t) * arr_len)
            pygame.draw.line(target_surf, (80, 255, 140), (irx, iry), (ax, ay), 3)
            
            # 도착 지점 거리 정보 배지 (예: "GOAL 24.5m")
            dist_m = dist_t / 50.0
            badge_txt = f"GOAL {dist_m:.1f}m"
            lbl_badge = self.bold_font.render(badge_txt, True, (80, 255, 150))
            badge_w = lbl_badge.get_width() + 10
            badge_h = lbl_badge.get_height() + 4
            
            # 배지 위치: 테두리 안쪽으로 약간 오프셋하여 화면 및 마스크 내 가독성 확보
            tag_offset = 32
            bx_tag = rim_x - math.cos(ang_t) * tag_offset - badge_w / 2
            by_tag = rim_y - math.sin(ang_t) * tag_offset - badge_h / 2
            bx_tag = max(8, min(surf_w - badge_w - 8, bx_tag))
            by_tag = max(8, min(surf_h - badge_h - 8, by_tag))
            
            tag_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
            tag_surf.fill((10, 30, 20, 210))
            pygame.draw.rect(tag_surf, (40, 255, 120), (0, 0, badge_w, badge_h), 1, border_radius=3)
            tag_surf.blit(lbl_badge, (5, 2))
            target_surf.blit(tag_surf, (int(bx_tag), int(by_tag)))

    def _draw_minimap(self):
        """전체 맵에서 현재 위치를 표시하는 미니맵 오버레이 (주행화면 우측 하단)"""
        env = self.env
        
        # 미니맵 크기 및 위치 설정 (주행화면 맨아래 우측, 텔레메트리 HUD 패널과 겹치지 않도록 배치)
        mm_w = 340   # 미니맵 가로
        mm_h = 32    # 미니맵 세로
        mm_x = env.w - mm_w - 15  # 우측
        mm_y = env.sim_h - mm_h - 10  # 주행화면(sim_h) 맨아래
        
        # 스케일 팩터
        scale_x = mm_w / env.map_w
        scale_y = mm_h / env.sim_h
        
        # 미니맵 배경 (반투명 다크블루)
        mm_surf = pygame.Surface((mm_w + 4, mm_h + 4), pygame.SRCALPHA)
        pygame.draw.rect(mm_surf, (8, 18, 38, 200), (0, 0, mm_w + 4, mm_h + 4), border_radius=3)
        pygame.draw.rect(mm_surf, (40, 100, 160, 180), (2, 2, mm_w, mm_h), border_radius=2)
        
        # 장애물 (작은 빨간 점) - 블라인드 모드 시 라이다 범위 밖 장애물은 은닉
        is_blind = getattr(env, 'blind_mode', False)
        for ox, oy, r in env.dynamic_obstacles:
            if is_blind:
                if math.hypot(ox - env.boat_pos[0], oy - env.boat_pos[1]) > env.lidar_range:
                    continue
            mx = int(2 + ox * scale_x)
            my = int(2 + oy * scale_y)
            pygame.draw.circle(mm_surf, (220, 60, 40, 200), (mx, my), max(1, int(r * scale_x)))
        
        # 항적 (env.trail 표면 축소 렌더링)
        mm_trail = pygame.transform.scale(env.trail, (mm_w, mm_h))
        mm_surf.blit(mm_trail, (2, 2))
        
        # 웨이포인트 (1st: 시안, 2nd: 보라)
        is_lt = getattr(env, 'linetrace_mode', False)
        if not is_lt:
            if env.current_wp is not None:
                wp = env.current_wp["pos"]
                pygame.draw.circle(mm_surf, (0, 255, 255), (int(2 + wp[0] * scale_x), int(2 + wp[1] * scale_y)), 3)
            if env.next_wp is not None:
                nwp = env.next_wp["pos"]
                pygame.draw.circle(mm_surf, (200, 100, 255), (int(2 + nwp[0] * scale_x), int(2 + nwp[1] * scale_y)), 2)
        
        # 목표점 (녹색)
        tgx_mm = int(2 + env.target[0] * scale_x)
        tgy_mm = int(2 + env.target[1] * scale_y)
        pygame.draw.circle(mm_surf, (0, 240, 80), (tgx_mm, tgy_mm), 4)
        pygame.draw.circle(mm_surf, (255, 255, 255), (tgx_mm, tgy_mm), 2)
        
        # 현재 뷰포트 범위 (밝은 테두리 사각형)
        vp_x = int(2 + env.cam_x * scale_x)
        vp_w = int(env.w * scale_x)
        vp_h = mm_h
        pygame.draw.rect(mm_surf, (255, 255, 255, 150), (vp_x, 2, vp_w, vp_h), 1)
        
        # 보트 현재 위치 (밝은 시안 점, 가장 마지막에 그려서 가장 위에 표시)
        bx_mm = int(2 + env.boat_pos[0] * scale_x)
        by_mm = int(2 + env.boat_pos[1] * scale_y)
        pygame.draw.circle(mm_surf, (0, 255, 255), (bx_mm, by_mm), 4)
        pygame.draw.circle(mm_surf, (255, 255, 255), (bx_mm, by_mm), 2)
        
        # "MAP" 라벨
        lbl = self.micro_font.render("MAP", True, (180, 210, 240))
        mm_surf.blit(lbl, (4, mm_h - 8))
        
        # 진행률 표시
        progress = min(100, max(0, env.boat_pos[0] / env.map_w * 100))
        prog_str = f"{progress:.0f}%"
        prog_lbl = self.micro_font.render(prog_str, True, (200, 255, 200))
        mm_surf.blit(prog_lbl, (mm_w - 26, mm_h - 8))
        
        env.screen.blit(mm_surf, (mm_x, mm_y))

    def _draw_boat_hull(self, bx, by, ch, sh, target_surf=None):
        env = self.env
        if target_surf is None:
            target_surf = env.screen
        GAP = 11; L = 84; W = 16
        left_center = (bx - sh*GAP, by + ch*GAP)
        right_center = (bx + sh*GAP, by - ch*GAP)
        
        hull_local = [
            (L*0.50, 0), (L*0.12, W),
            (-L*0.28, W*0.85), (-L*0.48, W*0.6),
            (-L*0.50, 0), (-L*0.48, -W*0.6),
            (-L*0.28, -W*0.85), (L*0.12, -W)
        ]
        
        def TR(c, px_l, py_l):
            return int(c[0] + px_l*ch - py_l*sh), int(c[1] + px_l*sh + py_l*ch)
            
        left_h = [TR(left_center, p[0], p[1]) for p in hull_local]
        right_h = [TR(right_center, p[0], p[1]) for p in hull_local]
        
        # 선체 하부 앰비언트 수중 그림자 (180x180 로컬 버퍼 최적화)
        self.shadow_surf.fill((0, 0, 0, 0))
        shadow_offset = 7
        sx_base = int(bx - 90)
        sy_base = int(by - 90)
        left_shadow = [(p[0] + shadow_offset - sx_base, p[1] + shadow_offset - sy_base) for p in left_h]
        right_shadow = [(p[0] + shadow_offset - sx_base, p[1] + shadow_offset - sy_base) for p in right_h]
        pygame.draw.polygon(self.shadow_surf, (8, 30, 55, 150), left_shadow)
        pygame.draw.polygon(self.shadow_surf, (8, 30, 55, 150), right_shadow)
        target_surf.blit(self.shadow_surf, (sx_base, sy_base))

        # 좌/우 선체 (군함/실험선 건메탈 그레이 - Tone 1: Gunmetal Grey)
        pygame.draw.polygon(target_surf, (52, 60, 70), left_h)
        pygame.draw.polygon(target_surf, (28, 34, 40), left_h, 2)
        pygame.draw.polygon(target_surf, (52, 60, 70), right_h)
        pygame.draw.polygon(target_surf, (28, 34, 40), right_h, 2)

        # 좌우 선체 상단 하이라이트 스트립
        left_deck_line = [TR(left_center, L*0.35, 0), TR(left_center, -L*0.35, 0)]
        right_deck_line = [TR(right_center, L*0.35, 0), TR(right_center, -L*0.35, 0)]
        pygame.draw.line(target_surf, (85, 96, 108), left_deck_line[0], left_deck_line[1], 2)
        pygame.draw.line(target_surf, (85, 96, 108), right_deck_line[0], right_deck_line[1], 2)

        # 중앙 연결 브릿지 데크 (투톤 대비 - Tone 2: Crisp Platinum Deck)
        deck_corners = [
            TR((bx, by), L*0.25, -GAP*0.85),
            TR((bx, by), L*0.25, GAP*0.85),
            TR((bx, by), -L*0.35, GAP*0.85),
            TR((bx, by), -L*0.35, -GAP*0.85)
        ]
        pygame.draw.polygon(target_surf, (210, 218, 228), deck_corners)
        pygame.draw.polygon(target_surf, (90, 100, 112), deck_corners, 1)

        # 데크 중앙 미끄럼 방지 패드 라인
        deck_pad = [
            TR((bx, by), L*0.20, -GAP*0.65),
            TR((bx, by), L*0.20, GAP*0.65),
            TR((bx, by), -L*0.30, GAP*0.65),
            TR((bx, by), -L*0.30, -GAP*0.65)
        ]
        pygame.draw.polygon(target_surf, (165, 175, 188), deck_pad)

        # 캐빈 조종실 팟 (Stealth Tactical Cabin)
        cabin_corners = [
            TR((bx, by), L*0.16, -GAP*0.55),
            TR((bx, by), L*0.16, GAP*0.55),
            TR((bx, by), -L*0.16, GAP*0.55),
            TR((bx, by), -L*0.16, -GAP*0.55)
        ]
        pygame.draw.polygon(target_surf, (75, 84, 96), cabin_corners)
        pygame.draw.polygon(target_surf, (35, 42, 50), cabin_corners, 1)

        # 틴팅 전면 윈드실드 창문 (Tinted Marine Cockpit Glass)
        windshield = [
            TR((bx, by), L*0.13, -GAP*0.42),
            TR((bx, by), L*0.13, GAP*0.42),
            TR((bx, by), L*0.04, GAP*0.42),
            TR((bx, by), L*0.04, -GAP*0.42)
        ]
        pygame.draw.polygon(target_surf, (28, 105, 160), windshield)
        pygame.draw.line(target_surf, (180, 230, 255), TR((bx, by), L*0.12, -GAP*0.3), TR((bx, by), L*0.06, GAP*0.3), 1)

        # 후방 GPS 수신기 마스트 돔 & 통신 휩 안테나 (GPS Dome & Whip Antenna)
        gps_pos = TR((bx, by), -L*0.22, GAP*0.35)
        pygame.draw.circle(target_surf, (245, 248, 255), gps_pos, 4)
        pygame.draw.circle(target_surf, (60, 70, 80), gps_pos, 4, 1)
        # 휩 안테나
        ant_pos = TR((bx, by), -L*0.24, -GAP*0.35)
        pygame.draw.circle(target_surf, (30, 35, 40), ant_pos, 2)
        pygame.draw.line(target_surf, (200, 210, 220), ant_pos, (ant_pos[0]-1, ant_pos[1]-6), 2)

        # 선체 일체형 소형 T500 덕트 쓰러스터 (Integrated Compact T500 Thrusters)
        t_ch, t_sh = ch, sh
        t_nx, t_ny = -sh, ch
        d_len = 11; d_rad = 4.8
        
        for m_center in [left_center, right_center]:
            p_center = TR(m_center, -L*0.50, 0)
            
            # 덕트 노즐 모서리
            d_fl = (int(p_center[0] + (d_len*0.5)*t_ch - d_rad*t_nx), int(p_center[1] + (d_len*0.5)*t_sh - d_rad*t_ny))
            d_fr = (int(p_center[0] + (d_len*0.5)*t_ch + d_rad*t_nx), int(p_center[1] + (d_len*0.5)*t_sh + d_rad*t_ny))
            d_rr = (int(p_center[0] - (d_len*0.5)*t_ch + d_rad*t_nx), int(p_center[1] - (d_len*0.5)*t_sh + d_rad*t_ny))
            d_rl = (int(p_center[0] - (d_len*0.5)*t_ch - d_rad*t_nx), int(p_center[1] - (d_len*0.5)*t_sh - d_rad*t_ny))
            
            # 일체형 덕트 쉘
            pygame.draw.polygon(target_surf, (32, 38, 46), [d_fl, d_fr, d_rr, d_rl])
            pygame.draw.polygon(target_surf, (68, 78, 92), [d_fl, d_fr, d_rr, d_rl], 1)
            
            # 중앙 모터 코어 & 프로펠러
            m_f = (int(p_center[0] + 3*t_ch), int(p_center[1] + 3*t_sh))
            m_r = (int(p_center[0] - 4*t_ch), int(p_center[1] - 4*t_sh))
            pygame.draw.line(target_surf, (18, 22, 28), m_f, m_r, 3)
            
            prop_c = (int(p_center[0] - 1*t_ch), int(p_center[1] - 1*t_sh))
            p_b1 = (int(prop_c[0] - 3.5*t_nx), int(prop_c[1] - 3.5*t_ny))
            p_b2 = (int(prop_c[0] + 3.5*t_nx), int(prop_c[1] + 3.5*t_ny))
            pygame.draw.line(target_surf, (225, 235, 245), p_b1, p_b2, 2)

        # 중앙 회전식 라이다 센서 돔 (Rotating LiDAR Sensor Pod)
        Lidar_pos = TR((bx, by), -L*0.05, 0)
        pygame.draw.circle(target_surf, (32, 36, 42), Lidar_pos, 6)
        pygame.draw.circle(target_surf, (255, 215, 30), Lidar_pos, 3)
        # 라이다 360도 스캔 레이저 펄스 회전선
        scan_ang = env.frame * 0.35
        sp_x = int(Lidar_pos[0] + math.cos(scan_ang) * 6)
        sp_y = int(Lidar_pos[1] + math.sin(scan_ang) * 6)
        pygame.draw.line(target_surf, (0, 255, 200), Lidar_pos, (sp_x, sp_y), 2)
        pygame.draw.circle(target_surf, (0, 255, 200), (sp_x, sp_y), 2)

    def _draw_dashboard(self, hits, sx=None):
        env = self.env
        if sx is None:
            sx = lambda wx: wx - env.cam_x
        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)

        pygame.draw.rect(env.screen, (15, 35, 60), (0, env.sim_h, env.w, env.h - env.sim_h))
        pygame.draw.line(env.screen, (0, 180, 255), (0, env.sim_h), (env.w, env.sim_h), 3)

        layout = get_dashboard_layout(env.w, env.sim_h)
        p1_x, p2_x, p3_x, p4_x, p5_x = layout['p1_x'], layout['p2_x'], layout['p3_x'], layout['p4_x'], layout['p5_x']
        p_y = layout['y']
        env.panel_3d_rect = pygame.Rect(p3_x, p_y, 320, 220)

        # 마우스 커서 위치 확인 (호버 인터랙션)
        mpos = pygame.mouse.get_pos()

        # 체크박스 렌더링
        is_lt = getattr(env, 'linetrace_mode', False)
        if is_lt:
            # [라인트레이싱 전용 UI] 3개 버튼 구성
            # 1. 가장 가까운 회피 장애물 지점 SHOW 버튼 (네온 마젠타)
            cb1_row = getattr(env, 'cb1_row_rect', env.cb1_rect)
            cb1_hover = cb1_row.collidepoint(mpos)
            if cb1_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb1_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb1_rect, 2)
            if getattr(env, 'show_closest_obstacle', True):
                pygame.draw.rect(env.screen, (255, 20, 190), env.cb1_rect.inflate(-6, -6))
            txt_col1 = (255, 140, 220) if cb1_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show Closest Obstacle", txt_col1), (70, env.cb1_rect.centery - 10))

            # 2. Show LiDAR Hits (소프트 옐로우)
            cb2_row = getattr(env, 'cb2_row_rect', env.cb2_rect)
            cb2_hover = cb2_row.collidepoint(mpos)
            if cb2_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb2_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb2_rect, 2)
            if env.show_lidar:
                pygame.draw.rect(env.screen, (225, 220, 130), env.cb2_rect.inflate(-6, -6))
            txt_col2 = (250, 245, 175) if cb2_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show LiDAR Hits", txt_col2), (70, env.cb2_rect.centery - 10))

            # 3. Show LiDAR Range (세이지 그린)
            cb3_row = getattr(env, 'cb3_row_rect', env.cb3_rect)
            cb3_hover = cb3_row.collidepoint(mpos)
            if cb3_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb3_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb3_rect, 2)
            if env.show_lidar_range:
                pygame.draw.rect(env.screen, (80, 175, 140), env.cb3_rect.inflate(-6, -6))
            txt_col3 = (130, 225, 180) if cb3_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show LiDAR Range", txt_col3), (70, env.cb3_rect.centery - 10))
        else:
            # [기본 갭 항법 모드 UI] 5개 체크박스 구성
            # 1. Show 1st Path (시안)
            cb1_row = getattr(env, 'cb1_row_rect', env.cb1_rect)
            cb1_hover = cb1_row.collidepoint(mpos)
            if cb1_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb1_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb1_rect, 2)
            if getattr(env, 'show_1st_path', True): pygame.draw.rect(env.screen, (0, 255, 200), env.cb1_rect.inflate(-6, -6))
            txt_col1 = (120, 255, 230) if cb1_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show 1st Path", txt_col1), (70, env.cb1_rect.centery - 10))

            # 2. Show 2nd Path (오렌지)
            cb2_row = getattr(env, 'cb2_row_rect', env.cb2_rect)
            cb2_hover = cb2_row.collidepoint(mpos)
            if cb2_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb2_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb2_rect, 2)
            if getattr(env, 'show_2nd_path', True): pygame.draw.rect(env.screen, (255, 140, 0), env.cb2_rect.inflate(-6, -6))
            txt_col2 = (255, 185, 95) if cb2_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show 2nd Path", txt_col2), (70, env.cb2_rect.centery - 10))

            # 3. Show Candidate WPs (연보라)
            cb3_row = getattr(env, 'cb3_row_rect', env.cb3_rect)
            cb3_hover = cb3_row.collidepoint(mpos)
            if cb3_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb3_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb3_rect, 2)
            if getattr(env, 'show_candidates', True): pygame.draw.rect(env.screen, (160, 180, 255), env.cb3_rect.inflate(-6, -6))
            txt_col3 = (195, 215, 255) if cb3_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show Candidate WPs", txt_col3), (70, env.cb3_rect.centery - 10))

            # 4. Show LiDAR Hits (소프트 옐로우)
            cb4_row = getattr(env, 'cb4_row_rect', env.cb4_rect)
            cb4_hover = cb4_row.collidepoint(mpos)
            if cb4_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb4_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb4_rect, 2)
            if env.show_lidar: pygame.draw.rect(env.screen, (225, 220, 130), env.cb4_rect.inflate(-6, -6))
            txt_col4 = (250, 245, 175) if cb4_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show LiDAR Hits", txt_col4), (70, env.cb4_rect.centery - 10))

            # 5. Show LiDAR Range (세이지 그린)
            cb5_row = getattr(env, 'cb5_row_rect', env.cb5_rect)
            cb5_hover = cb5_row.collidepoint(mpos)
            if cb5_hover:
                pygame.draw.rect(env.screen, (28, 56, 88), cb5_row, border_radius=4)
            pygame.draw.rect(env.screen, (255, 255, 255), env.cb5_rect, 2)
            if env.show_lidar_range: pygame.draw.rect(env.screen, (80, 175, 140), env.cb5_rect.inflate(-6, -6))
            txt_col5 = (130, 225, 180) if cb5_hover else (255, 255, 255)
            env.screen.blit(self.get_text_surf(self.font, "Show LiDAR Range", txt_col5), (70, env.cb5_rect.centery - 10))
        
        # 일시정지(PAUSE) 버튼
        is_paused = getattr(env, 'paused', False)
        p_hover = env.pause_btn.collidepoint(mpos)
        
        if is_paused:
            p_bg = (245, 130, 20) if p_hover else (225, 110, 15)
            p_border = (255, 255, 255)
            p_col = (255, 255, 255)
            border_w = 2
        else:
            p_bg = (35, 65, 100) if p_hover else (20, 40, 65)
            p_border = (0, 220, 255) if p_hover else (70, 110, 150)
            p_col = (255, 255, 255) if p_hover else (200, 225, 245)
            border_w = 2 if p_hover else 1
        
        pygame.draw.rect(env.screen, p_bg, env.pause_btn, border_radius=4)
        pygame.draw.rect(env.screen, p_border, env.pause_btn, border_w, border_radius=4)
        
        cx, cy = env.pause_btn.center
        if is_paused:
            # 선명한 재생 삼각형 아이콘 (Play Polygon)
            play_pts = [(cx - 5, cy - 7), (cx - 5, cy + 7), (cx + 6, cy)]
            pygame.draw.polygon(env.screen, p_col, play_pts)
        else:
            # 선명한 일시정지 더블 바 아이콘 (Pause Double Bars)
            pygame.draw.rect(env.screen, p_col, (cx - 6, cy - 7, 4, 14), border_radius=1)
            pygame.draw.rect(env.screen, p_col, (cx + 2, cy - 7, 4, 14), border_radius=1)

        # 배속 버튼
        cur_spd = getattr(env, 'sim_speed', 1)
        for spd, btn_rect in env.speed_btns.items():
            is_active = (cur_spd == spd and not is_paused)
            s_hover = btn_rect.collidepoint(mpos)
            if is_active:
                btn_bg = (0, 235, 255)
                btn_border = (255, 255, 255)
                border_w = 3
                text_color = (0, 15, 35)
            elif s_hover:
                btn_bg = (35, 60, 90)
                btn_border = (0, 200, 255)
                border_w = 1
                text_color = (240, 250, 255)
            else:
                btn_bg = (25, 45, 70)
                btn_border = (70, 105, 145)
                border_w = 1
                text_color = (210, 225, 240)
            
            pygame.draw.rect(env.screen, btn_bg, btn_rect, border_radius=4)
            pygame.draw.rect(env.screen, btn_border, btn_rect, border_w, border_radius=4)
            
            lbl = self.bold_font.render(f"{spd}x", True, text_color) if is_active else self.font.render(f"{spd}x", True, text_color)
            env.screen.blit(lbl, (btn_rect.centerx - lbl.get_width()//2, btn_rect.centery - lbl.get_height()//2))

        # --- 1. 180도 전방 확대 LiDAR View (2D) ---
        pov_w, pov_h = 320, 220
        self.pov_surf.fill((10, 25, 45, 240))
        pygame.draw.rect(self.pov_surf, (0, 180, 255), (0, 0, pov_w, pov_h), 2)
        
        pcx, pcy = pov_w // 2, pov_h - 25
        f_vec = np.array([ch, sh])
        r_vec = np.array([-sh, ch])
        
        scale_r = 0.55

        # 180도 전방 부채꼴 가이드라인 및 방위각 레이더 그리드
        angles_deg = [-90, -60, -30, 0, 30, 60, 90]
        for deg in angles_deg:
            rad = math.radians(deg)
            rx = pcx + math.sin(rad) * (env.lidar_range * scale_r)
            ry = pcy - math.cos(rad) * (env.lidar_range * scale_r)
            pygame.draw.line(self.pov_surf, (0, 70, 110), (pcx, pcy), (int(rx), int(ry)), 1)
            
            display_deg = deg + 90
            txt_ang = self.small_font.render(f"{display_deg}°", True, (0, 140, 200))
            tx_off = -12 if deg < 0 else (-6 if deg == 0 else 2)
            ty_off = -12 if ry < pcy else 2
            self.pov_surf.blit(txt_ang, (int(rx) + tx_off, int(ry) + ty_off))

        # 동심원 스케일 서클 (50px = 1m 기준: 2m, 4m, 6m)
        for dist in [100, 200, 300]:
            r_pixel = int(dist * scale_r)
            rect = pygame.Rect(pcx - r_pixel, pcy - r_pixel, r_pixel * 2, r_pixel * 2)
            pygame.draw.arc(self.pov_surf, (0, 90, 140), rect, 0, math.pi, 1)
            lbl = self.small_font.render(f"{dist // 50}m", True, (0, 120, 170))
            self.pov_surf.blit(lbl, (pcx + 4, pcy - r_pixel - 10))

        # 180도 스캔 레이 라인 (저채도 세이지 그린)
        if env.show_lidar_range:
            for ang in env.rel_angles:
                if -math.pi/2 <= ang <= math.pi/2:
                    rx = pcx + math.sin(ang) * (env.lidar_range * scale_r)
                    ry = pcy - math.cos(ang) * (env.lidar_range * scale_r)
                    pygame.draw.line(self.pov_surf, (55, 120, 95), (pcx, pcy), (int(rx), int(ry)), 1)

        # 라이다 히트 포인트 렌더링 (저채도 소프트 옐로우)
        if env.show_lidar:
            for hp in hits:
                if hp is not None:
                    hdx = hp[0] - bx; hdy = hp[1] - by
                    hlf = hdx * f_vec[0] + hdy * f_vec[1]
                    hlr = hdx * r_vec[0] + hdy * r_vec[1]
                    if hlf >= -10:
                        pygame.draw.circle(self.pov_surf, (225, 220, 130), (int(pcx + hlr * scale_r), int(pcy - hlf * scale_r)), 2)

        # 라인트레이싱 모드: POV 뷰에서 가장 가까운 장애물 히트지점 및 연결선 표출
        if getattr(env, 'linetrace_mode', False) and getattr(env, 'show_closest_obstacle', True):
            c_hit = getattr(env, 'closest_avoid_hit', None)
            if c_hit is not None:
                hdx = c_hit[0] - bx; hdy = c_hit[1] - by
                hlf = hdx * f_vec[0] + hdy * f_vec[1]
                hlr = hdx * r_vec[0] + hdy * r_vec[1]
                if hlf >= -10:
                    cx_p = int(pcx + hlr * scale_r)
                    cy_p = int(pcy - hlf * scale_r)
                    pygame.draw.line(self.pov_surf, (255, 20, 190), (pcx, pcy), (cx_p, cy_p), 2)
                    pygame.draw.circle(self.pov_surf, (255, 20, 190), (cx_p, cy_p), 5)
                    pygame.draw.circle(self.pov_surf, (255, 255, 255), (cx_p, cy_p), 2)

        # --- 목적지 인디케이터 & 테두리 트래킹 컴퍼스 ---
        dx_t = env.target[0] - bx; dy_t = env.target[1] - by
        lf_t = dx_t * f_vec[0] + dy_t * f_vec[1]
        lr_t = dx_t * r_vec[0] + dy_t * r_vec[1]
        
        tx_p = pcx + lr_t * scale_r
        ty_p = pcy - lf_t * scale_r
        
        margin = 0
        dist_total_m = math.hypot(dx_t, dy_t) / 50.0
        
        if margin <= tx_p <= pov_w - margin and margin <= ty_p <= pov_h - margin:
            pygame.draw.circle(self.pov_surf, (20, 250, 80), (int(tx_p), int(ty_p)), 7)
            pygame.draw.circle(self.pov_surf, (255, 255, 255), (int(tx_p), int(ty_p)), 3)
        else:
            dir_x = tx_p - pcx
            dir_y = ty_p - pcy
            
            t_candidates = []
            if dir_x < 0: t_candidates.append((margin - pcx) / dir_x)
            elif dir_x > 0: t_candidates.append(((pov_w - margin) - pcx) / dir_x)
            
            if dir_y < 0: t_candidates.append((margin - pcy) / dir_y)
            elif dir_y > 0: t_candidates.append(((pov_h - margin) - pcy) / dir_y)
            
            valid_t = [t for t in t_candidates if t > 0]
            if valid_t:
                t_edge = min(valid_t)
                edge_x = int(pcx + t_edge * dir_x)
                edge_y = int(pcy + t_edge * dir_y)
                
                pygame.draw.line(self.pov_surf, (20, 220, 80), (pcx, pcy), (edge_x, edge_y), 1)
                pygame.draw.circle(self.pov_surf, (20, 250, 80), (edge_x, edge_y), 6)
                pygame.draw.circle(self.pov_surf, (255, 255, 255), (edge_x, edge_y), 2)
                
                dist_txt = self.small_font.render(f"{dist_total_m:.0f}m", True, (20, 250, 80))
                lbl_x = max(10, min(edge_x - 12, pov_w - 40))
                lbl_y = max(10, min(edge_y - 12, pov_h - 18))
                self.pov_surf.blit(dist_txt, (lbl_x, lbl_y))

        # 내 선체 형상
        pygame.draw.circle(self.pov_surf, (20, 60, 180), (pcx, pcy), int(env.boat_radius * scale_r))
        pygame.draw.line(self.pov_surf, (255, 255, 255), (pcx, pcy), (pcx, pcy - 16), 2)
        
        txt_surf = self.font.render("LiDAR View", True, (255, 255, 255))
        self.pov_surf.blit(txt_surf, (10, pov_h - txt_surf.get_height() - 5))
        env.screen.blit(self.pov_surf, (p1_x, p_y))

        # --- 2. 180도 라이다 각도 세로 게이지 뷰 (LiDAR Gauge View) ---
        cam_w, cam_h = 320, 220
        self.cam_surf.fill((10, 20, 35, 240))
        pygame.draw.rect(self.cam_surf, (0, 180, 255), (0, 0, cam_w, cam_h), 2)

        n_slices = 180
        slice_angles = np.linspace(-np.pi/2, np.pi/2, n_slices)

        # 180개 각도 세로 직사각형 게이지 렌더링
        for i in range(n_slices):
            ang = slice_angles[i]
            idx = int((ang + np.pi) / (2 * np.pi) * len(env.rel_angles)) % len(env.rel_angles)
            hp = hits[idx] if idx < len(hits) else None
            
            if hp is not None:
                hdx = hp[0] - bx
                hdy = hp[1] - by
                d = math.hypot(hdx, hdy)
            else:
                d = env.lidar_range

            x1 = int(i * cam_w / n_slices)
            x2 = int((i + 1) * cam_w / n_slices)
            w_s = max(1, x2 - x1)

            if d < env.lidar_range:
                if d < 70:
                    color = (230, 60, 50)
                elif d < 140:
                    color = (240, 160, 40)
                elif d < 220:
                    color = (210, 210, 50)
                else:
                    color = (40, 170, 160)
                
                pygame.draw.rect(self.cam_surf, color, (x1, 2, w_s, cam_h - 4))

        # 웨이포인트 및 최종 목표 지점 수직 오버레이 신호선
        marker_objs = []
        show_1st = getattr(env, 'show_1st_path', getattr(env, 'show_paths', True))
        show_2nd = getattr(env, 'show_2nd_path', getattr(env, 'show_paths', True))
        if show_1st and env.current_wp is not None:
            dx_w = env.current_wp["pos"][0] - bx; dy_w = env.current_wp["pos"][1] - by
            lf_w = dx_w * f_vec[0] + dy_w * f_vec[1]; lr_w = dx_w * r_vec[0] + dy_w * r_vec[1]
            marker_objs.append(('wp1', lf_w, lr_w))

        if show_2nd and env.next_wp is not None:
            dx_w2 = env.next_wp["pos"][0] - bx; dy_w2 = env.next_wp["pos"][1] - by
            lf_w2 = dx_w2 * f_vec[0] + dy_w2 * f_vec[1]; lr_w2 = dx_w2 * r_vec[0] + dy_w2 * r_vec[1]
            marker_objs.append(('wp2', lf_w2, lr_w2))

        marker_objs.append(('target', lf_t, lr_t))

        for obj_type, lf, lr in marker_objs:
            ang_obj = math.atan2(lr, lf)
            if -math.pi/2 <= ang_obj <= math.pi/2:
                s_idx = int((ang_obj + math.pi/2) / math.pi * n_slices)
                mx = int(s_idx * cam_w / n_slices)
                
                if obj_type == 'wp1':
                    pygame.draw.line(self.cam_surf, (0, 255, 220), (mx, 0), (mx, cam_h - 26), 2)
                    pygame.draw.circle(self.cam_surf, (0, 255, 220), (mx, 55), 6)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 55), 2)
                    lbl_wp1 = self.micro_font.render("WP1", True, (0, 255, 220))
                    tx = mx + 8 if mx + 32 < cam_w else mx - lbl_wp1.get_width() - 8
                    self.cam_surf.blit(lbl_wp1, (tx, 49))
                elif obj_type == 'wp2':
                    pygame.draw.line(self.cam_surf, (200, 100, 255), (mx, 0), (mx, cam_h - 26), 2)
                    pygame.draw.circle(self.cam_surf, (200, 100, 255), (mx, 90), 6)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 90), 2)
                    lbl_wp2 = self.micro_font.render("WP2", True, (200, 100, 255))
                    tx = mx + 8 if mx + 32 < cam_w else mx - lbl_wp2.get_width() - 8
                    self.cam_surf.blit(lbl_wp2, (tx, 84))
                elif obj_type == 'target':
                    pygame.draw.line(self.cam_surf, (20, 250, 80), (mx, 0), (mx, cam_h - 26), 3)
                    pygame.draw.circle(self.cam_surf, (20, 250, 80), (mx, 125), 7)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (mx, 125), 3)
                    lbl_tgt = self.micro_font.render("Target", True, (20, 250, 80))
                    tx = mx + 9 if mx + 42 < cam_w else mx - lbl_tgt.get_width() - 8
                    self.cam_surf.blit(lbl_tgt, (tx, 119))

        # 패널 타이틀
        self.cam_surf.blit(self.font.render("LiDAR Gauge View", True, (255, 255, 255)), (10, 8))

        is_lt = getattr(env, 'linetrace_mode', False)

        if not is_lt:
            # 현재 고려 중인 모든 갭의 개수 표시 HUD 토글 버튼 (전방 180도 기준)
            front_gaps_count = 0
            if getattr(env, 'all_gaps', None):
                for g in env.all_gaps:
                    mid = g["pos"]
                    if (mid[0] - bx) * f_vec[0] + (mid[1] - by) * f_vec[1] >= 0:
                        front_gaps_count += 1
            total_gaps = front_gaps_count
            show_all = getattr(env, 'show_all_gaps', False)

            # 자리수 변화(1자리, 2자리)에 관계없이 버튼 크기 고정 (Fixed Width)
            btn_w = 94
            btn_h = 24
            bx_pos = cam_w - btn_w - 10
            by_pos = 7
            badge_rect = pygame.Rect(bx_pos, by_pos, btn_w, btn_h)
            # 화면 절대 좌표로 버튼 클릭 영역 저장 (환경 handle_click 연동)
            env.gaps_btn_rect = pygame.Rect(p2_x + bx_pos, p_y + by_pos, btn_w, btn_h)

            mpos = pygame.mouse.get_pos()
            is_hover = env.gaps_btn_rect.collidepoint(mpos)

            if getattr(env, 'current_wp', None) is not None and total_gaps > 0:
                gap_txt = f"Gaps: {total_gaps:02d}" if total_gaps < 100 else f"Gaps: {total_gaps}"
                
                if show_all:
                    # 활성화(ON) 상태: 갭 중간점 표시 켜짐 - 네온 시안 하이라이트
                    bg_col = (18, 55, 95, 240) if is_hover else (14, 42, 75, 230)
                    border_col = (0, 255, 255)
                    text_col = (0, 255, 255)
                    border_w = 2
                else:
                    # 비활성화(OFF) 상태: 차분한 다크 블루
                    bg_col = (20, 38, 62, 230) if is_hover else (12, 24, 42, 210)
                    border_col = (0, 200, 255) if is_hover else (0, 130, 180, 180)
                    text_col = (220, 245, 255) if is_hover else (145, 190, 220)
                    border_w = 1

                pygame.draw.rect(self.cam_surf, bg_col, badge_rect, border_radius=4)
                pygame.draw.rect(self.cam_surf, border_col, badge_rect, border_w, border_radius=4)

                # 버튼 내부 수직/수평 완벽한 정중앙 정렬
                lbl_gap = self.small_font.render(gap_txt, True, text_col)
                text_rect = lbl_gap.get_rect(center=badge_rect.center)
                self.cam_surf.blit(lbl_gap, text_rect)

            elif getattr(env, 'current_wp', None) is None:
                dir_txt = "Direct"
                bg_col = (14, 48, 30, 240) if is_hover else (10, 36, 22, 225)
                border_col = (20, 255, 90) if is_hover else (20, 220, 80)
                text_col = (30, 255, 100) if is_hover else (20, 250, 80)

                pygame.draw.rect(self.cam_surf, bg_col, badge_rect, border_radius=4)
                pygame.draw.rect(self.cam_surf, border_col, badge_rect, 1, border_radius=4)

                # 버튼 내부 수직/수평 완벽한 정중앙 정렬
                lbl_dir = self.small_font.render(dir_txt, True, text_col)
                text_rect = lbl_dir.get_rect(center=badge_rect.center)
                self.cam_surf.blit(lbl_dir, text_rect)
        else:
            env.gaps_btn_rect = None

        # 모든 갭 표시 활성화 시 각도 바로 위쪽 좌우 일렬 선상에 각 갭 및 웨이포인트(WP1, WP2)의 각도 위치를 점으로 표출
        legend_bar_y = cam_h - 25
        sample_lh = self.small_font.render("0°", True, (0, 0, 0)).get_height()
        dot_y = legend_bar_y - sample_lh - 9

        if not is_lt and getattr(env, 'show_all_gaps', False):
            # 좌우 일렬 수평 가이드선
            pygame.draw.line(self.cam_surf, (0, 180, 240, 90), (8, dot_y), (cam_w - 8, dot_y), 1)

            # 1. 탐지된 모든 장애물 쌍 틈새(갭) 각도 점 (번호 텍스트 없이 깔끔한 점으로 표출)
            if getattr(env, 'all_gaps', None):
                for g in env.all_gaps:
                    mid = g["pos"]
                    dx_g = mid[0] - bx
                    dy_g = mid[1] - by
                    lf_g = dx_g * f_vec[0] + dy_g * f_vec[1]
                    lr_g = dx_g * r_vec[0] + dy_g * r_vec[1]

                    ang_g = math.atan2(lr_g, lf_g)
                    if -math.pi / 2 <= ang_g <= math.pi / 2:
                        s_idx = int((ang_g + math.pi / 2) / math.pi * n_slices)
                        gx = int(s_idx * cam_w / n_slices)
                        gx = max(6, min(cam_w - 6, gx))

                        # 갭 위치 점 (섀도우 + 시안 링 + 화이트 코어)
                        pygame.draw.circle(self.cam_surf, (10, 20, 35), (gx, dot_y), 4)
                        pygame.draw.circle(self.cam_surf, (0, 220, 255), (gx, dot_y), 3)
                        pygame.draw.circle(self.cam_surf, (255, 255, 255), (gx, dot_y), 1)

            # 2. 1차 웨이포인트(WP1) 각도 점 표출 (제외하지 않고 반드시 포함)
            if getattr(env, 'current_wp', None) is not None:
                dx_w1 = env.current_wp["pos"][0] - bx
                dy_w1 = env.current_wp["pos"][1] - by
                lf_w1 = dx_w1 * f_vec[0] + dy_w1 * f_vec[1]
                lr_w1 = dx_w1 * r_vec[0] + dy_w1 * r_vec[1]
                ang_w1 = math.atan2(lr_w1, lf_w1)
                if -math.pi / 2 <= ang_w1 <= math.pi / 2:
                    s_idx1 = int((ang_w1 + math.pi / 2) / math.pi * n_slices)
                    gx1 = max(6, min(cam_w - 6, int(s_idx1 * cam_w / n_slices)))
                    # 1차 웨이포인트 점 (네온 시안 강조)
                    pygame.draw.circle(self.cam_surf, (10, 20, 35), (gx1, dot_y), 6)
                    pygame.draw.circle(self.cam_surf, (0, 255, 220), (gx1, dot_y), 5)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (gx1, dot_y), 2)

            # 3. 2차 웨이포인트(WP2) 각도 점 표출 (제외하지 않고 반드시 포함)
            if getattr(env, 'next_wp', None) is not None:
                dx_w2 = env.next_wp["pos"][0] - bx
                dy_w2 = env.next_wp["pos"][1] - by
                lf_w2 = dx_w2 * f_vec[0] + dy_w2 * f_vec[1]
                lr_w2 = dx_w2 * r_vec[0] + dy_w2 * r_vec[1]
                ang_w2 = math.atan2(lr_w2, lf_w2)
                if -math.pi / 2 <= ang_w2 <= math.pi / 2:
                    s_idx2 = int((ang_w2 + math.pi / 2) / math.pi * n_slices)
                    gx2 = max(6, min(cam_w - 6, int(s_idx2 * cam_w / n_slices)))
                    # 2차 웨이포인트 점 (퍼플/마젠타 강조)
                    pygame.draw.circle(self.cam_surf, (10, 20, 35), (gx2, dot_y), 6)
                    pygame.draw.circle(self.cam_surf, (200, 100, 255), (gx2, dot_y), 5)
                    pygame.draw.circle(self.cam_surf, (255, 255, 255), (gx2, dot_y), 2)

        # 전방 180도 화각 표시를 위한 하단 각도 단위 텍스트 (0°, 90°, 180°) - 버튼 박스 없이 숫자만 표시
        angles_spec = [(0, "0°", 8), (90, "90°", cam_w // 2), (180, "180°", cam_w - 8)]
        for deg, txt, anchor_x in angles_spec:
            lbl_ang = self.small_font.render(txt, True, (0, 180, 240))
            lbl_shadow = self.small_font.render(txt, True, (10, 20, 35))
            lw, lh = lbl_ang.get_width(), lbl_ang.get_height()
            if deg == 0:
                bx_pos = anchor_x
            elif deg == 180:
                bx_pos = anchor_x - lw
            else:
                bx_pos = anchor_x - lw // 2
            by_pos = legend_bar_y - lh - 4
            self.cam_surf.blit(lbl_shadow, (bx_pos + 1, by_pos + 1))
            self.cam_surf.blit(lbl_ang, (bx_pos, by_pos))

        # 패널 하단 거리 색상 범례 도킹 바 (Legend HUD Bar)
        # 50px = 1m 기준: <70px (~1.4m), 70~140px (~2.8m), 140~220px (~4.4m), >220px (>4.4m)
        pygame.draw.rect(self.cam_surf, (8, 16, 28, 235), (0, legend_bar_y, cam_w, 25))
        pygame.draw.line(self.cam_surf, (0, 140, 210), (0, legend_bar_y), (cam_w, legend_bar_y), 1)

        legend_items = [
            ((230, 60, 50), "<1.4m"),
            ((240, 160, 40), "<2.8m"),
            ((210, 210, 50), "<4.4m"),
            ((40, 170, 160), ">4.4m")
        ]
        
        # 총 4개 아이템을 패널 가로폭(320px)에 균등 정렬 배치 (small_font 18px로 시인성 향상)
        col_spacing = 76
        start_x = (cam_w - (col_spacing * 4 - 6)) // 2
        for i, (col, txt) in enumerate(legend_items):
            ix = start_x + i * col_spacing
            pygame.draw.rect(self.cam_surf, col, (ix, legend_bar_y + 7, 10, 10), border_radius=2)
            ltxt = self.small_font.render(txt, True, (225, 238, 255))
            self.cam_surf.blit(ltxt, (ix + 13, legend_bar_y + 5))

        env.screen.blit(self.cam_surf, (p2_x, p_y))

        # --- 3. 실시간 하드웨어 가속 ModernGL 3D 엔진 뷰포트 & 2D 화면 스왑 슬롯 (버튼 없음) ---
        env.cam_panel_btn_rect = None
        if getattr(self, 'engine_3d', None) is not None:
            try:
                if getattr(env, 'fullscreen_3d', False):
                    # 3D 전체화면 활성화 시: 하단 320x220 슬롯에 가로세로 비율(20:7)을 엄격히 고정한 2D 전체 맵 표출 (화면상 장애물 1:1 일치)
                    panel_surf = pygame.Surface((320, 220))
                    panel_surf.fill((8, 18, 30))
                    pygame.draw.rect(panel_surf, (0, 180, 255), (0, 0, 320, 220), 2)
                    
                    # 상단 라벨 (화면 크기 생략, 간략한 패널 이름 표기: "2D MAP")
                    t_mini = self.font.render("2D MAP", True, (240, 245, 255))
                    panel_surf.blit(t_mini, (10, 8))
                    
                    # 20:7 고정 비율 스케일링: 가로 316px, 세로 110px (상하 왜곡/잘림 완벽 방지)
                    mini_w, mini_h = 316, 110
                    mini_2d = pygame.transform.smoothscale(self.world_2d_surf, (mini_w, mini_h))
                    map_x, map_y = 2, 34
                    panel_surf.blit(mini_2d, (map_x, map_y))
                    pygame.draw.rect(panel_surf, (0, 140, 210), (map_x - 1, map_y - 1, mini_w + 2, mini_h + 2), 1)
                    
                    # 하단 엔진 정보 텍스트 (3D 엔진 텍스트와 동일한 14px 폰트)
                    lbl_eng = self.engine_info_font.render("Pygame 2D Engine", True, (0, 210, 255))
                    panel_surf.blit(lbl_eng, lbl_eng.get_rect(center=(160, 178)))
                    
                    env.screen.blit(panel_surf, (p3_x, p_y))
                else:
                    # 기본 2D 모드: 하단 슬롯에 320x220 3D 뷰포트 표출 (패널 상에 어떤 버튼도 배치하지 않음)
                    surf_3d = self.engine_3d.render(env, hits, 320, 220)
                    env.screen.blit(surf_3d, (p3_x, p_y))
            except Exception as e:
                print(f"[Warning] 3D render failed: {e}")

        # --- 4. 실시간 베지어 곡선 & 곡률 프로파일 그래프 & 5. 가중치 패널 (라인트레이싱 모드에서는 완전 제외) ---
        if not getattr(env, 'linetrace_mode', False):
            self._draw_bezier_profile(p4_x, p_y)
            self._draw_weight_breakdown(p5_x, p_y)

    def _draw_bezier_profile(self, x=None, y=None):
        """우측 하단: 실시간 3차 베지어 곡선(Cubic S-Curve) 2D 궤적 그래프 (X: 전진거리, Y: 좌우편차 - 상하반전 및 3차 수식 표기)"""
        env = self.env
        bw, bh = 190, 220
        surf = self.bezier_surf
        dest_x = x if x is not None else 1420
        dest_y = y if y is not None else (env.sim_h + 35)
        surf.fill((10, 22, 38, 240))
        pygame.draw.rect(surf, (0, 180, 255), (0, 0, bw, bh), 2)
        
        # 타이틀
        surf.blit(self.bold_font.render("Bezier Curve 2D", True, (255, 255, 255)), (10, 8))
        
        # Y축 단위 표기
        surf.blit(self.small_font.render("Y(m)", True, (140, 190, 240)), (4, 30))
        
        # 그래프 영역
        gx, gy, gw, gh = 36, 42, 144, 94
        pygame.draw.rect(surf, (15, 30, 50), (gx, gy, gw, gh))
        pygame.draw.rect(surf, (40, 80, 120), (gx, gy, gw, gh), 1)
        
        y_center = gy + gh // 2
        
        path = getattr(env, 'bezier_path', None)
        bx, by = env.boat_pos
        h = env.boat_heading
        ch, sh = math.cos(h), math.sin(h)
        
        if not hasattr(self, 'scale_x_max'): self.scale_x_max = 4.0; self.scale_y_max = 2.0
        
        if path is not None and len(path) >= 4:
            pts = np.array(path)
            diffs = pts - env.boat_pos
            # 선박 기준 로컬 좌표계 변환 (X: 전방 거리, Y: 좌/우 편차 거리)
            x_loc = diffs[:, 0] * ch + diffs[:, 1] * sh
            y_loc = -diffs[:, 0] * sh + diffs[:, 1] * ch
            
            xm = x_loc / 50.0  # 미터 단위
            ym = y_loc / 50.0  # 미터 단위
            
            target_x_max = max(2.0, float(np.max(xm)) * 1.15)
            target_y_max = max(1.2, float(np.max(np.abs(ym))) * 1.35)
            
            self.scale_x_max = self.scale_x_max * 0.90 + target_x_max * 0.10
            self.scale_y_max = self.scale_y_max * 0.90 + target_y_max * 0.10
            
            sx_max = self.scale_x_max
            sy_max = self.scale_y_max
            
            # Y축 눈금선 및 수치 표기 (상단 +Y, 하단 -Y)
            surf.blit(self.small_font.render(f"+{sy_max:.1f}", True, (160, 200, 230)), (2, gy - 2))
            surf.blit(self.small_font.render(" 0.0", True, (0, 200, 255)), (2, y_center - 6))
            surf.blit(self.small_font.render(f"-{sy_max:.1f}", True, (160, 200, 230)), (4, gy + gh - 8))
            
            # 배경 중심 기준선 (Y = 0: 전방 직진선)
            pygame.draw.line(surf, (0, 140, 180), (gx, y_center), (gx + gw, y_center), 1)
            pygame.draw.line(surf, (25, 55, 80), (gx + gw//2, gy), (gx + gw//2, gy + gh), 1)
            
            # 3차 베지어 곡선 궤적 포인트 생성 (+ y_val 방향으로 위쪽 매핑)
            plot_pts = []
            for x_val, y_val in zip(xm, ym):
                px = int(gx + (x_val / max(0.1, sx_max)) * (gw - 12))
                py = int(y_center - (y_val / max(0.1, sy_max)) * (gh * 0.44))
                px = max(gx, min(gx + gw, px))
                py = max(gy, min(gy + gh, py))
                plot_pts.append((px, py))
                
            if len(plot_pts) >= 2:
                # 3차 함수 곡선 본체 (빛나는 시안색)
                pygame.draw.lines(surf, (50, 225, 255), False, plot_pts, 3)
                
            # 시작점 P0 (선박 원점)
            p0_x, p0_y = plot_pts[0]
            pygame.draw.circle(surf, (0, 255, 200), (p0_x, p0_y), 4)
            pygame.draw.circle(surf, (255, 255, 255), (p0_x, p0_y), 2)
            
            # 목표점 P3 (골/웨이포인트)
            p3_x, p3_y = plot_pts[-1]
            pygame.draw.circle(surf, (255, 160, 40), (p3_x, p3_y), 5)
            pygame.draw.circle(surf, (255, 255, 255), (p3_x, p3_y), 2)
            
            path_len_m = float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))) / 50.0
            end_ym = ym[-1]
            
            # 실시간 3차 함수 계수 고속 계산 (y(0) = 0 제약 정규방정식 연산)
            if len(xm) >= 4 and (np.max(xm) - np.min(xm)) > 0.3:
                A = np.stack([xm**3, xm**2, xm], axis=1)
                ATA = A.T @ A
                ATA[0, 0] += 1e-4; ATA[1, 1] += 1e-4; ATA[2, 2] += 1e-4
                a, b, c = np.linalg.solve(ATA, A.T @ ym)
            else:
                a, b, c = 0.0, 0.0, 0.0
                
            if not hasattr(self, 'poly_coeffs'): self.poly_coeffs = (0.0, 0.0, 0.0)
            self.poly_coeffs = (
                self.poly_coeffs[0] * 0.80 + a * 0.20,
                self.poly_coeffs[1] * 0.80 + b * 0.20,
                self.poly_coeffs[2] * 0.80 + c * 0.20
            )
            sa, sb, sc = self.poly_coeffs
            formula_txt = f"y = {sa:+.3f}x³ {sb:+.2f}x² {sc:+.2f}x"
        else:
            sy_max = self.scale_y_max
            sx_max = self.scale_x_max
            surf.blit(self.small_font.render(f"+{sy_max:.1f}", True, (160, 200, 230)), (2, gy - 2))
            surf.blit(self.small_font.render(" 0.0", True, (0, 200, 255)), (2, y_center - 6))
            surf.blit(self.small_font.render(f"-{sy_max:.1f}", True, (160, 200, 230)), (4, gy + gh - 8))
            pygame.draw.line(surf, (0, 140, 180), (gx, y_center), (gx + gw, y_center), 1)
            pygame.draw.line(surf, (50, 225, 255), (gx, y_center), (gx + gw - 20, y_center), 3)
            path_len_m = 0.0
            end_ym = 0.0
            formula_txt = "y = +0.000x³ +0.00x² +0.00x"
            
        # X축 거리 눈금 및 수치 표기 (단위: m)
        surf.blit(self.small_font.render("0m", True, (140, 180, 220)), (gx, gy + gh + 2))
        mid_m_txt = f"{sx_max * 0.5:.1f}m"
        surf.blit(self.small_font.render(mid_m_txt, True, (140, 180, 220)), (gx + gw//2 - 10, gy + gh + 2))
        end_m_txt = f"{sx_max:.1f}m"
        surf.blit(self.small_font.render(end_m_txt, True, (140, 180, 220)), (gx + gw - len(end_m_txt)*7, gy + gh + 2))
        
        # 하단 실시간 3차 다항식 수식 표기 (실시간 수치 적용)
        surf.blit(self.small_font.render(formula_txt, True, (255, 220, 60)), (8, 164))
        
        # 하단 실시간 궤적 수치
        surf.blit(self.small_font.render(f"Len: {path_len_m:.1f}m | Lat Dev: {end_ym:+.1f}m", True, (220, 235, 255)), (8, 188))
        
        env.screen.blit(surf, (dest_x, dest_y))

    def _draw_weight_breakdown(self, x=None, y=None):
        """우측 하단: 웨이포인트 우선순위 가중치 비율 분포 막대 게이지"""
        env = self.env
        ww, wh = 190, 220
        surf = self.weights_surf
        dest_x = x if x is not None else 1630
        dest_y = y if y is not None else (env.sim_h + 35)
        surf.fill((10, 22, 38, 240))
        pygame.draw.rect(surf, (0, 180, 255), (0, 0, ww, wh), 2)
        
        # 타이틀
        surf.blit(self.bold_font.render("WP Score Weights", True, (255, 255, 255)), (10, 10))
        
        wp = getattr(env, 'current_wp', None)
        factors = wp.get('factors', None) if wp is not None else None
        
        FACTOR_ITEMS = [
            ("Align", (0, 230, 255)),      # GPS Alignment (align_exp)
            ("Heading", (170, 130, 255)),  # Boat Heading Alignment (heading_exp)
            ("Forward", (40, 225, 120)),   # Progress toward GPS (fwd_exp)
            ("Width", (255, 215, 40)),     # Gate Aperture Width (width_exp)
            ("Clear", (50, 220, 200)),     # Path Obstacle Clearance / Density (clear_exp)
            ("Perpend", (255, 140, 40))    # Vertical Orthogonality (perp_exp)
        ]
        
        bar_x = 64
        bar_w = 78
        bar_h = 8
        
        if wp is not None and factors is not None:
            # 가중치 w와 점수 raw를 곱하여 실제 기여 점수 산출 (w가 0이면 0% 표출)
            weighted_vals = []
            for k, _ in FACTOR_ITEMS:
                item = factors.get(k)
                if isinstance(item, dict):
                    w = float(item.get("w", 0.0))
                    raw = float(item.get("raw", 0.0))
                    weighted_vals.append(w * raw)
                elif isinstance(item, (int, float)):
                    weighted_vals.append(float(item))
                else:
                    weighted_vals.append(0.0)
                    
            tot = sum(weighted_vals)
            ratios = [v / tot if tot > 1e-6 else 0.0 for v in weighted_vals]
            
            for i, (name, col) in enumerate(FACTOR_ITEMS):
                y_pos = 36 + i * 29
                lbl = self.small_font.render(name, True, (210, 225, 240))
                surf.blit(lbl, (10, y_pos - 2))
                
                pygame.draw.rect(surf, (20, 40, 65), (bar_x, y_pos, bar_w, bar_h), border_radius=3)
                
                if ratios[i] > 0.001:
                    fill_w = max(2, int(ratios[i] * bar_w * 2.2))
                    fill_w = min(bar_w, fill_w)
                    pygame.draw.rect(surf, col, (bar_x, y_pos, fill_w, bar_h), border_radius=3)
                    
                pct = int(round(ratios[i] * 100))
                txt_pct = self.small_font.render(f"{pct}%", True, col if pct > 0 else (120, 140, 160))
                surf.blit(txt_pct, (bar_x + bar_w + 6, y_pos - 2))
        else:
            # 웨이포인트가 없을 때 (대기 상태 UI 유지)
            for i, (name, col) in enumerate(FACTOR_ITEMS):
                y_pos = 36 + i * 29
                lbl = self.small_font.render(name, True, (120, 145, 170))
                surf.blit(lbl, (10, y_pos - 2))
                
                pygame.draw.rect(surf, (20, 40, 65), (bar_x, y_pos, bar_w, bar_h), border_radius=3)
                
                txt_pct = self.small_font.render("--%", True, (90, 120, 150))
                surf.blit(txt_pct, (bar_x + bar_w + 6, y_pos - 2))
            
        env.screen.blit(surf, (dest_x, dest_y))

    def _draw_telemetry(self):
        """우상단 실시간 텔레메트리 HUD"""
        env = self.env
        is_manual = getattr(env, 'manual_mode', False)
        hud_w, hud_h = 210, 110
        hud_x = env.w - hud_w - 15
        hud_y = 12
        
        hud_surf = self.hud_surf
        hud_surf.fill((10, 20, 40, 190))
        pygame.draw.rect(hud_surf, (0, 160, 230), (0, 0, hud_w, hud_h), 2)
        
        # 모드 표시
        is_paused = getattr(env, 'paused', False)
        em = getattr(env, 'emergency_mode', False)
        is_lt = getattr(env, 'linetrace_mode', False)
        has_wp = env.current_wp is not None
        if is_paused:
            mode_txt = self.bold_font.render("PAUSED", True, (255, 140, 20))
        elif is_manual:
            mode_txt = self.bold_font.render("MANUAL RC", True, (255, 200, 30))
        elif is_lt:
            mode_txt = self.bold_font.render("LINE-TRACE", True, (255, 40, 195))
        elif em:
            mode_txt = self.bold_font.render("AVOIDING", True, (255, 80, 60))
        elif has_wp:
            mode_txt = self.bold_font.render("GAP PASS", True, (0, 255, 220))
        else:
            mode_txt = self.bold_font.render("CRUISING", True, (50, 230, 120))
        hud_surf.blit(mode_txt, (10, 6))
        
        # 속도
        speed = float(np.linalg.norm(env.boat_vel))
        speed_knots = speed * 0.9
        spd_txt = self.small_font.render(f"Speed: {speed_knots:.1f} kt", True, (220, 235, 255))
        hud_surf.blit(spd_txt, (10, 32))
        
        # 속도 바 (실제 최고 속도 45.0 px/s 기준 정밀 스케일링)
        bar_w = 125
        pygame.draw.rect(hud_surf, (30, 50, 70), (10, 48, bar_w, 7))
        speed_ratio = float(np.clip(speed / 45.0, 0.0, 1.0))
        fill_w = int(speed_ratio * bar_w)
        bar_color = (255, 80, 60) if em else (0, 200, 100)
        pygame.draw.rect(hud_surf, bar_color, (10, 48, fill_w, 7))

        # 목표 거리 (50px = 1m 기준 미터 단위 변환)
        d2t = float(np.linalg.norm(env.target - env.boat_pos))
        d2t_m = d2t / 50.0

        if is_manual:
            # 수동 조종 모드 전용 텔레메트리 (도달시간, 누적회전각, 목표거리 - 영문 표기)
            elapsed_sec = time.time() - getattr(env, 'manual_start_time', time.time())
            c_turn = getattr(env, 'manual_cum_turn', 0.0)

            time_txt = self.small_font.render(f"Time: {elapsed_sec:.1f} s", True, (220, 235, 255))
            hud_surf.blit(time_txt, (10, 60))

            trn_txt = self.small_font.render(f"Turn: {int(c_turn)}\u00b0", True, (220, 235, 255))
            hud_surf.blit(trn_txt, (10, 76))

            d2t_txt = self.small_font.render(f"Target: {d2t_m:.1f} m", True, (50, 230, 120))
            hud_surf.blit(d2t_txt, (10, 92))
        else:
            # 자율운항 모드 전용 텔레메트리 (조타각, 실시간 헤딩, 목표 거리)
            steer_val = getattr(env, 'prev_steer', 0)
            steer_txt = self.small_font.render(f"Steer: {steer_val:+.2f}", True, (220, 235, 255))
            hud_surf.blit(steer_txt, (10, 60))

            tgt_h = getattr(env, 'heading_target', env.boat_heading)
            hdg_deg = (math.degrees(tgt_h) + 90) % 360
            hdg_txt = self.small_font.render(f"Heading: {hdg_deg:.0f}\u00b0", True, (220, 235, 255))
            hud_surf.blit(hdg_txt, (10, 76))

            d2t_txt = self.small_font.render(f"Target: {d2t_m:.1f} m", True, (50, 230, 120))
            hud_surf.blit(d2t_txt, (10, 92))
        
        env.screen.blit(hud_surf, (hud_x, hud_y))

        # 충돌 발생 시 화면 중앙 상단에 실시간 충돌 횟수 경고 배너 표출
        if is_manual and getattr(env, 'manual_collision_flash', 0) > 0:
            c_count = getattr(env, 'manual_collisions', 0)
            alert_w, alert_h = 320, 44
            alert_x = (env.w - alert_w) // 2
            alert_y = 52
            alert_surf = pygame.Surface((alert_w, alert_h), pygame.SRCALPHA)
            pygame.draw.rect(alert_surf, (85, 14, 22, 235), (0, 0, alert_w, alert_h), border_radius=8)
            pygame.draw.rect(alert_surf, (255, 65, 75, 245), (0, 0, alert_w, alert_h), 2, border_radius=8)
            txt_surf = self.bold_font.render(f"COLLISION DETECTED (#{c_count})", True, (255, 240, 240))
            alert_surf.blit(txt_surf, txt_surf.get_rect(center=(alert_w // 2, alert_h // 2)))
            env.screen.blit(alert_surf, (alert_x, alert_y))

        # --- 우측 상단 텔레메트리 HUD 하단: 조이스틱 버튼 / 눈 깜빡임 버튼 / 새 에피소드 재시작 버튼 ---
        btn_y = hud_y + hud_h + 8
        rc_btn_w, rc_btn_h = 42, 38
        rc_x = hud_x + hud_w - rc_btn_w  # 우측 끝으로 밀착 정렬
        rc_rect = pygame.Rect(rc_x, btn_y, rc_btn_w, rc_btn_h)
        env.rc_btn_rect = rc_rect

        mpos = pygame.mouse.get_pos()
        is_hover = rc_rect.collidepoint(mpos)

        rc_surf = pygame.Surface((rc_btn_w, rc_btn_h), pygame.SRCALPHA)

        # 조이스틱 버튼 스타일 (저채도 미니멀 다크 슬레이트)
        if is_manual:
            bg_col = (25, 33, 44, 190) if is_hover else (18, 25, 34, 160)
            border_col = (75, 95, 120, 200) if is_hover else (50, 68, 88, 160)
            border_w = 1
            ball_col = (110, 130, 150)
            accent_col = (70, 92, 115)
        else:
            bg_col = (22, 28, 38, 170) if is_hover else (14, 20, 28, 140)
            border_col = (60, 78, 100, 180) if is_hover else (38, 50, 66, 130)
            border_w = 1
            ball_col = (85, 102, 120)
            accent_col = (55, 72, 90)

        pygame.draw.rect(rc_surf, bg_col, (0, 0, rc_btn_w, rc_btn_h), border_radius=6)
        pygame.draw.rect(rc_surf, border_col, (0, 0, rc_btn_w, rc_btn_h), border_w, border_radius=6)

        # 조이스틱 정밀 벡터 아이콘
        cx, cy = 21, 19
        pygame.draw.ellipse(rc_surf, (15, 20, 28), (cx - 10, cy + 4, 20, 9))
        pygame.draw.ellipse(rc_surf, accent_col, (cx - 10, cy + 4, 20, 9), 1)
        pygame.draw.ellipse(rc_surf, (10, 14, 20), (cx - 5, cy + 5, 10, 5))
        pygame.draw.line(rc_surf, (110, 125, 140), (cx, cy + 5), (cx - 2, cy - 4), 2)
        pygame.draw.circle(rc_surf, ball_col, (cx - 2, cy - 6), 5)
        pygame.draw.circle(rc_surf, (160, 180, 200), (cx - 3, cy - 7), 1)

        if is_manual:
            pygame.draw.circle(rc_surf, (120, 150, 180), (rc_btn_w - 6, 6), 2)

        env.screen.blit(rc_surf, (rc_x, btn_y))

        # --- RC 모드 진입 시에만 그 옆(좌측)에 나타나는 눈 깜빡임 버튼 및 새 에피소드 재시작 버튼 ---
        if is_manual:
            # [버튼 1] 눈 깜빡임(블라인드 시연 모드) 토글 버튼
            eye_btn_w, eye_btn_h = 42, 38
            eye_x = rc_x - eye_btn_w - 6  # 조이스틱 버튼 좌측
            eye_rect = pygame.Rect(eye_x, btn_y, eye_btn_w, eye_btn_h)
            env.blind_btn_rect = eye_rect

            is_eye_hover = eye_rect.collidepoint(mpos)
            is_blind = getattr(env, 'blind_mode', False)

            eye_surf = pygame.Surface((eye_btn_w, eye_btn_h), pygame.SRCALPHA)

            if is_blind:
                eye_bg = (28, 30, 40, 190) if is_eye_hover else (20, 24, 32, 160)
                eye_border = (85, 95, 115, 200) if is_eye_hover else (55, 68, 85, 160)
                eye_accent = (110, 125, 145)
                iris_col = (90, 105, 125)
            else:
                eye_bg = (22, 28, 38, 170) if is_eye_hover else (14, 20, 28, 140)
                eye_border = (60, 78, 100, 180) if is_eye_hover else (38, 50, 66, 130)
                eye_accent = (75, 95, 115)
                iris_col = (65, 82, 102)

            pygame.draw.rect(eye_surf, eye_bg, (0, 0, eye_btn_w, eye_btn_h), border_radius=6)
            pygame.draw.rect(eye_surf, eye_border, (0, 0, eye_btn_w, eye_btn_h), 1, border_radius=6)

            # 눈(Eye) 정밀 벡터 아이콘
            ecx, ecy = 21, 19
            pygame.draw.arc(eye_surf, eye_accent, (ecx - 11, ecy - 9, 22, 16), 0.18 * math.pi, 0.82 * math.pi, 1)
            pygame.draw.arc(eye_surf, eye_accent, (ecx - 11, ecy - 9, 22, 16), 1.18 * math.pi, 1.82 * math.pi, 1)
            pygame.draw.circle(eye_surf, iris_col, (ecx, ecy), 3)
            pygame.draw.circle(eye_surf, (12, 16, 24), (ecx, ecy), 1)

            if is_blind:
                pygame.draw.line(eye_surf, (130, 100, 100), (ecx - 8, ecy - 6), (ecx + 8, ecy + 6), 1)
                pygame.draw.circle(eye_surf, (130, 145, 165), (eye_btn_w - 6, 6), 2)

            env.screen.blit(eye_surf, (eye_x, btn_y))

            # [버튼 2] 새로운 에피소드로 재시작(Restart) 버튼 (눈 깜빡임 버튼 좌측)
            rst_btn_w, rst_btn_h = 42, 38
            rst_x = eye_x - rst_btn_w - 6
            rst_rect = pygame.Rect(rst_x, btn_y, rst_btn_w, rst_btn_h)
            env.restart_btn_rect = rst_rect

            is_rst_hover = rst_rect.collidepoint(mpos)
            rst_surf = pygame.Surface((rst_btn_w, rst_btn_h), pygame.SRCALPHA)

            rst_bg = (24, 30, 42, 190) if is_rst_hover else (16, 22, 32, 160)
            rst_border = (70, 90, 115, 200) if is_rst_hover else (45, 60, 80, 160)
            rst_accent = (175, 200, 230) if is_rst_hover else (105, 125, 148)

            pygame.draw.rect(rst_surf, rst_bg, (0, 0, rst_btn_w, rst_btn_h), border_radius=6)
            pygame.draw.rect(rst_surf, rst_border, (0, 0, rst_btn_w, rst_btn_h), 1, border_radius=6)

            # 세련된 고해상도 안티앨리어싱 회전 화살표(Restart Vector Icon) 렌더링
            S = 3
            w3, h3 = rst_btn_w * S, rst_btn_h * S
            cx3, cy3 = w3 // 2, h3 // 2
            s3 = pygame.Surface((w3, h3), pygame.SRCALPHA)
            r3 = 24.0
            arc_pts = []
            for deg in range(45, 320, 4):
                rad = math.radians(deg)
                arc_pts.append((cx3 + r3 * math.cos(rad), cy3 - r3 * math.sin(rad)))
            pygame.draw.lines(s3, rst_accent, False, arc_pts, 7)

            bx = cx3 + r3 * math.cos(math.radians(45))
            by = cy3 - r3 * math.sin(math.radians(45))
            tip = (bx - 14, by - 2)
            p_top = (bx + 2, by - 14)
            p_bot = (bx + 2, by + 10)
            pygame.draw.polygon(s3, rst_accent, [tip, p_top, p_bot])

            icon_surf = pygame.transform.smoothscale(s3, (rst_btn_w, rst_btn_h))
            rst_surf.blit(icon_surf, (0, 0))

            env.screen.blit(rst_surf, (rst_x, btn_y))
        else:
            env.blind_btn_rect = None
            env.restart_btn_rect = None

    def _draw_leaderboard_modal(self):
        """RC 수동 조종 모드 목적지 도달 시 상위 10등 랭킹 및 GAP 알고리즘 벤치마크 비교 모달 창 표출"""
        env = self.env
        mpos = pygame.mouse.get_pos()

        # 1. 전체 화면 어둡게 디밍 (Scrim)
        scrim = pygame.Surface((env.w, env.h), pygame.SRCALPHA)
        scrim.fill((6, 12, 22, 215))
        env.screen.blit(scrim, (0, 0))

        # 2. 모달 컨테이너 (840 x 560 px)
        mw, mh = 840, 560
        mx = (env.w - mw) // 2
        my = max(15, (env.h - mh) // 2)

        modal_surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        # 딥 네이비 슬레이트 배경 + 사이언 네온 테두리
        pygame.draw.rect(modal_surf, (12, 18, 28, 250), (0, 0, mw, mh), border_radius=12)
        pygame.draw.rect(modal_surf, (0, 180, 240, 210), (0, 0, mw, mh), 2, border_radius=12)

        # 3. 타이틀 헤더
        title_surf = self.ko_title_font.render("LEADERBOARD", True, (230, 245, 255))
        modal_surf.blit(title_surf, title_surf.get_rect(center=(mw // 2, 28)))

        sub_surf = self.ko_small_font.render("TOP 10 RANKINGS", True, (135, 165, 195))
        modal_surf.blit(sub_surf, sub_surf.get_rect(center=(mw // 2, 50)))

        pygame.draw.line(modal_surf, (30, 50, 75), (30, 66), (mw - 30, 66), 1)

        # 4. 데이터 로드 및 랭킹 계산
        last_rec = getattr(env, 'last_manual_result', None)
        if last_rec is None:
            cur_coll = getattr(env, 'manual_collisions', 0)
            cur_time = round(time.time() - getattr(env, 'manual_start_time', time.time()), 2)
            cur_turn = round(getattr(env, 'manual_cum_turn', 0.0), 1)
            last_rec = {"collisions": cur_coll, "time": cur_time, "cumulative_turn_deg": cur_turn, "date": "NOW", "timestamp": time.time()}
        else:
            cur_coll = last_rec.get("collisions", 0)
            cur_time = last_rec.get("time", 0.0)
            cur_turn = last_rec.get("cumulative_turn_deg", 0.0)

        player_rank = leaderboard.get_player_rank(last_rec)
        ai_rank = leaderboard.get_ai_benchmark_rank()

        # 5. 상단 비교 요약 카드 2개 (플레이어 기록 vs GAP 알고리즘 벤치마크)
        card_w, card_h = 380, 78
        card_y = 76

        # 좌측 카드: 플레이어 이번 주행 기록
        c1_x = 30
        pygame.draw.rect(modal_surf, (20, 32, 48, 230), (c1_x, card_y, card_w, card_h), border_radius=8)
        pygame.draw.rect(modal_surf, (255, 190, 40, 190), (c1_x, card_y, card_w, card_h), 1, border_radius=8)

        p_hdr = self.ko_bold_font.render("YOUR ATTEMPT", True, (255, 205, 70))
        modal_surf.blit(p_hdr, (c1_x + 14, card_y + 10))

        p_rank_str = f"#{player_rank}위" if player_rank else "-"
        p_rank_surf = self.ko_bold_font.render(p_rank_str, True, (255, 230, 150))
        modal_surf.blit(p_rank_surf, (c1_x + card_w - p_rank_surf.get_width() - 14, card_y + 10))

        m1_col = (100, 240, 130) if cur_coll == 0 else (255, 110, 100)
        m1_surf = self.ko_font.render(f"충돌: {cur_coll}회", True, m1_col)
        modal_surf.blit(m1_surf, (c1_x + 14, card_y + 42))

        m2_surf = self.ko_font.render(f"시간: {cur_time:.2f}s", True, (215, 235, 255))
        modal_surf.blit(m2_surf, (c1_x + 135, card_y + 42))

        m3_surf = self.ko_font.render(f"회전각: {cur_turn:.1f}\u00b0", True, (215, 235, 255))
        modal_surf.blit(m3_surf, (c1_x + 255, card_y + 42))

        # 우측 카드: GAP 알고리즘 벤치마크
        c2_x = mw - 30 - card_w
        pygame.draw.rect(modal_surf, (14, 36, 48, 230), (c2_x, card_y, card_w, card_h), border_radius=8)
        pygame.draw.rect(modal_surf, (0, 220, 240, 200), (c2_x, card_y, card_w, card_h), 1, border_radius=8)

        ai_hdr = self.ko_bold_font.render("GAP 알고리즘", True, (0, 235, 255))
        modal_surf.blit(ai_hdr, (c2_x + 14, card_y + 10))

        ai_rank_str = f"BENCHMARK (#{ai_rank}위)"
        ai_rank_surf = self.ko_bold_font.render(ai_rank_str, True, (150, 250, 255))
        modal_surf.blit(ai_rank_surf, (c2_x + card_w - ai_rank_surf.get_width() - 14, card_y + 10))

        ai_b = leaderboard.AI_BENCHMARK
        ai1_surf = self.ko_font.render(f"충돌: {ai_b['collisions']}회", True, (100, 245, 140))
        modal_surf.blit(ai1_surf, (c2_x + 14, card_y + 42))

        ai2_surf = self.ko_font.render(f"시간: {ai_b['time']:.1f}s", True, (190, 235, 255))
        modal_surf.blit(ai2_surf, (c2_x + 135, card_y + 42))

        ai3_surf = self.ko_font.render(f"회전각: {ai_b['cumulative_turn_deg']:.1f}\u00b0", True, (190, 235, 255))
        modal_surf.blit(ai3_surf, (c2_x + 255, card_y + 42))

        # 6. 상위 10등 랭킹 테이블 (TOP 10 LEADERBOARD)
        tbl_y = 166
        tbl_w = mw - 60
        tbl_h = 295
        pygame.draw.rect(modal_surf, (15, 22, 34, 210), (30, tbl_y, tbl_w, tbl_h), border_radius=6)
        pygame.draw.rect(modal_surf, (35, 55, 80), (30, tbl_y, tbl_w, tbl_h), 1, border_radius=6)

        # 테이블 헤더 행
        th_h = 28
        pygame.draw.rect(modal_surf, (22, 34, 52), (30, tbl_y, tbl_w, th_h), border_top_left_radius=6, border_top_right_radius=6)

        cols = [
            ("순위", 60),
            ("기록명", 200),
            ("충돌", 110),
            ("시간", 120),
            ("회전각", 120),
            ("일시", 140)
        ]
        col_x = 44
        for name, w in cols:
            lbl = self.ko_bold_font.render(name, True, (170, 200, 230))
            modal_surf.blit(lbl, (col_x, tbl_y + 5))
            col_x += w

        # 전체 목록 로드 및 GAP 알고리즘 벤치마크 항목 결합 정렬
        records = leaderboard.load_leaderboard()
        all_entries = [dict(r) for r in records]
        ai_entry = dict(leaderboard.AI_BENCHMARK)
        ai_entry["player"] = "GAP 알고리즘"
        all_entries.append(ai_entry)

        # 정렬: 1순위 충돌, 2순위 시간, 3순위 누적회전각
        all_entries.sort(key=lambda r: (
            r.get("collisions", 999),
            r.get("time", 9999.0),
            r.get("cumulative_turn_deg", 99999.0)
        ))

        top_10 = all_entries[:10]
        row_y = tbl_y + th_h + 3
        row_h = 25

        for idx, entry in enumerate(top_10):
            rank_num = idx + 1
            is_ai = entry.get("is_ai", False)
            is_cur_attempt = (
                not is_ai and
                last_rec and
                abs(entry.get("timestamp", 0) - last_rec.get("timestamp", -999)) < 0.05
            )

            # 행 배경 및 테두리 스타일
            if is_ai:
                row_bg = (12, 45, 62, 230)
                row_border = (0, 200, 235)
                t_col = (0, 240, 255)
            elif is_cur_attempt:
                row_bg = (50, 40, 16, 230)
                row_border = (255, 190, 40)
                t_col = (255, 220, 110)
            else:
                row_bg = (18, 27, 40, 160) if rank_num % 2 == 1 else (14, 21, 32, 160)
                row_border = None
                t_col = (215, 230, 245)

            pygame.draw.rect(modal_surf, row_bg, (32, row_y, tbl_w - 4, row_h - 2), border_radius=4)
            if row_border:
                pygame.draw.rect(modal_surf, row_border, (32, row_y, tbl_w - 4, row_h - 2), 1, border_radius=4)

            # 순위 텍스트
            col_x = 44
            rank_str = f"#{rank_num}"
            r_surf = self.ko_bold_font.render(rank_str, True, t_col)
            modal_surf.blit(r_surf, (col_x + 4, row_y + 3))
            col_x += 60

            # 구분 / 기록명
            p_name = entry.get("player", "Player")
            if is_ai:
                tag_surf = self.ko_bold_font.render(f"[AI] {p_name}", True, (0, 245, 255))
            elif is_cur_attempt:
                tag_surf = self.ko_bold_font.render(f"[YOU] {p_name}", True, (255, 215, 70))
            else:
                tag_surf = self.ko_font.render(p_name, True, t_col)
            modal_surf.blit(tag_surf, (col_x, row_y + 3))
            col_x += 200

            # 충돌 횟수
            c_val = entry.get("collisions", 0)
            c_str = f"{c_val}회"
            c_col = (100, 245, 140) if c_val == 0 else (255, 120, 110)
            c_surf = self.ko_font.render(c_str, True, c_col)
            modal_surf.blit(c_surf, (col_x + 6, row_y + 3))
            col_x += 110

            # 도달 시간
            tm_val = entry.get("time", 0.0)
            tm_str = f"{tm_val:.2f}s"
            tm_surf = self.ko_font.render(tm_str, True, t_col)
            modal_surf.blit(tm_surf, (col_x + 6, row_y + 3))
            col_x += 120

            # 누적 회전각
            trn_val = entry.get("cumulative_turn_deg", 0.0)
            trn_str = f"{trn_val:.1f}\u00b0"
            trn_surf = self.ko_font.render(trn_str, True, t_col)
            modal_surf.blit(trn_surf, (col_x + 6, row_y + 3))
            col_x += 120

            # 일시
            dt_str = entry.get("date", "-")
            dt_surf = self.ko_small_font.render(dt_str, True, (140, 165, 195))
            modal_surf.blit(dt_surf, (col_x, row_y + 4))

            row_y += row_h

        # 7. 하단 조작 버튼 (다시 도전 / 복귀)
        btn_w, btn_h = 170, 36
        btn_y = 476

        # [다시 도전] 버튼
        r_x = mw // 2 - btn_w - 14
        r_rect_global = pygame.Rect(mx + r_x, my + btn_y, btn_w, btn_h)
        env.leaderboard_retry_rect = r_rect_global
        r_hover = r_rect_global.collidepoint(mpos)

        r_bg = (24, 52, 42, 230) if r_hover else (16, 38, 30, 200)
        r_bd = (40, 240, 140) if r_hover else (25, 180, 105)
        r_txt_col = (200, 255, 220) if r_hover else (150, 235, 180)

        pygame.draw.rect(modal_surf, r_bg, (r_x, btn_y, btn_w, btn_h), border_radius=6)
        pygame.draw.rect(modal_surf, r_bd, (r_x, btn_y, btn_w, btn_h), 1, border_radius=6)
        r_lbl = self.ko_bold_font.render("다시 도전", True, r_txt_col)
        modal_surf.blit(r_lbl, r_lbl.get_rect(center=(r_x + btn_w // 2, btn_y + btn_h // 2)))

        # [복귀] 버튼
        e_x = mw // 2 + 14
        e_rect_global = pygame.Rect(mx + e_x, my + btn_y, btn_w, btn_h)
        env.leaderboard_exit_rect = e_rect_global
        e_hover = e_rect_global.collidepoint(mpos)

        e_bg = (42, 28, 48, 230) if e_hover else (30, 20, 36, 200)
        e_bd = (220, 100, 240) if e_hover else (170, 70, 190)
        e_txt_col = (250, 215, 255) if e_hover else (210, 175, 230)

        pygame.draw.rect(modal_surf, e_bg, (e_x, btn_y, btn_w, btn_h), border_radius=6)
        pygame.draw.rect(modal_surf, e_bd, (e_x, btn_y, btn_w, btn_h), 1, border_radius=6)
        e_lbl = self.ko_bold_font.render("복귀", True, e_txt_col)
        modal_surf.blit(e_lbl, e_lbl.get_rect(center=(e_x + btn_w // 2, btn_y + btn_h // 2)))

        # 하단 단축키 가이드: 스페이스(다시 도전)와 esc(복귀)만 간결하게 표시
        guide_str = "[SPACE] 다시 도전   |   [ESC] 복귀"
        g_surf = self.ko_small_font.render(guide_str, True, (130, 155, 185))
        modal_surf.blit(g_surf, g_surf.get_rect(center=(mw // 2, 528)))

        # 화면에 모달 최종 표출
        env.screen.blit(modal_surf, (mx, my))