import os
import math
import atexit
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import pygame
import moderngl

class _Engine3DCore:
    """
    내부 3D 렌더링 코어 (독립 워커 프로세스 내부에서 하드웨어 가속 실행)
    - ModernGL Core Profile 3.3+ EGL 기반 오프스크린 FBO 렌더링
    - Gerstner Harmonic Waves 해양 셰이더, KABOAT 쌍동선 3D 모델 및 연동 러더,
      항로 표지 부표, 3차원 라이다 포인트 클라우드, 베지에 리본, 다중 시점 카메라
    """
    def __init__(self, width=320, height=220):
        self.width = width
        self.height = height
        
        # EGL 독립형 컨텍스트 초기화 (X11 간섭 없는 순수 하드웨어 EGL 백엔드)
        self.ctx = None
        for backend_name in ['egl', None]:
            try:
                if backend_name:
                    self.ctx = moderngl.create_context(standalone=True, backend=backend_name)
                else:
                    self.ctx = moderngl.create_context(standalone=True)
                if self.ctx is not None:
                    break
            except Exception:
                continue
                
        if self.ctx is None:
            raise RuntimeError("ModernGL context creation failed across all backends.")
            
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        self._fbo_cache = {}
        # 사전 캐싱: 패널(320x220) 및 전체화면 FBO를 VRAM에 고정 상주
        self._get_fbo(320, 220)
        from config import WIDTH, SIM_H
        self._get_fbo(WIDTH, SIM_H)
        
        # GLSL 셰이더 컴파일
        self._init_shaders()
        
        # 3차원 지오메트리 메쉬 생성
        self._init_ocean_mesh()
        self._init_boat_mesh()
        self._init_rudder_mesh()
        self._init_buoy_meshes()
        self._init_beacon_mesh()
        
        # 동적 지오메트리 버퍼 (트라이앵글용 & 라인용 분리)
        self.tri_vbo = self.ctx.buffer(reserve=512 * 1024)
        self.tri_vao = self.ctx.vertex_array(self.prog_unlit, [(self.tri_vbo, '3f 4f', 'in_position', 'in_color')])
        
        self.line_vbo = self.ctx.buffer(reserve=512 * 1024)
        self.line_vao = self.ctx.vertex_array(self.prog_unlit, [(self.line_vbo, '3f 4f', 'in_position', 'in_color')])
        
        # 카메라 및 시간 변수
        self.cam_mode = 1  # 0: 1인칭 조타석, 1: 3인칭 추종 체이스, 2: 전술 드론
        self.cam_names = ["1st-Person Helm", "3rd-Person Chase", "Tactical Drone"]
        self.time = 0.0
        self.lidar_rot = 0.0
        
        # 텍스트 렌더용 폰트 (워커 프로세스 내부 렌더링)
        pygame.font.init()
        self.font = pygame.font.SysFont("sans-serif", 13, bold=True)
        self.panel_title_font = pygame.font.SysFont(None, 24)  # 2D MAP 패널 타이틀과 100% 동일한 폰트 및 크기
        self.micro_font = pygame.font.SysFont("sans-serif", 11)
        self.large_font = pygame.font.SysFont("sans-serif", 18, bold=True)
        self.large_info_font = pygame.font.SysFont("sans-serif", 14)

    def _get_fbo(self, w, h):
        key = (w, h)
        if key not in self._fbo_cache:
            col_tex = self.ctx.texture((w, h), 4)
            depth_rb = self.ctx.depth_renderbuffer((w, h))
            fbo = self.ctx.framebuffer(color_attachments=[col_tex], depth_attachment=depth_rb)
            self._fbo_cache[key] = (fbo, col_tex, depth_rb)
        return self._fbo_cache[key]

    def _init_shaders(self):
        # [1] 조명 및 깊이 안개 메쉬 셰이더 (선체, 부표, 비콘용)
        self.prog_mesh = self.ctx.program(
            vertex_shader='''
            #version 330
            in vec3 in_position;
            in vec3 in_normal;
            in vec3 in_color;
            
            uniform mat4 u_mvp;
            uniform mat4 u_model;
            
            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec3 v_color;
            
            void main() {
                vec4 world_pos = u_model * vec4(in_position, 1.0);
                v_world_pos = world_pos.xyz;
                v_normal = mat3(u_model) * in_normal;
                v_color = in_color;
                gl_Position = u_mvp * vec4(in_position, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec3 v_color;
            
            uniform vec3 u_cam_pos;
            uniform vec3 u_light_dir;
            uniform vec3 u_fog_color;
            uniform int u_blind_mode;
            uniform vec2 u_boat_pos;
            uniform float u_lidar_range;
            
            out vec4 fragColor;
            
            void main() {
                vec3 N = normalize(v_normal);
                vec3 L = normalize(u_light_dir);
                vec3 V = normalize(u_cam_pos - v_world_pos);
                vec3 H = normalize(L + V);
                
                float diff = max(dot(N, L), 0.0) * 0.72 + 0.28;
                float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.40;
                
                vec3 col = v_color * diff + vec3(spec);
                
                // 해무(Fog) 블렌딩
                float dist = length(v_world_pos - u_cam_pos);
                float fog = clamp((dist - 14.0) / 75.0, 0.0, 0.85);
                col = mix(col, u_fog_color, fog);
                
                // 라이다 블라인드 시연 모드: 라이다 범위(6.4m) 외곽 완전 암전(Pitch Black) 처리
                if (u_blind_mode == 1) {
                    float d_boat = length(v_world_pos.xz - u_boat_pos);
                    if (d_boat > u_lidar_range) {
                        col = vec3(0.0);
                    } else if (d_boat > u_lidar_range - 0.7) {
                        float edge_fade = (d_boat - (u_lidar_range - 0.7)) / 0.7;
                        col = mix(col, vec3(0.0), edge_fade);
                    }
                }
                
                fragColor = vec4(col, 1.0);
            }
            '''
        )
        
        # [2] 실시간 동적 해양 수면 셰이더 (하모닉 파고, 프레넬 반사, 태양광 글린트)
        self.prog_ocean = self.ctx.program(
            vertex_shader='''
            #version 330
            in vec2 in_pos;
            
            uniform mat4 u_vp;
            uniform vec2 u_center;
            uniform float u_time;
            
            out vec3 v_world_pos;
            out vec3 v_normal;
            out float v_wave_h;
            
            void main() {
                float x = in_pos.x + u_center.x;
                float z = in_pos.y + u_center.y;
                
                // 중첩 파고 함수 (Gerstner Harmonic Waves)
                float w1 = sin(x * 0.60 + z * 0.40 - u_time * 2.2) * 0.11;
                float w2 = sin(x * 1.25 - z * 0.75 - u_time * 3.3) * 0.05;
                float w3 = cos(x * 2.10 + z * 1.40 - u_time * 4.5) * 0.025;
                float y = w1 + w2 + w3;
                
                // 해석적 법선 벡터 계산
                float dx = 0.60 * 0.11 * cos(x * 0.60 + z * 0.40 - u_time * 2.2)
                         + 1.25 * 0.05 * cos(x * 1.25 - z * 0.75 - u_time * 3.3)
                         - 2.10 * 0.025 * sin(x * 2.10 + z * 1.40 - u_time * 4.5);
                float dz = 0.40 * 0.11 * cos(x * 0.60 + z * 0.40 - u_time * 2.2)
                         - 0.75 * 0.05 * cos(x * 1.25 - z * 0.75 - u_time * 3.3)
                         - 1.40 * 0.025 * sin(x * 2.10 + z * 1.40 - u_time * 4.5);
                
                v_normal = normalize(vec3(-dx, 1.0, -dz));
                v_world_pos = vec3(x, y, z);
                v_wave_h = y;
                
                gl_Position = u_vp * vec4(v_world_pos, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            in vec3 v_world_pos;
            in vec3 v_normal;
            in float v_wave_h;
            
            uniform vec3 u_cam_pos;
            uniform vec3 u_light_dir;
            uniform vec3 u_fog_color;
            uniform int u_blind_mode;
            uniform vec2 u_boat_pos;
            uniform float u_lidar_range;
            
            out vec4 fragColor;
            
            void main() {
                vec3 N = normalize(v_normal);
                vec3 L = normalize(u_light_dir);
                vec3 V = normalize(u_cam_pos - v_world_pos);
                vec3 H = normalize(L + V);
                
                // 입사각에 따른 프레넬 반사율 계산
                float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
                
                // 심해 딥블루와 파고 크레스트 청록색의 그라데이션
                vec3 deep_sea = vec3(0.04, 0.14, 0.28);
                vec3 crest_sea = vec3(0.10, 0.45, 0.62);
                vec3 sky_ref = vec3(0.40, 0.68, 0.90);
                
                vec3 water_color = mix(deep_sea, crest_sea, clamp(v_wave_h * 4.5 + 0.5, 0.0, 1.0));
                water_color = mix(water_color, sky_ref, fresnel * 0.72);
                
                // 햇빛 수면 글린트 (Blinn-Phong Specular)
                float spec = pow(max(dot(N, H), 0.0), 54.0) * 1.25;
                water_color += vec3(1.0, 0.96, 0.88) * spec;
                
                // 파고 최상단 백색 미세 거품 (Foam Crest)
                if (v_wave_h > 0.11) {
                    float foam = clamp((v_wave_h - 0.11) * 16.0, 0.0, 0.70);
                    water_color = mix(water_color, vec3(0.92, 0.97, 1.0), foam);
                }
                
                // 거리 비례 해무 적용
                float dist = length(v_world_pos - u_cam_pos);
                float fog = clamp((dist - 18.0) / 80.0, 0.0, 0.90);
                water_color = mix(water_color, u_fog_color, fog);
                
                // 라이다 블라인드 시연 모드: 라이다 범위(6.4m) 외곽 완전 암전(Pitch Black) 처리
                if (u_blind_mode == 1) {
                    float d_boat = length(v_world_pos.xz - u_boat_pos);
                    if (d_boat > u_lidar_range) {
                        water_color = vec3(0.0);
                    } else if (d_boat > u_lidar_range - 0.7) {
                        float edge_fade = (d_boat - (u_lidar_range - 0.7)) / 0.7;
                        water_color = mix(water_color, vec3(0.0), edge_fade);
                    }
                }
                
                fragColor = vec4(water_color, 1.0);
            }
            '''
        )
        
        # [3] 공간 발광 셰이더 (레이저 빔, 베지에 리본, 홀로그램 마커)
        self.prog_unlit = self.ctx.program(
            vertex_shader='''
            #version 330
            in vec3 in_position;
            in vec4 in_color;
            uniform mat4 u_mvp;
            out vec4 v_color;
            void main() {
                v_color = in_color;
                gl_Position = u_mvp * vec4(in_position, 1.0);
            }
            ''',
            fragment_shader='''
            #version 330
            in vec4 v_color;
            out vec4 fragColor;
            void main() {
                fragColor = v_color;
            }
            '''
        )

    def _init_ocean_mesh(self):
        # 56x56 해양 고밀도 메쉬 그리드 (반경 45m 커버)
        res = 56
        extent = 45.0
        xs = np.linspace(-extent, extent, res, dtype=np.float32)
        zs = np.linspace(-extent, extent, res, dtype=np.float32)
        
        verts = []
        for i in range(res - 1):
            for j in range(res - 1):
                p00 = (xs[i], zs[j])
                p10 = (xs[i + 1], zs[j])
                p01 = (xs[i], zs[j + 1])
                p11 = (xs[i + 1], zs[j + 1])
                
                # Tri 1
                verts.extend([p00[0], p00[1], p10[0], p10[1], p01[0], p01[1]])
                # Tri 2
                verts.extend([p10[0], p10[1], p11[0], p11[1], p01[0], p01[1]])
                
        ocean_data = np.array(verts, dtype=np.float32)
        self.ocean_vbo = self.ctx.buffer(ocean_data.tobytes())
        self.ocean_vao = self.ctx.vertex_array(self.prog_ocean, [(self.ocean_vbo, '2f', 'in_pos')])

    def _init_boat_mesh(self):
        # KABOAT 쌍동선 3차원 기하 형상 (좌현 선체, 우현 선체, 중앙 연결 데크, 레이더 마스트)
        verts = []
        
        def add_box(center, size, color):
            cx, cy, cz = center
            sx, sy, sz = [s * 0.5 for s in size]
            r, g, b = color
            
            # 6개 면 정의 (pos[3], normal[3], col[3])
            faces = [
                # Front (+X)
                ([cx+sx, cy-sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz+sz], [cx+sx, cy-sy, cz+sz], [1.0, 0.0, 0.0]),
                # Back (-X)
                ([cx-sx, cy-sy, cz+sz], [cx-sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz-sz], [cx-sx, cy-sy, cz-sz], [-1.0, 0.0, 0.0]),
                # Top (+Y)
                ([cx-sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz-sz], [0.0, 1.0, 0.0]),
                # Bottom (-Y)
                ([cx-sx, cy-sy, cz+sz], [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz+sz], [0.0, -1.0, 0.0]),
                # Right (+Z)
                ([cx+sx, cy-sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz], [cx-sx, cy-sy, cz+sz], [0.0, 0.0, 1.0]),
                # Left (-Z)
                ([cx-sx, cy-sy, cz-sz], [cx-sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx+sx, cy-sy, cz-sz], [0.0, 0.0, -1.0])
            ]
            
            for p0, p1, p2, p3, norm in faces:
                # Quad -> 2 Triangles
                for pt in [p0, p1, p2, p0, p2, p3]:
                    verts.extend(pt + norm + [r, g, b])

        # 좌현 선체 (Left Hull) - 짙은 메탈릭 네이비 (x: -0.80 ~ +0.50)
        add_box([-0.15, 0.05, -0.38], [1.30, 0.32, 0.28], [0.18, 0.26, 0.36])
        # 좌현 선수부 (Left Bow Nose) - 고시인성 레드 (x: +0.50 ~ +0.92, 테이퍼링 단차로 Z-fighting 100% 박멸)
        add_box([0.71, 0.05, -0.38], [0.42, 0.30, 0.26], [0.92, 0.22, 0.15])
        
        # 우현 선체 (Right Hull) - 짙은 메탈릭 네이비 (x: -0.80 ~ +0.50)
        add_box([-0.15, 0.05, 0.38], [1.30, 0.32, 0.28], [0.18, 0.26, 0.36])
        # 우현 선수부 (Right Bow Nose) - 고시인성 레드 (x: +0.50 ~ +0.92, 테이퍼링 단차로 Z-fighting 100% 박멸)
        add_box([0.71, 0.05, 0.38], [0.42, 0.30, 0.26], [0.92, 0.22, 0.15])
        
        # 중앙 연결 브릿지 데크 (Center Connecting Deck)
        add_box([0.0, 0.16, 0.0], [1.00, 0.12, 0.50], [0.75, 0.82, 0.90])
        
        # 상부 항법 제어 캐빈 (Avionics Cabin Pod)
        add_box([0.05, 0.28, 0.0], [0.60, 0.16, 0.38], [0.12, 0.18, 0.26])
        
        # 라이다 센서 마운트 타워 (LiDAR Tower)
        add_box([0.18, 0.44, 0.0], [0.12, 0.16, 0.12], [0.25, 0.25, 0.30])
        # 회전형 3D 라이다 센서 퍽 (LiDAR Puck - Golden Highlight)
        add_box([0.18, 0.54, 0.0], [0.16, 0.08, 0.16], [0.95, 0.75, 0.10])
        
        # 통신 안테나 돔 (GPS / V2X Dome)
        add_box([-0.22, 0.42, 0.10], [0.10, 0.12, 0.10], [0.95, 0.95, 0.98])
        add_box([-0.22, 0.42, -0.10], [0.10, 0.12, 0.10], [0.95, 0.95, 0.98])

        boat_data = np.array(verts, dtype=np.float32)
        self.boat_vbo = self.ctx.buffer(boat_data.tobytes())
        self.boat_vao = self.ctx.vertex_array(self.prog_mesh, [(self.boat_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _init_rudder_mesh(self):
        # 선미 조타기 러더 (Rudder Blade)
        verts = []
        cx, cy, cz = 0.0, -0.12, 0.0
        sx, sy, sz = 0.15, 0.24, 0.04
        r, g, b = 0.95, 0.15, 0.15 # Red Rudders
        
        faces = [
            ([cx+sx, cy-sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz+sz], [cx+sx, cy-sy, cz+sz], [1.0, 0.0, 0.0]),
            ([cx-sx, cy-sy, cz+sz], [cx-sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz-sz], [cx-sx, cy-sy, cz-sz], [-1.0, 0.0, 0.0]),
            ([cx-sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz-sz], [0.0, 1.0, 0.0]),
            ([cx-sx, cy-sy, cz+sz], [cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz+sz], [0.0, -1.0, 0.0]),
            ([cx+sx, cy-sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz], [cx-sx, cy-sy, cz+sz], [0.0, 0.0, 1.0]),
            ([cx-sx, cy-sy, cz-sz], [cx-sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx+sx, cy-sy, cz-sz], [0.0, 0.0, -1.0])
        ]
        for p0, p1, p2, p3, norm in faces:
            for pt in [p0, p1, p2, p0, p2, p3]:
                verts.extend(pt + norm + [r, g, b])
                
        rud_data = np.array(verts, dtype=np.float32)
        self.rudder_vbo = self.ctx.buffer(rud_data.tobytes())
        self.rudder_vao = self.ctx.vertex_array(self.prog_mesh, [(self.rudder_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _init_buoy_meshes(self):
        # 3D 원통/원뿔 항로 표지 부표 메쉬
        # 기본은 백색(White)이며, 선박과의 거리에 따라 안전(백색) -> 주의(황색) -> 경고(주황색) -> 위험(적색)으로 동적 표출
        def create_buoy_data(main_color, stripe_color=[1.0, 1.0, 1.0]):
            verts = []
            segments = 16
            r_body = 0.35
            y_base = -0.25
            h_base = 0.12
            h_mid = 0.85
            y_top = y_base + h_base + h_mid
            h_cone = 0.60
            ballast_col = [0.15, 0.20, 0.28]
            
            for i in range(segments):
                a1 = (i / segments) * 2 * math.pi
                a2 = ((i + 1) / segments) * 2 * math.pi
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)
                n1 = [c1, 0.0, s1]
                n2 = [c2, 0.0, s2]
                
                # 1. 하단 무게중심 밸러스트 링 (Ballast Ring)
                p0 = [c1 * r_body, y_base, s1 * r_body]
                p1 = [c1 * r_body, y_base + h_base, s1 * r_body]
                p2 = [c2 * r_body, y_base + h_base, s2 * r_body]
                p3 = [c2 * r_body, y_base, s2 * r_body]
                verts.extend(p0 + n1 + ballast_col + p1 + n1 + ballast_col + p2 + n2 + ballast_col)
                verts.extend(p0 + n1 + ballast_col + p2 + n2 + ballast_col + p3 + n2 + ballast_col)
                
                # 2. 하부 부력 원통체 (Lower Float Body - main_color)
                y1 = y_base + h_base
                y2 = y1 + 0.32
                p0 = [c1 * r_body, y1, s1 * r_body]
                p1 = [c1 * r_body, y2, s1 * r_body]
                p2 = [c2 * r_body, y2, s2 * r_body]
                p3 = [c2 * r_body, y1, s2 * r_body]
                verts.extend(p0 + n1 + main_color + p1 + n1 + main_color + p2 + n2 + main_color)
                verts.extend(p0 + n1 + main_color + p2 + n2 + main_color + p3 + n2 + main_color)
                
                # 3. 중간 고반사 안전 띠 (Reflective Stripe Band - stripe_color)
                y3 = y2 + 0.20
                p0 = [c1 * (r_body * 1.02), y2, s1 * (r_body * 1.02)]
                p1 = [c1 * (r_body * 1.02), y3, s1 * (r_body * 1.02)]
                p2 = [c2 * (r_body * 1.02), y3, s2 * (r_body * 1.02)]
                p3 = [c2 * (r_body * 1.02), y2, s2 * (r_body * 1.02)]
                verts.extend(p0 + n1 + stripe_color + p1 + n1 + stripe_color + p2 + n2 + stripe_color)
                verts.extend(p0 + n1 + stripe_color + p2 + n2 + stripe_color + p3 + n2 + stripe_color)
                
                # 4. 상부 부력 원통체 (Upper Float Body - main_color)
                p0 = [c1 * r_body, y3, s1 * r_body]
                p1 = [c1 * r_body, y_top, s1 * r_body]
                p2 = [c2 * r_body, y_top, s2 * r_body]
                p3 = [c2 * r_body, y3, s2 * r_body]
                verts.extend(p0 + n1 + main_color + p1 + n1 + main_color + p2 + n2 + main_color)
                verts.extend(p0 + n1 + main_color + p2 + n2 + main_color + p3 + n2 + main_color)
                
                # 5. 상단 원추형 톱마크 (Top Cone Mark - main_color)
                tip = [0.0, y_top + h_cone, 0.0]
                p_b1 = [c1 * (r_body * 0.72), y_top, s1 * (r_body * 0.72)]
                p_b2 = [c2 * (r_body * 0.72), y_top, s2 * (r_body * 0.72)]
                norm = [c1 * 0.7, 0.7, s1 * 0.7]
                verts.extend(p_b1 + norm + main_color + tip + norm + main_color + p_b2 + norm + main_color)

            return np.array(verts, dtype=np.float32)

        # 1. 기본 백색 부표 (White Default Buoy - 안전 / 거리 >= 4.4m / 220px)
        white_data = create_buoy_data([0.94, 0.95, 0.98], stripe_color=[0.82, 0.88, 0.95])
        self.buoy_white_vbo = self.ctx.buffer(white_data.tobytes())
        self.buoy_white_vao = self.ctx.vertex_array(self.prog_mesh, [(self.buoy_white_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

        # 2. 황색 주의 부표 (Yellow Caution Buoy - 주의 / 거리 < 4.4m / 220px)
        yellow_data = create_buoy_data([0.96, 0.88, 0.16], stripe_color=[1.0, 1.0, 1.0])
        self.buoy_yellow_vbo = self.ctx.buffer(yellow_data.tobytes())
        self.buoy_yellow_vao = self.ctx.vertex_array(self.prog_mesh, [(self.buoy_yellow_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

        # 3. 주황색 경고 부표 (Orange Warning Buoy - 경고 / 거리 < 2.8m / 140px)
        orange_data = create_buoy_data([0.96, 0.54, 0.10], stripe_color=[1.0, 1.0, 1.0])
        self.buoy_orange_vbo = self.ctx.buffer(orange_data.tobytes())
        self.buoy_orange_vao = self.ctx.vertex_array(self.prog_mesh, [(self.buoy_orange_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

        # 4. 적색 위험 부표 (Red Danger Buoy - 위험 / 거리 < 1.4m / 70px)
        red_data = create_buoy_data([0.94, 0.18, 0.18], stripe_color=[1.0, 1.0, 1.0])
        self.buoy_red_vbo = self.ctx.buffer(red_data.tobytes())
        self.buoy_red_vao = self.ctx.vertex_array(self.prog_mesh, [(self.buoy_red_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _init_beacon_mesh(self):
        # 최종 목적지 회전형 비콘 타워 (Lighthouse Beacon Tower - 2D 목표점과 동일한 에메랄드 녹색)
        verts = []
        segments = 16
        r_base = 0.55
        r_top = 0.28
        h_tower = 3.2
        col_green = [0.08, 0.95, 0.35]
        
        for i in range(segments):
            a1 = (i / segments) * 2 * math.pi
            a2 = ((i + 1) / segments) * 2 * math.pi
            c1, s1 = math.cos(a1), math.sin(a1)
            c2, s2 = math.cos(a2), math.sin(a2)
            
            p0 = [c1 * r_base, 0.0, s1 * r_base]
            p1 = [c1 * r_top, h_tower, s1 * r_top]
            p2 = [c2 * r_top, h_tower, s2 * r_top]
            p3 = [c2 * r_base, 0.0, s2 * r_base]
            norm = [c1, 0.1, s1]
            
            verts.extend(p0 + norm + col_green)
            verts.extend(p1 + norm + col_green)
            verts.extend(p2 + norm + col_green)
            verts.extend(p0 + norm + col_green)
            verts.extend(p2 + norm + col_green)
            verts.extend(p3 + norm + col_green)
            
        beacon_data = np.array(verts, dtype=np.float32)
        self.beacon_vbo = self.ctx.buffer(beacon_data.tobytes())
        self.beacon_vao = self.ctx.vertex_array(self.prog_mesh, [(self.beacon_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _wave_height(self, x, z, t):
        w1 = math.sin(x * 0.60 + z * 0.40 - t * 2.2) * 0.11
        w2 = math.sin(x * 1.25 - z * 0.75 - t * 3.3) * 0.05
        w3 = math.cos(x * 2.10 + z * 1.40 - t * 4.5) * 0.025
        return w1 + w2 + w3

    def _matrix_perspective(self, fovy_deg, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
        M = np.zeros((4, 4), dtype=np.float32)
        M[0, 0] = f / aspect
        M[1, 1] = f
        M[2, 2] = (far + near) / (near - far)
        M[2, 3] = (2.0 * far * near) / (near - far)
        M[3, 2] = -1.0
        return M

    def _matrix_look_at(self, eye, target, up):
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        
        f = target - eye
        fn = np.linalg.norm(f)
        f = f / (fn if fn > 1e-6 else 1.0)
        
        s = np.cross(f, up)
        sn = np.linalg.norm(s)
        s = s / (sn if sn > 1e-6 else 1.0)
        
        u = np.cross(s, f)
        
        M = np.identity(4, dtype=np.float32)
        M[0, 0:3] = s
        M[1, 0:3] = u
        M[2, 0:3] = -f
        M[0, 3] = -np.dot(s, eye)
        M[1, 3] = -np.dot(u, eye)
        M[2, 3] = np.dot(f, eye)
        return M

    def _matrix_model(self, x, y, z, yaw, pitch=0.0, roll=0.0):
        # Yaw -> Pitch -> Roll
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        
        Rz = np.array([
            [cr, -sr, 0, 0],
            [sr, cr, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        Rx = np.array([
            [1, 0, 0, 0],
            [0, cp, -sp, 0],
            [0, sp, cp, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        Ry = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        R = Ry @ Rx @ Rz
        R[0, 3] = x
        R[1, 3] = y
        R[2, 3] = z
        return R

    def render_into_buffer(self, env, hits, width, height, out_buf):
        """
        메인 3D 렌더링 파이프라인
        - 오프스크린 FBO 렌더링 후 공유 메모리 버퍼 out_buf에 직접 복사 및 HUD 오버레이 합성
        """
        w = width
        h = height
        fbo, col_tex, depth_rb = self._get_fbo(w, h)
            
        dt = getattr(env, 'dt', 0.04)
        is_paused = getattr(env, 'paused', False)
        if not is_paused:
            self.time += dt
            self.lidar_rot += 0.25
        
        # 1. 시뮬레이션 상태 벡터를 3차원 월드 미터 단위로 변환 (50px = 1.0m)
        bx, by = env.boat_pos
        boat_x = bx / 50.0
        boat_z = by / 50.0
        heading = env.boat_heading
        boat_vel = getattr(env, 'boat_vel', [0.0, 0.0])
        speed = math.hypot(boat_vel[0], boat_vel[1]) / 50.0
        steer = getattr(env, 'prev_steer', 0.0)
        
        fwd_x = math.cos(heading)
        fwd_z = math.sin(heading)
        right_x = -math.sin(heading)
        right_z = math.cos(heading)
        
        boat_wave_y = self._wave_height(boat_x, boat_z, self.time)
        
        # 2. 동적 선체 롤(Roll) 및 피치(Pitch) 동역학 계산
        boat_pitch = math.sin(self.time * 2.8) * 0.035
        boat_roll = -steer * 0.16 + math.cos(self.time * 2.2) * 0.02
        
        # 3. 다중 시점 카메라 좌표 및 목표점 산출
        self.cam_mode = getattr(env, 'cam_3d_mode', self.cam_mode)
        
        if self.cam_mode == 0:
            # 1인칭 조타석 뷰 (선수 앞머리와 파도를 가르는 시점)
            cam_eye = [
                boat_x + fwd_x * 0.35,
                boat_wave_y + 0.48,
                boat_z + fwd_z * 0.35
            ]
            cam_target = [
                cam_eye[0] + fwd_x * 18.0,
                cam_eye[1] - 0.15,
                cam_eye[2] + fwd_z * 18.0
            ]
            fovy = 60.0
        elif self.cam_mode == 1:
            # 3인칭 추종 체이스 뷰 (선체 후방 상공에서 조타와 주변 환경 조망)
            cam_eye = [
                boat_x - fwd_x * 3.6,
                boat_wave_y + 1.75,
                boat_z - fwd_z * 3.6
            ]
            cam_target = [
                boat_x + fwd_x * 2.8,
                boat_wave_y + 0.28,
                boat_z + fwd_z * 2.8
            ]
            fovy = 52.0
        else:
            # 전술 드론 뷰 (고각 45도에서 항로와 게이트 하향 조망)
            cam_eye = [
                boat_x - fwd_x * 8.0,
                boat_wave_y + 7.5,
                boat_z - fwd_z * 8.0
            ]
            cam_target = [
                boat_x + fwd_x * 6.0,
                boat_wave_y + 0.0,
                boat_z + fwd_z * 6.0
            ]
            fovy = 48.0

        # 투영 및 뷰 매트릭스
        aspect = float(w) / float(h)
        P = self._matrix_perspective(fovy, aspect, 0.1, 300.0)
        V = self._matrix_look_at(cam_eye, cam_target, [0.0, 1.0, 0.0])
        VP = P @ V
        
        # 4. FBO 활성화 및 해양 배경색 클리어
        fbo.use()
        is_blind = getattr(env, 'blind_mode', False)
        sky_fog = (0.0, 0.0, 0.0) if is_blind else (0.12, 0.28, 0.48)
        if is_blind:
            self.ctx.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
        else:
            self.ctx.clear(0.08, 0.22, 0.38, 1.0, depth=1.0)
        
        light_dir = (0.55, 0.80, 0.30)
        
        # 5. [수면 렌더링] 동적 해양 셰이더 실행
        self.prog_ocean['u_vp'].write(VP.T.tobytes())
        self.prog_ocean['u_center'].value = (float(boat_x), float(boat_z))
        self.prog_ocean['u_time'].value = float(self.time)
        self.prog_ocean['u_cam_pos'].value = tuple(cam_eye)
        self.prog_ocean['u_light_dir'].value = light_dir
        self.prog_ocean['u_fog_color'].value = sky_fog
        self.prog_ocean['u_blind_mode'].value = 1 if is_blind else 0
        self.prog_ocean['u_boat_pos'].value = (float(boat_x), float(boat_z))
        self.prog_ocean['u_lidar_range'].value = 6.4
        self.ocean_vao.render()
        
        # 6. [부표 장애물 렌더링] 시야 반경 내 부표 표출
        self.prog_mesh['u_cam_pos'].value = tuple(cam_eye)
        self.prog_mesh['u_light_dir'].value = light_dir
        self.prog_mesh['u_fog_color'].value = sky_fog
        self.prog_mesh['u_blind_mode'].value = 1 if is_blind else 0
        self.prog_mesh['u_boat_pos'].value = (float(boat_x), float(boat_z))
        self.prog_mesh['u_lidar_range'].value = 6.4
        
        max_buoy_dist_sq = (6.4 * 6.4) if is_blind else (55.0 * 55.0)
        for idx, (ox, oy, r) in enumerate(env.dynamic_obstacles):
            obs_x = ox / 50.0
            obs_z = oy / 50.0
            dx = obs_x - boat_x; dz = obs_z - boat_z
            if dx*dx + dz*dz > max_buoy_dist_sq:
                continue
                
            obs_y = self._wave_height(obs_x, obs_z, self.time)
            tilt_ang = math.sin(self.time * 2.0 + obs_x * 0.5) * 0.06
            
            # 수직 기립 상태의 3D 부표 매트릭스
            M_buoy = self._matrix_model(obs_x, obs_y, obs_z, 0.0, pitch=0.0, roll=tilt_ang)
            MVP_buoy = VP @ M_buoy
            self.prog_mesh['u_model'].write(M_buoy.T.tobytes())
            self.prog_mesh['u_mvp'].write(MVP_buoy.T.tobytes())
            
            # 거리 기반 동적 위험도 색상 표출 (기본: 백색, 접근 시 황색 -> 주황색 -> 적색)
            # 1m = 50px (d_px < 70 -> 1.4m, d_px < 140 -> 2.8m, d_px < 220 -> 4.4m)
            dist_m = max(0.0, math.sqrt(dx * dx + dz * dz) - (r / 50.0))
            if dist_m < 1.4:
                self.buoy_red_vao.render()
            elif dist_m < 2.8:
                self.buoy_orange_vao.render()
            elif dist_m < 4.4:
                self.buoy_yellow_vao.render()
            else:
                self.buoy_white_vao.render()
                
        # 7. [선체 렌더링] KABOAT 쌍동선 및 조타 연동 러더 표출
        M_boat = self._matrix_model(boat_x, boat_wave_y, boat_z, -heading, pitch=boat_pitch, roll=boat_roll)
        MVP_boat = VP @ M_boat
        self.prog_mesh['u_model'].write(M_boat.T.tobytes())
        self.prog_mesh['u_mvp'].write(MVP_boat.T.tobytes())
        self.boat_vao.render()
        
        # 선미 좌/우 조타기(Rudder) 회전 반영
        p_z = 0.38
        for rud_z in [-p_z, p_z]:
            rud_x_local = -0.80
            rud_x_w = boat_x + rud_x_local * fwd_x + rud_z * right_x
            rud_z_w = boat_z + rud_x_local * fwd_z + rud_z * right_z
            M_rud = self._matrix_model(rud_x_w, boat_wave_y + 0.02, rud_z_w, -heading - steer * 0.8, pitch=boat_pitch, roll=boat_roll)
            MVP_rud = VP @ M_rud
            self.prog_mesh['u_model'].write(M_rud.T.tobytes())
            self.prog_mesh['u_mvp'].write(MVP_rud.T.tobytes())
            self.rudder_vao.render()

        # 8. [목적지 비콘 타워 렌더링]
        tgt_x = env.target[0] / 50.0
        tgt_z = env.target[1] / 50.0
        tgt_dx = tgt_x - boat_x
        tgt_dz = tgt_z - boat_z
        tgt_dist = math.hypot(tgt_dx, tgt_dz)
        rim_ang = math.atan2(tgt_dz, tgt_dx)

        # 실제 목적지 비콘 타워는 가시 반경(6.4m) 내에 있거나 블라인드 모드가 아닐 때만 실제 위치에 표출
        # (원 테두리에 가짜 비콘 타워를 세우지 않고 수면 위 앰버 셰브론 항법 인디케이터로 차별화)
        if not is_blind or tgt_dist <= 6.4:
            tgt_y = self._wave_height(tgt_x, tgt_z, self.time)
            M_tgt = self._matrix_model(tgt_x, tgt_y, tgt_z, self.time * 0.5)
            MVP_tgt = VP @ M_tgt
            self.prog_mesh['u_model'].write(M_tgt.T.tobytes())
            self.prog_mesh['u_mvp'].write(MVP_tgt.T.tobytes())
            self.beacon_vao.render()

        # 9. [동적 발광 그래픽스 렌더링] (베지에 리본, 홀로그램 링, 라이다 광선)
        tri_verts = []
        line_verts = []
        
        # (1) 3차 베지에 계획 궤적 3D 발광 리본 (수면 위 0.08m, 트라이앵글 쿼드 스트립)
        path = getattr(env, 'bezier_path', None)
        if path is not None and len(path) >= 2:
            ribbon_w = 0.25
            col_ribbon = [0.10, 0.90, 1.0, 0.88] # Bright Cyan Glowing Ribbon
            
            pts_3d = []
            for px, py in path:
                wx = px / 50.0; wz = py / 50.0
                wy = self._wave_height(wx, wz, self.time) + 0.08
                pts_3d.append(np.array([wx, wy, wz]))
                
            for i in range(len(pts_3d) - 1):
                p_cur = pts_3d[i]
                p_nxt = pts_3d[i + 1]
                tangent = p_nxt - p_cur
                tangent[1] = 0.0
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-4:
                    tangent /= norm_t
                    side = np.array([-tangent[2], 0.0, tangent[0]]) * ribbon_w
                    
                    v1 = p_cur + side; v2 = p_cur - side
                    v3 = p_nxt + side; v4 = p_nxt - side
                    
                    tri_verts.extend(list(v1) + col_ribbon)
                    tri_verts.extend(list(v2) + col_ribbon)
                    tri_verts.extend(list(v3) + col_ribbon)
                    tri_verts.extend(list(v2) + col_ribbon)
                    tri_verts.extend(list(v4) + col_ribbon)
                    tri_verts.extend(list(v3) + col_ribbon)

        # (2) 웨이포인트 1 (WP1) & 2 (WP2) 3D 홀로그램 기둥 및 회전 링
        def add_holo_beacon(wp_pos, color_rgba, height=3.5):
            wx = wp_pos[0] / 50.0; wz = wp_pos[1] / 50.0
            wy = self._wave_height(wx, wz, self.time)
            
            # 수직 발광 광선
            line_verts.extend([wx, wy, wz] + color_rgba)
            line_verts.extend([wx, wy + height, wz] + color_rgba)
            
            # 회전하는 3D 홀로그램 링
            n_ring = 16
            r_ring = 0.60
            for r_lvl in [0.8, 1.9]:
                ring_y = wy + r_lvl + math.sin(self.time * 3.0) * 0.12
                for k in range(n_ring):
                    ang1 = (k / n_ring) * 2 * math.pi + self.time * 1.5
                    ang2 = ((k + 1) / n_ring) * 2 * math.pi + self.time * 1.5
                    p1 = [wx + math.cos(ang1)*r_ring, ring_y, wz + math.sin(ang1)*r_ring]
                    p2 = [wx + math.cos(ang2)*r_ring, ring_y, wz + math.sin(ang2)*r_ring]
                    line_verts.extend(p1 + color_rgba)
                    line_verts.extend(p2 + color_rgba)

        # 최종 목적지 녹색 홀로그램 비콘 및 발광 회전 링 (2D 녹색 타겟과 100% 색상 통일)
        if hasattr(env, 'target') and env.target is not None:
            if not is_blind or tgt_dist <= 6.4:
                add_holo_beacon(env.target, [0.08, 0.98, 0.35, 0.95], height=5.5)

        # 블라인드 모드 시 3D 수면 상에 라이다 시야 테두리 발광 원(반경 6.4m) 및 목표 방향 셰브론 가이드 렌더링
        if is_blind:
            n_segs = 72
            rim_pts = []
            for s_idx in range(n_segs + 1):
                th = (s_idx / n_segs) * 2.0 * math.pi
                rx = boat_x + math.cos(th) * 6.4
                rz = boat_z + math.sin(th) * 6.4
                ry = self._wave_height(rx, rz, self.time) + 0.05
                rim_pts.append((rx, ry, rz, th))
                
            for s_idx in range(n_segs):
                p1_info, p2_info = rim_pts[s_idx], rim_pts[s_idx + 1]
                p1, p2 = p1_info[:3], p2_info[:3]
                mid_th = 0.5 * (p1_info[3] + p2_info[3])
                # 목표 방위각 근처(±0.25 rad)는 골드/앰버 호(Arc)로 강조
                d_th = abs((mid_th - rim_ang + math.pi) % (2.0 * math.pi) - math.pi)
                if tgt_dist > 6.4 and d_th < 0.26:
                    c_rim = [1.0, 0.82, 0.15, 0.98] # Golden Compass Arc
                else:
                    c_rim = [0.0, 0.85, 1.0, 0.75] # Cyan Rim
                line_verts.extend(list(p1) + c_rim)
                line_verts.extend(list(p2) + c_rim)

            # 라이다 반경(6.4m) 너머에 목표가 있을 때:
            # 실제 목적지 비콘과 완전히 차별화된 3D 수면 앰버 셰브론 항법 포인터(Chevron Pointer) 표출
            if tgt_dist > 6.4:
                ux = math.cos(rim_ang)
                uz = math.sin(rim_ang)
                px = -uz
                pz = ux
                
                rim_x = boat_x + ux * 6.4
                rim_z = boat_z + uz * 6.4
                
                # 수면 위 앰버 셰브론 쐐기 화살표 기하 생성
                tip_x = rim_x + ux * 0.90
                tip_z = rim_z + uz * 0.90
                tip_y = self._wave_height(tip_x, tip_z, self.time) + 0.12
                
                notch_x = rim_x + ux * 0.15
                notch_z = rim_z + uz * 0.15
                notch_y = self._wave_height(notch_x, notch_z, self.time) + 0.12
                
                wing_len = 0.55
                back_offset = 0.35
                left_x = rim_x - ux * back_offset + px * wing_len
                left_z = rim_z - uz * back_offset + pz * wing_len
                left_y = self._wave_height(left_x, left_z, self.time) + 0.12
                
                right_x = rim_x - ux * back_offset - px * wing_len
                right_z = rim_z - uz * back_offset - pz * wing_len
                right_y = self._wave_height(right_x, right_z, self.time) + 0.12
                
                col_nav_tri = [1.0, 0.76, 0.12, 0.92]   # Amber Fill
                col_nav_line = [1.0, 0.95, 0.50, 0.98]  # Gold Highlight Outline
                
                # 셰브론 2개 삼각형 (tip-left-notch, tip-notch-right)
                v_tip = [tip_x, tip_y, tip_z]
                v_notch = [notch_x, notch_y, notch_z]
                v_left = [left_x, left_y, left_z]
                v_right = [right_x, right_y, right_z]
                
                tri_verts.extend(v_tip + col_nav_tri)
                tri_verts.extend(v_left + col_nav_tri)
                tri_verts.extend(v_notch + col_nav_tri)
                
                tri_verts.extend(v_tip + col_nav_tri)
                tri_verts.extend(v_notch + col_nav_tri)
                tri_verts.extend(v_right + col_nav_tri)
                
                # 셰브론 테두리 발광 선
                line_verts.extend(v_tip + col_nav_line); line_verts.extend(v_left + col_nav_line)
                line_verts.extend(v_left + col_nav_line); line_verts.extend(v_notch + col_nav_line)
                line_verts.extend(v_notch + col_nav_line); line_verts.extend(v_right + col_nav_line)
                line_verts.extend(v_right + col_nav_line); line_verts.extend(v_tip + col_nav_line)
                
                # 바깥 어둠 속으로 뻗어 나가는 원거리 항법 방향 지시선 (Directional Ray Beam)
                col_ray = [1.0, 0.82, 0.20, 0.85]
                for r_dist in [1.5, 3.2, 5.0]:
                    rs_x = rim_x + ux * (r_dist - 0.4)
                    rs_z = rim_z + uz * (r_dist - 0.4)
                    rs_y = self._wave_height(rs_x, rs_z, self.time) + 0.10
                    re_x = rim_x + ux * r_dist
                    re_z = rim_z + uz * r_dist
                    re_y = self._wave_height(re_x, re_z, self.time) + 0.10
                    line_verts.extend([rs_x, rs_y, rs_z] + col_ray)
                    line_verts.extend([re_x, re_y, re_z] + col_ray)

        if getattr(env, 'current_wp', None) is not None:
            add_holo_beacon(env.current_wp["pos"], [0.0, 1.0, 0.85, 0.95]) # Cyan WP1
        if getattr(env, 'next_wp', None) is not None:
            add_holo_beacon(env.next_wp["pos"], [0.82, 0.42, 1.0, 0.90])  # Purple WP2
            
        # 회피 지점 (Avoid Hit)
        if getattr(env, 'linetrace_mode', False):
            c_hit = getattr(env, 'closest_avoid_hit', None)
            if c_hit is not None:
                add_holo_beacon(c_hit, [1.0, 0.10, 0.80, 0.95], height=2.5) # Magenta Hazard

        # (3) 3차원 공간 라이다 광선 빔 및 히트 지점 포인트
        lidar_src = [boat_x, boat_wave_y + 0.68, boat_z]
        col_laser = [0.20, 0.95, 0.60, 0.22]
        col_hit = [1.0, 0.88, 0.20, 0.92] # Glowing Amber Hit Marker
        
        if hits is not None:
            for k in range(0, len(hits), 2):
                hp = hits[k]
                if hp is not None:
                    hx = hp[0] / 50.0; hz = hp[1] / 50.0
                    hy = self._wave_height(hx, hz, self.time) + 0.20
                    # 발사 광선
                    line_verts.extend(lidar_src + col_laser)
                    line_verts.extend([hx, hy, hz] + col_laser)
                    # 히트 지점 마커 (0.08m 3D 십자 스타)
                    s_m = 0.08
                    line_verts.extend([hx - s_m, hy, hz] + col_hit); line_verts.extend([hx + s_m, hy, hz] + col_hit)
                    line_verts.extend([hx, hy - s_m, hz] + col_hit); line_verts.extend([hx, hy + s_m, hz] + col_hit)
                    line_verts.extend([hx, hy, hz - s_m] + col_hit); line_verts.extend([hx, hy, hz + s_m] + col_hit)

        # 셰이더 uniform 바인딩 및 렌더링
        self.prog_unlit['u_mvp'].write(VP.T.tobytes())
        
        if len(tri_verts) > 0:
            tri_data = np.array(tri_verts, dtype=np.float32)
            self.tri_vbo.write(tri_data.tobytes())
            self.tri_vao.render(moderngl.TRIANGLES, vertices=len(tri_data) // 7)
            
        if len(line_verts) > 0:
            line_data = np.array(line_verts, dtype=np.float32)
            self.line_vbo.write(line_data.tobytes())
            self.line_vao.render(moderngl.LINES, vertices=len(line_data) // 7)

        # 10. FBO 버퍼를 공유 메모리에 직접 고속 읽기 (Zero-Copy)
        raw_pixels = fbo.read(components=4)
        raw_arr = np.frombuffer(raw_pixels, dtype=np.uint8).reshape((h, w, 4))
        # OpenGL Y축 반전 보정 (고속 슬라이싱)
        out_buf[:] = np.ascontiguousarray(raw_arr[::-1, :, :])
        
        # 11. 공유 메모리 버퍼 위에 직접 3D HUD 계기판 오버레이 합성
        surf_3d = pygame.image.frombuffer(out_buf, (w, h), 'RGBA')
        self._draw_hud_overlay(surf_3d, env, speed, steer, heading)

    def _draw_hud_overlay(self, surf, env, speed, steer, heading):
        w, h = surf.get_size()
        
        # [1] 테두리 프레임
        pygame.draw.rect(surf, (0, 200, 255), (0, 0, w, h), 2)
        
        # [2] 해상도별 반응형 폰트 및 바 크기 결정
        is_large = (w > 600)
        f_title = self.large_font if is_large else self.font
        f_info = self.large_info_font if is_large else self.micro_font
        
        # [3] 패널 이름 (작은 3D 패널일 때만 2D MAP과 동일한 크기/양식으로 좌측 상단에 표출, 전체화면일 땐 미표출)
        if not is_large:
            title_txt = "3D VIEW"
            lbl_title = self.panel_title_font.render(title_txt, True, (240, 245, 255))
            surf.blit(lbl_title, (10, 8))
        
        # [4] 1인칭 조타석 뷰 전용 조타 HUD (십자선만 유지, 상단 중앙 파란색 헤딩 텍스트는 제거)
        if self.cam_mode == 0:
            cx, cy = w // 2, h // 2
            ret_r = 24 if is_large else 16
            ret_arm = 36 if is_large else 24
            # 중앙 조타 십자선 (Tactical Reticle)
            pygame.draw.circle(surf, (0, 255, 220, 180), (cx, cy), ret_r, 1)
            pygame.draw.line(surf, (0, 255, 220, 200), (cx - ret_arm, cy), (cx - 8, cy), 1)
            pygame.draw.line(surf, (0, 255, 220, 200), (cx + 8, cy), (cx + ret_arm, cy), 1)
            pygame.draw.line(surf, (0, 255, 220, 200), (cx, cy - ret_arm // 2), (cx, cy - 6), 1)
            pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 2)
            
        # [5] 하단 정보 표출 (선속, 헤딩, 러더 각도, 엔진 정보)
        # 상단 중앙의 파란색 헤딩을 제거하고 하단 흰색 텍스트 아이템들 사이에 통합
        hdg_deg = int(math.degrees(heading)) % 360
        steer_deg = math.degrees(steer)
        knots = speed * 1.94384
        is_blind = getattr(env, 'blind_mode', False)

        if is_large:
            # 전체화면 3D 모드: 화면을 가리는 하단 검은색 바를 추가하지 않고 3D 화면을 100% 꽉 채우며, 우측 하단 텍스트는 그대로 유지
            blind_tag = " | BLIND VISION" if is_blind else ""
            rc_tag = f" | RC MANUAL [WASD]{blind_tag}" if getattr(env, 'manual_mode', False) else " | ModernGL 3.3 Core Profile"
            txt_str = f"SPEED: {speed:.1f} m/s ({knots:.1f} kt) | HDG: {hdg_deg:03d}° | RUDDER: {steer_deg:+.1f}°{rc_tag}"
            lbl_stat = f_info.render(txt_str, True, (225, 242, 255))
            lbl_stat_sh = f_info.render(txt_str, True, (10, 15, 25))
            txt_x = w - lbl_stat.get_width() - 16
            txt_y = h - 24
            surf.blit(lbl_stat_sh, (txt_x + 1, txt_y + 1))
            surf.blit(lbl_stat, (txt_x, txt_y))

            # 블라인드 모드 상단 중앙 목표 방향 및 방위각 가이드 HUD (앰버-골드 항법 컬러)
            if is_blind and hasattr(env, 'target') and env.target is not None:
                tgt_dx = (env.target[0] / 50.0) - (env.boat_pos[0] / 50.0)
                tgt_dz = (env.target[1] / 50.0) - (env.boat_pos[1] / 50.0)
                tgt_dist = math.hypot(tgt_dx, tgt_dz)
                t_bearing = int(math.degrees(math.atan2(tgt_dz, tgt_dx))) % 360
                rel_bearing = (t_bearing - hdg_deg) % 360
                if rel_bearing > 180: rel_bearing -= 360

                b_str = f"TARGET DIR -> {tgt_dist:.1f}m | BEARING: {t_bearing:03d}° (REL {rel_bearing:+d}°)"
                lbl_b = f_info.render(b_str, True, (255, 220, 80))
                badge_pad_x, badge_pad_y = 14, 5
                badge_w = lbl_b.get_width() + badge_pad_x * 2
                badge_h = lbl_b.get_height() + badge_pad_y * 2
                bx_hud = (w - badge_w) // 2
                by_hud = 36
                badge_surf = pygame.Surface((badge_w, badge_h), pygame.SRCALPHA)
                badge_surf.fill((28, 20, 8, 225))
                pygame.draw.rect(badge_surf, (255, 190, 30), (0, 0, badge_w, badge_h), 1, border_radius=4)
                badge_surf.blit(lbl_b, (badge_pad_x, badge_pad_y))
                surf.blit(badge_surf, (bx_hud, by_hud))
        else:
            # 기본 3D 패널(320x220): 하단 바 영역에 표출
            info_h = 20
            info_y = h - info_h
            pygame.draw.rect(surf, (8, 18, 30, 210), (0, info_y, w, info_h))
            pygame.draw.line(surf, (0, 140, 200), (0, info_y), (w, info_y), 1)
            txt_str = f"SPEED: {speed:.1f}m/s | HDG: {hdg_deg:03d}° | RUD: {steer_deg:+.1f}° | ModernGL 3.3 Core"
            lbl_stat = f_info.render(txt_str, True, (225, 242, 255))
            surf.blit(lbl_stat, (8, info_y + 3))


def _engine_3d_worker_proc(pipe, shm_panel_name, shm_full_name, full_w=1840, full_h=644):
    """
    독립 OS 프로세스에서 실행되는 3D 렌더링 워커 루프
    - 메인 Pygame 프로세스의 X11/Wayland 2D 그래픽스 파이프라인과 완벽히 격리
    """
    try:
        core = _Engine3DCore(320, 220)
    except Exception as e:
        pipe.send({'error': str(e)})
        return

    pipe.send({'status': 'ready'})
    
    shm_panel = shared_memory.SharedMemory(name=shm_panel_name)
    shm_full = shared_memory.SharedMemory(name=shm_full_name)
    
    buf_panel = np.ndarray((220, 320, 4), dtype=np.uint8, buffer=shm_panel.buf)
    buf_full = np.ndarray((full_h, full_w, 4), dtype=np.uint8, buffer=shm_full.buf)
    
    class ProxyEnv:
        pass
    p_env = ProxyEnv()
    
    while True:
        try:
            req = pipe.recv()
        except EOFError:
            break
            
        if req is None or req.get('cmd') == 'close':
            break
            
        w = req['w']
        h = req['h']
        hits = req.get('hits')
        
        p_env.boat_pos = req['boat_pos']
        p_env.boat_heading = req['boat_heading']
        p_env.boat_vel = req['boat_vel']
        p_env.prev_steer = req['prev_steer']
        p_env.cam_3d_mode = req['cam_3d_mode']
        p_env.dynamic_obstacles = req['dynamic_obstacles']
        p_env.target = req['target']
        p_env.bezier_path = req['bezier_path']
        p_env.current_wp = req['current_wp']
        p_env.next_wp = req['next_wp']
        p_env.linetrace_mode = req['linetrace_mode']
        p_env.closest_avoid_hit = req['closest_avoid_hit']
        p_env.dt = req.get('dt', 0.04)
        p_env.paused = req.get('paused', False)
        p_env.manual_mode = req.get('manual_mode', False)
        p_env.blind_mode = req.get('blind_mode', False)
        
        target_buf = buf_panel if (w, h) == (320, 220) else buf_full
        core.render_into_buffer(p_env, hits, w, h, target_buf)
        pipe.send(True)
        
    shm_panel.close()
    shm_full.close()


class Engine3D:
    """
    메인 애플리케이션용 고성능 프로세스 격리 3D 엔진 클라이언트
    - POSIX 공유 메모리(/dev/shm) 기반 마이크로초 단위 무복사(Zero-Copy) 버퍼 교환
    - 메인 Pygame 윈도우 서피스 검은 화면 및 드라이버 훅 충돌 원천 방지
    """
    def __init__(self, width=320, height=220, full_w=None, full_h=None):
        from config import WIDTH, SIM_H
        if full_w is None: full_w = WIDTH
        if full_h is None: full_h = SIM_H
        self.width = width
        self.height = height
        self.full_w = full_w
        self.full_h = full_h
        self._closed = False
        
        # 패널(320x220) 및 전체화면(full_w x full_h) 공유 메모리 블록 생성
        self.shm_panel = shared_memory.SharedMemory(create=True, size=320 * 220 * 4)
        self.shm_full = shared_memory.SharedMemory(create=True, size=full_w * full_h * 4)
        
        # Pygame Surface를 공유 메모리에 직접 매핑 (고정 참조)
        self.surf_panel = pygame.image.frombuffer(self.shm_panel.buf, (320, 220), 'RGBA')
        self.surf_full = pygame.image.frombuffer(self.shm_full.buf, (full_w, full_h), 'RGBA')
        
        # 클린 프로세스 스폰
        ctx_spawn = mp.get_context('spawn')
        self.parent_conn, self.child_conn = ctx_spawn.Pipe(duplex=True)
        self.proc = ctx_spawn.Process(
            target=_engine_3d_worker_proc,
            args=(self.child_conn, self.shm_panel.name, self.shm_full.name, full_w, full_h),
            daemon=True
        )
        self.proc.start()
        
        # 워커 시작 상태 확인
        init_res = self.parent_conn.recv()
        if 'error' in init_res:
            self.close()
            raise RuntimeError(f"3D Worker process initialization failed: {init_res['error']}")
            
        atexit.register(self.close)

    def render(self, env, hits, width=None, height=None):
        if self._closed or not self.proc.is_alive():
            # 워커가 비활성 상태인 경우 대체용 서피스 반환
            w = width or self.width
            h = height or self.height
            s = pygame.Surface((w, h))
            s.fill((20, 40, 60))
            return s
            
        w = width or self.width
        h = height or self.height
        
        # 최소 상태 페이로드 직렬화
        req = {
            'w': w, 'h': h,
            'boat_pos': tuple(env.boat_pos),
            'boat_heading': float(env.boat_heading),
            'boat_vel': tuple(getattr(env, 'boat_vel', [0.0, 0.0])),
            'prev_steer': float(getattr(env, 'prev_steer', 0.0)),
            'cam_3d_mode': int(getattr(env, 'cam_3d_mode', 1)),
            'dynamic_obstacles': [list(obs) for obs in env.dynamic_obstacles],
            'target': tuple(env.target),
            'bezier_path': [list(p) for p in env.bezier_path] if getattr(env, 'bezier_path', None) is not None else None,
            'current_wp': env.current_wp,
            'next_wp': env.next_wp,
            'linetrace_mode': bool(getattr(env, 'linetrace_mode', False)),
            'closest_avoid_hit': getattr(env, 'closest_avoid_hit', None),
            'hits': hits,
            'dt': float(getattr(env, 'dt', 0.04)),
            'paused': bool(getattr(env, 'paused', False)),
            'manual_mode': bool(getattr(env, 'manual_mode', False)),
            'blind_mode': bool(getattr(env, 'blind_mode', False))
        }
        
        self.parent_conn.send(req)
        self.parent_conn.recv()
        
        if (w, h) == (320, 220):
            return self.surf_panel
        else:
            return self.surf_full

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self, 'parent_conn') and self.parent_conn:
                self.parent_conn.send({'cmd': 'close'})
            if hasattr(self, 'proc') and self.proc:
                self.proc.join(timeout=0.5)
                if self.proc.is_alive():
                    self.proc.terminate()
        except Exception:
            pass
            
        try:
            if hasattr(self, 'shm_panel') and self.shm_panel:
                self.shm_panel.close()
                self.shm_panel.unlink()
        except Exception:
            pass
            
        try:
            if hasattr(self, 'shm_full') and self.shm_full:
                self.shm_full.close()
                self.shm_full.unlink()
        except Exception:
            pass

    def __del__(self):
        self.close()
