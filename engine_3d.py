import moderngl
import numpy as np
import math
import pygame

class Engine3D:
    """
    ModernGL 기반 실시간 하드웨어 가속 3D 그래픽스 엔진
    - 해양 환경(Gerstner Waves), KABOAT 쌍동선 3차원 모델, 실시간 항로 표지 부표,
      공간 라이다 포인트 클라우드, 베지에 궤적 리본, 다중 시점 카메라 시스템 렌더링
    """
    def __init__(self, width=320, height=220):
        self.width = width
        self.height = height
        
        # 1. ModernGL 독립형 컨텍스트 초기화 (Hardware GPU Acceleration)
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        self._fbo_cache = {}
        # 사전 캐싱: 패널(320x220) 및 전체화면(1800x630) FBO를 VRAM에 고정 상주
        self._get_fbo(320, 220)
        
        # 2. GLSL 셰이더 컴파일
        self._init_shaders()
        
        # 3. 3차원 지오메트리 메쉬 생성
        self._init_ocean_mesh()
        self._init_boat_mesh()
        self._init_rudder_mesh()
        self._init_buoy_meshes()
        self._init_beacon_mesh()
        
        # 4. 동적 지오메트리 버퍼 (트라이앵글용 & 라인용 분리)
        self.tri_vbo = self.ctx.buffer(reserve=512 * 1024)
        self.tri_vao = self.ctx.vertex_array(self.prog_unlit, [(self.tri_vbo, '3f 4f', 'in_position', 'in_color')])
        
        self.line_vbo = self.ctx.buffer(reserve=512 * 1024)
        self.line_vao = self.ctx.vertex_array(self.prog_unlit, [(self.line_vbo, '3f 4f', 'in_position', 'in_color')])
        
        # 5. 카메라 및 시간 변수
        self.cam_mode = 1  # 0: 1인칭 조타석, 1: 3인칭 추종 체이스, 2: 전술 드론
        self.cam_names = ["1st-Person Helm", "3rd-Person Chase", "Tactical Drone"]
        self.time = 0.0
        self.lidar_rot = 0.0
        
        # 텍스트 렌더용 폰트 (패널 및 전체화면 모드 반응형 크기)
        pygame.font.init()
        self.font = pygame.font.SysFont("sans-serif", 13, bold=True)
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
                
                fragColor = vec4(water_color, 0.96);
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
        grid_x, grid_z = np.meshgrid(xs, zs)
        
        verts = []
        for i in range(res - 1):
            for j in range(res - 1):
                p0 = [grid_x[i, j], grid_z[i, j]]
                p1 = [grid_x[i+1, j], grid_z[i+1, j]]
                p2 = [grid_x[i, j+1], grid_z[i, j+1]]
                p3 = [grid_x[i+1, j+1], grid_z[i+1, j+1]]
                verts.extend(p0 + p1 + p2)
                verts.extend(p1 + p3 + p2)
                
        ocean_data = np.array(verts, dtype=np.float32)
        self.ocean_count = len(ocean_data) // 2
        self.ocean_vbo = self.ctx.buffer(ocean_data.tobytes())
        self.ocean_vao = self.ctx.vertex_array(self.prog_ocean, [(self.ocean_vbo, '2f', 'in_pos')])

    def _init_boat_mesh(self):
        # KABOAT 정밀 쌍동선(Catamaran) 모델링
        verts = []
        
        def add_box(center, size, color):
            cx, cy, cz = center
            sx, sy, sz = size[0]*0.5, size[1]*0.5, size[2]*0.5
            faces = [
                ([cx-sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz], [0, 1, 0]),
                ([cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz-sz], [cx-sx, cy-sy, cz-sz], [0, -1, 0]),
                ([cx+sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz-sz], [1, 0, 0]),
                ([cx-sx, cy-sy, cz+sz], [cx-sx, cy-sy, cz-sz], [cx-sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz+sz], [-1, 0, 0]),
                ([cx+sx, cy-sy, cz+sz], [cx-sx, cy-sy, cz+sz], [cx-sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz+sz], [0, 0, 1]),
                ([cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz], [0, 0, -1]),
            ]
            for p0, p1, p2, p3, n in faces:
                verts.extend(p0 + n + list(color))
                verts.extend(p1 + n + list(color))
                verts.extend(p2 + n + list(color))
                verts.extend(p0 + n + list(color))
                verts.extend(p2 + n + list(color))
                verts.extend(p3 + n + list(color))

        def add_wedge(p_tip, p_base1, p_base2, p_base3, p_base4, color):
            triangles = [
                (p_tip, p_base1, p_base2),
                (p_tip, p_base2, p_base3),
                (p_tip, p_base3, p_base4),
                (p_tip, p_base4, p_base1)
            ]
            for t1, t2, t3 in triangles:
                v1 = np.array(t2) - np.array(t1)
                v2 = np.array(t3) - np.array(t1)
                n = np.cross(v1, v2)
                norm = list(n / (np.linalg.norm(n) + 1e-6))
                verts.extend(t1 + norm + list(color))
                verts.extend(t2 + norm + list(color))
                verts.extend(t3 + norm + list(color))

        # 1. 좌/우현 쌍동 폰툰 선체 (순백색 선체 + 해양 시안 레이싱 스트라이프)
        hull_w = 0.28
        hull_h = 0.26
        hull_l = 1.45
        p_z = 0.38
        
        # 좌현 (Port, -Z)
        add_box([-0.05, 0.05, -p_z], [hull_l, hull_h, hull_w], [0.95, 0.96, 0.98])
        add_box([-0.05, 0.16, -p_z], [hull_l, 0.05, hull_w + 0.02], [0.00, 0.72, 0.95])
        # 우현 (Starboard, +Z)
        add_box([-0.05, 0.05, p_z], [hull_l, hull_h, hull_w], [0.95, 0.96, 0.98])
        add_box([-0.05, 0.16, p_z], [hull_l, 0.05, hull_w + 0.02], [0.00, 0.72, 0.95])
        
        # 2. 유선형 선수 쇄파 웨지 (Bow Cutwaters)
        b_x = 0.675
        t_x = 0.98
        add_wedge([t_x, 0.08, -p_z], [b_x, 0.18, -p_z - hull_w*0.5], [b_x, 0.18, -p_z + hull_w*0.5], [b_x, -0.08, -p_z + hull_w*0.5], [b_x, -0.08, -p_z - hull_w*0.5], [0.96, 0.97, 0.99])
        add_wedge([t_x, 0.08, p_z], [b_x, 0.18, p_z - hull_w*0.5], [b_x, 0.18, p_z + hull_w*0.5], [b_x, -0.08, p_z + hull_w*0.5], [b_x, -0.08, p_z - hull_w*0.5], [0.96, 0.97, 0.99])

        # 3. 중앙 알루미늄 브리지 갑판 (Center Deck)
        add_box([-0.05, 0.17, 0.0], [1.02, 0.05, 0.64], [0.24, 0.30, 0.38])
        
        # 4. 방수 전장 박스 (Electronics Enclosure)
        add_box([-0.08, 0.28, 0.0], [0.60, 0.18, 0.44], [0.12, 0.18, 0.25])
        add_box([-0.08, 0.38, -0.14], [0.05, 0.03, 0.05], [0.10, 0.95, 0.30]) # Status Green LED
        add_box([-0.08, 0.38, 0.14], [0.05, 0.03, 0.05], [0.98, 0.70, 0.10])  # Warning Amber LED

        # 5. 센서 마스트
        add_box([0.14, 0.48, 0.0], [0.05, 0.28, 0.05], [0.82, 0.84, 0.88])
        
        # 6. YDLIDAR TG15 라이다 센서 바디
        add_box([0.14, 0.64, 0.0], [0.16, 0.06, 0.16], [0.08, 0.08, 0.10])
        add_box([0.14, 0.68, 0.0], [0.14, 0.03, 0.14], [0.96, 0.76, 0.16]) # Gold Optics Ring

        boat_data = np.array(verts, dtype=np.float32)
        self.boat_count = len(boat_data) // 9
        self.boat_vbo = self.ctx.buffer(boat_data.tobytes())
        self.boat_vao = self.ctx.vertex_array(self.prog_mesh, [(self.boat_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _init_rudder_mesh(self):
        # 조타각에 연동되어 회전하는 선미 러더/추진기 모듈
        verts = []
        cx, cy, cz = 0.0, 0.0, 0.0
        sx, sy, sz = 0.16, 0.22, 0.05
        col = [0.15, 0.16, 0.18]
        faces = [
            ([cx-sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx+sx, cy+sy, cz+sz], [cx-sx, cy+sy, cz+sz], [0, 1, 0]),
            ([cx-sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz+sz], [cx+sx, cy-sy, cz-sz], [cx-sx, cy-sy, cz-sz], [0, -1, 0]),
            ([cx+sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz+sz], [cx+sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz-sz], [1, 0, 0]),
            ([cx-sx, cy-sy, cz+sz], [cx-sx, cy-sy, cz-sz], [cx-sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz+sz], [-1, 0, 0]),
            ([cx+sx, cy-sy, cz+sz], [cx-sx, cy-sy, cz+sz], [cx-sx, cy+sy, cz+sz], [cx+sx, cy+sy, cz+sz], [0, 0, 1]),
            ([cx-sx, cy-sy, cz-sz], [cx+sx, cy-sy, cz-sz], [cx+sx, cy+sy, cz-sz], [cx-sx, cy+sy, cz-sz], [0, 0, -1]),
        ]
        for p0, p1, p2, p3, n in faces:
            verts.extend(p0 + n + col)
            verts.extend(p1 + n + col)
            verts.extend(p2 + n + col)
            verts.extend(p0 + n + col)
            verts.extend(p2 + n + col)
            verts.extend(p3 + n + col)
        
        rudder_data = np.array(verts, dtype=np.float32)
        self.rudder_count = len(rudder_data) // 9
        self.rudder_vbo = self.ctx.buffer(rudder_data.tobytes())
        self.rudder_vao = self.ctx.vertex_array(self.prog_mesh, [(self.rudder_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _init_buoy_meshes(self):
        # 3차원 해상 항로 표지 부표 (원통 바디 + 고반사 띠 + 원추 헤드)
        def create_buoy_data(main_color):
            verts = []
            n_segs = 18
            r = 0.36
            h_cone = 0.50
            angles = np.linspace(0, 2*np.pi, n_segs, endpoint=False)
            
            for i in range(n_segs):
                a1 = angles[i]
                a2 = angles[(i + 1) % n_segs]
                x1, z1 = math.cos(a1) * r, math.sin(a1) * r
                x2, z2 = math.cos(a2) * r, math.sin(a2) * r
                n1 = [math.cos(a1), 0.0, math.sin(a1)]
                n2 = [math.cos(a2), 0.0, math.sin(a2)]
                col_white = [0.95, 0.95, 0.95]
                
                # 하단 몸체 (수면 아래 ~ 수면 위)
                verts.extend([x1, -0.30, z1] + n1 + main_color)
                verts.extend([x2, -0.30, z2] + n2 + main_color)
                verts.extend([x2, 0.35, z2] + n2 + main_color)
                verts.extend([x1, -0.30, z1] + n1 + main_color)
                verts.extend([x2, 0.35, z2] + n2 + main_color)
                verts.extend([x1, 0.35, z1] + n1 + main_color)
                
                # 중단 백색 반사띠
                verts.extend([x1, 0.35, z1] + n1 + col_white)
                verts.extend([x2, 0.35, z2] + n2 + col_white)
                verts.extend([x2, 0.58, z2] + n2 + col_white)
                verts.extend([x1, 0.35, z1] + n1 + col_white)
                verts.extend([x2, 0.58, z2] + n2 + col_white)
                verts.extend([x1, 0.58, z1] + n1 + col_white)
                
                # 상단 몸체
                verts.extend([x1, 0.58, z1] + n1 + main_color)
                verts.extend([x2, 0.58, z2] + n2 + main_color)
                verts.extend([x2, 0.82, z2] + n2 + main_color)
                verts.extend([x1, 0.58, z1] + n1 + main_color)
                verts.extend([x2, 0.82, z2] + n2 + main_color)
                verts.extend([x1, 0.82, z1] + n1 + main_color)
                
                # 상단 원추형 탑마크
                apex = [0.0, 0.82 + h_cone, 0.0]
                v_cone1 = np.array([x1, 0.82, z1])
                v_cone2 = np.array([x2, 0.82, z2])
                n_cone = np.cross(v_cone2 - np.array(apex), v_cone1 - np.array(apex))
                n_cone = list(n_cone / (np.linalg.norm(n_cone) + 1e-6))
                
                verts.extend(apex + n_cone + main_color)
                verts.extend([x1, 0.82, z1] + n_cone + main_color)
                verts.extend([x2, 0.82, z2] + n_cone + main_color)
            return np.array(verts, dtype=np.float32)

        # 1. 홍색 좌현표지 (Port Buoy)
        red_data = create_buoy_data([0.92, 0.22, 0.18])
        self.buoy_red_count = len(red_data) // 9
        self.buoy_red_vbo = self.ctx.buffer(red_data.tobytes())
        self.buoy_red_vao = self.ctx.vertex_array(self.prog_mesh, [(self.buoy_red_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])
        
        # 2. 녹색 우현표지 (Starboard Buoy)
        green_data = create_buoy_data([0.16, 0.75, 0.40])
        self.buoy_green_count = len(green_data) // 9
        self.buoy_green_vbo = self.ctx.buffer(green_data.tobytes())
        self.buoy_green_vao = self.ctx.vertex_array(self.prog_mesh, [(self.buoy_green_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _init_beacon_mesh(self):
        # 최종 목적지 에메랄드 항해 등대 비콘 타워
        verts = []
        n_segs = 16
        r_base = 0.65
        r_top = 0.25
        h_tower = 4.5
        angles = np.linspace(0, 2*np.pi, n_segs, endpoint=False)
        col_tower = [0.10, 0.90, 0.45]
        
        for i in range(n_segs):
            a1 = angles[i]; a2 = angles[(i + 1) % n_segs]
            x1_b, z1_b = math.cos(a1)*r_base, math.sin(a1)*r_base
            x2_b, z2_b = math.cos(a2)*r_base, math.sin(a2)*r_base
            x1_t, z1_t = math.cos(a1)*r_top, math.sin(a1)*r_top
            x2_t, z2_t = math.cos(a2)*r_top, math.sin(a2)*r_top
            n1 = [math.cos(a1), 0.2, math.sin(a1)]
            n2 = [math.cos(a2), 0.2, math.sin(a2)]
            
            verts.extend([x1_b, 0.0, z1_b] + n1 + col_tower)
            verts.extend([x2_b, 0.0, z2_b] + n2 + col_tower)
            verts.extend([x2_t, h_tower, z2_t] + n2 + col_tower)
            verts.extend([x1_b, 0.0, z1_b] + n1 + col_tower)
            verts.extend([x2_t, h_tower, z2_t] + n2 + col_tower)
            verts.extend([x1_t, h_tower, z1_t] + n1 + col_tower)
            
        beacon_data = np.array(verts, dtype=np.float32)
        self.beacon_count = len(beacon_data) // 9
        self.beacon_vbo = self.ctx.buffer(beacon_data.tobytes())
        self.beacon_vao = self.ctx.vertex_array(self.prog_mesh, [(self.beacon_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')])

    def _wave_height(self, x, z, t):
        w1 = math.sin(x * 0.60 + z * 0.40 - t * 2.2) * 0.11
        w2 = math.sin(x * 1.25 - z * 0.75 - t * 3.3) * 0.05
        w3 = math.cos(x * 2.10 + z * 1.40 - t * 4.5) * 0.025
        return w1 + w2 + w3

    def _matrix_perspective(self, fovy, aspect, near, far):
        f = 1.0 / math.tan(math.radians(fovy) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2.0 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    def _matrix_look_at(self, eye, target, up):
        eye = np.array(eye, dtype=np.float32)
        target = np.array(target, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        f = target - eye
        f /= (np.linalg.norm(f) + 1e-7)
        u = up / (np.linalg.norm(up) + 1e-7)
        s = np.cross(f, u)
        s /= (np.linalg.norm(s) + 1e-7)
        u = np.cross(s, f)
        m = np.identity(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m

    def _matrix_model(self, tx, ty, tz, yaw, pitch=0.0, roll=0.0, sx=1.0, sy=1.0, sz=1.0):
        # 올바른 3축 오일러 회전 행렬 연산 (Yaw * Pitch * Roll)
        cy, sy_ang = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        
        Ry = np.array([
            [cy, 0.0, sy_ang],
            [0.0, 1.0, 0.0],
            [-sy_ang, 0.0, cy]
        ], dtype=np.float32)
        
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp]
        ], dtype=np.float32)
        
        Rz = np.array([
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        R = Ry @ Rx @ Rz
        
        m = np.identity(4, dtype=np.float32)
        m[:3, 0] = R[:, 0] * sx
        m[:3, 1] = R[:, 1] * sy
        m[:3, 2] = R[:, 2] * sz
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz
        return m

    def render(self, env, hits, width=None, height=None):
        """
        메인 3D 렌더링 파이프라인
        - 오프스크린 FBO 렌더링 후 Pygame Surface로 변환 반환
        """
        w = width or self.width
        h = height or self.height
        fbo, col_tex, depth_rb = self._get_fbo(w, h)
            
        dt = getattr(env, 'dt', 0.04)
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
        sky_fog = (0.12, 0.28, 0.48)
        self.ctx.clear(0.08, 0.22, 0.38, 1.0, depth=1.0)
        
        light_dir = (0.55, 0.80, 0.30)
        
        # 5. [수면 렌더링] 동적 해양 셰이더 실행
        self.prog_ocean['u_vp'].write(VP.T.tobytes())
        self.prog_ocean['u_center'].value = (float(boat_x), float(boat_z))
        self.prog_ocean['u_time'].value = float(self.time)
        self.prog_ocean['u_cam_pos'].value = tuple(cam_eye)
        self.prog_ocean['u_light_dir'].value = light_dir
        self.prog_ocean['u_fog_color'].value = sky_fog
        self.ocean_vao.render()
        
        # 6. [부표 장애물 렌더링] 시야 반경 내 부표 표출
        self.prog_mesh['u_cam_pos'].value = tuple(cam_eye)
        self.prog_mesh['u_light_dir'].value = light_dir
        self.prog_mesh['u_fog_color'].value = sky_fog
        
        for idx, (ox, oy, r) in enumerate(env.dynamic_obstacles):
            obs_x = ox / 50.0
            obs_z = oy / 50.0
            dx = obs_x - boat_x; dz = obs_z - boat_z
            if dx*dx + dz*dz > 55.0 * 55.0:
                continue
                
            obs_y = self._wave_height(obs_x, obs_z, self.time)
            tilt_ang = math.sin(self.time * 2.0 + obs_x * 0.5) * 0.06
            
            # 수직 기립 상태의 3D 부표 매트릭스
            M_buoy = self._matrix_model(obs_x, obs_y, obs_z, 0.0, pitch=0.0, roll=tilt_ang)
            MVP_buoy = VP @ M_buoy
            self.prog_mesh['u_model'].write(M_buoy.T.tobytes())
            self.prog_mesh['u_mvp'].write(MVP_buoy.T.tobytes())
            
            # 짝수/홀수 인덱스에 따라 홍색/녹색 부표 분기
            if idx % 2 == 0:
                self.buoy_red_vao.render()
            else:
                self.buoy_green_vao.render()
                
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
                    
                    # 2 Triangles for Quad
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

        # 10. FBO 버퍼를 초고속(10,000+ FPS)으로 읽어와 Pygame Surface로 변환
        raw_pixels = fbo.read(components=4)
        surf_3d = pygame.image.frombuffer(raw_pixels, (w, h), 'RGBA')
        surf_3d = pygame.transform.flip(surf_3d, False, True) # OpenGL Y축 반전 보정
        
        # 11. 3D 패널 오버레이 HUD 계기판 장식
        self._draw_hud_overlay(surf_3d, env, speed, steer, heading)
        
        return surf_3d

    def _draw_hud_overlay(self, surf, env, speed, steer, heading):
        w, h = surf.get_size()
        
        # [1] 테두리 프레임
        pygame.draw.rect(surf, (0, 200, 255), (0, 0, w, h), 2)
        
        # [2] 해상도별 반응형 폰트 및 바 크기 결정
        is_large = (w > 600)
        hdr_h = 34 if is_large else 24
        info_h = 26 if is_large else 20
        f_title = self.large_font if is_large else self.font
        f_hint = self.large_info_font if is_large else self.micro_font
        f_info = self.large_info_font if is_large else self.micro_font
        
        # 상단 타이틀 바
        header_surf = pygame.Surface((w, hdr_h), pygame.SRCALPHA)
        header_surf.fill((10, 24, 42, 220))
        surf.blit(header_surf, (0, 0))
        pygame.draw.line(surf, (0, 160, 230), (0, hdr_h), (w, hdr_h), 1)
        
        lbl_title = f_title.render(f"3D Engine ({self.cam_names[self.cam_mode]})", True, (255, 255, 255))
        surf.blit(lbl_title, (215 if is_large else 8, 7 if is_large else 4))
        
        # 단축키 안내 힌트
        lbl_hint = f_hint.render("[C: Camera View / V: Toggle Full 3D]", True, (0, 230, 255))
        surf.blit(lbl_hint, (w - lbl_hint.get_width() - (12 if is_large else 8), 8 if is_large else 6))
        
        # [3] 1인칭 조타석 뷰 전용 조타 HUD (인공 수평선 피치 사다리 및 나침반)
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
            
            # 상단 디지털 나침반 테이프
            hdg_deg = int(math.degrees(heading)) % 360
            lbl_hdg = (self.large_font if is_large else self.micro_font).render(f"HEADING {hdg_deg:03d}°", True, (0, 255, 220))
            surf.blit(lbl_hdg, (cx - lbl_hdg.get_width() // 2, hdr_h + 8))
            
        # [4] 하단 인포 바 (선속 및 러더 각도)
        info_y = h - info_h
        pygame.draw.rect(surf, (8, 18, 30, 210), (0, info_y, w, info_h))
        pygame.draw.line(surf, (0, 140, 200), (0, info_y), (w, info_y), 1)
        
        steer_deg = math.degrees(steer)
        knots = speed * 1.94384
        lbl_stat = f_info.render(
            f"SPEED: {speed:.1f} m/s ({knots:.1f} kt) | RUDDER: {steer_deg:+.1f}° | BACKEND: ModernGL 3.3 Core Profile",
            True, (225, 242, 255)
        )
        surf.blit(lbl_stat, (12 if is_large else 8, info_y + (5 if is_large else 3)))
