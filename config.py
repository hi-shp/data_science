import os
import subprocess

def detect_optimal_window_size(aspect_ratio=2.0, stadium_ratio=(20, 7), margin_w=9, margin_h=40):
    """
    모든 컴퓨터 및 모니터 환경에서 작업 영역(Dock, Taskbar, 상단 패널)을 실시간 측정하여
    기존 2:1 화면비와 20:7 경기장 비율을 엄격히 유지하는 최대 크기 해상도를 자동 계산합니다.
    """
    avail_w = None
    avail_h = None
    
    # 1. Linux X11/GNOME/KDE 작업 영역 (_NET_WORKAREA) 실시간 감지
    if os.name != 'nt' and 'DISPLAY' in os.environ:
        try:
            out = subprocess.check_output(
                ['xprop', '-root', '_NET_WORKAREA'],
                stderr=subprocess.DEVNULL,
                timeout=1
            ).decode()
            if '=' in out:
                raw_nums = out.split('=', 1)[1].strip().split(',')
                if len(raw_nums) >= 4:
                    work_w = int(raw_nums[2].strip())
                    work_h = int(raw_nums[3].strip())
                    if work_w > 600 and work_h > 400:
                        avail_w = work_w - margin_w
                        avail_h = work_h - margin_h
        except Exception:
            pass

    # 2. Windows 작업 영역 (SPI_GETWORKAREA) 실시간 감지 (작업표시줄 제외)
    if (avail_w is None or avail_h is None) and os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            # SPI_GETWORKAREA = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                work_w = rect.right - rect.left
                work_h = rect.bottom - rect.top
                if work_w > 600 and work_h > 400:
                    avail_w = work_w - margin_w
                    avail_h = work_h - margin_h
        except Exception:
            pass

    # 3. Pygame 디스플레이 정보 기반 Fallback (macOS 및 기타 환경)
    if avail_w is None or avail_h is None:
        try:
            import pygame
            if not pygame.get_init():
                pygame.init()
            if hasattr(pygame.display, 'get_desktop_sizes'):
                sizes = pygame.display.get_desktop_sizes()
                if sizes and len(sizes) > 0:
                    desk_w, desk_h = sizes[0]
                else:
                    info = pygame.display.Info()
                    desk_w, desk_h = info.current_w, info.current_h
            else:
                info = pygame.display.Info()
                desk_w, desk_h = info.current_w, info.current_h
            
            avail_w = int(desk_w * 0.95)
            avail_h = int(desk_h * 0.90)
        except Exception:
            return 1800, 900, 630, 270

    # 2:1 종횡비 제약 조건 하에서 가로/세로 중 화면에 가장 꽉 차는 최대 가로폭 산출
    max_w_from_h = int(avail_h * aspect_ratio)
    target_w = min(avail_w, max_w_from_h)
    
    # 20:7 경기장 비율 및 4px 점유 그리드 연산이 완벽한 정수로 떨어지도록 20의 배수로 스냅
    div = stadium_ratio[0]
    target_w = (target_w // div) * div
    target_w = max(1200, target_w)
    
    target_h = int(target_w / aspect_ratio)
    sim_h = int(target_w * stadium_ratio[1] / stadium_ratio[0])
    dash_h = target_h - sim_h
    
    return target_w, target_h, sim_h, dash_h

def get_dashboard_layout(width, sim_h):
    """
    모든 해상도에서 하단 대시보드의 5개 모니터링 패널을 화면 가로폭에 맞추어 균등 정렬 배치합니다.
    """
    base_widths = [340, 320, 320, 190, 190]
    total_panels_w = sum(base_widths)  # 1360
    left_ctrl_w = 340
    
    if width >= left_ctrl_w + total_panels_w + 40:
        gap = (width - left_ctrl_w - total_panels_w) // 5
        widths = list(base_widths)
        ctrl_w = left_ctrl_w
    else:
        scale = width / 1840.0
        widths = [int(w * scale) for w in base_widths]
        ctrl_w = int(left_ctrl_w * scale)
        gap = max(4, (width - ctrl_w - sum(widths)) // 5)

    p1_x = ctrl_w + gap
    p2_x = p1_x + widths[0] + gap
    p3_x = p2_x + widths[1] + gap
    p4_x = p3_x + widths[2] + gap
    p5_x = p4_x + widths[3] + gap
    y = sim_h + 35

    return {
        'widths': widths,
        'p1_x': p1_x, 'p2_x': p2_x, 'p3_x': p3_x, 'p4_x': p4_x, 'p5_x': p5_x,
        'y': y, 'gap': gap
    }

# 환경 변수 KABOAT_WIDTH 지정 시 수동 오버라이드 가능, 미지정 시 시스템 실시간 자동 감지
_env_w = os.environ.get('KABOAT_WIDTH')
if _env_w and _env_w.isdigit():
    WIDTH = (int(_env_w) // 20) * 20
    HEIGHT = WIDTH // 2
    SIM_H = WIDTH * 7 // 20
    DASH_H = HEIGHT - SIM_H
else:
    WIDTH, HEIGHT, SIM_H, DASH_H = detect_optimal_window_size()

# 맵 확장 배율 (기본값: 1배)
MAP_SCALE = 1
MAP_W = WIDTH * MAP_SCALE

GRID = 4
GRID_W = MAP_W // GRID   # 전체 맵 점유 그리드 가로 크기
GRID_H = HEIGHT // GRID  # 전체 맵 점유 그리드 세로 크기