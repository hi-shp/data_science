import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from scipy.interpolate import splprep, splev

# Font setup - using Noto Sans CJK JP which contains full Korean glyphs
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/soonhong/kaboat/report4'
SUBFIG_DIR = os.path.join(OUTPUT_DIR, 'sheet1_subfigures')
os.makedirs(SUBFIG_DIR, exist_ok=True)

MASTER_SHEET_FILE = os.path.join(OUTPUT_DIR, 'sheet1_background_and_problem.png')

# Color Palette
COLOR_BG = '#FFFFFF'
COLOR_PANEL = '#F8FAFC'
COLOR_BORDER = '#CBD5E1'
COLOR_BORDER_STRONG = '#94A3B8'
COLOR_TEXT_MAIN = '#0F172A'
COLOR_TEXT_SUB = '#334155'
COLOR_PRIMARY = '#0284C7'
COLOR_ACCENT = '#0369A1'
COLOR_WARN = '#EA580C'
COLOR_DANGER = '#DC2626'
COLOR_DANGER_BG = '#FEF2F2'
COLOR_DANGER_BORDER = '#FCA5A5'
COLOR_SUCCESS = '#16A34A'
COLOR_SUCCESS_BG = '#F0FDF4'
COLOR_SUCCESS_BORDER = '#86EFAC'

def draw_boat(ax, x, y, heading_rad, length=0.85, width=0.42, color='#0284C7', ec='#0F172A', alpha=0.95, zorder=10):
    hl = length * 0.5
    hw = width * 0.5
    dw = width * 0.28
    
    p_left = np.array([
        [-hl, hw - dw],
        [hl*0.55, hw - dw],
        [hl, hw],
        [hl*0.55, hw + dw],
        [-hl, hw + dw]
    ])
    p_right = np.array([
        [-hl, -hw - dw],
        [hl*0.55, -hw - dw],
        [hl, -hw],
        [hl*0.55, -hw + dw],
        [-hl, -hw + dw]
    ])
    p_deck = np.array([
        [-hl*0.7, -hw*0.9],
        [hl*0.35, -hw*0.9],
        [hl*0.35, hw*0.9],
        [-hl*0.7, hw*0.9]
    ])
    
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    R = np.array([[c, -s], [s, c]])
    
    for p, fc in [(p_deck, '#E2E8F0'), (p_left, color), (p_right, color)]:
        p_rot = (R @ p.T).T + np.array([x, y])
        poly = Polygon(p_rot, closed=True, facecolor=fc, edgecolor=ec, lw=1.2, alpha=alpha, zorder=zorder)
        ax.add_patch(poly)
    
    # Heading arrow
    ax.annotate('', xy=(x + length*0.65*c, y + length*0.65*s), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.6), zorder=zorder+2)

# ==============================================================================
# SUBFIGURE 1 GENERATOR: Line Tracer & Boat Adaptation
# ==============================================================================
def generate_subfig1():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax1 = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax1.set_facecolor(COLOR_PANEL)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5.5)
    ax1.axis('off')
    
    ax1.add_patch(FancyBboxPatch((0.05, 0.05), 9.9, 5.4, boxstyle="round,pad=0.08", 
                                 fc=COLOR_PANEL, ec=COLOR_BORDER_STRONG, lw=1.4))
    ax1.text(0.35, 5.18, "센서 반사 제어의 착안: 지상 라인트레이서 모티브와 보트 적용 원리", 
             fontsize=14.0, fontweight='bold', color=COLOR_ACCENT, va='top')
    
    # Left Column: Ground Line Tracer Robot
    ax1.add_patch(FancyBboxPatch((0.22, 0.25), 4.58, 4.60, boxstyle="round,pad=0.06", 
                                 fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax1.text(2.51, 4.58, "지상 라인트레이서 로봇의 제어 원리", ha='center', va='top', 
             fontsize=13.0, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    ax1.add_patch(Rectangle((0.45, 2.15), 4.12, 2.05, fc='#F1F5F9', ec='#CBD5E1', lw=1.0, zorder=2))
    t_x = np.linspace(0.60, 4.40, 150)
    t_y = 3.10 + 0.50 * np.sin((t_x - 0.60) * 1.5)
    ax1.plot(t_x, t_y, color='#0F172A', lw=8.0, zorder=3)
    ax1.text(0.65, 3.90, "검은색 주행선", fontsize=10.5, fontweight='bold', color='#0F172A', zorder=5)
    
    rx, ry = 2.45, 3.30
    r_th = np.deg2rad(22)
    rc, rs = np.cos(r_th), np.sin(r_th)
    
    ax1.add_patch(Rectangle((rx - 0.44, ry - 0.34), 0.88, 0.68, angle=np.rad2deg(r_th),
                            rotation_point=(rx, ry), fc='#38BDF8', ec='#0369A1', lw=1.4, zorder=6))
    ax1.add_patch(Rectangle((rx - 0.38, ry + 0.33), 0.28, 0.13, angle=np.rad2deg(r_th),
                            rotation_point=(rx, ry), fc='#334155', ec='#0F172A', lw=1.2, zorder=7))
    ax1.add_patch(Rectangle((rx - 0.38, ry - 0.46), 0.28, 0.13, angle=np.rad2deg(r_th),
                            rotation_point=(rx, ry), fc='#334155', ec='#0F172A', lw=1.2, zorder=7))
    
    sx = rx + 0.44 * rc
    sy = ry + 0.44 * rs
    ax1.add_patch(Circle((sx, sy), 0.11, fc='#DC2626', ec='#7F1D1D', lw=1.3, zorder=8))
    ax1.annotate('광센서 1개\n(선 경계면 감지)', xy=(sx, sy), xytext=(sx - 0.85, sy + 0.50),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.3),
                 fontsize=10.0, fontweight='bold', color='#DC2626', zorder=9)
    
    ax1.text(3.70, 3.85, "검은 선 감지 시 → 좌회전", ha='center', va='center',
             fontsize=10.0, fontweight='bold', color='#1E40AF',
             bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#93C5FD', lw=0.9), zorder=10)
    ax1.text(3.70, 2.45, "흰 바탕 감지 시 → 우회전", ha='center', va='center',
             fontsize=10.0, fontweight='bold', color='#B45309',
             bbox=dict(boxstyle='round,pad=0.2', fc='#FEF3C7', ec='#FCD34D', lw=0.9), zorder=10)
    
    ax1.text(2.51, 1.72, "1. 센서가 검은 선을 감지하면 좌회전합니다.", ha='center', va='center', fontsize=11.0, color='#0F172A')
    ax1.text(2.51, 1.38, "2. 센서가 흰 바탕을 감지하면 우회전합니다.", ha='center', va='center', fontsize=11.0, color='#0F172A')
    ax1.text(2.51, 0.76, "단순한 반사 규칙만으로도 곡선 경로를 이탈 없이 추종합니다.", 
             ha='center', va='center', fontsize=11.2, fontweight='bold', color='#0F172A')
    
    # Right Column: Boat Adaptation
    ax1.add_patch(FancyBboxPatch((5.20, 0.25), 4.58, 4.60, boxstyle="round,pad=0.06", 
                                 fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax1.text(7.49, 4.58, "보트 적용: 라인트레이싱 알고리즘", ha='center', va='top', 
             fontsize=13.0, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    ax1.add_patch(Rectangle((5.42, 2.05), 4.14, 2.15, fc='#F0F9FF', ec='#BAE6FD', lw=1.0, zorder=2))
    for wy in [2.35, 2.85, 3.35, 3.85]:
        ax1.plot([5.50, 9.45], [wy, wy], color='#E0F2FE', lw=1.2, ls=':', zorder=3)
        
    draw_boat(ax1, 6.15, 2.85, 0.0, length=1.05, width=0.52, color=COLOR_PRIMARY)
    
    goal_x, goal_y = 9.25, 2.85
    ax1.plot([goal_x, goal_x], [goal_y - 0.25, goal_y + 0.65], color='#15803D', lw=2.2, zorder=8)
    ax1.add_patch(Polygon([(goal_x, goal_y + 0.65), (goal_x + 0.38, goal_y + 0.45), (goal_x, goal_y + 0.25)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.1, zorder=9))
    ax1.text(goal_x, goal_y + 0.80, "목적지", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#15803D')
    
    ax1.plot([6.75, goal_x], [2.85, goal_y], color='#10B981', lw=1.6, ls='--', zorder=4)
    ax1.text(7.85, 3.05, "1순위: 목적지 추종", ha='center', va='bottom', fontsize=10.0, fontweight='bold', color='#047857')
    
    buoy_gx, buoy_gy = 7.75, 3.65
    ax1.add_patch(Circle((buoy_gx, buoy_gy), 0.24, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax1.text(buoy_gx, buoy_gy + 0.35, "장애물 부표", ha='center', va='bottom', fontsize=10.8, fontweight='bold', color='#9A3412')
    
    ax1.annotate('2순위: 센서 감지', xy=(buoy_gx - 0.20, buoy_gy - 0.15), xytext=(buoy_gx - 1.10, buoy_gy - 0.15),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.4),
                 ha='center', va='center', fontsize=9.8, fontweight='bold', color='#DC2626',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FEF2F2', ec='#FCA5A5', lw=0.9), zorder=12)
    
    ax1.annotate('우현 회피 조타!', xy=(6.55, 2.55), xytext=(6.55, 1.85),
                 arrowprops=dict(arrowstyle='->', color=COLOR_PRIMARY, lw=1.4),
                 ha='center', va='top', fontsize=10.2, fontweight='bold', color=COLOR_PRIMARY,
                 bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#93C5FD', lw=0.9), zorder=12)
    
    ax1.text(7.49, 1.72, "1. 평상시에는 목적지를 향해 직진 주행합니다.", ha='center', va='center', fontsize=11.0, color='#0F172A')
    ax1.text(7.49, 1.38, "2. 장애물이 감지되면 단순 반사 규칙으로 회피합니다.", ha='center', va='center', fontsize=11.0, color='#0F172A')
    ax1.text(7.49, 0.76, "간단한 반사 코드만으로도 장애물을 피해 목표에 도달합니다.", 
             ha='center', va='center', fontsize=11.2, fontweight='bold', color='#0F172A')
    
    out_path = os.path.join(SUBFIG_DIR, 'subfig1_linetracer_and_boat_adaptation.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 1:", out_path)

# ==============================================================================
# SUBFIGURE 2 GENERATOR: Initial Detour Success
# ==============================================================================
def generate_subfig2():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax2 = fig.add_axes([0.07, 0.07, 0.90, 0.88])
    ax2.set_facecolor(COLOR_PANEL)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5.5)
    
    ax2.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax2.set_title("초기 주행: 개방 수역에서 0.5m 안전거리를 유지하며 목표에 도달했습니다.", 
                  fontsize=13.2, fontweight='bold', pad=10, color=COLOR_TEXT_MAIN, loc='left')
    ax2.set_xlabel("전진 방향 X 좌표 (m)", fontsize=12.0, labelpad=5, color=COLOR_TEXT_SUB)
    ax2.set_ylabel("횡방향 Y 좌표 (m)", fontsize=12.0, labelpad=5, color=COLOR_TEXT_SUB)
    ax2.tick_params(labelsize=10.5)
    
    # Destination Goal
    gx, gy = 9.3, 2.7
    ax2.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.5, zorder=8)
    ax2.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.45, gy + 0.55), (gx, gy + 0.3)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax2.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=12.0, fontweight='bold', color='#15803D')
    
    ax2.plot([0.8, gx], [2.7, gy], color='#94A3B8', lw=1.5, ls=':', zorder=1, label='원래 목표 직진 경로')
    
    # Single Buoy
    buoy_x, buoy_y = 5.2, 3.4
    safe_circle = Circle((buoy_x, buoy_y), 0.90, fc='#FEF3C7', ec=COLOR_WARN, lw=1.4, ls='--', alpha=0.7, zorder=2)
    ax2.add_patch(safe_circle)
    ax2.add_patch(Circle((buoy_x, buoy_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.8, zorder=8))
    ax2.text(buoy_x, buoy_y + 0.35, "단일 부표", ha='center', va='bottom', fontsize=12.0, fontweight='bold', color='#9A3412')
    ax2.text(buoy_x, buoy_y + 1.05, "안전 거리 (0.5m 설정)", ha='center', va='bottom', fontsize=11.0, color='#B45309')
    
    pts = np.array([
        [0.8, 2.7],
        [2.8, 2.65],
        [4.2, 1.75],
        [5.2, 1.35],
        [6.5, 1.75],
        [7.8, 2.55],
        [gx, gy]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=3)
    u_new = np.linspace(0, 1, 200)
    path_x, path_y = splev(u_new, tck)
    dx = np.gradient(path_x)
    dy = np.gradient(path_y)
    headings = np.arctan2(dy, dx)
    
    ax2.plot(path_x, path_y, color=COLOR_PRIMARY, lw=3.2, zorder=4, label='선박 실제 주행 궤적')
    
    sample_indices = [10, 65, 130, 185]
    boat_labels = [
        "1. 목적지를 향해 직진합니다.",
        "2. 부표를 감지하고 우현으로 조타합니다.",
        "3. 0.5m 안전거리를 유지하며 통과합니다.",
        "4. 원래 경로로 복귀하여 도달합니다."
    ]
    label_pos = [
        (path_x[10], path_y[10] - 0.65),
        (path_x[65], path_y[65] - 0.65),
        (path_x[130], path_y[130] - 0.65),
        (path_x[185] - 0.15, path_y[185] - 0.65)
    ]
    
    for idx, blabel, (lx, ly) in zip(sample_indices, boat_labels, label_pos):
        x_i = path_x[idx]
        y_i = path_y[idx]
        hd_i = headings[idx]
        draw_boat(ax2, x_i, y_i, hd_i, length=0.85, width=0.42, color='#0284C7', zorder=10)
        ax2.text(lx, ly, blabel, ha='center', va='center', 
                 fontsize=10.5, fontweight='bold', color=COLOR_TEXT_MAIN,
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_BORDER, lw=0.8, alpha=0.95), zorder=15)
        
    ax2.text(5.0, 0.40, "단일 장애물 환경에서는 단순 반사 규칙만으로도 목표 지점에 도달했습니다.",
             ha='center', va='center', fontsize=11.8, fontweight='bold', color='#0F172A', zorder=20)
    ax2.legend(loc='upper left', fontsize=11.0, framealpha=0.9)
    
    out_path = os.path.join(SUBFIG_DIR, 'subfig2_initial_success_trajectory.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 2:", out_path)

# ==============================================================================
# SUBFIGURE 3 GENERATOR: Limitation 1 - Gate Closure
# ==============================================================================
def generate_subfig3():
    fig = plt.figure(figsize=(7.5, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax3 = fig.add_axes([0.10, 0.08, 0.86, 0.86])
    ax3.set_facecolor(COLOR_PANEL)
    ax3.set_xlim(0, 7.0)
    ax3.set_ylim(0, 7.5)
    
    ax3.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax3.set_title("한계 1: 안전마진 중첩으로 게이트 통로가 폐쇄됩니다.", 
                  fontsize=13.0, fontweight='bold', pad=10, color=COLOR_DANGER, loc='left')
    ax3.set_xlabel("X 좌표 (m)", fontsize=12.0, labelpad=5, color=COLOR_TEXT_SUB)
    ax3.set_ylabel("Y 좌표 (m)", fontsize=12.0, labelpad=5, color=COLOR_TEXT_SUB)
    ax3.tick_params(labelsize=11.0)
    
    # Outer Concrete Wall
    ax3.axhline(7.0, color='#64748B', lw=4.0, zorder=5)
    ax3.text(0.4, 7.15, "수조 외곽 콘크리트 벽", fontsize=11.5, fontweight='bold', color='#475569')
    
    b1_x, b1_y = 4.0, 4.0
    b2_x, b2_y = 4.0, 3.2
    margin_r = 0.5
    
    c1 = Circle((b1_x, b1_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    c2 = Circle((b2_x, b2_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    ax3.add_patch(c1)
    ax3.add_patch(c2)
    
    # Overlap Hatch
    ax3.fill_between([3.7, 4.0, 4.3], [3.6, 3.7, 3.6], [3.6, 3.5, 3.6], 
                     color='#DC2626', alpha=0.35, hatch='///', zorder=3)
    
    ax3.add_patch(Circle((b1_x, b1_y), 0.16, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax3.add_patch(Circle((b2_x, b2_y), 0.16, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax3.text(b1_x + 0.22, b1_y, "부표 A", ha='left', va='center', fontsize=11.5, fontweight='bold', color='#9A3412')
    ax3.text(b2_x + 0.22, b2_y, "부표 B", ha='left', va='center', fontsize=11.5, fontweight='bold', color='#9A3412')
    
    ax3.annotate('안전마진 중첩 구간\n(0.2m 겹침 발생)', xy=(3.9, 3.6), xytext=(2.2, 2.7),
                 arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.4),
                 ha='center', va='center', fontsize=10.8, fontweight='bold', color='#991B1B',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=15)
    
    ax3.annotate('', xy=(4.0, 3.95), xytext=(4.0, 3.25),
                 arrowprops=dict(arrowstyle='<->', color='#0F172A', lw=1.5), zorder=10)
    ax3.text(4.85, 3.6, "실제 통로\n폭 0.8m", ha='left', va='center', fontsize=11.0, fontweight='bold', color='#0F172A')

    # Margin dimension arrow
    ax3.annotate('', xy=(3.5, 4.0), xytext=(4.0, 4.0),
                 arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.2), zorder=10)
    ax3.text(3.75, 4.22, "안전마진 0.5m", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#DC2626')
    
    # Abort Trajectory
    pts = np.array([
        [0.8, 3.6],
        [2.2, 3.6],
        [2.9, 4.1],
        [3.5, 5.3],
        [4.2, 6.8]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=2)
    u_new = np.linspace(0, 1, 100)
    px, py = splev(u_new, tck)
    ax3.plot(px, py, color=COLOR_DANGER, lw=3.0, zorder=6, ls='-', label='회피 이탈 궤적')
    
    ax3.plot(4.2, 6.9, marker='X', markersize=16, color='#DC2626', markeredgecolor='#7F1D1D', zorder=20)
    ax3.text(4.45, 6.65, "외곽벽 충돌!", fontsize=12.0, fontweight='bold', color='#DC2626')
    
    draw_boat(ax3, 1.1, 3.6, 0.0, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax3, 2.7, 3.9, np.pi/5, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax3, 3.7, 5.7, np.pi/2.7, length=0.8, width=0.4, color=COLOR_DANGER)
    
    ax3.annotate('', xy=(6.5, 3.6), xytext=(4.8, 3.6),
                 arrowprops=dict(arrowstyle='->', color='#16A34A', lw=2.2, ls='--'), zorder=4)
    ax3.text(5.6, 3.9, "목표 게이트 출구", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#16A34A')
    
    ax3.add_patch(FancyBboxPatch((0.45, 0.40), 6.10, 1.48, boxstyle="round,pad=0.08", 
                                 fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2, zorder=24))
    ax3.text(3.5, 1.55, "1. 통로 폭(0.8m)보다 안전마진 합(1.0m)이 더 큽니다.", 
             ha='center', va='center', fontsize=10.5, color='#991B1B', fontweight='bold', zorder=25)
    ax3.text(3.5, 1.14, "2. 안전마진이 중첩되어 열린 통로를 벽으로 오판합니다.", 
             ha='center', va='center', fontsize=10.5, color='#991B1B', fontweight='bold', zorder=25)
    ax3.text(3.5, 0.72, "3. 통로 진입을 회피하다 외곽 수조 벽에 충돌합니다.", 
             ha='center', va='center', fontsize=10.5, color='#991B1B', fontweight='bold', zorder=25)
    
    out_path = os.path.join(SUBFIG_DIR, 'subfig3_limitation1_gate_closure.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 3:", out_path)

# ==============================================================================
# SUBFIGURE 4 GENERATOR: Limitation 2 - Chattering with Overlapping Boats
# ==============================================================================
def generate_subfig4():
    fig = plt.figure(figsize=(7.5, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)

    fig.text(0.08, 0.965, "한계 2: 매 프레임 단순 반사 계산의 반복으로 선체가 좌우로 진동합니다.", 
             fontsize=12.5, fontweight='bold', color=COLOR_DANGER, va='top')

    # Top Subplot: Overlapping Boats along nominal straight line
    ax_top = fig.add_axes([0.08, 0.52, 0.88, 0.40])
    ax_top.set_facecolor('#FFFFFF')
    ax_top.set_xlim(0.5, 9.5)
    ax_top.set_ylim(-3.7, 3.7)

    for spine in ax_top.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    ax_top.text(5.0, 3.35, "선체 거동: 매 프레임 단순 반사 계산으로 인해 직선 주행 중 좌우 요동 발생", 
                ha='center', va='top', fontsize=11.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=30)

    # Nominal straight path line down center
    ax_top.plot([0.6, 9.4], [0.0, 0.0], color='#94A3B8', lw=2.0, ls=':', zorder=5)

    # Smooth sinusoidal trajectory and 11 overlapping boats oscillating along wave
    x_fine = np.linspace(0.8, 9.2, 300)
    A = 0.75
    wavelength = 2.6
    y_fine = A * np.sin(2 * np.pi * (x_fine - 0.8) / wavelength)
    ax_top.plot(x_fine, y_fine, color='#DC2626', lw=2.0, ls='--', alpha=0.7, zorder=6)

    boat_x_samples = np.linspace(1.2, 8.8, 11)
    boat_y_samples = A * np.sin(2 * np.pi * (boat_x_samples - 0.8) / wavelength)
    dy_dx = A * (2 * np.pi / wavelength) * np.cos(2 * np.pi * (boat_x_samples - 0.8) / wavelength)
    boat_headings = np.arctan2(dy_dx, 1.0)

    for i, (bx, by, bhd) in enumerate(zip(boat_x_samples, boat_y_samples, boat_headings)):
        draw_boat(ax_top, bx, by, bhd, length=1.40, width=0.70, color='#38BDF8', ec='#0284C7', alpha=0.55, zorder=10 + i)

    ax_top.text(2.1, 2.30, "좌현 반사 조타", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#DC2626', 
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.9), zorder=35)
    ax_top.text(3.4, -2.30, "우현 반사 조타", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0284C7', 
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#93C5FD', lw=0.9), zorder=35)
    ax_top.text(5.0, -3.35, "시간순 중첩 관찰: 단순 반사 제어를 반복하여 직선 경로에서도 좌우로 요동치며 전진합니다.", 
                ha='center', va='bottom', fontsize=10.2, fontweight='bold', color='#475569', zorder=35)

    # Bottom Subplot: Quantitative Rudder Angle Time-Series Chattering
    ax_bot = fig.add_axes([0.08, 0.08, 0.88, 0.38])
    ax_bot.set_facecolor(COLOR_PANEL)
    ax_bot.set_xlim(0, 10)
    ax_bot.set_ylim(-36, 18)
    ax_bot.set_yticks([-10, 0, 10])

    ax_bot.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax_bot.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)

    ax_bot.set_xlabel("주행 시간 (s)", fontsize=12.0, labelpad=4, color=COLOR_TEXT_SUB)
    ax_bot.set_ylabel("조타각 (deg)", fontsize=12.0, labelpad=4, color=COLOR_TEXT_SUB)
    ax_bot.tick_params(labelsize=10.5)

    t = np.linspace(0, 10, 500)
    chatter = np.clip(10.0 * np.sin(2 * np.pi * 4.5 * t) + np.random.normal(0, 0.4, len(t)), -11, 11)

    ax_bot.plot(t, chatter, color='#DC2626', lw=1.2, zorder=4)
    ax_bot.axhline(0, color='#64748B', lw=1.0, ls='--', zorder=2)

    ax_bot.annotate('매 제어 주기마다 즉각 반사 조타 (좌우 고주파 진동)', xy=(3.5, 10), xytext=(3.5, 14),
                    ha='center', va='bottom',
                    fontsize=10.2, fontweight='bold', color='#B91C1C',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=20)

    ax_bot.add_patch(FancyBboxPatch((0.5, -34.5), 9.0, 16.5, boxstyle="round,pad=0.4", 
                                    fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2, zorder=24))
    ax_bot.text(5.0, -21.5, "1. 매 제어 프레임마다 단순 반사 규칙을 반복 계산하여 조타합니다.", 
                ha='center', va='center', fontsize=10.2, color='#991B1B', fontweight='bold', zorder=25)
    ax_bot.text(5.0, -26.5, "2. 상태 제어나 완충 없이 즉각 반응하므로 좌우로 심하게 진동합니다.", 
                ha='center', va='center', fontsize=10.2, color='#991B1B', fontweight='bold', zorder=25)
    ax_bot.text(5.0, -31.5, "3. 진동으로 인해 조타 구동부가 마모되고 전진 추진 효율이 저하됩니다.", 
                ha='center', va='center', fontsize=10.2, color='#991B1B', fontweight='bold', zorder=25)

    out_path = os.path.join(SUBFIG_DIR, 'subfig4_violent_chattering_overlapping_boats.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 4:", out_path)

# ==============================================================================
# SUBFIGURE 5 GENERATOR: Limitation 3 - Convincing Multi-Obstacle Collision
# ==============================================================================
def generate_subfig5():
    fig = plt.figure(figsize=(7.5, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax5 = fig.add_axes([0.10, 0.08, 0.86, 0.86])
    ax5.set_facecolor(COLOR_PANEL)
    ax5.set_xlim(0, 8.0)
    ax5.set_ylim(0, 8.0)

    ax5.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax5.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)

    ax5.set_title("한계 3: 단일 반사 제어로는 다중 장애물 충돌을 방지하지 못합니다.", 
                 fontsize=13.0, fontweight='bold', pad=10, color=COLOR_DANGER, loc='left')
    ax5.set_xlabel("전진 거리 X (m)", fontsize=12.0, labelpad=5, color=COLOR_TEXT_SUB)
    ax5.set_ylabel("횡방향 위치 Y (m)", fontsize=12.0, labelpad=5, color=COLOR_TEXT_SUB)
    ax5.tick_params(labelsize=11.0)

    # Top Context Banner
    ax5.add_patch(FancyBboxPatch((0.25, 6.70), 7.50, 1.05, boxstyle="round,pad=0.04",
                                 fc='#FFFFFF', ec=COLOR_BORDER_STRONG, lw=1.1, zorder=20))
    ax5.text(4.0, 7.52, "환경 지도가 없어 주변 장애물 배치를 인식하지 못합니다.", 
            ha='center', va='top', fontsize=11.0, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=21)
    ax5.text(4.0, 7.02, "장애물 A를 회피하는 과정에서 인접한 장애물 B와 충돌합니다.",
            ha='center', va='center', fontsize=10.2, color='#475569', zorder=21)

    # Destination Goal at (7.3, 2.5)
    gx, gy = 7.3, 2.5
    ax5.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.2, zorder=8)
    ax5.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.40, gy + 0.55), (gx, gy + 0.3)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax5.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=11.5, fontweight='bold', color='#15803D')

    # Nominal Target Path
    ax5.plot([0.5, gx], [2.5, gy], color='#94A3B8', lw=1.8, ls=':', zorder=1)
    ax5.text(5.8, 2.15, "원래 목표 직진 경로", fontsize=10.5, color='#64748B', fontweight='bold', zorder=2)

    # Obstacle A directly on the path at (3.4, 2.5)
    bA_x, bA_y = 3.4, 2.5
    ax5.add_patch(Circle((bA_x, bA_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax5.add_patch(Circle((bA_x, bA_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
    ax5.text(bA_x, bA_y - 0.35, "장애물 A (직진 차단)", ha='center', va='top', fontsize=10.5, fontweight='bold', color='#9A3412')

    # Obstacle B positioned offset to the left/upward at (5.2, 4.4)
    bB_x, bB_y = 5.2, 4.4
    ax5.add_patch(Circle((bB_x, bB_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax5.add_patch(Circle((bB_x, bB_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
    ax5.text(bB_x + 0.40, bB_y + 0.15, "장애물 B\n(사각지대 위치)", ha='left', va='center', fontsize=11.0, fontweight='bold', color='#9A3412')

    # Boat 1: Initial starting approach
    draw_boat(ax5, 0.9, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

    # Boat 2: At detection point
    draw_boat(ax5, 2.1, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

    # Sensor Beam detecting Obstacle A
    ax5.plot([2.55, bA_x - 0.25], [2.5, 2.5], color='#DC2626', lw=2.2, ls='-', zorder=15)
    ax5.annotate('센서 감지 (1.2m)\n즉시 좌현 회피 조타!', xy=(2.7, 2.5), xytext=(1.8, 3.7),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.4),
                fontsize=10.2, fontweight='bold', color='#DC2626',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.9), zorder=25)

    # Reactive avoidance trajectory swinging left directly into Obstacle B!
    pts = np.array([
        [2.1, 2.5],
        [2.8, 2.8],
        [3.7, 3.5],
        [4.5, 4.0],
        [5.0, 4.35]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=2)
    u_new = np.linspace(0, 1, 100)
    px, py = splev(u_new, tck)
    ax5.plot(px, py, color=COLOR_DANGER, lw=3.0, ls='-', zorder=5)

    # Boat 3: Mid-turn
    hd3 = np.arctan2(py[55] - py[50], px[55] - px[50])
    draw_boat(ax5, 3.6, 3.42, hd3, length=0.95, width=0.48, color='#38BDF8', alpha=0.75, zorder=10)

    # Boat 4: Crashing directly into Obstacle B
    hd4 = np.arctan2(py[-1] - py[-5], px[-1] - px[-5])
    draw_boat(ax5, 5.0, 4.35, hd4, length=0.95, width=0.48, color='#DC2626', zorder=12)

    # Crash Flash & Big Red X at Obstacle B
    ax5.plot(5.15, 4.4, marker='X', markersize=20, color='#DC2626', markeredgecolor='#7F1D1D', markeredgewidth=2.0, zorder=30)
    ax5.annotate('장애물 A 회피 중 장애물 B와 충돌!\n(인접 장애물 위치 미인식)', 
                xy=(5.15, 4.4), xytext=(3.4, 5.7),
                arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=1.8),
                fontsize=10.8, fontweight='bold', color='#991B1B',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#DC2626', lw=1.4), zorder=35)

    ax5.add_patch(FancyBboxPatch((0.5, 0.15), 7.0, 1.35, boxstyle="round,pad=0.06", 
                                 fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2, zorder=24))
    ax5.text(4.0, 1.25, "1. 전방 장애물 A를 감지한 직후 좌현으로 회피합니다.", 
             ha='center', va='center', fontsize=10.5, color='#991B1B', fontweight='bold', zorder=25)
    ax5.text(4.0, 0.85, "2. 인접 장애물 B를 인식하지 못해 충돌 경로로 진입합니다.", 
             ha='center', va='center', fontsize=10.5, color='#991B1B', fontweight='bold', zorder=25)
    ax5.text(4.0, 0.45, "3. 단순 반사 규칙으로는 다중 장애물 환경에 대응하지 못합니다.", 
             ha='center', va='center', fontsize=10.5, color='#991B1B', fontweight='bold', zorder=25)

    out_path = os.path.join(SUBFIG_DIR, 'subfig5_limitation3_scalability_collision.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 5:", out_path)

# ==============================================================================
# MASTER SHEET 1 COMPILER
# ==============================================================================
def generate_sheet_1():
    print("Generating all 5 individual subfigures first...")
    generate_subfig1()
    generate_subfig2()
    generate_subfig3()
    generate_subfig4()
    generate_subfig5()
    print("All individual subfigures successfully generated.")
    print("Compiling Master Sheet 1...")
    fig = plt.figure(figsize=(18.0, 12.0), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    
    # --------------------------------------------------------------------------
    # TOP MASTER HEADLINE (Prominent single-sentence summary, Strictly 1 line!)
    # --------------------------------------------------------------------------
    fig.text(0.025, 0.970, "라인트레이싱의 장애물 반사 제어로 초기 단일 회피에는 성공했으나, 실제 환경에서의 진동 문제와 확장성 한계에 직면했습니다.", 
             fontsize=19.5, fontweight='bold', color=COLOR_TEXT_MAIN, va='top')
    
    # --------------------------------------------------------------------------
    # SUBPLOT 1: 라인트레이서 vs 자율운항보트 개념 (left: 0.025, bottom: 0.525, width: 0.465, height: 0.380)
    # --------------------------------------------------------------------------
    ax1 = fig.add_axes([0.025, 0.525, 0.465, 0.380])
    ax1.set_facecolor(COLOR_PANEL)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5.5)
    ax1.axis('off')
    
    # Outer Card Frame
    ax1.add_patch(FancyBboxPatch((0.05, 0.05), 9.9, 5.4, boxstyle="round,pad=0.08", 
                                 fc=COLOR_PANEL, ec=COLOR_BORDER_STRONG, lw=1.4))
    ax1.text(0.35, 5.18, "센서 반사 제어의 착안: 지상 라인트레이서 모티브와 보트 적용 원리", 
             fontsize=13.5, fontweight='bold', color=COLOR_ACCENT, va='top')
    
    # Left Column: Ground Line Tracer Robot
    ax1.add_patch(FancyBboxPatch((0.22, 0.25), 4.58, 4.60, boxstyle="round,pad=0.06", 
                                 fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax1.text(2.51, 4.58, "지상 라인트레이서 로봇의 제어 원리", ha='center', va='top', 
             fontsize=12.5, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    ax1.add_patch(Rectangle((0.45, 2.15), 4.12, 2.05, fc='#F1F5F9', ec='#CBD5E1', lw=1.0, zorder=2))
    t_x = np.linspace(0.60, 4.40, 150)
    t_y = 3.10 + 0.50 * np.sin((t_x - 0.60) * 1.5)
    ax1.plot(t_x, t_y, color='#0F172A', lw=8.0, zorder=3)
    ax1.text(0.65, 3.90, "검은색 주행선", fontsize=10.2, fontweight='bold', color='#0F172A', zorder=5)
    
    rx, ry = 2.45, 3.30
    r_th = np.deg2rad(22)
    rc, rs = np.cos(r_th), np.sin(r_th)
    
    ax1.add_patch(Rectangle((rx - 0.44, ry - 0.34), 0.88, 0.68, angle=np.rad2deg(r_th),
                            rotation_point=(rx, ry), fc='#38BDF8', ec='#0369A1', lw=1.4, zorder=6))
    ax1.add_patch(Rectangle((rx - 0.38, ry + 0.33), 0.28, 0.13, angle=np.rad2deg(r_th),
                            rotation_point=(rx, ry), fc='#334155', ec='#0F172A', lw=1.2, zorder=7))
    ax1.add_patch(Rectangle((rx - 0.38, ry - 0.46), 0.28, 0.13, angle=np.rad2deg(r_th),
                            rotation_point=(rx, ry), fc='#334155', ec='#0F172A', lw=1.2, zorder=7))
    
    sx = rx + 0.44 * rc
    sy = ry + 0.44 * rs
    ax1.add_patch(Circle((sx, sy), 0.11, fc='#DC2626', ec='#7F1D1D', lw=1.3, zorder=8))
    ax1.annotate('광센서 1개\n(선 경계면 감지)', xy=(sx, sy), xytext=(sx - 0.85, sy + 0.50),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.3),
                 fontsize=9.8, fontweight='bold', color='#DC2626', zorder=9)
    
    ax1.text(3.70, 3.85, "검은 선 감지 시 → 좌회전", ha='center', va='center',
             fontsize=10.0, fontweight='bold', color='#1E40AF',
             bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#93C5FD', lw=0.9), zorder=10)
    ax1.text(3.70, 2.45, "흰 바탕 감지 시 → 우회전", ha='center', va='center',
             fontsize=10.0, fontweight='bold', color='#B45309',
             bbox=dict(boxstyle='round,pad=0.2', fc='#FEF3C7', ec='#FCD34D', lw=0.9), zorder=10)
    
    ax1.text(2.51, 1.72, "1. 센서가 검은 선을 감지하면 좌회전합니다.", ha='center', va='center', fontsize=10.6, color='#0F172A')
    ax1.text(2.51, 1.38, "2. 센서가 흰 바탕을 감지하면 우회전합니다.", ha='center', va='center', fontsize=10.6, color='#0F172A')
    ax1.text(2.51, 0.76, "단순한 반사 규칙만으로도 곡선 경로를 이탈 없이 추종합니다.", 
             ha='center', va='center', fontsize=10.8, fontweight='bold', color='#0F172A')
    
    # Right Column: Boat Adaptation
    ax1.add_patch(FancyBboxPatch((5.20, 0.25), 4.58, 4.60, boxstyle="round,pad=0.06", 
                                 fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax1.text(7.49, 4.58, "보트 적용: 라인트레이싱 알고리즘", ha='center', va='top', 
             fontsize=12.5, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    ax1.add_patch(Rectangle((5.42, 2.05), 4.14, 2.15, fc='#F0F9FF', ec='#BAE6FD', lw=1.0, zorder=2))
    for wy in [2.35, 2.85, 3.35, 3.85]:
        ax1.plot([5.50, 9.45], [wy, wy], color='#E0F2FE', lw=1.2, ls=':', zorder=3)
        
    draw_boat(ax1, 6.15, 2.85, 0.0, length=1.05, width=0.52, color=COLOR_PRIMARY)
    
    goal_x, goal_y = 9.25, 2.85
    ax1.plot([goal_x, goal_x], [goal_y - 0.25, goal_y + 0.65], color='#15803D', lw=2.2, zorder=8)
    ax1.add_patch(Polygon([(goal_x, goal_y + 0.65), (goal_x + 0.35, goal_y + 0.45), (goal_x, goal_y + 0.25)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax1.text(goal_x, goal_y + 0.78, "목적지", ha='center', va='bottom', fontsize=10.2, fontweight='bold', color='#15803D')
    
    ax1.plot([6.70, goal_x], [2.85, 2.85], color='#22C55E', lw=1.5, ls='--', zorder=4)
    ax1.text(7.05, 2.62, "1순위: 목적지 추종", fontsize=9.8, fontweight='bold', color='#15803D')
    
    buoy_x, buoy_y = 7.75, 3.55
    ax1.add_patch(Circle((buoy_x, buoy_y), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=8))
    ax1.text(buoy_x, buoy_y + 0.32, "장애물 부표", ha='center', va='bottom', fontsize=10.2, fontweight='bold', color='#9A3412')
    
    ax1.plot([6.70, buoy_x - 0.15], [2.85, buoy_y - 0.12], color='#DC2626', lw=1.8, ls='--', zorder=5)
    ax1.text(6.65, 3.55, "2순위: 센서 감지", fontsize=9.4, fontweight='bold', color='#DC2626', zorder=10,
             bbox=dict(boxstyle='round,pad=0.15', fc='#FFFFFF', ec='#FCA5A5', lw=0.8))
    
    ax1.annotate('우현 회피 조타!', xy=(6.15, 2.58), xytext=(6.15, 2.15),
                 arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.5),
                 fontsize=10.2, fontweight='bold', color='#0284C7', zorder=10)
    
    ax1.text(7.49, 1.72, "1. 평상시에는 목적지를 향해 직진 주행합니다.", ha='center', va='center', fontsize=10.6, color='#0F172A')
    ax1.text(7.49, 1.38, "2. 장애물이 감지되면 단순 반사 규칙으로 회피합니다.", ha='center', va='center', fontsize=10.6, color='#0F172A')
    ax1.text(7.49, 0.76, "간단한 반사 코드만으로도 장애물을 피해 목표에 도달합니다.", 
             ha='center', va='center', fontsize=10.8, fontweight='bold', color='#0F172A')
    
    # --------------------------------------------------------------------------
    # SUBPLOT 2: 초기 개방 수역 회피 성공 궤적 (left: 0.510, bottom: 0.525, width: 0.465, height: 0.380)
    # --------------------------------------------------------------------------
    ax2 = fig.add_axes([0.510, 0.525, 0.465, 0.380])
    ax2.set_facecolor(COLOR_PANEL)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5.5)
    
    ax2.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax2.set_title("초기 주행: 개방 수역에서 0.5m 안전거리를 유지하며 목표에 도달했습니다.", 
                  fontsize=12.2, fontweight='bold', pad=7, color=COLOR_TEXT_MAIN, loc='left')
    ax2.set_xlabel("전진 방향 X 좌표 (m)", fontsize=11.2, labelpad=4, color=COLOR_TEXT_SUB)
    ax2.set_ylabel("횡방향 Y 좌표 (m)", fontsize=11.2, labelpad=4, color=COLOR_TEXT_SUB)
    ax2.tick_params(labelsize=10.0)
    
    gx, gy = 9.3, 2.7
    ax2.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.5, zorder=8)
    ax2.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.45, gy + 0.55), (gx, gy + 0.3)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax2.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#15803D')
    
    ax2.plot([0.8, gx], [2.7, gy], color='#94A3B8', lw=1.5, ls=':', zorder=1, label='원래 목표 직진 경로')
    
    buoy_x, buoy_y = 5.2, 3.4
    safe_circle = Circle((buoy_x, buoy_y), 0.90, fc='#FEF3C7', ec=COLOR_WARN, lw=1.4, ls='--', alpha=0.7, zorder=2)
    ax2.add_patch(safe_circle)
    ax2.add_patch(Circle((buoy_x, buoy_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.8, zorder=8))
    ax2.text(buoy_x, buoy_y + 0.35, "단일 부표", ha='center', va='bottom', fontsize=11.0, fontweight='bold', color='#9A3412')
    ax2.text(buoy_x, buoy_y + 1.05, "안전 거리 (0.5m 설정)", ha='center', va='bottom', fontsize=10.2, color='#B45309')
    
    pts = np.array([
        [0.8, 2.7],
        [2.8, 2.65],
        [4.2, 1.75],
        [5.2, 1.35],
        [6.5, 1.75],
        [7.8, 2.55],
        [gx, gy]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=3)
    u_new = np.linspace(0, 1, 200)
    path_x, path_y = splev(u_new, tck)
    dx = np.gradient(path_x)
    dy = np.gradient(path_y)
    headings = np.arctan2(dy, dx)
    
    ax2.plot(path_x, path_y, color=COLOR_PRIMARY, lw=3.2, zorder=4, label='선박 실제 주행 궤적')
    
    sample_indices = [10, 65, 130, 185]
    boat_labels = [
        "1. 목적지를 향해 직진합니다.",
        "2. 부표를 감지하고 우현으로 조타합니다.",
        "3. 0.5m 안전거리를 유지하며 통과합니다.",
        "4. 원래 경로로 복귀하여 도달합니다."
    ]
    label_pos = [
        (path_x[10], path_y[10] - 0.65),
        (path_x[65], path_y[65] - 0.65),
        (path_x[130], path_y[130] - 0.65),
        (path_x[185] - 0.15, path_y[185] - 0.65)
    ]
    
    for idx, blabel, (lx, ly) in zip(sample_indices, boat_labels, label_pos):
        x_i = path_x[idx]
        y_i = path_y[idx]
        hd_i = headings[idx]
        draw_boat(ax2, x_i, y_i, hd_i, length=0.85, width=0.42, color='#0284C7', zorder=10)
        ax2.text(lx, ly, blabel, ha='center', va='center', 
                 fontsize=10.0, fontweight='bold', color=COLOR_TEXT_MAIN,
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_BORDER, lw=0.8, alpha=0.95), zorder=15)
        
    ax2.text(5.0, 0.40, "단일 장애물 환경에서는 단순 반사 규칙만으로도 목표 지점에 도달했습니다.",
             ha='center', va='center', fontsize=11.0, fontweight='bold', color='#0F172A', zorder=20)
    ax2.legend(loc='upper left', fontsize=10.2, framealpha=0.9)

    # --------------------------------------------------------------------------
    # MIDDLE SECTION HEADER (y: 0.485, perfectly centered, sub-sentence removed!)
    # --------------------------------------------------------------------------
    fig.text(0.025, 0.485, "실제 환경에서 직면한 라인트레이싱 알고리즘의 3대 구조적 한계", 
             fontsize=20.0, fontweight='bold', color=COLOR_DANGER, va='center')

    # --------------------------------------------------------------------------
    # SUBPLOT 3: 한계 1 - 게이트 폐쇄 (left: 0.025, bottom: 0.050, width: 0.304, height: 0.385)
    # --------------------------------------------------------------------------
    ax3 = fig.add_axes([0.025, 0.050, 0.304, 0.385])
    ax3.set_facecolor(COLOR_PANEL)
    ax3.set_xlim(0, 7.0)
    ax3.set_ylim(0, 7.5)
    
    ax3.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax3.set_title("한계 1: 안전마진 중첩으로 게이트 통로가 폐쇄됩니다.", 
                  fontsize=11.8, fontweight='bold', pad=7, color=COLOR_DANGER, loc='left')
    ax3.set_xlabel("X 좌표 (m)", fontsize=11.2, labelpad=4, color=COLOR_TEXT_SUB)
    ax3.set_ylabel("Y 좌표 (m)", fontsize=11.2, labelpad=4, color=COLOR_TEXT_SUB)
    ax3.tick_params(labelsize=10.0)
    
    # Outer Concrete Wall
    ax3.axhline(7.0, color='#64748B', lw=4.0, zorder=5)
    ax3.text(0.4, 7.15, "수조 외곽 콘크리트 벽", fontsize=10.8, fontweight='bold', color='#475569')
    
    b1_x, b1_y = 4.0, 4.0
    b2_x, b2_y = 4.0, 3.2
    margin_r = 0.5
    
    c1 = Circle((b1_x, b1_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    c2 = Circle((b2_x, b2_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    ax3.add_patch(c1)
    ax3.add_patch(c2)
    
    # Overlap Hatch
    ax3.fill_between([3.7, 4.0, 4.3], [3.6, 3.7, 3.6], [3.6, 3.5, 3.6], 
                     color='#DC2626', alpha=0.35, hatch='///', zorder=3)
    
    ax3.add_patch(Circle((b1_x, b1_y), 0.16, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax3.add_patch(Circle((b2_x, b2_y), 0.16, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax3.text(b1_x + 0.22, b1_y, "부표 A", ha='left', va='center', fontsize=11.2, fontweight='bold', color='#9A3412')
    ax3.text(b2_x + 0.22, b2_y, "부표 B", ha='left', va='center', fontsize=11.2, fontweight='bold', color='#9A3412')
    
    ax3.annotate('안전마진 중첩 구간\n(0.2m 겹침 발생)', xy=(3.9, 3.6), xytext=(2.2, 2.7),
                 arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.4),
                 ha='center', va='center', fontsize=9.8, fontweight='bold', color='#991B1B',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=15)
    
    ax3.annotate('', xy=(4.0, 3.95), xytext=(4.0, 3.25),
                 arrowprops=dict(arrowstyle='<->', color='#0F172A', lw=1.5), zorder=10)
    ax3.text(4.85, 3.6, "실제 통로\n폭 0.8m", ha='left', va='center', fontsize=10.2, fontweight='bold', color='#0F172A')

    # Margin dimension arrow
    ax3.annotate('', xy=(3.5, 4.0), xytext=(4.0, 4.0),
                 arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.2), zorder=10)
    ax3.text(3.75, 4.22, "안전마진 0.5m", ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#DC2626')
    
    # Abort Trajectory
    pts = np.array([
        [0.8, 3.6],
        [2.2, 3.6],
        [2.9, 4.1],
        [3.5, 5.3],
        [4.2, 6.8]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=2)
    u_new = np.linspace(0, 1, 100)
    px, py = splev(u_new, tck)
    ax3.plot(px, py, color=COLOR_DANGER, lw=3.0, zorder=6, ls='-', label='회피 이탈 궤적')
    
    ax3.plot(4.2, 6.9, marker='X', markersize=16, color='#DC2626', markeredgecolor='#7F1D1D', zorder=20)
    ax3.text(4.45, 6.65, "외곽벽 충돌!", fontsize=11.2, fontweight='bold', color='#DC2626')
    
    draw_boat(ax3, 1.1, 3.6, 0.0, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax3, 2.7, 3.9, np.pi/5, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax3, 3.7, 5.7, np.pi/2.7, length=0.8, width=0.4, color=COLOR_DANGER)
    
    ax3.annotate('', xy=(6.5, 3.6), xytext=(4.8, 3.6),
                 arrowprops=dict(arrowstyle='->', color='#16A34A', lw=2.2, ls='--'), zorder=4)
    ax3.text(5.6, 3.9, "목표 게이트 출구", ha='center', va='bottom', fontsize=10.2, fontweight='bold', color='#16A34A')
    
    ax3.add_patch(FancyBboxPatch((0.45, 0.40), 6.10, 1.48, boxstyle="round,pad=0.08", 
                                 fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2, zorder=24))
    ax3.text(3.5, 1.55, "1. 통로 폭(0.8m)보다 안전마진 합(1.0m)이 더 큽니다.", 
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold', zorder=25)
    ax3.text(3.5, 1.14, "2. 안전마진이 중첩되어 열린 통로를 벽으로 오판합니다.", 
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold', zorder=25)
    ax3.text(3.5, 0.72, "3. 통로 진입을 회피하다 외곽 수조 벽에 충돌합니다.", 
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold', zorder=25)

    # --------------------------------------------------------------------------
    # SUBPLOT 4: 한계 2 - 극심한 진동과 선체 불안정성 (left: 0.347, width: 0.304)
    # Divided cleanly into ax4_top (boats motion) and ax4_bot (rudder graph)
    # --------------------------------------------------------------------------
    # Top Card: 11 semi-transparent boats oscillating along sinusoidal wave
    ax4_top = fig.add_axes([0.347, 0.256, 0.304, 0.176])
    ax4_top.set_facecolor('#FFFFFF')
    ax4_top.set_xlim(0.5, 9.5)
    ax4_top.set_ylim(-3.7, 3.7)

    for spine in ax4_top.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
    ax4_top.set_xticks([])
    ax4_top.set_yticks([])

    ax4_top.set_title("한계 2: 매 프레임 단순 반사 계산의 반복으로 선체가 좌우로 진동합니다.", 
                      fontsize=11.8, fontweight='bold', pad=7, color=COLOR_DANGER, loc='left')

    ax4_top.text(5.0, 3.35, "선체 거동: 매 프레임 단순 반사 계산으로 인해 직선 주행 중 좌우 요동 발생", 
                 ha='center', va='top', fontsize=10.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=30)

    # Nominal straight line
    ax4_top.plot([0.6, 9.4], [0.0, 0.0], color='#94A3B8', lw=2.0, ls=':', zorder=5)

    # Smooth sinusoidal trajectory and 11 overlapping boats oscillating along wave
    x_fine = np.linspace(0.8, 9.2, 300)
    A = 0.75
    wavelength = 2.6
    y_fine = A * np.sin(2 * np.pi * (x_fine - 0.8) / wavelength)
    ax4_top.plot(x_fine, y_fine, color='#DC2626', lw=2.0, ls='--', alpha=0.7, zorder=6)

    boat_x_samples = np.linspace(1.2, 8.8, 11)
    boat_y_samples = A * np.sin(2 * np.pi * (boat_x_samples - 0.8) / wavelength)
    dy_dx = A * (2 * np.pi / wavelength) * np.cos(2 * np.pi * (boat_x_samples - 0.8) / wavelength)
    boat_headings = np.arctan2(dy_dx, 1.0)

    for i, (bx, by, bhd) in enumerate(zip(boat_x_samples, boat_y_samples, boat_headings)):
        draw_boat(ax4_top, bx, by, bhd, length=1.40, width=0.70, color='#38BDF8', ec='#0284C7', alpha=0.55, zorder=10 + i)

    ax4_top.text(2.1, 2.30, "좌현 반사 조타", ha='center', va='center', fontsize=9.6, fontweight='bold', color='#DC2626', 
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.9), zorder=35)
    ax4_top.text(3.4, -2.30, "우현 반사 조타", ha='center', va='center', fontsize=9.6, fontweight='bold', color='#0284C7', 
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#93C5FD', lw=0.9), zorder=35)
    ax4_top.text(5.0, -3.35, "시간순 중첩 관찰: 단순 반사 제어를 반복하여 직선 경로에서도 좌우로 요동치며 전진합니다.", 
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#475569', zorder=35)

    # Bottom Card: Rudder Angle Chattering Time Series
    ax4_bot = fig.add_axes([0.347, 0.050, 0.304, 0.172])
    ax4_bot.set_facecolor(COLOR_PANEL)
    ax4_bot.set_xlim(0, 10)
    ax4_bot.set_ylim(-36, 18)
    ax4_bot.set_yticks([-10, 0, 10])

    ax4_bot.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax4_bot.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)

    ax4_bot.set_xlabel("주행 시간 (s)", fontsize=11.2, labelpad=3, color=COLOR_TEXT_SUB)
    ax4_bot.set_ylabel("조타각 (deg)", fontsize=11.2, labelpad=3, color=COLOR_TEXT_SUB)
    ax4_bot.tick_params(labelsize=9.8)

    t = np.linspace(0, 10, 500)
    chatter = np.clip(10.0 * np.sin(2 * np.pi * 4.5 * t) + np.random.normal(0, 0.4, len(t)), -11, 11)

    ax4_bot.plot(t, chatter, color='#DC2626', lw=1.2, zorder=4)
    ax4_bot.axhline(0, color='#64748B', lw=1.0, ls='--', zorder=2)

    ax4_bot.annotate('매 제어 주기마다 즉각 반사 조타 (좌우 고주파 진동)', xy=(3.5, 10), xytext=(3.5, 14),
                     ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold', color='#B91C1C',
                     bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=20)

    ax4_bot.add_patch(FancyBboxPatch((0.5, -34.5), 9.0, 16.5, boxstyle="round,pad=0.4", 
                                     fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2, zorder=24))
    ax4_bot.text(5.0, -21.5, "1. 매 제어 프레임마다 단순 반사 규칙을 반복 계산하여 조타합니다.", 
                 ha='center', va='center', fontsize=9.6, color='#991B1B', fontweight='bold', zorder=25)
    ax4_bot.text(5.0, -26.5, "2. 상태 제어나 완충 없이 즉각 반응하므로 좌우로 심하게 진동합니다.", 
                 ha='center', va='center', fontsize=9.6, color='#991B1B', fontweight='bold', zorder=25)
    ax4_bot.text(5.0, -31.5, "3. 진동으로 인해 조타 구동부가 마모되고 전진 추진 효율이 저하됩니다.", 
                 ha='center', va='center', fontsize=9.6, color='#991B1B', fontweight='bold', zorder=25)

    # --------------------------------------------------------------------------
    # SUBPLOT 5: 한계 3 - 확장성의 한계 & 다중 장애물 연쇄 충돌 (left: 0.669, bottom: 0.050, width: 0.304, height: 0.385)
    # --------------------------------------------------------------------------
    ax5 = fig.add_axes([0.669, 0.050, 0.304, 0.385])
    ax5.set_facecolor(COLOR_PANEL)
    ax5.set_xlim(0, 8.0)
    ax5.set_ylim(0, 8.0)

    ax5.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax5.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)

    ax5.set_title("한계 3: 단일 반사 제어로는 다중 장애물 충돌을 방지하지 못합니다.", 
                  fontsize=11.8, fontweight='bold', pad=7, color=COLOR_DANGER, loc='left')
    ax5.set_xlabel("전진 거리 X (m)", fontsize=11.2, labelpad=4, color=COLOR_TEXT_SUB)
    ax5.set_ylabel("횡방향 위치 Y (m)", fontsize=11.2, labelpad=4, color=COLOR_TEXT_SUB)
    ax5.tick_params(labelsize=10.0)

    # Top Context Banner
    ax5.add_patch(FancyBboxPatch((0.25, 6.70), 7.50, 1.05, boxstyle="round,pad=0.04",
                                 fc='#FFFFFF', ec=COLOR_BORDER_STRONG, lw=1.1, zorder=20))
    ax5.text(4.0, 7.52, "환경 지도가 없어 주변 장애물 배치를 인식하지 못합니다.", 
             ha='center', va='top', fontsize=10.0, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=21)
    ax5.text(4.0, 7.02, "장애물 A를 회피하는 과정에서 인접한 장애물 B와 충돌합니다.",
             ha='center', va='center', fontsize=9.4, color='#475569', zorder=21)

    # Destination Goal at (7.3, 2.5)
    gx, gy = 7.3, 2.5
    ax5.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.2, zorder=8)
    ax5.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.40, gy + 0.55), (gx, gy + 0.3)],
                          closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax5.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#15803D')

    # Nominal Target Path
    ax5.plot([0.5, gx], [2.5, gy], color='#94A3B8', lw=1.8, ls=':', zorder=1)
    ax5.text(5.8, 1.85, "원래 목표 직진 경로", fontsize=9.6, color='#64748B', fontweight='bold', zorder=2)

    # Obstacle A directly on path at (3.4, 2.5)
    bA_x, bA_y = 3.4, 2.5
    ax5.add_patch(Circle((bA_x, bA_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax5.add_patch(Circle((bA_x, bA_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
    ax5.text(bA_x, bA_y - 0.35, "장애물 A (직진 차단)", ha='center', va='top', fontsize=9.8, fontweight='bold', color='#9A3412')

    # Obstacle B offset at (5.2, 4.4)
    bB_x, bB_y = 5.2, 4.4
    ax5.add_patch(Circle((bB_x, bB_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax5.add_patch(Circle((bB_x, bB_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
    ax5.text(bB_x + 0.40, bB_y + 0.15, "장애물 B\n(사각지대 위치)", ha='left', va='center', fontsize=10.2, fontweight='bold', color='#9A3412')

    # Boat 1: Initial approach
    draw_boat(ax5, 0.9, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

    # Boat 2: At detection point
    draw_boat(ax5, 2.1, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

    # Sensor Beam detecting A
    ax5.plot([2.55, bA_x - 0.25], [2.5, 2.5], color='#DC2626', lw=2.2, ls='-', zorder=15)
    ax5.annotate('센서 감지 (1.2m)\n즉시 좌현 회피 조타!', xy=(2.7, 2.5), xytext=(1.8, 3.7),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.4),
                 fontsize=9.5, fontweight='bold', color='#DC2626',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.9), zorder=25)

    # Reactive avoidance arc into Obstacle B!
    pts = np.array([
        [2.1, 2.5],
        [2.8, 2.8],
        [3.7, 3.5],
        [4.5, 4.0],
        [5.0, 4.35]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=2)
    u_new = np.linspace(0, 1, 100)
    px, py = splev(u_new, tck)
    ax5.plot(px, py, color=COLOR_DANGER, lw=3.0, ls='-', zorder=5)

    # Boat 3: Mid-turn
    hd3 = np.arctan2(py[55] - py[50], px[55] - px[50])
    draw_boat(ax5, 3.6, 3.42, hd3, length=0.95, width=0.48, color='#38BDF8', alpha=0.75, zorder=10)

    # Boat 4: Crashed at B
    hd4 = np.arctan2(py[-1] - py[-5], px[-1] - px[-5])
    draw_boat(ax5, 5.0, 4.35, hd4, length=0.95, width=0.48, color='#DC2626', zorder=12)

    # Crash Flash & Big Red X at Obstacle B
    ax5.plot(5.15, 4.4, marker='X', markersize=20, color='#DC2626', markeredgecolor='#7F1D1D', markeredgewidth=2.0, zorder=30)
    ax5.annotate('장애물 A 회피 중 장애물 B와 충돌!\n(인접 장애물 위치 미인식)', 
                 xy=(5.15, 4.4), xytext=(3.4, 5.7),
                 arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=1.8),
                 fontsize=10.0, fontweight='bold', color='#991B1B',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#DC2626', lw=1.4), zorder=35)

    ax5.add_patch(FancyBboxPatch((0.5, 0.15), 7.0, 1.35, boxstyle="round,pad=0.06", 
                                 fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2, zorder=24))
    ax5.text(4.0, 1.25, "1. 전방 장애물 A를 감지한 직후 좌현으로 회피합니다.", 
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold', zorder=25)
    ax5.text(4.0, 0.85, "2. 인접 장애물 B를 인식하지 못해 충돌 경로로 진입합니다.", 
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold', zorder=25)
    ax5.text(4.0, 0.45, "3. 단순 반사 규칙으로는 다중 장애물 환경에 대응하지 못합니다.", 
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold', zorder=25)

    # Save Master Sheet
    plt.savefig(MASTER_SHEET_FILE, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print(f"Master Sheet 1 successfully generated: {MASTER_SHEET_FILE}")

if __name__ == '__main__':
    generate_sheet_1()
