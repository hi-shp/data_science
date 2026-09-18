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

OUTPUT_FILE = '/home/soonhong/kaboat/report4/sheet1_background_and_problem.png'

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

def generate_sheet_1():
    fig = plt.figure(figsize=(18.0, 12.0), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    
    # --------------------------------------------------------------------------
    # TOP MASTER HEADLINE (Prominent single-sentence summary for passersby)
    # --------------------------------------------------------------------------
    fig.text(0.030, 0.978, "단순한 라인트레이싱 반사 제어로 초기 단일 회피에는 성공했으나, 경기 수조에서는 진동과 확장성의 한계에 직면했습니다.", 
             fontsize=19.0, fontweight='bold', color=COLOR_TEXT_MAIN, va='top')
    fig.text(0.030, 0.946, "지상 라인트레이서의 단순 반사 제어를 모티브로 한 '라인트레이싱 알고리즘'의 개발 배경과 실제 경기 환경에서의 3대 구조적 한계 규명", 
             fontsize=12.2, color=COLOR_TEXT_SUB, va='top')
    
    # --------------------------------------------------------------------------
    # SUBPLOT 1: 라인트레이서 vs 자율운항보트 개념 (left: 0.030 ~ 0.485, bottom: 0.540 ~ 0.895, height: 0.355)
    # --------------------------------------------------------------------------
    ax1 = fig.add_axes([0.030, 0.540, 0.455, 0.355])
    ax1.set_facecolor(COLOR_PANEL)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5.5)
    ax1.axis('off')
    
    # Outer Card Frame
    ax1.add_patch(FancyBboxPatch((0.05, 0.05), 9.9, 5.4, boxstyle="round,pad=0.08", 
                                 fc=COLOR_PANEL, ec=COLOR_BORDER_STRONG, lw=1.4))
    ax1.text(0.35, 5.18, "센서 반사 제어의 착안: 지상 라인트레이서 모티브와 보트 적용 원리", 
             fontsize=13.5, fontweight='bold', color=COLOR_ACCENT, va='top')
    
    # Left Column: Ground Line Tracer Robot (Intuitive principle: Black line detection)
    ax1.add_patch(FancyBboxPatch((0.22, 0.25), 4.58, 4.60, boxstyle="round,pad=0.06", 
                                 fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax1.text(2.51, 4.58, "지상 라인트레이서 로봇의 제어 원리", ha='center', va='top', 
             fontsize=12.5, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    # White floor with clear S-curved black track
    ax1.add_patch(Rectangle((0.45, 2.15), 4.12, 2.05, fc='#F1F5F9', ec='#CBD5E1', lw=1.0, zorder=2))
    t_x = np.linspace(0.60, 4.40, 150)
    t_y = 3.10 + 0.50 * np.sin((t_x - 0.60) * 1.5)
    ax1.plot(t_x, t_y, color='#0F172A', lw=8.0, zorder=3)
    ax1.text(0.65, 3.90, "검은색 주행선", fontsize=9.8, fontweight='bold', color='#0F172A', zorder=5)
    
    # Wheeled robot positioned on line edge
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
                 fontsize=9.2, fontweight='bold', color='#DC2626', zorder=9)
    
    # Visual Badges for binary steering logic
    ax1.text(3.70, 3.85, "검은 선 감지 시 → 우회전", ha='center', va='center',
             fontsize=9.0, fontweight='bold', color='#1E40AF',
             bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#93C5FD', lw=0.9), zorder=10)
    ax1.text(3.70, 2.45, "흰 바탕 감지 시 → 좌회전", ha='center', va='center',
             fontsize=9.0, fontweight='bold', color='#B45309',
             bbox=dict(boxstyle='round,pad=0.2', fc='#FEF3C7', ec='#FCD34D', lw=0.9), zorder=10)
    
    ax1.text(2.51, 1.72, "1. 센서가 검은 선을 감지하면 오른쪽으로 조향합니다.", ha='center', va='center', fontsize=10.2, color='#0F172A')
    ax1.text(2.51, 1.38, "2. 센서가 흰 바탕을 감지하면 왼쪽으로 조향합니다.", ha='center', va='center', fontsize=10.2, color='#0F172A')
    ax1.text(2.51, 0.76, "단 1줄의 단순한 반사 규칙만으로도\n복잡한 곡선 경로를 벗어나지 않고 추종합니다.", 
             ha='center', va='center', fontsize=10.8, fontweight='bold', color=COLOR_SUCCESS,
             bbox=dict(boxstyle='round,pad=0.25', fc=COLOR_SUCCESS_BG, ec=COLOR_SUCCESS_BORDER, lw=1.1))
    
    # Right Column: Boat Adaptation (라인트레이싱 알고리즘)
    ax1.add_patch(FancyBboxPatch((5.20, 0.25), 4.58, 4.60, boxstyle="round,pad=0.06", 
                                 fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax1.text(7.49, 4.58, "보트 적용: 라인트레이싱 알고리즘", ha='center', va='top', 
             fontsize=12.5, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    ax1.add_patch(Rectangle((5.42, 2.05), 4.14, 2.15, fc='#F0F9FF', ec='#BAE6FD', lw=1.0, zorder=2))
    for wy in [2.35, 2.85, 3.35, 3.85]:
        ax1.plot([5.50, 9.45], [wy, wy], color='#E0F2FE', lw=1.2, ls=':', zorder=3)
        
    draw_boat(ax1, 6.15, 2.85, 0.0, length=1.05, width=0.52, color=COLOR_PRIMARY)
    
    # Goal flag at right edge (x=9.25)
    goal_x, goal_y = 9.25, 2.85
    ax1.plot([goal_x, goal_x], [goal_y - 0.25, goal_y + 0.65], color='#15803D', lw=2.2, zorder=8)
    ax1.add_patch(Polygon([(goal_x, goal_y + 0.65), (goal_x + 0.35, goal_y + 0.45), (goal_x, goal_y + 0.25)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax1.text(goal_x, goal_y + 0.78, "목적지(Goal)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#15803D')
    
    # Priority 1: Destination tracking line
    ax1.plot([6.70, goal_x], [2.85, 2.85], color='#22C55E', lw=1.5, ls='--', zorder=4)
    ax1.text(7.05, 2.62, "1순위: 목적지 추종", fontsize=9.2, fontweight='bold', color='#15803D')
    
    # Priority 2: Buoy detection and reactive steering
    buoy_x, buoy_y = 7.75, 3.55
    ax1.add_patch(Circle((buoy_x, buoy_y), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=8))
    ax1.text(buoy_x, buoy_y + 0.32, "장애물 부표", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#9A3412')
    
    ax1.plot([6.70, buoy_x - 0.15], [2.85, buoy_y - 0.12], color='#DC2626', lw=1.8, ls='--', zorder=5)
    ax1.text(6.75, 3.40, "2순위: 센서 감지", fontsize=9.2, fontweight='bold', color='#DC2626', zorder=6)
    
    ax1.annotate('우현 회피 조타!', xy=(6.15, 2.58), xytext=(6.15, 2.15),
                 arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.5),
                 fontsize=9.8, fontweight='bold', color='#0284C7', zorder=10)
    
    ax1.text(7.49, 1.72, "1. 평상시에는 1순위로 최종 목적지를 향해 직진합니다.", ha='center', va='center', fontsize=10.2, color='#0F172A')
    ax1.text(7.49, 1.38, "2. 장애물이 감지되면 단순한 반사 규칙으로 회피 조타합니다.", ha='center', va='center', fontsize=10.2, color='#0F172A')
    ax1.text(7.49, 0.76, "단순하지만 강력한 제어:\n몇 줄의 반사 코드만으로도 목적지까지 안정적으로 도달합니다.", 
             ha='center', va='center', fontsize=10.8, fontweight='bold', color=COLOR_PRIMARY,
             bbox=dict(boxstyle='round,pad=0.25', fc='#F0F9FF', ec='#BAE6FD', lw=1.1))
    
    # --------------------------------------------------------------------------
    # SUBPLOT 2: 초기 개방 수역 회피 성공 궤적 (right: 0.505 ~ 0.970, bottom: 0.540 ~ 0.895, height: 0.355)
    # (Visually beautiful wide detour restored, safe distance labeled as 0.5m)
    # --------------------------------------------------------------------------
    ax2 = fig.add_axes([0.505, 0.540, 0.465, 0.355])
    ax2.set_facecolor(COLOR_PANEL)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5.5)
    
    ax2.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax2.set_title("초기 성공 경험: 개방 수역에서 단순 반사 제어로 0.5m 안전거리를 유지하며 목표에 도달했습니다.", 
                  fontsize=12.2, fontweight='bold', pad=8, color=COLOR_TEXT_MAIN, loc='left')
    ax2.set_xlabel("전진 방향 X 좌표 (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    ax2.set_ylabel("횡방향 Y 좌표 (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    
    # Destination Goal at (9.3, 2.7)
    gx, gy = 9.3, 2.7
    ax2.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.5, zorder=8)
    ax2.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.45, gy + 0.55), (gx, gy + 0.3)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax2.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#15803D')
    
    # Direct nominal path to goal
    ax2.plot([0.8, gx], [2.7, gy], color='#94A3B8', lw=1.5, ls=':', zorder=1, label='원래 목표 직진 경로')
    
    # Single Buoy at (5.2, 3.4), radius 0.25m
    buoy_x, buoy_y = 5.2, 3.4
    safe_circle = Circle((buoy_x, buoy_y), 0.90, fc='#FEF3C7', ec=COLOR_WARN, lw=1.4, ls='--', alpha=0.7, zorder=2)
    ax2.add_patch(safe_circle)
    ax2.add_patch(Circle((buoy_x, buoy_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.8, zorder=8))
    ax2.text(buoy_x, buoy_y + 0.35, "단일 부표", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#9A3412')
    ax2.text(buoy_x, buoy_y + 1.05, "안전 거리 (0.5m 설정)", ha='center', va='bottom', fontsize=10.2, color='#B45309')
    
    # Beautiful wide curved avoidance trajectory (generous smooth detour)
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
        "1. 목적지를 향해 직진 주행합니다.",
        "2. 부표를 감지하고 우현으로 조타합니다.",
        "3. 0.5m 안전 거리를 유지하며 통과합니다.",
        "4. 원래 목표 경로로 복귀하여 완주합니다."
    ]
    # Clean non-overlapping positions
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
                 fontsize=9.8, fontweight='bold', color=COLOR_TEXT_MAIN,
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_BORDER, lw=0.8, alpha=0.95), zorder=15)
        
    ax2.text(5.0, 0.40, "단일 장애물 환경에서는 단순 반사 규칙만으로도 목표 지점까지 이탈 없이 안정적으로 주행을 완주했습니다.",
             ha='center', va='center', fontsize=11, fontweight='bold', color=COLOR_SUCCESS,
             bbox=dict(boxstyle='round,pad=0.3', fc=COLOR_SUCCESS_BG, ec=COLOR_SUCCESS_BORDER, lw=1.2), zorder=20)
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # --------------------------------------------------------------------------
    # MIDDLE SECTION HEADER (y: 0.462 ~ 0.495, perfectly separated!)
    # --------------------------------------------------------------------------
    fig.text(0.030, 0.485, "실제 경기 수조에서 직면한 라인트레이싱 알고리즘의 3대 구조적 한계", 
             fontsize=19.5, fontweight='bold', color=COLOR_DANGER, va='top')
    fig.text(0.030, 0.458, "복잡한 경기 환경과 선박 유체역학적 특성으로 인해 단순 반사 제어가 직면한 본질적 결함입니다.", 
             fontsize=12.0, color=COLOR_TEXT_SUB, va='top')

    # --------------------------------------------------------------------------
    # SUBPLOT 3: 한계 1 - 게이트 폐쇄 (left: 0.030 ~ 0.330, bottom: 0.025 ~ 0.395, height: 0.370)
    # --------------------------------------------------------------------------
    ax3 = fig.add_axes([0.030, 0.025, 0.300, 0.370])
    ax3.set_facecolor(COLOR_PANEL)
    ax3.set_xlim(0, 7.0)
    ax3.set_ylim(0, 7.5)
    
    ax3.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax3.set_title("한계 1: 안전마진 중첩으로 게이트 통로가 폐쇄됩니다.", 
                  fontsize=12.2, fontweight='bold', pad=8, color=COLOR_DANGER, loc='left')
    ax3.set_xlabel("X 좌표 (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    ax3.set_ylabel("Y 좌표 (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    
    # Outer Concrete Wall
    ax3.axhline(7.0, color='#64748B', lw=4.0, zorder=5)
    ax3.text(0.4, 7.15, "수조 외곽 콘크리트 벽", fontsize=10.5, fontweight='bold', color='#475569')
    
    b1_x, b1_y = 4.0, 4.4
    b2_x, b2_y = 4.0, 2.8
    margin_r = 1.0
    
    c1 = Circle((b1_x, b1_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    c2 = Circle((b2_x, b2_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    ax3.add_patch(c1)
    ax3.add_patch(c2)
    
    # Overlap Hatch
    ax3.fill_between([3.4, 4.0, 4.6], [3.6, 3.8, 3.6], [3.6, 3.4, 3.6], 
                     color='#DC2626', alpha=0.35, hatch='///', zorder=3)
    
    ax3.add_patch(Circle((b1_x, b1_y), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax3.add_patch(Circle((b2_x, b2_y), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax3.text(b1_x, b1_y + 0.35, "부표 A", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#9A3412')
    ax3.text(b2_x, b2_y - 0.35, "부표 B", ha='center', va='top', fontsize=11, fontweight='bold', color='#9A3412')
    
    # Callout for overlap
    ax3.annotate('안전마진 중첩 구간\n(0.4m 겹침 발생)', xy=(3.8, 3.6), xytext=(2.2, 2.7),
                 arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.4),
                 ha='center', va='center', fontsize=9.8, fontweight='bold', color='#991B1B',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=15)
    
    ax3.annotate('', xy=(4.0, 4.18), xytext=(4.0, 3.02),
                 arrowprops=dict(arrowstyle='<->', color='#0F172A', lw=1.5), zorder=10)
    ax3.text(4.25, 3.6, "실제 통로\n폭 1.6m", ha='left', va='center', fontsize=10, fontweight='bold', color='#0F172A')
    
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
    ax3.text(4.45, 6.65, "외곽벽 충돌!", fontsize=11, fontweight='bold', color='#DC2626')
    
    draw_boat(ax3, 1.1, 3.6, 0.0, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax3, 2.7, 3.9, np.pi/5, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax3, 3.7, 5.7, np.pi/2.7, length=0.8, width=0.4, color=COLOR_DANGER)
    
    ax3.annotate('', xy=(6.5, 3.6), xytext=(4.8, 3.6),
                 arrowprops=dict(arrowstyle='->', color='#16A34A', lw=2.2, ls='--'), zorder=4)
    ax3.text(5.6, 3.9, "목표 게이트 출구", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#16A34A')
    
    # Complete Declarative Sentences
    ax3.text(3.5, 1.15, "1. 게이트 폭(1.6m)보다 안전마진 합(2.0m)이 더 큽니다.\n2. 양쪽 마진이 겹치면서 열린 통로를 가상 벽으로 오판합니다.\n3. 진입을 포기하고 급선회하여 외곽 콘크리트 벽에 충돌합니다.",
             ha='center', va='center', fontsize=10, color='#991B1B', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2), zorder=25)

    # --------------------------------------------------------------------------
    # SUBPLOT 4: 한계 2 - 너무 큰 진동과 선체 불안정성 (middle: 0.350 ~ 0.650, bottom: 0.025 ~ 0.395, height: 0.370)
    # (Visualizing multiple semi-transparent boats oscillating left/right along path)
    # --------------------------------------------------------------------------
    ax4 = fig.add_axes([0.350, 0.025, 0.300, 0.370])
    ax4.set_facecolor(COLOR_PANEL)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(-48, 52)
    
    ax4.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax4.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax4.set_title("한계 2: 원거리 궤적과 달리 실제로는 극심한 진동으로 불안정합니다.", 
                  fontsize=12.2, fontweight='bold', pad=8, color=COLOR_DANGER, loc='left')
    ax4.set_xlabel("주행 시간 (s)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    ax4.set_ylabel("조타각 (deg)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    
    # Top Card: Multiple semi-transparent boats showing violent left-right chattering along nominal path
    ax4.add_patch(FancyBboxPatch((0.4, 16.5), 9.2, 33.5, boxstyle="round,pad=0.04",
                                fc='#FFFFFF', ec=COLOR_BORDER_STRONG, lw=1.2, zorder=10))
    ax4.text(5.0, 47.8, "실제 선체 거동: 겉보기 일직선 주행 속 극심한 좌우 사행 진동", 
             ha='center', va='top', fontsize=10.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=11)
    
    # Nominal straight center line
    ax4.plot([0.8, 9.2], [31.5, 31.5], color='#94A3B8', lw=1.6, ls=':', zorder=12)
    
    # Actual high frequency zigzag trajectory
    zig_t = np.linspace(0.8, 9.2, 250)
    zig_y = 31.5 + 5.5 * np.sin((zig_t - 0.8) * 4.2)
    ax4.plot(zig_t, zig_y, color='#DC2626', lw=2.0, zorder=13)
    
    # Draw 8 clearly visible transparent boats vibrating left and right along the path!
    boat_x_samples = [1.2, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.6]
    boat_y_samples = [31.5 + 5.5 * np.sin((bx - 0.8) * 4.2) for bx in boat_x_samples]
    boat_headings = [np.deg2rad(30 * np.cos((bx - 0.8) * 4.2)) for bx in boat_x_samples]
    
    for bx, by, bhd in zip(boat_x_samples, boat_y_samples, boat_headings):
        draw_boat(ax4, bx, by, bhd, length=0.95, width=0.48, color='#38BDF8', alpha=0.60, zorder=15)
        
    ax4.text(2.2, 39.5, "좌현 30° 조타", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#DC2626', zorder=18)
    ax4.text(3.3, 23.5, "우현 30° 조타", ha='center', va='top', fontsize=8.5, fontweight='bold', color='#0284C7', zorder=18)
    ax4.text(5.0, 18.0, "겉보기에는 일직선 추종  <--->  실제로는 짧은 시간 내 좌우 30° 요동 반복", 
             ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#DC2626', zorder=18)
    
    # Bottom: Quantitative Rudder Angle Time-Series Chattering (y: -26 ~ 8)
    t = np.linspace(0, 10, 500)
    chatter = 15.0 * np.sin(2 * np.pi * 4.5 * t) + 3.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.6, len(t))
    chatter = np.clip(chatter, -22, 22) - 8.0 # centered around -8 deg, max +14, min -30
    
    ax4.plot(t, chatter, color='#DC2626', lw=1.3, zorder=4)
    ax4.axhline(-8, color='#64748B', lw=1.0, ls='--', zorder=2)
    
    ax4.annotate('조타각 극단적 진동 (4.5Hz, ±28°)', xy=(2.0, 7), xytext=(0.5, 9),
                 fontsize=9.0, fontweight='bold', color='#B91C1C',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=20)
    
    # Complete Declarative Sentences
    ax4.text(5.0, -36, "1. 멀리서 보면 경로를 아주 잘 따라가는 것처럼 보입니다.\n2. 그러나 실제로는 짧은 시간에 큰 각도로 좌우 진동합니다.\n3. 과도한 조타 진동으로 기어가 마모되고 선속이 40% 저하됩니다.",
             ha='center', va='center', fontsize=9.8, fontweight='bold', color='#991B1B',
             bbox=dict(boxstyle='round,pad=0.35', fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2), zorder=25)

    # --------------------------------------------------------------------------
    # SUBPLOT 5: 한계 3 - 확장성의 한계 & 다중 장애물 딜레마 (right: 0.670 ~ 0.970, bottom: 0.025 ~ 0.395, height: 0.370)
    # --------------------------------------------------------------------------
    ax5 = fig.add_axes([0.670, 0.025, 0.300, 0.370])
    ax5.set_facecolor(COLOR_PANEL)
    ax5.set_xlim(0, 7.0)
    ax5.set_ylim(0, 7.5)
    
    ax5.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax5.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax5.set_title("한계 3: 제어 파라미터 부족으로 확장성이 결여됩니다.", 
                  fontsize=12.2, fontweight='bold', pad=8, color=COLOR_DANGER, loc='left')
    ax5.set_xlabel("X 좌표 (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    ax5.set_ylabel("Y 좌표 (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
    
    # Top Card: Parameter & Environmental Blindspot Dilemma (Deep Engineering Insight)
    ax5.add_patch(FancyBboxPatch((0.25, 5.80), 6.5, 1.55, boxstyle="round,pad=0.04",
                                 fc='#FFFFFF', ec=COLOR_BORDER_STRONG, lw=1.2, zorder=20))
    ax5.text(3.5, 7.20, "파라미터 조절과 환경 인식의 구조적 딜레마", 
             ha='center', va='top', fontsize=10.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=21)
    ax5.text(3.5, 6.72, "1. 파라미터 모순: 감지거리 줄이면 최단경로이나 돌발상황 충돌",
             ha='center', va='center', fontsize=8.6, color='#334155', zorder=21)
    ax5.text(3.5, 6.32, "2. 단일 각도 반사: 특정 각도 진입 시 고정 각도만 선택 가능",
             ha='center', va='center', fontsize=8.6, color='#334155', zorder=21)
    ax5.text(3.5, 5.95, "3. 환경 종속성: 동일 각도라도 주변 배치 바뀌면 예측 불가",
             ha='center', va='center', fontsize=8.6, color='#334155', zorder=21)
    
    # Visualizing Multi-obstacle crash trap
    bA_x, bA_y = 2.6, 4.0
    bB_x, bB_y = 5.2, 4.6
    
    ax5.add_patch(Circle((bA_x, bA_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax5.add_patch(Circle((bB_x, bB_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax5.add_patch(Circle((bA_x, bA_y), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=5))
    ax5.add_patch(Circle((bB_x, bB_y), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=5))
    ax5.text(bA_x, bA_y - 0.38, "장애물 A", ha='center', va='top', fontsize=10, fontweight='bold', color='#9A3412')
    ax5.text(bB_x + 0.35, bB_y, "장애물 B", ha='left', va='center', fontsize=10, fontweight='bold', color='#9A3412')
    
    # Boat initial pose
    draw_boat(ax5, 1.3, 2.5, np.pi/4.5, length=0.85, width=0.42, color='#0284C7', zorder=8)
    
    # Avoidance path turning away from A into B
    pts = np.array([
        [1.3, 2.5],
        [2.3, 3.2],
        [3.6, 3.9],
        [5.0, 4.5]
    ])
    tck, u = splprep([pts[:,0], pts[:,1]], s=0, k=2)
    u_new = np.linspace(0, 1, 100)
    px, py = splev(u_new, tck)
    ax5.plot(px, py, color=COLOR_DANGER, lw=2.8, ls='--', zorder=6)
    
    # Boat crashed at B
    draw_boat(ax5, 5.0, 4.5, np.pi/4.5, length=0.85, width=0.42, color='#DC2626', zorder=10)
    ax5.plot(5.1, 4.6, marker='X', markersize=18, color='#DC2626', markeredgecolor='#7F1D1D', zorder=20)
    
    # Callout placed cleanly with no top-card overlap!
    ax5.annotate('A 회피하려다 B에 직접 충돌!\n(주변 장애물 배치 고려 불가)', xy=(4.9, 4.4), xytext=(2.1, 5.0),
                 arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5),
                 fontsize=9.5, fontweight='bold', color='#DC2626',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#F87171', lw=1.0), zorder=25)
    
    # Complete Declarative Sentences Box at bottom
    ax5.text(3.5, 1.15, "1. 회피 시점을 늦추면 최단 경로가 되나 돌발 상황에 충돌합니다.\n2. 특정 각도 반사는 주변의 다른 장애물 배치를 고려하지 못합니다.\n3. 동일한 감지 조건이라도 환경이 달라지면 성공을 보장하지 못합니다.",
             ha='center', va='center', fontsize=9.8, color='#991B1B', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2), zorder=25)

    # Save
    plt.savefig(OUTPUT_FILE, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print(f"Master Sheet 1 fully updated: {OUTPUT_FILE}")

if __name__ == '__main__':
    generate_sheet_1()
