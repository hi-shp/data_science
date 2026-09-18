import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from scipy.interpolate import splprep, splev

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_FILE = '/home/soonhong/kaboat/report4/sheet1_subfigures/subfig5_limitation3_scalability_collision.png'

COLOR_BG = '#FFFFFF'
COLOR_PANEL = '#F8FAFC'
COLOR_BORDER_STRONG = '#94A3B8'
COLOR_TEXT_MAIN = '#0F172A'
COLOR_TEXT_SUB = '#334155'
COLOR_PRIMARY = '#0284C7'
COLOR_WARN = '#EA580C'
COLOR_DANGER = '#DC2626'
COLOR_DANGER_BG = '#FEF2F2'
COLOR_DANGER_BORDER = '#FCA5A5'

def draw_boat(ax, x, y, heading_rad, length=1.05, width=0.52, color='#0284C7', ec='#0F172A', alpha=0.95, zorder=10):
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

fig = plt.figure(figsize=(7.2, 7.5), dpi=300)
ax = fig.add_axes([0.11, 0.08, 0.84, 0.86])
ax.set_facecolor(COLOR_PANEL)
ax.set_xlim(0, 8.0)
ax.set_ylim(0, 8.0)

ax.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
for spine in ax.spines.values():
    spine.set_color(COLOR_BORDER_STRONG)
    spine.set_linewidth(1.3)

ax.set_title("한계 3: 단일 반사 규칙은 다중 장애물 연쇄 충돌을 유발합니다.", 
             fontsize=12.2, fontweight='bold', pad=10, color=COLOR_DANGER, loc='left')
ax.set_xlabel("전진 거리 X (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
ax.set_ylabel("횡방향 위치 Y (m)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)

# Top Engineering Context Banner (Clean & compact, no overlap)
ax.add_patch(FancyBboxPatch((0.25, 6.70), 7.50, 1.05, boxstyle="round,pad=0.04",
                             fc='#FFFFFF', ec=COLOR_BORDER_STRONG, lw=1.1, zorder=20))
ax.text(4.0, 7.52, "구조적 원인: 주변 장애물 배치와 환경 지도를 고려하지 못하는 단일 센서 반사", 
        ha='center', va='top', fontsize=9.8, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=21)
ax.text(4.0, 7.02, "전방 장애물(A)만 감지하고 즉시 좌현 조타하여, 인접 장애물(B)의 위치를 보지 못하고 직격 충돌함",
        ha='center', va='center', fontsize=9.0, color='#475569', zorder=21)

# Destination Goal at (7.3, 2.5)
gx, gy = 7.3, 2.5
ax.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.2, zorder=8)
ax.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.40, gy + 0.55), (gx, gy + 0.3)],
                     closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
ax.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=10.2, fontweight='bold', color='#15803D')

# Nominal Target Path (Straight dashed line)
ax.plot([0.5, gx], [2.5, gy], color='#94A3B8', lw=1.8, ls=':', zorder=1)
ax.text(6.1, 2.15, "원래 목표 직진 경로", fontsize=9.2, color='#64748B', fontweight='bold', zorder=2)

# Obstacle A directly on the path at (3.4, 2.5)
bA_x, bA_y = 3.4, 2.5
ax.add_patch(Circle((bA_x, bA_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
ax.add_patch(Circle((bA_x, bA_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
ax.text(bA_x, bA_y - 0.40, "장애물 A\n(직진 경로 차단)", ha='center', va='top', fontsize=9.8, fontweight='bold', color='#9A3412')

# Obstacle B positioned offset to the left/upward at (5.2, 4.4)
bB_x, bB_y = 5.2, 4.4
ax.add_patch(Circle((bB_x, bB_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
ax.add_patch(Circle((bB_x, bB_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
ax.text(bB_x + 0.40, bB_y + 0.15, "장애물 B\n(사각지대 위치)", ha='left', va='center', fontsize=9.8, fontweight='bold', color='#9A3412')

# Boat 1: Initial starting approach (x=0.9, y=2.5, heading=0 deg)
draw_boat(ax, 0.9, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

# Boat 2: At detection point (x=2.1, y=2.5)
draw_boat(ax, 2.1, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

# Sensor Beam detecting Obstacle A
ax.plot([2.55, bA_x - 0.25], [2.5, 2.5], color='#DC2626', lw=2.2, ls='-', zorder=15)
ax.annotate('센서 감지 거리 (1.2m)\n즉시 좌현 40° 회피 조타!', xy=(2.7, 2.5), xytext=(1.8, 3.7),
            arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.4),
            fontsize=9.2, fontweight='bold', color='#DC2626',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.9), zorder=25)

# Reactive avoidance trajectory swinging left (upward) directly aiming into Obstacle B!
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
ax.plot(px, py, color=COLOR_DANGER, lw=3.0, ls='-', zorder=5)

# Boat 3: Mid-turn swinging toward B (x=3.6, y=3.4, heading=40 deg)
hd3 = np.arctan2(py[55] - py[50], px[55] - px[50])
draw_boat(ax, 3.6, 3.42, hd3, length=0.95, width=0.48, color='#38BDF8', alpha=0.75, zorder=10)

# Boat 4: Crashing directly into Obstacle B (x=5.0, y=4.35)
hd4 = np.arctan2(py[-1] - py[-5], px[-1] - px[-5])
draw_boat(ax, 5.0, 4.35, hd4, length=0.95, width=0.48, color='#DC2626', zorder=12)

# Crash Flash & Big Red X at Obstacle B
ax.plot(5.15, 4.4, marker='X', markersize=20, color='#DC2626', markeredgecolor='#7F1D1D', markeredgewidth=2.0, zorder=30)
ax.annotate('A를 회피하려다 B에 정면 충돌!\n(다중 장애물 환경에서 반사 제어 완전 실패)', 
            xy=(5.15, 4.4), xytext=(3.4, 5.7),
            arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=1.8),
            fontsize=9.8, fontweight='bold', color='#991B1B',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#DC2626', lw=1.4), zorder=35)

# Declarative 3-Sentence Summary at Bottom
ax.text(4.0, 0.95, 
        "1. 전방 장애물(A)을 감지하고 고정된 규칙에 따라 좌현으로 급회피를 시작합니다.\n"
        "2. 그러나 센서 시야 밖의 인접 장애물(B)을 전혀 인식하지 못해 회피 경로가 충돌 경로가 됩니다.\n"
        "3. 단일 센서 반사 제어는 다중 장애물이 배치된 복잡 수역에서 근본적인 한계를 드러냅니다.",
        ha='center', va='center', fontsize=9.6, fontweight='bold', color='#991B1B',
        bbox=dict(boxstyle='round,pad=0.35', fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2), zorder=25)

plt.savefig(OUTPUT_FILE, dpi=300, facecolor=COLOR_BG)
plt.close()
print("Subfig 5 test generated successfully:", OUTPUT_FILE)
