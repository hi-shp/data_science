import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_FILE = '/home/soonhong/kaboat/report4/sheet1_subfigures/subfig4_violent_chattering_overlapping_boats.png'

COLOR_BG = '#FFFFFF'
COLOR_PANEL = '#F8FAFC'
COLOR_BORDER_STRONG = '#94A3B8'
COLOR_TEXT_MAIN = '#0F172A'
COLOR_TEXT_SUB = '#334155'
COLOR_PRIMARY = '#0284C7'
COLOR_DANGER = '#DC2626'
COLOR_DANGER_BG = '#FEF2F2'
COLOR_DANGER_BORDER = '#FCA5A5'

def draw_boat(ax, x, y, heading_rad, length=1.4, width=0.75, color='#38BDF8', ec='#0284C7', alpha=0.55, zorder=10):
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
    ax.annotate('', xy=(x + length*0.70*c, y + length*0.70*s), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=2.0, alpha=min(1.0, alpha+0.35)), zorder=zorder+2)

fig = plt.figure(figsize=(7.2, 7.5), dpi=300)
fig.patch.set_facecolor(COLOR_BG)

# Title at the very top
fig.text(0.08, 0.965, "한계 2: 원거리 궤적과 달리 실제로는 극심한 진동으로 불안정합니다.", 
         fontsize=12.2, fontweight='bold', color=COLOR_DANGER, va='top')

# -------------------------------------------------------------------------
# Top Subplot: Overlapping Boats along nominal straight line
# -------------------------------------------------------------------------
ax_top = fig.add_axes([0.10, 0.52, 0.84, 0.40])
ax_top.set_facecolor('#FFFFFF')
ax_top.set_xlim(0.5, 9.5)
ax_top.set_ylim(-2.8, 2.8)

for spine in ax_top.spines.values():
    spine.set_color(COLOR_BORDER_STRONG)
    spine.set_linewidth(1.3)
ax_top.set_xticks([])
ax_top.set_yticks([])

ax_top.text(5.0, 2.50, "실제 선체 거동: 겉보기 일직선 주행 속 극심한 좌우 사행 진동", 
            ha='center', va='top', fontsize=10.5, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=30)

# Nominal straight path line down center
ax_top.plot([0.6, 9.4], [0.0, 0.0], color='#94A3B8', lw=2.0, ls=':', zorder=5)
ax_top.text(0.8, 0.25, "원래 목표 직진 경로", fontsize=9.0, color='#64748B', fontweight='bold', zorder=25)

# Place 11 transparent boats overlapping each other by ~50% sequentially in time!
boat_x_samples = np.linspace(1.2, 8.8, 11)
boat_headings = [np.deg2rad(28 if i % 2 == 0 else -28) for i in range(len(boat_x_samples))]
boat_y_samples = [0.42 if i % 2 == 0 else -0.42 for i in range(len(boat_x_samples))]

# Trajectory connecting centers
ax_top.plot(boat_x_samples, boat_y_samples, color='#DC2626', lw=1.6, ls='--', alpha=0.65, zorder=6)

for i, (bx, by, bhd) in enumerate(zip(boat_x_samples, boat_y_samples, boat_headings)):
    draw_boat(ax_top, bx, by, bhd, length=1.45, width=0.72, color='#38BDF8', ec='#0284C7', alpha=0.52, zorder=10 + i)

ax_top.text(2.0, 1.75, "좌현 28° 조타", ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#DC2626', 
            bbox=dict(boxstyle='round,pad=0.15', fc='#FFFFFF', ec='#FCA5A5', lw=0.8), zorder=35)
ax_top.text(2.7, -1.75, "우현 28° 조타", ha='center', va='top', fontsize=9.2, fontweight='bold', color='#0284C7', 
            bbox=dict(boxstyle='round,pad=0.15', fc='#FFFFFF', ec='#93C5FD', lw=0.8), zorder=35)
ax_top.text(5.0, -2.50, "시간순 50% 중첩: 일직선 경로 추종 시에도 선체는 좌우로 56° 진폭의 격렬한 진동을 반복함", 
            ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#DC2626', zorder=35)

# -------------------------------------------------------------------------
# Bottom Subplot: Quantitative Rudder Angle Time-Series Chattering
# -------------------------------------------------------------------------
ax_bot = fig.add_axes([0.10, 0.08, 0.84, 0.38])
ax_bot.set_facecolor(COLOR_PANEL)
ax_bot.set_xlim(0, 10)
ax_bot.set_ylim(-38, 22)

ax_bot.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
for spine in ax_bot.spines.values():
    spine.set_color(COLOR_BORDER_STRONG)
    spine.set_linewidth(1.3)

ax_bot.set_xlabel("주행 시간 (s)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)
ax_bot.set_ylabel("조타각 (deg)", fontsize=11, labelpad=4, color=COLOR_TEXT_SUB)

t = np.linspace(0, 10, 500)
chatter = 12.0 * np.sin(2 * np.pi * 4.5 * t) + 2.5 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.4, len(t))
chatter = np.clip(chatter, -16, 16) - 5.0 # centered around -5 deg, bounds [-21, +11]

ax_bot.plot(t, chatter, color='#DC2626', lw=1.2, zorder=4)
ax_bot.axhline(-5, color='#64748B', lw=1.0, ls='--', zorder=2)

ax_bot.annotate('조타각 극단적 고주파 진동 (4.5Hz, ±28°)', xy=(2.0, 11), xytext=(2.0, 16),
                ha='center', va='bottom',
                fontsize=9.2, fontweight='bold', color='#B91C1C',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=20)

# Complete Declarative Sentences Box at bottom
ax_bot.text(5.0, -28.0, 
            "1. 멀리서 보면 경로를 아주 잘 따라가는 것처럼 보입니다.\n"
            "2. 그러나 실제로는 짧은 시간에 큰 각도로 좌우 진동합니다.\n"
            "3. 과도한 조타 진동으로 기어가 마모되고 선속이 40% 저하됩니다.",
            ha='center', va='center', fontsize=9.6, fontweight='bold', color='#991B1B',
            bbox=dict(boxstyle='round,pad=0.35', fc=COLOR_DANGER_BG, ec=COLOR_DANGER_BORDER, lw=1.2), zorder=25)

plt.savefig(OUTPUT_FILE, dpi=300, facecolor=COLOR_BG)
plt.close()
print("Subfig 4 updated successfully:", OUTPUT_FILE)


