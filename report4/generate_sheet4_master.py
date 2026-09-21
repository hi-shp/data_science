#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Exhibition Sheet 4 Master Generator
Proposed Algorithm Solution Technologies: DBSCAN, Gap Extraction, Cubic Bezier, and Pure Pursuit
Generates 5 individual high-resolution subfigures and compiles them into a Master Sheet.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, Arc, Wedge
from scipy.interpolate import splprep, splev

# Matplotlib Korean font configuration
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/soonhong/kaboat/report4'
SUBFIG_DIR = os.path.join(OUTPUT_DIR, 'sheet4_subfigures')
os.makedirs(SUBFIG_DIR, exist_ok=True)

MASTER_PNG = os.path.join(OUTPUT_DIR, 'sheet4_proposed_algorithm_pipeline.png')

# Color Palette Constants
COLOR_BG = '#F8FAFC'
COLOR_PANEL = '#FFFFFF'
COLOR_BORDER = '#CBD5E1'
COLOR_BORDER_STRONG = '#94A3B8'
COLOR_TEXT_TITLE = '#0F172A'
COLOR_TEXT_MAIN = '#1E293B'
COLOR_TEXT_MUTED = '#475569'

COLOR_PRIMARY = '#0284C7'
COLOR_TEAL = '#0D9488'
COLOR_AMBER = '#D97706'
COLOR_INDIGO = '#4F46E5'
COLOR_GREEN = '#16A34A'
COLOR_DANGER = '#DC2626'
COLOR_DANGER_BG = '#FEF2F2'
COLOR_DANGER_BORDER = '#FCA5A5'
COLOR_INFO_BG = '#F0F9FF'
COLOR_INFO_BORDER = '#BAE6FD'
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
# SUBFIGURE 1: DBSCAN 밀도 기반 군집화 및 잡음 제거 원리
# ==============================================================================
def generate_subfig1():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax1 = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax1.set_facecolor(COLOR_PANEL)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 7.0)
    ax1.axis('off')
    
    # Outer frame
    frame = FancyBboxPatch((0.05, 0.05), 9.9, 6.9, boxstyle="round,pad=0.08",
                           ec=COLOR_BORDER, fc=COLOR_PANEL, lw=1.5, zorder=1)
    ax1.add_patch(frame)
    
    # Header tag
    ax1.add_patch(FancyBboxPatch((0.4, 6.20), 9.2, 0.60, boxstyle="round,pad=0.04",
                                 fc='#E0F2FE', ec='#0284C7', lw=1.4, zorder=2))
    ax1.text(0.65, 6.50, "기술 1: DBSCAN 공간 밀도 기반 군집화 및 수면 난반사 잡음 제거",
             fontsize=13.0, fontweight='bold', color='#0369A1', zorder=3)
    
    # USV Boat & LiDAR fan
    draw_boat(ax1, 1.2, 3.8, np.deg2rad(15), length=1.0, width=0.48, color='#0284C7', zorder=15)
    ax1.text(1.2, 2.90, "자율운항 USV\n(2D 라이다 탑재)", ha='center', fontsize=9.8, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=16)
    
    # 120 degree LiDAR FOV beam cone
    wedge = Wedge((1.2, 3.8), 7.2, -45, 75, facecolor='#E0F2FE', edgecolor='#38BDF8',
                  linestyle='--', linewidth=1.2, alpha=0.35, zorder=2)
    ax1.add_patch(wedge)
    ax1.text(3.6, 5.8, "라이다 120° 탐색 영역 (FOV)", fontsize=9.5, color='#0284C7', fontweight='bold', zorder=4)

    # Buoy A (Red Buoy, Obstacle Left/Top)
    cA = (6.4, 5.0)
    circle_buoyA = Circle(cA, 0.40, facecolor='#FCA5A5', edgecolor='#DC2626', lw=2.0, zorder=5)
    ax1.add_patch(circle_buoyA)
    ax1.text(cA[0], cA[1], "부표 A\n(좌현)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#991B1B', zorder=6)
    
    # Points on Buoy A
    np.random.seed(42)
    anglesA = np.linspace(np.pi*0.75, np.pi*1.55, 14)
    ptsA_x = cA[0] + 0.40 * np.cos(anglesA) + np.random.normal(0, 0.03, len(anglesA))
    ptsA_y = cA[1] + 0.40 * np.sin(anglesA) + np.random.normal(0, 0.03, len(anglesA))
    ax1.scatter(ptsA_x, ptsA_y, color='#0284C7', s=50, edgecolors='#0F172A', lw=0.8, zorder=10)

    # Buoy B (Green Buoy, Obstacle Right/Bottom)
    cB = (6.6, 2.6)
    circle_buoyB = Circle(cB, 0.40, facecolor='#86EFAC', edgecolor='#16A34A', lw=2.0, zorder=5)
    ax1.add_patch(circle_buoyB)
    ax1.text(cB[0], cB[1], "부표 B\n(우현)", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#14532D', zorder=6)
    
    anglesB = np.linspace(np.pi*0.65, np.pi*1.45, 12)
    ptsB_x = cB[0] + 0.40 * np.cos(anglesB) + np.random.normal(0, 0.03, len(anglesB))
    ptsB_y = cB[1] + 0.40 * np.sin(anglesB) + np.random.normal(0, 0.03, len(anglesB))
    ax1.scatter(ptsB_x, ptsB_y, color='#0284C7', s=50, edgecolors='#0F172A', lw=0.8, zorder=10)

    # Stray water reflection noise points
    noise_x = [4.2, 5.0, 4.8, 7.5, 7.8, 5.8]
    noise_y = [4.5, 3.8, 2.3, 5.8, 2.2, 3.8]
    ax1.scatter(noise_x, noise_y, marker='x', color='#DC2626', s=60, lw=2.2, zorder=12)
    for nx, ny in zip(noise_x[:3], noise_y[:3]):
        ax1.text(nx+0.12, ny+0.05, "수면 잡음 (제거)", fontsize=8.2, color='#DC2626', fontweight='bold', zorder=12)

    # DBSCAN Epsilon Radius Demo on Buoy A
    demo_pt = (ptsA_x[4], ptsA_y[4])
    eps_circle = Circle(demo_pt, 0.55, facecolor='#38BDF8', edgecolor='#0284C7', ls='--', lw=1.5, alpha=0.35, zorder=7)
    ax1.add_patch(eps_circle)
    ax1.annotate(r"$\epsilon$ 이웃 탐색 반경 ($\epsilon=0.6\mathrm{m}$)" + "\n" + r"이웃 점수 $\geq \mathrm{MinPts}(3)$ 조건 충족",
                 xy=(demo_pt[0]-0.25, demo_pt[1]+0.2), xytext=(2.9, 5.3),
                 arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.6),
                 fontsize=9.2, fontweight='bold', color='#0369A1',
                 bbox=dict(boxstyle='round,pad=0.25', fc='#FFFFFF', ec='#0284C7', lw=1.2), zorder=20)

    # Convex cluster boundary hulls
    hullA = FancyBboxPatch((cA[0]-0.65, cA[1]-0.65), 1.3, 1.3, boxstyle="round,pad=0.08",
                           ec='#0284C7', fc='none', ls='-', lw=2.0, zorder=8)
    ax1.add_patch(hullA)
    ax1.text(cA[0]+0.85, cA[1]+0.3, "군집 1 결속\n(부표 객체 A)", fontsize=9.5, fontweight='bold', color='#0369A1', zorder=12)

    hullB = FancyBboxPatch((cB[0]-0.65, cB[1]-0.65), 1.3, 1.3, boxstyle="round,pad=0.08",
                           ec='#16A34A', fc='none', ls='-', lw=2.0, zorder=8)
    ax1.add_patch(hullB)
    ax1.text(cB[0]+0.85, cB[1]-0.2, "군집 2 결속\n(부표 객체 B)", fontsize=9.5, fontweight='bold', color='#15803D', zorder=12)

    # Explanation banner at bottom
    ax1.add_patch(FancyBboxPatch((0.4, 0.20), 9.2, 1.70, boxstyle="round,pad=0.05",
                                 fc='#F8FAFC', ec='#CBD5E1', lw=1.2, zorder=20))
    ax1.text(0.65, 1.62, "핵심 동작 메커니즘 및 직관적 해설", fontsize=11.2, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax1.text(0.65, 1.25, "1. 120° 라이다가 수집한 수많은 점 중에서 수면 난반사 잡음을 밀도 기준(MinPts)으로 즉각 분리합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax1.text(0.65, 0.85, "2. 부표 표면에 오밀조밀 모인 점들만 자동으로 결속하여 독립된 2개의 부표 객체로 정밀하게 인식합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax1.text(0.65, 0.45, "3. 사전에 부표가 몇 개인지 지정하지 않아도 주변 환경에 맞춰 자율적으로 장애물 경계를 파악합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)

    out_path = os.path.join(SUBFIG_DIR, 'subfig1_dbscan_clustering.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 1:", out_path)

# ==============================================================================
# SUBFIGURE 2: 통과 가능 안전 갭(Gap) 추출 및 다중 가중치 평가
# ==============================================================================
def generate_subfig2():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax2 = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax2.set_facecolor(COLOR_PANEL)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 7.0)
    ax2.axis('off')
    
    frame = FancyBboxPatch((0.05, 0.05), 9.9, 6.9, boxstyle="round,pad=0.08",
                           ec=COLOR_BORDER, fc=COLOR_PANEL, lw=1.5, zorder=1)
    ax2.add_patch(frame)
    
    # Header tag
    ax2.add_patch(FancyBboxPatch((0.4, 6.20), 9.2, 0.60, boxstyle="round,pad=0.04",
                                 fc='#CCFBF1', ec='#0D9488', lw=1.4, zorder=2))
    ax2.text(0.65, 6.50, "기술 2: 안전 갭(Gap) 후보 추출 및 다중 목적 가중치 최적 평가",
             fontsize=13.0, fontweight='bold', color='#0F766E', zorder=3)
    
    # Two Buoys
    cA = (5.5, 5.2)
    cB = (5.5, 2.4)
    r_buoy = 0.45
    margin = 0.50
    boat_hw = 0.21
    
    # Safety Buffers
    ax2.add_patch(Circle(cA, r_buoy + margin + boat_hw, facecolor='#FEF2F2', edgecolor='#F87171', ls='--', lw=1.2, alpha=0.6, zorder=3))
    ax2.add_patch(Circle(cB, r_buoy + margin + boat_hw, facecolor='#FEF2F2', edgecolor='#F87171', ls='--', lw=1.2, alpha=0.6, zorder=3))
    
    # Buoys Solid
    ax2.add_patch(Circle(cA, r_buoy, facecolor='#FCA5A5', edgecolor='#DC2626', lw=2.0, zorder=5))
    ax2.add_patch(Circle(cB, r_buoy, facecolor='#86EFAC', edgecolor='#16A34A', lw=2.0, zorder=5))
    ax2.text(cA[0], cA[1], "부표 A", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#991B1B', zorder=6)
    ax2.text(cB[0], cB[1], "부표 B", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#14532D', zorder=6)
    
    # Dimension line: Total distance vs Effective gap
    ax2.annotate('', xy=(cA[0], cA[1]), xytext=(cB[0], cB[1]),
                arrowprops=dict(arrowstyle='<->', color='#64748B', lw=1.6), zorder=8)
    ax2.text(5.65, 3.8, "부표 중심 간 거리: 2.80m", fontsize=9.2, color='#475569', fontweight='bold', zorder=9)
    
    # Green safe portal bar in between
    gap_y_top = cA[1] - (r_buoy + margin + boat_hw)
    gap_y_bot = cB[1] + (r_buoy + margin + boat_hw)
    ax2.add_patch(FancyBboxPatch((5.3, gap_y_bot), 0.4, gap_y_top - gap_y_bot, boxstyle="round,pad=0.03",
                                 fc='#10B981', ec='#047857', lw=1.8, alpha=0.35, zorder=7))
    
    # Target Gap Center Point
    p_gap = (5.5, 3.8)
    ax2.plot(p_gap[0], p_gap[1], marker='o', markersize=12, color='#0D9488', markeredgecolor='#FFFFFF', markeredgewidth=2.0, zorder=12)
    ax2.text(p_gap[0] + 0.35, p_gap[1] + 0.15, "최적 갭 목표점 (P_gap)\n[선박 통과 중심 좌표]", fontsize=9.8, fontweight='bold', color='#0F766E', zorder=14)

    # Boat approaching
    draw_boat(ax2, 1.6, 3.2, np.deg2rad(12), length=1.0, width=0.48, color='#0284C7', zorder=15)
    ax2.text(1.6, 2.35, "USV 접근", ha='center', fontsize=9.5, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=16)

    # Evaluation Vector Arrows
    ax2.annotate('', xy=(p_gap[0], p_gap[1]), xytext=(1.6, 3.2),
                 arrowprops=dict(arrowstyle='->', color='#0D9488', lw=2.2, ls='-'), zorder=10)
    ax2.text(3.3, 3.7, "목표 갭 벡터 (거리: 3.9m)", fontsize=9.0, fontweight='bold', color='#0D9488', zorder=11)

    # Multi-objective criteria card on right
    ax2.add_patch(FancyBboxPatch((7.3, 2.3), 2.2, 3.6, boxstyle="round,pad=0.05",
                                 fc='#F0FDFA', ec='#0D9488', lw=1.2, zorder=10))
    ax2.text(8.4, 5.5, "6대 가중치 평가", ha='center', fontsize=10.2, fontweight='bold', color='#0F766E', zorder=11)
    
    weights = [
        ("전방 지향성 (Align)", "22%"),
        ("선박 헤딩각 (Heading)", "15%"),
        ("목표 진행성 (Forward)", "23%"),
        ("통로 안전폭 (Width)", "16%"),
        ("장애물 이격 (Clear)", "16%"),
        ("수직 법선도 (Perpend)", "8%")
    ]
    for idx, (wname, wval) in enumerate(weights):
        wy = 5.05 - idx * 0.45
        ax2.text(7.5, wy, wname, fontsize=8.4, color=COLOR_TEXT_MAIN, zorder=12)
        ax2.text(9.3, wy, wval, ha='right', fontsize=8.6, fontweight='bold', color='#0D9488', zorder=12)

    # Explanation banner at bottom
    ax2.add_patch(FancyBboxPatch((0.4, 0.20), 9.2, 1.70, boxstyle="round,pad=0.05",
                                 fc='#F8FAFC', ec='#CBD5E1', lw=1.2, zorder=20))
    ax2.text(0.65, 1.62, "핵심 동작 메커니즘 및 직관적 해설", fontsize=11.2, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax2.text(0.65, 1.25, "1. 인접한 두 부표 사이 거리를 측정하고 선폭과 안전마진(0.5m)을 뺀 실제 통과 공간을 계산합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax2.text(0.65, 0.85, "2. 통로 폭이 충분히 확보된 '안전 갭(Gap)'만을 추려내어 충돌 위험이 있는 좁은 틈을 원천 배제합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax2.text(0.65, 0.45, "3. 통로의 너비와 현재 선박 진행 방향을 종합 평가하여 가장 안전하고 빠른 최적 목표점을 선정합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)

    out_path = os.path.join(SUBFIG_DIR, 'subfig2_gap_extraction_weights.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 2:", out_path)

# ==============================================================================
# SUBFIGURE 3: 3차 베지어 곡선(Cubic Bezier) 기하학 및 곡률 연속성
# ==============================================================================
def generate_subfig3():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax3 = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax3.set_facecolor(COLOR_PANEL)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 7.0)
    ax3.axis('off')
    
    frame = FancyBboxPatch((0.05, 0.05), 9.9, 6.9, boxstyle="round,pad=0.08",
                           ec=COLOR_BORDER, fc=COLOR_PANEL, lw=1.5, zorder=1)
    ax3.add_patch(frame)
    
    # Header tag
    ax3.add_patch(FancyBboxPatch((0.4, 6.20), 9.2, 0.60, boxstyle="round,pad=0.04",
                                 fc='#FEF3C7', ec='#D97706', lw=1.4, zorder=2))
    ax3.text(0.65, 6.50, "기술 3: 3차 베지어(Cubic Bézier) 곡선 및 C2 곡률 연속 궤적 합성",
             fontsize=13.0, fontweight='bold', color='#B45309', zorder=3)
    
    # 4 Control Points
    p0 = np.array([1.2, 2.8])
    p1 = np.array([3.4, 2.8])
    p2 = np.array([5.2, 5.2])
    p3 = np.array([7.8, 5.2])
    
    # Compute Bezier Curve
    t = np.linspace(0, 1, 100)
    bezier_pts = (1-t)[:, None]**3 * p0 + \
                 3*(1-t)[:, None]**2 * t[:, None] * p1 + \
                 3*(1-t)[:, None] * t[:, None]**2 * p2 + \
                 t[:, None]**3 * p3
                 
    # Draw Boat at P0
    draw_boat(ax3, p0[0], p0[1], 0.0, length=1.0, width=0.48, color='#0284C7', zorder=15)
    
    # Draw Control Polygon
    poly_ctrl = np.array([p0, p1, p2, p3])
    ax3.plot(poly_ctrl[:, 0], poly_ctrl[:, 1], color='#94A3B8', ls='--', lw=1.6, zorder=5)
    
    # Draw Control Points
    labels = ["P0: 선박 위치 (출발점)", "P1: 헤딩 방향 연장 (출발 접선)",
              "P2: 갭 진입 법선 (도착 접선)", "P3: 최적 갭 중심점 (도착점)"]
    for idx, (pt, lbl) in enumerate(zip(poly_ctrl, labels)):
        ax3.plot(pt[0], pt[1], marker='o', markersize=11, color='#D97706', markeredgecolor='#0F172A', markeredgewidth=1.5, zorder=12)
        offset_y = 0.35 if idx in [1, 3] else -0.45
        ax3.text(pt[0], pt[1] + offset_y, f"P{idx}", ha='center', fontsize=11.0, fontweight='bold', color='#B45309', zorder=14)

    # Draw Smooth Bezier Curve (Bold Turquoise/Teal)
    ax3.plot(bezier_pts[:, 0], bezier_pts[:, 1], color='#0D9488', lw=4.2, zorder=10)
    
    # Contrast: Sharp Discontinuous Curve (Red Dashed)
    sharp_x = [p0[0], 4.2, p3[0]]
    sharp_y = [p0[1], p0[1], p3[1]]
    ax3.plot(sharp_x, sharp_y, color='#DC2626', ls=':', lw=2.2, alpha=0.75, zorder=6)
    ax3.plot(4.2, p0[1], marker='X', markersize=12, color='#DC2626', zorder=12)
    ax3.text(4.35, p0[1] - 0.35, "급격한 각도 꺾임!\n(조타 충격 및 진동 유발)", fontsize=8.4, fontweight='bold', color='#DC2626', zorder=12)

    # Tangent arrows
    ax3.annotate('', xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                 arrowprops=dict(arrowstyle='->', color='#2563EB', lw=2.0), zorder=8)
    ax3.annotate('', xy=(p3[0], p3[1]), xytext=(p2[0], p2[1]),
                 arrowprops=dict(arrowstyle='->', color='#16A34A', lw=2.0), zorder=8)

    # Inset badge: Curvature Continuity
    ax3.add_patch(FancyBboxPatch((5.8, 2.3), 3.6, 1.6, boxstyle="round,pad=0.04",
                                 fc='#FEF3C7', ec='#D97706', lw=1.2, zorder=10))
    ax3.text(7.6, 3.55, "C2 곡률 연속성의 공학적 이점", ha='center', fontsize=9.8, fontweight='bold', color='#B45309', zorder=11)
    ax3.text(6.0, 3.15, "• 조타각 급변 원천 차단 (dk/dt 유한)", fontsize=9.0, color=COLOR_TEXT_MAIN, zorder=11)
    ax3.text(6.0, 2.75, "• 선미 횡표류 및 전복 모멘트 억제", fontsize=9.0, color=COLOR_TEXT_MAIN, zorder=11)
    ax3.text(6.0, 2.40, "• 채터링 없는 부드러운 선회 유지", fontsize=9.0, fontweight='bold', color='#047857', zorder=11)

    # Explanation banner at bottom
    ax3.add_patch(FancyBboxPatch((0.4, 0.20), 9.2, 1.70, boxstyle="round,pad=0.05",
                                 fc='#F8FAFC', ec='#CBD5E1', lw=1.2, zorder=20))
    ax3.text(0.65, 1.62, "핵심 동작 메커니즘 및 직관적 해설", fontsize=11.2, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax3.text(0.65, 1.25, "1. 선박 위치(P0)와 진행 방향(P1), 갭 진입 방향(P2)과 목표점(P3) 등 4개 점으로 곡선을 생성합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax3.text(0.65, 0.85, "2. 시작할 때와 도착할 때의 각도를 매끄럽게 연결하여 각도가 꺾이지 않는 부드러운 곡선을 완성합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax3.text(0.65, 0.45, "3. 곡률이 연속적으로 변화하므로 배가 급격하게 흔들리지 않고 미끄러지듯 자연스럽게 회전합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)

    out_path = os.path.join(SUBFIG_DIR, 'subfig3_cubic_bezier_curvature.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 3:", out_path)

# ==============================================================================
# SUBFIGURE 4: 순수추종(Pure Pursuit) 기구학 및 차등 추진 조타 제어
# ==============================================================================
def generate_subfig4():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax4 = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax4.set_facecolor(COLOR_PANEL)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 7.0)
    ax4.axis('off')
    
    frame = FancyBboxPatch((0.05, 0.05), 9.9, 6.9, boxstyle="round,pad=0.08",
                           ec=COLOR_BORDER, fc=COLOR_PANEL, lw=1.5, zorder=1)
    ax4.add_patch(frame)
    
    # Header tag
    ax4.add_patch(FancyBboxPatch((0.4, 6.20), 9.2, 0.60, boxstyle="round,pad=0.04",
                                 fc='#EEF2FF', ec='#4F46E5', lw=1.4, zorder=2))
    ax4.text(0.65, 6.50, "기술 4: 순수추종(Pure Pursuit) 기구학 및 좌우 차등 추진 제어",
             fontsize=13.0, fontweight='bold', color='#3730A3', zorder=3)
    
    # Boat position
    bx, by = 2.2, 3.4
    heading = np.deg2rad(10)
    draw_boat(ax4, bx, by, heading, length=1.2, width=0.55, color='#4F46E5', zorder=15)
    
    # Ahead Bezier Path
    path_x = np.linspace(1.5, 8.5, 100)
    path_y = 2.8 + 1.6 * np.sin((path_x - 1.5) * 0.5)
    ax4.plot(path_x, path_y, color='#0D9488', lw=3.2, zorder=6)

    # Lookahead Distance Circle (within limits)
    L_fw = 2.9
    lookahead_circle = Circle((bx, by), L_fw, facecolor='#EEF2FF', edgecolor='#4F46E5', ls='--', lw=1.5, alpha=0.4, zorder=3)
    ax4.add_patch(lookahead_circle)
    
    # Lookahead Point Pt
    tx, ty = 4.8, 4.35
    ax4.plot(tx, ty, marker='o', markersize=11, color='#DC2626', markeredgecolor='#FFFFFF', markeredgewidth=2.0, zorder=16)
    ax4.text(tx + 0.25, ty + 0.15, "전방 주시점 Pt (Lfw=2.9m)", fontsize=9.5, fontweight='bold', color='#B91C1C', zorder=17)

    # Vector to Target
    ax4.annotate('', xy=(tx, ty), xytext=(bx, by),
                 arrowprops=dict(arrowstyle='->', color='#4F46E5', lw=2.2), zorder=10)
    
    # Heading line dashed
    hx = bx + L_fw * np.cos(heading)
    hy = by + L_fw * np.sin(heading)
    ax4.plot([bx, hx], [by, hy], color='#64748B', ls=':', lw=1.8, zorder=8)
    
    # Angle alpha arc
    alpha_arc = Arc((bx, by), 1.6, 1.6, angle=0, theta1=np.rad2deg(heading),
                    theta2=np.rad2deg(np.arctan2(ty-by, tx-bx)), color='#DC2626', lw=2.0, zorder=12)
    ax4.add_patch(alpha_arc)
    ax4.text(bx + 1.0, by + 0.35, r"$\alpha$ (오차각)", fontsize=10.0, fontweight='bold', color='#DC2626', zorder=14)

    # Pursuit Curvature Arc text
    ax4.text(3.0, 2.3, r"원호 곡률 추종: $\kappa = \frac{2\sin\alpha}{L_{fw}}$", fontsize=10.2, fontweight='bold', color='#4F46E5', zorder=14)

    # Dual Thruster Differential PWM Diagram (Right side)
    ax4.add_patch(FancyBboxPatch((6.6, 2.1), 3.0, 3.8, boxstyle="round,pad=0.04",
                                 fc='#F5F3FF', ec='#4F46E5', lw=1.2, zorder=10))
    ax4.text(8.1, 5.55, "좌우 차등 추력 제어 (PWM)", ha='center', fontsize=10.0, fontweight='bold', color='#3730A3', zorder=11)
    
    ax4.text(6.8, 5.05, "조타각 산출 공식:", fontsize=9.0, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=11)
    ax4.text(6.8, 4.60, r"$\delta = \arctan\left(\frac{2L \sin\alpha}{L_{fw}}\right)$", fontsize=10.2, color='#4F46E5', zorder=11)
    
    ax4.text(6.8, 4.00, "추진기 출력 분배:", fontsize=9.0, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=11)
    ax4.text(6.8, 3.55, r"$PWM_L = PWM_{\mathrm{base}} + \Delta PWM$", fontsize=8.8, color='#16A34A', zorder=11)
    ax4.text(6.8, 3.15, r"$PWM_R = PWM_{\mathrm{base}} - \Delta PWM$", fontsize=8.8, color='#DC2626', zorder=11)
    ax4.text(6.8, 2.60, "• 20Hz 고속 실시간 제어 루프\n• 러더 타각 물리적 지연 보상", fontsize=8.5, color=COLOR_TEXT_MUTED, zorder=11)

    # Explanation banner at bottom
    ax4.add_patch(FancyBboxPatch((0.4, 0.20), 9.2, 1.70, boxstyle="round,pad=0.05",
                                 fc='#F8FAFC', ec='#CBD5E1', lw=1.2, zorder=20))
    ax4.text(0.65, 1.62, "핵심 동작 메커니즘 및 직관적 해설", fontsize=11.2, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax4.text(0.65, 1.25, "1. 배 앞쪽으로 일정 거리(전방 주시 거리 Lfw)만큼 떨어진 궤적 상의 목표점을 실시간으로 바라봅니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax4.text(0.65, 0.85, "2. 배의 현재 방향과 목표점 사이의 각도 차이(alpha)를 줄일 수 있는 최적의 원호 회전 반경을 도출합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax4.text(0.65, 0.45, "3. 산출된 조타각에 맞춰 좌우 추진기의 추진력을 다르게 주어 목표 궤적을 정밀하게 추종합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)

    out_path = os.path.join(SUBFIG_DIR, 'subfig4_pure_pursuit_steering.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 4:", out_path)

# ==============================================================================
# SUBFIGURE 5: 3대 기술 실시간 연동 체계 및 실제 협수로 자율통과 실증
# ==============================================================================
def generate_subfig5():
    fig = plt.figure(figsize=(9.2, 7.5), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    ax5 = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax5.set_facecolor(COLOR_PANEL)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 7.0)
    ax5.axis('off')
    
    frame = FancyBboxPatch((0.05, 0.05), 9.9, 6.9, boxstyle="round,pad=0.08",
                           ec=COLOR_BORDER, fc=COLOR_PANEL, lw=1.5, zorder=1)
    ax5.add_patch(frame)
    
    # Header tag
    ax5.add_patch(FancyBboxPatch((0.4, 6.20), 9.2, 0.60, boxstyle="round,pad=0.04",
                                 fc='#F0FDF4', ec='#16A34A', lw=1.4, zorder=2))
    ax5.text(0.65, 6.50, "실전 통합: 3대 기술 실시간 연동 체계 및 실제 협수로 자율통과 실증",
             fontsize=13.0, fontweight='bold', color='#15803D', zorder=3)
    
    # Buoys forming S-curve Canal
    buoys = [
        (3.0, 5.4, '#DC2626', '부표 1'), (3.2, 2.6, '#16A34A', '부표 2'),
        (5.8, 4.8, '#DC2626', '부표 3'), (5.6, 2.0, '#16A34A', '부표 4'),
        (8.4, 5.6, '#DC2626', '부표 5'), (8.2, 2.8, '#16A34A', '부표 6')
    ]
    for bx, by, col, name in buoys:
        ax5.add_patch(Circle((bx, by), 0.38, facecolor=col, edgecolor='#0F172A', lw=1.5, alpha=0.85, zorder=5))
        ax5.add_patch(Circle((bx, by), 0.38 + 0.50 + 0.21, facecolor='#FEF2F2' if col=='#DC2626' else '#F0FDF4',
                             edgecolor=col, ls=':', lw=1.0, alpha=0.35, zorder=3))
        ax5.text(bx, by, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=6)

    # Proposed Bezier Path (Smooth S-curve through gap centers)
    path_pts = np.array([
        [0.8, 3.8],
        [2.0, 4.0],
        [3.1, 4.0],
        [4.4, 3.5],
        [5.7, 3.4],
        [7.0, 4.1],
        [8.3, 4.2],
        [9.5, 4.2]
    ])
    tck, u = splprep([path_pts[:, 0], path_pts[:, 1]], s=0, k=3)
    u_fine = np.linspace(0, 1, 150)
    sx, sy = splev(u_fine, tck)
    ax5.plot(sx, sy, color='#0D9488', lw=4.2, zorder=8)

    # Ghost Old Line Tracer Crash Path
    old_x = [0.8, 2.0, 2.7, 3.0, 3.1, 3.5]
    old_y = [3.8, 3.9, 4.6, 5.0, 5.2, 5.4]
    ax5.plot(old_x, old_y, color='#DC2626', ls='--', lw=2.2, alpha=0.7, zorder=7)
    ax5.plot(3.5, 5.4, marker='X', markersize=14, color='#DC2626', markeredgecolor='#7F1D1D', markeredgewidth=1.8, zorder=15)
    ax5.text(2.6, 5.85, "기존 방식:\n채터링 후 충돌!", fontsize=8.4, fontweight='bold', color='#DC2626', zorder=16)

    # Boat positions along successful path
    boat_steps = [(0.8, 3.8, 0.15), (3.1, 4.0, -0.2), (5.7, 3.4, 0.3), (8.3, 4.2, 0.0)]
    for step_x, step_y, step_hd in boat_steps:
        draw_boat(ax5, step_x, step_y, step_hd, length=0.85, width=0.40, color='#0284C7', alpha=0.9, zorder=12)

    # HUD Live Overlay Badge
    ax5.add_patch(FancyBboxPatch((5.6, 5.6), 4.0, 0.55, boxstyle="round,pad=0.03",
                                 fc='#0F172A', ec='#38BDF8', lw=1.2, zorder=14))
    ax5.text(5.8, 5.87, "HUD 실시간 상태: 완주 성공 (CLEAR) | 완주율 99.2%", fontsize=8.5, fontweight='bold', color='#38BDF8', zorder=15)

    # Explanation banner at bottom
    ax5.add_patch(FancyBboxPatch((0.4, 0.20), 9.2, 1.70, boxstyle="round,pad=0.05",
                                 fc='#F8FAFC', ec='#CBD5E1', lw=1.2, zorder=20))
    ax5.text(0.65, 1.62, "핵심 동작 메커니즘 및 직관적 해설", fontsize=11.2, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax5.text(0.65, 1.25, "1. 인지(DBSCAN), 계획(갭 추출 및 베지어), 제어(순수추종)가 초당 20회(20Hz) 실시간으로 유기 연동됩니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax5.text(0.65, 0.85, "2. 단순 반사 방식이 부표 사이에서 진동하며 충돌했던 난코스를 완벽한 중심선으로 돌파합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)
    ax5.text(0.65, 0.45, "3. 무거운 SLAM 지도 없이도 가벼운 온보드 컴퓨터에서 실시간으로 안전한 자율통과를 완성합니다.",
             fontsize=10.2, color=COLOR_TEXT_MAIN, zorder=21)

    out_path = os.path.join(SUBFIG_DIR, 'subfig5_integrated_pipeline_scenario.png')
    plt.savefig(out_path, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Saved Subfig 5:", out_path)

# ==============================================================================
# MASTER SHEET 4 COMPILER
# ==============================================================================
def compile_sheet4():
    print("Generating all 5 individual subfigures first...")
    generate_subfig1()
    generate_subfig2()
    generate_subfig3()
    generate_subfig4()
    generate_subfig5()
    print("All individual subfigures successfully generated.")
    
    print("Compiling Master Sheet 4...")
    # Master Canvas (18.0 x 12.0 inches at 300 DPI -> 5400 x 3600 px)
    fig = plt.figure(figsize=(18.0, 12.0), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    
    # --------------------------------------------------------------------------
    # 1. TOP MASTER HEADER
    # --------------------------------------------------------------------------
    ax_top = fig.add_axes([0.025, 0.915, 0.950, 0.068])
    ax_top.set_facecolor('none')
    ax_top.axis('off')
    
    # Category Tag
    ax_top.add_patch(FancyBboxPatch((0.0, 0.18), 0.120, 0.64, boxstyle="round,pad=0.015",
                                    fc='#0F172A', ec='none', zorder=2))
    ax_top.text(0.060, 0.50, "SOLUTION 04", ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#FFFFFF', zorder=3)
    
    # Main Headline
    ax_top.text(0.132, 0.68, "DBSCAN 군집화로 열린 공간(Gap)을 직접 탐색하고, 3차 베지어 곡선과 순수추종 제어로 급조타 없는 최적 통과 궤적을 완성했습니다.", 
                ha='left', va='center', fontsize=14.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=3)
    
    # Subtitle
    ax_top.text(0.132, 0.28, "|  인지(DBSCAN 군집화) -> 경로 계획(안전 갭 & 3차 베지어) -> 궤적 제어(Pure Pursuit) 3대 핵심 기술 연동 체계", 
                ha='left', va='center', fontsize=11.2, fontweight='bold', color=COLOR_TEXT_MUTED, zorder=3)
    
    # Right Meta Badge
    ax_top.add_patch(FancyBboxPatch((0.855, 0.18), 0.145, 0.64, boxstyle="round,pad=0.015",
                                    fc='#F1F5F9', ec='#94A3B8', lw=1.0, zorder=2))
    ax_top.text(0.9275, 0.50, "KABOAT 2026 자율운항", ha='center', va='center',
                fontsize=11.0, fontweight='bold', color='#0F172A', zorder=3)

    # --------------------------------------------------------------------------
    # 2. TOP ROW (Subplots 1 & 2, Y: 0.510 ~ 0.900, Height: 0.390)
    # --------------------------------------------------------------------------
    # SUBPLOT 1: DBSCAN Clustering (top left, width: 0.465)
    ax1 = fig.add_axes([0.025, 0.510, 0.465, 0.390])
    img1 = mpimg.imread(os.path.join(SUBFIG_DIR, 'subfig1_dbscan_clustering.png'))
    ax1.imshow(img1)
    ax1.axis('off')
    
    # SUBPLOT 2: Safe Gap Extraction (top right, width: 0.465)
    ax2 = fig.add_axes([0.510, 0.510, 0.465, 0.390])
    img2 = mpimg.imread(os.path.join(SUBFIG_DIR, 'subfig2_gap_extraction_weights.png'))
    ax2.imshow(img2)
    ax2.axis('off')

    # --------------------------------------------------------------------------
    # 3. MIDDLE SECTION DIVIDER
    # --------------------------------------------------------------------------
    fig.text(0.025, 0.472, "곡률 연속 궤적 합성부터 기구학적 경로 추종 및 실전 자율통과 검증까지의 제어 체계", 
             fontsize=13.0, fontweight='bold', color='#334155', va='center')

    # --------------------------------------------------------------------------
    # 4. BOTTOM ROW (Subplots 3, 4, 5, Y: 0.040 ~ 0.440, Height: 0.400)
    # --------------------------------------------------------------------------
    # SUBPLOT 3: Cubic Bezier Curve (bottom left, width: 0.304)
    ax3 = fig.add_axes([0.025, 0.040, 0.304, 0.400])
    img3 = mpimg.imread(os.path.join(SUBFIG_DIR, 'subfig3_cubic_bezier_curvature.png'))
    ax3.imshow(img3)
    ax3.axis('off')

    # SUBPLOT 4: Pure Pursuit & Thrusters (bottom center, width: 0.304)
    ax4 = fig.add_axes([0.347, 0.040, 0.304, 0.400])
    img4 = mpimg.imread(os.path.join(SUBFIG_DIR, 'subfig4_pure_pursuit_steering.png'))
    ax4.imshow(img4)
    ax4.axis('off')

    # SUBPLOT 5: Integrated S-Curve Scenario (bottom right, width: 0.304)
    ax5 = fig.add_axes([0.669, 0.040, 0.304, 0.400])
    img5 = mpimg.imread(os.path.join(SUBFIG_DIR, 'subfig5_integrated_pipeline_scenario.png'))
    ax5.imshow(img5)
    ax5.axis('off')

    plt.savefig(MASTER_PNG, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print("Master Sheet 4 successfully compiled:", MASTER_PNG)

if __name__ == '__main__':
    compile_sheet4()
