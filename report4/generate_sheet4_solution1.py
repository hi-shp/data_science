#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Exhibition Sheet 4 Generator: [해결 방법 1] 인지 및 공간 탐색
Covering Technology 1 (DBSCAN Clustering) & Technology 2 (Safe Gap Extraction & Priority Scoring)
Canvas: 140 : 100 Aspect Ratio (14.0 x 10.0 inches at 300 DPI -> 4200 x 3000 px)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, Wedge

# Matplotlib Korean font configuration
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/soonhong/kaboat/report4'
SUBFIG_DIR = os.path.join(OUTPUT_DIR, 'sheet4_subfigures')
os.makedirs(SUBFIG_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUTPUT_DIR, 'sheet4_solution1_dbscan_gap.png')

# Color Palette Constants
COLOR_BG = '#F8FAFC'
COLOR_CARD_BG = '#FFFFFF'
COLOR_BORDER = '#CBD5E1'
COLOR_BORDER_STRONG = '#94A3B8'
COLOR_TEXT_TITLE = '#0F172A'
COLOR_TEXT_MAIN = '#1E293B'
COLOR_TEXT_MUTED = '#475569'

COLOR_PRIMARY = '#0284C7'
COLOR_TEAL = '#0D9488'
COLOR_DANGER = '#DC2626'

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

def generate_sheet4():
    # Canvas 14.0 x 10.0 inches (140:100 aspect ratio)
    fig = plt.figure(figsize=(14.0, 10.0), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    
    # -------------------------------------------------------------------------
    # 1. TOP MASTER HEADER (Y: 0.915 ~ 0.985, Height: 0.070)
    # -------------------------------------------------------------------------
    ax_top = fig.add_axes([0.025, 0.915, 0.950, 0.070])
    ax_top.axis('off')
    
    # Category Tag
    ax_top.add_patch(FancyBboxPatch((0.0, 0.15), 0.125, 0.70, boxstyle="round,pad=0.015",
                                    fc='#0F172A', ec='none', zorder=2))
    ax_top.text(0.0625, 0.50, "SOLUTION 01", ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#FFFFFF', zorder=3)
    
    # Title & Master Headline
    ax_top.text(0.148, 0.72, "해결 방법 1: 인지 및 공간 탐색 (DBSCAN 군집화 & 안전 갭 추출)", 
                ha='left', va='center', fontsize=17.5, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=3)
    
    # Declarative Subtitle
    ax_top.text(0.148, 0.28, "|  밀도 기반 군집화로 수면 잡음을 제거하고, 선체 폭과 안전마진(0.5m)을 확보한 통과 갭을 탐색합니다.", 
                ha='left', va='center', fontsize=11.5, fontweight='bold', color=COLOR_TEXT_MUTED, zorder=3)
    
    # Right Meta Badge
    ax_top.add_patch(FancyBboxPatch((0.835, 0.15), 0.165, 0.70, boxstyle="round,pad=0.015",
                                    fc='#F1F5F9', ec='#94A3B8', lw=1.0, zorder=2))
    ax_top.text(0.9175, 0.50, "KABOAT 2026 자율운항", ha='center', va='center',
                fontsize=11.0, fontweight='bold', color='#0F172A', zorder=3)

    # -------------------------------------------------------------------------
    # 2. TWO-COLUMN MAIN BODY (Y: 0.025 ~ 0.895, Height: 0.870)
    # Left Column: 기술 1 - DBSCAN 공간 밀도 기반 군집화
    # Right Column: 기술 2 - 안전 갭(Gap) 탐색 및 우선순위 평가
    # -------------------------------------------------------------------------
    col_w = 0.465
    col_h = 0.870
    y_base = 0.025
    
    # =========================================================================
    # LEFT COLUMN: 기술 1 - DBSCAN
    # =========================================================================
    ax_col1 = fig.add_axes([0.025, y_base, col_w, col_h])
    ax_col1.set_facecolor(COLOR_CARD_BG)
    ax_col1.set_xlim(0, 10)
    ax_col1.set_ylim(0, 10)
    ax_col1.axis('off')
    
    # Outer Card Frame
    frame1 = FancyBboxPatch((0.05, 0.05), 9.9, 9.9, boxstyle="round,pad=0.08",
                            ec=COLOR_BORDER, fc=COLOR_CARD_BG, lw=1.5, zorder=1)
    ax_col1.add_patch(frame1)
    
    # Section Header Ribbon
    ax_col1.add_patch(FancyBboxPatch((0.3, 9.15), 9.4, 0.65, boxstyle="round,pad=0.04",
                                     fc='#E0F2FE', ec='#0284C7', lw=1.4, zorder=2))
    ax_col1.text(0.55, 9.48, "1. DBSCAN 밀도 기반 군집화 및 수면 난반사 잡음 제거",
                 fontsize=13.5, fontweight='bold', color='#0369A1', zorder=3)

    # Visualization Area (Y: 4.60 ~ 9.00)
    # Boat at (1.4, 6.6)
    draw_boat(ax_col1, 1.4, 6.6, np.deg2rad(15), length=1.2, width=0.55, color='#0284C7', zorder=15)
    ax_col1.text(1.4, 5.65, "자율운항 USV\n(2D 라이다 탑재)", ha='center', fontsize=10.0, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=16)
    
    # 120 degree LiDAR FOV beam cone
    wedge = Wedge((1.4, 6.6), 7.6, -45, 75, facecolor='#E0F2FE', edgecolor='#38BDF8',
                  linestyle='--', linewidth=1.4, alpha=0.40, zorder=2)
    ax_col1.add_patch(wedge)
    ax_col1.text(3.8, 8.65, "라이다 120° 전방 탐색 영역 (FOV)", fontsize=9.8, color='#0284C7', fontweight='bold', zorder=4)

    # Buoy A (Red Buoy, Left/Top)
    cA = (6.6, 7.8)
    ax_col1.add_patch(Circle(cA, 0.48, facecolor='#FCA5A5', edgecolor='#DC2626', lw=2.2, zorder=5))
    ax_col1.text(cA[0], cA[1], "부표 A\n(좌현)", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#991B1B', zorder=6)
    
    np.random.seed(42)
    anglesA = np.linspace(np.pi*0.75, np.pi*1.55, 15)
    ptsA_x = cA[0] + 0.48 * np.cos(anglesA) + np.random.normal(0, 0.03, len(anglesA))
    ptsA_y = cA[1] + 0.48 * np.sin(anglesA) + np.random.normal(0, 0.03, len(anglesA))
    ax_col1.scatter(ptsA_x, ptsA_y, color='#0284C7', s=60, edgecolors='#0F172A', lw=1.0, zorder=10)

    # Buoy B (Green Buoy, Right/Bottom)
    cB = (6.8, 5.2)
    ax_col1.add_patch(Circle(cB, 0.48, facecolor='#86EFAC', edgecolor='#16A34A', lw=2.2, zorder=5))
    ax_col1.text(cB[0], cB[1], "부표 B\n(우현)", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#14532D', zorder=6)
    
    anglesB = np.linspace(np.pi*0.65, np.pi*1.45, 14)
    ptsB_x = cB[0] + 0.48 * np.cos(anglesB) + np.random.normal(0, 0.03, len(anglesB))
    ptsB_y = cB[1] + 0.48 * np.sin(anglesB) + np.random.normal(0, 0.03, len(anglesB))
    ax_col1.scatter(ptsB_x, ptsB_y, color='#0284C7', s=60, edgecolors='#0F172A', lw=1.0, zorder=10)

    # Stray water reflection noise points (placed safely within FOV whitespace)
    noise_x = [3.6, 4.2, 5.0, 8.2]
    noise_y = [7.3, 5.3, 4.7, 6.6]
    ax_col1.scatter(noise_x, noise_y, marker='x', color='#DC2626', s=70, lw=2.2, zorder=12)
    ax_col1.text(3.75, 7.30, "수면 잡음 (제거)", fontsize=8.8, color='#DC2626', fontweight='bold', zorder=12)
    ax_col1.text(4.35, 5.30, "수면 잡음 (제거)", fontsize=8.8, color='#DC2626', fontweight='bold', zorder=12)
    ax_col1.text(5.15, 4.70, "수면 잡음 (제거)", fontsize=8.8, color='#DC2626', fontweight='bold', zorder=12)
    ax_col1.text(8.35, 6.60, "수면 잡음", fontsize=8.8, color='#DC2626', fontweight='bold', zorder=12)

    # Epsilon Radius Circle Demo
    demo_pt = (ptsA_x[5], ptsA_y[5])
    eps_circle = Circle(demo_pt, 0.65, facecolor='#38BDF8', edgecolor='#0284C7', ls='--', lw=1.6, alpha=0.35, zorder=7)
    ax_col1.add_patch(eps_circle)
    ax_col1.annotate(r"$\epsilon$ 이웃 탐색 반경 ($\epsilon=0.6\mathrm{m}$)" + "\n" + r"이웃 점수 $\geq \mathrm{MinPts}(3)$ 만족",
                     xy=(demo_pt[0]-0.25, demo_pt[1]+0.2), xytext=(2.6, 8.2),
                     arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.8),
                     fontsize=9.5, fontweight='bold', color='#0369A1',
                     bbox=dict(boxstyle='round,pad=0.25', fc='#FFFFFF', ec='#0284C7', lw=1.2), zorder=20)

    # Cluster Bounding Hulls
    hullA = FancyBboxPatch((cA[0]-0.75, cA[1]-0.75), 1.5, 1.5, boxstyle="round,pad=0.08",
                           ec='#0284C7', fc='none', ls='-', lw=2.2, zorder=8)
    ax_col1.add_patch(hullA)
    ax_col1.text(cA[0]+0.90, cA[1]+0.2, "군집 1 결속\n(부표 객체 A)", fontsize=10.0, fontweight='bold', color='#0369A1', zorder=12)

    hullB = FancyBboxPatch((cB[0]-0.75, cB[1]-0.75), 1.5, 1.5, boxstyle="round,pad=0.08",
                           ec='#16A34A', fc='none', ls='-', lw=2.2, zorder=8)
    ax_col1.add_patch(hullB)
    ax_col1.text(cB[0]+0.90, cB[1]-0.2, "군집 2 결속\n(부표 객체 B)", fontsize=10.0, fontweight='bold', color='#15803D', zorder=12)

    # Middle Technical Spec Box (Y: 2.70 ~ 4.40)
    ax_col1.add_patch(FancyBboxPatch((0.3, 2.70), 9.4, 1.70, boxstyle="round,pad=0.04",
                                     fc='#F0F9FF', ec='#BAE6FD', lw=1.2, zorder=10))
    ax_col1.text(0.55, 4.15, "DBSCAN 알고리즘의 주요 파라미터 및 노이즈 필터링 기준", fontsize=11.2, fontweight='bold', color='#0369A1', zorder=11)
    specs1 = [
        ("• 탐색 반경 (Epsilon)", r"$\epsilon = 0.60\mathrm{m}$ : 부표 직경(0.4m) 대비 최적의 점군 결속 거리 확보"),
        ("• 최소 점 개수 (MinPts)", r"$\mathrm{MinPts} = 3$ : 수면 난반사로 단발성 발생하는 잡음 점군 즉각 분리"),
        ("• 연산 처리 속도", "20Hz 실시간 루프 (프레임당 연산 소요 시간 3.8ms 이내 고속 처리)")
    ]
    for idx, (shead, sbody) in enumerate(specs1):
        sy = 3.75 - idx * 0.40
        ax_col1.text(0.65, sy, shead, fontsize=9.8, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=11)
        ax_col1.text(3.55, sy, sbody, fontsize=9.5, color=COLOR_TEXT_MAIN, zorder=11)

    # Bottom Core Explanatory Card (Y: 0.30 ~ 2.50)
    ax_col1.add_patch(FancyBboxPatch((0.3, 0.30), 9.4, 2.20, boxstyle="round,pad=0.05",
                                     fc='#F8FAFC', ec='#CBD5E1', lw=1.4, zorder=20))
    ax_col1.text(0.55, 2.18, "핵심 동작 원리 및 엔지니어링 해설 (완결형 문장)", fontsize=12.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax_col1.text(0.55, 1.68, "1. 120° 라이다가 수집한 수많은 2D 점군 중 수면 난반사 잡음을 밀도 기준(MinPts)으로 즉각 걸러냅니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col1.text(0.55, 1.20, "2. 부표 표면에 오밀조밀 모인 유효 점들만 자동으로 결속하여 독립된 부표 객체(중심점, 반경)로 변환합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col1.text(0.55, 0.72, "3. 사전에 부표의 개수나 위치를 지정하지 않아도 주변 환경의 장애물 변화를 실시간으로 유연하게 인식합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)

    # =========================================================================
    # RIGHT COLUMN: 기술 2 - 안전 갭 탐색 및 우선순위 평가
    # =========================================================================
    ax_col2 = fig.add_axes([0.510, y_base, col_w, col_h])
    ax_col2.set_facecolor(COLOR_CARD_BG)
    ax_col2.set_xlim(0, 10)
    ax_col2.set_ylim(0, 10)
    ax_col2.axis('off')
    
    # Outer Card Frame
    frame2 = FancyBboxPatch((0.05, 0.05), 9.9, 9.9, boxstyle="round,pad=0.08",
                            ec=COLOR_BORDER, fc=COLOR_CARD_BG, lw=1.5, zorder=1)
    ax_col2.add_patch(frame2)
    
    # Section Header Ribbon
    ax_col2.add_patch(FancyBboxPatch((0.3, 9.15), 9.4, 0.65, boxstyle="round,pad=0.04",
                                     fc='#CCFBF1', ec='#0D9488', lw=1.4, zorder=2))
    ax_col2.text(0.55, 9.48, "2. 안전 갭(Gap) 후보 추출 및 다중 목적 가중치 평가",
                 fontsize=13.5, fontweight='bold', color='#0F766E', zorder=3)

    # Visualization Area (Y: 4.60 ~ 9.00)
    c2A = (4.1, 7.8)
    c2B = (4.1, 4.8)
    r_buoy2 = 0.48
    margin2 = 0.50
    boat_hw2 = 0.21
    
    # Safety Buffers
    ax_col2.add_patch(Circle(c2A, r_buoy2 + margin2 + boat_hw2, facecolor='#FEF2F2', edgecolor='#F87171', ls='--', lw=1.4, alpha=0.55, zorder=3))
    ax_col2.add_patch(Circle(c2B, r_buoy2 + margin2 + boat_hw2, facecolor='#FEF2F2', edgecolor='#F87171', ls='--', lw=1.4, alpha=0.55, zorder=3))
    
    # Buoys Solid
    ax_col2.add_patch(Circle(c2A, r_buoy2, facecolor='#FCA5A5', edgecolor='#DC2626', lw=2.2, zorder=5))
    ax_col2.add_patch(Circle(c2B, r_buoy2, facecolor='#86EFAC', edgecolor='#16A34A', lw=2.2, zorder=5))
    ax_col2.text(c2A[0], c2A[1], "부표 A", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#991B1B', zorder=6)
    ax_col2.text(c2B[0], c2B[1], "부표 B", ha='center', va='center', fontsize=9.8, fontweight='bold', color='#14532D', zorder=6)
    
    # Dimension Line: Buoy center distance (placed to the left side of buoys)
    ax_col2.annotate('', xy=(3.2, c2A[1]), xytext=(3.2, c2B[1]),
                     arrowprops=dict(arrowstyle='<->', color='#64748B', lw=1.8), zorder=8)
    ax_col2.text(3.05, 6.30, "부표 간 거리\n3.00m", ha='right', va='center', fontsize=9.2, color='#475569', fontweight='bold', zorder=9)

    # Green safe portal bar in between
    gap_top = c2A[1] - (r_buoy2 + margin2 + boat_hw2)
    gap_bot = c2B[1] + (r_buoy2 + margin2 + boat_hw2)
    ax_col2.add_patch(FancyBboxPatch((3.85, gap_bot), 0.5, gap_top - gap_bot, boxstyle="round,pad=0.03",
                                     fc='#10B981', ec='#047857', lw=2.0, alpha=0.45, zorder=7))
    ax_col2.text(4.1, 7.05, "유효 통과 폭: 1.82m", ha='center', fontsize=8.8, fontweight='bold', color='#047857',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#10B981', lw=1.0), zorder=12)

    # Target Gap Midpoint
    p_gap2 = (4.1, 6.30)
    ax_col2.plot(p_gap2[0], p_gap2[1], marker='o', markersize=13, color='#0D9488', markeredgecolor='#FFFFFF', markeredgewidth=2.2, zorder=14)
    ax_col2.text(p_gap2[0] + 0.30, p_gap2[1] + 0.05, "최적 갭 목표점\n(P_gap, 중심점)", fontsize=9.2, fontweight='bold', color='#0F766E', zorder=15)

    # Approaching Boat
    draw_boat(ax_col2, 1.0, 5.6, np.deg2rad(10), length=1.2, width=0.55, color='#0284C7', zorder=15)
    ax_col2.text(1.0, 4.65, "USV 주행 진입", ha='center', fontsize=10.0, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=16)

    # Approach Vector
    ax_col2.annotate('', xy=(p_gap2[0] - 0.2, p_gap2[1]), xytext=(1.0, 5.6),
                     arrowprops=dict(arrowstyle='->', color='#0D9488', lw=2.4), zorder=10)
    ax_col2.text(1.85, 5.25, "목표 갭 벡터 (거리: 3.2m)", fontsize=8.8, fontweight='bold', color='#0D9488',
                 bbox=dict(boxstyle='round,pad=0.15', fc='#FFFFFF', ec='#0D9488', lw=0.8), zorder=11)

    # Evaluation Criteria Card on right (Clean non-overlapping table)
    ax_col2.add_patch(FancyBboxPatch((7.0, 4.6), 2.7, 4.2, boxstyle="round,pad=0.05",
                                     fc='#F0FDFA', ec='#0D9488', lw=1.2, zorder=10))
    ax_col2.text(8.30, 8.45, "6대 가중치 우선순위 체계", ha='center', fontsize=10.5, fontweight='bold', color='#0F766E', zorder=11)
    
    weights2 = [
        ("목표 진행성", "23%"),
        ("전방 지향성", "22%"),
        ("통로 안전폭", "16%"),
        ("장애물 이격", "16%"),
        ("선박 헤딩각", "15%"),
        ("통로 수직도", "8%")
    ]
    for idx, (wname, wval) in enumerate(weights2):
        wy = 7.95 - idx * 0.50
        ax_col2.text(7.10, wy, f"• {wname}", fontsize=9.2, color=COLOR_TEXT_MAIN, zorder=12)
        ax_col2.text(9.45, wy, wval, ha='right', fontsize=9.2, fontweight='bold', color='#0D9488', zorder=12)

    # Middle Technical Spec Box (Y: 2.70 ~ 4.40)
    ax_col2.add_patch(FancyBboxPatch((0.3, 2.70), 9.4, 1.70, boxstyle="round,pad=0.04",
                                     fc='#CCFBF1', ec='#99F6E4', lw=1.2, zorder=10))
    ax_col2.text(0.55, 4.15, "안전 갭 필터링 및 다중 목적 우선순위 평가 수식", fontsize=11.2, fontweight='bold', color='#0F766E', zorder=11)
    specs2 = [
        ("• 통과 가능성 검증", r"$W_{\mathrm{eff}} = W_{\mathrm{raw}} - (R_A + R_B) - 2 \cdot (\mathrm{Margin} + B/2) \geq 0.50\mathrm{m}$"),
        ("• 종합 가중합 점수", r"$\mathrm{Score}(\mathrm{Gap}_k) = \sum_{i=1}^6 w_i \cdot f_i(\mathrm{Gap}_k) \quad (\sum w_i = 1.0)$"),
        ("• 능동 공간 추종", "장애물 반사 회피의 한계를 극복하고 열린 자유 공간(Gap)으로 직접 지향")
    ]
    for idx, (shead, sbody) in enumerate(specs2):
        sy = 3.75 - idx * 0.40
        ax_col2.text(0.65, sy, shead, fontsize=9.8, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=11)
        ax_col2.text(3.55, sy, sbody, fontsize=9.5, color=COLOR_TEXT_MAIN, zorder=11)

    # Bottom Core Explanatory Card (Y: 0.30 ~ 2.50)
    ax_col2.add_patch(FancyBboxPatch((0.3, 0.30), 9.4, 2.20, boxstyle="round,pad=0.05",
                                     fc='#F8FAFC', ec='#CBD5E1', lw=1.4, zorder=20))
    ax_col2.text(0.55, 2.18, "핵심 동작 원리 및 엔지니어링 해설 (완결형 문장)", fontsize=12.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax_col2.text(0.55, 1.68, "1. 인접한 두 부표 사이 거리를 측정하고 선폭과 안전마진(0.5m)을 제외한 실제 통과 가능 폭을 계산합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col2.text(0.55, 1.20, "2. 통로 폭이 충분히 확보된 유효 갭만을 후보로 추출하여 충돌 위험이 있는 좁은 틈은 계획 단계에서 배제합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col2.text(0.55, 0.72, "3. 통로 너비와 진행 방향 등 6대 평가 항목의 가중합을 계산하여 가장 안전하고 빠른 목표점을 선정합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)

    plt.savefig(OUT_PNG, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print(f"Sheet 4 successfully generated: {OUT_PNG}")

if __name__ == '__main__':
    generate_sheet4()

