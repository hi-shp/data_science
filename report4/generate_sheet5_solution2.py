#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Exhibition Sheet 5 Generator: [해결 방법 2] 궤적 생성 및 운동 제어
Covering Technology 3 (Cubic Bézier Curve) & Technology 4 (Pure Pursuit & Differential Thrust)
Canvas: 140 : 100 Aspect Ratio (14.0 x 10.0 inches at 300 DPI -> 4200 x 3000 px)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch, Wedge, Arc

# Matplotlib Korean font configuration
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/soonhong/kaboat/report4'
OUT_PNG = os.path.join(OUTPUT_DIR, 'sheet5_solution2_bezier_purepursuit.png')

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
COLOR_INDIGO = '#4F46E5'
COLOR_DANGER = '#DC2626'
COLOR_SUCCESS = '#16A34A'

def draw_boat_detailed(ax, x, y, heading_rad, length=1.2, width=0.55, color='#0284C7', ec='#0F172A',
                       show_thrusters=False, tl_len=0.7, tr_len=0.35, zorder=10):
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
        poly = Polygon(p_rot, closed=True, facecolor=fc, edgecolor=ec, lw=1.2, alpha=0.95, zorder=zorder)
        ax.add_patch(poly)
    
    # Heading line
    ax.annotate('', xy=(x + length*0.65*c, y + length*0.65*s), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.6), zorder=zorder+2)
    
    # Twin thrusters at stern
    if show_thrusters:
        prop_y = hw * 0.75
        # Left Thruster at (-hl, prop_y)
        pt_l = np.array([x, y]) + R @ np.array([-hl, prop_y])
        pt_r = np.array([x, y]) + R @ np.array([-hl, -prop_y])
        
        # Propeller motors
        for pt in [pt_l, pt_r]:
            ax.add_patch(Circle(pt, 0.08, facecolor='#334155', edgecolor='#0F172A', lw=1.0, zorder=zorder+3))
            
        # Left thrust vector
        vec_l = pt_l + R @ np.array([tl_len, 0])
        ax.annotate('', xy=(vec_l[0], vec_l[1]), xytext=(pt_l[0], pt_l[1]),
                    arrowprops=dict(arrowstyle='->', color='#2563EB', lw=2.4), zorder=zorder+4)
        ax.text(pt_l[0] - 0.35, pt_l[1] + 0.15, r"$T_L$",
                fontsize=9.2, fontweight='bold', color='#1D4ED8',
                bbox=dict(boxstyle='circle,pad=0.1', fc='#FFFFFF', ec='#1D4ED8', lw=0.8), zorder=zorder+5)
        
        # Right thrust vector
        vec_r = pt_r + R @ np.array([tr_len, 0])
        ax.annotate('', xy=(vec_r[0], vec_r[1]), xytext=(pt_r[0], pt_r[1]),
                    arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.8), zorder=zorder+4)
        ax.text(pt_r[0] - 0.35, pt_r[1] - 0.25, r"$T_R$",
                fontsize=9.2, fontweight='bold', color='#0369A1',
                bbox=dict(boxstyle='circle,pad=0.1', fc='#FFFFFF', ec='#0369A1', lw=0.8), zorder=zorder+5)

def generate_sheet5():
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
    ax_top.text(0.0625, 0.50, "SOLUTION 02", ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#FFFFFF', zorder=3)
    
    # Title & Master Headline
    ax_top.text(0.148, 0.72, "해결 방법 2: 궤적 생성 및 운동 제어 (3차 베지어 곡선 & 순수추종)", 
                ha='left', va='center', fontsize=17.5, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=3)
    
    # Declarative Subtitle
    ax_top.text(0.148, 0.28, "|  곡률 연속(C2) 궤적으로 급격한 조향 변동을 방지하고, 전방 주시 기반 순수추종으로 외란에 강인하게 추종합니다.", 
                ha='left', va='center', fontsize=11.5, fontweight='bold', color=COLOR_TEXT_MUTED, zorder=3)
    
    # Right Meta Badge
    ax_top.add_patch(FancyBboxPatch((0.835, 0.15), 0.165, 0.70, boxstyle="round,pad=0.015",
                                    fc='#F1F5F9', ec='#94A3B8', lw=1.0, zorder=2))
    ax_top.text(0.9175, 0.50, "KABOAT 2026 자율운항", ha='center', va='center',
                fontsize=11.0, fontweight='bold', color='#0F766E', zorder=3)

    # -------------------------------------------------------------------------
    # 2. TWO-COLUMN MAIN BODY (Y: 0.025 ~ 0.895, Height: 0.870)
    # Left Column: 기술 3 - 3차 베지어 곡선 기반 부드러운 궤적 합성
    # Right Column: 기술 4 - 순수추종 제어 및 좌우 차등 추진 제어
    # -------------------------------------------------------------------------
    col_w = 0.465
    col_h = 0.870
    y_base = 0.025
    
    # =========================================================================
    # LEFT COLUMN: 기술 3 - 3차 베지어 곡선
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
                                     fc='#EEF2FF', ec='#4F46E5', lw=1.4, zorder=2))
    ax_col1.text(0.55, 9.48, "3. 3차 베지어 곡선(Cubic Bézier) 기반 부드러운 궤적 합성",
                 fontsize=13.5, fontweight='bold', color='#4338CA', zorder=3)

    # Visualization Area (Y: 4.60 ~ 9.00)
    # Control Points Definition (spaced for maximum visual clarity)
    P0 = np.array([1.3, 6.0])   # USV current position
    P1 = np.array([3.4, 6.8])   # Tangent forward from current heading
    P2 = np.array([5.2, 7.9])   # Approach alignment towards gate
    P3 = np.array([7.8, 6.8])   # Target safe gap midpoint
    
    # Buoys at gate P3
    c3A = (7.8, 8.2)
    c3B = (7.8, 5.4)
    ax_col1.add_patch(Circle(c3A, 0.42, facecolor='#FCA5A5', edgecolor='#DC2626', lw=2.0, zorder=5))
    ax_col1.add_patch(Circle(c3B, 0.42, facecolor='#86EFAC', edgecolor='#16A34A', lw=2.0, zorder=5))
    ax_col1.text(c3A[0], c3A[1], "부표 A", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#991B1B', zorder=6)
    ax_col1.text(c3B[0], c3B[1], "부표 B", ha='center', va='center', fontsize=9.0, fontweight='bold', color='#14532D', zorder=6)
    
    # USV Boat at P0
    draw_boat_detailed(ax_col1, P0[0], P0[1], np.deg2rad(20), length=1.1, width=0.50, color='#4F46E5', zorder=15)
    ax_col1.text(P0[0], P0[1]-0.80, "현재 선박 위치 (P0)", ha='center', fontsize=9.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=16)

    # Control Polygon (P0 -> P1 -> P2 -> P3)
    poly_pts = np.vstack([P0, P1, P2, P3])
    ax_col1.plot(poly_pts[:, 0], poly_pts[:, 1], 'o--', color='#94A3B8', lw=1.5, markersize=8,
                 markerfacecolor='#F8FAFC', markeredgecolor='#475569', markeredgewidth=1.8, zorder=8)
    
    # Cubic Bézier Curve calculation
    t_vals = np.linspace(0, 1, 100)
    bezier_curve = np.array([
        ((1-t)**3)*P0 + 3*((1-t)**2)*t*P1 + 3*(1-t)*(t**2)*P2 + (t**3)*P3
        for t in t_vals
    ])
    ax_col1.plot(bezier_curve[:, 0], bezier_curve[:, 1], color='#2563EB', lw=3.6, zorder=12)
    
    # Bézier Curve label
    ax_col1.text(4.2, 7.05, "3차 베지어 곡선 B(t)\n[C2 곡률 연속 경로]", fontsize=9.3, fontweight='bold', color='#1D4ED8',
                 bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#3B82F6', lw=1.2), zorder=15)

    # Control Point Markers and Annotations
    ax_col1.plot(P0[0], P0[1], marker='o', markersize=10, color='#4F46E5', zorder=20)
    ax_col1.plot(P1[0], P1[1], marker='s', markersize=9, color='#0284C7', zorder=20)
    ax_col1.plot(P2[0], P2[1], marker='s', markersize=9, color='#0284C7', zorder=20)
    ax_col1.plot(P3[0], P3[1], marker='o', markersize=11, color='#10B981', markeredgecolor='#047857', markeredgewidth=2.0, zorder=20)
    
    ax_col1.text(P1[0]-0.2, P1[1]-0.45, "P1 (헤딩 접선 제어점)", fontsize=9.0, fontweight='bold', color='#0284C7', zorder=21)
    ax_col1.text(P2[0]-0.3, P2[1]+0.35, "P2 (통로 정렬 제어점)", fontsize=9.0, fontweight='bold', color='#0284C7', zorder=21)
    ax_col1.text(P3[0]+0.45, P3[1], "P3 (목표 갭 중심)", fontsize=9.2, fontweight='bold', color='#047857', zorder=21)

    # Comparison: Conventional Kinked Path (Ghost red dashed line, placed at safe height)
    P_kink = np.array([4.4, 8.4])
    ax_col1.plot([P0[0], P_kink[0], P3[0]], [P0[1], P_kink[1], P3[1]], ':', color='#EF4444', lw=2.0, alpha=0.85, zorder=7)
    ax_col1.plot(P_kink[0], P_kink[1], marker='^', markersize=8, color='#DC2626', zorder=8)
    ax_col1.text(P_kink[0]+0.15, P_kink[1]+0.22, "기존 직선 경유 (급격한 꺾임: 조타 진동)",
                 ha='center', fontsize=8.6, fontweight='bold', color='#DC2626',
                 bbox=dict(boxstyle='round,pad=0.15', fc='#FEF2F2', ec='#FCA5A5', lw=0.9), zorder=14)

    # Dynamic Feature Callout (Curvature limit)
    ax_col1.add_patch(FancyBboxPatch((0.4, 7.80), 2.8, 1.15, boxstyle="round,pad=0.04",
                                     fc='#F8FAFC', ec='#CBD5E1', lw=1.0, zorder=10))
    ax_col1.text(0.55, 8.65, "동역학 곡률 제약 준수", fontsize=9.5, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=11)
    ax_col1.text(0.55, 8.20, "• 곡률 반경: R ≥ 1.5m 보장\n• 최대 조타각: δ ≤ 28.5° 구속", fontsize=8.6, color=COLOR_TEXT_MUTED, zorder=11)

    # Middle Technical Spec Box (Y: 2.70 ~ 4.40)
    ax_col1.add_patch(FancyBboxPatch((0.3, 2.70), 9.4, 1.70, boxstyle="round,pad=0.04",
                                     fc='#EEF2FF', ec='#C7D2FE', lw=1.2, zorder=10))
    ax_col1.text(0.55, 4.15, "3차 베지어 곡선의 수학적 정식화 및 동역학적 특성", fontsize=11.2, fontweight='bold', color='#4338CA', zorder=11)
    specs3 = [
        ("• 매개변수 곡선식", r"$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3 \quad (t \in [0, 1])$"),
        ("• 경계 접선 조건", r"$B'(0) = 3(P_1 - P_0) \parallel \mathbf{v}_{\mathrm{USV}}, \quad B'(1) = 3(P_3 - P_2) \parallel \mathbf{n}_{\mathrm{gap}}$"),
        ("• 곡률 연속성(C2)", r"곡률 변화율 $\frac{d\kappa}{dt}$가 유계되어 선체 롤링 및 조타 채터링을 물리적으로 차단")
    ]
    for idx, (shead, sbody) in enumerate(specs3):
        sy = 3.75 - idx * 0.40
        ax_col1.text(0.65, sy, shead, fontsize=9.8, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=11)
        ax_col1.text(3.35, sy, sbody, fontsize=9.4, color=COLOR_TEXT_MAIN, zorder=11)

    # Bottom Core Explanatory Card (Y: 0.30 ~ 2.50)
    ax_col1.add_patch(FancyBboxPatch((0.3, 0.30), 9.4, 2.20, boxstyle="round,pad=0.05",
                                     fc='#F8FAFC', ec='#CBD5E1', lw=1.4, zorder=20))
    ax_col1.text(0.55, 2.18, "핵심 동작 원리 및 엔지니어링 해설 (완결형 문장)", fontsize=12.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax_col1.text(0.55, 1.68, "1. 현재 선박 위치(P0)와 갭 중심(P3) 사이에 조향 연속성을 보장하는 4개의 제어점을 배치합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col1.text(0.55, 1.20, "2. 선박의 현재 헤딩과 통로 진입 방향을 접선 벡터로 구속하여 급격한 꺾임 없는 매끄러운 궤적을 합성합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col1.text(0.55, 0.72, "3. 곡률(κ)의 급격한 변화를 억제함으로써 선체 관성으로 인한 오버슈트와 조향 진동을 근본적으로 차단합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)

    # =========================================================================
    # RIGHT COLUMN: 기술 4 - 순수추종 제어 및 좌우 차등 추진 제어
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
                                     fc='#E0F2FE', ec='#0284C7', lw=1.4, zorder=2))
    ax_col2.text(0.55, 9.48, "4. 순수추종(Pure Pursuit) 및 좌우 차등 추진 제어",
                 fontsize=13.5, fontweight='bold', color='#0369A1', zorder=3)

    # Visualization Area (Y: 4.60 ~ 9.00)
    boat_x, boat_y = 1.8, 5.8
    heading_deg = 15
    heading_rad = np.deg2rad(heading_deg)
    
    # Draw USV with twin thrusters
    draw_boat_detailed(ax_col2, boat_x, boat_y, heading_rad, length=1.3, width=0.58, color='#0284C7',
                       show_thrusters=True, tl_len=0.8, tr_len=0.4, zorder=15)
    ax_col2.text(boat_x, boat_y - 1.05, "자율운항 USV\n(좌우 듀얼 트러스터)", ha='center', fontsize=9.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=16)

    # Path Curve (Bézier Path being followed)
    t_path = np.linspace(0, 1, 100)
    path_x = 0.8 + 8.8 * t_path
    path_y = 4.8 + 2.9 * np.sin(t_path * np.pi * 0.75)
    ax_col2.plot(path_x, path_y, color='#0284C7', lw=2.4, ls='-', zorder=6)
    ax_col2.text(8.6, 7.85, "계획 경로", fontsize=9.0, fontweight='bold', color='#0284C7', zorder=8)

    # Lookahead circle centered at USV
    L_fw = 2.8  # Lookahead distance
    lookahead_circle = Circle((boat_x, boat_y), L_fw, facecolor='none', edgecolor='#38BDF8', ls='--', lw=1.6, alpha=0.85, zorder=5)
    ax_col2.add_patch(lookahead_circle)
    
    # Target Lookahead Point Pt (Intersection of lookahead circle and path)
    pt_x = boat_x + L_fw * np.cos(np.deg2rad(40))
    pt_y = boat_y + L_fw * np.sin(np.deg2rad(40))
    
    ax_col2.plot([boat_x, pt_x], [boat_y, pt_y], color='#0D9488', lw=2.2, ls='-', zorder=10)
    ax_col2.plot(pt_x, pt_y, marker='o', markersize=12, color='#0D9488', markeredgecolor='#FFFFFF', markeredgewidth=2.2, zorder=14)
    ax_col2.text(pt_x + 0.20, pt_y + 0.15, r"목표 주시점 ($P_t$)" + "\n[거리 $L_{fw}=2.9\mathrm{m}$]",
                 fontsize=9.2, fontweight='bold', color='#0F766E', zorder=15)

    # Lookahead Distance dimension label
    ax_col2.text(boat_x + 0.95, boat_y + 1.25, r"전방주시 거리 $L_{fw}$", fontsize=8.8, fontweight='bold', color='#0D9488',
                 rotation=36, zorder=12)

    # Heading reference ray & Alpha Angle
    ray_len = 2.3
    ax_col2.plot([boat_x, boat_x + ray_len * np.cos(heading_rad)],
                 [boat_y, boat_y + ray_len * np.sin(heading_rad)],
                 color='#DC2626', lw=1.5, ls='-.', zorder=8)
    ax_col2.text(boat_x + 1.15, boat_y + 0.18, "헤딩선", fontsize=8.5, fontweight='bold', color='#DC2626', zorder=9)

    # Steering error angle alpha arc
    alpha_arc = Arc((boat_x, boat_y), 1.9, 1.9, angle=0, theta1=heading_deg, theta2=40,
                    color='#D97706', lw=2.0, zorder=12)
    ax_col2.add_patch(alpha_arc)
    ax_col2.text(boat_x + 1.15, boat_y + 0.52, r"$\alpha$ (조향 오차각)", fontsize=9.0, fontweight='bold', color='#B45309', zorder=13)

    # Pure pursuit following arc (Curvature kappa) placed cleanly away from Pt
    t_arc = np.linspace(0, 1, 40)
    arc_x = boat_x + (pt_x - boat_x)*t_arc - 0.35 * np.sin(t_arc * np.pi)
    arc_y = boat_y + (pt_y - boat_y)*t_arc + 0.45 * np.sin(t_arc * np.pi)
    ax_col2.plot(arc_x, arc_y, color='#10B981', lw=2.8, ls='--', zorder=11)
    ax_col2.text(2.3, 7.35, r"추종 호 궤적 ($\kappa = \frac{2\sin\alpha}{L_{fw}}$)",
                 fontsize=8.8, fontweight='bold', color='#047857',
                 bbox=dict(boxstyle='round,pad=0.15', fc='#ECFDF5', ec='#10B981', lw=0.9), zorder=13)

    # Differential Thruster Status Box on Right Side (Formatted as clean 1-line rows)
    ax_col2.add_patch(FancyBboxPatch((6.8, 4.6), 2.9, 2.7, boxstyle="round,pad=0.05",
                                     fc='#F0F9FF', ec='#0284C7', lw=1.2, zorder=10))
    ax_col2.text(8.25, 7.00, "좌우 차등 추진 분배 제어", ha='center', fontsize=10.2, fontweight='bold', color='#0369A1', zorder=11)
    
    ax_col2.text(7.00, 6.50, "• 좌현(TL): 1680 µs (48.5 N)", fontsize=8.8, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=12)
    ax_col2.text(7.00, 6.05, "• 우현(TR): 1420 µs (21.3 N)", fontsize=8.8, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=12)
    ax_col2.text(7.00, 5.60, "• 추진 편차: ΔT = 27.2 N", fontsize=8.8, fontweight='bold', color='#0284C7', zorder=12)
    ax_col2.text(7.00, 5.15, "• 선회 토크: Mz = +8.7 N·m", fontsize=8.8, fontweight='bold', color='#0369A1', zorder=12)
    ax_col2.text(7.00, 4.75, "  [조타타 없는 민첩 차등 선회]", fontsize=8.3, fontweight='bold', color='#0D9488', zorder=12)

    # Middle Technical Spec Box (Y: 2.70 ~ 4.40)
    ax_col2.add_patch(FancyBboxPatch((0.3, 2.70), 9.4, 1.70, boxstyle="round,pad=0.04",
                                     fc='#F0F9FF', ec='#BAE6FD', lw=1.2, zorder=10))
    ax_col2.text(0.55, 4.15, "순수추종 기하학 제어식 및 차등 추진 분배 알고리즘", fontsize=11.2, fontweight='bold', color='#0369A1', zorder=11)
    specs4 = [
        ("• 목표 곡률 및 조타각", r"$\kappa = \frac{2 \sin \alpha}{L_{fw}}, \quad \delta_{\mathrm{cmd}} = \arctan(\kappa \cdot L_{\mathrm{ship}})$"),
        ("• 가변 전방주시 거리", r"$L_{fw} = \max(L_{\min}, k_v \cdot V_x) \quad (L_{\min}=2.0\mathrm{m}, k_v=1.2\mathrm{s})$ : 속도 연동 안정성"),
        ("• 차등 추진 분배", r"$\tau_{\mathrm{yaw}} = K_p \alpha + K_d \dot{\alpha}, \quad T_{L,R} = \frac{T_{\mathrm{base}}}{2} \pm \frac{\tau_{\mathrm{yaw}}}{B_{\mathrm{prop}}}$ (조타타 없이 민첩 선회)")
    ]
    for idx, (shead, sbody) in enumerate(specs4):
        sy = 3.75 - idx * 0.40
        ax_col2.text(0.65, sy, shead, fontsize=9.8, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=11)
        ax_col2.text(3.45, sy, sbody, fontsize=9.4, color=COLOR_TEXT_MAIN, zorder=11)

    # Bottom Core Explanatory Card (Y: 0.30 ~ 2.50)
    ax_col2.add_patch(FancyBboxPatch((0.3, 0.30), 9.4, 2.20, boxstyle="round,pad=0.05",
                                     fc='#F8FAFC', ec='#CBD5E1', lw=1.4, zorder=20))
    ax_col2.text(0.55, 2.18, "핵심 동작 원리 및 엔지니어링 해설 (완결형 문장)", fontsize=12.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=21)
    ax_col2.text(0.55, 1.68, "1. 선박 전방 일정 거리(Lfw=2.9m) 앞의 목표점을 지속적으로 주시하며 호(Arc) 궤적을 그리도록 조타각을 연산합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col2.text(0.55, 1.20, "2. 선속이 빠를 때는 전방주시 거리를 늘려 주행 안정성을 높이고 저속에서는 거리를 좁혀 회두 응답성을 극대화합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    ax_col2.text(0.55, 0.72, "3. 계산된 조타량을 좌우 듀얼 트러스터의 차등 추진력(PWM)으로 직접 분배하여 조타타 없이도 민첩한 선회를 수행합니다.",
                 fontsize=10.0, color=COLOR_TEXT_MAIN, zorder=21)
    plt.savefig(OUT_PNG, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print(f"Sheet 5 successfully generated: {OUT_PNG}")

if __name__ == '__main__':
    generate_sheet5()

