#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Exhibition Sheet 1 PowerPoint (.pptx) Generator
Generates a fully editable presentation for Sheet 1:
"라인트레이싱의 장애물 반사 제어로 초기 단일 회피에는 성공했으나, 실제 환경에서의 진동 문제와 확장성 한계에 직면했습니다."

All headers, titles, cards, bullet points, and summary sentences are native
PowerPoint shapes and text boxes, allowing the user to select, edit, and restyle any element.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from scipy.interpolate import splprep, splev

# Matplotlib Korean font configuration
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# -----------------------------------------------------------------------------
# Configuration and Output Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(SCRIPT_DIR, 'pptx_assets')
os.makedirs(ASSET_DIR, exist_ok=True)
PPTX_FILE = os.path.join(SCRIPT_DIR, 'sheet1_background_and_problem.pptx')
MASTER_PNG = os.path.join(SCRIPT_DIR, 'sheet1_background_and_problem.png')

FONT_KOREAN = 'Malgun Gothic'

# Color Palette Constants
COLOR_BG = '#FFFFFF'
COLOR_PANEL = '#F8FAFC'
COLOR_BORDER = '#CBD5E1'
COLOR_BORDER_STRONG = '#94A3B8'
COLOR_TEXT_MAIN = '#0F172A'
COLOR_TEXT_SUB = '#475569'
COLOR_PRIMARY = '#0284C7'
COLOR_WARN = '#F59E0B'
COLOR_DANGER = '#DC2626'
COLOR_DANGER_BG = '#FEF2F2'
COLOR_DANGER_BORDER = '#FCA5A5'

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
    
    ax.annotate('', xy=(x + length*0.65*c, y + length*0.65*s), xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.6), zorder=zorder+2)

# -----------------------------------------------------------------------------
# 1. Export Clean Graphics (Without Title & Summary Cards) for Native PPT Layout
# -----------------------------------------------------------------------------
def export_panel1_graphic():
    fig, ax = plt.subplots(figsize=(8.8, 3.2), dpi=300)
    fig.patch.set_facecolor(COLOR_PANEL)
    ax.set_facecolor(COLOR_PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis('off')
    
    # Left Box: Linetracer Robot
    ax.add_patch(FancyBboxPatch((0.20, 0.10), 4.60, 4.60, boxstyle="round,pad=0.06", 
                               fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax.text(2.50, 4.35, "지상 라인트레이서 로봇의 제어 원리", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    # Track curve
    t_vals = np.linspace(0, 1, 100)
    line_x = 0.5 + 4.0 * t_vals
    line_y = 2.4 + 1.2 * np.sin(np.pi * t_vals)
    ax.plot(line_x, line_y, color='#0F172A', lw=8.0, zorder=2)
    ax.text(0.9, 3.4, "검은색 주행선", fontsize=9.2, fontweight='bold', color='#475569')
    
    # Robot Body
    rx, ry, rhd = 2.4, 2.8, np.deg2rad(22)
    R_mat = np.array([[np.cos(rhd), -np.sin(rhd)], [np.sin(rhd), np.cos(rhd)]])
    r_body = np.array([[-0.55, -0.42], [0.55, -0.42], [0.55, 0.42], [-0.55, 0.42]])
    r_rot = (R_mat @ r_body.T).T + np.array([rx, ry])
    ax.add_patch(Polygon(r_rot, closed=True, fc='#0284C7', ec='#0369A1', lw=1.5, zorder=5))
    
    # Wheels
    for wy_off in [-0.48, 0.48]:
        w_poly = np.array([[-0.25, wy_off-0.08], [0.25, wy_off-0.08], [0.25, wy_off+0.08], [-0.25, wy_off+0.08]])
        w_rot = (R_mat @ w_poly.T).T + np.array([rx, ry])
        ax.add_patch(Polygon(w_rot, closed=True, fc='#0F172A', zorder=6))
        
    # Sensor at front edge
    sensor_pt = np.array([rx + 0.58*np.cos(rhd), ry + 0.58*np.sin(rhd)])
    ax.plot(sensor_pt[0], sensor_pt[1], marker='s', markersize=9, color='#DC2626', zorder=8)
    ax.annotate('광센서 1개\n(선 경계면 감지)', xy=(sensor_pt[0], sensor_pt[1]), xytext=(sensor_pt[0] - 0.45, sensor_pt[1] + 0.80),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.3),
                ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#DC2626', zorder=12)
    
    ax.text(3.30, 3.25, "검은 선 감지 시 → 좌회전", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=COLOR_PRIMARY,
            bbox=dict(boxstyle='round,pad=0.2', fc='#EFF6FF', ec='#93C5FD', lw=0.9), zorder=10)
    ax.text(3.70, 2.05, "흰 바탕 감지 시 → 우회전", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#B45309',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FEF3C7', ec='#FCD34D', lw=0.9), zorder=10)
    
    # Right Box: Boat Adaptation
    ax.add_patch(FancyBboxPatch((5.20, 0.10), 4.60, 4.60, boxstyle="round,pad=0.06", 
                               fc='#FFFFFF', ec=COLOR_BORDER, lw=1.2))
    ax.text(7.50, 4.35, "보트 적용: 라인트레이싱 알고리즘", ha='center', va='center', 
            fontsize=11.5, fontweight='bold', color=COLOR_TEXT_MAIN)
    
    ax.add_patch(Rectangle((5.45, 1.45), 4.10, 2.55, fc='#F0F9FF', ec='#BAE6FD', lw=1.0, zorder=2))
    for wy in [1.8, 2.4, 3.0, 3.6]:
        ax.plot([5.55, 9.45], [wy, wy], color='#E0F2FE', lw=1.2, ls=':', zorder=3)
        
    draw_boat(ax, 6.20, 2.70, 0.0, length=1.05, width=0.52, color=COLOR_PRIMARY)
    
    goal_x, goal_y = 9.20, 2.70
    ax.plot([goal_x, goal_x], [goal_y - 0.25, goal_y + 0.65], color='#15803D', lw=2.2, zorder=8)
    ax.add_patch(Polygon([(goal_x, goal_y + 0.65), (goal_x + 0.35, goal_y + 0.45), (goal_x, goal_y + 0.25)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax.text(goal_x, goal_y + 0.78, "목적지", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#15803D')
    
    ax.plot([6.75, goal_x], [2.70, 2.70], color='#22C55E', lw=1.5, ls='--', zorder=4)
    ax.text(7.10, 2.48, "1순위: 목적지 추종", fontsize=9.0, fontweight='bold', color='#15803D')
    
    buoy_gx, buoy_gy = 7.75, 3.45
    ax.add_patch(Circle((buoy_gx, buoy_gy), 0.22, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=8))
    ax.text(buoy_gx, buoy_gy + 0.32, "장애물 부표", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#9A3412')
    
    ax.plot([6.75, buoy_gx - 0.15], [2.70, buoy_gy - 0.12], color='#DC2626', lw=1.8, ls='--', zorder=5)
    ax.text(6.75, 3.45, "2순위: 센서 감지", fontsize=8.8, fontweight='bold', color='#DC2626', zorder=10,
            bbox=dict(boxstyle='round,pad=0.15', fc='#FFFFFF', ec='#FCA5A5', lw=0.8))
    
    ax.annotate('우현 회피 조타!', xy=(6.20, 2.45), xytext=(6.20, 1.80),
                arrowprops=dict(arrowstyle='->', color='#0284C7', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#0284C7', zorder=10)
    
    out_file = os.path.join(ASSET_DIR, 'panel1_visual.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=COLOR_PANEL)
    plt.close()
    return out_file

def export_panel2_graphic():
    fig, ax = plt.subplots(figsize=(8.8, 3.5), dpi=300)
    fig.patch.set_facecolor(COLOR_PANEL)
    ax.set_facecolor(COLOR_PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    
    ax.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax.set_xlabel("전진 방향 X 좌표 (m)", fontsize=10.5, labelpad=4, color=COLOR_TEXT_SUB)
    ax.set_ylabel("횡방향 Y 좌표 (m)", fontsize=10.5, labelpad=4, color=COLOR_TEXT_SUB)
    ax.tick_params(labelsize=9.5)
    
    gx, gy = 9.3, 2.7
    ax.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.5, zorder=8)
    ax.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.45, gy + 0.55), (gx, gy + 0.3)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#15803D')
    
    ax.plot([0.8, gx], [2.7, gy], color='#94A3B8', lw=1.5, ls=':', zorder=1, label='원래 목표 직진 경로')
    
    buoy_x, buoy_y = 5.2, 3.4
    safe_circle = Circle((buoy_x, buoy_y), 0.90, fc='#FEF3C7', ec=COLOR_WARN, lw=1.4, ls='--', alpha=0.7, zorder=2)
    ax.add_patch(safe_circle)
    ax.add_patch(Circle((buoy_x, buoy_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.8, zorder=8))
    ax.text(buoy_x, buoy_y + 0.35, "단일 부표", ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#9A3412')
    ax.text(buoy_x, buoy_y + 1.05, "안전 거리 (0.5m 설정)", ha='center', va='bottom', fontsize=9.8, color='#B45309')
    
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
    
    ax.plot(path_x, path_y, color=COLOR_PRIMARY, lw=3.2, zorder=4, label='선박 실제 주행 궤적')
    
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
        draw_boat(ax, x_i, y_i, hd_i, length=0.85, width=0.42, color='#0284C7', zorder=10)
        ax.text(lx, ly, blabel, ha='center', va='center', 
                fontsize=9.5, fontweight='bold', color=COLOR_TEXT_MAIN,
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_BORDER, lw=0.8, alpha=0.95), zorder=15)
        
    ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)
    
    out_file = os.path.join(ASSET_DIR, 'panel2_visual.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=COLOR_PANEL)
    plt.close()
    return out_file

def export_panel3_graphic():
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=300)
    fig.patch.set_facecolor(COLOR_PANEL)
    ax.set_facecolor(COLOR_PANEL)
    ax.set_xlim(0, 7.0)
    ax.set_ylim(2.2, 7.5) # Cropped above summary card
    
    ax.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)
        
    ax.set_xlabel("X 좌표 (m)", fontsize=10.0, labelpad=3, color=COLOR_TEXT_SUB)
    ax.set_ylabel("Y 좌표 (m)", fontsize=10.0, labelpad=3, color=COLOR_TEXT_SUB)
    ax.tick_params(labelsize=9.2)
    
    # Outer Concrete Wall
    ax.axhline(7.0, color='#64748B', lw=4.0, zorder=5)
    ax.text(0.4, 7.15, "수조 외곽 콘크리트 벽", fontsize=10.0, fontweight='bold', color='#475569')
    
    b1_x, b1_y = 4.0, 4.0
    b2_x, b2_y = 4.0, 3.2
    margin_r = 0.5
    
    c1 = Circle((b1_x, b1_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    c2 = Circle((b2_x, b2_y), margin_r, fc='#FEE2E2', ec='#EF4444', lw=1.2, ls='--', alpha=0.6, zorder=2)
    ax.add_patch(c1)
    ax.add_patch(c2)
    
    # Overlap Hatch
    ax.fill_between([3.7, 4.0, 4.3], [3.6, 3.7, 3.6], [3.6, 3.5, 3.6], 
                    color='#DC2626', alpha=0.35, hatch='///', zorder=3)
    
    ax.add_patch(Circle((b1_x, b1_y), 0.16, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax.add_patch(Circle((b2_x, b2_y), 0.16, fc=COLOR_WARN, ec='#9A3412', lw=1.5, zorder=8))
    ax.text(b1_x + 0.22, b1_y, "부표 A", ha='left', va='center', fontsize=10.5, fontweight='bold', color='#9A3412')
    ax.text(b2_x + 0.22, b2_y, "부표 B", ha='left', va='center', fontsize=10.5, fontweight='bold', color='#9A3412')
    
    ax.annotate('안전마진 중첩 구간\n(0.2m 겹침 발생)', xy=(3.9, 3.6), xytext=(2.2, 2.7),
                arrowprops=dict(arrowstyle='->', color='#991B1B', lw=1.4),
                ha='center', va='center', fontsize=9.2, fontweight='bold', color='#991B1B',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.9), zorder=15)
    
    ax.annotate('', xy=(4.0, 3.95), xytext=(4.0, 3.25),
                arrowprops=dict(arrowstyle='<->', color='#0F172A', lw=1.5), zorder=10)
    ax.text(4.85, 3.6, "실제 통로\n폭 0.8m", ha='left', va='center', fontsize=9.8, fontweight='bold', color='#0F172A')

    # Margin dimension arrow
    ax.annotate('', xy=(3.5, 4.0), xytext=(4.0, 4.0),
                arrowprops=dict(arrowstyle='<->', color='#DC2626', lw=1.2), zorder=10)
    ax.text(3.75, 4.22, "안전마진 0.5m", ha='center', va='bottom', fontsize=8.8, fontweight='bold', color='#DC2626')
    
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
    ax.plot(px, py, color=COLOR_DANGER, lw=3.0, zorder=6, ls='-', label='회피 이탈 궤적')
    
    ax.plot(4.2, 6.9, marker='X', markersize=15, color='#DC2626', markeredgecolor='#7F1D1D', zorder=20)
    ax.text(4.45, 6.65, "외곽벽 충돌!", fontsize=10.5, fontweight='bold', color='#DC2626')
    
    draw_boat(ax, 1.1, 3.6, 0.0, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax, 2.7, 3.9, np.pi/5, length=0.8, width=0.4, color=COLOR_PRIMARY)
    draw_boat(ax, 3.7, 5.7, np.pi/2.7, length=0.8, width=0.4, color=COLOR_DANGER)
    
    ax.annotate('', xy=(6.5, 3.6), xytext=(4.8, 3.6),
                arrowprops=dict(arrowstyle='->', color='#16A34A', lw=2.2, ls='--'), zorder=4)
    ax.text(5.6, 3.9, "목표 게이트 출구", ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#16A34A')
    
    out_file = os.path.join(ASSET_DIR, 'panel3_visual.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=COLOR_PANEL)
    plt.close()
    return out_file

def export_panel4_graphics():
    # Top Boat Sinusoidal Wave
    fig_top, ax_top = plt.subplots(figsize=(5.6, 1.6), dpi=300)
    fig_top.patch.set_facecolor('#FFFFFF')
    ax_top.set_facecolor('#FFFFFF')
    ax_top.set_xlim(0.5, 9.5)
    ax_top.set_ylim(-3.7, 3.7)
    for spine in ax_top.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.2)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    
    ax_top.text(5.0, 3.35, "선체 거동: 매 프레임 단순 반사 계산으로 인해 직선 주행 중 좌우 요동 발생", 
                ha='center', va='top', fontsize=9.2, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=30)
    ax_top.plot([0.6, 9.4], [0.0, 0.0], color='#94A3B8', lw=1.8, ls=':', zorder=5)

    x_fine = np.linspace(0.8, 9.2, 300)
    A = 0.75
    wavelength = 2.6
    y_fine = A * np.sin(2 * np.pi * (x_fine - 0.8) / wavelength)
    ax_top.plot(x_fine, y_fine, color='#DC2626', lw=1.8, ls='--', alpha=0.7, zorder=6)

    boat_x_samples = np.linspace(1.2, 8.8, 11)
    boat_y_samples = A * np.sin(2 * np.pi * (boat_x_samples - 0.8) / wavelength)
    dy_dx = A * (2 * np.pi / wavelength) * np.cos(2 * np.pi * (boat_x_samples - 0.8) / wavelength)
    boat_headings = np.arctan2(dy_dx, 1.0)

    for i, (bx, by, bhd) in enumerate(zip(boat_x_samples, boat_y_samples, boat_headings)):
        draw_boat(ax_top, bx, by, bhd, length=1.35, width=0.68, color='#38BDF8', ec='#0284C7', alpha=0.55, zorder=10 + i)

    ax_top.text(2.1, 2.30, "좌현 반사 조타", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#DC2626', 
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.8), zorder=35)
    ax_top.text(3.4, -2.30, "우현 반사 조타", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#0284C7', 
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#93C5FD', lw=0.8), zorder=35)
    ax_top.text(5.0, -3.35, "시간순 중첩 관찰: 단순 반사 제어를 반복하여 직선 경로에서도 좌우로 요동치며 전진합니다.", 
                ha='center', va='bottom', fontsize=8.6, fontweight='bold', color='#475569', zorder=35)

    out_file_top = os.path.join(ASSET_DIR, 'panel4_top_visual.png')
    plt.tight_layout()
    plt.savefig(out_file_top, dpi=300, facecolor='#FFFFFF')
    plt.close()

    # Bottom Rudder Curve
    fig_bot, ax_bot = plt.subplots(figsize=(5.6, 1.5), dpi=300)
    fig_bot.patch.set_facecolor(COLOR_PANEL)
    ax_bot.set_facecolor(COLOR_PANEL)
    ax_bot.set_xlim(0, 10)
    ax_bot.set_ylim(-16, 18) # Cropped above summary card
    ax_bot.set_yticks([-10, 0, 10])
    ax_bot.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax_bot.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.2)

    ax_bot.set_xlabel("주행 시간 (s)", fontsize=9.5, labelpad=2, color=COLOR_TEXT_SUB)
    ax_bot.set_ylabel("조타각 (deg)", fontsize=9.5, labelpad=2, color=COLOR_TEXT_SUB)
    ax_bot.tick_params(labelsize=8.8)

    t = np.linspace(0, 10, 500)
    chatter = np.clip(10.0 * np.sin(2 * np.pi * 4.5 * t) + np.random.normal(0, 0.4, len(t)), -11, 11)

    ax_bot.plot(t, chatter, color='#DC2626', lw=1.2, zorder=4)
    ax_bot.axhline(0, color='#64748B', lw=1.0, ls='--', zorder=2)

    ax_bot.annotate('매 제어 주기마다 즉각 반사 조타 (좌우 고주파 진동)', xy=(3.5, 10), xytext=(3.5, 14),
                    ha='center', va='bottom',
                    fontsize=8.8, fontweight='bold', color='#B91C1C',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec=COLOR_DANGER_BORDER, lw=0.8), zorder=20)

    out_file_bot = os.path.join(ASSET_DIR, 'panel4_bot_visual.png')
    plt.tight_layout()
    plt.savefig(out_file_bot, dpi=300, facecolor=COLOR_PANEL)
    plt.close()

    return out_file_top, out_file_bot

def export_panel5_graphic():
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=300)
    fig.patch.set_facecolor(COLOR_PANEL)
    ax.set_facecolor(COLOR_PANEL)
    ax.set_xlim(0, 8.0)
    ax.set_ylim(1.6, 8.0) # Cropped above summary card
    
    ax.grid(True, color='#E2E8F0', ls='--', lw=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER_STRONG)
        spine.set_linewidth(1.3)

    ax.set_xlabel("전진 거리 X (m)", fontsize=10.0, labelpad=3, color=COLOR_TEXT_SUB)
    ax.set_ylabel("횡방향 위치 Y (m)", fontsize=10.0, labelpad=3, color=COLOR_TEXT_SUB)
    ax.tick_params(labelsize=9.2)

    # Top Context Banner
    ax.add_patch(FancyBboxPatch((0.25, 6.70), 7.50, 1.05, boxstyle="round,pad=0.04",
                               fc='#FFFFFF', ec=COLOR_BORDER_STRONG, lw=1.1, zorder=20))
    ax.text(4.0, 7.52, "환경 지도가 없어 주변 장애물 배치를 인식하지 못합니다.", 
            ha='center', va='top', fontsize=9.4, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=21)
    ax.text(4.0, 7.02, "장애물 A를 회피하는 과정에서 인접한 장애물 B와 충돌합니다.",
            ha='center', va='center', fontsize=8.8, color='#475569', zorder=21)

    gx, gy = 7.3, 2.5
    ax.plot([gx, gx], [gy - 0.3, gy + 0.8], color='#15803D', lw=2.2, zorder=8)
    ax.add_patch(Polygon([(gx, gy + 0.8), (gx + 0.40, gy + 0.55), (gx, gy + 0.3)],
                         closed=True, fc='#22C55E', ec='#15803D', lw=1.2, zorder=9))
    ax.text(gx, gy + 0.95, "최종 목적지", ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#15803D')

    ax.plot([0.5, gx], [2.5, gy], color='#94A3B8', lw=1.8, ls=':', zorder=1)
    ax.text(5.8, 1.85, "원래 목표 직진 경로", fontsize=9.0, color='#64748B', fontweight='bold', zorder=2)

    bA_x, bA_y = 3.4, 2.5
    ax.add_patch(Circle((bA_x, bA_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax.add_patch(Circle((bA_x, bA_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
    ax.text(bA_x, bA_y - 0.35, "장애물 A (직진 차단)", ha='center', va='top', fontsize=9.2, fontweight='bold', color='#9A3412')

    bB_x, bB_y = 5.2, 4.4
    ax.add_patch(Circle((bB_x, bB_y), 0.70, fc='#FEE2E2', ec='#EF4444', lw=1.0, ls='--', alpha=0.5, zorder=2))
    ax.add_patch(Circle((bB_x, bB_y), 0.25, fc=COLOR_WARN, ec='#9A3412', lw=1.6, zorder=6))
    ax.text(bB_x + 0.40, bB_y + 0.15, "장애물 B\n(사각지대 위치)", ha='left', va='center', fontsize=9.5, fontweight='bold', color='#9A3412')

    draw_boat(ax, 0.9, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)
    draw_boat(ax, 2.1, 2.5, 0.0, length=0.95, width=0.48, color='#0284C7', zorder=10)

    ax.plot([2.55, bA_x - 0.25], [2.5, 2.5], color='#DC2626', lw=2.2, ls='-', zorder=15)
    ax.annotate('센서 감지 (1.2m)\n즉시 좌현 회피 조타!', xy=(2.7, 2.5), xytext=(1.8, 3.7),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.4),
                fontsize=8.8, fontweight='bold', color='#DC2626',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FFFFFF', ec='#FCA5A5', lw=0.8), zorder=25)

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

    hd3 = np.arctan2(py[55] - py[50], px[55] - px[50])
    draw_boat(ax, 3.6, 3.42, hd3, length=0.95, width=0.48, color='#38BDF8', alpha=0.75, zorder=10)

    hd4 = np.arctan2(py[-1] - py[-5], px[-1] - px[-5])
    draw_boat(ax, 5.0, 4.35, hd4, length=0.95, width=0.48, color='#DC2626', zorder=12)

    ax.plot(5.15, 4.4, marker='X', markersize=18, color='#DC2626', markeredgecolor='#7F1D1D', markeredgewidth=1.8, zorder=30)
    ax.annotate('장애물 A 회피 중 장애물 B와 충돌!\n(인접 장애물 위치 미인식)', 
                xy=(5.15, 4.4), xytext=(3.4, 5.7),
                arrowprops=dict(arrowstyle='->', color='#B91C1C', lw=1.8),
                fontsize=9.4, fontweight='bold', color='#991B1B',
                bbox=dict(boxstyle='round,pad=0.3', fc='#FEF2F2', ec='#DC2626', lw=1.3), zorder=35)

    out_file = os.path.join(ASSET_DIR, 'panel5_visual.png')
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=COLOR_PANEL)
    plt.close()
    return out_file

# -----------------------------------------------------------------------------
# 2. PowerPoint Slide Builder (Native Editable Presentation)
# -----------------------------------------------------------------------------
def build_pptx():
    print("Exporting clean graphics for PowerPoint...")
    p1_img = export_panel1_graphic()
    p2_img = export_panel2_graphic()
    p3_img = export_panel3_graphic()
    p4_top_img, p4_bot_img = export_panel4_graphics()
    p5_img = export_panel5_graphic()
    print("All component graphics ready.")

    prs = Presentation()
    # Aspect Ratio 3:2 matching 18x12 inches master figure
    prs.slide_width = Inches(18.0)
    prs.slide_height = Inches(12.0)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Fully Component-based Editable Master Poster
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)

    # 1. Top Master Headline (Native Editable Text)
    tb_headline = slide1.shapes.add_textbox(Inches(0.45), Inches(0.25), Inches(17.10), Inches(0.65))
    tf_h = tb_headline.text_frame
    tf_h.word_wrap = True
    p_h = tf_h.paragraphs[0]
    p_h.text = "라인트레이싱의 장애물 반사 제어로 초기 단일 회피에는 성공했으나, 실제 환경에서의 진동 문제와 확장성 한계에 직면했습니다."
    p_h.font.name = FONT_KOREAN
    p_h.font.size = Pt(21)
    p_h.font.bold = True
    p_h.font.color.rgb = RGBColor(15, 23, 42)

    # 2. Upper Row - Left Card: Linetracer & Boat Adaptation
    c1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(0.95), Inches(8.35), Inches(4.75))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(248, 250, 252)
    c1.line.color.rgb = RGBColor(203, 213, 225)
    c1.line.width = Pt(1.5)

    tb_c1_title = slide1.shapes.add_textbox(Inches(0.65), Inches(1.05), Inches(7.95), Inches(0.40))
    p_c1_t = tb_c1_title.text_frame.paragraphs[0]
    p_c1_t.text = "센서 반사 제어의 착안: 지상 라인트레이서 모티브와 보트 적용 원리"
    p_c1_t.font.name = FONT_KOREAN
    p_c1_t.font.size = Pt(14)
    p_c1_t.font.bold = True
    p_c1_t.font.color.rgb = RGBColor(3, 105, 161)

    slide1.shapes.add_picture(p1_img, Inches(0.65), Inches(1.50), width=Inches(7.95))

    # Editable descriptions below panel 1
    tb_c1_sub1 = slide1.shapes.add_textbox(Inches(0.65), Inches(4.45), Inches(3.85), Inches(1.15))
    tf_c1_sub1 = tb_c1_sub1.text_frame
    tf_c1_sub1.word_wrap = True
    p1 = tf_c1_sub1.paragraphs[0]
    p1.text = "1. 센서가 검은 선을 감지하면 좌회전합니다."
    p1.font.name = FONT_KOREAN
    p1.font.size = Pt(10)
    p1.font.color.rgb = RGBColor(15, 23, 42)
    p2 = tf_c1_sub1.add_paragraph()
    p2.text = "2. 센서가 흰 바탕을 감지하면 우회전합니다."
    p2.font.name = FONT_KOREAN
    p2.font.size = Pt(10)
    p2.font.color.rgb = RGBColor(15, 23, 42)
    p3 = tf_c1_sub1.add_paragraph()
    p3.text = "단순한 반사 규칙만으로도 곡선 경로를 이탈 없이 추종합니다."
    p3.font.name = FONT_KOREAN
    p3.font.size = Pt(10.5)
    p3.font.bold = True
    p3.font.color.rgb = RGBColor(15, 23, 42)

    tb_c1_sub2 = slide1.shapes.add_textbox(Inches(4.75), Inches(4.45), Inches(3.85), Inches(1.15))
    tf_c1_sub2 = tb_c1_sub2.text_frame
    tf_c1_sub2.word_wrap = True
    p1 = tf_c1_sub2.paragraphs[0]
    p1.text = "1. 평상시에는 목적지를 향해 직진 주행합니다."
    p1.font.name = FONT_KOREAN
    p1.font.size = Pt(10)
    p1.font.color.rgb = RGBColor(15, 23, 42)
    p2 = tf_c1_sub2.add_paragraph()
    p2.text = "2. 장애물이 감지되면 단순 반사 규칙으로 회피합니다."
    p2.font.name = FONT_KOREAN
    p2.font.size = Pt(10)
    p2.font.color.rgb = RGBColor(15, 23, 42)
    p3 = tf_c1_sub2.add_paragraph()
    p3.text = "간단한 반사 코드만으로도 장애물을 피해 목표에 도달합니다."
    p3.font.name = FONT_KOREAN
    p3.font.size = Pt(10.5)
    p3.font.bold = True
    p3.font.color.rgb = RGBColor(15, 23, 42)

    # 3. Upper Row - Right Card: Initial Success Trajectory
    c2 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.20), Inches(0.95), Inches(8.35), Inches(4.75))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(248, 250, 252)
    c2.line.color.rgb = RGBColor(203, 213, 225)
    c2.line.width = Pt(1.5)

    tb_c2_title = slide1.shapes.add_textbox(Inches(9.40), Inches(1.05), Inches(7.95), Inches(0.40))
    p_c2_t = tb_c2_title.text_frame.paragraphs[0]
    p_c2_t.text = "초기 주행: 개방 수역에서 0.5m 안전거리를 유지하며 목표에 도달했습니다."
    p_c2_t.font.name = FONT_KOREAN
    p_c2_t.font.size = Pt(14)
    p_c2_t.font.bold = True
    p_c2_t.font.color.rgb = RGBColor(15, 23, 42)

    slide1.shapes.add_picture(p2_img, Inches(9.40), Inches(1.50), width=Inches(7.95))

    tb_c2_bot = slide1.shapes.add_textbox(Inches(9.40), Inches(5.15), Inches(7.95), Inches(0.40))
    p_c2_b = tb_c2_bot.text_frame.paragraphs[0]
    p_c2_b.text = "단일 장애물 환경에서는 단순 반사 규칙만으로도 목표 지점에 도달했습니다."
    p_c2_b.font.name = FONT_KOREAN
    p_c2_b.font.size = Pt(11.5)
    p_c2_b.font.bold = True
    p_c2_b.font.color.rgb = RGBColor(15, 23, 42)
    p_c2_b.alignment = PP_ALIGN.CENTER

    # 4. Middle Section Divider Header
    tb_mid = slide1.shapes.add_textbox(Inches(0.45), Inches(5.85), Inches(17.10), Inches(0.45))
    p_mid = tb_mid.text_frame.paragraphs[0]
    p_mid.text = "실제 환경에서 직면한 라인트레이싱 알고리즘의 3대 구조적 한계"
    p_mid.font.name = FONT_KOREAN
    p_mid.font.size = Pt(21)
    p_mid.font.bold = True
    p_mid.font.color.rgb = RGBColor(220, 38, 38)

    # 5. Lower Row - Column 1: Limitation 1
    col_w = Inches(5.43)
    c3 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(6.40), col_w, Inches(5.35))
    c3.fill.solid()
    c3.fill.fore_color.rgb = RGBColor(248, 250, 252)
    c3.line.color.rgb = RGBColor(203, 213, 225)
    c3.line.width = Pt(1.5)

    tb_c3_t = slide1.shapes.add_textbox(Inches(0.55), Inches(6.48), Inches(5.23), Inches(0.40))
    p_c3_t = tb_c3_t.text_frame.paragraphs[0]
    p_c3_t.text = "한계 1: 안전마진 중첩으로 게이트 통로가 폐쇄됩니다."
    p_c3_t.font.name = FONT_KOREAN
    p_c3_t.font.size = Pt(12)
    p_c3_t.font.bold = True
    p_c3_t.font.color.rgb = RGBColor(220, 38, 38)

    slide1.shapes.add_picture(p3_img, Inches(0.55), Inches(6.92), width=Inches(5.23))

    # Limitation 1 Summary Box (Native Editable Card)
    box3 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.65), Inches(10.35), Inches(5.03), Inches(1.25))
    box3.fill.solid()
    box3.fill.fore_color.rgb = RGBColor(254, 242, 242)
    box3.line.color.rgb = RGBColor(252, 165, 165)
    box3.line.width = Pt(1.2)
    tf3 = box3.text_frame
    tf3.word_wrap = True
    p = tf3.paragraphs[0]
    p.text = "1. 통로 폭(0.8m)보다 안전마진 합(1.0m)이 더 큽니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)
    p = tf3.add_paragraph()
    p.text = "2. 안전마진이 중첩되어 열린 통로를 벽으로 오판합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)
    p = tf3.add_paragraph()
    p.text = "3. 통로 진입을 회피하다 외곽 수조 벽에 충돌합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)

    # 6. Lower Row - Column 2: Limitation 2
    c4 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.28), Inches(6.40), col_w, Inches(5.35))
    c4.fill.solid()
    c4.fill.fore_color.rgb = RGBColor(248, 250, 252)
    c4.line.color.rgb = RGBColor(203, 213, 225)
    c4.line.width = Pt(1.5)

    tb_c4_t = slide1.shapes.add_textbox(Inches(6.38), Inches(6.48), Inches(5.23), Inches(0.40))
    p_c4_t = tb_c4_t.text_frame.paragraphs[0]
    p_c4_t.text = "한계 2: 매 프레임 단순 반사 계산의 반복으로 선체가 좌우로 진동합니다."
    p_c4_t.font.name = FONT_KOREAN
    p_c4_t.font.size = Pt(11.5)
    p_c4_t.font.bold = True
    p_c4_t.font.color.rgb = RGBColor(220, 38, 38)

    slide1.shapes.add_picture(p4_top_img, Inches(6.38), Inches(6.92), width=Inches(5.23))
    slide1.shapes.add_picture(p4_bot_img, Inches(6.38), Inches(8.75), width=Inches(5.23))

    # Limitation 2 Summary Box (Native Editable Card)
    box4 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.48), Inches(10.35), Inches(5.03), Inches(1.25))
    box4.fill.solid()
    box4.fill.fore_color.rgb = RGBColor(254, 242, 242)
    box4.line.color.rgb = RGBColor(252, 165, 165)
    box4.line.width = Pt(1.2)
    tf4 = box4.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "1. 매 제어 프레임마다 단순 반사 규칙을 반복 계산하여 조타합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)
    p = tf4.add_paragraph()
    p.text = "2. 상태 제어나 완충 없이 즉각 반응하므로 좌우로 심하게 진동합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)
    p = tf4.add_paragraph()
    p.text = "3. 진동으로 인해 조타 구동부가 마모되고 전진 추진 효율이 저하됩니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)

    # 7. Lower Row - Column 3: Limitation 3
    c5 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(12.11), Inches(6.40), col_w, Inches(5.35))
    c5.fill.solid()
    c5.fill.fore_color.rgb = RGBColor(248, 250, 252)
    c5.line.color.rgb = RGBColor(203, 213, 225)
    c5.line.width = Pt(1.5)

    tb_c5_t = slide1.shapes.add_textbox(Inches(12.21), Inches(6.48), Inches(5.23), Inches(0.40))
    p_c5_t = tb_c5_t.text_frame.paragraphs[0]
    p_c5_t.text = "한계 3: 단일 반사 제어로는 다중 장애물 충돌을 방지하지 못합니다."
    p_c5_t.font.name = FONT_KOREAN
    p_c5_t.font.size = Pt(11.5)
    p_c5_t.font.bold = True
    p_c5_t.font.color.rgb = RGBColor(220, 38, 38)

    slide1.shapes.add_picture(p5_img, Inches(12.21), Inches(6.92), width=Inches(5.23))

    # Limitation 3 Summary Box (Native Editable Card)
    box5 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(12.31), Inches(10.35), Inches(5.03), Inches(1.25))
    box5.fill.solid()
    box5.fill.fore_color.rgb = RGBColor(254, 242, 242)
    box5.line.color.rgb = RGBColor(252, 165, 165)
    box5.line.width = Pt(1.2)
    tf5 = box5.text_frame
    tf5.word_wrap = True
    p = tf5.paragraphs[0]
    p.text = "1. 전방 장애물 A를 감지한 직후 좌현으로 회피합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)
    p = tf5.add_paragraph()
    p.text = "2. 인접 장애물 B를 인식하지 못해 충돌 경로로 진입합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)
    p = tf5.add_paragraph()
    p.text = "3. 단순 반사 규칙으로는 다중 장애물 환경에 대응하지 못합니다."
    p.font.name = FONT_KOREAN
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(153, 27, 27)

    prs.save(PPTX_FILE)
    print(f"Editable PowerPoint presentation successfully created: {PPTX_FILE}")

if __name__ == '__main__':
    build_pptx()
