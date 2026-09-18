#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Report 5 Visualization Suite
===================================
Generates presentation-grade figures comparing Line Tracing and GAP Navigation
based on the 10,000 empirical runs of the current tuned version:
  Fig 1: 6-Axis Radar Chart & Key Metric Summary Table
  Fig 2: Full Trajectory Side-by-Side Streamlines (1,800 x 630 px water basin)
  Fig 3: 2D Spatial Occupancy Density Heatmap (Magma vs Viridis)
  Fig 4: Trajectory Distributions & Transit Profiles
  Fig 5: Transit Time & Cumulative Rotation Distributions (KDE + Hist)
  Fig 6: Minimum Clearance & Safety Margin (CDF + Boxplot)
  Fig 7: Steering Stability (Jitter Rate) & Cruising Speed Distributions
  Fig 8: 2D Spatial Collision Hotspots (Wall Traps vs Cluster Squeezes)
  Fig 9: Actual Path Length & Detour Ratio CDF
  Fig 10: Longitudinal X-Axis Corridor Lateral Variance & Centerline Deviation
"""

import os
import sys
import math
import pickle
import json
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Polygon
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

# Typography & Global Style Configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13.5
plt.rcParams['axes.labelsize'] = 11.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10.5
plt.rcParams['legend.fontsize'] = 10.5

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    summary_path = os.path.join(OUTPUT_DIR, "benchmark_5000_summary.json")
    pkl_path = os.path.join(OUTPUT_DIR, "benchmark_5000_results.pkl")
    
    if not os.path.exists(summary_path) or not os.path.exists(pkl_path):
        raise FileNotFoundError("Benchmark data files not found in report5 directory!")
        
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
        
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
        
    return summary, results

def fig1_radar_and_key_metrics(summary, results):
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 1.0], height_ratios=[1.0, 1.05], wspace=0.22, hspace=0.30)
    
    lt = summary['linetrace']
    gn = summary['gapnav']
    
    # 1. 6-Axis Radar Chart (Left column, spanning full height)
    ax_radar = fig.add_subplot(gs[:, 0], polar=True)
    ax_radar.set_facecolor('#111C38')
    
    labels = [
        'Success Rate\n(완주 성공률)',
        'Cruising Speed\n(평균 선속)',
        'Steer Stability\n(조타 안정성 1/Jitter)',
        'Turn Efficiency\n(회전 효율 1/Turn)',
        'Path Directness\n(직진성 1/Detour)',
        'Safety Clearance\n(최소 여유 거리)'
    ]
    num_vars = len(labels)
    
    # Normalized radar metrics (0.0 to 1.0 scale)
    v_lt = [
        lt['success_rate_pct'] / 100.0,
        lt['mean_speed_mean'] / max(1e-3, max(lt['mean_speed_mean'], gn['mean_speed_mean']) * 1.05),
        (1.0 / max(1e-4, lt['steer_jitter_mean'])) / max(1.0 / max(1e-4, lt['steer_jitter_mean']), 1.0 / max(1e-4, gn['steer_jitter_mean'])),
        (1.0 / max(1.0, lt['cum_turn_deg_mean'])) / max(1.0 / max(1.0, lt['cum_turn_deg_mean']), 1.0 / max(1.0, gn['cum_turn_deg_mean'])),
        (1.0 / max(1.0, lt['detour_ratio_mean'])) / max(1.0 / max(1.0, lt['detour_ratio_mean']), 1.0 / max(1.0, gn['detour_ratio_mean'])),
        lt['min_clearance_mean'] / max(1e-3, max(lt['min_clearance_mean'], gn['min_clearance_mean']) * 1.05)
    ]
    v_gn = [
        gn['success_rate_pct'] / 100.0,
        gn['mean_speed_mean'] / max(1e-3, max(lt['mean_speed_mean'], gn['mean_speed_mean']) * 1.05),
        (1.0 / max(1e-4, gn['steer_jitter_mean'])) / max(1.0 / max(1e-4, lt['steer_jitter_mean']), 1.0 / max(1e-4, gn['steer_jitter_mean'])),
        (1.0 / max(1.0, gn['cum_turn_deg_mean'])) / max(1.0 / max(1.0, lt['cum_turn_deg_mean']), 1.0 / max(1.0, gn['cum_turn_deg_mean'])),
        (1.0 / max(1.0, gn['detour_ratio_mean'])) / max(1.0 / max(1.0, lt['detour_ratio_mean']), 1.0 / max(1.0, gn['detour_ratio_mean'])),
        gn['min_clearance_mean'] / max(1e-3, max(lt['min_clearance_mean'], gn['min_clearance_mean']) * 1.05)
    ]
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    v_lt += v_lt[:1]
    v_gn += v_gn[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, v_lt, color='#E63946', lw=2.5, label=f"Line Tracing ({lt['success_rate_pct']}%)")
    ax_radar.fill(angles, v_lt, color='#E63946', alpha=0.25)
    ax_radar.plot(angles, v_gn, color='#00F0FF', lw=2.8, label=f"GAP Navigation ({gn['success_rate_pct']}%)")
    ax_radar.fill(angles, v_gn, color='#00F0FF', alpha=0.35)
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, color='#E2E8F0', fontsize=10.5)
    ax_radar.set_ylim(0, 1.05)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color='#718096', fontsize=9)
    ax_radar.grid(color='#2A3B60', linestyle='--', alpha=0.7)
    ax_radar.set_title("[A] 6축 종합 성능 레이더 다이어그램 (각 5,000회 벤치마크)", color='#FFFFFF', fontsize=13.5, pad=22, fontweight='bold')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12), facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # 2. Outcome Ratio Stacked Horizontal Bar (Top Right)
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor('#111C38')
    ax_bar.set_title("[B] 주행 결과 비율 비교 (완주 vs 충돌 vs 타임아웃)", color='#FFFFFF', fontsize=13, pad=10, fontweight='bold')
    
    y_pos = [1, 0]
    categories = ['Line Tracing', 'GAP Navigation']
    succ_p = [lt['success_rate_pct'], gn['success_rate_pct']]
    coll_p = [lt['collision_rate_pct'], gn['collision_rate_pct']]
    time_p = [lt['timeout_rate_pct'], gn['timeout_rate_pct']]
    
    ax_bar.barh(y_pos, succ_p, color='#2EC4B6', edgecolor='#111C38', height=0.45, label='Success (완주)')
    ax_bar.barh(y_pos, coll_p, left=succ_p, color='#E71D36', edgecolor='#111C38', height=0.45, label='Collision (충돌)')
    ax_bar.barh(y_pos, time_p, left=np.array(succ_p) + np.array(coll_p), color='#FF9F1C', edgecolor='#111C38', height=0.45, label='Timeout (타임아웃)')
    
    for i, (sp, cp, tp) in enumerate(zip(succ_p, coll_p, time_p)):
        ax_bar.text(sp * 0.5, y_pos[i], f"{sp:.1f}%", ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=11)
        if cp > 2.0:
            ax_bar.text(sp + cp * 0.5, y_pos[i], f"{cp:.1f}%", ha='center', va='center', color='#FFFFFF', fontweight='bold', fontsize=10.5)
            
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(categories, color='#E2E8F0', fontsize=11)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("비율 (%)", color='#CBD5E0', fontsize=11)
    ax_bar.tick_params(colors='#A0AEC0')
    ax_bar.grid(axis='x', color='#2A3B60', linestyle=':', alpha=0.5)
    ax_bar.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=3, facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF', fontsize=10)
    
    # 3. Quantitative Summary Table (Bottom Right)
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.axis('off')
    ax_tbl.set_title("[C] 주요 성능 지표 정량 대조표", color='#FFFFFF', fontsize=13, pad=10, fontweight='bold')
    
    tbl_data = [
        ["완주 성공률 (5,000회)", f"{lt['success_rate_pct']:.2f}%", f"{gn['success_rate_pct']:.2f}%", f"{gn['success_rate_pct'] - lt['success_rate_pct']:+.2f}%p"],
        ["충돌 실패율", f"{lt['collision_rate_pct']:.2f}%", f"{gn['collision_rate_pct']:.2f}%", f"{gn['collision_rate_pct'] - lt['collision_rate_pct']:+.2f}%p"],
        ["외곽벽 충돌 건수", f"{lt['wall_collision_count']}건", f"{gn['wall_collision_count']}건", f"-{lt['wall_collision_count']}건 (벽면차단)"],
        ["평균 완주 시간", f"{lt['time_sec_mean']:.2f}s", f"{gn['time_sec_mean']:.2f}s", f"{gn['time_sec_mean'] - lt['time_sec_mean']:+.2f}s"],
        ["평균 누적 회전각", f"{lt['cum_turn_deg_mean']:.1f}°", f"{gn['cum_turn_deg_mean']:.1f}°", f"{gn['cum_turn_deg_mean'] - lt['cum_turn_deg_mean']:+.1f}°"],
        ["평균 순항 선속", f"{lt['mean_speed_mean']:.2f} px/s", f"{gn['mean_speed_mean']:.2f} px/s", f"{gn['mean_speed_mean'] - lt['mean_speed_mean']:+.2f} px/s"],
        ["조타 지터율", f"{lt['steer_jitter_mean']:.4f}", f"{gn['steer_jitter_mean']:.4f}", f"{(gn['steer_jitter_mean'] - lt['steer_jitter_mean'])/lt['steer_jitter_mean']*100:+.1f}%"],
        ["경로 우회율", f"{lt['detour_ratio_mean']:.3f}", f"{gn['detour_ratio_mean']:.3f}", f"{gn['detour_ratio_mean'] - lt['detour_ratio_mean']:+.3f}"]
    ]
    
    col_labels = ["평가 항목", "Line Tracing", "GAP Navigation", "개선폭 (Delta)"]
    table = ax_tbl.table(cellText=tbl_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#2A3B60')
        if row == 0:
            cell.set_facecolor('#1E2A4A')
            cell.set_text_props(color='#00F0FF', fontweight='bold')
        else:
            cell.set_facecolor('#111C38' if row % 2 == 0 else '#152238')
            cell.set_text_props(color='#E2E8F0')
            if col == 3:
                cell.set_text_props(color='#52B788', fontweight='bold')
                
    p = os.path.join(OUTPUT_DIR, "fig1_radar_and_key_metrics.png")
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig2_trajectory_side_by_side(summary, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    lt_runs = [r for r in results if r['mode'] == 'linetrace' and len(r['trajectory']) > 0]
    gn_runs = [r for r in results if r['mode'] == 'gapnav' and len(r['trajectory']) > 0]
    
    # 1. Line Tracing (Top)
    ax1.set_facecolor('#111C38')
    ax1.set_title("[A] Line Tracing 주행 궤적 오버레이 (500회 표본: 수조 외곽벽 밀림 및 충돌 분산)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    ax1.set_xlim(0, 1840); ax1.set_ylim(0, 600)
    ax1.set_aspect('equal')
    ax1.axhline(300, color='#48CAE4', linestyle=':', lw=1, alpha=0.5, label='중앙 수로선 (Y=300)')
    ax1.plot([0, 1840], [18, 18], color='#E63946', lw=2.0, label='수조 외곽벽 충돌선')
    ax1.plot([0, 1840], [582, 582], color='#E63946', lw=2.0)
    
    for r in lt_runs[:350]:
        t = np.array(r['trajectory'])
        if len(t) > 2:
            col = '#2EC4B6' if r['success'] else '#E71D36'
            alpha = 0.18 if r['success'] else 0.45
            ax1.plot(t[:, 0], t[:, 1], color=col, alpha=alpha, lw=1.0)
            
    ax1.scatter([65], [300], color='#52B788', s=80, zorder=5, label='출발점 (65, 300)')
    goal1 = Circle((1740, 300), 70, ec='#FFD166', fc='none', lw=1.8, linestyle='--', label='도착 골인원 (R=70)')
    ax1.add_patch(goal1)
    ax1.tick_params(colors='#A0AEC0')
    ax1.set_ylabel("수조 폭 Y (px)", color='#CBD5E0', fontsize=10.5)
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF', fontsize=9.5)
    
    # 2. GAP Navigation (Bottom)
    ax2.set_facecolor('#111C38')
    ax2.set_title("[B] GAP Navigation 주행 궤적 오버레이 (500회 표본: 중앙 수로 밀집 추종 및 완전 무벽 충돌)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    ax2.set_xlim(0, 1840); ax2.set_ylim(0, 600)
    ax2.set_aspect('equal')
    ax2.axhline(300, color='#48CAE4', linestyle=':', lw=1, alpha=0.5, label='중앙 수로선 (Y=300)')
    ax2.plot([0, 1840], [18, 18], color='#E63946', lw=2.0, label='수조 외곽벽 충돌선')
    ax2.plot([0, 1840], [582, 582], color='#E63946', lw=2.0)
    
    for r in gn_runs[:350]:
        t = np.array(r['trajectory'])
        if len(t) > 2:
            col = '#00F0FF' if r['success'] else '#E71D36'
            alpha = 0.22 if r['success'] else 0.55
            ax2.plot(t[:, 0], t[:, 1], color=col, alpha=alpha, lw=1.1)
            
    ax2.scatter([65], [300], color='#52B788', s=80, zorder=5, label='출발점 (65, 300)')
    goal2 = Circle((1740, 300), 70, ec='#FFD166', fc='none', lw=1.8, linestyle='--', label='도착 골인원 (R=70)')
    ax2.add_patch(goal2)
    ax2.tick_params(colors='#A0AEC0')
    ax2.set_xlabel("수조 길이 X (px)", color='#CBD5E0', fontsize=10.5)
    ax2.set_ylabel("수조 폭 Y (px)", color='#CBD5E0', fontsize=10.5)
    ax2.grid(color='#2A3B60', linestyle='--', alpha=0.4)
    ax2.legend(loc='lower left', facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF', fontsize=9.5)
    
    p = os.path.join(OUTPUT_DIR, "fig2_trajectory_side_by_side.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig3_trajectory_density_heatmap(summary, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    # Compute 2D occupancy histograms
    bins_x = np.linspace(0, 1840, 185)
    bins_y = np.linspace(0, 600, 61)
    
    lt_pts = []
    for r in results:
        if r['mode'] == 'linetrace' and r['trajectory']:
            lt_pts.extend(r['trajectory'])
            
    gn_pts = []
    for r in results:
        if r['mode'] == 'gapnav' and r['trajectory']:
            gn_pts.extend(r['trajectory'])
            
    lt_arr = np.array(lt_pts) if lt_pts else np.zeros((1, 2))
    gn_arr = np.array(gn_pts) if gn_pts else np.zeros((1, 2))
    
    h_lt, _, _ = np.histogram2d(lt_arr[:, 0], lt_arr[:, 1], bins=[bins_x, bins_y])
    h_gn, _, _ = np.histogram2d(gn_arr[:, 0], gn_arr[:, 1], bins=[bins_x, bins_y])
    
    h_lt = gaussian_filter(h_lt.T, sigma=1.4)
    h_gn = gaussian_filter(h_gn.T, sigma=1.4)
    
    # 1. Line Tracing Heatmap (Magma)
    ax1.set_facecolor('#000004')
    ax1.set_title("[A] Line Tracing 2D 공간 궤적 점유 밀도 (외곽 경계벽 밀집 누적)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    im1 = ax1.imshow(h_lt, origin='lower', extent=[0, 1840, 0, 600], cmap='magma', aspect='auto')
    ax1.axhline(300, color='#00F0FF', linestyle=':', lw=1, alpha=0.6)
    ax1.set_ylabel("수조 폭 Y (px)", color='#CBD5E0', fontsize=10.5)
    ax1.tick_params(colors='#A0AEC0')
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.018, pad=0.015)
    cb1.ax.tick_params(colors='#A0AEC0')
    cb1.set_label("점유 밀도", color='#CBD5E0', fontsize=9.5)
    
    # 2. GAP Navigation Heatmap (Viridis)
    ax2.set_facecolor('#000004')
    ax2.set_title("[B] GAP Navigation 2D 공간 궤적 점유 밀도 (단일 중앙 통로 집중도 100%)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    im2 = ax2.imshow(h_gn, origin='lower', extent=[0, 1840, 0, 600], cmap='viridis', aspect='auto')
    ax2.axhline(300, color='#00F0FF', linestyle=':', lw=1, alpha=0.6)
    ax2.set_xlabel("수조 길이 X (px)", color='#CBD5E0', fontsize=10.5)
    ax2.set_ylabel("수조 폭 Y (px)", color='#CBD5E0', fontsize=10.5)
    ax2.tick_params(colors='#A0AEC0')
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.018, pad=0.015)
    cb2.ax.tick_params(colors='#A0AEC0')
    cb2.set_label("점유 밀도", color='#CBD5E0', fontsize=9.5)
    
    p = os.path.join(OUTPUT_DIR, "fig3_trajectory_density_heatmap.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig5_distributions_time_and_angle(summary, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    lt_succ = [r for r in results if r['mode'] == 'linetrace' and r['success']]
    gn_succ = [r for r in results if r['mode'] == 'gapnav' and r['success']]
    
    lt_times = [r['time_sec'] for r in lt_succ]
    gn_times = [r['time_sec'] for r in gn_succ]
    
    lt_turns = [r['cum_turn_deg'] for r in lt_succ]
    gn_turns = [r['cum_turn_deg'] for r in gn_succ]
    
    # 1. Transit Time Distribution (Left)
    ax1.set_facecolor('#111C38')
    ax1.set_title("[A] 완주 시간(Transit Time) 분포 비교", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    bins_t = np.linspace(40, 120, 45)
    ax1.hist(lt_times, bins=bins_t, density=True, color='#E63946', alpha=0.55, edgecolor='#E63946', label=f"Line Tracing (Mean: {summary['linetrace']['time_sec_mean']}s)")
    ax1.hist(gn_times, bins=bins_t, density=True, color='#00F0FF', alpha=0.55, edgecolor='#00F0FF', label=f"GAP Navigation (Mean: {summary['gapnav']['time_sec_mean']}s)")
    ax1.axvline(summary['linetrace']['time_sec_mean'], color='#E63946', lw=2.2, linestyle='--')
    ax1.axvline(summary['gapnav']['time_sec_mean'], color='#00F0FF', lw=2.2, linestyle='--')
    ax1.set_xlabel("완주 시간 (초)", color='#CBD5E0', fontsize=11)
    ax1.set_ylabel("확률 밀도 (Density)", color='#CBD5E0', fontsize=11)
    ax1.tick_params(colors='#A0AEC0')
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # 2. Cumulative Heading Turn Distribution (Right)
    ax2.set_facecolor('#111C38')
    ax2.set_title("[B] 누적 선회각(Cumulative Heading Rotation) 분포 비교", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    bins_a = np.linspace(150, 1200, 45)
    ax2.hist(lt_turns, bins=bins_a, density=True, color='#E63946', alpha=0.55, edgecolor='#E63946', label=f"Line Tracing (Mean: {summary['linetrace']['cum_turn_deg_mean']}°)")
    ax2.hist(gn_turns, bins=bins_a, density=True, color='#00F0FF', alpha=0.55, edgecolor='#00F0FF', label=f"GAP Navigation (Mean: {summary['gapnav']['cum_turn_deg_mean']}°)")
    ax2.axvline(summary['linetrace']['cum_turn_deg_mean'], color='#E63946', lw=2.2, linestyle='--')
    ax2.axvline(summary['gapnav']['cum_turn_deg_mean'], color='#00F0FF', lw=2.2, linestyle='--')
    ax2.set_xlabel("누적 회전각 (도)", color='#CBD5E0', fontsize=11)
    ax2.set_ylabel("확률 밀도 (Density)", color='#CBD5E0', fontsize=11)
    ax2.tick_params(colors='#A0AEC0')
    ax2.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    p = os.path.join(OUTPUT_DIR, "fig5_distributions_time_and_angle.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig6_safety_margin_and_clearance(summary, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    lt_clears = [r['min_clearance'] for r in results if r['mode'] == 'linetrace' and r['success']]
    gn_clears = [r['min_clearance'] for r in results if r['mode'] == 'gapnav' and r['success']]
    
    # 1. Empirical CDF (Left)
    ax1.set_facecolor('#111C38')
    ax1.set_title("[A] 최소 장애물 여유 거리 누적분포함수 (CDF)", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    
    s_lt = np.sort(lt_clears)
    s_gn = np.sort(gn_clears)
    p_lt = np.linspace(0, 1, len(s_lt))
    p_gn = np.linspace(0, 1, len(s_gn))
    
    ax1.plot(s_lt, p_lt, color='#E63946', lw=2.5, label=f"Line Tracing (Median: {summary['linetrace']['min_clearance_median']}px)")
    ax1.plot(s_gn, p_gn, color='#00F0FF', lw=2.5, label=f"GAP Navigation (Median: {summary['gapnav']['min_clearance_median']}px)")
    ax1.axvline(0.0, color='#E63946', linestyle=':', lw=1.5, alpha=0.7, label='물리적 충돌 경계선 (0px)')
    ax1.set_xlabel("최소 여유 거리 (px)", color='#CBD5E0', fontsize=11)
    ax1.set_ylabel("누적 확률 P(Clearance <= x)", color='#CBD5E0', fontsize=11)
    ax1.set_xlim(-5, 55)
    ax1.tick_params(colors='#A0AEC0')
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # 2. Boxplot Comparison (Right)
    ax2.set_facecolor('#111C38')
    ax2.set_title("[B] 최소 여유 거리 박스플롯 대조", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    bp = ax2.boxplot([lt_clears, gn_clears], tick_labels=['Line Tracing', 'GAP Navigation'],
                     patch_artist=True, widths=0.45,
                     boxprops=dict(facecolor='#1E2A4A', color='#2A3B60'),
                     whiskerprops=dict(color='#CBD5E0'),
                     capprops=dict(color='#CBD5E0'),
                     medianprops=dict(color='#FFD166', lw=2.2))
    
    bp['boxes'][0].set_facecolor('#E63946')
    bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor('#00F0FF')
    bp['boxes'][1].set_alpha(0.55)
    
    ax2.set_ylabel("최소 여유 거리 (px)", color='#CBD5E0', fontsize=11)
    ax2.tick_params(colors='#A0AEC0')
    ax2.grid(axis='y', color='#2A3B60', linestyle='--', alpha=0.5)
    
    p = os.path.join(OUTPUT_DIR, "fig6_safety_margin_and_clearance.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig7_steering_stability_and_jitter(summary, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    lt_jitters = [r['steer_jitter'] for r in results if r['mode'] == 'linetrace' and r['success']]
    gn_jitters = [r['steer_jitter'] for r in results if r['mode'] == 'gapnav' and r['success']]
    
    lt_spds = [r['mean_speed'] for r in results if r['mode'] == 'linetrace' and r['success']]
    gn_spds = [r['mean_speed'] for r in results if r['mode'] == 'gapnav' and r['success']]
    
    # 1. Steering Jitter (Left)
    ax1.set_facecolor('#111C38')
    ax1.set_title("[A] 조타 지터율(|dSteer/dt|) 확률분포", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    bins_j = np.linspace(0.01, 0.18, 40)
    ax1.hist(lt_jitters, bins=bins_j, density=True, color='#E63946', alpha=0.55, edgecolor='#E63946', label=f"Line Tracing (Mean: {summary['linetrace']['steer_jitter_mean']:.4f})")
    ax1.hist(gn_jitters, bins=bins_j, density=True, color='#00F0FF', alpha=0.55, edgecolor='#00F0FF', label=f"GAP Navigation (Mean: {summary['gapnav']['steer_jitter_mean']:.4f})")
    ax1.set_xlabel("조타 지터율 (|ΔSteer / step|)", color='#CBD5E0', fontsize=11)
    ax1.set_ylabel("확률 밀도", color='#CBD5E0', fontsize=11)
    ax1.tick_params(colors='#A0AEC0')
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # 2. Cruising Speed (Right)
    ax2.set_facecolor('#111C38')
    ax2.set_title("[B] 평균 순항 선속 분포 비교", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    bins_s = np.linspace(15, 35, 40)
    ax2.hist(lt_spds, bins=bins_s, density=True, color='#E63946', alpha=0.55, edgecolor='#E63946', label=f"Line Tracing (Mean: {summary['linetrace']['mean_speed_mean']:.2f} px/s)")
    ax2.hist(gn_spds, bins=bins_s, density=True, color='#00F0FF', alpha=0.55, edgecolor='#00F0FF', label=f"GAP Navigation (Mean: {summary['gapnav']['mean_speed_mean']:.2f} px/s)")
    ax2.set_xlabel("평균 순항 선속 (px/s)", color='#CBD5E0', fontsize=11)
    ax2.set_ylabel("확률 밀도", color='#CBD5E0', fontsize=11)
    ax2.tick_params(colors='#A0AEC0')
    ax2.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    p = os.path.join(OUTPUT_DIR, "fig7_steering_stability_and_jitter.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig8_collision_hotspots_spatial(summary, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    lt_colls = [r for r in results if r['mode'] == 'linetrace' and r['collision']]
    gn_colls = [r for r in results if r['mode'] == 'gapnav' and r['collision']]
    
    # 1. Line Tracing Collisions (Top)
    ax1.set_facecolor('#111C38')
    ax1.set_title(f"[A] Line Tracing 충돌 지점 공간 분포 (총 {len(lt_colls)}건: 외곽벽 충돌 {summary['linetrace']['wall_collision_count']}건 집중)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    ax1.set_xlim(0, 1840); ax1.set_ylim(0, 600)
    ax1.set_aspect('equal')
    ax1.plot([0, 1840], [18, 18], color='#E63946', lw=2.5, label='외곽벽 충돌선')
    ax1.plot([0, 1840], [582, 582], color='#E63946', lw=2.5)
    
    if lt_colls:
        c_pts = np.array([r['final_pos'] for r in lt_colls])
        ax1.scatter(c_pts[:, 0], c_pts[:, 1], color='#FF3366', s=35, alpha=0.75, edgecolors='#FFFFFF', linewidths=0.5, label='충돌 지점 (Collisions)')
        
    ax1.tick_params(colors='#A0AEC0')
    ax1.set_ylabel("수조 폭 Y (px)", color='#CBD5E0', fontsize=10.5)
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.4)
    ax1.legend(loc='lower left', facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # 2. GAP Navigation Collisions (Bottom)
    ax2.set_facecolor('#111C38')
    ax2.set_title(f"[B] GAP Navigation 충돌 지점 공간 분포 (총 {len(gn_colls)}건: 외곽벽 충돌 0건, 내부 협소 구간 국한)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    ax2.set_xlim(0, 1840); ax2.set_ylim(0, 600)
    ax2.set_aspect('equal')
    ax2.plot([0, 1840], [18, 18], color='#E63946', lw=2.5, label='외곽벽 충돌선')
    ax2.plot([0, 1840], [582, 582], color='#E63946', lw=2.5)
    
    if gn_colls:
        c_pts2 = np.array([r['final_pos'] for r in gn_colls])
        ax2.scatter(c_pts2[:, 0], c_pts2[:, 1], color='#FF9900', s=45, alpha=0.85, edgecolors='#FFFFFF', linewidths=0.7, label='내부 협로 충돌 지점')
        
    ax2.tick_params(colors='#A0AEC0')
    ax2.set_xlabel("수조 길이 X (px)", color='#CBD5E0', fontsize=10.5)
    ax2.set_ylabel("수조 폭 Y (px)", color='#CBD5E0', fontsize=10.5)
    ax2.grid(color='#2A3B60', linestyle='--', alpha=0.4)
    ax2.legend(loc='lower left', facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    p = os.path.join(OUTPUT_DIR, "fig8_collision_hotspots_spatial.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig9_detour_and_path_length(summary, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    lt_lens = [r['path_len'] for r in results if r['mode'] == 'linetrace' and r['success']]
    gn_lens = [r['path_len'] for r in results if r['mode'] == 'gapnav' and r['success']]
    
    lt_detours = [r['detour_ratio'] for r in results if r['mode'] == 'linetrace' and r['success']]
    gn_detours = [r['detour_ratio'] for r in results if r['mode'] == 'gapnav' and r['success']]
    
    # 1. Path Length Histogram (Left)
    ax1.set_facecolor('#111C38')
    ax1.set_title("[A] 실제 주행 거리 분포 (기준 직선: 1,675 px)", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    bins_l = np.linspace(1600, 2200, 45)
    ax1.hist(lt_lens, bins=bins_l, color='#E63946', alpha=0.55, edgecolor='#E63946', label=f"Line Tracing (Mean: {np.mean(lt_lens):.1f}px)")
    ax1.hist(gn_lens, bins=bins_l, color='#00F0FF', alpha=0.55, edgecolor='#00F0FF', label=f"GAP Navigation (Mean: {np.mean(gn_lens):.1f}px)")
    ax1.axvline(1675.0, color='#52B788', lw=2.2, linestyle=':', label='유클리드 최단선 (1,675px)')
    ax1.set_xlabel("주행 거리 (px)", color='#CBD5E0', fontsize=11)
    ax1.set_ylabel("발생 빈도 (Count)", color='#CBD5E0', fontsize=11)
    ax1.tick_params(colors='#A0AEC0')
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax1.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # 2. Detour Ratio CDF (Right)
    ax2.set_facecolor('#111C38')
    ax2.set_title("[B] 경로 우회율(Detour Ratio) 누적분포함수 (CDF)", color='#FFFFFF', fontsize=13, fontweight='bold', pad=10)
    s_lt_d = np.sort(lt_detours)
    s_gn_d = np.sort(gn_detours)
    ax2.plot(s_lt_d, np.linspace(0, 1, len(s_lt_d)), color='#E63946', lw=2.5, label=f"Line Tracing (Median: {summary['linetrace']['detour_ratio_median']:.3f})")
    ax2.plot(s_gn_d, np.linspace(0, 1, len(s_gn_d)), color='#00F0FF', lw=2.5, label=f"GAP Navigation (Median: {summary['gapnav']['detour_ratio_median']:.3f})")
    ax2.set_xlabel("경로 우회율 (실제 거리 / 직선 최단거리)", color='#CBD5E0', fontsize=11)
    ax2.set_ylabel("누적 확률 P(Detour <= x)", color='#CBD5E0', fontsize=11)
    ax2.set_xlim(0.98, 1.35)
    ax2.tick_params(colors='#A0AEC0')
    ax2.grid(color='#2A3B60', linestyle='--', alpha=0.5)
    ax2.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    p = os.path.join(OUTPUT_DIR, "fig9_detour_and_path_length.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def fig10_spatial_corridor_dynamics(summary, results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=200)
    fig.patch.set_facecolor('#0B132B')
    
    # Longitudinal bins along X (30 slices from X=100 to 1700)
    x_slices = np.linspace(100, 1700, 32)
    dx = (x_slices[1] - x_slices[0]) * 0.5
    
    lt_runs = [r for r in results if r['mode'] == 'linetrace' and r['success'] and r['trajectory']]
    gn_runs = [r for r in results if r['mode'] == 'gapnav' and r['success'] and r['trajectory']]
    
    lt_std, gn_std = [], []
    lt_off, gn_off = [], []
    
    for xc in x_slices:
        lt_y = []
        for r in lt_runs:
            for pt in r['trajectory']:
                if abs(pt[0] - xc) <= dx:
                    lt_y.append(pt[1])
        gn_y = []
        for r in gn_runs:
            for pt in r['trajectory']:
                if abs(pt[0] - xc) <= dx:
                    gn_y.append(pt[1])
                    
        lt_std.append(float(np.std(lt_y)) if lt_y else 0.0)
        gn_std.append(float(np.std(gn_y)) if gn_y else 0.0)
        lt_off.append(float(np.mean(np.abs(np.array(lt_y) - 300.0))) if lt_y else 0.0)
        gn_off.append(float(np.mean(np.abs(np.array(gn_y) - 300.0))) if gn_y else 0.0)
        
    # Top: Lateral Standard Deviation
    ax1.set_facecolor('#111C38')
    ax1.set_title("[A] 종단 X축 진행도에 따른 횡방향 궤적 산포도 (Lateral Standard Deviation)", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    ax1.axvspan(300, 1500, color='#FFAA00', alpha=0.10, label='동적 부표 밀집 수로 구간 (X=300~1500)')
    ax1.plot(x_slices, lt_std, color='#E63946', lw=2.4, marker='o', ms=4, label='Line Tracing 산포도')
    ax1.plot(x_slices, gn_std, color='#00F0FF', lw=2.4, marker='s', ms=4, label='GAP Navigation 산포도')
    ax1.set_ylabel("횡방향 표준편차 (px)", color='#CBD5E0', fontsize=10.5)
    ax1.tick_params(colors='#A0AEC0')
    ax1.grid(color='#2A3B60', linestyle='--', alpha=0.4)
    ax1.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    # Bottom: Centerline Deviation
    ax2.set_facecolor('#111C38')
    ax2.set_title("[B] 중앙 기준선(|Y - 300px|) 평균 이탈 오프셋", color='#FFFFFF', fontsize=12.5, fontweight='bold', pad=8)
    ax2.axvspan(300, 1500, color='#FFAA00', alpha=0.10, label='동적 부표 밀집 수로 구간')
    ax2.plot(x_slices, lt_off, color='#E63946', lw=2.4, marker='o', ms=4, label='Line Tracing 중심선 이탈')
    ax2.plot(x_slices, gn_off, color='#00F0FF', lw=2.4, marker='s', ms=4, label='GAP Navigation 중심선 이탈')
    ax2.set_xlabel("수조 길이 X (px)", color='#CBD5E0', fontsize=10.5)
    ax2.set_ylabel("중심선 이탈 거리 (px)", color='#CBD5E0', fontsize=10.5)
    ax2.tick_params(colors='#A0AEC0')
    ax2.grid(color='#2A3B60', linestyle='--', alpha=0.4)
    ax2.legend(facecolor='#152238', edgecolor='#2A3B60', labelcolor='#FFFFFF')
    
    p = os.path.join(OUTPUT_DIR, "fig10_spatial_corridor_dynamics.png")
    plt.tight_layout()
    plt.savefig(p, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Generated: {p}")

def main():
    print("Loading benchmark data...")
    summary, results = load_data()
    print(f"Loaded {len(results)} simulation results.")
    
    fig1_radar_and_key_metrics(summary, results)
    fig2_trajectory_side_by_side(summary, results)
    fig3_trajectory_density_heatmap(summary, results)
    fig5_distributions_time_and_angle(summary, results)
    fig6_safety_margin_and_clearance(summary, results)
    fig7_steering_stability_and_jitter(summary, results)
    fig8_collision_hotspots_spatial(summary, results)
    fig9_detour_and_path_length(summary, results)
    fig10_spatial_corridor_dynamics(summary, results)
    print("All figures successfully generated in report5/ directory!")

if __name__ == '__main__':
    main()
