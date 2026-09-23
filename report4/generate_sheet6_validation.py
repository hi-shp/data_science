#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KABOAT Exhibition Sheet 6 Generator: [성능 비교 및 검증]
Visualizing 6 key analytical figures from Report 2 benchmark (5,000 paired runs)
Canvas: 140 : 100 Aspect Ratio (14.0 x 10.0 inches at 300 DPI -> 4200 x 3000 px)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

# Matplotlib Korean font configuration
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/soonhong/kaboat/report4'
REPORT2_DIR = '/home/soonhong/kaboat/report2'
OUT_PNG = os.path.join(OUTPUT_DIR, 'sheet6_performance_validation.png')

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

CARDS_CONFIG = [
    {
        'col': 0, 'row': 0,
        'title': '1. 종합 지표 및 완주 성공률',
        'img_file': 'fig1_radar_and_key_metrics.png',
        'line1': '동일한 난수 조건에서 5,000회 주행한 결과, 완주 성공률이 85.1%에서 96.2%로 향상되었습니다.',
        'line2': '주행 시간 단축과 조타 안정성을 포함한 주요 지표 전반에서 고른 개선을 확인했습니다.',
        'ribbon_bg': '#E0F2FE', 'ribbon_ec': '#0284C7', 'ribbon_tc': '#0369A1'
    },
    {
        'col': 1, 'row': 0,
        'title': '2. 5,000회 전수 주행 궤적 분포',
        'img_file': 'fig2_trajectory_side_by_side.png',
        'line1': '기존 방식은 장애물을 피해 수조 외곽 벽면으로 밀려나는 경향이 관찰되었습니다.',
        'line2': '제안 방식은 통로를 향해 주행하며 중앙 수로를 비교적 안정적으로 유지했습니다.',
        'ribbon_bg': '#CCFBF1', 'ribbon_ec': '#0D9488', 'ribbon_tc': '#0F766E'
    },
    {
        'col': 0, 'row': 1,
        'title': '3. 2D 공간 궤적 점유 밀도',
        'img_file': 'fig3_trajectory_density_heatmap.png',
        'line1': '외곽 벽면으로의 치우침 없이 중앙 통로를 따라 주행 선로가 조밀하게 형성되었습니다.',
        'line2': '불필요한 우회가 줄어들면서 전체 주행 위치 데이터 수가 약 26% 절감되었습니다.',
        'ribbon_bg': '#EEF2FF', 'ribbon_ec': '#6366F1', 'ribbon_tc': '#4338CA'
    },
    {
        'col': 1, 'row': 1,
        'title': '4. 완주 소요 시간 및 누적 선회각',
        'img_file': 'fig5_distributions_time_and_angle.png',
        'line1': '평균 완주 시간이 약 66.8초에서 48.9초로 단축되었고 편차도 안정화되었습니다.',
        'line2': '지그재그 회피가 줄어들면서 배의 누적 선회 각도도 약 60% 감소했습니다.',
        'ribbon_bg': '#E0F2FE', 'ribbon_ec': '#0284C7', 'ribbon_tc': '#0369A1'
    },
    {
        'col': 0, 'row': 2,
        'title': '5. 조타 지터율 및 실효 순항 속도',
        'img_file': 'fig7_steering_stability_and_jitter.png',
        'line1': '매 프레임 급하게 꺾이지 않아 조타 변화율(지터)이 절반 수준(54%)으로 낮아졌습니다.',
        'line2': '선회 시 발생하는 감속이 줄어들어 평균 순항 속도도 소폭(약 10%) 상승했습니다.',
        'ribbon_bg': '#CCFBF1', 'ribbon_ec': '#0D9488', 'ribbon_tc': '#0F766E'
    },
    {
        'col': 1, 'row': 2,
        'title': '6. 충돌 발생 위치 공간 분포',
        'img_file': 'fig8_collision_hotspots_spatial.png',
        'line1': '기존 방식에서 92건 발생했던 수조 외곽 벽면 충돌이 발생하지 않았습니다.',
        'line2': '남은 충돌 사례들은 부표 사이 간격이 좁은 일부 구간에서 관찰되었습니다.',
        'ribbon_bg': '#EEF2FF', 'ribbon_ec': '#6366F1', 'ribbon_tc': '#4338CA'
    }
]

def generate_sheet6():
    fig_w, fig_h = 14.0, 10.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300, facecolor=COLOR_BG)

    # Master Background Axis
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, fig_w)
    ax_bg.set_ylim(0, fig_h)
    ax_bg.axis('off')

    # Top Master Header Banner
    banner_x, banner_y, banner_w, banner_h = 0.35, 9.15, 13.30, 0.68
    ax_bg.add_patch(FancyBboxPatch((banner_x, banner_y), banner_w, banner_h,
                                   boxstyle="round,pad=0.03", fc=COLOR_CARD_BG, ec=COLOR_BORDER_STRONG, lw=1.5, zorder=2))
    
    ax_bg.text(0.60, 9.54, "시트 6: [성능 비교 및 검증] 5,000회 전수 시뮬레이션을 통한 주행 안정성 검증",
               fontsize=16.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=3)
    ax_bg.text(0.60, 9.27, "라인트레이싱(반발 제어)과 갭네비게이션(경로 계획)의 10,000회 대조 실험 결과 분석",
               fontsize=10.5, color=COLOR_TEXT_MUTED, zorder=3)

    # Grid Geometry
    card_w = 6.48
    card_h = 2.70
    col_xs = [0.35, 7.17]
    row_ys = [6.25, 3.30, 0.35]  # row 0 (top), row 1 (mid), row 2 (bot)

    for cfg in CARDS_CONFIG:
        c_x = col_xs[cfg['col']]
        c_y = row_ys[cfg['row']]

        # Card Outer Frame
        ax_bg.add_patch(FancyBboxPatch((c_x, c_y), card_w, card_h,
                                       boxstyle="round,pad=0.04", fc=COLOR_CARD_BG, ec=COLOR_BORDER, lw=1.4, zorder=2))

        # Card Title Ribbon
        ribbon_h = 0.30
        ribbon_y = c_y + card_h - ribbon_h - 0.08
        ax_bg.add_patch(FancyBboxPatch((c_x + 0.12, ribbon_y), card_w - 0.24, ribbon_h,
                                       boxstyle="round,pad=0.02", fc=cfg['ribbon_bg'], ec=cfg['ribbon_ec'], lw=1.1, zorder=3))
        ax_bg.text(c_x + 0.25, ribbon_y + 0.08, cfg['title'],
                   fontsize=11.2, fontweight='bold', color=cfg['ribbon_tc'], zorder=4)

        # Image Placement with clean dark viewport framing
        img_path = os.path.join(REPORT2_DIR, cfg['img_file'])
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Sub-axes for image viewport
            img_x_in = c_x + 0.18
            img_y_in = c_y + 0.64
            img_w_in = card_w - 0.36
            img_h_in = 1.58
            
            # Viewport container background (dark navy matching image canvas)
            ax_bg.add_patch(FancyBboxPatch((img_x_in - 0.04, img_y_in - 0.03), img_w_in + 0.08, img_h_in + 0.06,
                                           boxstyle="round,pad=0.02", fc='#0B0F19', ec='#1E293B', lw=1.2, zorder=3))

            ax_img = fig.add_axes([img_x_in / fig_w, img_y_in / fig_h, img_w_in / fig_w, img_h_in / fig_h], zorder=4)
            ax_img.set_facecolor('#0B0F19')
            ax_img.imshow(img)
            ax_img.axis('off')

        # Bottom Caption Card (2 complete declarative sentences with high readability)
        cap_h = 0.52
        cap_y = c_y + 0.06
        ax_bg.add_patch(FancyBboxPatch((c_x + 0.14, cap_y), card_w - 0.28, cap_h,
                                       boxstyle="round,pad=0.03", fc='#F8FAFC', ec='#CBD5E1', lw=1.1, zorder=3))
        
        ax_bg.text(c_x + 0.26, cap_y + 0.30, cfg['line1'],
                   fontsize=10.4, fontweight='bold', color=COLOR_TEXT_MAIN, zorder=5)
        ax_bg.text(c_x + 0.26, cap_y + 0.10, cfg['line2'],
                   fontsize=10.0, color='#334155', zorder=5)

    plt.savefig(OUT_PNG, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print(f"Sheet 6 successfully generated: {OUT_PNG}")

if __name__ == '__main__':
    generate_sheet6()
