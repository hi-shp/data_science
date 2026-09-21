import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = '/home/soonhong/kaboat/report4'
OUT_PNG = os.path.join(OUTPUT_DIR, 'sheet1_project_overview.png')

IMG_2D = '/home/soonhong/kaboat/report4/review_pool/2d_cockpit/2D_pos9_frame578_x837.png'
IMG_3D_CHASE = '/home/soonhong/kaboat/report4/review_pool/3d_wide_cropped/3D_Chase_pos6_frame318_x491.png'
IMG_3D_DRONE = '/home/soonhong/kaboat/report4/review_pool/3d_wide_cropped/3D_Drone_pos6_frame318_x491.png'

# Color Palette
COLOR_BG = '#F8FAFC'
COLOR_PANEL_BG = '#FFFFFF'
COLOR_PANEL_BORDER = '#CBD5E1'
COLOR_TEXT_TITLE = '#0F172A'
COLOR_TEXT_MAIN = '#1E293B'
COLOR_TEXT_MUTED = '#475569'

def generate_sheet1():
    # 140 : 100 aspect ratio (14.0 in x 10.0 in at 300 DPI -> 4200 x 3000 px)
    fig = plt.figure(figsize=(14.0, 10.0), dpi=300)
    fig.patch.set_facecolor(COLOR_BG)
    
    # -------------------------------------------------------------------------
    # 1. TOP HEADER (Y: 0.915 ~ 0.985, Height = 0.070)
    # -------------------------------------------------------------------------
    ax_top = fig.add_axes([0.025, 0.915, 0.950, 0.068])
    ax_top.set_facecolor('none')
    ax_top.axis('off')
    
    # Category Tag
    ax_top.add_patch(FancyBboxPatch((0.0, 0.20), 0.120, 0.60, boxstyle="round,pad=0.015",
                                   fc='#0F172A', ec='none', zorder=2))
    ax_top.text(0.060, 0.50, "PROJECT 01", ha='center', va='center',
                fontsize=11.5, fontweight='bold', color='#FFFFFF', zorder=3)
    
    # Main Title
    ax_top.text(0.135, 0.50, "프로젝트 개요", ha='left', va='center',
                fontsize=24.0, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=3)
    
    # Subtitle
    ax_top.text(0.305, 0.49, "|   협수로 자율통과를 위한 라이다 기반 갭 네비게이션(Gap Navigation) USV 시스템", 
                ha='left', va='center', fontsize=12.2, fontweight='bold', color=COLOR_TEXT_MUTED, zorder=3)
    
    # Right Meta Badge
    ax_top.add_patch(FancyBboxPatch((0.815, 0.20), 0.185, 0.60, boxstyle="round,pad=0.015",
                                   fc='#F1F5F9', ec='#94A3B8', lw=1.0, zorder=2))
    ax_top.text(0.9075, 0.50, "KABOAT 2026 자율운항", ha='center', va='center',
                fontsize=11.0, fontweight='bold', color='#0F172A', zorder=3)

    # -------------------------------------------------------------------------
    # 2. THREE CORE SUMMARY CARDS (Y: 0.745 ~ 0.898, Height = 0.153)
    # -------------------------------------------------------------------------
    cards_data = [
        {
            "tag": "01. 연구 배경 및 필요성",
            "tag_color": '#0284C7',
            "tag_bg": '#E0F2FE',
            "points": [
                ("복잡 협수로", "다수 부표가 밀집된 협수역 안전 자율통과"),
                ("기존 한계", "고가 센서 비용 및 무거운 SLAM 연산 탑재 한계"),
                ("개발 목표", "저전력 보드(20Hz+) 즉각 반응형 경량 제어 구현")
            ]
        },
        {
            "tag": "02. 핵심 제어: 갭 네비게이션",
            "tag_color": '#0D9488',
            "tag_bg": '#CCFBF1',
            "points": [
                ("공간 추종", "장애물 반사 회피 대신 사이 열린 공간 직접 탐색"),
                ("점군 군집화", "2D 라이다 DBSCAN 실시간 군집화 및 통로 검출"),
                ("궤적 제어", "곡률 연속 3차 베지어 곡선 생성 및 Pure Pursuit 조타")
            ]
        },
        {
            "tag": "03. 자체 시뮬레이터 및 검증",
            "tag_color": '#4F46E5',
            "tag_bg": '#EEF2FF',
            "points": [
                ("동역학 모델", "관성, 항력, 조타 지연 및 센서 노이즈 물리 반영"),
                ("실시간 콕핏", "라이다 스캔, 갭 가중치, 조타 상태 통합 대시보드"),
                ("성능 검증", "10,000회 무작위 시험 기반 99.2% 완주 성공률 입증")
            ]
        }
    ]
    
    card_width = 0.306
    card_gap = 0.016
    left_base = 0.025
    
    for i, cdata in enumerate(cards_data):
        c_left = left_base + i * (card_width + card_gap)
        ax_card = fig.add_axes([c_left, 0.745, card_width, 0.153])
        ax_card.set_facecolor('none')
        ax_card.axis('off')
        
        # Background
        ax_card.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99, boxstyle="round,pad=0.015",
                                         fc=COLOR_PANEL_BG, ec=COLOR_PANEL_BORDER, lw=1.2, zorder=1))
        
        # Card Tag Header
        ax_card.add_patch(FancyBboxPatch((0.03, 0.76), 0.94, 0.20, boxstyle="round,pad=0.01",
                                         fc=cdata["tag_bg"], ec='none', zorder=2))
        ax_card.text(0.06, 0.86, cdata["tag"], ha='left', va='center',
                     fontsize=12.2, fontweight='bold', color=cdata["tag_color"], zorder=3)
        
        # 3 Bullets with fixed tabular alignment
        y_positions = [0.55, 0.33, 0.11]
        for (head, desc), yp in zip(cdata["points"], y_positions):
            ax_card.add_patch(Rectangle((0.04, yp + 0.025), 0.015, 0.075,
                                        fc=cdata["tag_color"], ec='none', zorder=3))
            ax_card.text(0.070, yp + 0.062, f"{head}:", ha='left', va='center',
                         fontsize=11.2, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=4)
            ax_card.text(0.280, yp + 0.062, desc, ha='left', va='center',
                         fontsize=10.5, fontweight='normal', color=COLOR_TEXT_MUTED, zorder=4)

    # -------------------------------------------------------------------------
    # 3. VISUAL DISPLAY AREA (Y: 0.020 ~ 0.725)
    # Left: 2D Full Cockpit View (Frame 578)
    # Right: 3D Chase View (Frame 318) + 3D Drone View (Frame 318)
    # -------------------------------------------------------------------------
    
    # Left: 2D Cockpit Ribbon & Image
    # Exact 2:1 physical aspect ratio: W_norm = 0.560, H_norm = 0.392
    # Physical: (0.560 * 14.0) / (0.392 * 10.0) = 7.84 / 3.92 = 2.0000!
    # Let's scale up to fit available vertical height ~ 0.68:
    # If W = 0.58, H = 0.58 * 14.0 / (2.0 * 10.0) = 0.406
    # Wait, let's make it even bigger:
    # What if Left = 0.585, Height = 0.4095?
    # What if 2D is at the TOP (Width 0.95, Height = 0.44)?
    # Let's use the balanced two-column layout:
    w_2d = 0.575
    h_2d = (w_2d * 14.0) / (2.0 * 10.0) # 0.4025
    y_visual_base = 0.285
    
    # Left Ribbon
    ax_r1 = fig.add_axes([0.025, y_visual_base + h_2d + 0.005, w_2d, 0.030])
    ax_r1.axis('off')
    ax_r1.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.005", fc='#0F172A', ec='none'))
    ax_r1.text(0.02, 0.50, "2D 통합 자율운항 콕핏 인터페이스 (Frame 578: 전역 경로 추종 및 텔레메트리)",
               ha='left', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF')
    ax_r1.text(0.98, 0.50, "[ 120° 라이다  |  베지어 궤적  |  5채널 계측 ]",
               ha='right', va='center', fontsize=9.2, fontweight='bold', color='#38BDF8')

    # Left Image (2D Cockpit Frame 578)
    img_2d = mpimg.imread(IMG_2D)
    ax_img_2d = fig.add_axes([0.025, y_visual_base, w_2d, h_2d])
    ax_img_2d.imshow(img_2d, aspect='auto')
    ax_img_2d.axis('off')
    for spine in ax_img_2d.spines.values():
        spine.set_visible(True); spine.set_color('#334155'); spine.set_linewidth(1.5)

    # Right Column: 3D Images
    # W_right = 0.355
    # H_3d = (0.355 * 14.0) / (2.857 * 10.0) = 0.1739
    w_3d = 0.355
    h_3d = (w_3d * 14.0) / ((1840.0 / 644.0) * 10.0) # 0.1739
    x_3d = 0.620
    
    # 3D Chase (Top-Right)
    y_3d_top = y_visual_base + h_2d - h_3d
    ax_r2 = fig.add_axes([x_3d, y_3d_top + h_3d + 0.005, w_3d, 0.030])
    ax_r2.axis('off')
    ax_r2.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.005", fc='#0F172A', ec='none'))
    ax_r2.text(0.03, 0.50, "3D 실시간 물리 엔진 (Frame 318: 3인칭 추종 뷰)",
               ha='left', va='center', fontsize=10.2, fontweight='bold', color='#FFFFFF')
    ax_r2.text(0.97, 0.50, "[ ModernGL 3D ]", ha='right', va='center', fontsize=9.0, fontweight='bold', color='#38BDF8')

    img_3d_chase = mpimg.imread(IMG_3D_CHASE)
    ax_img_3d1 = fig.add_axes([x_3d, y_3d_top, w_3d, h_3d])
    ax_img_3d1.imshow(img_3d_chase, aspect='auto')
    ax_img_3d1.axis('off')
    for spine in ax_img_3d1.spines.values():
        spine.set_visible(True); spine.set_color('#334155'); spine.set_linewidth(1.5)

    # 3D Drone (Bottom-Right)
    y_3d_bot = y_visual_base
    ax_r3 = fig.add_axes([x_3d, y_3d_bot + h_3d + 0.005, w_3d, 0.030])
    ax_r3.axis('off')
    ax_r3.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.005", fc='#0F172A', ec='none'))
    ax_r3.text(0.03, 0.50, "3D 전술 드론 쿼터뷰 (Frame 318: 전역 조망)",
               ha='left', va='center', fontsize=10.2, fontweight='bold', color='#FFFFFF')
    ax_r3.text(0.97, 0.50, "[ 수역 전역 관측 ]", ha='right', va='center', fontsize=9.0, fontweight='bold', color='#38BDF8')

    img_3d_drone = mpimg.imread(IMG_3D_DRONE)
    ax_img_3d2 = fig.add_axes([x_3d, y_3d_bot, w_3d, h_3d])
    ax_img_3d2.imshow(img_3d_drone, aspect='auto')
    ax_img_3d2.axis('off')
    for spine in ax_img_3d2.spines.values():
        spine.set_visible(True); spine.set_color('#334155'); spine.set_linewidth(1.5)

    # -------------------------------------------------------------------------
    # 4. BOTTOM ARCHITECTURE & WORKFLOW STRIP (Y: 0.020 ~ 0.250, Height = 0.230)
    # -------------------------------------------------------------------------
    ax_bot = fig.add_axes([0.025, 0.020, 0.950, 0.240])
    ax_bot.axis('off')
    ax_bot.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.015",
                                    fc=COLOR_PANEL_BG, ec=COLOR_PANEL_BORDER, lw=1.2, zorder=1))
    
    # Section Title
    ax_bot.text(0.025, 0.88, "자율운항 제어 파이프라인 및 계측 인터페이스 구성",
                fontsize=13.5, fontweight='bold', color=COLOR_TEXT_TITLE, zorder=2)
    ax_bot.text(0.420, 0.88, "|   센서 인지부터 궤적 추종 및 동역학 시뮬레이션까지의 실시간 연동 체계",
                fontsize=11.0, color=COLOR_TEXT_MUTED, zorder=2)
    
    # 4 Architecture Flow Blocks
    flow_steps = [
        ("1단계: 라이다 인지 및 군집화", "• 120° 라이다 점군 실시간 수집\n• DBSCAN 군집화 알고리즘 적용\n• 개별 부표 장애물 경계 추출", '#0284C7', '#E0F2FE'),
        ("2단계: 안전 갭(Gap) 후보 산출", "• 인접 부표 사이 통로 정밀 검출\n• 선체 전폭 및 안전마진(0.5m) 필터링\n• 다중 목적 함수 가중치 최적 평가", '#0D9488', '#CCFBF1'),
        ("3단계: 3차 베지어 궤적 생성", "• 최적 갭 중심점 통과 목표 경로\n• C2 곡률 연속 부드러운 궤적 합성\n• 조타 진동(Chattering) 원천 방지", '#D97706', '#FEF3C7'),
        ("4단계: Pure Pursuit 조타 제어", "• 전방 주시 거리(Lfw) 기반 추종\n• 선박 유체동역학 3-DOF 모델 반영\n• 좌우 추진기 차등 추력(PWM) 제어", '#4F46E5', '#EEF2FF')
    ]
    
    b_w = 0.228
    b_gap = 0.015
    b_base = 0.020
    
    for j, (stitle, sdesc, scolor, sbg) in enumerate(flow_steps):
        bx = b_base + j * (b_w + b_gap)
        ax_bot.add_patch(FancyBboxPatch((bx, 0.09), b_w, 0.69, boxstyle="round,pad=0.01",
                                       fc=sbg, ec=scolor, lw=1.2, zorder=2))
        ax_bot.text(bx + 0.015, 0.65, stitle, fontsize=11.2, fontweight='bold', color=scolor, zorder=3)
        ax_bot.text(bx + 0.015, 0.35, sdesc, fontsize=10.0, color=COLOR_TEXT_MAIN, va='center', linespacing=1.35, zorder=3)

    plt.savefig(OUT_PNG, dpi=300, facecolor=COLOR_BG)
    plt.close()
    print(f"Sheet 1 successfully created: {OUT_PNG}")

if __name__ == '__main__':
    generate_sheet1()
