WIDTH = 1840
HEIGHT = 920
SIM_H = 644   # 1840 * 7 // 20 = 644 (기존 1800x630과 100% 동일한 20:7 경기장 종횡비 유지)
DASH_H = HEIGHT - SIM_H  # 276px (기존 900-630=270과 동일한 30% 비율)

# 맵 확장 배율 (기본값: 1배 = 1840px)
# 4배 맵 확장 사용 시 MAP_SCALE = 4 로 변경하면 카메라 추종 및 우측하단 미니맵이 자동 활성화됩니다.
MAP_SCALE = 1
MAP_W = WIDTH * MAP_SCALE

GRID = 4
GRID_W = MAP_W // GRID   # 전체 맵을 커버하는 점유 그리드 가로 크기 (460)
GRID_H = HEIGHT // GRID  # 전체 맵을 커버하는 점유 그리드 세로 크기 (230)