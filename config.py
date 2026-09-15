WIDTH = 1800
HEIGHT = 900

# 맵 확장 배율 (기본값: 1배 = 1800px, 4dae4d1 표준)
# 4배 맵 확장 사용 시 MAP_SCALE = 4 로 변경하면 카메라 추종 및 우측하단 미니맵이 자동 활성화됩니다.
MAP_SCALE = 1
MAP_W = WIDTH * MAP_SCALE

GRID = 4
GRID_W = MAP_W // GRID   # 전체 맵을 커버하는 점유 그리드 가로 크기
GRID_H = HEIGHT // GRID