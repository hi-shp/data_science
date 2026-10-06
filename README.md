# LiDAR 기반 자율운항 보트 시뮬레이션

Gap Navigation · Bezier · 2D/3D Simulation

조선해양공학을 공부하면서 자율운항 제어를 직접 이해해 보고 싶어 만든 프로젝트입니다. 단순한 장애물 회피부터 시작해, 통과할 공간을 찾고 배의 관성을 고려하는 방식까지 하나씩 시험해 봤습니다.

![2D 자율주행 연속 5회 완주](images/readme/main_2d_5runs.gif)

2D — 4x playback, 연속 5회 완주

LiDAR가 보는 장애물과 앞으로 지나갈 경로를 한 화면에서 확인할 수 있습니다. 같은 주행을 3D 추종 카메라로도 볼 수 있습니다.

![3D 자율주행 연속 5회 완주](images/readme/main_3d_5runs.gif)

3D — 4x playback, 연속 5회 완주

## 처음에는 하나씩 피했습니다

처음에는 가까운 장애물이 보이면 반대쪽으로 꺾고, 공간이 생기면 다시 목적지를 향했습니다. 장애물이 연속해서 나타나면 회피 방향이 자주 바뀌었고, 회전하는 동안 다음 부표에 부딪히기도 했습니다.

이 단순한 방식을 Line Tracing 비교 모드로 남겨 두었습니다. 현재는 이전 MAIN의 물리와 제어를 함께 사용하는 호환 모드입니다.

![Line Tracing 연속 3회 완주](images/readme/line_trace_3runs.gif)

Line Tracing — 4x playback, 연속 3회 완주

## 통과할 공간과 앞으로의 움직임을 봅니다

장애물을 하나씩 피하는 대신, 장애물 사이의 공간을 먼저 찾아보면 어떨까 생각했습니다. 처음에는 GAP에 waypoint를 놓고 Bezier 곡선을 만든 뒤, 앞쪽 점을 따라가는 Pure Pursuit로 조종했습니다. 이 구현은 `main_light` 브랜치에 남아 있습니다.

배가 무거워지면서 경로만 잘 그려서는 부족했습니다. 지금의 `main`은 관성과 추력 응답을 반영해 여러 움직임을 미리 계산하고, 선체가 장애물과 벽을 안전하게 지나갈 수 있는 후보를 고릅니다.

```text
LiDAR 관측 → 관측맵과 A* 경로 안내 → 후보 움직임 예측
          → 선체 안전 검사 → 속도·회전 명령 → 좌우 추력
```

![LiDAR 거리 정보와 Gauge](images/readme/lidar_panels.png)

LiDAR 거리 정보로 주변 장애물을 확인합니다. 센서는 주변 360°를 관측하고, 하단 Gauge는 전방 180°를 펼쳐 보여줍니다.

![GAP과 경로의 실제 교차점](images/readme/gap_crossing.png)

화면의 1차·2차 GAP은 현재 예상경로가 지나갈 통로를 설명합니다. waypoint는 장애물 사이 정중앙 대신 경로와 GAP 선분의 교점에 놓입니다. 같은 통로를 지나가는 동안에는 선택한 GAP을 유지합니다.

Bezier와 분홍색 Pure Pursuit 점은 경로를 읽기 쉽게 보여주는 표시입니다. 현재 기본 모드의 실제 조종은 CODEX 계열의 예측 제어가 맡습니다.

![현재 2D 화면](images/readme/main_2d_cockpit.png)

경로, LiDAR, GAP 표시를 켜고 끄면서 선박의 판단과 움직임을 함께 볼 수 있습니다.

## 반복해서 돌려본 결과

개발 초기에는 Line Tracing과 GAP 방식을 같은 시뮬레이션 조건에서 반복 실행했습니다. 아래는 [보고서 2에 저장된 5,000회 요약](report/report2/benchmark_5000_summary.json)입니다.

| 당시 방식 | 시행 | 성공 | 충돌 | Timeout |
| --- | ---: | ---: | ---: | ---: |
| Line Tracing | 5,000 | 4,256 | 726 | 18 |
| Gap Navigation | 5,000 | 4,811 | 189 | 0 |

통과할 공간을 찾는 방식에서 충돌이 줄었습니다. 이 표는 이전 물리 모델의 결과이며, 현재 버전의 평가 기록은 [data/](data/), 개발 과정은 [report/](report/)에 정리했습니다.

현재 리더보드의 고정 시간은 `GAP NAVI AVG 13.2048 s`, `GAP NAVI BEST 9.7339 s`입니다. 이전 CODEX의 실제 fullscreen 3D 1,000회 측정값을 가져와 이름을 바꾼 기록이며, 조건은 [benchmark metadata](leaderboard_benchmarks.json)에 남아 있습니다.

## 구현 구성

- `main`: 현재 주 구현. 예측 제어와 GAP/Bezier 화면 표시
- `main_light`: 이전 GAP → Bezier → Pure Pursuit 구현
- `codex`: A*와 예측 제어 비교 버전

```text
main.py / environment.py   실행 루프와 선박 상태
heavy/                     예측 제어와 GAP 표시
ui_renderer.py             2D 화면
engine_3d.py                3D 화면
experiments/ · tests/       평가 도구와 테스트
data/ · report/            실험 기록과 보고서
```

## 실행

Python 3.10 / Ubuntu X11에서 실행했습니다. 3D 화면에는 OpenGL 3.3이 필요합니다.

```bash
git clone --branch main https://github.com/hi-shp/data_science.git
cd data_science
python3 -m pip install numpy pygame numba moderngl
python3 main.py
```

첫 실행에는 Numba 컴파일과 3D 렌더러 준비 시간이 들어갑니다.

| 조작 | 기능 |
| --- | --- |
| Space | 일시정지 / 재생 |
| V / C | 2D·3D 전환 / 카메라 전환 |
| M / WASD | RC 모드 전환 / 수동 조종 |
| 하단 버튼 | 배속·Line Tracing·표시 설정 |
| F11 / ESC | 전체화면 / 종료 |

```bash
python3 -m unittest discover -s tests -t .
```

## 아직 남아 있는 점

현재는 시뮬레이터가 선박의 위치와 자세를 알려줍니다. 실제 센서와 선박에 적용하려면 상태 추정과 물리 모델 보정이 필요합니다. 장애물·센서 모델도 단순해서 복잡한 배치나 관측 오차를 더 시험해 보고 싶습니다.

완주시간은 화면 모드와 컴퓨터 성능의 영향을 받습니다. 일부 HUD 속도 표기도 이전 픽셀 환산이 남아 있어 정리가 필요합니다.

[영상 촬영 조건](images/readme/capture_metadata.json) · [과거 보고서](report/)
