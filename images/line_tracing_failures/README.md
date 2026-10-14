# Line Tracing 실패 장면 영상

출처: `main_light` 624f4acb6d71의 실제 Line Tracing 실행 화면. 1x와 2x는 각각 UI에서 직접 실행했고 후처리 배속 변환을 하지 않았습니다.
LiDAR hit·범위·최근접 장애물, 1·2차 경로·GAP·웨이포인트 표시를 끄고 녹화했습니다. 발표용 영상은 실제 화면에서 하단 패널만 잘라낸 1840×644 영상이며, 1840×920 원본도 로컬 자산 폴더에 보관했습니다.

| 사례 | seed | min_obs | 1x | 2x | 확인 결과 |
|---|---:|---:|---|---|---|
| 좌우진동 | 2065 | 120 px | [line_tracing_oscillation_seed2065_1x.mp4](oscillation/line_tracing_oscillation_seed2065_1x.mp4) | [line_tracing_oscillation_seed2065_2x.mp4](oscillation/line_tracing_oscillation_seed2065_2x.mp4) | 100초 관찰 동안 목적지 미도달 |
| 좌우진동 | 2092 | 120 px | [line_tracing_oscillation_seed2092_1x.mp4](oscillation/line_tracing_oscillation_seed2092_1x.mp4) | [line_tracing_oscillation_seed2092_2x.mp4](oscillation/line_tracing_oscillation_seed2092_2x.mp4) | 100초 관찰 동안 목적지 미도달 |
| 좌우진동 | 2201 | 120 px | [line_tracing_oscillation_seed2201_1x.mp4](oscillation/line_tracing_oscillation_seed2201_1x.mp4) | [line_tracing_oscillation_seed2201_2x.mp4](oscillation/line_tracing_oscillation_seed2201_2x.mp4) | 전방 부표 접촉 |
| 통로진입실패 | 2063 | 120 px | [line_tracing_passage_refusal_seed2063_1x.mp4](passage_refusal/line_tracing_passage_refusal_seed2063_1x.mp4) | [line_tracing_passage_refusal_seed2063_2x.mp4](passage_refusal/line_tracing_passage_refusal_seed2063_2x.mp4) | 100초 관찰 동안 목적지 미도달 |
| 통로진입실패 | 2100 | 120 px | [line_tracing_passage_refusal_seed2100_1x.mp4](passage_refusal/line_tracing_passage_refusal_seed2100_1x.mp4) | [line_tracing_passage_refusal_seed2100_2x.mp4](passage_refusal/line_tracing_passage_refusal_seed2100_2x.mp4) | 100초 관찰 동안 목적지 미도달 |
| 통로진입실패 | 2244 | 120 px | [line_tracing_passage_refusal_seed2244_1x.mp4](passage_refusal/line_tracing_passage_refusal_seed2244_1x.mp4) | [line_tracing_passage_refusal_seed2244_2x.mp4](passage_refusal/line_tracing_passage_refusal_seed2244_2x.mp4) | 100초 관찰 동안 목적지 미도달 |
| 후미충돌 | 2069 | 100 px | [line_tracing_stern_collision_seed2069_1x_minobs100.mp4](stern_collision/line_tracing_stern_collision_seed2069_1x_minobs100.mp4) | [line_tracing_stern_collision_seed2069_2x_minobs100.mp4](stern_collision/line_tracing_stern_collision_seed2069_2x_minobs100.mp4) | 선미 쪽 부표 접촉 |
| 후미충돌 | 2085 | 100 px | [line_tracing_stern_collision_seed2085_1x_minobs100.mp4](stern_collision/line_tracing_stern_collision_seed2085_1x_minobs100.mp4) | [line_tracing_stern_collision_seed2085_2x_minobs100.mp4](stern_collision/line_tracing_stern_collision_seed2085_2x_minobs100.mp4) | 선미 쪽 부표 접촉 |
| 후미충돌 | 2113 | 100 px | [line_tracing_stern_collision_seed2113_1x_minobs100.mp4](stern_collision/line_tracing_stern_collision_seed2113_1x_minobs100.mp4) | [line_tracing_stern_collision_seed2113_2x_minobs100.mp4](stern_collision/line_tracing_stern_collision_seed2113_2x_minobs100.mp4) | 선미 쪽 부표 접촉 |

좌우진동은 같은 주행 중 조향 방향이 반복적으로 바뀌는 장면입니다. 후미충돌은 충돌 부표 중심의 선박 좌표계 종방향 값이 음수임을 충돌 프레임에서 확인했습니다.
통로진입실패는 길이 보이는 장애물 구간에서 우회와 재접근이 반복되는 현재 Line Tracing의 장면입니다. 과거 보고서의 광선 차폐 방식에 의한 정확한 “게이트 폐쇄” 기전을 재현한 영상으로 설명하지 마세요.
100초는 자료 선정을 위한 관찰 길이입니다. 프로그램의 공식 timeout 판정으로 해석하지 않습니다.
