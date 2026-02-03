# [Project] 자율운항보트 경로 계획 및 제어 알고리즘 최적화
**Autonomous Surface Vehicle Navigation: From Reactive Avoidance to Map-Based Path Planning**

본 프로젝트는 단순한 제어 파라미터 최적화의 한계를 데이터 과학적 관점에서 분석하고, 이를 극복하기 위해 **인지(Perception) - 판단(Planning) - 제어(Control)** 프로세스를 완전히 재설계한 자율운항보트 소프트웨어 스택 개발 과정을 담고 있습니다.

---

## 1. 프로젝트 배경 및 문제 정의

### 1.1 초기 접근 방식의 한계 (시시행착오)
처음에는 `avoidance_strength`, `gps_gain`, `gap_gain`이라는 3가지 핵심 파라미터를 조정하며 완주율을 예측하는 기계학습 모델을 구축하려 했습니다. 그러나 실제 시뮬레이션 데이터 생성 과정에서 다음과 같은 결정적 결함을 발견했습니다.

* **연산 효율 저하**: Python의 GIL 문제와 시뮬레이션 환경의 제약으로 인해 단일 완주에 50초 이상 소요되어 대규모 데이터 수집이 불가능했습니다.
* **구조적 비일관성**: 장애물 배치의 무작위성이 시스템 안정성보다 커서, 동일 파라미터 내에서도 완주율 편차가 극심하게 나타났습니다.
* **제어 로직의 원시성**: 단순히 벡터를 가감하는 1차적 회피 로직은 특정 상황에서 좌/우 조향이 번갈아 활성화되는 진동 패턴(Oscillation)을 유발했습니다.

![초기 시뮬레이션 장면](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/early_sim.png)
*시행착오가 담긴 초기 시뮬레이션: 연산 시간이 과도하게 길고 구조적으로 불안정했던 구간*

### 1.2 패러다임의 전환: "학습 가능한 구조 설계"
단순한 '값의 최적화'를 포기하고, **'학습이 가능할 정도로 안정적인 알고리즘 구조'**를 만드는 것으로 방향을 수정했습니다. 즉, 매 프레임 즉흥적인 판단이 아닌 **지도(Map)와 경로(Path)**에 기반한 논리적 주행 체계를 구축했습니다.

---

## 2. 시스템 아키텍처 및 구현 (System Architecture)

### 2.1 실시간 환경 인지 및 격자 지도 (Perception)
* **Coarse Grid Map**: $600 \times 2000$ px의 경기장을 $8 \times 8$ px 단위의 격자로 압축하여 연산 효율을 극대화했습니다.
* **Occupancy & Decay**: 점유도(Occupancy) 업데이트 시 이전 프레임 정보를 Decay를 통해 감쇠시켜 잔상을 제거하고 지도의 맥락을 유지했습니다.

### 2.2 DBSCAN 기반 장애물 클러스터링
불연속적인 라이다 히트 포인트를 의미 있는 객체로 분류하기 위해 `sklearn.cluster.DBSCAN`을 도입했습니다.

![DBSCAN 클러스터링](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/dbscan.png)
*DBSCAN 도입 장면: 라이다 점들을 군집화하여 장애물의 형태와 중심을 정확하게 추정*

### 2.3 스코어링 기반 웨이포인트 생성 (Planning)
장애물 사이의 공간(Gap) 중 최적의 통로를 찾기 위해 다중 목표 최적화 함수를 설계했습니다.
* **Heuristic Scoring**: GPS 목표 방향 일치도, 통로 너비, 안전 여유(Clearance), 주행 관성(Hysteresis)을 가중합하여 웨이포인트 점수를 계산합니다.
* **Hysteresis Logic**: 경로가 급격히 바뀌어 충돌하는 것을 방지하기 위해 기존 경로를 유지하려는 관성을 부여했습니다.

---

## 3. 경로 계획 및 제어 (Control)

보트의 동역학적 특성을 고려한 부드러운 경로 생성을 위해 **2차 베지어 곡선(Quadratic Bézier Curve)**과 **Pure Pursuit** 알고리즘을 결합했습니다.

### 3.1 베지어 곡선 경로 생성
보트의 현재 위치($p_0$), 진행 방향의 예측 위치($p_1$), 타겟 웨이포인트($p_2$)를 제어점으로 하여 부드러운 궤적을 생성합니다.
$$B(t) = (1-t)^2p_0 + 2(1-t)tp_1 + t^2p_2$$

![베지어 및 Pure Pursuit 원리](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/bezier_logic.png)
*Pure Pursuit와 베지어 곡선을 결합하여 경로의 lookahead 지점을 추종하도록 구성*

### 3.2 고도화된 제어 로직
* **Pure Pursuit**: 생성된 곡선 위에서 일정 거리 앞의 점을 실시간 추종하여 조향의 불연속성을 제거했습니다.
* **Hybrid Control**: 삼각함수 기반 steering mix, exponential 감쇠, PID 속도 조절을 통합하여 자연스러운 움직임을 구현했습니다.

---

## 4. 실험 결과 및 성능 분석

알고리즘 구조 전환 후, 시스템의 신뢰성과 효율성이 획기적으로 개선되었습니다.

| 분석 항목 | 기존 알고리즘 (Reactive) | 개선된 알고리즘 (Proposed) |
| :--- | :--- | :--- |
| **평균 완주 시간** | 약 50초 내외 (실패 빈번) | **약 10초 내외 (최대 5배 개선)** |
| **완주 성공률** | 매우 낮음 | **약 96.6% (30회 중 29회 성공)** |
| **주행 안정성** | 좌우 미세 진동 및 급선회 | **매끄러운 베지어 경로 추종** |

![최종 시뮬레이션 구동](https://raw.githubusercontent.com/hi-shp/Data_science/main/images/final_sim.png)
*최종 시뮬레이션 장면: 장애물 지도 위에 웨이포인트를 선정하고 경로를 생성하여 주행하는 모습*

---

## 5. 결론 및 성취

본 프로젝트는 단순한 기능 구현을 넘어, 실제 업계 알고리즘 구조와 유사한 **지도 기반 자율운항 시스템**을 직접 설계하고 검증했습니다.
* **기술적 성취**: `sklearn`의 기계학습 모델(DBSCAN)을 제어 중핵에 적용하고, 벡터화 연산을 통해 실시간성을 확보했습니다.
* **실전성**: 라이다와 GPS라는 최소한의 센서 구성으로도 높은 완성도를 달성했으며, 본 알고리즘은 실제 전국 자율운항보트 경진대회(KABOAT)의 핵심 소프트웨어 스택으로 채택되었습니다.

---

## 6. 설치 및 실행 (Quick Start)

### Prerequisites
* Python 3.8+
* Pygame, NumPy, SciPy, Scikit-learn

### Execution
```bash
python simulation.py
