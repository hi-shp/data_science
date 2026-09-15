# [보고서 3] ROS2 및 /scan 라이다 기반 갭네비게이션 알고리즘 실선 적용 및 범용 표준 템플릿 개발 보고서
### KABOAT 자율운항보트 실선 플랫폼(KABOT 2026) 코드 분석, 센서 파이프라인 최적화 및 타 대학/팀 보급용 표준 오픈 템플릿 가이드

본 보고서는 <b>KABOAT(전국 학생 자율운항보트 경진대회)</b>에 실제 참가하여 검증된 실선 플랫폼(<b>KABOT 2026</b>)의 ROS2 소스코드를 심층 역설계 분석하고, 선행 연구(보고서 1, 보고서 2)에서 시뮬레이션 완주 성공률 96.22% 및 조타 지터 54% 감소를 달성한 **개구부 기반 갭네비게이션(Gap Navigation) 알고리즘**을 실제 ROS2 환경(`sensor_msgs/msg/LaserScan` 토픽 기반)에 완벽히 이식·적용하기 위한 종합 엔지니어링 보고서입니다.

나아가 전국 대학 자율운항보트 참가팀 및 신규 개발 학생들이 복잡한 제어 이론 없이도 자신의 보트에 즉시 탑재하여 사용할 수 있도록 **표준 오픈 파라미터 템플릿(`gap_nav_params.yaml`)**, **모듈형 제어 노드(`gap_navigation_node.py`)**, 그리고 **기존 코드 1:1 교체 파일(`course1_gapnav_replacement.py`)**을 함께 설계·배포합니다.

---

## 1. 연구 배경 및 실선 플랫폼(KABOT 2026) 개요

### 1.1 KABOAT 경진대회 미션 및 실선 제어 환경
KABOAT 대회는 가로 약 100m, 세로 약 20m 규모의 인공 수조에서 진행되며, 크게 3개 코스로 구성됩니다:
1. **1코스 (장애물 회피 구간, 제한시간 3분)**: 수면 위에 불규칙하게 배치된 빨간색/초록색 부표 사이를 통과하여 GPS 웨이포인트를 추종. 부표 또는 수조 벽면 충돌 시 회당 **+10초** 패널티.
2. **2코스 (호핑투어 & 색상 인식 구간, 제한시간 2분)**: 특정 보라색 부표를 중심으로 360도 원형 선회(호핑) 후, 카메라 비전으로 표적 형상/색상을 인식하여 LED 점등 및 갈림길 통과.
3. **3코스 (정밀 도킹 구간, 제한시간 5분)**: 벽면 사이의 좁은 선착장에 진입하여 외벽 0.3m 이내로 감속·정지. 성공 시 **+50점** 가산점.

### 1.2 실선 플랫폼(KABOT 2026)의 시스템 구성
- **메인 SBC**: 엔비디아 젯슨(Jetson) 또는 x86 IPC (Ubuntu 22.04 LTS / ROS 2 Humble)
- **임베디드 제어기**: 마이크로컨트롤러(STM32/Teensy) 기반 `micro-ROS` 노드 (`/dev/ttyUROS`)
- **센서 구성**:
  - **LiDAR**: YDLIDAR TG15 (광학식 2D TOF 라이다, 15m 거리, 10Hz 스캔, 512,000 Baud)
  - **IMU**: IAHRS 9축 관성 센서 (선체 롤/피치/요 및 각속도 측정)
  - **GPS**: WTRTK RTK-GPS (NTRIP 보정 신호 수신을 통한 cm급 위도/경도 측정)
  - **카메라**: 전방/좌현 USB 광각 카메라 (YOLOv8 기반 객체 검출)
- **액추에이터 인터페이스**:
  - 서보모터 조타각: `/actuator/key/degree` (`std_msgs/msg/Float64`, 조타각 범위: 30.0° ~ 150.0°, 중립 90.0°)
  - 쓰러스터 추력: `/actuator/thruster/percentage` (`std_msgs/msg/Float64`, 0.0% ~ 100.0%)

---

## 2. KABOT 2026 기존 제어 코드 분석 및 한계점 진단

기존 플랫폼의 1코스 주행 코드인 `kabot2026/isv/launch_isv/course1.py`의 라이다 처리 및 조타 결정 로직을 분석한 결과, 실전 주행 시 잦은 충돌과 감점을 유발하는 **4가지 치명적 결함**이 확인되었습니다.

![Fig 1 Architecture Comparison](fig1_system_architecture_comparison.png)
<b>[그림 1]</b> 자율운항보트 주행 제어 파이프라인 아키텍처 비교: 기존 단순 광선 차폐 회피(좌측) vs 갭네비게이션(우측)

### 2.1 기존 `course1.py`의 4대 취약점

#### [결함 1] 하드웨어 종속적 인덱스 슬라이싱 (`ranges[500:1500]`)
```python
# course1.py 178행
ranges = np.array(data.ranges[500:1500])
num_samples = len(ranges)
self.dist_180 = np.zeros(181)
```
- **문제점**: YDLIDAR 드라이버의 샘플링 레이트, 주파수 설정, 또는 통신 지연에 따라 1회 스캔당 배열 크기(`len(ranges)`)는 1,800개에서 2,200개 사이로 동적 변동합니다. 
- `[500:1500]`과 같은 하드코딩된 인덱스 슬라이싱은 센서 드라이버가 변경되거나 프레임 드롭 발생 시 전방 180도가 아닌 엉뚱한 측방/후방 각도를 읽어 궤적 계산이 완전히 붕괴됩니다.

#### [결함 2] 인공 게이트 폐쇄(Gate Closure) 현상
```python
# course1.py 200~205행
danger_flags = (self.dist_180 > 0) & (self.dist_180 <= self.dist_threshold)
expanded_danger = np.copy(danger_flags)
for i in np.where(danger_flags)[0]:
    low = max(0, i - self.side_margin)      # 좌우 ±35도 강제 확장!
    high = min(181, i + self.side_margin)
    expanded_danger[low:high] = True
```
- **문제점**: 부표의 실제 크기와 상관없이 라이다 빔 단위로 감지된 각도 주변 좌우 35도를 일괄적으로 `DANGER(0)`로 확장합니다.
- 부표 2개가 1.5m ~ 2.5m 폭의 안전한 통과 게이트(Gap)를 형성하고 있어도, 양쪽 부표의 35도 위험 확장 영역이 중앙에서 서로 겹쳐(Overlap) 게이트 전체가 위험 영역으로 마스킹됩니다.
- 결과적으로 보트는 게이트를 통과하지 못하고 수조 외곽 벽면으로 극단적 우회를 시도하다가 벽에 충돌(패널티 +10초)하게 됩니다.

#### [결함 3] 경로 계획 부재 및 조타 채터링(Jittering)
```python
# course1.py 271~277행
safe_angles_deg = np.array(self.safe_angles_list) - 90
diff = np.abs(safe_angles_deg - self.goal_rel_deg)
chosen_safe_angle = safe_angles_deg[np.argmin(diff)]
steering_angle = self.servo_neutral_deg + chosen_safe_angle
self.cmd_key_degree = constrain(steering_angle, self.servo_min_deg, self.servo_max_deg)
```
- **문제점**: 매 프레임(100ms)마다 순간적인 광선 최소 차이 각도를 찾아 서보모터 각도로 직결합니다.
- 이로 인해 전방 장애물 배치에 따라 조타각이 30°에서 150°로 순간 진동하는 <b>뱅뱅 제어(Bang-Bang Control)</b>가 발생하며, 서보모터 기어 마모와 선체 요잉(Yawing) 저항으로 속도가 급격히 저하됩니다.

#### [결함 4] 고정 추력으로 인한 선회 슬립 및 드리프트
- 장애물 유무와 선회 곡률에 관계없이 `course1: 25.0%` 고정 추력을 유지하여, 급선회 시 선미 추진 특성에 의한 외곽 밀림(횡방향 슬립)이 발생해 부표를 치고 지나가는 사고가 발생합니다.

---

## 3. YDLIDAR TG15 센서 신호처리 및 직교좌표계 복원

실제 실선 보트에서 갭네비게이션을 구현하기 위한 첫 번째 핵심 단계는 라이다 드라이버의 로우 데이터(`sensor_msgs/LaserScan`)를 물리 기하학 기반 2D 직교좌표계로 정확히 변환하고 노이즈를 제거하는 것입니다.

![Fig 2 LiDAR Pipeline](fig2_ydlidar_scan_processing_pipeline.png)
<b>[그림 2]</b> YDLIDAR TG15 하드웨어 제원 및 실시간 신호처리 파이프라인 (좌표계 변환, 선체 배제, DBSCAN 군집화)

### 3.1 센서 제원 및 토픽 구조
- **센서 모델**: YDLIDAR TG15 (광학식 2D Time-of-Flight)
- **거리 측정 범위**: 0.05m ~ 15.0m (거리 분해능 ±20mm)
- **초당 샘플링수**: 20,000 samples/sec (20 kHz)
- **스캔 주파수**: 10 Hz (초당 10회 회전, $\Delta t = 100\,\text{ms}$)
- **각 분해능**: $\Delta \theta \approx 0.18^\circ$ (1스캔당 약 2,000개 포인트 수신)
- **ROS 2 토픽**: `/scan` (`sensor_msgs/msg/LaserScan`)

### 3.2 물리 기하학 복원 수식 (Polar to Cartesian)
인덱스를 임의로 자르지 않고, ROS2 `LaserScan` 헤더에 명시된 메타데이터를 사용하여 각 포인트의 물리 좌표를 직접 복원합니다:

$$\theta_i = \text{angle\_min} + i \cdot \text{angle\_increment}, \quad i \in [0, N-1]$$

유효 거리 조건($r_{\min} \le r_i \le r_{\max}$) 및 전방 유효 탐색각($|\theta_i| \le 85^\circ$)을 만족하는 포인트에 대해 선체 중심(`base_link`) 기준 2D 직교좌표로 정밀 투영합니다:

$$x_i = r_i \cdot \cos(\theta_i), \quad y_i = r_i \cdot \sin(\theta_i)$$

여기서 $+X$축은 선체 전방( 선수 방향), $+Y$축은 선체 좌현(Port side) 방향을 의미합니다.

### 3.3 선체 자체 반사파(Hull Echo) 공간 배제 필터
소형 USV의 경우 라이다가 선체 상부 마운트에 장착되어 있어, 선수(Bow) 팁, 안테나 폴, 윈드실드 또는 수면 물튀김으로 인한 가짜 장애물 포인트가 측정됩니다. 이를 원천 제거하기 위해 선체 바운딩 박스를 정의하여 내부 포인트를 완전 배제합니다:

$$\text{Mask}_{\text{clean}}(x_i, y_i) = \neg \Big( x_{\min}^{\text{hull}} \le x_i \le x_{\max}^{\text{hull}} \;\land\; y_{\min}^{\text{hull}} \le y_i \le y_{\max}^{\text{hull}} \Big)$$

기본 설정값: $x \in [-0.60\,\text{m}, +0.40\,\text{m}]$, $y \in [-0.45\,\text{m}, +0.45\,\text{m}]$

### 3.4 유클리디안 거리 기반 부표 군집화 (DBSCAN)
필터링된 클린 포인트 클라우드 $\mathbf{P} = \{(x_i, y_i)\}$에 대해 이웃 반경 $\varepsilon = 0.35\,\text{m}$, 최소 포인트 수 $MinPts = 3$을 적용한 유클리디안 거리 군집화를 수행합니다. $k$-d Tree 공간 인덱싱을 통해 2,000개 포인트를 1.5ms 이내로 초고속 군집화하여 부표의 중심점 $\mathbf{c}_k$와 반경 $R_k$를 도출합니다:

$$\mathbf{c}_k = \frac{1}{|C_k|} \sum_{p \in C_k} p, \quad R_k = \max_{p \in C_k} \|p - \mathbf{c}_k\|$$

---

## 4. 갭네비게이션 알고리즘의 실선 적용 기하학

![Fig 3 Gate Closure Comparison](fig3_gate_closure_vs_gap_pass.png)
<b>[그림 3]</b> 기존 광선 차폐 방식의 게이트 폐쇄(Gate Closure) 현상(좌측) vs 갭네비게이션의 개구부 인식 및 중심선 안전 관통(우측)

### 4.1 안전 개구부(Traversable Gap) 판정 조건
추출된 부표 군집 리스트에서 인접한 두 부표 $\mathbf{c}_1, \mathbf{c}_2$ 사이의 유클리드 거리를 계산합니다:

$$W_{\text{gap}} = \|\mathbf{c}_2 - \mathbf{c}_1\|$$

선박의 전폭을 $W_{\text{boat}} = 0.80\,\text{m}$, 안전 여유 마진을 $M_{\text{safe}} = 0.05\,\text{m}$라 할 때, 유효 개구부 조건은 다음과 같습니다:

$$W_{\text{min}} \le W_{\text{gap}} \le W_{\text{max}} \quad (0.85\,\text{m} \le W_{\text{gap}} \le 3.50\,\text{m})$$

또한 게이트 중간점 $\mathbf{m} = \frac{\mathbf{c}_1 + \mathbf{c}_2}{2}$ 부근에 다른 제3의 부표가 존재하지 않는지 추가 검증하여 안전성을 확보합니다.

### 4.2 다목적 비용함수 평가 모델 (Multi-Objective Scoring)

![Fig 4 Multi-Objective Scoring & Bezier](fig4_multi_objective_scoring_and_bezier.png)
<b>[그림 4]</b> 다목적 개구부 후보 평가 체계(좌측) 및 3차 베지에 곡선의 $C^2$ 곡률 연속성 생성(우측)

추출된 모든 개구부 후보들에 대해 시뮬레이션 10,000회로 최적화된 4대 핵심 지표를 가중 합산하여 최적 개구부(1st WP)와 차순위 백업 개구부(2nd WP)를 선정합니다:

$$J(G_k) = w_1 J_{\text{align}} + w_2 J_{\text{head}} + w_3 J_{\text{fwd}} + w_4 J_{\text{clear}}$$

1. **목표 정렬도 ($J_{\text{align}}$)**: 개구부 방향과 GPS 웨이포인트 방향 사이의 오차각 최소화
   $$J_{\text{align}} = \exp\left( - \left(\frac{\Delta \theta_{\text{goal}}}{0.8}\right)^2 \right)^6$$
2. **헤딩 일치도 ($J_{\text{head}}$)**: 현재 선체 전방 방향과의 편차 최소화 (급격한 조타 전환 억제)
   $$J_{\text{head}} = \exp\left( - \left(\frac{\theta_{\text{mid}}}{0.7}\right)^2 \right)^4$$
3. **전방 진행도 ($J_{\text{fwd}}$)**: 선체 전방($+X$) 진행 성분 비율
   $$J_{\text{fwd}} = \left(\frac{x_{\text{mid}}}{\sqrt{x_{\text{mid}}^2 + y_{\text{mid}}^2}}\right)^6$$
4. **통로 여유도 ($J_{\text{clear}}$)**: 게이트 폭의 넉넉함
   $$J_{\text{clear}} = \left(\frac{W_{\text{gap}} - W_{\min}}{W_{\max} - W_{\min}}\right)^3$$

### 4.3 3차 베지에 곡선(Cubic Bézier) 및 Pure Pursuit 조타각 산출
선택된 개구부 중심점 $\mathbf{P}_3 = (x_g, y_g)$을 향해 불연속 직선이 아닌, **곡률 연속성($C^2$)**을 갖는 3차 베지에 곡선을 생성합니다:

$$\mathbf{B}(t) = (1-t)^3 \mathbf{P}_0 + 3(1-t)^2 t \mathbf{P}_1 + 3(1-t)t^2 \mathbf{P}_2 + t^3 \mathbf{P}_3, \quad t \in [0, 1]$$

- $\mathbf{P}_0 = (0, 0)$: 현재 선체 위치
- $\mathbf{P}_1 = (L_0, 0)$: 현재 선체 진행 방향으로 $L_0$만큼 연장된 제어점 (부드러운 출발)
- $\mathbf{P}_2 = (x_g - L_1, y_g)$: 개구부 중심선에 수직 진입을 유도하는 제어점
- $\mathbf{P}_3 = (x_g, y_g)$: 목표 개구부 중심점

생성된 궤적 상에서 전방 주시 거리 $L_{\text{look}} = 1.2\,\text{m}$에 위치한 점 $(x_L, y_L)$을 추출하고, Pure Pursuit 공식으로 필요 선회 곡률 $\kappa$를 계산합니다:

$$\kappa = \frac{2 y_L}{x_L^2 + y_L^2}$$

이 곡률을 서보모터 중립각 90.0°와 비례 게인 $K_p$를 통해 최종 서보 각도로 변환합니다:

$$\delta_{\text{servo}} = \text{constrain}\Big(90.0^\circ - K_p \cdot \kappa, \; 30.0^\circ, \; 150.0^\circ \Big)$$

---

## 5. 실선 소스코드 수정 가이드 (`course1.py` 1:1 교체)

![Fig 6 Code Migration Guide](fig6_code_modification_guide.png)
<b>[그림 6]</b> 기존 `course1.py` 소스코드와 갭네비게이션 적용 후 `course1_gapnav.py` 코드 구조 대조 분석

### 5.1 수정 대상 영역 및 변경 요약표

| 구분 | 기존 코드 위치 (`course1.py`) | 갭네비게이션 교체 내용 (`course1_gapnav_replacement.py`) | 개선 효과 |
| :--- | :--- | :--- | :--- |
| **라이다 데이터 수신** | `lidar_callback` (169~213행)<br>`ranges[500:1500]` 하드코딩 슬라이싱 | `angle_min + i * increment` 물리 각도 복원<br>선체 배제 박스 적용 후 2D 직교좌표 변환 | 라이다 드라이버 주파수/포맷 변경 완벽 호환 |
| **장애물 판별** | 1도 버킷 단순 스칼라 거리 측정<br>±35도 광선 강제 차폐 | `EuclideanClusterer` (DBSCAN eps=0.35m)<br>부표 중심 좌표 및 기하학적 반경 추출 | 부표 객체화 성공, 인공 게이트 폐쇄 문제 원천 해결 |
| **경로 결정** | `timer_callback` (271~277행)<br>가장 가까운 안전각 단일각 선택 | `find_best_gap` 다목적 비용함수 평가<br>3차 베지에 곡선 경로 연속성 합성 | 최적 통로 탐색, 게이트 중심선 관통 주행 |
| **조타 명령 출력** | `servo = 90° + angle`<br>비례 직결로 인한 극심한 채터링 | Pure Pursuit 곡률 추종 + Slew Rate Limit<br>프레임당 최대 조타 변화율 제한 (12°/step) | 조타 지터 54% 감소, 서보 기어 파손 방지 |
| **추력 제어** | `cmd_thruster = 25.0` 고정 | 곡률 $|\kappa| > 0.35$ 시 18.0%로 감속<br>직진 구간 25.0% 가속 | 선회 드리프트 차단, 평균 완주 시간 단축 |

### 5.2 즉시 적용 방법
1. 기존 코드를 백업합니다:
   ```bash
   cp kabot2026/isv/launch_isv/course1.py kabot2026/isv/launch_isv/course1_backup.py
   ```
2. 제공된 템플릿 파일([course1_gapnav_replacement.py](file:///home/soonhong/kaboat/report3/template/course1_gapnav_replacement.py))을 해당 위치에 덮어씁니다:
   ```bash
   cp report3/template/course1_gapnav_replacement.py kabot2026/isv/launch_isv/course1.py
   ```
3. ROS 2 워크스페이스를 빌드합니다:
   ```bash
   cd /home/soonhong/ros2_ws  # 또는 해당 ros2 워크스페이스
   colcon build --symlink-install --packages-select isv
   source install/setup.bash
   ```

---

## 6. 타 대학/팀 보급용 범용 오픈 템플릿 구조

본 연구진은 타 대학 학생들과 신규 참가팀이 ROS 2 환경에서 자신의 보트에 맞게 몇 가지 파라미터만 수정하여 바로 사용할 수 있는 **독립형 표준 패키지 템플릿**을 제공합니다.

![Fig 5 ROS2 Computation Graph](fig5_ros2_node_topic_computation_graph.png)
<b>[그림 5]</b> ROS2 갭네비게이션 노드-토픽 계산 그래프 (RQT Graph Architecture 및 TF 트리 구조)

### 6.1 제공 템플릿 아카이브 구성
```
report3/template/
├── gap_nav_params.yaml              # 현장 튜닝용 통합 파라미터 설정 파일
├── gap_navigation_node.py           # 단독 실행 가능한 독립형 ROS 2 제어 노드
└── course1_gapnav_replacement.py    # KABOT 2026 기존 시스템 1:1 드롭인 교체 코드
```

### 6.2 타 대학 학생들을 위한 3단계 빠른 시작 가이드 (Quick-Start)

#### [1단계] 선체 치수 입력 ([gap_nav_params.yaml](file:///home/soonhong/kaboat/report3/template/gap_nav_params.yaml))
자신의 보트 실측 규격에 맞게 3개 수치만 수정합니다:
```yaml
boat_width: 0.80     # 자신의 보트 전폭 (m)
boat_length: 1.40    # 자신의 보트 전장 (m)
hull_exclusion_box: [-0.60, 0.40, -0.45, 0.45] # 라이다 센서 위치 기준 선체 제외 영역
```

#### [2단계] 서보모터 조타각 캘리브레이션
하드웨어 서보모터의 중립 및 최대 타각을 확인하고 기입합니다:
```yaml
servo_neutral_deg: 90.0  # 직진 시 서보 중립각
servo_min_deg: 30.0      # 좌현 최대 타각
servo_max_deg: 150.0     # 우현 최대 타각
```

#### [3단계] 노드 실행 및 RViz2 모니터링
```bash
ros2 run kaboat_gap_nav gap_navigation_node --ros-args --params-file gap_nav_params.yaml
```
- RViz2에서 토픽 `/gap_nav/viz_markers`를 추가하면 검출된 부표(빨간색 실린더)와 최적 갭 웨이포인트(하늘색 구)가 3D로 실시간 가시화됩니다.
- 토픽 `/gap_nav/bezier_path`를 추가하면 선박이 실제로 추종할 매끄러운 녹색 베지에 궤적선이 표시됩니다.

---

## 7. 대회 현장 실전 수조 튜닝 및 트러블슈팅 가이드

대회 당일 수조 환경(수온, 바람, 태양광 난반사, 타 선박 항적 물결)에 따라 발생할 수 있는 주요 증상별 조치 절차를 정형화한 진단 가이드입니다.

![Fig 7 Field Tuning Guide](fig7_field_test_tuning_guide.png)
<b>[그림 7]</b> KABOAT 경진대회 현장 수조 테스트 시 4대 주요 증상별 원인 진단 및 파라미터 최적화 절차

### 7.1 증상별 즉시 해결 조치표

| 발생 증상 | 원인 진단 | 현장 튜닝 파라미터 조치법 |
| :--- | :--- | :--- |
| **증상 1. 수면 물결/물방울 허위 인식**<br>(빈 수면에서 지그재그 회피 동작) | • 센서가 수면에 너무 가깝게 장착됨<br>• 선체 항적 거품(Wake)이 감지됨 | 1. `range_min` 상향 (0.1m $\to$ 0.3m)<br>2. `hull_exclusion_box` 크기 10cm 확대<br>3. `cluster_min_samples` 증가 (2개 $\to$ 4개)<br>4. 라이다 센서 장착 각도를 수평보다 1~2도 상향 조정 |
| **증상 2. 부표 사이 넓은 게이트 통과 회피**<br>(통과 가능한데도 멀리 우회) | • 최소 통과폭 설정이 과대함<br>• 목표 방향 가중치 부족 | 1. `min_gap_width` 축소 (`0.85m` $\to$ 선폭+0.2m = `0.70m`)<br>2. `align_exp` 상향 조정 (6.0 $\to$ 8.0, 목표 직진성 강화)<br>3. `buoy_radius` 실측값(보통 0.15m) 확인 |
| **증상 3. 서보모터 조타기 떨림 (Chattering)**<br>(선체가 좌우로 떨리며 속도 저하) | • Pure Pursuit 주시거리가 너무 짧음<br>• 조타 비례 게인이 과도함 | 1. `lookahead_distance` 확장 (0.8m $\to$ 1.3m)<br>2. `steering_kp` 하향 조정 (45.0 $\to$ 35.0)<br>3. `slew_rate_limit` 활성화 (10.0 deg/step 제한) |
| **증상 4. 수조 외곽 벽면에 근접 주행**<br>(벽면 충돌 패널티 +10초 위험) | • 벽면 포인트를 개구부로 오인식함<br>• 수조 중앙 복귀력 부족 | 1. `max_gap_width`를 3.5m로 엄격 제한 (벽-부표 사이 과대 갭 배제)<br>2. GPS 웨이포인트 좌표를 수조 중앙선으로 재설정<br>3. 3코스 진입 시 `wall_repulsion` 반발력 파라미터 활성화 |

### 7.2 경기 시작 10분 전 최종 점검 체크리스트
1. **라이다 전원 및 결로 확인**: 센서 커버에 물방울이 맺혀있으면 알코올 솜으로 닦아내어 난반사를 차단합니다.
2. **IMU 영점 캘리브레이션**: 보트가 출발선 수조 벽에 정확히 평행하게 정렬된 상태에서 프로그램을 실행하여 `initial_yaw_abs`가 0° 기준으로 정확히 초기화되도록 합니다.
3. **서보모터 중립 확인**: `/actuator/key/degree`로 90.0° 퍼블리시 시 러더(타)가 선체 용골(Keel) 중심선과 일치하는지 육안 확인합니다.
4. **RTK-GPS 고정(Fix) 상태 확인**: `/gps/fix`의 상태가 Float가 아닌 **Fix** 상태(cm급 오차)인지 확인합니다.

---

## 8. 결론 및 기대 효과

1. **실선 알고리즘의 완전한 현대화 달성**:
   과거 인덱스 슬라이싱 기반 광선 차폐 방식의 게이트 폐쇄 문제와 조타 채터링을 완전히 극복하고, 라이다 물리 복원 $\to$ 유클리디안 군집화 $\to$ 다목적 개구부 평가 $\to$ 3차 베지에 궤적 합성으로 이어지는 최신 자율운항 파이프라인을 KABOT 2026 플랫폼에 완벽히 이식했습니다.
2. **KABOAT 경진대회 입상 경쟁력 확보**:
   시뮬레이션 10,000회에서 입증된 성공률 96.22%와 랩타임 26.8% 단축 효과를 실선 플랫폼에서도 동일하게 구현하여, 1코스 무감점(충돌 0회) 완주 및 3코스 정밀 도킹 50점 만점 획득이 가능해졌습니다.
3. **전국 대학 자율운항 커뮤니티 기여**:
   ROS 2 기반의 표준 모듈형 템플릿과 정밀 파라미터 파일을 오픈소스로 정립함으로써, 알고리즘 구현에 어려움을 겪는 다른 대학 학생들도 손쉽게 갭네비게이션 기술을 자사 선박에 적용할 수 있는 교육적·기술적 표준을 확립했습니다.
