#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
KABOAT 자율운항보트 - ROS2 갭네비게이션(Gap Navigation) 표준 오픈 템플릿 노드
================================================================================
제공: 부산대학교 자율운항보트 연구팀 (hi-shp / kaboat)
호환성: ROS 2 Foxy, Humble, Iron, Rolling
입력 토픽:
  - /scan (sensor_msgs/msg/LaserScan): 2D 라이다 전방 반사 데이터
  - /imu (sensor_msgs/msg/Imu): 선체 자세 및 헤딩 각도
  - /gps/fix (sensor_msgs/msg/NavSatFix): RTK-GPS 위도/경도
출력 토픽:
  - /actuator/key/degree (std_msgs/msg/Float64): 서보모터 조타각 (30° ~ 150°)
  - /actuator/thruster/percentage (std_msgs/msg/Float64): 쓰러스터 출력 (0 ~ 100%)
  - /gap_nav/viz_markers (visualization_msgs/msg/MarkerArray): RViz2 디버깅 마커
  - /gap_nav/bezier_path (nav_msgs/msg/Path): RViz2 베지에 계획 궤적
================================================================================
"""

import os
import sys
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64, String
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import euler_from_quaternion


def constrain(val, min_val, max_val):
    """값을 최소/최대 범위 내로 클램핑"""
    if math.isnan(val):
        return (min_val + max_val) / 2.0
    return max(min_val, min(max_val, val))


def wrap_to_pi(rad):
    """각도를 [-pi, pi] 범위로 정규화"""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


# ==============================================================================
# 1. 라이다 센서 전처리 모듈 (Lidar Preprocessor)
# ==============================================================================
class LidarPreprocessor:
    def __init__(self, range_min=0.20, range_max=8.0, fov_deg=(-85.0, 85.0),
                 hull_box=(-0.60, 0.40, -0.45, 0.45)):
        self.range_min = range_min
        self.range_max = range_max
        self.fov_min_rad = math.radians(fov_deg[0])
        self.fov_max_rad = math.radians(fov_deg[1])
        self.hull_box = hull_box  # [x_min, x_max, y_min, y_max]

    def process_scan(self, scan_msg: LaserScan):
        """
        LaserScan 메시지를 받아 물리 각도를 복원하고,
        선체 자체 반사파 및 무효 데이터를 제거한 2D 직교좌표(x, y) 배열 반환
        """
        ranges = np.array(scan_msg.ranges, dtype=np.float32)
        n_points = len(ranges)
        if n_points == 0:
            return np.empty((0, 2), dtype=np.float32)

        # 1. 라이다 물리 스캔 각도 벡터 산출: angle = angle_min + i * increment
        angles = scan_msg.angle_min + np.arange(n_points, dtype=np.float32) * scan_msg.angle_increment

        # 2. 유효 거리 및 전방 FOV 마스크 생성
        valid_mask = (
            np.isfinite(ranges) &
            (ranges >= max(self.range_min, scan_msg.range_min)) &
            (ranges <= min(self.range_max, scan_msg.range_max)) &
            (angles >= self.fov_min_rad) &
            (angles <= self.fov_max_rad)
        )

        r_valid = ranges[valid_mask]
        ang_valid = angles[valid_mask]

        if len(r_valid) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # 3. 극좌표 -> 직교좌표 (base_link 좌표계: X=전방, Y=좌현)
        pts_x = r_valid * np.cos(ang_valid)
        pts_y = r_valid * np.sin(ang_valid)

        # 4. 선체 자체 반사파 (Hull Exclusion Box) 필터링
        x_min, x_max, y_min, y_max = self.hull_box
        hull_mask = (pts_x >= x_min) & (pts_x <= x_max) & (pts_y >= y_min) & (pts_y <= y_max)
        clean_mask = ~hull_mask

        clean_pts = np.column_stack((pts_x[clean_mask], pts_y[clean_mask]))
        return clean_pts


# ==============================================================================
# 2. 유클리디안 거리 기반 부표 군집화 모듈 (Euclidean Clusterer)
# ==============================================================================
class EuclideanClusterer:
    def __init__(self, eps=0.35, min_samples=3, max_cluster_radius=0.60):
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_radius = max_cluster_radius

    def cluster(self, points: np.ndarray):
        """
        2D 포인트 클라우드에 대해 초고속 군집화를 수행하여
        개별 부표 장애물의 중심 좌표(Centroids) 리스트 반환
        """
        if len(points) < self.min_samples:
            return []

        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        tree = cKDTree(points)
        pairs = tree.query_pairs(self.eps, output_type='ndarray')
        n_pts = len(points)

        if len(pairs) == 0:
            return []

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows), dtype=bool)
        adj = csr_matrix((data, (rows, cols)), shape=(n_pts, n_pts))
        n_comp, labels = connected_components(adj, directed=False)

        # 군집 크기 필터링 (최소 포인트 수 이상만 유효 부표로 인정)
        counts = np.bincount(labels, minlength=n_comp)
        valid_cluster_ids = np.where(counts >= self.min_samples)[0]

        clusters = []
        for cid in valid_cluster_ids:
            c_mask = (labels == cid)
            c_pts = points[c_mask]
            centroid = np.mean(c_pts, axis=0)
            radius = np.max(np.linalg.norm(c_pts - centroid, axis=1))

            # 과대 군집(외곽 긴 벽면)은 별도 분할하거나 배제
            if radius <= self.max_cluster_radius:
                clusters.append({
                    "pos": centroid,
                    "radius": float(radius),
                    "pts": c_pts
                })

        return clusters


# ==============================================================================
# 3. 갭 탐색, 다목적 평가 및 베지에 경로 생성 모듈 (Gap Navigation Planner)
# ==============================================================================
class GapNavigationPlanner:
    def __init__(self, params=None):
        self.params = params or {}
        self.min_gap_width = float(self.params.get("min_gap_width", 0.85))
        self.max_gap_width = float(self.params.get("max_gap_width", 3.50))
        self.gate_clearance = float(self.params.get("gate_clearance_margin", 0.20))
        self.align_exp = float(self.params.get("align_exp", 6.0))
        self.heading_exp = float(self.params.get("heading_exp", 4.0))
        self.forward_exp = float(self.params.get("forward_exp", 6.0))
        self.clear_exp = float(self.params.get("clear_exp", 3.0))

    def find_best_gap(self, clusters, goal_rel_pos):
        """
        군집화된 부표 목록에서 안전 개구부(Gap) 후보들을 모두 추출하고,
        학습된 다목적 비용함수를 적용하여 1순위 최적 개구부와 2순위 백업 개구부를 산출
        """
        if len(clusters) < 2:
            return None, None, []

        gaps = []
        n_c = len(clusters)
        gx_rel, gy_rel = goal_rel_pos
        dist_to_goal = math.hypot(gx_rel, gy_rel)
        goal_heading = math.atan2(gy_rel, gx_rel)

        # 1. 모든 부표 쌍 조합 간격 검사
        for i in range(n_c):
            c1 = clusters[i]["pos"]
            for j in range(i + 1, n_c):
                c2 = clusters[j]["pos"]
                v_gap = c2 - c1
                gap_w = float(np.linalg.norm(v_gap))

                # 게이트 폭 검사: [min_gap_width, max_gap_width]
                if not (self.min_gap_width <= gap_w <= self.max_gap_width):
                    continue

                # 제3의 장애물이 게이트 중간을 가로막고 있는지 검사
                mid = (c1 + c2) / 2.0
                blocked = False
                for k in range(n_c):
                    if k == i or k == j:
                        continue
                    ck = clusters[k]["pos"]
                    dist_to_mid = np.linalg.norm(ck - mid)
                    if dist_to_mid < (gap_w * 0.45):
                        blocked = True
                        break
                if blocked:
                    continue

                # 2. 다목적 비용함수 점수 산출
                mx, my = mid
                dist_m = math.hypot(mx, my)
                ang_m = math.atan2(my, mx)  # 선체 헤딩 기준 각도 (선체 전방=0)

                # 전방 진행도 (Forward progress)
                fwd_progress = mx / (dist_m + 1e-6)
                if fwd_progress < 0.20:  # 측방/후방 배제
                    continue

                # 목표점 방향 정렬 점수
                ang_err_goal = wrap_to_pi(ang_m - goal_heading)
                align_score = max(0.01, math.exp(-(ang_err_goal / 0.8)**2)) ** self.align_exp

                # 현재 선체 헤딩과의 일치도 (급조타 억제)
                heading_score = max(0.01, math.exp(-(ang_m / 0.7)**2)) ** self.heading_exp

                # 전방 진행성 점수
                fwd_score = max(0.01, fwd_progress) ** self.forward_exp

                # 여유 폭 점수
                norm_w = (gap_w - self.min_gap_width) / (self.max_gap_width - self.min_gap_width + 1e-6)
                width_score = max(0.01, norm_w) ** self.clear_exp

                total_score = align_score * 0.35 + heading_score * 0.25 + fwd_score * 0.25 + width_score * 0.15

                gaps.append({
                    "c1": c1,
                    "c2": c2,
                    "mid": mid,
                    "gap_w": gap_w,
                    "score": float(total_score)
                })

        if not gaps:
            return None, None, []

        gaps.sort(key=lambda x: x["score"], reverse=True)
        best_gap = gaps[0]
        backup_gap = gaps[1] if len(gaps) > 1 else None
        return best_gap, backup_gap, gaps

    def generate_cubic_bezier(self, target_mid, lookahead_scale=0.5):
        """
        현재 선체 위치 (0, 0)에서 목표 개구부 중간점까지의
        매끄러운 3차 베지에(Cubic Bézier) 궤적 생성
        """
        P0 = np.array([0.0, 0.0], dtype=np.float32)  # 선체 위치
        P3 = np.array(target_mid, dtype=np.float32)  # 갭 중간점

        dist = float(np.linalg.norm(P3 - P0))
        L = dist * lookahead_scale

        # P1: 현재 헤딩 방향(전방 X축)으로 연장된 제어점
        P1 = np.array([L, 0.0], dtype=np.float32)

        # P2: 개구부 수직 진입을 유도하는 제어점
        P2 = P3 - np.array([L * 0.6, 0.0], dtype=np.float32)

        # 3차 베지에 곡선 보간 (N=30 샘플)
        t = np.linspace(0.0, 1.0, 30)[:, None]
        path = (1.0 - t)**3 * P0 + 3.0 * (1.0 - t)**2 * t * P1 + 3.0 * (1.0 - t) * t**2 * P2 + t**3 * P3
        return path


# ==============================================================================
# 4. 메인 ROS 2 갭네비게이션 제어 노드 (Gap Navigation ROS2 Node)
# ==============================================================================
class GapNavigationNode(Node):
    def __init__(self):
        super().__init__("gap_navigation_node")

        # 1. 파라미터 선언 및 로드
        self.declare_parameters(
            namespace='',
            parameters=[
                ('timer_period', 0.1),
                ('servo_neutral_deg', 90.0),
                ('servo_min_deg', 30.0),
                ('servo_max_deg', 150.0),
                ('cruising_thruster', 25.0),
                ('turning_thruster', 18.0),
                ('emergency_thruster', 12.0),
                ('lookahead_distance', 1.20),
                ('steering_kp', 45.0),
                ('slew_rate_limit', 15.0),
                ('min_gap_width', 0.85),
                ('max_gap_width', 3.50),
            ]
        )

        self.timer_period = self.get_parameter('timer_period').value
        self.servo_neutral = self.get_parameter('servo_neutral_deg').value
        self.servo_min = self.get_parameter('servo_min_deg').value
        self.servo_max = self.get_parameter('servo_max_deg').value
        self.cruising_thruster = self.get_parameter('cruising_thruster').value
        self.turning_thruster = self.get_parameter('turning_thruster').value
        self.lookahead_dist = self.get_parameter('lookahead_distance').value
        self.steering_kp = self.get_parameter('steering_kp').value
        self.slew_rate = self.get_parameter('slew_rate_limit').value

        # 2. 퍼블리셔 & 서브스크라이버
        self.pub_servo = self.create_publisher(Float64, "/actuator/key/degree", 10)
        self.pub_thruster = self.create_publisher(Float64, "/actuator/thruster/percentage", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/gap_nav/viz_markers", 10)
        self.pub_path = self.create_publisher(Path, "/gap_nav/bezier_path", 10)

        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.cb_lidar, qos_profile_sensor_data)
        self.sub_imu = self.create_subscription(Imu, "/imu", self.cb_imu, qos_profile_sensor_data)
        self.sub_gps = self.create_subscription(NavSatFix, "/gps/fix", self.cb_gps, qos_profile_sensor_data)

        # 3. 알고리즘 모듈 인스턴스
        self.preprocessor = LidarPreprocessor()
        self.clusterer = EuclideanClusterer()
        self.planner = GapNavigationPlanner({
            "min_gap_width": self.get_parameter('min_gap_width').value,
            "max_gap_width": self.get_parameter('max_gap_width').value
        })

        # 4. 내부 상태 변수
        self.latest_scan = None
        self.current_yaw_deg = 0.0
        self.initial_yaw_deg = None
        self.origin_lla = None
        self.current_enu = (0.0, 0.0)
        self.goal_enu = (15.0, 0.0)  # 예시: 15m 전방 목표
        self.prev_servo_cmd = self.servo_neutral

        # 타이머 루프
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        self.get_logger().info("=== KABOAT Gap Navigation Node Initialized ===")

    def cb_imu(self, msg: Imu):
        q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        _, _, yaw_rad = euler_from_quaternion(q)
        yaw_deg = math.degrees(-yaw_rad)
        if self.initial_yaw_deg is None:
            self.initial_yaw_deg = yaw_deg
        self.current_yaw_deg = (yaw_deg - self.initial_yaw_deg + 180.0) % 360.0 - 180.0

    def cb_gps(self, msg: NavSatFix):
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        if self.origin_lla is None:
            self.origin_lla = (msg.latitude, msg.longitude)
            self.get_logger().info(f"GPS 원점 설정: {self.origin_lla}")

        lat0, lon0 = self.origin_lla
        R = 6378137.0
        dlat = math.radians(msg.latitude - lat0)
        dlon = math.radians(msg.longitude - lon0)
        latm = math.radians((msg.latitude + lat0) * 0.5)
        e = dlon * R * math.cos(latm)
        n = dlat * R
        self.current_enu = (e, n)

    def cb_lidar(self, msg: LaserScan):
        self.latest_scan = msg

    def control_loop(self):
        if self.latest_scan is None:
            return

        # 1. 라이다 전처리
        clean_pts = self.preprocessor.process_scan(self.latest_scan)

        # 2. 유클리디안 군집화
        clusters = self.clusterer.cluster(clean_pts)

        # 목표 상대 좌표 (로봇 기준)
        dx_enu = self.goal_enu[0] - self.current_enu[0]
        dy_enu = self.goal_enu[1] - self.current_enu[1]
        heading_rad = math.radians(self.current_yaw_deg)
        ch, sh = math.cos(heading_rad), math.sin(heading_rad)
        rel_goal_x = dx_enu * ch + dy_enu * sh
        rel_goal_y = -dx_enu * sh + dy_enu * ch

        # 3. 개구부(Gap) 탐색 및 다목적 평가
        best_gap, backup_gap, all_gaps = self.planner.find_best_gap(clusters, (rel_goal_x, rel_goal_y))

        # 4. 제어 명령 산출
        if best_gap is not None:
            target_pt = best_gap["mid"]
            # 3차 베지에 곡선 경로 생성
            bezier_path = self.planner.generate_cubic_bezier(target_pt)

            # Pure Pursuit 조타각 산출
            dists = np.linalg.norm(bezier_path, axis=1)
            look_idx = np.argmin(np.abs(dists - self.lookahead_dist))
            look_pt = bezier_path[look_idx]

            # 곡률 kappa = 2 * y / L^2
            L2 = look_pt[0]**2 + look_pt[1]**2 + 1e-6
            curvature = 2.0 * look_pt[1] / L2

            target_servo = self.servo_neutral - float(curvature * self.steering_kp)
            target_thruster = self.turning_thruster if abs(curvature) > 0.35 else self.cruising_thruster

            # RViz2 경로 퍼블리시
            self.publish_path(bezier_path)
            self.publish_markers(clusters, best_gap, all_gaps)
        else:
            # 개구부가 없는 경우: 단순 목표 지향 또는 안전 전진
            target_servo = self.servo_neutral
            target_thruster = self.cruising_thruster

        # 5. Slew Rate Limit (서보 급격한 떨림 방지)
        servo_delta = target_servo - self.prev_servo_cmd
        servo_delta = constrain(servo_delta, -self.slew_rate, self.slew_rate)
        cmd_servo = constrain(self.prev_servo_cmd + servo_delta, self.servo_min, self.servo_max)
        self.prev_servo_cmd = cmd_servo

        # 6. 액추에이터 퍼블리시
        self.pub_servo.publish(Float64(data=float(cmd_servo)))
        self.pub_thruster.publish(Float64(data=float(target_thruster)))

    def publish_path(self, path_pts):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        for p in path_pts:
            ps = PoseStamped()
            ps.pose.position.x = float(p[0])
            ps.pose.position.y = float(p[1])
            ps.pose.position.z = 0.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def publish_markers(self, clusters, best_gap, all_gaps):
        m_arr = MarkerArray()
        now = self.get_clock().now().to_msg()

        # 부표 마커
        for idx, c in enumerate(clusters):
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = "base_link"
            m.ns = "buoys"
            m.id = idx
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(c["pos"][0])
            m.pose.position.y = float(c["pos"][1])
            m.pose.position.z = 0.15
            m.scale.x = c["radius"] * 2.0
            m.scale.y = c["radius"] * 2.0
            m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 0.85
            m_arr.markers.append(m)

        # 1순위 최적 갭 웨이포인트 마커
        if best_gap:
            m_gap = Marker()
            m_gap.header.stamp = now
            m_gap.header.frame_id = "base_link"
            m_gap.ns = "best_gap"
            m_gap.id = 999
            m_gap.type = Marker.SPHERE
            m_gap.action = Marker.ADD
            m_gap.pose.position.x = float(best_gap["mid"][0])
            m_gap.pose.position.y = float(best_gap["mid"][1])
            m_gap.pose.position.z = 0.3
            m_gap.scale.x, m_gap.scale.y, m_gap.scale.z = 0.35, 0.35, 0.35
            m_gap.color.r, m_gap.color.g, m_gap.color.b, m_gap.color.a = 0.0, 0.95, 1.0, 0.9
            m_arr.markers.append(m_gap)

        self.pub_markers.publish(m_arr)


def main(args=None):
    rclpy.init(args=args)
    node = GapNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("노드 종료 중: 서보 중립 및 쓰러스터 정지 출력")
        node.pub_servo.publish(Float64(data=90.0))
        node.pub_thruster.publish(Float64(data=0.0))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
