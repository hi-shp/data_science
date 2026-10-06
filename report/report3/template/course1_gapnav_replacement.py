#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
KABOAT 2026 - Course 1 갭네비게이션(Gap Navigation) 1:1 교체형 실선 코드
================================================================================
기존 isv/launch_isv/course1.py의 광선 차폐 회피 로직을
신규 개발된 갭네비게이션 알고리즘으로 100% 1:1 대체한 코드입니다.
================================================================================
"""

import os, sys, yaml, math, time, signal
import numpy as np
from math import degrees
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64, String
from sensor_msgs.msg import NavSatFix, Imu, LaserScan
from vision_msgs.msg import Detection2DArray
from mechaship_interfaces.msg import RgbwLedColor
from tf_transformations import euler_from_quaternion

def constrain(v, lo, hi):
    if math.isnan(v): return lo + (hi - lo) / 2.0
    return lo if v < lo else hi if v > hi else v

def norm_text(s):
    return " ".join(str(s).strip().lower().split())

def wrap_180(deg):
    return (deg + 180.0) % 360.0 - 180.0


class ISV_2026_GapNav(Node):
    def __init__(self):
        super().__init__("ISV_2026")
        self.load_params_from_yaml()

        # 액추에이터 및 디버그 퍼블리셔
        self.key_publisher = self.create_publisher(Float64, "/actuator/key/degree", 10)
        self.thruster_publisher = self.create_publisher(Float64, "/actuator/thruster/percentage", 10)
        self.dist_publisher = self.create_publisher(Float64, "/waypoint/distance", 10)
        self.rel_deg_publisher = self.create_publisher(Float64, "/waypoint/rel_deg", 10)
        self.goal_publisher = self.create_publisher(NavSatFix, "/waypoint/goal", 10)
        self.curr_yaw_publisher = self.create_publisher(Float64, "/current_yaw", 10)
        self.safe_angle_list_publisher = self.create_publisher(String, "/safe_angles_list", 10)
        self.safe_angle_publisher = self.create_publisher(Float64, "/safe_angle", 10)
        self.led_publisher = self.create_publisher(RgbwLedColor, "/actuator/rgbwled/color", 10)
        self.led_string_publisher = self.create_publisher(String, "/led_color", 10)
        self.target_name_publisher = self.create_publisher(String, "/target_name", 10)
        self.target_angle_publisher = self.create_publisher(Float64, "/target_angle", 10)
        self.state_publisher = self.create_publisher(String, "/state", 10)
        self.zero_count_publisher = self.create_publisher(String, "/zero_count", 10)

        # 센서 서브스크라이버
        self.imu_sub = self.create_subscription(Imu, "/imu", self.imu_callback, qos_profile_sensor_data)
        self.gps_sub = self.create_subscription(NavSatFix, "/gps/fix", self.gps_callback, qos_profile_sensor_data)
        self.lidar_sub = self.create_subscription(LaserScan, "/scan", self.lidar_callback, qos_profile_sensor_data)
        self.det_sub = self.create_subscription(Detection2DArray, "/detections", self.detection_callback, qos_profile_sensor_data)

        # 상태 제어 변수
        self.phase = "GPS"  # (GPS -> HOPING -> DETECTION -> DONE -> WALL)
        self.origin = None
        self.origin_set = False
        self.wp_index = 0
        self.initial_yaw_abs = None
        self.current_yaw_rel = 0.0
        self.dist_to_goal_m = None
        self.goal_rel_deg = None
        self.latest_det = None
        self.arrived_all = False
        self.start_time = self.get_clock().now()
        self.yaw_zero_count = 0
        self.last_zero_time = 0.0
        self.zero_count_cooldown = 8.0

        # ----------------------------------------------------------------------
        # [신규] 갭네비게이션 내부 파라미터 및 상태
        # ----------------------------------------------------------------------
        self.latest_clusters = []
        self.best_gap = None
        self.prev_steering = self.servo_neutral_deg
        self.lookahead_dist = 1.20   # Pure Pursuit 주시거리 (m)
        self.steering_kp = 42.0       # 곡률 -> 서보 변환 게인
        self.slew_rate_limit = 12.0  # 스텝당 최대 조타 변화율 (deg/step)

        self.create_timer(self.timer_period, self.timer_callback)
        self.led_by_name("off")
        self.get_logger().info("Course 1 with Gap Navigation Initialized")

    def load_params_from_yaml(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(script_dir, "isv_params.yaml")
        with open(yaml_path, "r") as file:
            params = yaml.safe_load(file)
        self.timer_period = float(params["node_settings"]["timer_period"])
        self.servo_neutral_deg = float(params["servo"]["neutral_deg"])
        self.servo_min_deg = float(params["servo"]["min_deg"])
        self.servo_max_deg = float(params["servo"]["max_deg"])
        self.waypoints = params["navigation"]["waypoints"]
        self.arrival_radii = params["navigation"]["arrival_radius"]
        self.default_thruster = float(params["thruster"]["course1"])
        v = params["vision"]
        self.screen_width = int(v["screen_width"])
        self.angle_factor = float(v["angle_conversion_factor"])
        self.available_objects = v["available_objects"]
        self.hoping_target = v["hoping_target"]
        self.detection_target = v["detection_target"]

    def led_by_name(self, name):
        name = norm_text(name)
        r, g, b, w = 0, 0, 0, 0
        pub_name = "off"
        for color in ["blue", "green", "red", "white"]:
            if name.startswith(color):
                pub_name = color
                if color == "red": g = 200
                elif color == "green": r = 200
                elif color == "blue": b = 200
                elif color == "white": w = 200
                break
        self.led_publisher.publish(RgbwLedColor(red=r, green=g, blue=b, white=w))
        self.led_string_publisher.publish(String(data=pub_name))

    def gps_enu_converter(self, lla):
        if self.origin is None: return 0.0, 0.0
        lat, lon, _ = lla
        lat0, lon0, _ = self.origin
        R = 6378137.0
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        latm = math.radians((lat + lat0) * 0.5)
        return dlon * R * math.cos(latm), dlat * R

    def imu_callback(self, msg: Imu):
        q = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        _, _, yaw_rad = euler_from_quaternion(q)
        current_yaw_abs = -yaw_rad
        if self.initial_yaw_abs is None:
            self.initial_yaw_abs = current_yaw_abs
        rel_yaw_deg = wrap_180(degrees(current_yaw_abs - self.initial_yaw_abs))
        self.current_yaw_rel = rel_yaw_deg
        self.curr_yaw_publisher.publish(Float64(data=float(rel_yaw_deg)))

        current_time = time.time()
        if self.phase == "HOPING" and -10.0 < rel_yaw_deg < 0.0:
            if (current_time - self.last_zero_time) > self.zero_count_cooldown:
                self.yaw_zero_count += 1
                self.last_zero_time = current_time
                if self.yaw_zero_count >= 2:
                    self.zero_count_publisher.publish(String(data=str(self.yaw_zero_count)))
                    self.led_by_name("off")
                    self.phase = "DETECTION"

    # ==========================================================================
    # [핵심 교체] 라이다 콜백: 물리 삼각함수 복원 및 유클리디안 군집화
    # ==========================================================================
    def lidar_callback(self, data: LaserScan):
        ranges = np.array(data.ranges, dtype=np.float32)
        n = len(ranges)
        if n == 0: return

        # 1. 센서 독립적 물리 각도 배열 산출
        angles = data.angle_min + np.arange(n, dtype=np.float32) * data.angle_increment

        # 2. 전방 170도 (±85도) & 유효 거리 (0.2m ~ 8.0m) 필터링
        fov_mask = (angles >= -math.radians(85.0)) & (angles <= math.radians(85.0)) & \
                   np.isfinite(ranges) & (ranges >= 0.20) & (ranges <= 8.0)

        r_val = ranges[fov_mask]
        ang_val = angles[fov_mask]
        if len(r_val) == 0:
            self.latest_clusters = []
            return

        # 3. 극좌표 -> 직교좌표 (base_link 기준: x=전방, y=좌현)
        px = r_val * np.cos(ang_val)
        py = r_val * np.sin(ang_val)

        # 4. 선체 자체 반사파(Hull Echo) 제거 (전방 -0.5~0.4m, 좌우 ±0.45m)
        clean_mask = ~((px >= -0.50) & (px <= 0.40) & (py >= -0.45) & (py <= 0.45))
        clean_px = px[clean_mask]
        clean_py = py[clean_mask]

        if len(clean_px) < 3:
            self.latest_clusters = []
            return

        # 5. 초고속 유클리디안 군집화 (DBSCAN 대치)
        pts = np.column_stack((clean_px, clean_py))
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        tree = cKDTree(pts)
        pairs = tree.query_pairs(0.35, output_type='ndarray')
        if len(pairs) == 0:
            self.latest_clusters = []
            return

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        adj = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(pts), len(pts)))
        n_comp, labels = connected_components(adj, directed=False)

        counts = np.bincount(labels, minlength=n_comp)
        valid_cids = np.where(counts >= 3)[0]

        clusters = []
        for cid in valid_cids:
            c_pts = pts[labels == cid]
            centroid = np.mean(c_pts, axis=0)
            rad = np.max(np.linalg.norm(c_pts - centroid, axis=1))
            if rad <= 0.60:
                clusters.append({"pos": centroid, "radius": float(rad)})

        self.latest_clusters = clusters

    def gps_callback(self, gps: NavSatFix):
        if math.isnan(gps.latitude) or math.isnan(gps.longitude) or self.initial_yaw_abs is None:
            return
        if not self.origin_set:
            self.origin = [gps.latitude, gps.longitude, gps.altitude]
            self.origin_set = True
            self.get_logger().info(f"시작 위치: {self.origin[:2]}")
            self.update_current_goal()
        curr_e, curr_n = self.gps_enu_converter([gps.latitude, gps.longitude, gps.altitude])
        if self.current_goal_enu is not None:
            goal_e, goal_n = self.current_goal_enu
            dx, dy = goal_e - curr_e, goal_n - curr_n
            self.dist_to_goal_m = math.hypot(dx, dy)
            target_ang_abs = degrees(math.atan2(dx, dy))
            target_ang_rel = wrap_180(target_ang_abs - degrees(self.initial_yaw_abs))
            self.goal_rel_deg = wrap_180(target_ang_rel - self.current_yaw_rel)
            self.dist_publisher.publish(Float64(data=float(self.dist_to_goal_m)))
            self.rel_deg_publisher.publish(Float64(data=float(self.goal_rel_deg)))

    def update_current_goal(self):
        if self.wp_index < len(self.waypoints):
            target_lat, target_lon = self.waypoints[self.wp_index]
            self.current_goal_enu = self.gps_enu_converter([target_lat, target_lon, 0.0])
            goal_msg = NavSatFix()
            goal_msg.latitude = target_lat
            goal_msg.longitude = target_lon
            self.goal_publisher.publish(goal_msg)

    def detection_callback(self, msg):
        self.latest_det = msg

    # ==========================================================================
    # [핵심 교체] 개구부 탐색 및 3차 베지에 Pure Pursuit 조타각 산출
    # ==========================================================================
    def compute_gap_navigation_steering(self):
        clusters = self.latest_clusters
        if len(clusters) < 2 or self.goal_rel_deg is None:
            # 장애물이 거의 없는 경우: GPS 목표 방향으로 완만하게 주행
            goal_ang = constrain(self.goal_rel_deg if self.goal_rel_deg else 0.0, -35.0, 35.0)
            return self.servo_neutral_deg - goal_ang, self.default_thruster

        goal_rad = math.radians(self.goal_rel_deg)
        best_gap = None
        highest_score = -1e9

        for i in range(len(clusters)):
            c1 = clusters[i]["pos"]
            for j in range(i + 1, len(clusters)):
                c2 = clusters[j]["pos"]
                v_gap = c2 - c1
                gap_w = float(np.linalg.norm(v_gap))

                # 게이트 통과 가능 폭 (0.85m ~ 3.5m)
                if not (0.85 <= gap_w <= 3.50):
                    continue

                mid = (c1 + c2) / 2.0
                mx, my = mid
                dist_m = math.hypot(mx, my)
                ang_m = math.atan2(my, mx)

                if mx <= 0.3: continue  # 후방 배제

                # 다목적 점수 계산
                align_err = wrap_180(degrees(ang_m - goal_rad))
                align_score = math.exp(-(align_err / 45.0)**2) * 4.0
                heading_score = math.exp(-(degrees(ang_m) / 40.0)**2) * 3.0
                fwd_score = (mx / (dist_m + 1e-6)) * 3.0
                total_score = align_score + heading_score + fwd_score

                if total_score > highest_score:
                    highest_score = total_score
                    best_gap = mid

        if best_gap is None:
            goal_ang = constrain(self.goal_rel_deg if self.goal_rel_deg else 0.0, -30.0, 30.0)
            return self.servo_neutral_deg - goal_ang, self.default_thruster

        # 베지에 궤적 기반 Pure Pursuit
        gx, gy = best_gap
        dist_g = math.hypot(gx, gy)
        L = dist_g * 0.5
        P0 = np.array([0.0, 0.0])
        P1 = np.array([L, 0.0])
        P2 = np.array([gx - L * 0.5, gy])
        P3 = np.array([gx, gy])

        t = np.linspace(0, 1, 20)[:, None]
        path = (1-t)**3 * P0 + 3*(1-t)**2*t * P1 + 3*(1-t)*t**2 * P2 + t**3 * P3

        # Pure Pursuit
        dists = np.linalg.norm(path, axis=1)
        look_idx = np.argmin(np.abs(dists - self.lookahead_dist))
        look_pt = path[look_idx]

        L2 = look_pt[0]**2 + look_pt[1]**2 + 1e-6
        curvature = 2.0 * look_pt[1] / L2

        target_servo = self.servo_neutral_deg - float(curvature * self.steering_kp)
        cmd_thruster = 18.0 if abs(curvature) > 0.35 else self.default_thruster
        return target_servo, cmd_thruster

    def timer_callback(self):
        if self.phase == "GPS":
            self.state_publisher.publish(String(data="GPS 갭네비게이션 모드"))
            elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            if not self.arrived_all and self.wp_index == 0:
                lat, lon = self.waypoints[self.wp_index]
                self.goal_publisher.publish(NavSatFix(latitude=lat, longitude=lon))

            if self.arrived_all or self.dist_to_goal_m is None or self.goal_rel_deg is None:
                self.cmd_thruster = 0.0
                self.cmd_key_degree = self.servo_neutral_deg
            else:
                if elapsed_time >= 40.0 and self.latest_det and self.latest_det.detections:
                    for d in self.latest_det.detections:
                        c_id = int(d.results[0].hypothesis.class_id)
                        if norm_text(self.available_objects[c_id]) == norm_text(self.hoping_target) or self.dist_to_goal_m <= self.arrival_radii[0]:
                            self.wp_index = 1
                            self.phase = "HOPING"
                            self.update_current_goal()

                # [신규] 갭네비게이션 조타 제어 실행
                target_servo, target_thruster = self.compute_gap_navigation_steering()

                # Slew Rate 제한으로 서보모터 기어 보호
                delta = target_servo - self.prev_steering
                delta = constrain(delta, -self.slew_rate_limit, self.slew_rate_limit)
                self.cmd_key_degree = constrain(self.prev_steering + delta, self.servo_min_deg, self.servo_max_deg)
                self.prev_steering = self.cmd_key_degree
                self.cmd_thruster = target_thruster

            self.key_publisher.publish(Float64(data=float(self.cmd_key_degree)))
            self.thruster_publisher.publish(Float64(data=float(self.cmd_thruster)))

        elif self.phase == "HOPING":
            self.state_publisher.publish(String(data="Hoping 모드"))
            self.led_by_name("blue")
            self.key_publisher.publish(Float64(data=40.0))
            if self.latest_det:
                for d in self.latest_det.detections:
                    c_id = int(d.results[0].hypothesis.class_id)
                    if norm_text(self.available_objects[c_id]) == norm_text(self.hoping_target):
                        cx = float(d.bbox.center.position.x if hasattr(d.bbox.center, 'position') else d.bbox.center.x)
                        ang = ((cx - (self.screen_width/2)) / (self.screen_width/2)) * self.angle_factor
                        self.key_publisher.publish(Float64(data=40.0 if ang <= 10.0 else self.servo_neutral_deg))

        elif self.phase == "DETECTION":
            self.state_publisher.publish(String(data="Detection 모드"))
            error = 5.0 - self.current_yaw_rel
            steer = self.servo_neutral_deg + error
            self.key_publisher.publish(Float64(data=constrain(steer, self.servo_min_deg, self.servo_max_deg)))
            if self.latest_det and self.latest_det.detections:
                for d in self.latest_det.detections:
                    c_id = int(d.results[0].hypothesis.class_id)
                    name = self.available_objects[c_id]
                    if norm_text(name) == norm_text(self.detection_target):
                        cx = float(d.bbox.center.position.x if hasattr(d.bbox.center, 'position') else d.bbox.center.x)
                        ang = ((cx - (self.screen_width/2)) / (self.screen_width/2)) * self.angle_factor
                        if ang <= 30.0:
                            self.led_by_name(name)
                            self.phase = "DONE"

        elif self.phase == "DONE":
            self.state_publisher.publish(String(data="Done 모드"))
            lat, lon = self.waypoints[1]
            self.goal_publisher.publish(NavSatFix(latitude=lat, longitude=lon))
            if self.dist_to_goal_m and self.dist_to_goal_m <= self.arrival_radii[1]:
                self.wp_index = 2
                self.led_by_name("off")
                self.phase = "WALL"
                self.update_current_goal()
            error = -90.0 - self.current_yaw_rel
            self.cmd_key_degree = constrain(self.servo_neutral_deg + error, self.servo_min_deg, self.servo_max_deg)
            self.cmd_thruster = self.default_thruster
            self.key_publisher.publish(Float64(data=float(self.cmd_key_degree)))
            self.thruster_publisher.publish(Float64(data=float(self.cmd_thruster)))

        elif self.phase == "WALL":
            self.state_publisher.publish(String(data="Wall 모드"))
            lat, lon = self.waypoints[2]
            self.goal_publisher.publish(NavSatFix(latitude=lat, longitude=lon))
            if self.dist_to_goal_m and self.dist_to_goal_m <= self.arrival_radii[2]:
                self.led_by_name("green")
                self.key_publisher.publish(Float64(data=self.servo_neutral_deg))
                self.thruster_publisher.publish(Float64(data=0.0))
                self.get_logger().info("최종 도킹 성공")
                self.destroy_node()
                sys.exit(0)

            # 도킹 접근 시에도 갭네비게이션 안전 경로 적용
            target_servo, target_thruster = self.compute_gap_navigation_steering()
            self.key_publisher.publish(Float64(data=float(target_servo)))
            self.thruster_publisher.publish(Float64(data=float(target_thruster)))

    def send_stop_commands(self):
        if not rclpy.ok(): return
        for _ in range(5):
            self.key_publisher.publish(Float64(data=float(self.servo_neutral_deg)))
            self.thruster_publisher.publish(Float64(data=0.0))
            self.led_by_name("off")
            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    node = ISV_2026_GapNav()
    def signal_handler(sig, frame):
        node.get_logger().warn("Stopped")
        node.send_stop_commands()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
