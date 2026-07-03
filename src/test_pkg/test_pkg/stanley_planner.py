#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, Twist


class StanleyPlannerNode(Node):
    def __init__(self):
        super().__init__('stanley_planner_node')

        # ---- I/O 토픽 ----
        self.sub_topic = self.declare_parameter('sub_points_topic', '/lane/target_points').value
        self.cmd_topic = self.declare_parameter('cmd_vel_topic', '/stanley/cmd_vel').value

        # ---- 기하/좌표 및 스케일 변환 ----
        self.img_w    = self.declare_parameter('image_width', 640).value
        # ★ 수정: 차량 앞바퀴 기준점을 ROI 높이가 아닌 전체 원본 이미지 바닥(480)으로 설정
        self.img_h    = self.declare_parameter('image_height', 480).value 
        self.center_x = self.declare_parameter('center_x', self.img_w / 2.0).value
        
        # 실제 차선 폭 61cm (0.61m) 적용
        self.lane_width_px = self.declare_parameter('lane_width_px', 240.0).value
        self.lane_width_m  = self.declare_parameter('lane_width_m', 0.61).value
        self.px_to_m       = self.lane_width_m / self.lane_width_px

        # ---- 스탠리 제어 게인 ----
        self.k_gain       = self.declare_parameter('stanley_k_gain', 0.5).value  
        self.invert_steer = self.declare_parameter('invert_steering', False).value

        # ---- 이상치 제거 ----
        self.outlier_threshold = self.declare_parameter('outlier_threshold_m', 0.15).value

        # ---- 속도 제어 ----
        self.v_max   = self.declare_parameter('max_linear_speed', 0.7).value
        self.v_min   = self.declare_parameter('min_linear_speed', 0.4).value
        self.w_max   = self.declare_parameter('max_angular_speed', 2.0).value
        self.curve_slowdown = self.declare_parameter('curve_slowdown', 1.5).value

        # ---- 안전 장치 ----
        self.timeout = self.declare_parameter('input_timeout', 0.5).value

        self.sub = self.create_subscription(PoseArray, self.sub_topic,
                                            self.points_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self._last_msg_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog)

        self.get_logger().info(f'stanley_planner_node 시작됨 (차량 기준 좌표 Y=480 적용)')

    def _to_vehicle_meters(self, px, py):
        """픽셀 좌표를 미터(m) 단위의 차량 기준 좌표계로 변환"""
        # 전방 (X축, 미터): 480에서 py(300~430)를 빼면 무조건 양수가 나옵니다!
        lon_m = (self.img_h - py) * self.px_to_m   
        # 좌측 (Y축, 미터)
        lat_m = (self.center_x - px) * self.px_to_m  
        return lon_m, lat_m

    def points_callback(self, msg: PoseArray):
        self._last_msg_time = self.get_clock().now()

        if len(msg.poses) < 4:
            self._publish_stop()
            return

        lons_m = []
        lats_m = []
        
        for p in msg.poses[2:]:
            lon, lat = self._to_vehicle_meters(p.position.x, p.position.y)
            if lon > 0: 
                lons_m.append(lon)
                lats_m.append(lat)

        if len(lons_m) < 2:
            self._publish_stop()
            return

        # 1. 이상치 제거 (Outlier Rejection in Meters)
        m_temp, c_temp = np.polyfit(lons_m, lats_m, 1)
        inlier_lons = []
        inlier_lats = []

        for lon, lat in zip(lons_m, lats_m):
            expected_lat = m_temp * lon + c_temp
            error = abs(lat - expected_lat)

            if error <= self.outlier_threshold:
                inlier_lons.append(lon)
                inlier_lats.append(lat)

        if len(inlier_lons) >= 2:
            m, c = np.polyfit(inlier_lons, inlier_lats, 1)
            used_points_count = len(inlier_lons)
        else:
            m, c = m_temp, c_temp
            used_points_count = len(lons_m)

        # ---------------------------------------------------------
        # 2. 스탠리 알고리즘 제어 계산 (High-Speed 보정 적용)
        # ---------------------------------------------------------
        heading_error = math.atan(m)  # psi (라디안)
        
        # [핵심 추가] 고속 주행을 위한 Preview (미래 오차 예측)
        # 속도(v_max)에 비례하여 살짝 앞(Look-ahead)을 봅니다. (예: 1.0m/s일 때 0.15m 앞)
        lookahead_distance = 0.0 * self.v_max 
        
        # y = mx + c 공식에 앞을 내다본 x값을 대입하여 미래의 횡방향 오차를 구합니다.
        cross_track_error = (m * lookahead_distance) + c

        # 속도 계산 (커브 감속)
        v = self.v_max / (1.0 + self.curve_slowdown * abs(heading_error))
        v = float(np.clip(v, self.v_min, self.v_max))

        # [스탠리 공식] 타이어 조향각(delta) 계산
        safe_v = max(v, 0.1)
        stanley_term = math.atan2(self.k_gain * cross_track_error, safe_v)
        
        delta = heading_error + stanley_term
        
        if self.invert_steer:
            delta = -delta
            
        # 최대 조향각 클리핑 (Ackermann 최소 회전 반경 한계 등 고려)
        max_steer_rad = math.radians(25.0) # 예: 최대 25도
        delta = float(np.clip(delta, -max_steer_rad, max_steer_rad))

        # ---------------------------------------------------------
        # [핵심] 3. Kinematic Model을 이용해 조향각(delta)을 각속도(w)로 변환
        # ---------------------------------------------------------
        wheelbase = 0.20  # Limo Pro 매뉴얼 기준 축거 200mm = 0.2m
        
        # 공식: w = (v * tan(delta)) / L
        if v > 0.0:
            w = (v * math.tan(delta)) / wheelbase
        else:
            w = 0.0

        w = float(np.clip(w, -self.w_max, self.w_max))

        # ---------------------------------------------------------
        # 4. 명령 발행
        # ---------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'점: {used_points_count} | HE(deg): {math.degrees(heading_error):+5.1f} | CTE(cm): {cross_track_error*100:+5.1f} || v: {v:.2f}, w: {w:.2f}'
        )

    def _watchdog(self):
        dt = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if dt > self.timeout:
            self._publish_stop()

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = StanleyPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n종료 중...\n')
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()