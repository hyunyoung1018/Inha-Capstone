#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, Twist

class StanleyPlannerNode(Node):
    def __init__(self):
        super().__init__('dh_stanley_planner_node')

        # ---- I/O 토픽 ----
        self.sub_topic = self.declare_parameter('sub_points_topic', '/lane/target_points').value
        # Behavior Planner로 넘기기 위해 임시 토픽 이름 사용
        self.cmd_topic = self.declare_parameter('cmd_vel_topic', '/stanley/cmd_vel').value

        # ---- 기하/좌표 및 스케일 변환 ----
        self.img_h    = self.declare_parameter('image_height', 480).value 
        self.img_w    = self.declare_parameter('image_width', 640).value
        self.center_x = self.declare_parameter('center_x', self.img_w / 2.0).value
        
        self.lane_width_px = self.declare_parameter('lane_width_px', 250.0).value
        self.lane_width_m  = self.declare_parameter('lane_width_m', 0.61).value
        self.px_to_m       = self.lane_width_m / self.lane_width_px

        # ---- [핵심 1] 스탠리 게인 파라미터 ----
        self.steer_k = self.declare_parameter('steer_k', 0.005).value # 횡방향 오차 복원력
        self.yaw_k   = self.declare_parameter('yaw_k', 1.5).value   # 헤딩 오차 복원력 (1.0 권장)
        
        # 최대 조향각 한계 (약 34도)
        self.max_steer = self.declare_parameter('max_steer', 0.45).value 

        # ---- [핵심 2] 스무딩 및 속도 제어 파라미터 (타인 코드 이식) ----
        # 조향 부드러움 정도 (낮을수록 부드럽지만 반응이 느림, 0.3~0.5 권장)
        self.steer_smoothing_alpha = self.declare_parameter('steer_smoothing_alpha', 0.5).value
        # 핸들을 급하게 꺾을 때 속도를 얼마나 줄일 것인가?
        self.steer_slowdown_ratio  = self.declare_parameter('steer_slowdown_ratio', 0.4).value
        
        self.v_max = self.declare_parameter('max_linear_speed', 1.0).value
        self.v_min = self.declare_parameter('min_linear_speed', 0.45).value
        
        self.wheelbase = self.declare_parameter('wheelbase_m', 0.23).value

        # 상태 변수
        self.prev_steer = None
        self.outlier_threshold = 0.15
        self.timeout = 0.5
        self.regression_history_size = int(self.declare_parameter('regression_history_size', 5).value)
        self.slope_jump_threshold = self.declare_parameter('slope_jump_threshold', 0.35).value
        self.intercept_jump_threshold = self.declare_parameter('intercept_jump_threshold_m', 0.08).value
        self.slope_jump_limit = self.declare_parameter('slope_jump_limit', 0.20).value
        self.intercept_jump_limit = self.declare_parameter('intercept_jump_limit_m', 0.05).value
        self.regression_history = deque(maxlen=max(self.regression_history_size, 1))

        self.sub = self.create_subscription(PoseArray, self.sub_topic,
                                            self.points_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self._last_msg_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog)

        self.get_logger().info('DH Stanley Planner (회귀식 안정화, 스무딩 & 다이나믹 감속) 적용 완료')

    def _to_vehicle_meters(self, px, py):
        lon_m = (self.img_h - py) * self.px_to_m   
        lat_m = (self.center_x - px) * self.px_to_m  
        return lon_m, lat_m

    def _stabilize_regression(self, m_new, c_new):
        if not self.regression_history:
            self.regression_history.append((m_new, c_new))
            return m_new, c_new

        prev_m = np.array([item[0] for item in self.regression_history], dtype=float)
        prev_c = np.array([item[1] for item in self.regression_history], dtype=float)
        m_ref = float(np.median(prev_m))
        c_ref = float(np.median(prev_c))

        m_delta = m_new - m_ref
        c_delta = c_new - c_ref

        if abs(m_delta) > self.slope_jump_threshold:
            m_new = m_ref + float(np.clip(m_delta, -self.slope_jump_limit, self.slope_jump_limit))

        if abs(c_delta) > self.intercept_jump_threshold:
            c_new = c_ref + float(np.clip(c_delta, -self.intercept_jump_limit, self.intercept_jump_limit))

        self.regression_history.append((m_new, c_new))
        return m_new, c_new

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

        # 선형 회귀 및 이상치 제거
        m_temp, c_temp = np.polyfit(lons_m, lats_m, 1)
        inlier_lons = [lon for lon, lat in zip(lons_m, lats_m) if abs(lat - (m_temp * lon + c_temp)) <= self.outlier_threshold]
        inlier_lats = [lat for lon, lat in zip(lons_m, lats_m) if abs(lat - (m_temp * lon + c_temp)) <= self.outlier_threshold]

        if len(inlier_lons) >= 2:
            m, c = np.polyfit(inlier_lons, inlier_lats, 1)
        else:
            m, c = m_temp, c_temp

        m, c = self._stabilize_regression(float(m), float(c))

        # ---------------------------------------------------------
        # [고급 제어 핵심 로직 시작]
        # ---------------------------------------------------------
        yaw = math.atan(m)  # 헤딩 오차
        
        error = c

        base_speed = self.v_max

        # 1. Stanley 제어기로 순수 목표 조향각(Raw Steering) 계산
        # 보내주신 코드의 로직 적용 (safe_v 방어 포함)
        raw_steering_angle = (self.yaw_k * yaw) + np.arctan2(self.steer_k * error, max(abs(base_speed), 0.01))
        
        raw_steering_angle = float(np.clip(raw_steering_angle, -self.max_steer, self.max_steer))

        # 2. 조향 스무딩 필터 적용 (이전 조향각과 섞어주기)
        if self.prev_steer is None:
            steering_delta = 0.0
            steering_angle = raw_steering_angle
        else:
            steering_delta = raw_steering_angle - self.prev_steer
            alpha = float(np.clip(self.steer_smoothing_alpha, 0.0, 1.0))
            steering_angle = self.prev_steer + (alpha * steering_delta)
            steering_angle = float(np.clip(steering_angle, -self.max_steer, self.max_steer))

        self.prev_steer = steering_angle

        # 3. 조향 기반 다이나믹 감속 (급격히 꺾을 때만 속도 늦추기)
        # max_steer 대비 현재 조향 변화량이 얼마나 큰지 비율(0~1)로 계산
        steer_change_ratio = min(abs(steering_delta) / max(abs(self.max_steer), 0.01), 1.0)
        
        # 감속 비율 적용
        speed_scale = 1.0 - (self.steer_slowdown_ratio * steer_change_ratio)
        final_speed = max(base_speed * speed_scale, self.v_min)

        # 4. 자전거 모델(Bicycle Model) 기반 각속도 변환
        angular_velocity = final_speed * np.tan(steering_angle) / self.wheelbase

        # ---------------------------------------------------------
        # 최종 명령 하달
        # ---------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = float(final_speed)
        cmd.angular.z = float(angular_velocity)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'err:{error*100:+3.1f}cm | steer_deg:{math.degrees(steering_angle):+4.1f} | v:{final_speed:.2f} | w:{angular_velocity:.2f}'
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
