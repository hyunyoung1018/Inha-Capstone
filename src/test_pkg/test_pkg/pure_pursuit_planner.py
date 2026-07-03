#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pure_pursuit_node.py
- Pure Pursuit 알고리즘 + PD 제어기 결합
- [NEW] EMA 스무딩 필터 적용 (직진 주행 시 덜덜거리는 진동 완벽 억제)
- [NEW] 조향 변화량 연동 다이나믹 감속 (급격한 핸들링 시 속도를 낮춰 전복 및 탈선 방지)
- 자전거 모델(Bicycle Model) 기반 정밀 각속도 변환
"""

import math
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, Twist

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ==========================================================
        # [튜닝 파라미터]
        # ==========================================================
        
        # 1. Pure Pursuit 파라미터
        self.lookahead_dist = self.declare_parameter('lookahead_dist', 1.2).value 
        self.wheelbase      = self.declare_parameter('wheelbase_m', 0.20).value # 자전거 모델 L 값 (0.23)

        # 2. PD 제어기 파라미터
        self.Kp = self.declare_parameter('Kp', 0.8).value   
        self.Kd = self.declare_parameter('Kd', 0.3).value  

        # 3. [NEW] 스무딩 및 감속 제어 파라미터
        self.steer_smoothing_alpha = self.declare_parameter('steer_smoothing_alpha', 0.35).value # EMA 스무딩 계수
        self.steer_slowdown_ratio  = self.declare_parameter('steer_slowdown_ratio', 0.35).value  # 감속 계수
        
        # 4. 속도 파라미터
        self.target_speed = self.declare_parameter('target_speed', 0.7).value 
        self.max_speed    = self.declare_parameter('max_speed', 1.0).value    
        self.min_speed    = self.declare_parameter('min_speed', 0.5).value 
        
        # 조향각 한계값 (클리핑용, ±0.6 rad)
        self.max_steer = self.declare_parameter('max_steer', 0.6).value 

        # 5. 카메라 및 스케일 정보
        self.img_h    = self.declare_parameter('image_height', 480).value 
        self.img_w    = self.declare_parameter('image_width', 640).value
        self.center_x = self.declare_parameter('center_x', self.img_w / 2.0).value
        
        self.lane_width_px = self.declare_parameter('lane_width_px', 240.0).value
        self.lane_width_m  = self.declare_parameter('lane_width_m', 0.61).value
        self.px_to_m       = self.lane_width_m / self.lane_width_px
        # ==========================================================

        # 토픽 설정
        self.sub_topic = self.declare_parameter('sub_points_topic', '/lane/target_points').value
        self.cmd_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value

        # 상태 저장용 변수
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.prev_steer = 0.0 # 스무딩을 위한 이전 조향각 저장
        self.timeout = 0.5 

        self.sub = self.create_subscription(PoseArray, self.sub_topic, self.points_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self._last_msg_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog)

        self.get_logger().info('Pure Pursuit + EMA 스무딩 가동 시작 (Alpha: {}, L: {}m)'.format(
            self.steer_smoothing_alpha, self.wheelbase))

    def _to_vehicle_meters(self, px, py):
        lon_m = (self.img_h - py) * self.px_to_m   
        lat_m = (self.center_x - px) * self.px_to_m  
        return lon_m, lat_m

    def points_callback(self, msg: PoseArray):
        self._last_msg_time = self.get_clock().now()

        if len(msg.poses) < 3:
            self._publish_stop()
            return

        path_points = []
        for p in msg.poses:
            x_m, y_m = self._to_vehicle_meters(p.position.x, p.position.y)
            if x_m > 0:
                path_points.append((x_m, y_m))

        if not path_points:
            self._publish_stop()
            return

        target_pt = None
        for (x, y) in path_points:
            dist = math.hypot(x, y)
            if dist >= self.lookahead_dist:
                target_pt = (x, y)
                break
        
        if target_pt is None:
            target_pt = path_points[-1]

        target_x, target_y = target_pt
        Ld = math.hypot(target_x, target_y)

        if Ld < 0.01:
            raw_steering = 0.0
        else:
            sin_alpha = target_y / Ld
            raw_steering = math.atan2(2.0 * self.wheelbase * sin_alpha, Ld)

        # ---------------------------------------------------------
        # 제어 로직 시작 (PD -> EMA 스무딩 -> 클리핑 -> 속도 -> 각속도)
        # ---------------------------------------------------------
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0.0: dt = 0.01
            
        # 1. PD 제어 연산
        error = raw_steering
        derivative = (error - self.prev_error) / dt
        pd_steering = (self.Kp * error) + (self.Kd * derivative)
        
        # 2. [핵심] EMA 스무딩 적용 (조향 급변 완화)
        # 공식: prev_steer + alpha * (raw - prev_steer)
        smoothed_steering = self.prev_steer + self.steer_smoothing_alpha * (pd_steering - self.prev_steer)
        
        # 3. 클리핑 (±0.6 rad 제한)
        final_steering = float(np.clip(smoothed_steering, -self.max_steer, self.max_steer))
        
        # 조향 변화량 (감속 계산용)
        steering_delta = final_steering - self.prev_steer

        # 상태 업데이트
        self.prev_error = error
        self.prev_time = current_time
        self.prev_steer = final_steering

        # 4. [핵심] 조향 연동 감속 (조향 변화량이 클수록 감속)
        # 변화량 비율: max_steer 대비 얼마나 급하게 꺾었는가 (0.0 ~ 1.0)
        ratio = min(abs(steering_delta) / max(abs(self.max_steer), 0.01), 1.0)
        
        # 공식: speed_scale = 1 - 0.35 * ratio
        speed_scale = 1.0 - (self.steer_slowdown_ratio * ratio)
        
        # 기본 최고 속도에서 스케일 다운 및 최소 속도(0.45) 보장
        current_speed = self.max_speed * speed_scale
        current_speed = float(np.clip(current_speed, self.min_speed, self.max_speed))

        # 5. [핵심] 자전거 모델 기반 각속도 변환
        # 공식: ω = v * tan(δ) / L
        angular_velocity = current_speed * math.tan(final_steering) / self.wheelbase

        # ---------------------------------------------------------
        # 퍼블리시
        # ---------------------------------------------------------
        cmd = Twist()
        cmd.linear.x = current_speed
        cmd.angular.z = float(angular_velocity)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'Ld:{Ld:.2f}m | Steer:{math.degrees(final_steering):+4.1f}deg | V:{current_speed:.2f}m/s | W:{angular_velocity:.2f}rad/s'
        )

    def _watchdog(self):
        dt = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if dt > self.timeout:
            self._publish_stop()

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
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