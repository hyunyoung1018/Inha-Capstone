#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray, Twist
from scipy.optimize import minimize

class MPCPlannerNode(Node):
    def __init__(self):
        super().__init__('mpc_planner_node')

        # ---- I/O 토픽 ----
        self.sub_topic = self.declare_parameter('sub_points_topic', '/lane/target_points').value
        self.cmd_topic = self.declare_parameter('cmd_vel_topic', '/stanley/cmd_vel').value

        # ---- 차량 기하학 정보 ----
        self.img_h = self.declare_parameter('image_height', 480).value 
        self.img_w = self.declare_parameter('image_width', 640).value
        self.center_x = self.declare_parameter('center_x', self.img_w / 2.0).value
        
        self.lane_width_px = self.declare_parameter('lane_width_px', 250.0).value
        self.lane_width_m  = self.declare_parameter('lane_width_m', 0.61).value
        self.px_to_m       = self.lane_width_m / self.lane_width_px
        self.wheelbase     = self.declare_parameter('wheelbase_m', 0.20).value # Limo Pro 축거 0.2m

        # ---- MPC 파라미터 ----
        self.N = 6         # 예측 지평선 (Prediction Horizon)
        self.dt = 0.1       # 시간 간격 (Time step)
        
        # 가중치 제어 (Cost Weights)
        self.Q_pos = 1.5    # 위치 오차 가중치 (X, Y)
        self.Q_yaw = 2.0    # 헤딩 오차 가중치
        self.R_steer = 1.0  # 조향 입력 가중치 (급격한 조향 방지)
        self.R_vel = 0.1    # 속도 유지 가중치

        # 제어 제약 조건
        self.v_max = 0.7
        self.v_min = 0.45
        self.max_steer = 0.45

        # 안전 장치
        self.timeout = 0.5
        self._last_msg_time = self.get_clock().now()

        # 이전 제어 입력 저장 (솔버의 초기 예측값으로 사용되어 연산 속도 극대화)
        self.last_u = np.zeros(2 * self.N)

        self.sub = self.create_subscription(PoseArray, self.sub_topic, self.points_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        
        self.create_timer(0.1, self._watchdog)
        self.get_logger().info('Model Predictive Control (MPC) 플래너 노드 시작됨')

    def _to_vehicle_meters(self, px, py):
        lon_m = (self.img_h - py) * self.px_to_m   
        lat_m = (self.center_x - px) * self.px_to_m  
        return lon_m, lat_m

    def motion_model(self, x, y, yaw, v, delta):
        """차량 운동학 기하 모델을 통한 1개 스텝 미래 상태 예측"""
        next_x = x + v * math.cos(yaw) * self.dt
        next_y = y + v * math.sin(yaw) * self.dt
        next_yaw = yaw + (v * math.tan(delta) / self.wheelbase) * self.dt
        return next_x, next_y, next_yaw

    def cost_function(self, u, ref_m, ref_c):
        """미래 예측 궤적과 목표 궤적 간의 비용(Cost)을 계산"""
        cost = 0.0
        # 현재 차량 위치는 차량 기준 좌표계이므로 무조건 (0, 0, 0)에서 출발합니다.
        cx, cy, cyaw = 0.0, 0.0, 0.0

        for k in range(self.N):
            v = u[2 * k]
            delta = u[2 * k + 1]
            
            # 모델을 통해 미래 상태 업데이트
            cx, cy, cyaw = self.motion_model(cx, cy, cyaw, v, delta)

            # 레퍼런스 라인(직선 피팅 기준) 상의 목표 Y 좌표 및 헤딩 각도 계산
            ref_y = ref_m * cx + ref_c
            ref_yaw = math.atan(ref_m)

            # 오차 누적
            cost += self.Q_pos * ((cy - ref_y) ** 2)
            cost += self.Q_yaw * ((cyaw - ref_yaw) ** 2)
            cost += self.R_steer * (delta ** 2)
            cost += self.R_vel * ((v - self.v_max) ** 2)

        return cost

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

        # 차선 중심점들의 대표 직선 방정식 도출 (y = mx + c)
        m, c = np.polyfit(lons_m, lats_m, 1)

        # 제약 조건 범위 설정 (Bounds)
        bounds = []
        for _ in range(self.N):
            bounds.append((self.v_min, self.v_max))       # 속도 제약
            bounds.append((-self.max_steer, self.max_steer)) # 조향각 제약

        # 비선형 최적화 수행 (SLSQP 솔버 사용)
        res = minimize(
            self.cost_function,
            self.last_u,
            args=(m, c),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 15, 'ftol': 1e-3}
        )

        if res.success:
            # 최적화된 제어 입력 배열 중 첫 번째 스텝의 입력을 물리 명령어로 채택
            self.last_u = res.x
            optimal_v = res.x[0]
            optimal_delta = res.x[1]
        else:
            # 최적화 실패 시 차선 이탈 방지를 위해 이전의 유효한 제어 입력을 홀딩 유지
            optimal_v = self.v_min
            optimal_delta = self.last_u[1] * 0.5
            self.get_logger().warn('MPC 최적화 실패: 제약 조건 내 수렴 실패')

        # 자전거 모델 수식을 기반으로 차량의 각속도(w) 최종 변환
        angular_velocity = optimal_v * math.tan(optimal_delta) / self.wheelbase

        cmd = Twist()
        cmd.linear.x = float(optimal_v)
        cmd.angular.z = float(angular_velocity)
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'MPC -> Opt_Steer:{math.degrees(optimal_delta):+4.1f}° | Opt_V:{optimal_v:.2f} m/s | w:{angular_velocity:.2f}'
        )

    def _watchdog(self):
        dt = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if dt > self.timeout:
            self._publish_stop()

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = MPCPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('\n노드 종료 중...\n')
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()