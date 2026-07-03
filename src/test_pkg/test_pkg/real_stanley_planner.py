#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseArray, Twist


# ============================================================
#        튜닝 파라미터 (첫 번째 코드에서 통째로 이식)
# ============================================================
IMG_WIDTH            = 640
IMG_HEIGHT           = 480

# --- Stanley 게인 (첫 번째 코드 값 그대로) ---
STEER_K              = 0.001    # cross track error(픽셀) 게인
YAW_K                = 1.0      # heading error(yaw) 게인
MAX_STEER            = 0.6     # 최대 조향각 [rad]

# --- 조향 스무딩 (저역통과 필터) ---
STEER_SMOOTHING_ALPHA = 0.35    # 변화량 반영 비율 (작을수록 부드러움)

# --- 조향 변화량 기반 감속 ---
STEER_SLOWDOWN_RATIO  = 0.4    # 급조향 시 감속 강도
MIN_SMOOTH_SPEED      = 0.45    # 감속 하한 [m/s]

# --- 속도 / 차량 ---
BASE_SPEED            = 1.0     # 기본 속도 [m/s]
WHEELBASE             = 0.23    # 휠베이스 [m]

# --- 안전 ---
INPUT_TIMEOUT         = 0.5     # 입력 끊김 정지 [s]
# ============================================================


class StanleyPlannerNode(Node):
    def __init__(self):
        super().__init__('stanley_planner_node')

        self.sub_topic = self.declare_parameter('sub_points_topic', '/lane/target_points').value
        self.cmd_topic = self.declare_parameter('cmd_vel_topic', '/stanley/cmd_vel').value

        self.img_w = IMG_WIDTH
        self.img_h = IMG_HEIGHT
        self.center_x = self.img_w / 2.0

        # 제어 상태
        self.prev_steer = None
        self.steer = 0.0
        self.cmd_speed = 0.0
        self.angular_velocity = 0.0

        self.sub = self.create_subscription(
            PoseArray, self.sub_topic,
            self.points_callback, qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self._last_msg_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog)

        self.get_logger().info('Stanley Planner 시작됨 (파라미터 통째 이식)')

    # -----------------------------------------------
    # 중점들에서 yaw(기울기) + error(횡오차) 추출
    # -----------------------------------------------
    def extract_yaw_error(self, poses):
        """
        target_points(중점들)을 직선 피팅해서
        - yaw: 중심선 기울기 (heading error)
        - error: 차량 중심 기준 횡방향 오차 (픽셀)
        를 첫 번째 코드와 동일한 정의로 계산
        """
        xs = np.array([p.position.x for p in poses])
        ys = np.array([p.position.y for p in poses])

        # x = a*y + b 피팅 (첫 번째 코드 fit 방식과 동일: y기준 x)
        a, b = np.polyfit(ys, xs, 1)

        # 평가 높이: 화면 아래쪽(가까운 곳) 기준
        # 첫 번째 코드: y_eval = h * 0.9
        y_eval = self.img_h * 0.2
        x_center = a * y_eval + b

        # yaw = arctan(기울기)  (첫 번째 코드와 동일)
        yaw = math.atan(a)

        # error = -x_center + img_center_x  (첫 번째 코드와 동일)
        error = -x_center + self.center_x

        return yaw, error

    # -----------------------------------------------
    # 첫 번째 코드 cal_steering 통째 이식
    # -----------------------------------------------
    def cal_steering(self, yaw, error):
        base_speed = BASE_SPEED
        wheelbase = WHEELBASE

        # Stanley 제어기로 조향각 계산
        steering_angle = (
            YAW_K * yaw
            + np.arctan2(STEER_K * error, max(abs(base_speed), 0.01))
        )

        # 조향각 제한
        raw_steering_angle = float(
            np.clip(steering_angle, -MAX_STEER, MAX_STEER)
        )

        # 조향 스무딩 (저역통과 필터)
        if self.prev_steer is None:
            steering_delta = 0.0
            steering_angle = raw_steering_angle
        else:
            steering_delta = raw_steering_angle - self.prev_steer
            alpha = float(np.clip(STEER_SMOOTHING_ALPHA, 0.0, 1.0))
            steering_angle = self.prev_steer + alpha * steering_delta
            steering_angle = float(
                np.clip(steering_angle, -MAX_STEER, MAX_STEER)
            )

        # 조향 변화량 기반 감속
        steer_change_ratio = min(
            abs(steering_delta) / max(abs(MAX_STEER), 0.01), 1.0
        )
        speed_scale = 1.0 - STEER_SLOWDOWN_RATIO * steer_change_ratio
        base_speed = max(base_speed * speed_scale, MIN_SMOOTH_SPEED)

        # 조향각 → 각속도 변환
        angular_velocity = base_speed * np.tan(steering_angle) / wheelbase

        self.steer = steering_angle
        self.prev_steer = steering_angle
        self.cmd_speed = float(base_speed)
        self.angular_velocity = float(angular_velocity)

        # 발행
        msg = Twist()
        msg.linear.x = float(base_speed)
        msg.angular.z = float(angular_velocity)
        self.cmd_pub.publish(msg)

    def points_callback(self, msg: PoseArray):
        self._last_msg_time = self.get_clock().now()

        # 중점이 2개 미만이면 피팅 불가
        if len(msg.poses) < 2:
            self._publish_stop()
            return

        yaw, error = self.extract_yaw_error(msg.poses)
        self.cal_steering(yaw, error)

        self.get_logger().info(
            f'yaw(deg): {math.degrees(yaw):+5.1f} | '
            f'err(px): {error:+6.1f} | '
            f'steer(deg): {math.degrees(self.steer):+5.1f} | '
            f'v: {self.cmd_speed:.2f} | w: {self.angular_velocity:+.2f}'
        )

    def _watchdog(self):
        dt = (self.get_clock().now() - self._last_msg_time).nanoseconds * 1e-9
        if dt > INPUT_TIMEOUT:
            self._publish_stop()

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = StanleyPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('종료 중...')
    finally:
        node._publish_stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()