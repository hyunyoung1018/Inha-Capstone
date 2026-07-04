#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_planner_node.py
- /stanley/cmd_vel(조향+속도)을 받아 상태에 따라 게이팅 후 /cmd_vel 발행
- 상태: WAITING_GREEN → DRIVING ↔ PERSON_STOP → PERSON_PASS → DRIVING
        DRIVING ↔ CONE_AIM → CONE_PUSH → DRIVING
        DRIVING → TUNNEL → DRIVING
- Car: 상태 전환 없이 속도만 거리별 캡 (터널 중에도 유지)
- Cone 갈림길 (3단계):
    CONE_AIM  : 콘 2개 감지(≤AIM_DIST) 시 차선 무시, 가운데 콘(ly≈0) 조준하며 접근
    CONE_PUSH : 가운데 콘이 PUSH_DIST 안에 오면 열린 쪽으로 강하게 짧게 조향
    → DRIVING : 정상 차선 주행 복귀
- Tunnel:
    TunnelActive(YOLO Tunnel) 뜨는 순간 즉시 TUNNEL 진입 → Cone/Person 무시(Car만),
    좌/우 벽(TunnelWall) 중점 추종. TUNNEL_HOLD_SEC 뒤 시간으로 종료.
"""

import json
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from std_msgs.msg import String


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_CTRL_CMD   = '/stanley/cmd_vel'
SUB_OBSTACLES  = '/obstacles/fused'
SUB_TRAFFIC    = '/traffic_light'
PUB_CMD_VEL    = '/cmd_vel'

# --- 신호등 출발 ---
USE_TRAFFIC_START = False

# --- 사람 정지 (종방향 거리만) ---
PERSON_STOP_DIST  = 0.8
PERSON_WAIT_SEC   = 2.5
PERSON_PASS_SEC   = 5.0

# --- Car 추종 (속도 캡) ---
CAR_GATE_LAT      = 0.6
CAR_STOP_DIST     = 0.40
CAR_RESUME_DIST   = 0.55
CAR_MID_DIST      = 0.80
CAR_CRUISE_DIST   = 1.50
CAR_MID_SPEED     = 0.5
CAR_MAX_CAP       = 1.0

# --- Cone 갈림길 (3단계) ---
CONE_ENABLE       = True
CONE_MIN_COUNT    = 2       # 최소 콘 개수
CONE_AIM_DIST     = 2.0     # 콘 2개가 이 안에 들어오면 AIM 시작 [m]
CONE_PUSH_DIST    = 1.4     # 가운데 콘이 이 안에 오면 PUSH 전환 [m]
CONE_CENTER_DEAD  = 0.1     # 가운데 콘 판단 제외 |lateral_y| [m] (좌/우 판단용)
CONE_AIM_GAIN     = 4.0     # AIM 조준 게인 (angular.z = GAIN * center_ly) [1/s]
CONE_AIM_WMAX     = 3.0     # AIM 각속도 제한 [rad/s]
CONE_AIM_SPEED    = 0.8     # AIM 접근 속도 [m/s]
CONE_STEER        = 2.5     # PUSH 각속도 크기 [rad/s] (강하게)
CONE_STEER_SEC    = 0.7     # PUSH 지속 시간 [s] (짧게)
CONE_SPEED        = 1.0     # PUSH 속도 [m/s]
# PUSH 이후: GAP(직진) → COUNTER(반대 되틀기)
CONE_GAP_SEC       = 0.4    # PUSH 후 직진 유지 시간 [s]
CONE_GAP_SPEED     = 1.0    # GAP 직진 속도 [m/s]
CONE_COUNTER       = 2.0    # 반대 되틀기 각속도 [rad/s]
CONE_COUNTER_SEC   = 0.5    # 되틀기 지속 시간 [s]
CONE_COUNTER_SPEED = 1.0    # 되틀기 속도 [m/s]

# --- Tunnel 주행 ---
TUNNEL_ENABLE     = True
TUNNEL_GAIN       = 3.0     # 벽 중점 추종 게인 (angular.z = GAIN * mid_y) [1/s]
TUNNEL_WMAX       = 3.0     # 터널 각속도 제한 [rad/s]
TUNNEL_SPEED      = 0.8     # 터널 주행 속도 [m/s]
TUNNEL_HOLD_SEC   = 1.0     # 진입 후 터널 주행 유지 시간 [s] (시간 종료)

# --- Class Name ---
PERSON_CLASS      = 'Person'
CAR_CLASS         = 'Car'
CONE_CLASS        = 'Cone'
# ============================================================


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class BehaviorPlannerNode(Node):
    def __init__(self):
        super().__init__('behavior_planner_node')

        self.state = 'WAITING_GREEN' if USE_TRAFFIC_START else 'DRIVING'
        self.timer_target = 0.0

        # Car
        self.car_front_dist = None
        self.car_stopped = False
        self.car_log_state = 'none'

        # 신호등
        self.green_seen = False

        # 콘
        self.cone_aim_ly = 0.0       # AIM 중 가운데 콘 횡오차 (조준용)
        self.cone_steer_dir = 0.0    # PUSH 각속도 (+왼쪽 / -오른쪽)
        self.cone_done = False       # 이번 콘 구간 처리 완료 래치

        # 터널
        self.tunnel_mid_y = 0.0      # 좌/우 벽 중점 횡오차

        self.create_subscription(Twist, SUB_CTRL_CMD, self.cmd_callback, 10)
        self.create_subscription(String, SUB_OBSTACLES, self.obstacle_callback, qos_profile_sensor_data)
        if USE_TRAFFIC_START:
            self.create_subscription(String, SUB_TRAFFIC, self.traffic_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.get_logger().info(f'Behavior Planner Started (State: {self.state})')

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ============================================================
    #  신호등
    # ============================================================
    def traffic_callback(self, msg: String):
        if self.green_seen:
            return
        if msg.data == 'Green' and self.state == 'WAITING_GREEN':
            self.green_seen = True
            self.state = 'DRIVING'
            self.get_logger().info('🟢 Green Light Start Driving')

    # ============================================================
    #  장애물 콜백
    # ============================================================
    def obstacle_callback(self, msg: String):
        try:
            obstacles = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._update_car_follow(obstacles)

        # ---- 터널 진입 (최우선): TunnelActive 뜨면 즉시 TUNNEL ----
        tunnel_active = any(o.get('class') == 'TunnelActive' for o in obstacles)
        if TUNNEL_ENABLE:
            if tunnel_active and self.state != 'TUNNEL':
                self.state = 'TUNNEL'
                self.timer_target = self.now_sec() + TUNNEL_HOLD_SEC
                self.get_logger().warn('🚇 Tunnel detected → TUNNEL mode (ignore all but Car)')

            if self.state == 'TUNNEL':
                self._update_tunnel_mid(obstacles)
                return   # 터널 중엔 Cone/Person 판단 스킵 (Car는 위에서 갱신됨)

        # 콘 처리 (DRIVING에서 트리거, AIM 중에는 갱신)
        if CONE_ENABLE and self.state in ('DRIVING', 'CONE_AIM'):
            self._check_cone(obstacles)

        # 사람 정지는 DRIVING일 때만
        if self.state != 'DRIVING':
            return

        for obs in obstacles:
            if obs.get('class') != PERSON_CLASS:
                continue
            fx = obs.get('forward_x')
            if fx is None:
                continue
            if 0.0 < fx < PERSON_STOP_DIST:
                self.state = 'PERSON_STOP'
                self.timer_target = self.now_sec() + PERSON_WAIT_SEC
                self.get_logger().warn(f'Person {fx:.2f}m → {PERSON_WAIT_SEC} second wait')
                break

    # ============================================================
    #  터널 벽 중점
    # ============================================================
    def _update_tunnel_mid(self, obstacles):
        left_ly = None
        right_ly = None
        for o in obstacles:
            if o.get('class') != 'TunnelWall':
                continue
            if o.get('side') == 'left':
                left_ly = o.get('lateral_y')
            elif o.get('side') == 'right':
                right_ly = o.get('lateral_y')

        if left_ly is not None and right_ly is not None:
            self.tunnel_mid_y = (left_ly + right_ly) / 2.0
        elif left_ly is not None:
            # 왼쪽 벽만 → 그 벽에서 약간 오른쪽을 목표 (벽 붙음 방지)
            self.tunnel_mid_y = left_ly - 0.3
        elif right_ly is not None:
            self.tunnel_mid_y = right_ly + 0.3
        # 둘 다 없으면 직전 mid 유지

    # ============================================================
    #  Cone: 감지 → AIM(가운데 콘 조준) → PUSH(열린쪽 강하게)
    # ============================================================
    def _check_cone(self, obstacles):
        cones = []   # (fx, ly)
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS:
                continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is None or ly is None:
                continue
            if fx <= 0.0 or fx > CONE_AIM_DIST:
                continue
            cones.append((fx, ly))

        if len(cones) < CONE_MIN_COUNT:
            self.cone_done = False
            if self.state == 'CONE_AIM':
                self.state = 'DRIVING'
                self.get_logger().info('Cone lost during AIM → lane following')
            return

        if self.cone_done:
            return

        center = min(cones, key=lambda c: abs(c[1]))
        center_fx, center_ly = center
        self.cone_aim_ly = center_ly

        if self.state == 'DRIVING':
            self.state = 'CONE_AIM'
            self.get_logger().warn(f'🚧 Cone detected → AIM center cone (ly={center_ly:+.2f})')
            return

        if self.state == 'CONE_AIM' and center_fx <= CONE_PUSH_DIST:
            side_ys = [ly for (fx, ly) in cones if abs(ly) > CONE_CENTER_DEAD]
            if not side_ys:
                return
            mean_y = sum(side_ys) / len(side_ys)
            if mean_y < 0:
                self.cone_steer_dir = +CONE_STEER
                open_side = 'LEFT'
            else:
                self.cone_steer_dir = -CONE_STEER
                open_side = 'RIGHT'

            self.cone_done = True
            self.state = 'CONE_PUSH'
            self.timer_target = self.now_sec() + CONE_STEER_SEC
            self.get_logger().warn(f'🚧 Push → open {open_side} ({CONE_STEER_SEC}s)')

    def _update_car_follow(self, obstacles):
        nearest = None
        for obs in obstacles:
            if obs.get('class') != CAR_CLASS:
                continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is None or ly is None:
                continue
            if fx <= 0.0 or abs(ly) > CAR_GATE_LAT:
                continue
            if nearest is None or fx < nearest:
                nearest = fx
        self.car_front_dist = nearest

    def _car_speed_cap(self):
        d = self.car_front_dist
        if d is None:
            self.car_stopped = False
            return None

        if self.car_stopped:
            if d > CAR_RESUME_DIST:
                self.car_stopped = False
            else:
                return 0.0
        elif d < CAR_STOP_DIST:
            self.car_stopped = True
            return 0.0

        if d < CAR_MID_DIST:
            frac = (d - CAR_STOP_DIST) / max(1e-6, CAR_MID_DIST - CAR_STOP_DIST)
            return max(0.0, frac) * CAR_MID_SPEED
        if d < CAR_CRUISE_DIST:
            frac = (d - CAR_MID_DIST) / max(1e-6, CAR_CRUISE_DIST - CAR_MID_DIST)
            return CAR_MID_SPEED + frac * (CAR_MAX_CAP - CAR_MID_SPEED)
        return None

    # ============================================================
    #  제어 명령 게이팅 (메인 출력)
    # ============================================================
    def cmd_callback(self, ctrl_msg: Twist):
        t = self.now_sec()
        out = Twist()

        if self.state == 'WAITING_GREEN':
            self.cmd_pub.publish(out)
            return

        # ---- 터널 주행: 벽 중점 추종, 시간 종료 ----
        if self.state == 'TUNNEL':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('🚇 Tunnel end (time) → lane following')
                out = ctrl_msg
                out = self._apply_car_cap(out)
                self.cmd_pub.publish(out)
                return
            w = clamp(TUNNEL_GAIN * self.tunnel_mid_y, -TUNNEL_WMAX, TUNNEL_WMAX)
            out.linear.x = TUNNEL_SPEED
            out.angular.z = w
            out = self._apply_car_cap(out)   # 앞차 있으면 속도 캡
            self.cmd_pub.publish(out)
            return

        # ---- 콘 AIM: 차선 무시, 가운데 콘 조준 ----
        if self.state == 'CONE_AIM':
            w = clamp(CONE_AIM_GAIN * self.cone_aim_ly, -CONE_AIM_WMAX, CONE_AIM_WMAX)
            out.linear.x = CONE_AIM_SPEED
            out.angular.z = w
            self.cmd_pub.publish(out)
            return

        # ---- 콘 PUSH: 열린 쪽으로 강하게 짧게 ----
        if self.state == 'CONE_PUSH':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('Cone push done → lane following')
                out = ctrl_msg
                out = self._apply_car_cap(out)
            else:
                out.linear.x = CONE_SPEED
                out.angular.z = self.cone_steer_dir
            self.cmd_pub.publish(out)
            return

        if self.state == 'PERSON_STOP':
            if t >= self.timer_target:
                self.state = 'PERSON_PASS'
                self.timer_target = t + PERSON_PASS_SEC
                self.get_logger().info('Person wait done → PASSING')
                out = ctrl_msg
            else:
                out.linear.x = 0.0
                out.angular.z = 0.0
            self.cmd_pub.publish(out)
            return

        if self.state == 'PERSON_PASS':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('Normal Driving')
            out = ctrl_msg
            out = self._apply_car_cap(out)
            self.cmd_pub.publish(out)
            return

        # DRIVING
        out = ctrl_msg
        out = self._apply_car_cap(out)
        self.cmd_pub.publish(out)

    def _apply_car_cap(self, cmd: Twist):
        cap = self._car_speed_cap()
        d = self.car_front_dist

        if cap is None:
            if self.car_log_state != 'none':
                if d is None:
                    self.get_logger().info('🚗 앞차 벗어남 → 캡 해제, 정상 주행')
                else:
                    self.get_logger().info(f'🚗 앞차 {d:.2f}m (순항) → 캡 해제')
                self.car_log_state = 'none'
        elif cap == 0.0:
            if self.car_log_state != 'stop':
                self.get_logger().warn(f'🛑 앞차 {d:.2f}m → 정지')
                self.car_log_state = 'stop'
        else:
            if self.car_log_state != 'cap':
                self.get_logger().info(f'🚗 앞차 {d:.2f}m → 추종 (속도캡 {cap:.2f})')
                self.car_log_state = 'cap'

        if cap is not None and cmd.linear.x > cap:
            if cmd.linear.x > 1e-3:
                cmd.angular.z *= (cap / cmd.linear.x)
            cmd.linear.x = cap
        return cmd


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()