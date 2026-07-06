#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_planner_node.py
- /stanley/cmd_vel(조향+속도)을 받아 상태에 따라 게이팅 후 /cmd_vel 발행
- 상태: WAITING_GREEN → DRIVING ↔ PERSON_STOP → PERSON_PASS → DRIVING
        DRIVING ↔ CONE_W1 → DRIVING
        DRIVING → TUNNEL → DRIVING
- Cone 미션 (Local Waypoint 기반):
    오도메트리 없이 오직 라이다 기준 상대 좌표 사용.
    최초 2개 감지 시 중앙콘과 측면콘의 오프셋을 기억.
    이후 지속적으로 감지되는 중앙콘 위치에 오프셋을 더해 가상의 W1(빈공간)을 실시간 추종.
"""

import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_CTRL_CMD        = '/stanley/cmd_vel'
SUB_OBSTACLES       = '/obstacles/fused'
SUB_TRAFFIC         = '/traffic_light'
PUB_CMD_VEL         = '/cmd_vel'
PUB_PHASE           = '/behavior/phase'
PUB_WAYPOINT_MARKER = '/behavior/waypoints'  # RViz 웨이포인트 (laser_link 프레임)

# --- 신호등 출발 ---
USE_TRAFFIC_START = False

# --- 사람 정지 ---
PERSON_STOP_DIST  = 0.8
PERSON_WAIT_SEC   = 2.5
PERSON_PASS_SEC   = 5.0

# --- Car 추종 ---
CAR_GATE_LAT      = 0.6
CAR_STOP_DIST     = 0.40
CAR_RESUME_DIST   = 0.55
CAR_MID_DIST      = 0.80
CAR_CRUISE_DIST   = 1.50
CAR_MID_SPEED     = 0.5
CAR_MAX_CAP       = 1.0

# --- Cone 갈림길 (Local Navigation) ---
CONE_ENABLE       = True
CONE_AIM_DIST     = 1.5    # 콘 2개가 이 안에 들어오면 미션 시작 [m]
CONE_PASS_DIST    = 0.2     # 중앙콘이 차 앞 이 거리 이내로 들어오면 통과로 간주 [m]
CONE_TIMEOUT_SEC  = 6.0     # 무한루프 방지 타이머

CONE_AIM_GAIN     = 4.0     # 조향 게인 (angular.z = GAIN * 가상_ly) [1/s]
CONE_AIM_WMAX     = 3.0     # 각속도 제한 [rad/s]
CONE_SPEED_MAX    = 1.0     # 직진(조향 0) 시 최대 속도 [m/s]
CONE_SPEED_MIN    = 0.5     # 최대 조향 시 최소 속도 [m/s]

# --- Tunnel 주행 ---
TUNNEL_ENABLE     = True
TUNNEL_GAIN       = 3.0
TUNNEL_WMAX       = 3.0
TUNNEL_SPEED      = 0.8
TUNNEL_HOLD_SEC   = 1.0
PRE_CAR_FOLLOW_SEC = 4.0    # 터널 종료 직후 PRE_CAR_FOLLOW(=CAR_FOLLOW) 유지 시간 [s]

# --- LAST_CURVE 테스트용 ---
LAST_CURVE_TEST_TIMEOUT_SEC = 10.0  # [테스트용] LAST_CURVE phase 진입 후 이 시간 지나면 강제로 NORMAL 복귀

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

        # Cone Local Waypoint
        self.cone_done = False
        self.cone_offset_x = 0.0  # 중앙콘 대비 W1의 X 오프셋
        self.cone_offset_y = 0.0  # 중앙콘 대비 W1의 Y 오프셋
        self.w1_local = None      # 차량 기준 W1 현재 좌표 (fx, ly)

        # 터널
        self.tunnel_mid_y = 0.0

        # phase 발행용
        self.last_cap = None
        self._cone_was_active = False
        self.last_curve_latched = False
        self._tunnel_was_active = False
        self.pre_car_follow_target = None   # 터널 종료 후 PRE_CAR_FOLLOW 만료 시각 (None=비활성)
        self.last_curve_start_t = None

        self.create_subscription(Twist, SUB_CTRL_CMD, self.cmd_callback, 10)
        self.create_subscription(String, SUB_OBSTACLES, self.obstacle_callback, qos_profile_sensor_data)
        if USE_TRAFFIC_START:
            self.create_subscription(String, SUB_TRAFFIC, self.traffic_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, PUB_CMD_VEL, 10)
        self.phase_pub = self.create_publisher(String, PUB_PHASE, 10)
        self.marker_pub = self.create_publisher(MarkerArray, PUB_WAYPOINT_MARKER, 10)
        
        self.get_logger().info(f'Behavior Planner Started (Local Waypoint Mode)')

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def traffic_callback(self, msg: String):
        if self.green_seen: return
        if msg.data == 'Green' and self.state == 'WAITING_GREEN':
            self.green_seen = True
            self.state = 'DRIVING'
            self.get_logger().info('🟢 Green Light Start Driving')

    def obstacle_callback(self, msg: String):
        try:
            obstacles = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self._update_car_follow(obstacles)

        # 터널 진입
        tunnel_active = any(o.get('class') == 'TunnelActive' for o in obstacles)
        if TUNNEL_ENABLE:
            if tunnel_active and self.state != 'TUNNEL':
                self.state = 'TUNNEL'
                self.timer_target = self.now_sec() + TUNNEL_HOLD_SEC
                self.get_logger().warn('🚇 Tunnel detected → TUNNEL mode')

            if self.state == 'TUNNEL':
                self._update_tunnel_mid(obstacles)
                return

        # 🚧 Cone 미션 (Local) 🚧
        if CONE_ENABLE and not self.cone_done:
            if self.state == 'DRIVING':
                self._trigger_cone_mission(obstacles)
            elif self.state == 'CONE_W1':
                self._update_cone_w1(obstacles)

        # 사람 정지
        if self.state != 'DRIVING':
            return

        for obs in obstacles:
            if obs.get('class') != PERSON_CLASS: continue
            fx = obs.get('forward_x')
            if fx and 0.0 < fx < PERSON_STOP_DIST:
                self.state = 'PERSON_STOP'
                self.timer_target = self.now_sec() + PERSON_WAIT_SEC
                self.get_logger().warn(f'Person {fx:.2f}m → wait')
                break

    def _update_tunnel_mid(self, obstacles):
        left_ly, right_ly = None, None
        for o in obstacles:
            if o.get('class') != 'TunnelWall': continue
            if o.get('side') == 'left': left_ly = o.get('lateral_y')
            elif o.get('side') == 'right': right_ly = o.get('lateral_y')

        if left_ly is not None and right_ly is not None:
            self.tunnel_mid_y = (left_ly + right_ly) / 2.0
        elif left_ly is not None: self.tunnel_mid_y = left_ly - 0.3
        elif right_ly is not None: self.tunnel_mid_y = right_ly + 0.3

    def _trigger_cone_mission(self, obstacles):
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS: continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None and 0.0 < fx <= CONE_AIM_DIST:
                cones.append((math.hypot(fx, ly), fx, ly))

        if len(cones) < 2:
            return

        # 거리순 정렬: [0]이 가장 가까운 중앙콘, [1]이 측면콘
        cones.sort(key=lambda c: c[0])
        c_dist, cx, cy = cones[0]
        s_dist, sx, sy = cones[1]

        # 측면 콘의 정반대 방향(빈 공간)으로 가기 위한 오프셋 기억
        # W1 = 2C - S = C + (C - S)
        self.cone_offset_x = cx - sx
        self.cone_offset_y = cy - sy

        self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)
        self.state = 'CONE_W1'
        self.timer_target = self.now_sec() + CONE_TIMEOUT_SEC
        self.get_logger().warn(f'🚧 Cone Trigger! Offset locked. Aiming W1')

    def _update_cone_w1(self, obstacles):
        """매 프레임마다 중앙 콘의 위치를 다시 찾고 W1 좌표를 최신화"""
        cones = []
        for obs in obstacles:
            if obs.get('class') != CONE_CLASS: continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is not None and ly is not None:
                cones.append((math.hypot(fx, ly), fx, ly))

        if not cones:
            # 콘이 시야에서 아예 사라지면 미션 완료로 간주 (지나쳤다고 판단)
            self.get_logger().info('✅ Cones lost (passed) -> Return to DRIVING')
            self.state = 'DRIVING'
            self.cone_done = True
            self.w1_local = None
            return

        # 가장 가까운 콘(중앙콘)의 위치를 지속적으로 추적
        cones.sort(key=lambda c: c[0])
        c_dist, cx, cy = cones[0]

        if cx < CONE_PASS_DIST:
            self.get_logger().warn('✅ Passed Center Cone! -> Return to Lane')
            self.state = 'DRIVING'
            self.cone_done = True
            self.w1_local = None
        else:
            # 실시간으로 중앙콘 위치에 기억해둔 오프셋을 더해 W1 갱신
            self.w1_local = (cx + self.cone_offset_x, cy + self.cone_offset_y)

    def _update_car_follow(self, obstacles):
        nearest = None
        for obs in obstacles:
            if obs.get('class') != CAR_CLASS: continue
            fx = obs.get('forward_x')
            ly = obs.get('lateral_y')
            if fx is None or ly is None: continue
            if fx <= 0.0 or abs(ly) > CAR_GATE_LAT: continue
            if nearest is None or fx < nearest:
                nearest = fx
        self.car_front_dist = nearest

    def _car_speed_cap(self):
        d = self.car_front_dist
        if d is None:
            self.car_stopped = False
            return None

        if self.car_stopped:
            if d > CAR_RESUME_DIST: self.car_stopped = False
            else: return 0.0
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

    def _publish_waypoint_markers(self):
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        if self.w1_local is None or self.state != 'CONE_W1':
            self.marker_pub.publish(marker_array)
            return

        # W1 마커 (초록색) - 레이저 프레임 기준
        m1 = Marker()
        m1.header.frame_id = 'laser_link'
        m1.header.stamp = current_time
        m1.ns = 'waypoints'
        m1.id = 1
        m1.type = Marker.SPHERE
        m1.action = Marker.ADD
        m1.pose.position.x = float(self.w1_local[0])
        m1.pose.position.y = float(self.w1_local[1])
        m1.pose.position.z = 0.0
        m1.pose.orientation.w = 1.0
        m1.scale.x = 0.4; m1.scale.y = 0.4; m1.scale.z = 0.4
        m1.color.r = 0.0; m1.color.g = 1.0; m1.color.b = 0.0; m1.color.a = 0.8
        marker_array.markers.append(m1)

        self.marker_pub.publish(marker_array)

    def cmd_callback(self, ctrl_msg: Twist):
        t = self.now_sec()
        out = Twist()

        self._publish_phase()
        self._publish_waypoint_markers()

        if self.state == 'WAITING_GREEN':
            self.cmd_pub.publish(out)
            return

        if self.state == 'TUNNEL':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('🚇 Tunnel end (time) → lane following')
                out = self._apply_car_cap(ctrl_msg)
                self.cmd_pub.publish(out)
                return
            w = clamp(TUNNEL_GAIN * self.tunnel_mid_y, -TUNNEL_WMAX, TUNNEL_WMAX)
            out.linear.x = TUNNEL_SPEED
            out.angular.z = w
            out = self._apply_car_cap(out)
            self.cmd_pub.publish(out)
            return

        # ============================================================
        # 🚧 Cone 미션 조향 제어 (W1 추종)
        # ============================================================
        if self.state == 'CONE_W1':
            if t > self.timer_target:
                self.state = 'DRIVING'
                self.cone_done = True
                self.get_logger().warn('🚧 Cone Mission Timeout!')
                out = self._apply_car_cap(ctrl_msg)
                self.cmd_pub.publish(out)
                return

            if self.w1_local is not None:
                # 계산된 W1의 가상 측면 오차(ly)를 향해 조향
                w = clamp(CONE_AIM_GAIN * self.w1_local[1], -CONE_AIM_WMAX, CONE_AIM_WMAX)
                
                # 조향각(w)의 크기에 비례하여 속도를 부드럽게 감속 (직진=MAX, 최대조향=MIN)
                steer_ratio = abs(w) / CONE_AIM_WMAX
                current_speed = CONE_SPEED_MAX - steer_ratio * (CONE_SPEED_MAX - CONE_SPEED_MIN)
                
                out.linear.x = current_speed
                out.angular.z = w
            else:
                out.linear.x = CONE_SPEED_MAX
                out.angular.z = 0.0
            
            self.cmd_pub.publish(out)
            return
        # ============================================================

        if self.state == 'PERSON_STOP':
            if t >= self.timer_target:
                self.state = 'PERSON_PASS'
                self.timer_target = t + PERSON_PASS_SEC
                self.get_logger().info('Person wait done → PASSING')
                out = ctrl_msg
            else:
                out.linear.x, out.angular.z = 0.0, 0.0
            self.cmd_pub.publish(out)
            return

        if self.state == 'PERSON_PASS':
            if t >= self.timer_target:
                self.state = 'DRIVING'
                self.get_logger().info('Normal Driving')
            out = self._apply_car_cap(ctrl_msg)
            self.cmd_pub.publish(out)
            return

        # 일반 주행 (Stanley 기반)
        out = self._apply_car_cap(ctrl_msg)
        self.cmd_pub.publish(out)

    def _publish_phase(self):
        if self.state == 'CONE_W1':
            phase = 'CONE'
            self._cone_was_active = True
        elif self.state == 'TUNNEL':
            phase = 'TUNNEL'
            self._tunnel_was_active = True
        else:
            if self._cone_was_active:
                self.last_curve_latched = True
                self.last_curve_start_t = self.now_sec()
                self._cone_was_active = False
            if self._tunnel_was_active:
                # 터널 종료 순간 → PRE_CAR_FOLLOW 4초 발동 (CAR_FOLLOW와 동일 동작)
                self.pre_car_follow_target = self.now_sec() + PRE_CAR_FOLLOW_SEC
                self._tunnel_was_active = False

            if self.pre_car_follow_target is not None and \
                    self.now_sec() < self.pre_car_follow_target:
                # PRE_CAR_FOLLOW 창: 4초간 무조건 CAR_FOLLOW (앞차 없어도 유지)
                phase = 'CAR_FOLLOW'
            elif self.last_cap is not None:
                # 실제 앞차 추종 → CAR_FOLLOW (PRE_CAR_FOLLOW 창 안팎 동일)
                self.pre_car_follow_target = None
                phase = 'CAR_FOLLOW'
            elif self.last_curve_latched:
                self.pre_car_follow_target = None
                # [테스트용] LAST_CURVE 진입 후 LAST_CURVE_TEST_TIMEOUT_SEC 지나면 강제로 NORMAL 복귀
                if self.now_sec() - self.last_curve_start_t >= LAST_CURVE_TEST_TIMEOUT_SEC:
                    self.last_curve_latched = False
                    self.last_curve_start_t = None
                    # [테스트용] 콘 미션도 다시 트리거 가능하도록 원상 복구
                    self.cone_done = False
                    self.w1_local = None
                    phase = 'NORMAL'
                else:
                    phase = 'LAST_CURVE'
            else:
                # PRE_CAR_FOLLOW 4초 경과 & 앞차 없음 → NORMAL 복귀
                self.pre_car_follow_target = None
                phase = 'NORMAL'

        msg = String()
        msg.data = phase
        self.phase_pub.publish(msg)

    def _apply_car_cap(self, cmd: Twist):
        cap = self._car_speed_cap()
        self.last_cap = cap
        if cap is None:
            if self.car_log_state != 'none':
                self.car_log_state = 'none'
        elif cap == 0.0:
            if self.car_log_state != 'stop':
                self.car_log_state = 'stop'
        else:
            if self.car_log_state != 'cap':
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