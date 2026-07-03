#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_perception_node.py
- 고정 ROI(화면 왼쪽 신호등 영역)에서 HSV로 빨강/초록 판단
- "빨강 먼저 확인 → 초록 전환" 시 /traffic_light 에 'Green' 발행
- 초록 확정되면 스스로 카메라 구독 해제 (출발 후 연산 0)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

# ============================================================
#                  튜닝 파라미터
# ============================================================
SUB_IMAGE_TOPIC = '/camera/color/image_raw'
PUB_TOPIC       = '/traffic_light'

# --- ROI (화면 비율 0~1, 신호등이 있는 왼쪽 영역) ---
ROI_X_MIN = 0.02
ROI_X_MAX = 0.22
ROI_Y_MIN = 0.30
ROI_Y_MAX = 0.58

# --- HSV 색 범위 ---
# 빨강 (H가 0 근처 + 180 근처 양쪽)
RED_LOWER1 = (0, 80, 80)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (160, 80, 80)
RED_UPPER2 = (180, 255, 255)
# 초록
GREEN_LOWER = (40, 40, 40)   # 밝기(V) 임계 낮게: 초록이 약하게 빛나도 잡히게
GREEN_UPPER = (90, 255, 255)

# --- 판단 ---
COLOR_RATIO_THRES = 0.03     # ROI 대비 해당 색 픽셀 비율 임계
RED_CONFIRM       = 3        # 빨강 연속 프레임 (빨강 상태 확정)
GREEN_CONFIRM     = 3        # 초록 연속 프레임 (초록 확정 → 출발)

PUBLISH_DEBUG     = True      # 디버그 이미지 발행 여부
DEBUG_TOPIC       = '/traffic_light/debug'
# ============================================================


class TrafficLightNode(Node):
    def __init__(self):
        super().__init__('traffic_light_perception_node')
        self.bridge = CvBridge()

        self.red_seen = False        # 빨강을 먼저 봤는가
        self.red_count = 0
        self.green_count = 0
        self.done = False

        self.image_sub = self.create_subscription(
            Image, SUB_IMAGE_TOPIC, self.image_callback, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(String, PUB_TOPIC, 10)
        if PUBLISH_DEBUG:
            self.pub_debug = self.create_publisher(Image, DEBUG_TOPIC, 10)

        self.get_logger().info('신호등 인식 노드 시작 (빨강 대기 중)')

    def image_callback(self, msg):
        if self.done:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = img.shape[:2]

        # ROI 자르기
        x1, x2 = int(w * ROI_X_MIN), int(w * ROI_X_MAX)
        y1, y2 = int(h * ROI_Y_MIN), int(h * ROI_Y_MAX)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        # 빨강 비율
        red_mask = (cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
                    + cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
        red_ratio = cv2.countNonZero(red_mask) / total

        # 초록 비율
        green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        green_ratio = cv2.countNonZero(green_mask) / total

        # --- 상태 판단 ---
        # 1단계: 빨강 먼저 확인
        if not self.red_seen:
            if red_ratio > COLOR_RATIO_THRES:
                self.red_count += 1
                if self.red_count >= RED_CONFIRM:
                    self.red_seen = True
                    self.get_logger().info('🔴 빨강 확인 → 초록 전환 대기')
            else:
                self.red_count = 0
        # 2단계: 빨강 본 뒤 초록 전환 감지
        else:
            if green_ratio > COLOR_RATIO_THRES and green_ratio > red_ratio:
                self.green_count += 1
                if self.green_count >= GREEN_CONFIRM:
                    self.pub.publish(String(data='Green'))
                    self.get_logger().info('🟢 초록 확정 → 출발 신호 발행, 감지 종료')
                    self.done = True
                    self.destroy_subscription(self.image_sub)  # 스스로 종료
            else:
                self.green_count = 0

        # 디버그
        if PUBLISH_DEBUG:
            dbg = img.copy()
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)
            state = 'GREEN-WAIT' if self.red_seen else 'RED-WAIT'
            cv2.putText(dbg, f'{state} R:{red_ratio:.2f} G:{green_ratio:.2f}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            try:
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()