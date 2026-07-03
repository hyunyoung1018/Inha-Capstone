#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_extractor_node.py  (순수 차선 추출 - 판단 없음)

역할: 지정된 클래스(기본 'Lane')의 폴리곤을 면으로 채워 BEV+ROI 후
      좌(100)/우(200) 엣지를 뽑아 /lane/roi_image (mono8) 로 발행한다.

- 어느 도로/브랜치를 뽑을지 '판단'은 하지 않는다.
  뽑을 클래스는 /driving/target_branch (std_msgs/String) 로 외부(behavior_planner)가 지정한다.
    'NONE' 또는 빈 문자열 -> 'Lane'
    'Branch_L' / 'Branch_R' -> 해당 클래스
- 실제 장애물(obstacle_classes)만 마스킹해서 그 위로 차선이 그려지지 않게 한다.
  (Branch / Cone 은 마스킹하지 않는다.)
"""

import json
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ============ 영상처리 함수 ============
def warpping(image, src_mat, dst_mat):
    h, w = image.shape[:2]
    M = cv2.getPerspectiveTransform(src_mat, dst_mat)
    return cv2.warpPerspective(image, M, (w, h))

def bird_convert(img, src_mat, dst_mat):
    return warpping(img, np.float32(src_mat), np.float32(dst_mat))

def draw_lane_mask(polygons, img_h, img_w, color=255):
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], color=color)
    return canvas

def keep_largest_contours(image, max_count=1, min_area=400):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    clean_mask = np.zeros_like(image)
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            count += 1
            if count >= max_count:
                break
    return clean_mask


# ============================== Node ==============================
class LaneExtractorNode(Node):
    def __init__(self):
        super().__init__('lane_extractor_node')

        self.sub_mask_topic   = self.declare_parameter('sub_mask_topic', '/detections/masks').value
        self.pub_roi_topic    = self.declare_parameter('pub_roi_topic', '/lane/roi_image').value
        # 뽑을 클래스를 외부에서 지정받는 토픽 (behavior_planner 가 발행)
        self.sub_target_topic = self.declare_parameter('sub_target_branch', '/driving/target_branch').value

        self.lane_class = self.declare_parameter('lane_class_name', 'Lane').value
        # 차선 위에 그려지면 안 되는 실제 장애물 클래스들 (Branch/Cone 은 제외)
        self.obstacle_classes = list(self.declare_parameter(
            'obstacle_classes', ['Car', 'Box', 'person']).value)

        self.img_w          = self.declare_parameter('image_width', 640).value
        self.img_h          = self.declare_parameter('image_height', 480).value
        self.roi_cut_top    = self.declare_parameter('roi_cut_top', 300).value
        self.roi_cut_bottom = self.declare_parameter('roi_cut_bottom', 0).value
        self.edge_thickness = self.declare_parameter('edge_thickness', 7).value

        # 현재 뽑을 클래스 (외부 지정, 기본은 Lane)
        self.target_class = self.lane_class

        self.src_points = np.float32([[215.0, 250.0], [435.0, 250.0], [0.0, 445.0], [636.0, 445.0]])
        self.dst_points = np.float32([[240.0, 230.0], [400.0, 230.0], [240.0, 470.0], [400.0, 470.0]])

        self.safe_mask_roi = self._create_safe_zone_mask()
        self.bridge = CvBridge()
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.sub        = self.create_subscription(String, self.sub_mask_topic, self.masks_callback, qos)
        self.sub_target = self.create_subscription(String, self.sub_target_topic, self.target_callback, 10)
        self.roi_pub    = self.create_publisher(Image, self.pub_roi_topic, qos)
        self.get_logger().info('차선 추출기 가동 (순수 인지, 뽑을 클래스=%s)' % self.target_class)

    def _create_safe_zone_mask(self):
        white_canvas = np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255
        M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        bev_footprint = cv2.warpPerspective(white_canvas, M, (self.img_w, self.img_h))
        kernel = np.ones((15, 15), np.uint8)
        safe_mask_full = cv2.erode(bev_footprint, kernel, iterations=1)
        return safe_mask_full[self.roi_cut_top : self.img_h - self.roi_cut_bottom, :]

    def target_callback(self, msg: String):
        cls = msg.data
        # 'NONE'/빈 문자열이면 기본 차선(Lane)을 뽑는다
        self.target_class = self.lane_class if (not cls or cls == 'NONE') else cls

    def masks_callback(self, msg: String):
        try:
            masks = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        target = self.target_class

        best_polygon = None
        best_score = -1.0
        obstacle_polygons = []
        obstacle_boxes = []

        for m in masks:
            cls = m.get('class_name')
            if not cls:
                continue

            # 1) 뽑을 대상 클래스 -> 최고 신뢰도 폴리곤 1개 선택
            if cls == target:
                if m.get('polygon'):
                    score = float(m.get('confidence', m.get('score', 0.0)))
                    if score > best_score:
                        best_score = score
                        best_polygon = m['polygon']

            # 2) 실제 장애물 클래스 -> 마스킹용 수집
            elif cls in self.obstacle_classes:
                if m.get('polygon'):
                    obstacle_polygons.append(m['polygon'])
                elif 'box2d' in m or 'bbox' in m:
                    obstacle_boxes.append(m.get('box2d', m.get('bbox')))
            # 그 외(다른 Branch, Cone 등)는 무시

        lane_polygons = [best_polygon] if best_polygon else []

        # ===================== 장애물 마스크 =====================
        obstacle_canvas = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        for poly in obstacle_polygons:
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(obstacle_canvas, [pts], color=255)
        for box in obstacle_boxes:
            if len(box) >= 4:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(obstacle_canvas, (x1, y1), (x2, y2), 255, -1)

        obstacle_bev = bird_convert(obstacle_canvas, self.src_points, self.dst_points)
        obstacle_roi = obstacle_bev[self.roi_cut_top : self.img_h - self.roi_cut_bottom, :]
        kernel_dilate = np.ones((31, 31), np.uint8)
        obstacle_roi_dilated = cv2.dilate(obstacle_roi, kernel_dilate, iterations=1)
        valid_safe_zone = cv2.bitwise_not(obstacle_roi_dilated)
        final_safe_mask = cv2.bitwise_and(self.safe_mask_roi, valid_safe_zone)

        # ===================== 차선(도로) 면 -> 테두리 =====================
        filled_image = draw_lane_mask(lane_polygons, self.img_h, self.img_w, color=255)
        bird_image = bird_convert(filled_image, self.src_points, self.dst_points)
        roi_image = bird_image[self.roi_cut_top : self.img_h - self.roi_cut_bottom, :]

        kernel_open = np.ones((5, 5), np.uint8)
        roi_opened = cv2.morphologyEx(roi_image, cv2.MORPH_OPEN, kernel_open)
        roi_cleaned = keep_largest_contours(roi_opened, max_count=1, min_area=500)

        h, w = roi_cleaned.shape
        t = self.edge_thickness

        M_left = np.float32([[1, 0, -t], [0, 1, 0]])
        shifted_left = cv2.warpAffine(roi_cleaned, M_left, (w, h))
        raw_left_edge = cv2.subtract(shifted_left, roi_cleaned)

        M_right = np.float32([[1, 0, t], [0, 1, 0]])
        shifted_right = cv2.warpAffine(roi_cleaned, M_right, (w, h))
        raw_right_edge = cv2.subtract(shifted_right, roi_cleaned)

        final_left_edge = cv2.bitwise_and(raw_left_edge, raw_left_edge, mask=final_safe_mask)
        final_right_edge = cv2.bitwise_and(raw_right_edge, raw_right_edge, mask=final_safe_mask)

        final_roi = np.zeros_like(roi_cleaned)
        final_roi[final_left_edge > 0] = 100
        final_roi[final_right_edge > 0] = 200

        roi_u8 = cv2.convertScaleAbs(final_roi)
        self.roi_pub.publish(self.bridge.cv2_to_imgmsg(roi_u8, encoding='mono8'))

def main(args=None):
    rclpy.init(args=args)
    node = LaneExtractorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()