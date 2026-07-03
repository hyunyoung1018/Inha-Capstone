#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lane_fitting_node.py

[역할]
1. /lane/roi_image (640x180, mono8) 구독
2. 히스토그램 및 슬라이딩 윈도우 기반 차선 픽셀 추출
3. 2차 곡선(Polynomial) 피팅 (x = ay^2 + by + c)
4. 차선이 1개만 보일 경우 차선 폭(Offset)을 이용해 가상의 중앙 궤적 추정
5. 스탠리 제어기를 위한 BEV 좌표계 기준 목표 궤적(Target Points) 발행

- 슬라이딩 윈도우 최대 꺾임 각도(Max Shift) 제한 적용
- 중복 차선 방지 및 예외 처리 강화 (안정적인 Yellow Line 보장)

lane_fitting_node.py
- [NEW] 색상 값(100=Left, 200=Right) 기반 절대적 좌우 분류 (완벽 분리)
- 불필요한 기울기 검사 및 영역 판단 로직 제거
"""

import cv2
import numpy as np
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge

class LaneFittingNode(Node):
    def __init__(self):
        super().__init__('lane_fitting_node')

        self.sub_roi_topic    = self.declare_parameter('sub_roi_topic', '/lane/roi_image').value
        self.pub_points_topic = self.declare_parameter('pub_points_topic', '/lane/target_points').value
        self.pub_debug_topic  = self.declare_parameter('pub_debug_topic', '/lane/debug_fitting').value

        self.img_w       = self.declare_parameter('image_width', 640).value
        self.roi_h       = self.declare_parameter('roi_height', 180).value
        self.roi_cut_top = self.declare_parameter('roi_cut_top', 300).value 

        self.nwindows = self.declare_parameter('nwindows', 9).value
        self.margin   = self.declare_parameter('margin', 60).value          
        self.minpix   = self.declare_parameter('minpix', 20).value          
        self.max_shift = self.declare_parameter('max_shift_px', 40).value
        
        self.lane_width = self.declare_parameter('lane_width', 240.0).value 
        self.frame_id = self.declare_parameter('frame_id', 'bev').value

        self.bridge = CvBridge()
        qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        self.sub        = self.create_subscription(Image, self.sub_roi_topic, self.image_callback, qos)
        self.points_pub = self.create_publisher(PoseArray, self.pub_points_topic, qos)
        self.debug_pub  = self.create_publisher(Image, self.pub_debug_topic, qos)

    def image_callback(self, msg: Image):
        try:
            binary_warped = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception: return

        # 좌/우 차선 픽셀을 완전히 분리
        left_mask = (binary_warped == 100).astype(np.uint8) * 255
        right_mask = (binary_warped == 200).astype(np.uint8) * 255

        # 각각의 마스크에서 히스토그램을 구해 확실한 베이스라인 탐색
        left_hist = np.sum(left_mask[left_mask.shape[0]//2:, :], axis=0)
        right_hist = np.sum(right_mask[right_mask.shape[0]//2:, :], axis=0)
        
        leftx_base = np.argmax(left_hist)
        rightx_base = np.argmax(right_hist)

        min_hist_threshold = 10
        left_valid = left_hist[leftx_base] > min_hist_threshold
        right_valid = right_hist[rightx_base] > min_hist_threshold

        # 디버그용 이미지 (픽셀들을 흰색으로 표시)
        out_img = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
        out_img[left_mask > 0] = [255, 255, 255]
        out_img[right_mask > 0] = [255, 255, 255]

        leftx_current = leftx_base
        rightx_current = rightx_base
        window_height = int(binary_warped.shape[0] // self.nwindows)
        
        left_lane_inds = []
        right_lane_inds = []

        # 미리 좌표 인덱스 추출 (각 마스크별로 독립적으로!)
        left_nonzeroy, left_nonzerox = left_mask.nonzero()
        right_nonzeroy, right_nonzerox = right_mask.nonzero()

        for window in range(self.nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low, win_xleft_high = leftx_current - self.margin, leftx_current + self.margin
            win_xright_low, win_xright_high = rightx_current - self.margin, rightx_current + self.margin

            if left_valid:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                good_left_inds = ((left_nonzeroy >= win_y_low) & (left_nonzeroy < win_y_high) & 
                                  (left_nonzerox >= win_xleft_low) & (left_nonzerox < win_xleft_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                
                if len(good_left_inds) > self.minpix:
                    new_x = int(np.mean(left_nonzerox[good_left_inds]))
                    if abs(new_x - leftx_current) > self.max_shift:
                        leftx_current = leftx_current + int(math.copysign(self.max_shift, new_x - leftx_current))
                    else:
                        leftx_current = new_x

            if right_valid:
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                good_right_inds = ((right_nonzeroy >= win_y_low) & (right_nonzeroy < win_y_high) & 
                                   (right_nonzerox >= win_xright_low) & (right_nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                
                if len(good_right_inds) > self.minpix:
                    new_x = int(np.mean(right_nonzerox[good_right_inds]))
                    if abs(new_x - rightx_current) > self.max_shift:
                        rightx_current = rightx_current + int(math.copysign(self.max_shift, new_x - rightx_current))
                    else:
                        rightx_current = new_x

        if left_valid: left_lane_inds = np.concatenate(left_lane_inds)
        if right_valid: right_lane_inds = np.concatenate(right_lane_inds)

        # 수집된 차선은 디버그 색상 정상화 (왼쪽: 빨강, 오른쪽: 파랑)
        if left_valid and len(left_lane_inds) > 0:
            out_img[left_nonzeroy[left_lane_inds], left_nonzerox[left_lane_inds]] = [0, 0, 255] 
        if right_valid and len(right_lane_inds) > 0:
            out_img[right_nonzeroy[right_lane_inds], right_nonzerox[right_lane_inds]] = [255, 0, 0] 

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        center_fitx = None
        left_fitx_vals = None
        right_fitx_vals = None

        if left_valid and len(left_lane_inds) > 50:
            leftx, lefty = left_nonzerox[left_lane_inds], left_nonzeroy[left_lane_inds]
            # 1차 피팅 테스트용
            # left_fit = np.polyfit(lefty, leftx, 1)
            # left_fitx_vals = left_fit[0] * ploty + left_fit[1]
            #2차 피팅
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fitx_vals = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

        if right_valid and len(right_lane_inds) > 50:
            rightx, righty = right_nonzerox[right_lane_inds], right_nonzeroy[right_lane_inds]
            # 1차 피팅
            # right_fit = np.polyfit(righty, rightx, 1)
            # right_fitx_vals = right_fit[0] * ploty + right_fit[1]
            # 2차 피팅
            right_fit = np.polyfit(righty, rightx, 2)
            right_fitx_vals = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # 완벽하게 분리된 좌우 차선 기반 중앙 궤적
        if left_fitx_vals is not None and right_fitx_vals is not None:
            center_fitx = (left_fitx_vals + right_fitx_vals) / 2.0
        elif left_fitx_vals is not None and right_fitx_vals is None:
            center_fitx = left_fitx_vals + (self.lane_width / 2.0)
        elif right_fitx_vals is not None and left_fitx_vals is None:
            center_fitx = right_fitx_vals - (self.lane_width / 2.0)

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.frame_id

        if center_fitx is not None:
            for y_idx in range(len(ploty) - 1, 0, -10):
                target_x = center_fitx[y_idx]
                target_y = ploty[y_idx]

                if -100 <= target_x <= self.img_w + 100:
                    p = Pose()
                    p.position.x = float(target_x)
                    p.position.y = float(target_y + self.roi_cut_top)
                    pose_array.poses.append(p)

                    if 0 <= target_x < self.img_w:
                        cv2.circle(out_img, (int(target_x), int(target_y)), 3, (0, 255, 255), -1)

            if left_fitx_vals is not None:
                pts_left = np.array([np.transpose(np.vstack([left_fitx_vals, ploty]))])
                cv2.polylines(out_img, np.int_([pts_left]), isClosed=False, color=(255, 0, 255), thickness=2)
            if right_fitx_vals is not None:
                pts_right = np.array([np.transpose(np.vstack([right_fitx_vals, ploty]))])
                cv2.polylines(out_img, np.int_([pts_right]), isClosed=False, color=(255, 0, 255), thickness=2)

        self.points_pub.publish(pose_array)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(out_img, encoding='bgr8'))

def main(args=None):
    rclpy.init(args=args)
    node = LaneFittingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()