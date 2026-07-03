# This node uses Ultralytics YOLO, which is licensed under AGPL-3.0.
# https://github.com/ultralytics/ultralytics

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

# --------------- 설정값 ---------------
MODEL_PATH   = '/home/wego/hyunyoung_ws/obstacle.engine'  # TensorRT 엔진
DEVICE       = 0                     # Jetson GPU
CONF_THRES   = 0.5                   # 신뢰도 임계값
IMGSZ        = 640
SUB_TOPIC    = '/camera/color/image_raw'
PUB_DET      = '/detections'
PUB_IMG      = '/yolo/result'
PUBLISH_DEBUG = True                 # 디버그 이미지 발행 여부
# ------------------------------------

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        self.model = YOLO(MODEL_PATH)
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, SUB_TOPIC, self.image_callback, qos_profile_sensor_data
        )
        self.pub_det = self.create_publisher(Detection2DArray, PUB_DET, 10)
        if PUBLISH_DEBUG:
            self.pub_img = self.create_publisher(Image, PUB_IMG, 10)

        self.get_logger().info('YOLO Detect Node 시작됨')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 추론
        results = self.model(cv_image, conf=CONF_THRES, imgsz=IMGSZ,
                             device=DEVICE, verbose=False)
        result = results[0]

        # Detection2DArray 구성
        det_array = Detection2DArray()
        det_array.header = msg.header

        if result.boxes is not None:
            for box, cls_id, conf in zip(
                result.boxes.xywh,    # 중심x, 중심y, 폭, 높이
                result.boxes.cls,
                result.boxes.conf
            ):
                cls_name = self.model.names[int(cls_id)]

                det = Detection2D()
                det.header = msg.header

                cx, cy, w, h = map(float, box)
                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = w
                det.bbox.size_y = h

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = cls_name
                hyp.hypothesis.score = float(conf)
                det.results.append(hyp)

                det_array.detections.append(det)

        self.pub_det.publish(det_array)

        # 디버그 이미지
        if PUBLISH_DEBUG:
            annotated = result.plot()
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header = msg.header
            self.pub_img.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()