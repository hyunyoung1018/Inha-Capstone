import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import json

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String

class Yolov8nNode(Node):
    def __init__(self):
        super().__init__('yolov8n_node')

        self.model = YOLO('/home/wego/hyunyoung_ws/best.engine', task='segment')
        # self.model.to('cuda')  # CPU 사용, GPU 사용 시 'cuda'로 변경 가능
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        # 시각화 이미지
        self.pub_img = self.create_publisher(Image, '/yolo/result', 10)
        # 바운딩박스 + 클래스 (표준 메시지)
        self.pub_det = self.create_publisher(Detection2DArray, '/detections', 10)
        # 세그멘테이션 마스크 폴리곤 (JSON String)
        self.pub_mask = self.create_publisher(String, '/detections/masks', 10)

        self.get_logger().info('YOLOv8n Node 시작됨')
        self.get_logger().info(f'디바이스: {self.model.device}')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, verbose=False)
        result = results[0]

        # 시각화 발행
        annotated = result.plot()
        result_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        result_msg.header = msg.header
        self.pub_img.publish(result_msg)

        # Detection2DArray 구성
        det_array = Detection2DArray()
        det_array.header = msg.header
        masks_data = []

        if result.boxes is not None:
            for i, (box, cls_id, conf) in enumerate(zip(
                result.boxes.xyxy,
                result.boxes.cls,
                result.boxes.conf
            )):
                cls_name = self.model.names[int(cls_id)]

                # 바운딩박스
                det = Detection2D()
                det.header = msg.header
                x1, y1, x2, y2 = map(float, box)
                det.bbox.center.position.x = (x1 + x2) / 2
                det.bbox.center.position.y = (y1 + y2) / 2
                det.bbox.size_x = x2 - x1
                det.bbox.size_y = y2 - y1

                # 클래스 + 신뢰도
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = cls_name
                hyp.hypothesis.score = float(conf)
                det.results.append(hyp)
                det_array.detections.append(det)

                # 마스크 폴리곤
                mask_entry = {
                    'class_name': cls_name,
                    'score': float(conf),
                    'polygon': []
                }
                if result.masks is not None and i < len(result.masks):
                    mask_entry['polygon'] = result.masks.xy[i].tolist()
                masks_data.append(mask_entry)

        self.pub_det.publish(det_array)

        mask_msg = String()
        mask_msg.data = json.dumps(masks_data)
        self.pub_mask.publish(mask_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8nNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()