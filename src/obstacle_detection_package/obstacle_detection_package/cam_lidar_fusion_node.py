import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray


# ============================================================
#                  튜닝 파라미터
# ============================================================
# --- 토픽 ---
SUB_YOLO_TOPIC   = '/detections'
SUB_SCAN_TOPIC   = '/scan'
PUB_FUSED_TOPIC  = '/obstacles/fused'
PUB_MARKER_TOPIC = '/obstacles/markers'

# --- 카메라 ---
IMG_W            = 640.0
CAMERA_FOV_DEG   = 66.74

# --- 추적 대상 클래스 (Tunnel 제외) ---
TARGET_CLASSES   = ['Cone', 'Person', 'Box', 'Car']
TUNNEL_CLASS     = 'Tunnel'

# --- 씨앗 트래킹 (모든 클래스 공통) ---
SEED_RADIUS      = 0.35     # 씨앗 근처 탐색 반경 [m]
LOST_FRAMES      = 10       # Lost 판정 프레임 (20Hz * 10 = 0.5초)
MIN_POINTS       = 4        # 클러스터 최소 점 개수
MATCH_DIST       = 0.4      # YOLO 검출 ↔ 기존 트래커 매칭 거리 [m]

# --- 유효 영역 필터 ---
MAX_RANGE        = 2.0      # 이 거리 밖 장애물 무시 [m]
FORWARD_MIN      = 0.1      # 이 값보다 뒤(작으면) 무시 [m] (후방/측면 제거)

# --- 거리 측정 ---
SCAN_WINDOW      = 5        # YOLO 각도 주변 스캔 탐색 폭

# --- 터널 벽 인식 (트래킹 없음, bbox 각도 범위 안 2클러스터) ---
TUNNEL_ANGLE_PAD_DEG = 3.0   # bbox 각도 범위 좌우 여유 [deg]
TUNNEL_MAX_RANGE     = 2.0   # 터널 벽 최대 거리 [m]
TUNNEL_CLUSTER_BREAK = 0.15  # 클러스터 경계 거리 점프 [m]
TUNNEL_MIN_POINTS    = 2     # 벽 클러스터 최소 점 (치우치면 점 몇 개)
TUNNEL_BRIDGE_GAP    = 3     # 무효빔 브리징 허용

# --- 기타 ---
YOLO_TIMEOUT     = 0.3      # YOLO 검출 유효 시간 [s]
LASER_FRAME_ID   = 'laser_link'
PUBLISH_DEBUG    = True     # RViz 마커 발행 on/off
RATE_HZ          = 20.0     # 발행 주기
# ============================================================


class Tracker:
    """장애물 하나의 추적 상태"""
    _next_id = 0

    def __init__(self, class_name, pos):
        self.id = Tracker._next_id
        Tracker._next_id += 1
        self.class_name = class_name
        self.pos = pos          # (fx, fy) 미터
        self.lost_count = 0
        self.source = 'fusion'


class CameraLidarFusionNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_fusion_node')

        self.camera_fov_rad = CAMERA_FOV_DEG * (math.pi / 180.0)
        self.tunnel_pad_rad = TUNNEL_ANGLE_PAD_DEG * (math.pi / 180.0)

        self.latest_scan = None
        self.latest_detections = []      # [(class_name, cx_pixel, bbox_w), ...]
        self.latest_det_time = self.get_clock().now()

        self.trackers = []

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        self.scan_sub = self.create_subscription(LaserScan, SUB_SCAN_TOPIC, self.scan_callback, qos)
        self.yolo_sub = self.create_subscription(Detection2DArray, SUB_YOLO_TOPIC, self.yolo_callback, qos)

        self.fused_pub = self.create_publisher(String, PUB_FUSED_TOPIC, 10)
        if PUBLISH_DEBUG:
            self.marker_pub = self.create_publisher(MarkerArray, PUB_MARKER_TOPIC, 10)

        self.create_timer(1.0 / RATE_HZ, self.fuse_and_publish)
        self.get_logger().info('Camera-LiDAR Fusion 시작 (씨앗 트래킹 + 터널 벽 2클러스터)')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def yolo_callback(self, msg: Detection2DArray):
        dets = []
        for det in msg.detections:
            if not det.results:
                continue
            cls = det.results[0].hypothesis.class_id     # 문자열 클래스명
            cx = det.bbox.center.position.x               # 박스 중심 x (픽셀)
            bw = det.bbox.size_x                          # 박스 폭 (픽셀)
            dets.append((cls, cx, bw))
        self.latest_detections = dets
        self.latest_det_time = self.get_clock().now()

    # ================= 메인 루프 =================
    def fuse_and_publish(self):
        scan = self.latest_scan
        if scan is None:
            return

        age = (self.get_clock().now() - self.latest_det_time).nanoseconds * 1e-9
        detections = self.latest_detections if age < YOLO_TIMEOUT else []

        # 터널 감지: YOLO Tunnel bbox 찾기
        tunnel_det = self._find_tunnel(detections)  # (cx, bw) or None
        tunnel_on = tunnel_det is not None

        # YOLO Car 인식 여부 (behavior 터널 종료용)
        yolo_car_seen = any(cls == 'Car' for (cls, _cx, _bw) in detections)

        # 1) YOLO 검출 → 차량 기준 좌표(m). 터널 중이면 Cone/Box 제외
        yolo_obs = self._yolo_to_positions(detections, scan, tunnel_on)

        # 2) YOLO 검출로 트래커 갱신/생성 (우선권)
        matched = set()
        for cls, pos in yolo_obs:
            tr = self._match_tracker(cls, pos)
            if tr is None:
                tr = Tracker(cls, pos)
                self.trackers.append(tr)
            else:
                tr.pos = pos
            tr.lost_count = 0
            tr.source = 'fusion'
            matched.add(tr.id)

        # 3) YOLO가 못 본 트래커 → 씨앗 근처 클러스터로 추적 유지
        survivors = []
        for tr in self.trackers:
            # 터널 중이면 Cone/Box 트래커 즉시 폐기
            if tunnel_on and tr.class_name in ('Cone', 'Box'):
                continue
            if tr.id in matched:
                survivors.append(tr)
                continue
            new_pos = self._find_cluster_near(scan, tr.pos, SEED_RADIUS)
            if new_pos is not None:
                if new_pos[0] <= FORWARD_MIN:
                    continue
                tr.pos = new_pos
                tr.lost_count = 0
                tr.source = 'lidar'
                survivors.append(tr)
            else:
                tr.lost_count += 1
                if tr.lost_count < LOST_FRAMES:
                    tr.source = 'lidar'
                    survivors.append(tr)
        self.trackers = survivors

        # 4) 발행 (JSON) — 전방 + 2m 이내만
        combined = [{
            'class': tr.class_name,
            'source': tr.source,
            'distance_r': round(math.hypot(tr.pos[0], tr.pos[1]), 3),
            'forward_x': round(tr.pos[0], 3),
            'lateral_y': round(tr.pos[1], 3),
        } for tr in self.trackers
          if tr.pos[0] > FORWARD_MIN and math.hypot(tr.pos[0], tr.pos[1]) <= MAX_RANGE]

        # 5) 터널 벽: bbox 각도 범위 안 라이다를 2클러스터로 → 좌/우 벽
        if tunnel_on:
            walls = self._detect_tunnel_walls(tunnel_det, scan)
            combined.extend(walls)
            combined.append({'class': 'TunnelActive', 'source': 'yolo'})

        # YOLO Car 인식 신호
        if yolo_car_seen:
            combined.append({'class': 'CarDetected', 'source': 'yolo'})

        if not combined:
            return

        out = String()
        out.data = json.dumps(combined)
        self.fused_pub.publish(out)

        if PUBLISH_DEBUG:
            self.publish_rviz_markers(combined)

    # ================= 터널 벽 인식 =================
    def _find_tunnel(self, detections):
        """YOLO 검출 중 Tunnel → (cx, bw) or None (가장 큰 것)"""
        best = None
        for cls, cx, bw in detections:
            if cls == TUNNEL_CLASS:
                if best is None or bw > best[1]:
                    best = (cx, bw)
        return best

    def _detect_tunnel_walls(self, tunnel_det, scan):
        """Tunnel bbox 각도 범위 안 라이다 점을 클러스터링해서
        가장 큰 2개 덩어리를 좌/우 벽으로 인식. 트래킹 없음."""
        cx, bw = tunnel_det
        x_left = cx - bw / 2.0
        x_right = cx + bw / 2.0
        ang_a = -(x_left - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        ang_b = -(x_right - IMG_W / 2.0) * (self.camera_fov_rad / IMG_W)
        ang_lo = min(ang_a, ang_b) - self.tunnel_pad_rad
        ang_hi = max(ang_a, ang_b) + self.tunnel_pad_rad

        amin = scan.angle_min
        ainc = scan.angle_increment
        rmin = scan.range_min

        # 각도 범위 안 유효 점을 각도 순으로 (i, r, angle) 수집
        pts = []
        for i, r in enumerate(scan.ranges):
            angle = amin + i * ainc
            if angle < ang_lo or angle > ang_hi:
                continue
            if not (math.isfinite(r) and rmin < r < TUNNEL_MAX_RANGE):
                continue
            pts.append((i, r, angle))

        if len(pts) < TUNNEL_MIN_POINTS:
            return []

        # 인접 점 거리 점프로 클러스터 분할 (브리징 허용)
        clusters = []
        cur = [pts[0]]
        gap = 0
        for k in range(1, len(pts)):
            prev_r = cur[-1][1]
            r = pts[k][1]
            idx_gap = pts[k][0] - cur[-1][0]
            if abs(r - prev_r) > TUNNEL_CLUSTER_BREAK or idx_gap > (TUNNEL_BRIDGE_GAP + 1):
                clusters.append(cur)
                cur = [pts[k]]
            else:
                cur.append(pts[k])
        clusters.append(cur)

        # 점 개수 충분한 클러스터만
        clusters = [c for c in clusters if len(c) >= TUNNEL_MIN_POINTS]
        if not clusters:
            return []

        # 각 클러스터 대표 위치(평균) 계산
        reps = []
        for c in clusters:
            xs = [p[1] * math.cos(p[2]) for p in c]
            ys = [p[1] * math.sin(p[2]) for p in c]
            fx = sum(xs) / len(xs)
            fy = sum(ys) / len(ys)
            reps.append((fx, fy, len(c)))

        # 점 많은 순 상위 2개를 벽으로
        reps.sort(key=lambda p: p[2], reverse=True)
        top = reps[:2]

        walls = []
        if len(top) == 2:
            # y가 큰 쪽이 왼쪽(+), 작은 쪽이 오른쪽(-)
            top.sort(key=lambda p: p[1], reverse=True)
            sides = ['left', 'right']
        else:
            # 하나만 잡히면 y 부호로 판단
            sides = ['left' if top[0][1] >= 0 else 'right']

        for (fx, fy, _n), side in zip(top, sides):
            walls.append({
                'class': 'TunnelWall',
                'side': side,
                'source': 'lidar',
                'distance_r': round(math.hypot(fx, fy), 3),
                'forward_x': round(fx, 3),
                'lateral_y': round(fy, 3),
            })
        return walls

    # ================= YOLO → 위치 =================
    def _yolo_to_positions(self, detections, scan, tunnel_on=False):
        result = []
        for cls, cx, bw in detections:
            if cls not in TARGET_CLASSES:
                continue
            # 터널 중이면 Cone/Box 무시 (Car/Person만)
            if tunnel_on and cls in ('Cone', 'Box'):
                continue
            angle = -(cx - (IMG_W / 2.0)) * (self.camera_fov_rad / IMG_W)
            distance = self.get_distance_from_scan(angle, scan)
            if distance is not None and distance <= MAX_RANGE:
                result.append((cls, (distance * math.cos(angle),
                                     distance * math.sin(angle))))
        return result

    def _match_tracker(self, cls, pos):
        best, best_d = None, MATCH_DIST
        for tr in self.trackers:
            if tr.class_name != cls:
                continue
            d = math.hypot(tr.pos[0] - pos[0], tr.pos[1] - pos[1])
            if d < best_d:
                best, best_d = tr, d
        return best

    def _find_cluster_near(self, scan, seed, radius):
        sx, sy = seed
        amin = scan.angle_min
        ainc = scan.angle_increment
        rmin = scan.range_min
        rmax = scan.range_max

        xs, ys = [], []
        for i, r in enumerate(scan.ranges):
            if not (math.isfinite(r) and rmin < r < rmax):
                continue
            angle = amin + i * ainc
            px = r * math.cos(angle)
            py = r * math.sin(angle)
            if math.hypot(px - sx, py - sy) <= radius:
                xs.append(px)
                ys.append(py)

        if len(xs) < MIN_POINTS:
            return None
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def get_distance_from_scan(self, target_angle, scan):
        if target_angle < scan.angle_min or target_angle > scan.angle_max:
            return None
        idx = int((target_angle - scan.angle_min) / scan.angle_increment)
        start_idx = max(0, idx - SCAN_WINDOW)
        end_idx = min(len(scan.ranges), idx + SCAN_WINDOW + 1)
        valid = [scan.ranges[i] for i in range(start_idx, end_idx)
                 if scan.range_min < scan.ranges[i] < scan.range_max]
        return min(valid) if valid else None

    # ================= RViz 마커 =================
    def publish_rviz_markers(self, obstacles):
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        delete_all = Marker(); delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        idx = 0
        for obs in obstacles:
            if 'forward_x' not in obs:
                continue
            is_wall = obs.get('class') == 'TunnelWall'
            is_lidar = obs.get('source') == 'lidar'

            cyl = Marker()
            cyl.header.frame_id = LASER_FRAME_ID
            cyl.header.stamp = current_time
            cyl.ns = 'obstacles_shape'; cyl.id = idx * 2
            cyl.type = Marker.CYLINDER; cyl.action = Marker.ADD
            cyl.pose.position.x = float(obs['forward_x'])
            cyl.pose.position.y = float(obs['lateral_y'])
            cyl.pose.position.z = 0.5
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = 0.3; cyl.scale.y = 0.3; cyl.scale.z = 1.0
            if is_wall:
                cyl.color.r = 0.0; cyl.color.g = 0.4; cyl.color.b = 1.0  # 파랑: 벽
            else:
                cyl.color.r = 1.0
                cyl.color.g = 0.5 if is_lidar else 0.0
                cyl.color.b = 0.0
            cyl.color.a = 0.7
            cyl.lifetime.sec = 0; cyl.lifetime.nanosec = 200000000

            text = Marker()
            text.header.frame_id = LASER_FRAME_ID
            text.header.stamp = current_time
            text.ns = 'obstacles_text'; text.id = idx * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
            text.pose.position.x = float(obs['forward_x'])
            text.pose.position.y = float(obs['lateral_y'])
            text.pose.position.z = 1.2
            text.pose.orientation.w = 1.0
            label = obs['class'] + (f"({obs.get('side','')})" if is_wall else f"({obs.get('source','')})")
            text.text = f"{label}\n{obs['distance_r']}m"
            text.scale.z = 0.3
            text.color.r = 1.0; text.color.g = 1.0; text.color.b = 0.0; text.color.a = 1.0
            text.lifetime.sec = 0; text.lifetime.nanosec = 200000000

            marker_array.markers.append(cyl)
            marker_array.markers.append(text)
            idx += 1

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Camera-LiDAR Fusion 종료')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()