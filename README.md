# Inha-Capstone

ROS2 (Humble) 기반 자율주행 소형 로봇 워크스페이스. 카메라+YOLO+LiDAR로 장애물을 인식하고, Stanley 방식 차선추종과 상태머신으로 주행을 제어합니다.

## 패키지 구성

```
src/
├── decision_making_package/
│   ├── stanley_follow.py       # 차선 인식 + Stanley 조향 → /stanley/cmd_vel
│   └── state_machine_node.py   # 장애물/터널/신호 상태머신 → /cmd_vel (최종 출력)
└── obstacle_detection_package/
    ├── yolov8n_node.py                   # 카메라 → YOLO(TensorRT) → /detections
    ├── cam_lidar_fusion_node.py          # /detections + /scan 융합 → /obstacles/fused
    └── traffic_light_perception_node.py  # 신호등 색 인식 → /traffic_light
```

launch 파일은 없으며, 각 노드를 `ros2 run`으로 개별 실행합니다.

```
ros2 run obstacle_detection_package yolov8n_node
ros2 run obstacle_detection_package cam_lidar_fusion_node
ros2 run obstacle_detection_package traffic_light_perception_node
ros2 run decision_making_package stanley_follow
ros2 run decision_making_package state_machine
```

## 데이터 흐름

```
camera → yolov8n_node ─(/detections)─→ cam_lidar_fusion_node ─(/obstacles/fused)─┬→ stanley_follow (Box 회피)
                                              (+ /scan)                            └→ state_machine (Person/Car/Cone/Tunnel 처리)
traffic_light_perception_node ─(/traffic_light)──────────────────────────────────→ state_machine (출발 게이트)
stanley_follow ─(/stanley/cmd_vel)───────────────────────────────────────────────→ state_machine → /cmd_vel
```

## stanley_follow.py — 차선 인식 파이프라인

1. `warpping` — 원근 변환으로 BEV(Bird's-Eye View) 생성 (좌우 대칭 보정된 `src_points` 사용)
2. `roi_set` — BEV 하단부만 ROI로 사용
3. `white_color_filter_hsv` + `tophat_filter` — HSV 흰색 필터와 top-hat 모폴로지를 AND 결합해, 바닥 반사광 같은 넓은 밝은 영역은 제거하고 얇은 차선만 남김
4. `sliding_window` — 좌/우 차선을 슬라이딩 윈도우로 검출, 상황별로 상태 분기:
   - `both`: 두 차선 모두 정상 검출 (간격 ≥ `narrow_both_gap_px`)
   - `narrow_both_{left,right}`: 두 차선이 검출됐지만 간격이 비정상적으로 좁을 때 — 기울기로 좌/우를 재판정해 단일 차선 추적으로 전환
   - `tracked_{left,right}_only`: 한쪽 차선만 검출되어 추적
   - `previous` / `default`: 검출 실패 시 이전 값 또는 기본값 사용
5. Stanley 컨트롤러로 조향각 계산 → `/stanley/cmd_vel` 발행
6. 디버그 토픽: `/roi_img`, `/binary_img`, `/debugging_image1`(슬라이딩 윈도우), `/debugging_image2`(최종 차선/조향 오버레이)

## state_machine_node.py — 상태머신

`/stanley/cmd_vel`을 받아 상태에 따라 게이팅/오버라이드 후 `/cmd_vel`로 재발행합니다.

- **Person**: 근접 시 정지 → 대기 → 통과
- **Car**: 상태 전환 없이 거리 기반 속도 캡 (터널 중에도 유지)
- **Cone (갈림길)**: `CONE_AIM`(가운데 콘 조준 접근) → `CONE_PUSH`(열린 쪽으로 강하게 조향) → `DRIVING` 복귀
- **Tunnel**: `TunnelActive` 검출 즉시 최우선으로 진입, 좌/우 벽(`TunnelWall`) 중점을 추종하며 주행, 일정 시간 후 종료
- **Traffic Light**: 활성화 시 `'Green'` 수신 전까지 출발 대기

## obstacle_detection_package

- `cam_lidar_fusion_node.py`: YOLO 검출과 LiDAR를 융합해 클래스별(`Cone`, `Person`, `Box`, `Car`) 씨앗 기반 추적을 유지하고, `Tunnel` 검출 시에는 별도로 bbox 각도 범위 내 LiDAR 포인트를 클러스터링해 좌/우 벽(`TunnelWall`)을 인식합니다.
