import os
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

def extract_images_from_bag(bag_path, topic_name, output_dir):
    """
    ROS 2 Bag 파일에서 이미지를 추출하여 저장합니다.
    메시지 타입 정의 누락 에러 방지를 위해 기본 Typestore를 주입합니다.
    """
    # 저장할 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[{bag_path}] 파일 읽기를 시작합니다...")
    print(f"타겟 토픽: {topic_name}")
    
    frame_count = 0
    saved_count = 0
    
    # [핵심 수정] ROS 2 Humble 버전에 맞는 기본 타입스토어(설명서) 생성
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    # AnyReader에 default_typestore를 명시적으로 전달
    with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
        # 원하는 토픽의 커넥션만 필터링
        connections = [x for x in reader.connections if x.topic == topic_name]
        
        if not connections:
            print(f"경고: Bag 파일 내에 '{topic_name}' 토픽이 존재하지 않거나 데이터가 없습니다!")
            return

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # 메시지 역직렬화
            msg = reader.deserialize(rawdata, connection.msgtype)
            img = None

            # 1. Raw 이미지 처리 (sensor_msgs/Image)
            if connection.msgtype == 'sensor_msgs/msg/Image':
                img = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
                if msg.encoding in ['rgb8', 'RGB8']:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
            # 2. Compressed 이미지 처리 (sensor_msgs/CompressedImage)
            elif connection.msgtype == 'sensor_msgs/msg/CompressedImage':
                np_arr = np.frombuffer(msg.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 이미지 저장 로직
            if img is not None:
                frame_count += 1
                
                # 5장당 1장씩 저장 (Frame Skipping)
                if frame_count % 10 == 0:
                    filename = f"{timestamp}.jpg"
                    save_path = os.path.join(output_dir, filename)
                    
                    cv2.imwrite(save_path, img)
                    saved_count += 1
                
                if frame_count % 100 == 0:
                    print(f"진행 상황: 총 {frame_count} 프레임 스캔 완료 (추출 및 저장됨: {saved_count}장)...")

    print(f"\n[작업 완료] 스캔된 총 프레임: {frame_count}개 | 최종 저장된 이미지: {saved_count}장")
    print(f"저장 위치: '{os.path.abspath(output_dir)}'")

if __name__ == '__main__':
    # ================= 설정 부분 =================
    # 1. 녹화한 rosbag 폴더의 경로 (실행하신 터미널에 맞게 yolo_hard_case_01 로 설정)
    BAG_FILE_PATH = 'yolo_hard_case_01' 
    
    # 2. 추출할 토픽 이름
    TARGET_TOPIC = '/camera/color/image_raw' 
    
    # 3. 이미지가 저장될 결과 폴더 이름
    OUTPUT_DIRECTORY = 'extracted_yolo_dataset'
    # =============================================
    
    extract_images_from_bag(BAG_FILE_PATH, TARGET_TOPIC, OUTPUT_DIRECTORY)