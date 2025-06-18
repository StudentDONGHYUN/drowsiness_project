# src/main.py (여러 비디오 파일을 순차적으로 처리하는 버전)

import cv2
import mediapipe as mp
import numpy as np
import os
import glob # 파일 경로를 다루기 위한 라이브러리
from utils import calculate_ear

# --- 설정 (Configuration) ---
# 1. 경로 설정
VIDEO_SOURCE_DIR = "../data/videos/"  # 동영상들이 있는 폴더
OUTPUT_IMAGE_DIR = "../data/images/"  # 결과 이미지를 저장할 폴더

# 2. 졸음 판단 기준값 설정
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 48 # 48 프레임 (약 2초) 동안 기준값 미만이면 졸음으로 판단

# 3. MediaPipe 설정
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = '../models/face_landmarker_v2_with_blendshapes.task'

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

# --- 메인 로직 ---

# 결과 저장 폴더가 없으면 생성
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# VIDEO_SOURCE_DIR 폴더 안에 있는 모든 .mp4 파일을 찾기
video_files = glob.glob(os.path.join(VIDEO_SOURCE_DIR, "*.mp4"))
video_files.sort() # 파일 이름 순서대로 정렬

print(f"[INFO] {len(video_files)}개의 비디오 파일을 찾았습니다.")

# FaceLandmarker 인스턴스 생성
with FaceLandmarker.create_from_options(options) as landmarker:
    # 찾은 모든 비디오 파일에 대해 순차적으로 처리
    for video_path in video_files:
        print(f"[INFO] 처리 시작: {video_path}")

        # 변수 초기화
        COUNTER = 0

        # 비디오 파일 열기
        vs = cv2.VideoCapture(video_path)

        frame_number = 0
        while vs.isOpened():
            ret, frame = vs.read()
            if not ret:
                break

            frame_number += 1
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(vs.get(cv2.CAP_PROP_POS_MSEC))
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            is_drowsy = False
            # 랜드마크가 감지된 경우
            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    LEFT_EYE_INDICES = [33, 160, 158, 133, 159, 144]
                    RIGHT_EYE_INDICES = [362, 385, 387, 263, 386, 373]

                    left_eye_points = np.array([[int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)] for i in LEFT_EYE_INDICES])
                    right_eye_points = np.array([[int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)] for i in RIGHT_EYE_INDICES])

                    ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0

                    if ear < EAR_THRESHOLD:
                        COUNTER += 1
                        if COUNTER >= EAR_CONSEC_FRAMES:
                            is_drowsy = True
                    else:
                        COUNTER = 0

                    # (시각화) 화면에 EAR 값 표시
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (width - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 졸음 상태일 때만 결과 이미지 저장
            if is_drowsy:
                # (시각화) 알림 텍스트 표시
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 저장할 파일 이름 생성 (예: SGA2101519S0005_frame_00123.jpg)
                base_filename = os.path.basename(video_path) # SGA2101519S0005.mp4
                video_name = os.path.splitext(base_filename)[0] # SGA2101519S0005

                output_filename = f"{video_name}_frame_{frame_number:05d}_alert.jpg"
                output_path = os.path.join(OUTPUT_IMAGE_DIR, output_filename)

                # 이미지 파일로 저장
                cv2.imwrite(output_path, frame)
                print(f"  -> 졸음 감지! {output_filename} 저장됨")

        # 비디오 처리 완료 후 자원 해제
        vs.release()

print("[INFO] 모든 비디오 파일 처리가 완료되었습니다.")
