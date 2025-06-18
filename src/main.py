# src/main.py (최신 MediaPipe Tasks API 버전)

import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_ear # utils.py는 그대로 사용

# MediaPipe Tasks API를 위한 설정
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 모델 파일 경로 (.task 파일)
model_path = '../models/face_landmarker_v2_with_blendshapes.task'

# FaceLandmarker 인스턴스 생성
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO) # 비디오 모드로 설정

with FaceLandmarker.create_from_options(options) as landmarker:
    # MediaPipe에서 눈 랜드마크 인덱스 (EAR 계산에 필요한 6개 포인트)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 159, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 386, 373]

    # 졸음 판단 기준값 설정
    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 48
    COUNTER = 0

    # 비디오 파일 열기
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture("../data/videos/test_video.mp4")

    frame_timestamp_ms = 0
    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            print("End of stream, exiting...")
            break

        # 프레임 크기 가져오기
        height, width, _ = frame.shape

        # BGR 이미지를 RGB로 변환하여 MediaPipe Image 객체로 만듦
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 현재 프레임의 타임스탬프 계산
        frame_timestamp_ms = int(vs.get(cv2.CAP_PROP_POS_MSEC))

        # 랜드마크 감지 실행
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # 얼굴 랜드마크가 감지된 경우
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # 눈 좌표 추출 (정규화된 좌표 -> 픽셀 좌표)
                left_eye_points = np.array([[int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)] for i in LEFT_EYE_INDICES])
                right_eye_points = np.array([[int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)] for i in RIGHT_EYE_INDICES])

                # EAR 계산
                leftEAR = calculate_ear(left_eye_points)
                rightEAR = calculate_ear(right_eye_points)
                ear = (leftEAR + rightEAR) / 2.0

                # 눈 윤곽선 그리기
                cv2.drawContours(frame, [cv2.convexHull(left_eye_points)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(right_eye_points)], -1, (0, 255, 0), 1)

                # 졸음 감지 로직
                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0

                # 화면에 EAR 값 표시
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (width - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 결과 프레임을 파일로 저장
        cv2.imwrite("../data/images/output_frame_mediapipe_tasks.jpg", frame)

    # 작업 완료 후 정리
    vs.release()
