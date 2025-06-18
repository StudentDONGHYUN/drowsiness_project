# src/video_thread.py
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import mediapipe as mp
import numpy as np
from detection_logic import calculate_ear, get_head_pose
import os
import time

class VideoThread(QThread):
    # 시그널 정의: 처리된 프레임(numpy.ndarray)과 상태 메시지(str)를 전달
    update_frame = pyqtSignal(np.ndarray, str)

    def __init__(self, source):
        super().__init__()
        self.source = source # 비디오 파일 경로 또는 웹캠 ID
        self.running = True

    def run(self):
        # MediaPipe 설정
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path = '../models/face_landmarker_v2_with_blendshapes.task'

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=VisionRunningMode.VIDEO)

        with FaceLandmarker.create_from_options(options) as landmarker:
            vs = cv2.VideoCapture(self.source)

            # 카운터 초기화
            drowsy_counter = 0
            inattention_counter = 0

            while self.running and vs.isOpened():
                ret, frame = vs.read()
                if not ret:
                    self.update_frame.emit(None, "비디오의 끝에 도달했습니다.")
                    break

                height, width, _ = frame.shape

                # MediaPipe 처리
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                frame_timestamp_ms = int(vs.get(cv2.CAP_PROP_POS_MSEC))
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                status_text = "정상"

                if detection_result.face_landmarks:
                    face_landmarks_list = detection_result.face_landmarks[0]
                    face_landmarks_for_drawing = np.array([[int(lm.x * width), int(lm.y * height)] for lm in face_landmarks_list])

                    # 1. 졸음 감지 (EAR)
                    LEFT_EYE = [33, 160, 158, 133, 159, 144]
                    RIGHT_EYE = [362, 385, 387, 263, 386, 373]
                    left_pts = np.array([[face_landmarks_list[i].x * width, face_landmarks_list[i].y * height] for i in LEFT_EYE])
                    right_pts = np.array([[face_landmarks_list[i].x * width, face_landmarks_list[i].y * height] for i in RIGHT_EYE])
                    ear = (calculate_ear(left_pts) + calculate_ear(right_pts)) / 2.0

                    if ear < 0.25:
                        drowsy_counter += 1
                        if drowsy_counter > 48:
                            status_text = "경고: 졸음 감지!"
                            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.save_frame(frame, "drowsiness")
                    else:
                        drowsy_counter = 0

                    # 2. 부주의 감지 (머리 방향)
                    head_angles = get_head_pose(face_landmarks_for_drawing, frame.shape)
                    if abs(head_angles[1]) > 25: # y축 회전(좌우)이 25도 이상일 때
                        inattention_counter += 1
                        if inattention_counter > 30:
                            status_text = "경고: 전방 주시 태만!"
                            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.save_frame(frame, "inattention")
                    else:
                        inattention_counter = 0

                    # 랜드마크 그리기 (시각화)
                    cv2.drawContours(frame, [cv2.convexHull(left_pts)], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [cv2.convexHull(right_pts)], -1, (0, 255, 0), 1)

                # 처리된 프레임과 상태 메시지를 GUI로 전송
                self.update_frame.emit(frame, status_text)
                time.sleep(0.01) # 다른 스레드에 CPU 점유를 양보

            vs.release()

    def save_frame(self, frame, event_type):
        output_dir = "../data/images/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}{event_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
