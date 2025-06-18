# src/video_thread.py (예측 랜드마크 시각화 최종 완전판)
from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import mediapipe as mp
import numpy as np
from detection_logic import (
    calculate_ear,
    get_head_pose,
    analyze_torso_twist,
    analyze_posture,
    is_hand_near_face,
)
import os
import time
from PIL import Image, ImageDraw, ImageFont

# MediaPipe 그리기 유틸리티와 Holistic 모델 임포트
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def draw_text(frame, text, pos, color, font_path, font_size):
    """Pillow를 사용하여 프레임에 한글 텍스트를 그리는 함수"""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        # 폰트 로드 실패 시 OpenCV 기본 폰트로 대체
        cv2.putText(
            frame, "Font Error", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        return frame


def custom_draw_landmarks(image, landmark_list, connections, connection_drawing_spec):
    """
    신뢰도가 낮은 랜드마크도 연결하여 '예측된' 뼈대를 그려주는 함수
    """
    if not landmark_list:
        return

    h, w, _ = image.shape
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        # 랜드마크가 화면 밖으로 벗어나지 않았을 때만 좌표 저장
        if landmark.x * w > 0 and landmark.y * h > 0:
            plotted_landmarks[idx] = (int(landmark.x * w), int(landmark.y * h))

    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                continue

            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                cv2.line(
                    image,
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                    connection_drawing_spec.color,
                    connection_drawing_spec.thickness,
                )


class VideoThread(QThread):
    update_frame = pyqtSignal(np.ndarray, str)

    def __init__(self, source, enable_segmentation=False, flip_horizontally=False):
        super().__init__()
        self.source = source
        self.running = True
        self.enable_segmentation = enable_segmentation
        self.flip_horizontally = flip_horizontally

    def run(self):
        # --- 설정부 (Configuration) ---
        FONT_PATH = "../assets/fonts/D2Coding-Ver1.3.2-20180524.ttf"
        FONT_SIZE = 24

        # 임계값 설정
        EAR_THRESHOLD, EAR_CONSEC_FRAMES = 0.25, 48
        INATTENTION_SCORE_THRESHOLD, INATTENTION_CONSEC_FRAMES = 100, 30
        POSTURE_THRESHOLD, POSTURE_CONSEC_FRAMES = 15, 60

        # 부주의 점수 가중치
        (
            HEAD_YAW_SCORE,
            HEAD_PITCH_SCORE,
            TORSO_TWIST_SCORE,
            POSTURE_SCORE,
            HAND_NEAR_FACE_SCORE,
        ) = 40, 30, 60, 25, 50

        # --- Holistic 모델 초기화 ---
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=self.enable_segmentation,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            vs = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)

            drowsy_counter, inattention_counter, posture_counter = 0, 0, 0

            while self.running and vs.isOpened():
                ret, frame = vs.read()
                if not ret:
                    self.update_frame.emit(
                        np.zeros((600, 800, 3), dtype=np.uint8),
                        "비디오의 끝에 도달했습니다.",
                    )
                    break

                if self.source == 0 and self.flip_horizontally:
                    frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = holistic.process(rgb_frame)
                rgb_frame.flags.writeable = True

                annotated_image = frame.copy()
                if self.enable_segmentation and results.segmentation_mask is not None:
                    condition = (
                        np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                    )
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = (192, 192, 192)  # 회색 배경
                    annotated_image = np.where(condition, annotated_image, bg_image)

                status_text = "정상 운행"
                inattention_score = 0

                face_lm = (
                    np.array(
                        [
                            [int(lm.x * width), int(lm.y * height)]
                            for lm in results.face_landmarks.landmark
                        ],
                        dtype=np.int32,
                    )
                    if results.face_landmarks
                    else None
                )
                pose_lm = (
                    np.array(
                        [
                            [int(lm.x * width), int(lm.y * height)]
                            for lm in results.pose_landmarks.landmark
                        ],
                        dtype=np.int32,
                    )
                    if results.pose_landmarks
                    else None
                )
                left_hand_lm = (
                    np.array(
                        [
                            [int(lm.x * width), int(lm.y * height)]
                            for lm in results.left_hand_landmarks.landmark
                        ],
                        dtype=np.int32,
                    )
                    if results.left_hand_landmarks
                    else None
                )
                right_hand_lm = (
                    np.array(
                        [
                            [int(lm.x * width), int(lm.y * height)]
                            for lm in results.right_hand_landmarks.landmark
                        ],
                        dtype=np.int32,
                    )
                    if results.right_hand_landmarks
                    else None
                )

                if face_lm is not None:
                    LEFT_EYE, RIGHT_EYE = (
                        [33, 160, 158, 133, 159, 144],
                        [362, 385, 387, 263, 386, 373],
                    )
                    ear = (
                        calculate_ear(face_lm[LEFT_EYE])
                        + calculate_ear(face_lm[RIGHT_EYE])
                    ) / 2.0
                    if ear < EAR_THRESHOLD:
                        drowsy_counter += 1
                        if drowsy_counter >= EAR_CONSEC_FRAMES:
                            status_text = "경고: 졸음 감지!"
                            self.save_frame(annotated_image, "drowsiness")
                    else:
                        drowsy_counter = 0

                    head_angles = get_head_pose(face_lm, frame.shape)
                    if abs(head_angles[1]) > 25:
                        inattention_score += HEAD_YAW_SCORE
                    if head_angles[0] < -10:
                        inattention_score += HEAD_PITCH_SCORE
                    if (
                        pose_lm is not None
                        and analyze_torso_twist(face_lm, pose_lm) > 20
                    ):
                        inattention_score += TORSO_TWIST_SCORE
                    if left_hand_lm is not None and is_hand_near_face(
                        left_hand_lm, face_lm
                    ):
                        inattention_score += HAND_NEAR_FACE_SCORE
                    if right_hand_lm is not None and is_hand_near_face(
                        right_hand_lm, face_lm
                    ):
                        inattention_score += HAND_NEAR_FACE_SCORE

                if pose_lm is not None:
                    if analyze_posture(pose_lm) > POSTURE_THRESHOLD:
                        posture_counter += 1
                        if posture_counter >= POSTURE_CONSEC_FRAMES:
                            status_text = "경고: 자세 불량!"
                            self.save_frame(annotated_image, "slouching")
                    else:
                        posture_counter = 0

                if inattention_score >= INATTENTION_SCORE_THRESHOLD:
                    inattention_counter += 1
                    if inattention_counter >= INATTENTION_CONSEC_FRAMES:
                        status_text = "경고: 복합적 부주의 운전!"
                        self.save_frame(annotated_image, "complex_inattention")
                else:
                    inattention_counter = 0

                # 시각화 그리기
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                custom_draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(121, 44, 80), thickness=2
                    ),  # <--- 단순한 DrawingSpec 객체로 변경
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                )

                if status_text != "정상 운행":
                    annotated_image = draw_text(
                        annotated_image,
                        status_text,
                        (10, 30),
                        (0, 0, 255),
                        FONT_PATH,
                        24,
                    )
                annotated_image = draw_text(
                    annotated_image,
                    f"Inattention Score: {inattention_score}",
                    (width - 250, 30),
                    (255, 255, 0),
                    FONT_PATH,
                    20,
                )
                if face_lm is not None:
                    annotated_image = draw_text(
                        annotated_image,
                        f"EAR: {ear:.2f}",
                        (width - 250, 60),
                        (255, 255, 0),
                        FONT_PATH,
                        20,
                    )

                self.update_frame.emit(annotated_image, status_text)
                time.sleep(0.01)
            vs.release()

    def save_frame(self, frame, event_type):
        output_dir = "../data/images/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(output_dir, f"{event_type}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        time.sleep(1)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
