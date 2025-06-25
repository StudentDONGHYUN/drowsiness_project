# FILE: utils/visualizer.py
# 역할: 모든 분석 결과(포즈, 손, 얼굴, 객체 등)를 화면에 시각화.
#       최신 MediaPipe Tasks API의 출력(list)을 drawing_utils가 요구하는 protobuf로 변환하는 브릿지 함수 포함.
#       UI 텍스트는 영문 표준, 실무에서 바로 쓸 수 있는 구조.

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import FaceLandmarkerResult, PoseLandmarkerResult, ObjectDetectorResult
import config as cfg

# For drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def _convert_to_protobuf(landmarks):
    """Converts a list of NormalizedLandmark objects to a NormalizedLandmarkList protobuf object."""
    if not landmarks:
        return landmark_pb2.NormalizedLandmarkList()
        
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for landmark_data in landmarks:
        landmark_proto = landmark_list.landmark.add()
        landmark_proto.x = landmark_data.x
        landmark_proto.y = landmark_data.y
        landmark_proto.z = landmark_data.z
        if landmark_data.visibility is not None:
            landmark_proto.visibility = landmark_data.visibility
        if landmark_data.presence is not None:
            landmark_proto.presence = landmark_data.presence
    return landmark_list

class Visualizer:
    def __init__(self, width, height):
        self.frame_width = width
        self.frame_height = height
        """
        MediaPipe 공식 drawing_utils 및 스타일 객체 초기화.
        - drawing_spec: 랜드마크 점/선의 색상, 두께, 원 크기 등 스타일 지정
        - connection_spec: 연결선 스타일 지정
        """
        self.mp_drawing = mp_drawing
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.connection_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)

    def draw_face_landmarks(self, image, detection_result: FaceLandmarkerResult):
        if not detection_result or not detection_result.face_landmarks:
            return

        for landmarks in detection_result.face_landmarks:
            face_landmarks_proto = _convert_to_protobuf(landmarks)
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    def draw_pose_landmarks(self, image, detection_result: PoseLandmarkerResult):
        if not detection_result or not detection_result.pose_landmarks:
            return

        for landmarks in detection_result.pose_landmarks:
            pose_landmarks_proto = _convert_to_protobuf(landmarks)
            self.mp_drawing.draw_landmarks(
                image,
                pose_landmarks_proto,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    def draw_object_detections(self, image, detection_result: ObjectDetectorResult):
        if not detection_result or not detection_result.detections:
            return

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

            category = detection.categories[0]
            category_name = category.category_name
            score = round(category.score, 2)
            result_text = f'{category_name} ({score})'
            text_location = (bbox.origin_x + 10, bbox.origin_y + 20)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    def draw_all(self, image, face_result, pose_result, object_result, judgment_results, fps=0.0, system_state=""):
        annotated_image = image.copy()
        self.draw_face_landmarks(annotated_image, face_result)
        self.draw_pose_landmarks(annotated_image, pose_result)
        self.draw_object_detections(annotated_image, object_result)
        
        # --- Display Judgment Status ---
        if judgment_results:
            status_message = judgment_results.get("status_message", "Initializing...")
            
            # Set color based on message prefix
            color = (0, 255, 0) # Green for Normal
            if "CAUTION" in status_message:
                color = (0, 255, 255) # Yellow
            elif "WARNING" in status_message:
                color = (0, 0, 255) # Red

            cv2.putText(annotated_image, status_message, 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # --- Display System Info (FPS and State) ---
        info_text = f"FPS: {fps:.1f} | STATE: {system_state}"
        cv2.putText(annotated_image, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_image 