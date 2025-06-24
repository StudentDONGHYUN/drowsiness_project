# FILE: utils/visualizer.py
# 역할: 모든 분석 결과(포즈, 손, 얼굴, 객체 등)를 화면에 시각화.
#       최신 MediaPipe Tasks API의 출력(list)을 drawing_utils가 요구하는 protobuf로 변환하는 브릿지 함수 포함.
#       UI 텍스트는 영문 표준, 실무에서 바로 쓸 수 있는 구조.

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import config as cfg

class Visualizer:
    def __init__(self):
        """
        MediaPipe 공식 drawing_utils 및 스타일 객체 초기화.
        - drawing_spec: 랜드마크 점/선의 색상, 두께, 원 크기 등 스타일 지정
        - connection_spec: 연결선 스타일 지정
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.connection_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)

    def _convert_to_protobuf(self, landmarks_list):
        """
        [핵심 브릿지 함수]
        Tasks API가 반환하는 Python list 타입의 랜드마크를,
        drawing_utils가 요구하는 protobuf 타입(NormalizedLandmarkList)으로 변환.
        Args:
            landmarks_list (list): Tasks API의 NormalizedLandmark 객체 리스트
        Returns:
            landmark_pb2.NormalizedLandmarkList: drawing_utils에서 사용할 수 있는 protobuf 객체
        """
        if not landmarks_list:
            return None
        landmark_list_proto = landmark_pb2.NormalizedLandmarkList()
        for landmark_obj in landmarks_list:
            landmark_proto = landmark_pb2.NormalizedLandmark(
                x=landmark_obj.x, y=landmark_obj.y, z=landmark_obj.z
            )
            landmark_list_proto.landmark.append(landmark_proto)
        return landmark_list_proto

    def draw_all(self, image, results, system_info, gaze_debug=None):
        """
        모든 분석 결과(포즈, 손, 객체 등)와 시스템 상태 정보를 이미지에 시각화.
        Args:
            image (np.ndarray): OpenCV BGR 이미지
            results (dict): 각 모델의 추론 결과
            system_info (dict): 시스템/차량/운전자 상태 텍스트
            gaze_debug (dict): 시선 추적 디버그/시각화 정보
        Returns:
            np.ndarray: 시각화가 적용된 이미지
        """
        # 1. Pose 랜드마크
        if results.get('pose') and results['pose'].pose_landmarks:
            pose_landmarks_proto = self._convert_to_protobuf(results['pose'].pose_landmarks[0])
            self.mp_drawing.draw_landmarks(
                image, landmark_list=pose_landmarks_proto,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.connection_spec)

        # 2. Hands 랜드마크 (양손 모두)
        if results.get('hands') and results['hands'].hand_landmarks:
            for hand_landmarks in results['hands'].hand_landmarks:
                hand_landmarks_proto = self._convert_to_protobuf(hand_landmarks)
                self.mp_drawing.draw_landmarks(
                    image, landmark_list=hand_landmarks_proto,
                    connections=self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.connection_spec)

        # 3. ObjectDetector Bounding Box (휴대폰만)
        if results.get('object') and hasattr(results['object'], 'detections'):
            for det in results['object'].detections:
                if det.categories and det.categories[0].category_name.lower() in ['cell phone', 'mobile phone', 'phone']:
                    bbox = det.bounding_box
                    x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    x2, y2 = x1 + w, y1 + h
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    # 휴대폰 박스 시각화
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(image, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # 3-1. PoseDetector 기반 얼굴 ROI Bounding Box 시각화
        if results.get('face_roi_bbox'):
            x, y, w, h = results['face_roi_bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(image, "FaceROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        # 3-2. PoseDetector 상반신 ROI Bounding Box 시각화 (파란색)
        if results.get('pose_roi_bbox'):
            x, y, w, h = results['pose_roi_bbox']
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, "PoseROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 2-1. Face Mesh(얼굴 랜드마크) 시각화
        if results.get('face_landmarker') and hasattr(results['face_landmarker'], 'face_landmarks') and results['face_landmarker'].face_landmarks:
            face_landmarks_list = results['face_landmarker'].face_landmarks
            for face_landmarks in face_landmarks_list:
                # protobuf 변환
                face_landmarks_proto = self._convert_to_protobuf(face_landmarks)
                # 얼굴 전체 메쉬(삼각형) 시각화 - 공식 스타일 적용
                if hasattr(mp.solutions.face_mesh, 'FACEMESH_TESSELATION'):
                    self.mp_drawing.draw_landmarks(
                        image,
                        landmark_list=face_landmarks_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                # 얼굴 윤곽선 시각화 - 공식 스타일 적용
                if hasattr(mp.solutions.face_mesh, 'FACEMESH_CONTOURS'):
                    self.mp_drawing.draw_landmarks(
                        image,
                        landmark_list=face_landmarks_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
                    )
                # 양쪽 눈동자(iris) landmark 및 연결선 시각화 - 공식 스타일 적용
                if hasattr(mp.solutions.face_mesh, 'FACEMESH_IRISES'):
                    self.mp_drawing.draw_landmarks(
                        image,
                        landmark_list=face_landmarks_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

        # 4. 시스템 상태 텍스트 (영문)
        system_text = f"System: {system_info['system_state']}"
        vehicle_text = f"Vehicle: {system_info['vehicle_status']}"
        status_text = f"Driver Status: {system_info['driver_status']}"
        # System
        cv2.putText(image, system_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, system_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Vehicle
        cv2.putText(image, vehicle_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, vehicle_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Driver Status
        cv2.putText(image, status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(image, status_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # FPS 표시
        if 'fps' in system_info and system_info['fps'] > 0:
            fps_text = f"FPS: {system_info['fps']:.1f}"
            cv2.putText(image, fps_text, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(image, fps_text, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # --- 시선 추적 디버그/시각화 ---
        if gaze_debug:
            # 1. iris 중심(좌/우) 및 eye_center
            if 'left_iris' in gaze_debug and 'right_iris' in gaze_debug:
                h, w = image.shape[:2]
                lx, ly = int(gaze_debug['left_iris'][0] * w), int(gaze_debug['left_iris'][1] * h)
                rx, ry = int(gaze_debug['right_iris'][0] * w), int(gaze_debug['right_iris'][1] * h)
                cv2.circle(image, (lx, ly), 4, (255, 128, 0), -1)  # 파란색(좌)
                cv2.circle(image, (rx, ry), 4, (255, 128, 0), -1)  # 파란색(우)
            if 'eye_center' in gaze_debug:
                h, w = image.shape[:2]
                ex, ey = int(gaze_debug['eye_center'][0] * w), int(gaze_debug['eye_center'][1] * h)
                cv2.circle(image, (ex, ey), 6, (0, 0, 255), -1)  # 빨간색(중점)
            # 2. 시선 벡터(eye_center에서 직선)
            if 'gaze_dir' in gaze_debug and 'eye_center' in gaze_debug:
                h, w = image.shape[:2]
                ex, ey = int(gaze_debug['eye_center'][0] * w), int(gaze_debug['eye_center'][1] * h)
                dx, dy = gaze_debug['gaze_dir'][0], gaze_debug['gaze_dir'][1]
                scale = 200  # 시선 벡터 길이(픽셀)
                x2, y2 = int(ex + dx * scale), int(ey + dy * scale)
                cv2.arrowedLine(image, (ex, ey), (x2, y2), (0, 0, 255), 2, tipLength=0.2)
            # 3. 휴대폰 박스 중심
            if 'phone_center' in gaze_debug:
                h, w = image.shape[:2]
                px, py = int(gaze_debug['phone_center'][0] * w), int(gaze_debug['phone_center'][1] * h)
                cv2.circle(image, (px, py), 6, (0, 255, 0), -1)  # 초록색
            # 4. is_gazing 결과 텍스트
            if 'is_gazing' in gaze_debug:
                txt = f"is_gazing: {gaze_debug['is_gazing']}"
                cv2.putText(image, txt, (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if gaze_debug['is_gazing'] else (0, 0, 255), 2, cv2.LINE_AA)

        return image 