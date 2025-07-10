# processing/judgment_engine.py
# 역할: 모든 분석 결과를 종합하여 최종 운전자 상태를 추론하는 규칙 기반 엔진.

from utils import geometry
import config as cfg
import numpy as np
import time

class JudgmentEngine:
    def __init__(self):
        """졸음/하품 상태 지속 카운터 초기화."""
        self.drowsy_counter = 0
        self.yawn_counter = 0
        self.frame_counter = 0  # 프레임 샘플링용 카운터
        self._last_log_time = 0  # 마지막 로그 출력 시각

    def get_blendshape_score(self, blendshapes, category_name):
        """
        Blendshape 결과 리스트에서 특정 카테고리의 점수를 안전하게 반환.
        Args:
            blendshapes (list): MediaPipe Blendshape 결과
            category_name (str): 예) 'eyeBlinkLeft'
        Returns:
            float: 해당 카테고리의 점수(0~1)
        """
        if not blendshapes: return 0.0
        for category in blendshapes[0]:
            if category.category_name == category_name: return category.score
        return 0.0

    def judge(self, results, image_shape):
        """
        모든 감지 결과를 종합하여 최종 운전자 상태를 판단하고,
        상태 머신과 시각화에 사용할 딕셔너리를 반환합니다.
        """
        is_drowsy = False
        is_distracted = False
        status_message = "Normal"

        # 1. 졸음 여부 판정 (눈 감김, 고개 숙임)
        drowsy_status, drowsy_msg = self._check_drowsiness(results)
        if drowsy_status:
            is_drowsy = True
            status_message = drowsy_msg

        # 2. 주의 분산 여부 판정 (휴대폰 사용, 곁눈질)
        # 졸음이 아닐 때만 주의 분산을 확인하여 경고의 우선순위를 정함
        if not is_drowsy:
            distracted_status, distracted_msg = self._check_distraction(results, image_shape)
            if distracted_status:
                is_distracted = True
                status_message = distracted_msg

        return {
            "is_drowsy": is_drowsy,
            "is_distracted": is_distracted,
            "status_message": status_message,
        }

    def _check_phone_usage(self, results, image_shape):
        """
        손과 휴대폰의 근접 여부를 판단합니다.
        캘리브레이션 없이 정규화된 좌표계의 거리를 사용합니다.
        """
        object_results = results.get('object')
        hand_results = results.get('hands')

        if not (object_results and object_results.detections and hand_results and hand_results.hand_landmarks):
            return False

        phone_detections = [det for det in object_results.detections if det.categories and det.categories[0].category_name == 'cell phone']
        if not phone_detections:
            return False

        main_phone = sorted(phone_detections, key=lambda x: x.categories[0].score, reverse=True)[0]
        bbox = main_phone.bounding_box
        h_img, w_img, _ = image_shape
        
        phone_center_x = (bbox.origin_x + bbox.width / 2) / w_img
        phone_center_y = (bbox.origin_y + bbox.height / 2) / h_img
        phone_center = type('Landmark', (), {'x': phone_center_x, 'y': phone_center_y, 'z': 0})()

        for hand_landmarks in hand_results.hand_landmarks:
            wrist = hand_landmarks[0] # 0: Wrist landmark
            dist = geometry.calculate_model_unit_distance(wrist, phone_center)
            
            if dist < cfg.HAND_OBJECT_PROXIMITY_THRESHOLD:
                return True # 손과 휴대폰이 가까움
        
        return False

    def _check_drowsiness(self, results):
        """
        Blendshapes와 얼굴 변환 행렬을 이용해 졸음 상태를 판단.
        - 눈 감김: 일정 프레임 이상 지속 시 졸음
        - 고개 숙임: 변환 행렬의 pitch 각도 기준
        """
        face_results = results.get('face_landmarker')
        if not face_results:
            # Face detection failed, reset counters to prevent false positives
            self.drowsy_counter = 0
            return False, ""

        # 눈 감김 확인
        if face_results.face_blendshapes:
            eye_blink_left = self.get_blendshape_score(face_results.face_blendshapes, 'eyeBlinkLeft')
            eye_blink_right = self.get_blendshape_score(face_results.face_blendshapes, 'eyeBlinkRight')
            
            if eye_blink_left > cfg.EYE_BLINK_THRESHOLD and eye_blink_right > cfg.EYE_BLINK_THRESHOLD:
                self.drowsy_counter += 1
            else:
                self.drowsy_counter = 0
            
            if self.drowsy_counter > cfg.DROWSY_FRAME_LIMIT:
                return True, "WARNING: Eyes Closed"
        else:
            # No blendshapes data available, reset counter
            self.drowsy_counter = 0

        # 고개 숙임 확인
        if face_results.facial_transformation_matrixes:
            matrix = face_results.facial_transformation_matrixes[0]
            _, pitch, _ = geometry.get_angles_from_transformation_matrix(matrix)
            if pitch < cfg.HEAD_PITCH_THRESHOLD: # 음수값이므로 작을수록 숙인 것
                return True, "WARNING: Nodding Off"

        return False, ""

    def _check_distraction(self, results, image_shape):
        """
        휴대폰 사용 또는 곁눈질(심한 고개 돌림)을 판단.
        """
        # 휴대폰 사용 확인
        if self._check_phone_usage(results, image_shape):
            return True, "WARNING: Phone Usage"

        # 곁눈질 확인
        face_results = results.get('face_landmarker')
        if face_results and face_results.facial_transformation_matrixes:
            matrix = face_results.facial_transformation_matrixes[0]
            yaw, _, _ = geometry.get_angles_from_transformation_matrix(matrix)

            if abs(yaw) > cfg.HEAD_YAW_THRESHOLD:
                return True, "CAUTION: Distracted"
        
        return False, "" 