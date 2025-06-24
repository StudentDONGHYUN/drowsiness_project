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

    def judge(self, results, ruler_value, image_shape):
        self.frame_counter += 1
        now = time.time()
        print_log = False
        # if self.frame_counter % 30 == 0:
        #     print_log = True
        # if print_log:
        #     print(f"[JudgmentEngine] judge() 동작 (frame {self.frame_counter})")
        if ruler_value == 0:
            # 캘리브레이션 미완료 시 바로 반환
            return cfg.DRIVER_STATUS_CALIBRATING

        # 1. 휴대폰 사용 여부 판정
        phone_status = self.check_phone_usage(results, ruler_value, image_shape, print_log)
        if phone_status != cfg.DRIVER_STATUS_NORMAL:
            return phone_status
        # 2. 졸음 여부 판정
        drowsy_status = self.check_drowsiness(results, print_log)
        if drowsy_status != cfg.DRIVER_STATUS_NORMAL:
            return drowsy_status
        # 3. 곁눈질(집중력 저하) 여부 판정
        distraction_status = self.check_distraction(results, print_log)
        if distraction_status != cfg.DRIVER_STATUS_NORMAL:
            return distraction_status
        # 4. 모두 정상일 때
        return cfg.DRIVER_STATUS_NORMAL

    def check_phone_usage(self, results, ruler_value, image_shape, print_log=False):
        """
        손, 휴대폰, 시선, 머리 각도를 종합하여 휴대폰 사용 여부를 정밀하게 판단.
        (실제 구현: 손목-휴대폰 중심 거리, 시선 추정, 통화 판정)
        """
        object_results = results.get('object')
        hand_results = results.get('hands')
        face_results = results.get('face_landmarker')
        # 모든 디버그/print/log 비활성화
        # if print_log:
        #     print(f"[JudgmentEngine] check_phone_usage: object={type(object_results)}, hands={type(hand_results)}, face_mesh={type(face_results)}")
        if not (object_results and hand_results and hand_results.hand_landmarks):
            # 필수 데이터가 없으면 정상 상태로 간주
            return cfg.DRIVER_STATUS_NORMAL

        # 2. 휴대폰 박스 추출 및 별도 출력
        phone_boxes = []
        for det in getattr(object_results, 'detections', []):
            if det.categories and det.categories[0].category_name.lower() in ['cell phone', 'mobile phone', 'phone']:
                bbox = det.bounding_box
                x1, y1, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                x2, y2 = x1 + w, y1 + h
                phone_boxes.append((x1, y1, x2, y2))
                print(f"[ObjectDetector] PHONE DETECTED: bbox=({x1},{y1},{x2},{y2})")
        # if print_log:
        #     print(f"  - [check_phone_usage] phone_boxes: {len(phone_boxes)}")
        if not phone_boxes:
            # 휴대폰 미탐지 시 정상
            return cfg.DRIVER_STATUS_NORMAL
        x1, y1, x2, y2 = phone_boxes[0]
        h_img, w_img, _ = image_shape
        # bbox 중심을 모델 좌표계로 변환
        phone_center = type('Landmark', (), {'x': ((x1 + x2) / 2) / w_img, 'y': ((y1 + y2) / 2) / h_img, 'z': 0})()

        # 3. 손목-휴대폰 중심 거리 계산 (개인화된 자 기준)
        for hand_landmarks in hand_results.hand_landmarks:
            wrist = hand_landmarks[0]  # 0번: 손목
            dist = geometry.calculate_model_unit_distance(wrist, phone_center)
            # if print_log:
            #     print(f"    - wrist-phone dist: {dist:.4f} (ruler={ruler_value})")
            if dist < (ruler_value * cfg.HAND_PHONE_PROXIMITY_RATIO):
                # 4. 시선 추정 (정밀)
                is_gazing = False
                if face_results:
                    is_gazing = geometry.estimate_gaze_intersection(face_results, [phone_boxes[0]], None, image_shape, cfg)
                print(f"[ObjectDetector] is_gazing: {is_gazing}")
                if is_gazing:
                    # 시선이 휴대폰을 향하면 PHONE_VIEW
                    print(f"[ObjectDetector] 판정: DRIVER_STATUS_PHONE_VIEW")
                    return cfg.DRIVER_STATUS_PHONE_VIEW
                # 5. 통화 판정 (휴대폰-귀 거리, 간단화)
                if face_results and face_results.face_landmarks:
                    face_lm = face_results.face_landmarks[0]
                    right_ear = face_lm[4]
                    left_ear = face_lm[234] if len(face_lm) > 234 else None
                    # 더 가까운 귀를 선택
                    ear = right_ear if geometry.calculate_model_unit_distance(wrist, right_ear) < (geometry.calculate_model_unit_distance(wrist, left_ear) if left_ear else 1e9) else left_ear
                    if ear and geometry.calculate_model_unit_distance(wrist, ear) < (ruler_value * 0.3):
                        print(f"[ObjectDetector] 판정: DRIVER_STATUS_PHONE_CALL")
                        return cfg.DRIVER_STATUS_PHONE_CALL
                # 귀 근처만 접근 시 PHONE_CALL
                print(f"[ObjectDetector] 판정: DRIVER_STATUS_PHONE_CALL (근접만)")
                return cfg.DRIVER_STATUS_PHONE_CALL
        return cfg.DRIVER_STATUS_NORMAL
    
    def check_drowsiness(self, results, print_log=False):
        """
        Blendshapes와 얼굴 변환 행렬을 이용해 졸음 상태를 판단.
        - 눈 감김: 일정 프레임 이상 지속 시 졸음
        - 고개 숙임: 변환 행렬의 pitch 각도 기준
        """
        face_results = results.get('face_landmarker')
        if print_log:
            print(f"[JudgmentEngine] check_drowsiness: face_mesh={type(face_results)}")
        if not face_results:
            # 얼굴 결과 없으면 정상
            return cfg.DRIVER_STATUS_NORMAL
        # blendshapes, transformation matrixes 실제 값/길이 출력
        if print_log:
            print(f"    - blendshapes: {face_results.face_blendshapes}")
            print(f"    - blendshapes len: {len(face_results.face_blendshapes) if face_results.face_blendshapes else 0}")
            print(f"    - transformation_matrixes: {face_results.facial_transformation_matrixes}")
            print(f"    - transformation_matrixes len: {len(face_results.facial_transformation_matrixes) if face_results.facial_transformation_matrixes else 0}")
        if face_results.face_blendshapes:
            eye_blink_left = self.get_blendshape_score(face_results.face_blendshapes, 'eyeBlinkLeft')
            eye_blink_right = self.get_blendshape_score(face_results.face_blendshapes, 'eyeBlinkRight')
            if print_log:
                print(f"    - eye_blink_left: {eye_blink_left:.3f}, eye_blink_right: {eye_blink_right:.3f}")
            # 양쪽 눈이 모두 감겼을 때 카운터 증가
            if eye_blink_left > cfg.EYE_BLINK_THRESHOLD and eye_blink_right > cfg.EYE_BLINK_THRESHOLD:
                self.drowsy_counter += 1
            else:
                self.drowsy_counter = 0
            if print_log:
                print(f"    - drowsy_counter: {self.drowsy_counter}")
            # 일정 프레임 이상 눈 감김 지속 시 졸음 판정
            if self.drowsy_counter > cfg.DROWSY_FRAME_LIMIT:
                if print_log:
                    print("    - 졸음 판정: 눈 감김 지속")
                return cfg.DRIVER_STATUS_DROWSY_BLINK
        if face_results.facial_transformation_matrixes:
            matrix = face_results.facial_transformation_matrixes[0]
            _, pitch, _ = geometry.get_angles_from_transformation_matrix(matrix)
            if print_log:
                print(f"    - pitch: {pitch:.2f}")
            # pitch(고개 숙임 각도) 임계값 초과 시 졸음(고개 숙임) 판정
            if pitch > cfg.HEAD_PITCH_THRESHOLD:
                if print_log:
                    print("    - 졸음 판정: 고개 숙임")
                return cfg.DRIVER_STATUS_DROWSY_NOD
        return cfg.DRIVER_STATUS_NORMAL

    def check_distraction(self, results, print_log=False):
        """
        얼굴 변환 행렬로 곁눈질(심한 고개 돌림) 판단.
        """
        face_results = results.get('face_landmarker')
        if print_log:
            print(f"[JudgmentEngine] check_distraction: face_mesh={type(face_results)}")
        if face_results and face_results.facial_transformation_matrixes:
            matrix = face_results.facial_transformation_matrixes[0]
            yaw, pitch, _ = geometry.get_angles_from_transformation_matrix(matrix)
            if print_log:
                print(f"    - yaw: {yaw:.2f}, pitch: {pitch:.2f}")
            # yaw(좌우 회전 각도) 임계값 초과 시, pitch가 정상 범위면 전방주시로 인정
            if abs(yaw) > cfg.HEAD_YAW_THRESHOLD:
                if cfg.HEAD_PITCH_DISTRACT_MIN <= pitch <= cfg.HEAD_PITCH_DISTRACT_MAX:
                    if print_log:
                        print(f"    - yaw 초과지만 pitch({pitch:.2f})가 정상 범위({cfg.HEAD_PITCH_DISTRACT_MIN}~{cfg.HEAD_PITCH_DISTRACT_MAX})이므로 전방주시로 인정")
                    return cfg.DRIVER_STATUS_NORMAL
                else:
                    if print_log:
                        print("    - 곁눈질 판정: yaw 초과 & pitch도 비정상")
                    return cfg.DRIVER_STATUS_DISTRACTED_GAZE
        return cfg.DRIVER_STATUS_NORMAL 