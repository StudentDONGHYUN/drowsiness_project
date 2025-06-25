# FILE: inference/hand_detector.py
# 역할: MediaPipe HandLandmarker 모델을 로드하고 실행.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
import mediapipe as mp
import platform

class HandDetector:
    def __init__(self, model_path):
        """
        HandLandmarker 객체를 초기화합니다.
        
        Args:
            model_path: 손 랜드마크 모델 파일 경로
        """
        delegate_option = None
        if platform.system() == "Linux":
            delegate_option = base_options_module.BaseOptions.Delegate.GPU
            
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate_option)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, image):
        """
        이미지에서 손을 검출하고 랜드마크를 추출합니다.
        
        Args:
            image: MediaPipe Image 객체
            
        Returns:
            HandLandmarkerResult 객체. 다음 정보를 포함합니다:
            - hand_landmarks: 이미지 내 정규화된 좌표 (x, y: 0~1 범위, z: 상대적 깊이)
            - hand_world_landmarks: 실제 3D 좌표 (미터 단위)
            - handedness: 손의 좌/우 구분
        """
        return self.detector.detect(image)

    def close(self):
        """
        검출기의 리소스를 해제합니다.
        """
        self.detector.close()

    @property
    def is_running(self):
        """detector의 실행 상태를 반환합니다."""
        return True 