# FILE: inference/hand_detector.py
# 역할: MediaPipe HandLandmarker 모델을 로드하고 실행.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path, callback):
        """
        모델 파일을 받아 HandLandmarker 객체를 생성하고 초기화합니다.
        LIVE_STREAM 모드로 실행하고, 결과를 비동기적으로 처리할 콜백을 등록합니다.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=callback)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect_async(self, image, timestamp_ms):
        """
        MediaPipe 이미지 객체를 입력받아 손 랜드마크 추론을 비동기적으로 시작합니다.
        """
        self.detector.detect_async(image, timestamp_ms)

    def close(self):
        """
        Landmarker 객체의 리소스를 해제합니다.
        """
        self.detector.close() 