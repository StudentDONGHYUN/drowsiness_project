# FILE: inference/pose_detector.py
# 역할: MediaPipe Pose 모델을 로드하고 실행. 차량 이동 상태 분석에 필요한 분할 마스크 출력을 활성화.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class PoseDetector:
    def __init__(self, model_path, callback):
        """
        모델 파일을 받아 PoseLandmarker 객체를 생성하고 초기화합니다.
        LIVE_STREAM 모드로 실행하고, 결과를 비동기적으로 처리할 콜백을 등록합니다.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,  # ROI에는 운전자 한 명만 있을 것으로 가정
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
            result_callback=callback)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect_async(self, image, timestamp_ms):
        """
        MediaPipe 이미지 객체(주로 ROI)를 입력받아 자세 랜드마크 추론을 비동기적으로 시작합니다.
        """
        self.detector.detect_async(image, timestamp_ms)

    def close(self):
        """
        Landmarker 객체의 리소스를 해제합니다.
        """
        self.detector.close() 