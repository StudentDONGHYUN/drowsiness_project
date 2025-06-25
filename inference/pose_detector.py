# FILE: inference/pose_detector.py
# 역할: MediaPipe Pose 모델을 로드하고 실행. 차량 이동 상태 분석에 필요한 분할 마스크 출력을 활성화.
import cv2
import mediapipe as mp
import platform
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

class PoseDetector:
    def __init__(self, model_path, result_callback):
        """
        모델 파일을 받아 PoseLandmarker 객체를 생성하고 초기화합니다.
        LIVE_STREAM 모드로 실행하고, 결과를 비동기적으로 처리할 콜백을 등록합니다.
        Linux 환경에서는 GPU 가속을 사용합니다.
        """
        delegate_option = None
        if platform.system() == "Linux":
            delegate_option = base_options_module.BaseOptions.Delegate.GPU

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate_option)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            output_segmentation_masks=True,
            result_callback=result_callback)
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect_async(self, image: mp.Image, timestamp_ms: int):
        """
        MediaPipe 이미지 객체와 선택적인 ROI를 입력받아 자세 랜드마크 추론을 비동기적으로 시작합니다.
        """
        self.landmarker.detect_async(image, timestamp_ms)

    def close(self):
        """
        Landmarker 객체의 리소스를 해제합니다.
        """
        self.landmarker.close() 