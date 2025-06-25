# FILE: inference/object_detector.py
# 역할: MediaPipe ObjectDetector 모델을 로드하고 실행. 휴대폰 등 특정 객체 탐지.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
import mediapipe as mp
import platform

class ObjectDetector:
    def __init__(self, model_path, result_callback):
        """
        모델 파일을 받아 ObjectDetector 객체를 생성하고 초기화합니다.
        LIVE_STREAM 모드로 실행하고, 결과를 비동기적으로 처리할 콜백을 등록합니다.
        Linux 환경에서는 GPU 가속을 사용합니다.
        """
        delegate_option = None
        if platform.system() == "Linux":
            delegate_option = base_options_module.BaseOptions.Delegate.GPU
            
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate_option)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            score_threshold=0.5,
            category_allowlist=['cell phone'],
            result_callback=result_callback)
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect_async(self, image: mp.Image, timestamp_ms: int):
        """
        MediaPipe 이미지 객체를 입력받아 객체 탐지를 비동기적으로 시작합니다.
        """
        self.detector.detect_async(image, timestamp_ms)

    def close(self):
        """
        Detector 객체의 리소스를 해제합니다.
        """
        self.detector.close() 