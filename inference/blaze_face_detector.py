# FILE: inference/blaze_face_detector.py
# 역할: 가볍고 빠른 BlazeFace 모델을 로드하고 실행. 운전자의 '얼굴 위치'를 신속하게 탐지하는 1단계 역할.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
import mediapipe as mp
import platform
from mediapipe.tasks.python.components.containers.rect import NormalizedRect

class BlazeFaceDetector:
    def __init__(self, model_path):
        """
        BlazeFace 검출기를 초기화합니다.
        
        Args:
            model_path: BlazeFace 모델 파일 경로
        """
        delegate_option = None
        if platform.system() == "Linux":
            delegate_option = base_options_module.BaseOptions.Delegate.GPU

        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=delegate_option)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3
        )
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, image):
        """
        이미지에서 얼굴을 검출합니다.
        
        Args:
            image: MediaPipe Image 객체
            
        Returns:
            검출된 얼굴 목록. 각 얼굴은 bounding box와 신뢰도 점수를 포함
        """
        return self.detector.detect(image)

    def detect_roi(self, image: mp.Image) -> NormalizedRect | None:
        detection_result = self.detector.detect(image)
        if not detection_result.detections:
            return None

        # Assume the largest face is the driver
        largest_detection = max(detection_result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bbox = largest_detection.bounding_box
        
        # Expand the ROI for pose detection
        # Make it wider and taller
        roi_width = bbox.width * 1.5
        roi_height = bbox.height * 2.0
        
        # Center the ROI
        x_center = (bbox.origin_x + bbox.width / 2) / image.width
        y_center = (bbox.origin_y + bbox.height / 2) / image.height

        # Convert pixel dimensions to normalized dimensions
        norm_width = roi_width / image.width
        norm_height = roi_height / image.height
        
        return NormalizedRect(
            x_center=x_center,
            y_center=y_center,
            width=norm_width,
            height=norm_height
        )

    def close(self):
        """
        검출기의 리소스를 해제합니다.
        """
        self.detector.close()

    @property
    def is_running(self):
        """detector의 실행 상태를 반환합니다."""
        return True 