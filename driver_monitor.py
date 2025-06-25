# FILE: driver_monitor.py
# 역할: 모든 모듈을 조립하고, 시스템의 전체 흐름(상태 머신, 캘리브레이션, 스케줄링)을 관리.

import cv2
import time
import threading
import mediapipe as mp

from config import (
    POSE_MODEL_PATH, FACE_LANDMARKER_MODEL_PATH, OBJECT_MODEL_PATH,
    STATE_TRACKING_NORMAL, STATE_HIGH_ALERT,
    DETECTION_INTERVAL_NORMAL, HIGH_ALERT_FRAME_THRESHOLD
)
from inference.face_detector import FaceDetector
from inference.pose_detector import PoseDetector
from inference.object_detector import ObjectDetector
from processing.motion_analyzer import MotionAnalyzer
from processing.judgment_engine import JudgmentEngine
from utils.visualizer import Visualizer

class DriverMonitor:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {camera_id}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.latest_face_result = None
        self.latest_pose_result = None
        self.latest_object_result = None
        self.result_lock = threading.Lock()

        self.fps = 0
        self.prev_time = 0

        # --- Intelligent Architecture States & Counters ---
        self.system_state = STATE_TRACKING_NORMAL
        self.frame_counter = 0
        self.alert_counter = 0

        self._init_modules()

    def _init_modules(self):
        self.face_detector = FaceDetector(FACE_LANDMARKER_MODEL_PATH, self._result_callback('face'))
        self.pose_detector = PoseDetector(POSE_MODEL_PATH, self._result_callback('pose'))
        self.object_detector = ObjectDetector(OBJECT_MODEL_PATH, self._result_callback('object'))

        self.motion_analyzer = MotionAnalyzer()
        self.judgment_engine = JudgmentEngine()
        self.visualizer = Visualizer(self.frame_width, self.frame_height)

    def _result_callback(self, detector_name):
        def callback(result, output_image: mp.Image, timestamp_ms: int):
            with self.result_lock:
                if detector_name == 'face':
                    self.latest_face_result = result
                elif detector_name == 'pose':
                    self.latest_pose_result = result
                elif detector_name == 'object':
                    self.latest_object_result = result
        return callback

    def run(self):
        timestamp_ms = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                break
            
            self.frame_counter += 1

            # --- FPS Calculation ---
            now = time.time()
            if self.prev_time > 0:
                self.fps = 1 / (now - self.prev_time)
            self.prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)

            # --- State-based Asynchronous Detection ---
            # Pose is always detected to keep track of the driver
            self.pose_detector.detect_async(mp_image, timestamp_ms)

            # Face and Object detection are scheduled based on the system state
            run_secondary_detectors = True
            if self.system_state == STATE_TRACKING_NORMAL:
                if self.frame_counter % DETECTION_INTERVAL_NORMAL != 0:
                    run_secondary_detectors = False
            
            if run_secondary_detectors:
                self.face_detector.detect_async(mp_image, timestamp_ms)
                self.object_detector.detect_async(mp_image, timestamp_ms)
            
            # Get the latest results safely
            with self.result_lock:
                face_result = self.latest_face_result
                pose_result = self.latest_pose_result
                object_result = self.latest_object_result
            
            # --- Prepare data for processing modules ---
            
            # 1. Create the results dictionary for the judgment engine
            results_for_judgment = {
                'face_landmarker': face_result,
                'pose': pose_result,
                'object': object_result,
                'hands': None  # Placeholder, as hand detection is not implemented
            }

            # Since calibration is removed, use a fixed threshold value from config
            # The judgment engine will need to be adapted to use this.
            judgment_status = self.judgment_engine.judge(
                results_for_judgment,
                frame.shape
            )

            # --- State Machine Update ---
            if judgment_status['is_drowsy'] or judgment_status['is_distracted']:
                self.alert_counter = min(self.alert_counter + 1, HIGH_ALERT_FRAME_THRESHOLD)
            else:
                self.alert_counter = max(self.alert_counter - 1, 0)
            
            if self.alert_counter >= HIGH_ALERT_FRAME_THRESHOLD:
                self.system_state = STATE_HIGH_ALERT
            elif self.alert_counter == 0:
                self.system_state = STATE_TRACKING_NORMAL
            
            # --- Visualization ---
            display_frame = self.visualizer.draw_all(
                frame,
                face_result,
                pose_result,
                object_result,
                judgment_status,
                self.fps,
                self.system_state
            )

            cv2.imshow('Driver Monitor', display_frame)

            if cv2.waitKey(5) & 0xFF == 27: # ESC key
                break
        
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_detector.close()
        self.pose_detector.close()
        self.object_detector.close()
        print("System shut down and resources released.")

if __name__ == '__main__':
    monitor = DriverMonitor(camera_id=0)
    monitor.run()