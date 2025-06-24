# FILE: driver_monitor.py
# 역할: 모든 모듈을 조립하고, 시스템의 전체 흐름(상태 머신, 캘리브레이션, 스케줄링)을 관리.

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import config as cfg
from inference.blaze_face_detector import BlazeFaceDetector
from inference.pose_detector import PoseDetector
from inference.hand_detector import HandDetector
from inference.face_detector import FaceDetector as FaceLandmarkerDetector
from inference.object_detector import ObjectDetector
from processing.motion_analyzer import MotionAnalyzer
from processing.judgment_engine import JudgmentEngine
from utils.visualizer import Visualizer
from utils import geometry
import time
import threading

class DriverMonitor:
    def __init__(self):
        self.system_state = cfg.STATE_CALIBRATING
        self.vehicle_status = cfg.VEHICLE_STATUS_DRIVING
        self.prev_vehicle_status = cfg.VEHICLE_STATUS_DRIVING
        self.frame_count = 0
        self.this_driver_shoulder_width = 0.0
        self.calibration_samples = []
        self.alert_state_counter = 0

        self.latest_results = {}
        self.results_lock = threading.Lock()
        
        self.detectors = {
            'face_detector': BlazeFaceDetector(cfg.FACE_DETECTOR_MODEL_PATH, self.on_face_detection_result),
            'pose': PoseDetector(cfg.POSE_MODEL_PATH, self.on_pose_result),
            'hands': HandDetector(cfg.HAND_MODEL_PATH, self.on_hand_result),
            'face_landmarker': FaceLandmarkerDetector(cfg.FACE_LANDMARKER_MODEL_PATH, self.on_face_landmarker_result),
            'object': ObjectDetector(cfg.OBJECT_MODEL_PATH, self.on_object_detection_result)
        }
        self.motion_analyzer = MotionAnalyzer()
        self.judgment_engine = JudgmentEngine()
        self.visualizer = Visualizer()
        
        self.last_frame = None
        self.last_frame_shape = None
        self.fps = 0.0
        self.prev_time = None

    def on_face_detection_result(self, result: vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        driver_face_bbox = self._select_driver_face(result, self.last_frame_shape)
        with self.results_lock:
            self.latest_results['face_roi_bbox'] = driver_face_bbox

    def on_pose_result(self, result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        with self.results_lock:
            self.latest_results['pose'] = result

    def on_hand_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        with self.results_lock:
            self.latest_results['hands'] = result

    def on_face_landmarker_result(self, result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        with self.results_lock:
            self.latest_results['face_landmarker'] = result

    def on_object_detection_result(self, result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int):
        with self.results_lock:
            self.latest_results['object'] = result

    def _select_driver_face(self, detection_result, image_shape):
        if not detection_result or not detection_result.detections or not image_shape:
            return None
        
        h, w = image_shape[:2]
        candidates = []
        for det in detection_result.detections:
            bbox = det.bounding_box
            area = bbox.width * bbox.height
            cx = bbox.origin_x + bbox.width / 2
            cy = bbox.origin_y + bbox.height / 2
            center_dist = ((cx/w - 0.5)**2 + (cy/h - 0.5)**2)**0.5
            
            score = -area * 1.5 + center_dist
            candidates.append({'score': score, 'bbox': bbox})
            
        if not candidates: return None
        
        best_candidate = sorted(candidates, key=lambda x: x['score'])[0]
        bbox = best_candidate['bbox']
        return (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)

    def _get_pose_roi_from_face_bbox(self, face_bbox, image_shape):
        x, y, w, h = face_bbox
        img_h, img_w = image_shape[:2]
        roi_w = int(w * 3.0)
        roi_h = int(h * 4.0)
        face_cx = x + w / 2
        roi_x = int(face_cx - roi_w / 2)
        roi_y = int(y)
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        if roi_x + roi_w > img_w: roi_w = img_w - roi_x
        if roi_y + roi_h > img_h: roi_h = img_h - roi_y
        return (roi_x, roi_y, roi_w, roi_h)

    def process_frame(self, frame):
        self.frame_count += 1
        now = time.time()
        if self.prev_time: self.fps = 1.0 / (now - self.prev_time)
        self.prev_time = now
        timestamp_ms = int(now * 1000)
        
        self.last_frame = frame
        self.last_frame_shape = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        self.detectors['face_detector'].detect_async(mp_image, timestamp_ms)

        with self.results_lock:
            all_results = self.latest_results.copy()

        driver_face_bbox = all_results.get('face_roi_bbox')
        
        if driver_face_bbox:
            pose_roi_bbox = self._get_pose_roi_from_face_bbox(driver_face_bbox, frame.shape)
            all_results['pose_roi_bbox'] = pose_roi_bbox
            x, y, w, h = pose_roi_bbox
            if w > 0 and h > 0:
                cropped_frame = frame[y:y+h, x:x+w]
                mp_cropped_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
                self.detectors['pose'].detect_async(mp_cropped_image, timestamp_ms)
            
            if self.system_state == cfg.STATE_CALIBRATING:
                pose_results = all_results.get('pose')
                if pose_results and pose_results.pose_landmarks:
                    shoulder_width = geometry.get_shoulder_width_from_pose(pose_results.pose_landmarks[0])
                    if shoulder_width > 0: self.calibration_samples.append(shoulder_width)
                    if len(self.calibration_samples) >= cfg.CALIBRATION_FRAME_COUNT:
                        self.this_driver_shoulder_width = np.mean(self.calibration_samples)
                        self.system_state = cfg.STATE_TRACKING_NORMAL
                        print(f"캘리브레이션 완료: 어깨너비={self.this_driver_shoulder_width:.4f}")
                        self.calibration_samples = []
            
            if self.this_driver_shoulder_width > 0:
                is_high_alert = self.system_state == cfg.STATE_HIGH_ALERT
                if is_high_alert or self.frame_count % cfg.FACE_INTERVAL_NORMAL == 0:
                    self.detectors['face_landmarker'].detect_async(mp_image, timestamp_ms)
                if is_high_alert or self.frame_count % cfg.HAND_INTERVAL_NORMAL == 0:
                    self.detectors['hands'].detect_async(mp_image, timestamp_ms)
                if is_high_alert or self.frame_count % cfg.OBJECT_INTERVAL_NORMAL == 0:
                    self.detectors['object'].detect_async(mp_image, timestamp_ms)
        else:
            self.system_state = cfg.STATE_SEARCHING
            self.this_driver_shoulder_width = 0.0

        driver_status = cfg.DRIVER_STATUS_NORMAL
        if self.this_driver_shoulder_width > 0:
             driver_status = self.judgment_engine.judge(all_results, self.this_driver_shoulder_width, frame.shape)
        
        if driver_status != cfg.DRIVER_STATUS_NORMAL and self.system_state != cfg.STATE_CALIBRATING:
            self.alert_state_counter += 1
            if self.alert_state_counter > cfg.HIGH_ALERT_FRAME_THRESHOLD:
                self.system_state = cfg.STATE_HIGH_ALERT
        else:
            self.alert_state_counter = 0
            if self.system_state == cfg.STATE_HIGH_ALERT:
                self.system_state = cfg.STATE_TRACKING_NORMAL
        
        system_info = {'system_state': self.system_state, 'vehicle_status': self.vehicle_status, 'driver_status': driver_status, 'fps': self.fps}
        return self.visualizer.draw_all(frame, all_results, system_info)

    def start(self):
        cap = cv2.VideoCapture(cfg.VIDEO_PATH)
        if not cap.isOpened():
            print(f"오류: 비디오 스트림을 열 수 없습니다 ({cfg.VIDEO_PATH})")
            return

        print("\n[INFO] 키보드 단축키:")
        print("  d: 주행 시작 (Driving)\n  s: 차량 정지 (Stopped)\n  c: 재캘리브레이션 (Re-Calibrate)\n  h: 고위험 상태 강제 설정 (High Alert)\n  n: 정상 상태로 복귀 (Normal)\n  q: 종료\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if self.prev_vehicle_status == cfg.VEHICLE_STATUS_STOPPED and self.vehicle_status == cfg.VEHICLE_STATUS_DRIVING:
                self.system_state = cfg.STATE_CALIBRATING
                self.calibration_samples = []
                self.this_driver_shoulder_width = 0.0
            self.prev_vehicle_status = self.vehicle_status
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Driver Monitoring System', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('d'): self.vehicle_status = cfg.VEHICLE_STATUS_DRIVING
            elif key == ord('s'): self.vehicle_status = cfg.VEHICLE_STATUS_STOPPED
            elif key == ord('c'):
                self.system_state = cfg.STATE_CALIBRATING
                self.calibration_samples = []
                self.this_driver_shoulder_width = 0.0
            elif key == ord('h'):
                self.system_state = cfg.STATE_HIGH_ALERT
                if self.this_driver_shoulder_width == 0.0: self.this_driver_shoulder_width = 0.2 
            elif key == ord('n'): self.system_state = cfg.STATE_TRACKING_NORMAL
        self.close_all(cap)

    def close_all(self, cap=None):
        if cap: cap.release()
        cv2.destroyAllWindows()
        for detector in self.detectors.values():
            detector.close()
        print("시스템 종료. 모든 리소스를 해제했습니다.")

if __name__ == '__main__':
    monitor = DriverMonitor()
    monitor.start()