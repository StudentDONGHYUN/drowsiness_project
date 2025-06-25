# config.py
# 모든 설정값, 경로, 임계값을 한 곳에서 관리하여 유지보수를 용이하게 함.

import os

# -----------------------------------------------------------------------------
# 1. 모델 파일 경로 (Model File Paths)
# -----------------------------------------------------------------------------
POSE_MODEL_PATH = os.path.join('models', 'pose_landmarker_full.task')
FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker.task"
OBJECT_MODEL_PATH = "models/efficientdet_lite2.tflite"

# -----------------------------------------------------------------------------
# 2. 비디오 입력 설정 (Video Input Settings)
# -----------------------------------------------------------------------------
VIDEO_PATH = 0  # 0 for webcam, or "path/to/video.mp4" for video file

# -----------------------------------------------------------------------------
# 3. 시스템 상태 정의 (State Machine Definitions)
# -----------------------------------------------------------------------------
STATE_TRACKING_NORMAL = "NORMAL"      # 평시 상태
STATE_HIGH_ALERT = "HIGH_ALERT"       # 위험 상황(스마트폰 사용 등) 감지 상태

# -----------------------------------------------------------------------------
# 4. 스케줄링 설정 (Scheduling Settings)
# -----------------------------------------------------------------------------
DETECTION_INTERVAL_NORMAL = 5   # 평시(NORMAL) 상태에서 객체 감지 실행 간격 (프레임)
HIGH_ALERT_FRAME_THRESHOLD = 30 # HIGH_ALERT 상태로 전환/해제되기까지 필요한 연속적인 위험 감지/미감지 프레임 수

# -----------------------------------------------------------------------------
# 5. 판단 로직 임계값 (Judgment Engine Thresholds)
# -----------------------------------------------------------------------------
# Blendshape 판단 임계값 (0.0 ~ 1.0)
EYE_BLINK_THRESHOLD = 0.45      # 눈 감김 판단 기준
JAW_OPEN_THRESHOLD = 0.5        # 하품 판단 기준
DROWSY_FRAME_LIMIT = 45         # 눈 감음이 이 프레임 수 이상 지속되면 졸음으로 판단

# 머리 각도 판단 임계값 (degrees)
HEAD_PITCH_THRESHOLD = -15.0    # 고개 숙임(졸음) 판단 기준 (음수값이 숙임)
HEAD_YAW_THRESHOLD = 25.0       # 좌우 곁눈질(주의 분산) 판단 기준

# 손-객체 근접 판단 임계값 (정규화된 좌표 기준, 캘리브레이션 없음)
HAND_OBJECT_PROXIMITY_THRESHOLD = 0.1 # 손-휴대폰 근접 판단 기준 (단위: 정규화된 좌표계)

# YOLO를 사용하지 않으므로 관련 ID는 제거합니다.
# PHONE_CLASS_ID = 67 