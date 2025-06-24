# config.py
# 모든 설정값, 경로, 임계값을 한 곳에서 관리하여 유지보수를 용이하게 함.

# --- 모델 파일 경로 ---
# 각 모델은 MediaPipe나 Ultralytics에서 제공하는 공식 파일을 다운로드하여 지정된 경로에 위치시킵니다.
FACE_DETECTOR_MODEL_PATH = "models/blaze_face_short_range.tflite"
POSE_MODEL_PATH = "models/pose_landmarker_heavy.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker.task"
OBJECT_MODEL_PATH = "models/efficientdet_lite2.tflite"

# -----------------------------------------------------------------------------
# 2. 비디오 입력 설정
# -----------------------------------------------------------------------------
VIDEO_PATH = 0 # 0 for webcam, or "path/to/video.mp4" for video file

# -----------------------------------------------------------------------------
# 3. 시스템 상태 (State Machine)
# -----------------------------------------------------------------------------
STATE_SEARCHING = "SEARCHING"         # 1단계: 운전자 얼굴 탐색
STATE_CALIBRATING = "CALIBRATING"     # 2단계: 어깨너비 측정/캘리브레이션
STATE_TRACKING_NORMAL = "NORMAL" # 3단계: 정상 추적
STATE_HIGH_ALERT = "HIGH_ALERT"       # 4단계: 위험 상황(스마트폰 사용 등) 감지
STATE_IDLE = "IDLE"                   # (미사용) 차량 정차 시 유휴 상태

# --- 차량 상태 정의 (영문) ---
VEHICLE_STATUS_STOPPED = "STOPPED"
VEHICLE_STATUS_DRIVING = "DRIVING"

# --- 운전자 상태 메시지 정의 (영문) ---
DRIVER_STATUS_NORMAL = "Normal"
DRIVER_STATUS_CALIBRATING = "Calibrating..."
DRIVER_STATUS_PHONE_VIEW = "WARNING: Phone View"
DRIVER_STATUS_PHONE_CALL = "CAUTION: Phone Call"
DRIVER_STATUS_DROWSY_BLINK = "WARNING: Eyes Closed"
DRIVER_STATUS_DROWSY_YAWN = "WARNING: Yawning"
DRIVER_STATUS_DROWSY_NOD = "WARNING: Nodding Off"
DRIVER_STATUS_DISTRACTED_GAZE = "CAUTION: Distracted"

# -----------------------------------------------------------------------------
# 4. 캘리브레이션 설정
# -----------------------------------------------------------------------------
CALIBRATION_FRAME_COUNT = 50 # 어깨너비 측정을 위한 샘플 프레임 수

# -----------------------------------------------------------------------------
# 5. 모델 실행 주기 (프레임 단위)
# -----------------------------------------------------------------------------
# --- 캘리브레이션 및 스케줄링 ---
FACE_INTERVAL_NORMAL = 1        # 평시 Face Mesh 실행 간격 (2프레임당 1회)
HAND_INTERVAL_NORMAL = 5        # 평시 Hand 실행 간격
OBJECT_INTERVAL_NORMAL = 10       # 평시 Object Detector 실행 간격
# HIGH_ALERT 상태에서는 모든 모델을 매 프레임 실행

# --- 판단 로직 임계값 ---
MOTION_STOP_THRESHOLD = 1.0     # 광학 흐름 정지 판단 기준
MOTION_STOP_DURATION = 45       # 정차 상태로 판단할 지속 프레임 (15fps 기준 3초)
POSE_FAIL_LIMIT = 75            # 운전자 탐지 실패로 간주할 지속 프레임 (5초)
ALERT_TIMEOUT = 30              # 위험 상황 해제 판단 지속 프레임 (2초)

# Blendshape 판단 임계값 (0.0 ~ 1.0)
EYE_BLINK_THRESHOLD = 0.45       # 눈 감김 판단 기준
JAW_OPEN_THRESHOLD = 0.5        # 이 값보다 턱 벌어짐 점수가 높으면 하품으로 간주
DROWSY_FRAME_LIMIT = 45         # 눈 감음이 이 프레임 수 이상 지속되면 졸음으로 판단
YAWN_FRAME_LIMIT = 20           # 하품 지속 프레임

# 머리 각도 판단 임계값 (degrees)
HEAD_PITCH_THRESHOLD = 20.0     # 고개 숙임(졸음) 판단 기준 (degree)
HEAD_YAW_THRESHOLD = 30.0       # 곁눈질(주의 분산) 판단 기준 (degree)
# 전방주시로 인정하는 pitch 범위 (룸미러 등 허용)
HEAD_PITCH_DISTRACT_MIN = -15.0 # 곁눈질 판정 시 허용되는 고개 숙임 최소 각도
HEAD_PITCH_DISTRACT_MAX = 15.0  # 곁눈질 판정 시 허용되는 고개 숙임 최대 각도

# '개인화된 자'에 대한 상대 비율 임계값
HAND_PHONE_PROXIMITY_RATIO = 0.8 # 손-휴대폰 근접 판단 기준 (어깨너비 대비 비율)
HAND_MOUTH_PROXIMITY_RATIO = 0.15  # 손-입 거리가 어깨너비의 15% 이내면 근접으로 판단

# YOLO 클래스 ID
PHONE_CLASS_ID = 67 # 예시: COCO 데이터셋의 cell phone ID

# HIGH_ALERT 상태에서는 모든 모델을 매 프레임 실행
HIGH_ALERT_FRAME_THRESHOLD = 30 # HIGH_ALERT 상태로 전환되기까지 필요한 연속 프레임 수 