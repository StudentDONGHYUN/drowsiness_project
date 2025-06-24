# utils/geometry.py
# 역할: 모든 기하학적 계산을 담당.
import numpy as np
import cv2
import math

def calculate_model_unit_distance(point1, point2):
    """두 3D 랜드마크의 '모델 유닛' 거리를 계산합니다. 이는 실제 물리적 거리가 아닙니다."""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def get_shoulder_width_from_pose(pose_landmarks):
    """주어진 Pose 랜드마크로부터 '모델 유닛' 단위의 어깨너비를 계산합니다."""
    # MediaPipe Pose의 어깨 랜드마크 인덱스는 11번(왼쪽), 12번(오른쪽)입니다.
    if len(pose_landmarks) > 12:
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        return calculate_model_unit_distance(left_shoulder, right_shoulder)
    return 0

def get_angles_from_transformation_matrix(transformation_matrix):
    """FaceLandmarker가 출력한 4x4 변환 행렬에서 Yaw, Pitch, Roll 각도를 추출합니다."""
    if transformation_matrix is None: return 0, 0, 0
    # 4x4 행렬에서 회전 정보를 담고 있는 좌상단 3x3 부분 행렬만 추출합니다.
    rotation_matrix = transformation_matrix[0:3, 0:3]
    sy = math.sqrt(rotation_matrix[0,0] ** 2 +  rotation_matrix[1,0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = math.atan2(-rotation_matrix[2,0], sy)
        z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = math.atan2(-rotation_matrix[2,0], sy)
        z = 0
    # 라디안을 우리에게 익숙한 각도(degree)로 변환하여 반환합니다.
    return math.degrees(y), math.degrees(x), math.degrees(z) # Yaw, Pitch, Roll

def check_ray_box_intersection(ray_origin, ray_dir, box):
    """
    3D 광선(ray)과 3D AABB(축 정렬 박스) 교차 판정 (표준 알고리즘)
    box: (min_xyz, max_xyz) 튜플
    """
    tmin = (box[0] - ray_origin) / (ray_dir + 1e-8)
    tmax = (box[1] - ray_origin) / (ray_dir + 1e-8)
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    t_near = np.max(t1)
    t_far = np.min(t2)
    return t_near < t_far and t_far > 0

def adjust_face_landmarks_to_global(face_results_raw, roi_offset, cropped_shape, original_shape):
    """
    크롭된 이미지에서 계산된 얼굴 랜드마크 좌표를 원본 이미지의 전역 좌표로 재조정합니다.
    """
    if not face_results_raw or not face_results_raw.face_landmarks:
        return face_results_raw

    x_offset, y_offset = roi_offset
    cropped_h, cropped_w = cropped_shape[:2]
    original_h, original_w = original_shape[:2]

    # 랜드마크 좌표 재조정
    for face_landmark_list in face_results_raw.face_landmarks:
        for lm in face_landmark_list:
            # 1. 크롭 이미지 기준의 정규화된 좌표 -> 픽셀 좌표로 변환
            lm_pixel_x = lm.x * cropped_w
            lm_pixel_y = lm.y * cropped_h
            
            # 2. 오프셋을 더해 원본 이미지 기준의 픽셀 좌표로 변환
            global_pixel_x = x_offset + lm_pixel_x
            global_pixel_y = y_offset + lm_pixel_y

            # 3. 원본 이미지 기준으로 다시 정규화
            lm.x = global_pixel_x / original_w
            lm.y = global_pixel_y / original_h
    
    # 변환 행렬은 회전 정보(머리 각도)에 주로 사용되며, 이는 이미지 위치에 불변하므로
    # 별도의 변환 없이 그대로 반환합니다. Blendshapes 역시 위치와 무관합니다.
    return face_results_raw

def adjust_pose_landmarks_to_global(pose_results_raw, roi_offset, cropped_shape, original_shape):
    """
    크롭된 이미지에서 계산된 포즈 랜드마크 좌표를 원본 이미지의 전역 좌표로 재조정합니다.
    """
    if not pose_results_raw or not pose_results_raw.pose_landmarks:
        return pose_results_raw

    x_offset, y_offset = roi_offset
    cropped_h, cropped_w = cropped_shape[:2]
    original_h, original_w = original_shape[:2]

    # 랜드마크 좌표 재조정
    for pose_landmark_list in pose_results_raw.pose_landmarks:
        for lm in pose_landmark_list:
            # 1. 크롭 이미지 기준의 정규화된 좌표 -> 픽셀 좌표로 변환
            lm_pixel_x = lm.x * cropped_w
            lm_pixel_y = lm.y * cropped_h
            
            # 2. 오프셋을 더해 원본 이미지 기준의 픽셀 좌표로 변환
            global_pixel_x = x_offset + lm_pixel_x
            global_pixel_y = y_offset + lm_pixel_y

            # 3. 원본 이미지 기준으로 다시 정규화
            lm.x = global_pixel_x / original_w
            lm.y = global_pixel_y / original_h
    
    # segmentation_mask도 동일하게 변환해야 하지만, 현재 구현에서는 생략.
    # 필요 시, 마스크에 대한 리사이즈 및 패딩 추가 필요.
    return pose_results_raw

def estimate_gaze_intersection(face_results, yolo_results, hand_results, image_shape, config, return_debug=False):
    """
    시선 벡터와 휴대폰 Bounding Box의 3D 공간상 교차 여부를 판정합니다.
    얼굴 변환 행렬, 홍채(Iris) 랜드마크, YOLO 박스, 손 Z값을 모두 활용합니다.
    return_debug=True일 때, 시각화/디버그용 dict도 함께 반환
    """
    # --- 2.1. 필수 데이터 유효성 검사 ---
    if not (face_results and hasattr(face_results, 'facial_transformation_matrixes') and 
            hasattr(face_results, 'face_landmarks') and face_results.facial_transformation_matrixes and 
            face_results.face_landmarks and yolo_results):
        if return_debug:
            return False, {}
        return False

    # --- 2.2. 목표물 정의: 휴대폰의 3D 위치 추정 ---
    phone_box_2d = None
    for r in yolo_results:
        for box in getattr(r, 'boxes', []):
            if int(box.cls[0]) == config.PHONE_CLASS_ID:
                phone_box_2d = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                break
        if phone_box_2d is not None:
            break
    if phone_box_2d is None:
        if return_debug:
            return False, {}
        return False

    # Z 깊이 추정
    phone_depth_z = 0.0
    if hand_results and hasattr(hand_results, 'hand_world_landmarks') and hand_results.hand_world_landmarks:
        phone_cx = (phone_box_2d[0] + phone_box_2d[2]) / 2 / image_shape[1]
        phone_cy = (phone_box_2d[1] + phone_box_2d[3]) / 2 / image_shape[0]
        min_dist = float('inf')
        for hand in hand_results.hand_world_landmarks:
            wrist = hand[0]
            dist = np.sqrt((wrist.x - phone_cx)**2 + (wrist.y - phone_cy)**2)
            if dist < min_dist:
                min_dist = dist
                phone_depth_z = wrist.z
    elif face_results.face_landmarks:
        face_z_coords = [lm.z for lm in face_results.face_landmarks[0]]
        phone_depth_z = np.mean(face_z_coords)

    phone_center_x = (phone_box_2d[0] + phone_box_2d[2]) / 2 / image_shape[1]
    phone_center_y = (phone_box_2d[1] + phone_box_2d[3]) / 2 / image_shape[0]
    phone_center_3d = np.array([phone_center_x, phone_center_y, phone_depth_z])

    # 3D 박스 크기 추정 (가로, 세로: 2D 박스 비율, 두께: 어깨너비의 1/10)
    w_2d = abs(phone_box_2d[2] - phone_box_2d[0]) / image_shape[1]
    h_2d = abs(phone_box_2d[3] - phone_box_2d[1]) / image_shape[0]
    d_3d = max(w_2d, h_2d) * 0.2  # 두께는 대략적으로 추정
    min_xyz = phone_center_3d - np.array([w_2d/2, h_2d/2, d_3d/2])
    max_xyz = phone_center_3d + np.array([w_2d/2, h_2d/2, d_3d/2])
    phone_3d_box = (min_xyz, max_xyz)

    # --- 2.3. 시선 벡터 정의 ---
    face_landmarks = face_results.face_landmarks[0]
    transformation_matrix = face_results.facial_transformation_matrixes[0]
    left_iris_center = face_landmarks[473]
    right_iris_center = face_landmarks[468]
    eye_center_3d = np.array([
        (left_iris_center.x + right_iris_center.x) / 2,
        (left_iris_center.y + right_iris_center.y) / 2,
        (left_iris_center.z + right_iris_center.z) / 2
    ])
    gaze_direction_vector = transformation_matrix[0:3, 2]
    gaze_direction_vector = gaze_direction_vector / (np.linalg.norm(gaze_direction_vector) + 1e-8)

    # --- 2.4. 교차 판정 ---
    is_intersecting = check_ray_box_intersection(eye_center_3d, gaze_direction_vector, phone_3d_box)

    if return_debug:
        debug = {
            'left_iris': (left_iris_center.x, left_iris_center.y),
            'right_iris': (right_iris_center.x, right_iris_center.y),
            'eye_center': (eye_center_3d[0], eye_center_3d[1]),
            'gaze_dir': (gaze_direction_vector[0], gaze_direction_vector[1]),
            'phone_center': (phone_center_3d[0], phone_center_3d[1]),
            'is_gazing': is_intersecting
        }
        return is_intersecting, debug
    return is_intersecting 