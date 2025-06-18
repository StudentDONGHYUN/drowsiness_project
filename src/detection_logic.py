# src/detection_logic.py
import numpy as np
import cv2
from scipy.spatial import distance as dist

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(shape, frame_shape):
    # 3D 모델의 얼굴 특징점
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 코 끝
        (0.0, -330.0, -65.0),        # 턱
        (-225.0, 170.0, -135.0),     # 왼쪽 눈의 왼쪽 끝
        (225.0, 170.0, -135.0),      # 오른쪽 눈의 오른쪽 끝
        (-150.0, -150.0, -125.0),    # 왼쪽 입 끝
        (150.0, -150.0, -125.0)      # 오른쪽 입 끝
    ], dtype="double")

    # 2D 이미지의 얼굴 특징점 인덱스
    image_points = np.array([
        shape[30], # 코 끝
        shape[8],  # 턱
        shape[36], # 왼쪽 눈의 왼쪽 끝
        shape[45], # 오른쪽 눈의 오른쪽 끝
        shape[48], # 왼쪽 입 끝
        shape[54]  # 오른쪽 입 끝
    ], dtype="double")

    (focal_length, center) = (frame_shape[1], (frame_shape[1]/2, frame_shape[0]/2))
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4,1)) # 렌즈 왜곡은 없다고 가정
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # x, y, z 축 회전 각도 계산
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles # x, y, z 회전 각도 반환
