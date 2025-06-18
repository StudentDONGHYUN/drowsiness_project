# src/detection_logic.py (Holistic 기능 및 인덱스 오류 수정 최종 버전)
import numpy as np
import cv2
from scipy.spatial import distance as dist


def calculate_ear(eye):
    # 눈의 수직/수평 거리를 계산하여 EAR(눈 깜빡임 비율)을 반환
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_head_pose(shape, frame_shape):
    # 3D 모델과 2D 이미지의 특징점을 이용해 머리 방향 각도(x, y, z)를 계산하여 반환
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # 코 끝
            (0.0, -330.0, -65.0),  # 턱
            (-225.0, 170.0, -135.0),  # 왼쪽 눈의 왼쪽 끝
            (225.0, 170.0, -135.0),  # 오른쪽 눈의 오른쪽 끝
            (-150.0, -150.0, -125.0),  # 왼쪽 입 끝
            (150.0, -150.0, -125.0),  # 오른쪽 입 끝
        ],
        dtype="double",
    )

    # 2D 이미지의 얼굴 특징점 (MediaPipe 478개 랜드마크 기준 인덱스)
    image_points = np.array(
        [
            shape[1],  # 코 끝 (Nose tip)
            shape[152],  # 턱 (Chin)
            shape[263],  # 왼쪽 눈의 왼쪽 끝 (Left eye left corner)
            shape[33],  # 오른쪽 눈의 오른쪽 끝 (Right eye right corner)
            shape[291],  # 왼쪽 입 끝 (Left mouth corner)
            shape[61],  # 오른쪽 입 끝 (Right mouth corner)
        ],
        dtype="double",
    )

    (focal_length, center) = (frame_shape[1], (frame_shape[1] / 2, frame_shape[0] / 2))
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))  # 렌즈 왜곡은 없다고 가정
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles  # x, y, z 회전 각도 반환


def analyze_torso_twist(face_landmarks, pose_landmarks):
    """
    얼굴의 방향과 어깨의 방향을 비교하여 몸의 뒤틀림 각도를 계산합니다.
    정면을 응시할 때 이 각도는 0에 가깝습니다.
    """
    try:
        # 얼굴 방향 벡터 (오른쪽 귀 -> 왼쪽 귀)
        face_v_x = face_landmarks[234][0] - face_landmarks[454][0]
        face_v_y = face_landmarks[234][1] - face_landmarks[454][1]

        # 어깨 방향 벡터 (오른쪽 어깨 -> 왼쪽 어깨)
        shoulder_v_x = pose_landmarks[11][0] - pose_landmarks[12][0]
        shoulder_v_y = pose_landmarks[11][1] - pose_landmarks[12][1]

        # 두 벡터 사이의 각도 계산 (라디안 -> 각도)
        angle = np.degrees(
            np.arccos(
                np.dot([face_v_x, face_v_y], [shoulder_v_x, shoulder_v_y])
                / (
                    np.linalg.norm([face_v_x, face_v_y])
                    * np.linalg.norm([shoulder_v_x, shoulder_v_y])
                )
            )
        )
        return angle
    except:
        return 0.0


def analyze_posture(pose_landmarks):
    """
    자세의 구부정함(slouching)과 쏠림을 분석합니다.
    """
    try:
        # 어깨 중심점과 엉덩이 중심점 계산
        shoulder_mid = (pose_landmarks[11] + pose_landmarks[12]) / 2
        hip_mid = (pose_landmarks[23] + pose_landmarks[24]) / 2

        # 척추의 기울기 (x축 기준)
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # 정자세(90도)에서 벗어난 각도를 반환
        return abs(angle - 90)
    except:
        return 0.0


def is_hand_near_face(hand_landmarks, face_landmarks, threshold=100):
    """
    손이 얼굴 근처에 있는지 확인합니다. (스마트폰 사용 등 의심)
    """
    try:
        # 손의 중심점 계산 (손목, 중지 뿌리)
        hand_center = (hand_landmarks[0] + hand_landmarks[9]) / 2
        # 얼굴의 중심점 계산 (코, 턱)
        face_center = (face_landmarks[1] + face_landmarks[152]) / 2

        distance = np.linalg.norm(hand_center - face_center)

        return distance < threshold
    except:
        return False
