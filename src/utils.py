# src/utils.py

from scipy.spatial import distance as dist
import numpy as np

def calculate_ear(eye):
    # 눈의 수직 거리를 계산합니다.
    # p2, p6와 p3, p5 사이의 거리를 계산합니다.
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # 눈의 수평 거리를 계산합니다.
    # p1, p4 사이의 거리를 계산합니다.
    C = dist.euclidean(eye[0], eye[3])

    # 눈 깜빡임 비율(EAR)을 계산합니다.
    ear = (A + B) / (2.0 * C)

    return ear
