# processing/motion_analyzer.py
# 역할: 차량의 이동 상태를 분석. 전역적 움직임 필터링(Global Motion Filtering) 적용.
import cv2
import numpy as np

class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None  # 이전 프레임(흑백)
        self.prev_pts = None   # 이전 프레임에서 추적한 특징점 좌표
        # 특징점 추출 파라미터
        self.feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=7, blockSize=7)
        # 움직임 일관성 판단 임계값
        self.std_angle_thresh = 0.3  # 각도 표준편차 임계값 (라디안)
        self.std_mag_thresh = 0.7    # 크기 표준편차 임계값 (픽셀)

    def analyze(self, frame_gray, pose_results):
        """
        배경 특징점 추적 및 움직임 벡터 일관성 분석을 통해 차량의 이동 상태를 판단합니다.
        1. 운전자(사람) 영역을 segmentation mask로 제외한 배경에서만 특징점 추출
        2. 이전 프레임과의 LK Optical Flow로 특징점 이동 추적
        3. 모든 벡터의 방향(각도)·크기(길이) 표준편차 계산
        4. 각도는 np.unwrap으로 순환성 보정 후 표준편차 계산
        5. 두 표준편차가 모두 임계값 이하이면 '주행', 아니면 '정차'로 판단
        6. 특징점은 50개 미만일 때만 새로 보충하여 성능 최적화
        반환값: motion_consistency (1.0=주행, 0.0=정차)
        """
        motion_consistency = 0.0  # 일관성(0=정차, 1=주행)
        std_angle = 0.0
        std_mag = 0.0
        num_vectors = 0

        # 1. 배경 마스크 생성 (사람=1.0, 배경=0.0)
        background_mask = np.ones_like(frame_gray, dtype=np.uint8) * 255  # 기본값: 전체를 배경으로
        if pose_results and hasattr(pose_results, 'segmentation_masks') and pose_results.segmentation_masks:
            try:
                segmentation_mask_raw = pose_results.segmentation_masks[0].numpy_view()
                # 마스크를 원본 이미지 크기로 리사이즈
                resized_mask = cv2.resize(segmentation_mask_raw, (frame_gray.shape[1], frame_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                background_mask = (resized_mask < 0.5).astype(np.uint8) * 255
            except Exception as e:
                # print(f"[MotionAnalyzer] 마스크 생성 오류: {e}")
                pass # 오류 발생 시 기본 마스크 사용
        
        # 2. Optical Flow 기반 특징점 추적 및 일관성 분석
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            # 이전 프레임의 특징점들을 현재 프레임에서 추적
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_pts, None)
            good_prev = self.prev_pts[status.flatten() == 1]
            good_next = next_pts[status.flatten() == 1]

            # (N, 2) shape으로 강제 변환
            good_prev = np.reshape(good_prev, (-1, 2))
            good_next = np.reshape(good_next, (-1, 2))

            # 3. 움직임 벡터 계산 및 일관성 분석
            if good_prev.shape[0] > 0 and good_next.shape[0] > 0 and good_prev.shape[1] == 2 and good_next.shape[1] == 2:
                vectors = good_next - good_prev  # 각 특징점의 이동 벡터
                mags = np.linalg.norm(vectors, axis=1)  # 벡터 크기(길이)
                angles = np.arctan2(vectors[:,1], vectors[:,0])  # 벡터 방향(라디안)
                num_vectors = len(mags)

                if num_vectors > 10:  # 충분한 특징점이 있을 때만 분석
                    # [중요] 각도는 -π~+π 경계에서 순환성이 있으므로 np.unwrap으로 보정
                    unwrapped_angles = np.unwrap(angles)
                    std_angle = np.std(unwrapped_angles)
                    std_mag = np.std(mags)
                    # 두 표준편차가 모두 임계값 이하이면 '주행', 아니면 '정차'
                    if std_angle < self.std_angle_thresh and std_mag < self.std_mag_thresh:
                        motion_consistency = 1.0  # 주행
                    else:
                        motion_consistency = 0.0  # 정차
                else:
                    # 특징점 부족 시 정차로 간주
                    motion_consistency = 0.0
            else:
                # shape 오류 발생 시 경고 출력
                # print(f"[MotionAnalyzer] shape error: good_prev {good_prev.shape}, good_next {good_next.shape}")
                motion_consistency = 0.0

            # 4. 특징점 보충: 50개 미만이면 새로 찾고, 아니면 추적된 점을 그대로 사용
            if num_vectors < 50:
                self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=background_mask, **self.feature_params)
            else:
                self.prev_pts = good_next.reshape(-1, 1, 2)
        else:
            # 첫 프레임 등, 이전 정보가 없으면 새로 특징점 추출
            self.prev_pts = cv2.goodFeaturesToTrack(frame_gray, mask=background_mask, **self.feature_params)

        # 5. 다음 프레임을 위한 상태 갱신
        self.prev_gray = frame_gray.copy()
        return motion_consistency  # 1.0=주행, 0.0=정차 (추가 정보 필요시 std_angle, std_mag, num_vectors도 반환 가능) 