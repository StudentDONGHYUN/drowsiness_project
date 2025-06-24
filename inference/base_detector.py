# FILE: inference/base_detector.py
# 역할: 모든 Detector 클래스가 따라야 할 설계도(Interface)를 정의하는 추상 기반 클래스.
# 이를 통해 메인 시스템은 어떤 모델이든 동일한 방식으로 호출할 수 있어 코드의 일관성과 확장성이 높아짐.
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image):
        """이미지를 입력받아 추론을 수행하고 결과를 반환하는, 모든 하위 클래스가 반드시 구현해야 할 함수."""
        pass  # 반드시 하위 클래스에서 구현 필요

    @abstractmethod
    def close(self):
        """모델이 사용한 모든 리소스를 안전하게 해제하는, 모든 하위 클래스가 반드시 구현해야 할 함수."""
        pass  # 반드시 하위 클래스에서 구현 필요 