"""
Base OCR Engine

모든 OCR 엔진의 기본 인터페이스 정의.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import OCRConfig


class BaseOCREngine(ABC):
    """
    OCR 엔진 기본 클래스

    모든 OCR 엔진은 이 클래스를 상속받아 구현합니다.

    Attributes:
        name: 엔진 이름

    Example:
        ```python
        class MyOCREngine(BaseOCREngine):
            def __init__(self):
                super().__init__(name="MyOCR")

            def recognize(self, image, config: OCRConfig) -> dict:
                # OCR 로직 구현
                return {
                    "text": "recognized text",
                    "lines": [],
                    "confidence": 0.95,
                    "language": config.language,
                }
        ```
    """

    def __init__(self, name: str):
        """
        Args:
            name: 엔진 이름
        """
        self.name = name

    @abstractmethod
    def recognize(self, image: Any, config: OCRConfig) -> Dict:
        """
        이미지에서 텍스트 인식

        Args:
            image: 이미지 (numpy array 또는 PIL Image)
            config: OCR 설정

        Returns:
            dict: OCR 결과
                - text (str): 전체 텍스트
                - lines (List[OCRTextLine]): 라인별 결과
                - confidence (float): 평균 신뢰도
                - language (str): 인식된 언어
                - metadata (dict, optional): 추가 메타데이터

        Raises:
            NotImplementedError: 하위 클래스에서 구현 필요
        """
        raise NotImplementedError(f"{self.name} engine must implement recognize()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
