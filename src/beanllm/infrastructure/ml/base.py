"""
ML Model Base - 추상 기본 클래스 및 공유 타입

모든 ML 프레임워크 래퍼가 상속받는 추상 인터페이스
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
else:
    NDArray = Any  # noqa: N816


class BaseMLModel(ABC):
    """
    ML 모델 베이스 클래스

    모든 ML 프레임워크의 통합 인터페이스.
    load, predict, save 메서드를 구현해야 합니다.

    Example:
        ```python
        class MyModel(BaseMLModel):
            def load(self, model_path):
                ...
            def predict(self, inputs):
                ...
            def save(self, save_path):
                ...
        ```
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Args:
            model_path: 모델 파일 경로 (옵션)
        """
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load(self, model_path: Union[str, Path]) -> None:
        """모델 로드"""
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """예측"""
        pass

    @abstractmethod
    def save(self, save_path: Union[str, Path]) -> None:
        """모델 저장"""
        pass
