"""
TensorFlow Model - Keras/SavedModel 래퍼

TensorFlow/Keras 모델 로드, 예측, 저장을 위한 래퍼 클래스
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from beanllm.infrastructure.ml.base import BaseMLModel

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
else:
    NDArray = Any  # noqa: N816


class TensorFlowModel(BaseMLModel):
    """
    TensorFlow 모델 래퍼

    Keras HDF5 및 SavedModel 형식 지원.

    Example:
        ```python
        # Keras 모델 로드
        model = TensorFlowModel.from_keras("model.h5")
        predictions = model.predict(data)

        # SavedModel 로드
        model = TensorFlowModel.from_saved_model("saved_model/")
        ```
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        super().__init__(model_path)
        if model_path:
            self.load(model_path)

    def load(self, model_path: Union[str, Path]) -> None:
        """
        모델 로드

        Args:
            model_path: 모델 파일/디렉토리 경로
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow 필요:\npip install tensorflow")

        model_path = Path(model_path)

        if model_path.is_dir():
            # SavedModel 형식
            self.model = tf.keras.models.load_model(str(model_path))
        else:
            # HDF5 형식
            self.model = tf.keras.models.load_model(str(model_path))

        self.model_path = model_path

    def predict(
        self,
        inputs: Union[NDArray, List],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> NDArray:
        """
        예측

        Args:
            inputs: 입력 데이터
            batch_size: 배치 크기
            **kwargs: 추가 파라미터

        Returns:
            예측 결과
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        return self.model.predict(inputs, batch_size=batch_size, **kwargs)

    def save(self, save_path: Union[str, Path], format: str = "tf") -> None:
        """
        모델 저장

        Args:
            save_path: 저장 경로
            format: 저장 형식 (tf, h5)
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = Path(save_path)

        if format == "tf":
            # SavedModel 형식
            self.model.save(str(save_path))
        elif format == "h5":
            # HDF5 형식
            self.model.save(str(save_path), save_format="h5")
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def from_keras(cls, model_path: Union[str, Path]) -> "TensorFlowModel":
        """Keras 모델에서 생성"""
        return cls(model_path)

    @classmethod
    def from_saved_model(cls, model_path: Union[str, Path]) -> "TensorFlowModel":
        """SavedModel에서 생성"""
        return cls(model_path)
