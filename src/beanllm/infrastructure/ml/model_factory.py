"""
ML Model Factory - 프레임워크 자동 감지 및 모델 로드

경로 확장자로 프레임워크를 자동 감지하여 적절한 래퍼를 생성합니다.
"""

from pathlib import Path
from typing import Any, Optional, Union

from beanllm.infrastructure.ml.base import BaseMLModel
from beanllm.infrastructure.ml.pytorch_model import PyTorchModel
from beanllm.infrastructure.ml.sklearn_model import SklearnModel
from beanllm.infrastructure.ml.tensorflow_model import TensorFlowModel


class MLModelFactory:
    """
    ML 모델 팩토리

    프레임워크를 자동으로 감지하여 적절한 래퍼 생성
    """

    @staticmethod
    def load(
        model_path: Union[str, Path],
        framework: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseMLModel:
        """
        모델 로드 (자동 감지)

        Args:
            model_path: 모델 경로
            framework: 프레임워크 (tf, torch, sklearn 또는 auto)
            **kwargs: 추가 파라미터

        Returns:
            ML 모델 인스턴스

        Example:
            ```python
            # 자동 감지
            model = MLModelFactory.load("model.h5")

            # 명시적 지정
            model = MLModelFactory.load("model.pth", framework="torch")
            ```
        """
        model_path = Path(model_path)

        if framework is None:
            framework = MLModelFactory._detect_framework(model_path)

        if framework == "tensorflow" or framework == "tf":
            return TensorFlowModel(model_path)
        elif framework == "pytorch" or framework == "torch":
            return PyTorchModel(model_path=model_path, **kwargs)
        elif framework == "sklearn":
            return SklearnModel.from_pickle(model_path)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    @staticmethod
    def _detect_framework(model_path: Path) -> str:
        """프레임워크 자동 감지"""
        suffix = model_path.suffix.lower()

        # TensorFlow
        if suffix in [".h5", ".hdf5"] or model_path.name == "saved_model":
            return "tensorflow"

        # PyTorch
        if suffix in [".pt", ".pth", ".ckpt"]:
            return "pytorch"

        # Scikit-learn
        if suffix in [".pkl", ".pickle", ".joblib"]:
            return "sklearn"

        # 디렉토리 체크 (SavedModel)
        if model_path.is_dir():
            if (model_path / "saved_model.pb").exists():
                return "tensorflow"

        raise ValueError(
            f"Cannot detect framework from path: {model_path}. "
            "Please specify framework explicitly."
        )


def load_ml_model(
    model_path: Union[str, Path],
    framework: Optional[str] = None,
    **kwargs: Any,
) -> BaseMLModel:
    """
    ML 모델 로드 (간편 함수)

    Args:
        model_path: 모델 경로
        framework: 프레임워크 (옵션)
        **kwargs: 추가 파라미터

    Returns:
        ML 모델 인스턴스

    Example:
        ```python
        model = load_ml_model("model.h5")
        predictions = model.predict(data)
        ```
    """
    return MLModelFactory.load(model_path, framework, **kwargs)
