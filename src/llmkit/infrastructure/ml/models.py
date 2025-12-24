"""
ML Models Integration - TensorFlow, PyTorch, Scikit-learn 등 머신러닝 모델 통합
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None


class BaseMLModel(ABC):
    """
    ML 모델 베이스 클래스

    모든 ML 프레임워크의 통합 인터페이스
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Args:
            model_path: 모델 파일 경로 (옵션)
        """
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def load(self, model_path: Union[str, Path]):
        """모델 로드"""
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """예측"""
        pass

    @abstractmethod
    def save(self, save_path: Union[str, Path]):
        """모델 저장"""
        pass


class TensorFlowModel(BaseMLModel):
    """
    TensorFlow 모델 래퍼

    Example:
        # Keras 모델 로드
        model = TensorFlowModel.from_keras("model.h5")
        predictions = model.predict(data)

        # SavedModel 로드
        model = TensorFlowModel.from_saved_model("saved_model/")
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        super().__init__(model_path)
        if model_path:
            self.load(model_path)

    def load(self, model_path: Union[str, Path]):
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
        self, inputs: Union[np.ndarray, List], batch_size: Optional[int] = None, **kwargs
    ) -> np.ndarray:
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

    def save(self, save_path: Union[str, Path], format: str = "tf"):
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


class PyTorchModel(BaseMLModel):
    """
    PyTorch 모델 래퍼

    Example:
        # 모델 로드
        model = PyTorchModel.from_checkpoint("model.pth")
        predictions = model.predict(data)

        # 추론 모드
        model.eval_mode()
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        super().__init__(model_path)
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")
        self.model = model

        if model_path:
            self.load(model_path)

    def _is_cuda_available(self) -> bool:
        """CUDA 사용 가능 여부"""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def load(self, model_path: Union[str, Path]):
        """
        모델 로드

        Args:
            model_path: 체크포인트 경로
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch 필요:\npip install torch")

        checkpoint = torch.load(str(model_path), map_location=self.device)

        # 체크포인트 형식 확인
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # state_dict가 딕셔너리에 있는 경우
            if self.model is None:
                raise ValueError(
                    "Model architecture not provided. "
                    "Pass model instance or use from_checkpoint_with_model()."
                )
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 모델 전체가 저장된 경우
            self.model = checkpoint

        self.model.to(self.device)
        self.model_path = model_path

    def predict(self, inputs: Union[np.ndarray, Any], **kwargs) -> np.ndarray:
        """
        예측

        Args:
            inputs: 입력 데이터
            **kwargs: 추가 파라미터

        Returns:
            예측 결과
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")

        if self.model is None:
            raise ValueError("Model not loaded")

        # numpy를 tensor로 변환
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)

        # 추론 모드
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(inputs, **kwargs)

        # numpy로 변환
        if isinstance(outputs, torch.Tensor):
            return outputs.cpu().numpy()
        else:
            return outputs

    def save(self, save_path: Union[str, Path], save_full_model: bool = False):
        """
        모델 저장

        Args:
            save_path: 저장 경로
            save_full_model: 전체 모델 저장 여부 (False면 state_dict만)
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")

        if self.model is None:
            raise ValueError("No model to save")

        if save_full_model:
            # 전체 모델 저장
            torch.save(self.model, str(save_path))
        else:
            # state_dict만 저장
            torch.save({"model_state_dict": self.model.state_dict()}, str(save_path))

    def eval_mode(self):
        """평가 모드로 전환"""
        if self.model:
            self.model.eval()

    def train_mode(self):
        """학습 모드로 전환"""
        if self.model:
            self.model.train()

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Union[str, Path], device: Optional[str] = None
    ) -> "PyTorchModel":
        """체크포인트에서 생성"""
        return cls(model_path=checkpoint_path, device=device)

    @classmethod
    def from_checkpoint_with_model(
        cls, checkpoint_path: Union[str, Path], model: Any, device: Optional[str] = None
    ) -> "PyTorchModel":
        """체크포인트 + 모델 아키텍처로 생성"""
        instance = cls(model=model, device=device)
        instance.load(checkpoint_path)
        return instance


class SklearnModel(BaseMLModel):
    """
    Scikit-learn 모델 래퍼

    Example:
        # 모델 로드
        model = SklearnModel.from_pickle("model.pkl")
        predictions = model.predict(data)

        # 모델 학습
        model = SklearnModel()
        model.fit(X_train, y_train)
        model.save("model.pkl")
    """

    def __init__(self, model: Optional[Any] = None):
        super().__init__()
        self.model = model

    def load(self, model_path: Union[str, Path]):
        """
        모델 로드 (pickle 또는 joblib)

        Args:
            model_path: 모델 파일 경로
        """
        model_path = Path(model_path)

        # joblib 시도
        try:
            import joblib

            self.model = joblib.load(str(model_path))
            self.model_path = model_path
            return
        except Exception:
            pass

        # pickle 시도
        try:
            import pickle

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_path = model_path
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def predict(self, inputs: Union[np.ndarray, List], **kwargs) -> np.ndarray:
        """
        예측

        Args:
            inputs: 입력 데이터
            **kwargs: 추가 파라미터

        Returns:
            예측 결과
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        return self.model.predict(inputs, **kwargs)

    def predict_proba(self, inputs: Union[np.ndarray, List], **kwargs) -> np.ndarray:
        """
        확률 예측 (분류 모델)

        Args:
            inputs: 입력 데이터
            **kwargs: 추가 파라미터

        Returns:
            확률 예측
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba")

        return self.model.predict_proba(inputs, **kwargs)

    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List], **kwargs):
        """
        모델 학습

        Args:
            X: 학습 데이터
            y: 레이블
            **kwargs: 추가 파라미터
        """
        if self.model is None:
            raise ValueError("Model not initialized")

        self.model.fit(X, y, **kwargs)

    def save(self, save_path: Union[str, Path], use_joblib: bool = True):
        """
        모델 저장

        Args:
            save_path: 저장 경로
            use_joblib: joblib 사용 여부 (False면 pickle)
        """
        if self.model is None:
            raise ValueError("No model to save")

        save_path = Path(save_path)

        if use_joblib:
            try:
                import joblib

                joblib.dump(self.model, str(save_path))
            except ImportError:
                # joblib 없으면 pickle 사용
                import pickle

                with open(save_path, "wb") as f:
                    pickle.dump(self.model, f)
        else:
            import pickle

            with open(save_path, "wb") as f:
                pickle.dump(self.model, f)

    @classmethod
    def from_pickle(cls, model_path: Union[str, Path]) -> "SklearnModel":
        """Pickle 파일에서 생성"""
        instance = cls()
        instance.load(model_path)
        return instance

    @classmethod
    def from_estimator(cls, estimator: Any) -> "SklearnModel":
        """Scikit-learn estimator에서 생성"""
        return cls(model=estimator)


# ML 모델 팩토리
class MLModelFactory:
    """
    ML 모델 팩토리

    프레임워크를 자동으로 감지하여 적절한 래퍼 생성
    """

    @staticmethod
    def load(
        model_path: Union[str, Path], framework: Optional[str] = None, **kwargs
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
            # 자동 감지
            model = MLModelFactory.load("model.h5")

            # 명시적 지정
            model = MLModelFactory.load("model.pth", framework="torch")
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
            f"Cannot detect framework from path: {model_path}. Please specify framework explicitly."
        )


# 편의 함수
def load_ml_model(
    model_path: Union[str, Path], framework: Optional[str] = None, **kwargs
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
        model = load_ml_model("model.h5")
        predictions = model.predict(data)
    """
    return MLModelFactory.load(model_path, framework, **kwargs)
