"""
PyTorch Model - PyTorch 체크포인트 래퍼

PyTorch 모델 로드, 예측, 저장 및 GPU/Device 관리를 위한 래퍼 클래스
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from beanllm.infrastructure.ml.base import BaseMLModel

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
else:
    NDArray = Any  # noqa: N816

np: Optional[object] = None
try:
    import numpy as _np

    np = _np
except ImportError:
    pass


class PyTorchModel(BaseMLModel):
    """
    PyTorch 모델 래퍼

    체크포인트 로드, CUDA/CPU 자동 선택, inference 모드 지원.

    Example:
        ```python
        # 모델 로드
        model = PyTorchModel.from_checkpoint("model.pth")
        predictions = model.predict(data)

        # 추론 모드
        model.eval_mode()
        ```
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

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def load(self, model_path: Union[str, Path]) -> None:
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

        if self.model is not None:
            self.model.to(self.device)
        self.model_path = model_path

    def predict(self, inputs: Union[NDArray, Any], **kwargs: Any) -> NDArray:
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
        if np is not None and isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(self.device)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)

        # 추론 모드
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(inputs, **kwargs)

        # numpy로 변환
        if isinstance(outputs, torch.Tensor):
            return cast(NDArray, outputs.cpu().numpy())
        return cast(NDArray, outputs)

    def save(self, save_path: Union[str, Path], save_full_model: bool = False) -> None:
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

    def eval_mode(self) -> None:
        """평가 모드로 전환"""
        if self.model:
            self.model.eval()

    def train_mode(self) -> None:
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
