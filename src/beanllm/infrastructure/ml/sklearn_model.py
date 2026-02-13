"""
Scikit-learn Model - Pickle/Joblib 래퍼

Scikit-learn 모델 로드, 예측, 학습, 저장 및 HMAC 서명 검증
"""

import hashlib
import hmac
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

from beanllm.infrastructure.ml.base import BaseMLModel
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from numpy import ndarray as NDArray
else:
    NDArray = Any  # noqa: N816

logger = get_logger(__name__)

# 보안: 모델 서명용 비밀 키 (환경변수에서 로드)
MODEL_SIGNATURE_KEY = os.getenv("MODEL_SIGNATURE_KEY", "change-this-secret-key-in-production")


class SklearnModel(BaseMLModel):
    """
    Scikit-learn 모델 래퍼

    Pickle/Joblib 형식 지원, HMAC 서명 검증, fit/predict_proba 지원.

    Example:
        ```python
        # 모델 로드
        model = SklearnModel.from_pickle("model.pkl")
        predictions = model.predict(data)

        # 모델 학습
        model = SklearnModel()
        model.fit(X_train, y_train)
        model.save("model.pkl")
        ```
    """

    def __init__(self, model: Optional[Any] = None):
        super().__init__()
        self.model = model

    def load(self, model_path: Union[str, Path], verify_signature: bool = True) -> None:
        """
        모델 로드 (pickle 또는 joblib) - HMAC 서명 검증 포함

        Args:
            model_path: 모델 파일 경로
            verify_signature: 서명 검증 여부 (기본: True)

        Warning:
            pickle/joblib 역직렬화는 보안 위험이 있습니다.
            신뢰할 수 있는 소스의 모델만 로드하세요.
        """
        model_path = Path(model_path)

        # 서명 검증 (보안 강화)
        if verify_signature:
            sig_path = Path(f"{model_path}.sig")
            if not sig_path.exists():
                logger.warning(
                    f"No signature file found for {model_path}. "
                    "Set verify_signature=False to skip verification."
                )
                raise ValueError(
                    f"Signature file {sig_path} not found. Model integrity cannot be verified."
                )

            # 파일 내용 읽기
            with open(model_path, "rb") as f:
                model_bytes = f.read()

            # 서명 읽기
            with open(sig_path, "r") as f:
                expected_sig = f.read().strip()

            # 서명 계산
            actual_sig = hmac.new(
                MODEL_SIGNATURE_KEY.encode(), model_bytes, hashlib.sha256
            ).hexdigest()

            # 서명 검증 (타이밍 공격 방지)
            if not hmac.compare_digest(expected_sig, actual_sig):
                raise ValueError(
                    f"Signature verification failed for {model_path}! "
                    "Model may be tampered or corrupted."
                )

            logger.info(f"Signature verified successfully for {model_path}")

        # 보안 경고
        logger.warning("Loading model using pickle/joblib. Only load models from trusted sources!")

        # joblib 시도
        try:
            import joblib

            self.model = joblib.load(str(model_path))
            self.model_path = model_path
            return
        except Exception as e:
            logger.debug(f"Failed to load model with joblib (trying pickle): {e}")

        # pickle 시도
        try:
            import pickle

            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.model_path = model_path
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e

    def predict(self, inputs: Union[NDArray, List], **kwargs: Any) -> NDArray:
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

    def predict_proba(self, inputs: Union[NDArray, List], **kwargs: Any) -> NDArray:
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

    def fit(
        self,
        X: Union[NDArray, List],
        y: Union[NDArray, List],
        **kwargs: Any,
    ) -> None:
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

    def save(
        self,
        save_path: Union[str, Path],
        use_joblib: bool = True,
        sign: bool = True,
    ) -> None:
        """
        모델 저장 - HMAC 서명 생성 포함

        Args:
            save_path: 저장 경로
            use_joblib: joblib 사용 여부 (False면 pickle)
            sign: 서명 생성 여부 (기본: True)
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

        # 서명 생성 (무결성 보호)
        if sign:
            with open(save_path, "rb") as f:
                model_bytes = f.read()

            signature = hmac.new(
                MODEL_SIGNATURE_KEY.encode(), model_bytes, hashlib.sha256
            ).hexdigest()

            sig_path = Path(f"{save_path}.sig")
            with open(sig_path, "w") as f:
                f.write(signature)

            logger.info(f"Model saved with signature: {sig_path}")

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
