"""
Base Vision Task Model - 비전 태스크 모델 추상 클래스

beanLLM의 모든 비전 태스크 모델 래퍼는 이 추상 클래스를 상속해야 합니다.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import numpy as np


class BaseVisionTaskModel(ABC):
    """
    비전 태스크 모델 베이스 클래스

    SAM, Florence-2, YOLO 등 비전 태스크 전용 모델을 통합하기 위한
    공통 인터페이스를 정의합니다.

    BaseEmbedding과의 차이:
    - BaseEmbedding: 이미지를 임베딩 벡터로 변환 (CLIP, SigLIP 등)
    - BaseVisionTaskModel: 특정 비전 태스크 수행 (Segmentation, Detection, Captioning 등)

    Example:
        ```python
        from beanllm.domain.vision import BaseVisionTaskModel

        class MyVisionModel(BaseVisionTaskModel):
            def _load_model(self):
                # 모델 로딩 로직
                self._model = load_my_model()

            def predict(self, image, **kwargs):
                # 예측 로직
                self._load_model()
                return self._model(image)
        ```
    """

    @abstractmethod
    def _load_model(self):
        """
        모델 로딩 (lazy loading)

        모델을 메모리에 로드합니다. 이 메서드는 첫 예측 시점에 호출되어야 합니다.
        self._model이 None이 아니면 조기 반환하여 중복 로딩을 방지합니다.

        Example:
            ```python
            def _load_model(self):
                if self._model is not None:
                    return

                from transformers import AutoModel
                self._model = AutoModel.from_pretrained("model-name")
                self._model.to(self.device)

                logger.info("Model loaded successfully")
            ```
        """
        pass

    @abstractmethod
    def predict(self, image: Union[str, Path, np.ndarray], **kwargs) -> Any:
        """
        예측 실행

        이미지에 대해 모델의 주요 태스크를 실행합니다.
        각 모델마다 태스크와 파라미터가 다르므로 **kwargs로 받습니다.

        Args:
            image: 이미지 (파일 경로 또는 numpy array)
            **kwargs: 모델별 추가 파라미터

        Returns:
            모델별 예측 결과
            - SAM: {"masks": np.ndarray, "scores": List[float], ...}
            - Florence-2: str (caption), List[Dict] (objects), etc.
            - YOLO: List[Dict] (detections/segments)

        Example:
            ```python
            # SAM
            result = model.predict(
                image="photo.jpg",
                points=[[500, 375]],
                labels=[1]
            )

            # Florence-2
            caption = model.predict(
                image="photo.jpg",
                task="caption"
            )

            # YOLO
            detections = model.predict(
                image="photo.jpg",
                conf=0.25
            )
            ```
        """
        pass

    def __repr__(self) -> str:
        """모델 정보 출력"""
        return f"{self.__class__.__name__}()"
