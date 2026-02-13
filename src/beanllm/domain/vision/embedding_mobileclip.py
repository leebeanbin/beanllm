"""
MobileCLIP embedding implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union, cast

from beanllm.domain.embeddings import BaseEmbedding


class MobileCLIPEmbedding(BaseEmbedding):
    """
    MobileCLIP2 임베딩 (Apple, 2025)

    모바일 및 엣지 디바이스에 최적화된 경량 비전-언어 모델.
    SigLIP-SO400M과 동급 성능을 2배 적은 파라미터로 달성.

    Features:
    - 모바일 최적화 (2x fewer parameters)
    - SigLIP-SO400M 동급 성능
    - 2.5x faster inference on mobile
    - 효율적인 아키텍처

    Example:
        # 모바일/엣지 디바이스용
        embed = MobileCLIPEmbedding(model_size="s2")

        # 이미지 임베딩 (모바일에서 빠름)
        image_vec = embed.embed_images(["cat.jpg"])
    """

    def __init__(self, model_size: str = "s2", device: Optional[str] = None):
        """
        Args:
            model_size: 모델 크기 (s0, s1, s2 - s2가 가장 성능 좋음)
            device: 디바이스 (cuda, cpu 등)
        """
        # MobileCLIP 모델 이름 매핑
        model_map = {
            "s0": "apple/mobileclip-s0",
            "s1": "apple/mobileclip-s1",
            "s2": "apple/mobileclip-s2",
        }
        model_name = model_map.get(model_size, model_map["s2"])

        super().__init__(model=model_name)
        self.model_size = model_size
        self.device = device or "cpu"
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _load_model(self) -> None:
        """모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError:
                raise ImportError("transformers 및 torch 필요:\npip install transformers torch")

            self._processor = AutoProcessor.from_pretrained(self.model)
            self._model = AutoModel.from_pretrained(self.model)
            self._model.to(self.device)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트 임베딩"""
        self._load_model()

        import torch

        # Type assertions after load
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)

        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return cast(List[List[float]], outputs.cpu().numpy().tolist())

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """비동기 텍스트 임베딩"""
        return self.embed_sync(texts)

    def embed_images(self, images: List[Union[str, Path]], **kwargs: Any) -> List[List[float]]:
        """이미지 임베딩 (모바일 최적화)"""
        self._load_model()

        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow 필요:\npip install pillow")

        # Type assertions after load
        assert self._processor is not None
        assert self._model is not None

        pil_images = [Image.open(img) for img in images]

        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return cast(List[List[float]], outputs.cpu().numpy().tolist())

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도"""
        try:
            import numpy as np
        except ImportError:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b))
