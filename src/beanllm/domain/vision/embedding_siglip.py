"""
SigLIP embedding implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union, cast

from beanllm.domain.embeddings import BaseEmbedding


class SigLIPEmbedding(BaseEmbedding):
    """
    SigLIP 2 임베딩 (Google DeepMind, 2025).
    CLIP을 능가하는 최신 비전-언어 모델.
    """

    def __init__(
        self, model: str = "google/siglip-so400m-patch14-384", device: Optional[str] = None
    ):
        super().__init__(model=model)
        self.device = device or "cpu"
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _load_model(self) -> None:
        if self._model is None:
            try:
                from transformers import AutoModel, AutoProcessor
            except ImportError:
                raise ImportError("transformers 및 torch 필요:\npip install transformers torch")
            self._processor = AutoProcessor.from_pretrained(self.model)
            self._model = AutoModel.from_pretrained(self.model)
            self._model.to(self.device)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        import torch

        assert self._processor is not None
        assert self._model is not None
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return cast(List[List[float]], outputs.cpu().numpy().tolist())

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embed_sync(texts)

    def embed_images(self, images: List[Union[str, Path]], **kwargs: Any) -> List[List[float]]:
        self._load_model()
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow 필요:\npip install pillow")
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
