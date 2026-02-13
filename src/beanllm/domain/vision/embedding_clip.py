"""
CLIP embedding implementation.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from beanllm.domain.embeddings import BaseEmbedding

if TYPE_CHECKING:
    from beanllm.domain.protocols import CacheProtocol, EventLoggerProtocol, RateLimiterProtocol

# 환경변수로 분산 모드 활성화 여부 확인
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"


class CLIPEmbedding(BaseEmbedding):
    """
    CLIP 임베딩

    텍스트와 이미지를 동일한 벡터 공간에 임베딩

    Example:
        embed = CLIPEmbedding()

        # 텍스트 임베딩
        text_vec = embed.embed_sync(["a cat"])

        # 이미지 임베딩
        image_vec = embed.embed_images(["cat.jpg"])

        # 유사도 계산
        similarity = embed.similarity(text_vec[0], image_vec[0])
    """

    def __init__(
        self,
        model: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache: Optional["CacheProtocol"] = None,
        rate_limiter: Optional["RateLimiterProtocol"] = None,
        event_logger: Optional["EventLoggerProtocol"] = None,
    ):
        """
        Args:
            model: CLIP 모델 이름
            device: 디바이스 (cuda, cpu 등)
            cache: 캐시 프로토콜 (옵션, Service layer에서 주입)
            rate_limiter: Rate Limiter 프로토콜 (옵션, Service layer에서 주입)
            event_logger: Event Logger 프로토콜 (옵션, Service layer에서 주입)
        """
        super().__init__(model=model)
        self.device = device or "cpu"
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._event_logger = event_logger

    def _load_model(self):
        """모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError:
                raise ImportError("transformers 및 torch 필요:\npip install transformers torch")

            self._processor = CLIPProcessor.from_pretrained(self.model)
            self._model = CLIPModel.from_pretrained(self.model)
            self._model.to(self.device)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 임베딩

        Args:
            texts: 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        import asyncio

        # 비동기 래퍼 사용
        if USE_DISTRIBUTED:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 동기 실행
                    return self._embed_sync_internal(texts)
                else:
                    return loop.run_until_complete(self._embed_async(texts))
            except RuntimeError:
                return asyncio.run(self._embed_async(texts))
        else:
            return self._embed_sync_internal(texts)

    def _embed_sync_internal(self, texts: List[str]) -> List[List[float]]:
        """내부 동기 임베딩 메서드"""
        self._load_model()

        import torch

        # Type assertions after load
        assert self._processor is not None
        assert self._model is not None

        # 입력 처리
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 임베딩 생성
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return cast(List[List[float]], text_features.cpu().numpy().tolist())

    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        """비동기 임베딩 메서드 (Rate Limiting + Caching)"""
        # 캐싱: 텍스트 해시 기반 임베딩 캐싱 (옵션)
        cache_key = ""
        if self._cache is not None:
            text_hash = hashlib.md5("|".join(texts).encode()).hexdigest()
            cache_key = f"vision_embedding:clip:text:{text_hash}"
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                return cast(List[List[float]], cached_result)

        # Rate Limiting: Vision 임베딩 모델 호출 (옵션)
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire("vision:embedding")

        # 임베딩 생성
        result = self._embed_sync_internal(texts)

        # 캐시 저장 (옵션)
        if self._cache is not None:
            await self._cache.set(cache_key, result, ttl=7200)

        # 이벤트 발행 (옵션)
        if self._event_logger is not None:
            await self._event_logger.log_event(
                "vision_embedding.clip.text",
                {"text_count": len(texts), "model": self.model},
            )

        return result

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """비동기 텍스트 임베딩"""
        return self.embed_sync(texts)

    def embed_images(self, images: List[Union[str, Path]], **kwargs: Any) -> List[List[float]]:
        """
        이미지 임베딩

        Args:
            images: 이미지 파일 경로 리스트

        Returns:
            임베딩 벡터 리스트

        Example:
            vecs = embed.embed_images(["cat.jpg", "dog.jpg"])
        """
        self._load_model()

        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow 필요:\npip install pillow")

        # Type assertions after load
        assert self._processor is not None
        assert self._model is not None

        # 이미지 로드
        pil_images = [Image.open(img) for img in images]

        # 입력 처리
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 임베딩 생성
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return cast(List[List[float]], image_features.cpu().numpy().tolist())

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        코사인 유사도

        Args:
            vec1: 벡터 1
            vec2: 벡터 2

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        try:
            import numpy as np
        except ImportError:
            # numpy 없으면 수동 계산
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b))  # 이미 normalized됨
