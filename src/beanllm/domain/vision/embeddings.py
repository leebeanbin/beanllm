"""
Vision Embeddings - 이미지 임베딩 및 멀티모달 임베딩
"""

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

if TYPE_CHECKING:
    from beanllm.domain.protocols import CacheProtocol, EventLoggerProtocol, RateLimiterProtocol

from beanllm.domain.embeddings import BaseEmbedding

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


class SigLIPEmbedding(BaseEmbedding):
    """
    SigLIP 2 임베딩 (Google DeepMind, 2025)

    CLIP을 능가하는 최신 비전-언어 모델.
    Sigmoid loss + self-distillation + 다국어 지원.

    Features:
    - CLIP 능가하는 성능
    - 다국어 zero-shot 분류
    - Self-distillation으로 향상된 semantic understanding
    - 개선된 localization 및 dense features

    Example:
        embed = SigLIPEmbedding()

        # 텍스트 임베딩
        text_vec = embed.embed_sync(["a cat"])

        # 이미지 임베딩
        image_vec = embed.embed_images(["cat.jpg"])

        # 유사도 계산
        similarity = embed.similarity(text_vec[0], image_vec[0])
    """

    def __init__(
        self, model: str = "google/siglip-so400m-patch14-384", device: Optional[str] = None
    ):
        """
        Args:
            model: SigLIP 모델 이름 (기본: SigLIP-SO400M)
            device: 디바이스 (cuda, cpu 등)
        """
        super().__init__(model=model)
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
        """
        텍스트 임베딩

        Args:
            texts: 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
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
            outputs = self._model.get_text_features(**inputs)

        # Normalize
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)

        return cast(List[List[float]], outputs.cpu().numpy().tolist())

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
            outputs = self._model.get_image_features(**inputs)

        # Normalize
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


class MultimodalEmbedding(BaseEmbedding):
    """
    멀티모달 임베딩

    텍스트와 이미지를 함께 처리

    Example:
        embed = MultimodalEmbedding()

        # 텍스트 + 이미지 임베딩
        vec = embed.embed_multimodal(
            text="a cat sitting on a mat",
            image="cat.jpg"
        )
    """

    def __init__(
        self,
        text_model: str = "text-embedding-3-small",
        vision_model: str = "openai/clip-vit-base-patch32",
        fusion_method: str = "concat",  # concat, average, weighted
    ):
        """
        Args:
            text_model: 텍스트 임베딩 모델
            vision_model: 비전 임베딩 모델
            fusion_method: 융합 방법 (concat, average, weighted)
        """
        super().__init__(model=text_model)
        from beanllm.domain.embeddings import Embedding

        self.text_embedder: BaseEmbedding = cast(BaseEmbedding, Embedding(model=text_model))
        self.vision_embedder = CLIPEmbedding(model=vision_model)
        self.fusion_method = fusion_method

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트만 임베딩"""
        return cast(List[List[float]], self.text_embedder.embed_sync(texts))

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """비동기 텍스트 임베딩"""
        return self.embed_sync(texts)

    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path]] = None,
        text_weight: float = 0.5,
    ) -> List[float]:
        """
        멀티모달 임베딩

        Args:
            text: 텍스트 (옵션)
            image: 이미지 경로 (옵션)
            text_weight: 텍스트 가중치 (fusion_method='weighted'일 때)

        Returns:
            임베딩 벡터
        """
        if not text and not image:
            raise ValueError("At least one of text or image must be provided")

        vectors: List[tuple[str, List[float]]] = []

        # 텍스트 임베딩
        if text:
            text_vec = cast(List[float], self.text_embedder.embed_sync([text])[0])
            vectors.append(("text", text_vec))

        # 이미지 임베딩
        if image:
            image_vec = self.vision_embedder.embed_images([image])[0]
            vectors.append(("vision", image_vec))

        # 융합
        if len(vectors) == 1:
            return vectors[0][1]

        return self._fuse_vectors(vectors, text_weight)

    def _fuse_vectors(
        self, vectors: List[tuple[str, List[float]]], text_weight: float
    ) -> List[float]:
        """
        벡터 융합

        Args:
            vectors: [(type, vector), ...] 리스트
            text_weight: 텍스트 가중치

        Returns:
            융합된 벡터
        """
        try:
            import numpy as np
        except ImportError:
            # numpy 없으면 concat만 지원
            if self.fusion_method == "concat":
                return [v for _, vec in vectors for v in vec]
            else:
                raise ImportError("numpy required for fusion methods other than 'concat'")

        if self.fusion_method == "concat":
            # 연결
            return [v for _, vec in vectors for v in vec]

        elif self.fusion_method == "average":
            # 평균
            arrays = [np.array(vec) for _, vec in vectors]
            return cast(List[float], np.mean(arrays, axis=0).tolist())

        elif self.fusion_method == "weighted":
            # 가중 평균
            text_vecs = [vec for vec_type, vec in vectors if vec_type == "text"]
            vision_vecs = [vec for vec_type, vec in vectors if vec_type == "vision"]

            if text_vecs and vision_vecs:
                text_arr = np.array(text_vecs[0])
                vision_arr = np.array(vision_vecs[0])
                fused = text_weight * text_arr + (1 - text_weight) * vision_arr
                return cast(List[float], fused.tolist())
            else:
                # 하나만 있으면 그대로 반환
                return vectors[0][1]

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


# 편의 함수
def create_vision_embedding(model: str = "clip", **kwargs) -> BaseEmbedding:
    """
    Vision 임베딩 생성 (간편 함수)

    Args:
        model: 모델 타입
            - "clip": OpenAI CLIP
            - "siglip": Google SigLIP 2 (CLIP 능가, 2025)
            - "mobileclip": Apple MobileCLIP2 (모바일 최적화, 2025)
            - "multimodal": 멀티모달 임베딩
        **kwargs: 추가 파라미터

    Returns:
        임베딩 인스턴스

    Example:
        # CLIP (기본)
        embed = create_vision_embedding("clip")

        # SigLIP 2 (최신, 고성능)
        embed = create_vision_embedding("siglip")

        # MobileCLIP2 (모바일 최적화)
        embed = create_vision_embedding("mobileclip", model_size="s2")

        # Multimodal
        embed = create_vision_embedding("multimodal", fusion_method="concat")
    """
    if model == "clip":
        return CLIPEmbedding(**kwargs)
    elif model == "siglip":
        return SigLIPEmbedding(**kwargs)
    elif model == "mobileclip":
        return MobileCLIPEmbedding(**kwargs)
    elif model == "multimodal":
        return MultimodalEmbedding(**kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model}. " f"Supported models: clip, siglip, mobileclip, multimodal"
        )
