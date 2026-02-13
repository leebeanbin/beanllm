"""
Vision Embeddings - 이미지 임베딩 및 멀티모달 임베딩

Re-export hub. Implementations live in:
- embedding_clip: CLIPEmbedding
- embedding_siglip: SigLIPEmbedding
- embedding_mobileclip: MobileCLIPEmbedding
- embedding_multimodal: MultimodalEmbedding
"""

from __future__ import annotations

from beanllm.domain.embeddings import BaseEmbedding
from beanllm.domain.vision.embedding_clip import CLIPEmbedding
from beanllm.domain.vision.embedding_mobileclip import MobileCLIPEmbedding
from beanllm.domain.vision.embedding_multimodal import MultimodalEmbedding
from beanllm.domain.vision.embedding_siglip import SigLIPEmbedding


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
            f"Unknown model: {model}. Supported models: clip, siglip, mobileclip, multimodal"
        )


__all__ = [
    "CLIPEmbedding",
    "SigLIPEmbedding",
    "MobileCLIPEmbedding",
    "MultimodalEmbedding",
    "create_vision_embedding",
]
