"""
Multimodal embedding implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union, cast

from beanllm.domain.embeddings import BaseEmbedding
from beanllm.domain.vision.embedding_clip import CLIPEmbedding


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
