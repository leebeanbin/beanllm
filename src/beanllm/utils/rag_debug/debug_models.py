"""RAG Debug - 데이터 모델 (EmbeddingInfo, SimilarityInfo)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EmbeddingInfo:
    """임베딩 정보"""

    text: str
    vector: List[float]
    dimension: int
    norm: float  # 벡터 크기
    preview: List[float]  # 앞 10개 값


@dataclass
class SimilarityInfo:
    """유사도 정보"""

    text1: str
    text2: str
    cosine_similarity: float
    euclidean_distance: float
    interpretation: str  # 해석
