"""
RAG Debug Response DTOs - RAG 디버깅 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class DebugSessionResponse(BaseResponse):
    """
    디버그 세션 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str
    session_name: str
    vector_store_id: str
    num_documents: int
    num_embeddings: int
    embedding_dim: int
    status: str
    created_at: str
    metadata: dict[str, object] = {}


class AnalyzeEmbeddingsResponse(BaseResponse):
    """
    Embedding 분석 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str
    method: str
    num_clusters: int
    cluster_labels: list[int]
    cluster_sizes: dict[int, int]
    outliers: list[int]
    reduced_embeddings: Optional[list[list[float]]] = None
    silhouette_score: Optional[float] = None
    metadata: dict[str, object] = {}


class ValidateChunksResponse(BaseResponse):
    """
    청크 검증 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str
    total_chunks: int
    valid_chunks: int
    issues: list[dict[str, object]]
    size_distribution: dict[str, int]
    overlap_stats: Optional[dict[str, object]] = None
    duplicate_chunks: Optional[list[tuple[str, ...]]] = None
    recommendations: list[str] = []
    metadata: dict[str, object] = {}


class TuneParametersResponse(BaseResponse):
    """
    파라미터 튜닝 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    session_id: str
    parameters: dict[str, object]
    test_results: list[dict[str, object]]
    avg_score: float
    comparison_with_baseline: Optional[dict[str, float]] = None
    recommendations: list[str] = []
    metadata: dict[str, object] = {}
