"""
RAG Debug Response DTOs - RAG 디버깅 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class DebugSessionResponse(BaseResponse):
    """
    디버그 세션 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    session_name: str
    vector_store_id: str
    num_documents: int
    num_embeddings: int
    embedding_dim: int
    status: str
    created_at: str
    metadata: Dict[str, Any] = {}


class AnalyzeEmbeddingsResponse(BaseResponse):
    """
    Embedding 분석 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    method: str
    num_clusters: int
    cluster_labels: List[int]
    cluster_sizes: Dict[int, int]
    outliers: List[int]
    reduced_embeddings: Optional[List[List[float]]] = None
    silhouette_score: Optional[float] = None
    metadata: Dict[str, Any] = {}


class ValidateChunksResponse(BaseResponse):
    """
    청크 검증 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    total_chunks: int
    valid_chunks: int
    issues: List[Dict[str, Any]]
    size_distribution: Dict[str, int]
    overlap_stats: Optional[Dict[str, Any]] = None
    duplicate_chunks: Optional[List[tuple]] = None
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}


class TuneParametersResponse(BaseResponse):
    """
    파라미터 튜닝 응답 DTO
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    parameters: Dict[str, Any]
    test_results: List[Dict[str, Any]]
    avg_score: float
    comparison_with_baseline: Optional[Dict[str, float]] = None
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}
