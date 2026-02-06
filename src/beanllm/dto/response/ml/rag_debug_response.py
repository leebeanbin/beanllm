"""
RAG Debug Response DTOs - RAG 디버깅 응답 데이터 전송 객체
책임: 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DebugSessionResponse:
    """
    디버그 세션 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    session_id: str
    session_name: str
    vector_store_id: str
    num_documents: int
    num_embeddings: int
    embedding_dim: int
    status: str
    created_at: str
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AnalyzeEmbeddingsResponse:
    """
    Embedding 분석 응답 DTO
    """

    session_id: str
    method: str
    num_clusters: int
    cluster_labels: List[int]
    cluster_sizes: Dict[int, int]
    outliers: List[int]  # Indices of outlier embeddings
    reduced_embeddings: Optional[List[List[float]]] = None  # 2D/3D coordinates
    silhouette_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidateChunksResponse:
    """
    청크 검증 응답 DTO
    """

    session_id: str
    total_chunks: int
    valid_chunks: int
    issues: List[Dict[str, Any]]  # [{"type": "size", "chunk_id": ..., "details": ...}]
    size_distribution: Dict[str, int]  # {"0-200": 10, "200-500": 50, ...}
    overlap_stats: Optional[Dict[str, Any]] = None
    duplicate_chunks: Optional[List[tuple]] = None  # [(chunk_id1, chunk_id2), ...]
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TuneParametersResponse:
    """
    파라미터 튜닝 응답 DTO
    """

    session_id: str
    parameters: Dict[str, Any]
    test_results: List[Dict[str, Any]]  # Results for each test query
    avg_score: float
    comparison_with_baseline: Optional[Dict[str, float]] = None
    recommendations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}
