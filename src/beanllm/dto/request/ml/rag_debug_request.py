"""
RAG Debug Request DTOs - RAG 디버깅 요청 데이터 전송 객체
책임: 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True, kw_only=True)
class StartDebugSessionRequest:
    """
    디버그 세션 시작 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    """

    vector_store_id: str
    session_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class AnalyzeEmbeddingsRequest:
    """
    Embedding 분석 요청 DTO
    """

    session_id: str
    method: str = "umap"
    n_clusters: int = 5
    detect_outliers: bool = True
    sample_size: Optional[int] = None


@dataclass(slots=True, kw_only=True)
class ValidateChunksRequest:
    """
    청크 검증 요청 DTO
    """

    session_id: str
    check_size: bool = True
    check_overlap: bool = True
    check_metadata: bool = True
    check_duplicates: bool = True
    size_threshold: int = 1000


@dataclass(slots=True, kw_only=True)
class TuneParametersRequest:
    """
    파라미터 튜닝 요청 DTO
    """

    session_id: str
    parameters: Dict[str, Any]
    test_queries: List[str] = field(default_factory=list)
