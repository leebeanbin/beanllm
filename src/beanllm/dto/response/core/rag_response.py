"""
RAGResponse - RAG 응답 DTO
책임: RAG 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RAGResponse:
    """
    RAG 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    answer: str
    sources: List[Any]  # VectorSearchResult 타입
    metadata: Dict[str, Any]

    def __post_init__(self):
        """기본값 설정"""
        if self.metadata is None:
            self.metadata = {}
