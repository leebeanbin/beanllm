"""
RAGResponse - RAG 응답 DTO
책임: RAG 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class RAGResponse(BaseResponse):
    """
    RAG 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    answer: str
    sources: List[Any]  # VectorSearchResult 타입
    metadata: Dict[str, Any] = {}
