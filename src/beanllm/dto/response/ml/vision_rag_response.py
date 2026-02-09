"""
VisionRAGResponse - Vision RAG 응답 DTO
책임: Vision RAG 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class VisionRAGResponse(BaseResponse):
    """
    Vision RAG 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    # query 메서드 응답
    answer: Optional[str] = None
    sources: Optional[List[Any]] = None  # VectorSearchResult 타입

    # retrieve 메서드 응답
    results: Optional[List[Any]] = None  # VectorSearchResult 타입

    # batch_query 메서드 응답
    answers: Optional[List[str]] = None

    # 메타데이터
    metadata: Dict[str, Any] = {}
