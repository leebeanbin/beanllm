"""
VisionRAGResponse - Vision RAG 응답 DTO
책임: Vision RAG 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class VisionRAGResponse(BaseResponse):
    """
    Vision RAG 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    # query 메서드 응답
    answer: Optional[str] = None
    sources: Optional[list[object]] = None

    # retrieve 메서드 응답
    results: Optional[list[object]] = None

    # batch_query 메서드 응답
    answers: Optional[list[str]] = None

    # 메타데이터
    metadata: dict[str, object] = {}
