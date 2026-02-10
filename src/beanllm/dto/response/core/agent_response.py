"""
AgentResponse - 에이전트 응답 DTO
책임: 에이전트 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class AgentResponse(BaseResponse):
    """
    에이전트 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    answer: str
    steps: list[object] = []
    total_steps: int = 0
    success: bool = True
    error: Optional[str] = None
