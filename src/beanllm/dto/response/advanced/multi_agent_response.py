"""
MultiAgentResponse - Multi-Agent 응답 DTO
책임: Multi-Agent 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class MultiAgentResponse(BaseResponse):
    """
    Multi-Agent 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    final_result: object
    strategy: str
    intermediate_results: Optional[list[object]] = None
    all_steps: Optional[list[object]] = None
    metadata: dict[str, object] = {}
