"""
StateGraphResponse - StateGraph 응답 DTO
책임: StateGraph 응답 데이터만 전달
"""

from __future__ import annotations

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class StateGraphResponse(BaseResponse):
    """
    StateGraph 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    final_state: dict[str, object]
    execution_id: str
    nodes_executed: list[str] = []
    iterations: int = 0
    metadata: dict[str, object] = {}
