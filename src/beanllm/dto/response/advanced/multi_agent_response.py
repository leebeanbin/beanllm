"""
MultiAgentResponse - Multi-Agent 응답 DTO
책임: Multi-Agent 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class MultiAgentResponse(BaseResponse):
    """
    Multi-Agent 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    final_result: Any
    strategy: str
    intermediate_results: Optional[List[Any]] = None
    all_steps: Optional[List[Any]] = None
    metadata: Dict[str, Any] = {}
