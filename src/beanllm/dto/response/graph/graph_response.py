"""
GraphResponse - Graph 응답 DTO
책임: Graph 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class GraphResponse(BaseResponse):
    """
    Graph 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    final_state: dict[str, object]
    metadata: dict[str, object] = {}
    cache_stats: Optional[dict[str, object]] = None
    visited_nodes: list[str] = []
    iterations: int = 0
