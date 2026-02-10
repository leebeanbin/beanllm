"""
ChainResponse - Chain 응답 DTO
책임: Chain 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class ChainResponse(BaseResponse):
    """
    Chain 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    output: str
    steps: list[dict[str, object]] = []
    metadata: dict[str, object] = {}
    success: bool = True
    error: Optional[str] = None
