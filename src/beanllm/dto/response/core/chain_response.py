"""
ChainResponse - Chain 응답 DTO
책임: Chain 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse


class ChainResponse(BaseResponse):
    """
    Chain 응답 DTO

    책임:
    - 데이터 구조 정의만
    - 변환 로직 없음
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: str
    steps: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None
