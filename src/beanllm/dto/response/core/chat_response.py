"""
ChatResponse - 채팅 응답 DTO
책임: 채팅 응답 데이터만 전달
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse
from beanllm.dto.shared_types import TokenUsage


class ChatResponse(BaseResponse):
    """
    채팅 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    model: str
    provider: str
    usage: Optional[TokenUsage] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    @classmethod
    def from_provider_response(
        cls, provider_response: Dict[str, Any], **kwargs: Any
    ) -> "ChatResponse":
        """
        Provider 응답을 ChatResponse로 변환

        책임: 데이터 변환만 (비즈니스 로직 없음)

        Args:
            provider_response: Provider 응답 딕셔너리
            **kwargs: model (str), provider (str) 필수
        """
        return cls(
            content=provider_response.get("content", ""),
            model=kwargs["model"],
            provider=kwargs["provider"],
            usage=provider_response.get("usage"),
            finish_reason=provider_response.get("finish_reason"),
            raw_response=provider_response,
        )
