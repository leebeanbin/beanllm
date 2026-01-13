"""
ChatResponse - 채팅 응답 DTO
책임: 채팅 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ChatResponse:
    """
    채팅 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None

    @classmethod
    def from_provider_response(
        cls, provider_response: Dict[str, Any], model: str, provider: str
    ) -> "ChatResponse":
        """
        Provider 응답을 ChatResponse로 변환

        책임: 데이터 변환만 (비즈니스 로직 없음)
        """
        return cls(
            content=provider_response.get("content", ""),
            model=model,
            provider=provider,
            usage=provider_response.get("usage"),
            finish_reason=provider_response.get("finish_reason"),
            raw_response=provider_response,
        )
