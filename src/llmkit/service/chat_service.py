"""
IChatService - 채팅 서비스 인터페이스
SOLID 원칙:
- ISP: 채팅 관련 메서드만 포함 (작은 인터페이스)
- DIP: 인터페이스에 의존하도록 설계
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..dto.request.chat_request import ChatRequest
from ..dto.response.chat_response import ChatResponse


class IChatService(ABC):
    """
    채팅 서비스 인터페이스

    책임:
    - 채팅 비즈니스 로직 정의만 (구현 없음)
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: 채팅 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        채팅 요청 처리

        Args:
            request: 채팅 요청 DTO

        Returns:
            ChatResponse: 채팅 응답 DTO

        책임:
            - 비즈니스 로직만 (파라미터 변환, Provider 호출 등)
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        스트리밍 채팅 요청 처리

        Args:
            request: 채팅 요청 DTO

        Yields:
            str: 스트리밍 청크

        책임:
            - 스트리밍 비즈니스 로직만
            - 검증, 에러 처리 없음
        """
        pass
