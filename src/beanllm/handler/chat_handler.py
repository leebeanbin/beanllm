"""
ChatHandler - 채팅 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
- 비즈니스 로직 없음 (Service에 위임)
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.chat_request import ChatRequest
from ..dto.response.chat_response import ChatResponse
from ..service.chat_service import IChatService
from .base_handler import BaseHandler


class ChatHandler(BaseHandler):
    """
    채팅 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 결과 출력/포맷팅
    - 비즈니스 로직 없음
    """

    def __init__(self, chat_service: IChatService) -> None:
        """
        의존성 주입

        Args:
            chat_service: 채팅 서비스 (인터페이스에 의존 - DIP)
        """
        super().__init__(chat_service)
        self._chat_service = chat_service  # BaseHandler._service와 동일하지만 명시적으로 유지

    @log_handler_call
    @handle_errors(error_message="Chat request failed")
    @validate_input(
        required_params=["messages", "model"],
        param_types={"messages": list, "model": str},
        param_ranges={"temperature": (0, 2), "max_tokens": (1, None)},
    )
    async def handle_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        채팅 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            messages: 메시지 리스트
            model: 모델 이름
            temperature: 온도
            max_tokens: 최대 토큰 수
            top_p: Top-p
            system: 시스템 프롬프트
            stream: 스트리밍 여부
            **kwargs: 추가 파라미터

        Returns:
            ChatResponse: 채팅 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = ChatRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system=system,
            stream=stream,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._call_service("chat", request)

    @log_handler_call
    @handle_errors(error_message="Stream chat failed")
    @validate_input(
        required_params=["messages", "model"],
        param_types={"messages": list, "model": str},
        param_ranges={"temperature": (0, 2), "max_tokens": (1, None)},
    )
    async def handle_stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        스트리밍 채팅 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            messages: 메시지 리스트
            model: 모델 이름
            temperature: 온도
            max_tokens: 최대 토큰 수
            top_p: Top-p
            system: 시스템 프롬프트
            **kwargs: 추가 파라미터

        Yields:
            str: 스트리밍 청크

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = ChatRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system=system,
            stream=True,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        # BaseHandler._call_service는 async generator를 직접 반환하지 않으므로 직접 호출
        async for chunk in self._chat_service.stream_chat(request):
            yield chunk
