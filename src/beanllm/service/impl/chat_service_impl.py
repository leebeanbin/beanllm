"""
ChatServiceImpl - 채팅 서비스 구현체
SOLID 원칙:
- SRP: 채팅 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
- OCP: 확장 가능 (새 Provider 추가 시 수정 불필요)
- DRY: BaseService로 공통 로직 재사용
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Optional

from ...decorators.logger import log_service_call
from ...dto.request.chat_request import ChatRequest
from ...dto.response.chat_response import ChatResponse
from ...infrastructure.adapter import ParameterAdapter
from ..chat_service import IChatService
from .base_service import BaseService

if TYPE_CHECKING:
    from ...service.types import ProviderFactoryProtocol


class ChatServiceImpl(BaseService, IChatService):
    """
    채팅 서비스 구현체

    책임:
    - 채팅 비즈니스 로직만 (파라미터 변환, Provider 호출)
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: 채팅 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    - OCP: Provider 변경 시 수정 불필요
    """

    def __init__(
        self,
        provider_factory: "ProviderFactoryProtocol",
        parameter_adapter: Optional[ParameterAdapter] = None,
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            provider_factory: Provider 생성 팩토리
            parameter_adapter: 파라미터 변환 어댑터 (선택적)
        """
        super().__init__(provider_factory, parameter_adapter)

    @log_service_call
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        채팅 요청 처리 (비즈니스 로직만)

        Args:
            request: 채팅 요청 DTO

        Returns:
            ChatResponse: 채팅 응답 DTO

        책임:
            - 파라미터 변환 (비즈니스 로직)
            - Provider 호출 (비즈니스 로직)
            - 응답 변환 (비즈니스 로직)
            - if-else/try-catch 없음
        """
        # 1. Provider 생성 (공통 로직 재사용)
        provider = self._create_provider(request.model, request.extra_params.get("provider"))

        # 2. 파라미터 변환 (공통 로직 재사용)
        raw_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": request.stream,
            **request.extra_params,
        }
        params = self._adapt_parameters(provider.name, request.model, raw_params)

        # 3. Provider 호출 (비즈니스 로직)
        provider_response = await provider.chat(
            messages=request.messages,
            model=request.model,
            system=request.system,
            **params,
        )

        # 4. 응답 변환 (비즈니스 로직)
        return ChatResponse.from_provider_response(provider_response, request.model, provider.name)

    @log_service_call
    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        스트리밍 채팅 요청 처리 (비즈니스 로직만)

        Args:
            request: 채팅 요청 DTO

        Yields:
            str: 스트리밍 청크

        책임:
            - 스트리밍 비즈니스 로직만
            - if-else/try-catch 없음
        """
        # 1. Provider 생성 (공통 로직 재사용)
        provider = self._create_provider(request.model, request.extra_params.get("provider"))

        # 2. 파라미터 변환 (공통 로직 재사용)
        raw_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": True,
            **request.extra_params,
        }
        params = self._adapt_parameters(provider.name, request.model, raw_params)

        # 3. 스트리밍 호출
        async for chunk in provider.stream_chat(
            messages=request.messages,
            model=request.model,
            system=request.system,
            **params,
        ):
            yield chunk
