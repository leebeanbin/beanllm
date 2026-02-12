"""
BeanllmBackend - beanllm ChatBackend Protocol 구현

beantui.protocols.ChatBackend을 구현하여
beanllm의 Client를 통해 LLM 스트리밍을 제공합니다.
"""

from __future__ import annotations

from typing import AsyncIterator


class BeanllmBackend:
    """beanllm Client 기반 ChatBackend 구현

    Example:
        >>> from beantui import TUIEngine, TUIConfig
        >>> from beanllm.ui.interactive.backend import BeanllmBackend
        >>>
        >>> engine = TUIEngine(config=TUIConfig.auto_discover(), backend=BeanllmBackend())
        >>> engine.run()
    """

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        temperature: float,
    ) -> AsyncIterator[str]:
        """beanllm Client를 사용한 스트리밍 채팅

        Args:
            messages: 대화 메시지 목록
            system: 시스템 프롬프트
            model: 모델명
            temperature: 생성 온도

        Yields:
            응답 텍스트 청크
        """
        from beanllm import Client

        client = Client(model=model)
        stream = client.stream_chat(
            messages=messages,
            system=system,
            temperature=temperature,
        )
        async for chunk in stream:
            yield chunk

    async def get_default_model(self) -> str:
        """beanllm 모델 레지스트리에서 기본 모델 탐색

        Returns:
            활성 프로바이더 중 첫 번째 사용 가능 모델명
        """
        try:
            from beanllm.infrastructure.registry import get_model_registry

            registry = get_model_registry()
            models = registry.get_available_models()
            active_providers = {p.name for p in registry.get_active_providers()}
            for m in models:
                if m.provider in active_providers:
                    return m.model_name
        except Exception:
            pass
        return "gpt-4o-mini"

    def sanitize_error(self, error: Exception) -> str:
        """API 키 등 민감 정보 마스킹

        Args:
            error: 원본 예외

        Returns:
            안전한 에러 메시지 문자열
        """
        try:
            from beanllm.utils.integration.security import sanitize_error_message

            return sanitize_error_message(error)
        except ImportError:
            return str(error)
