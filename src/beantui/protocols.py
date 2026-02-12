"""
Protocols - TUI 엔진이 의존하는 추상 인터페이스

ChatBackend: LLM 스트리밍 채팅 추상화
  - 어떤 LLM 클라이언트든 이 프로토콜을 구현하면 beantui와 함께 사용 가능
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class ChatBackend(Protocol):
    """LLM 채팅 백엔드 프로토콜

    TUI 엔진은 이 프로토콜만 알며, 실제 LLM 클라이언트(OpenAI, Anthropic 등)는
    이 프로토콜의 구현체를 통해 주입됩니다.

    Example:
        >>> class MyBackend:
        ...     async def stream_chat(self, messages, system, model, temperature):
        ...         async for chunk in my_llm.stream(messages):
        ...             yield chunk
        ...
        ...     async def get_default_model(self) -> str:
        ...         return "my-model"
        ...
        ...     def sanitize_error(self, error: Exception) -> str:
        ...         return str(error)
    """

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        temperature: float,
    ) -> AsyncIterator[str]:
        """LLM 스트리밍 채팅

        Args:
            messages: 대화 메시지 목록 [{"role": "user", "content": "..."}]
            system: 시스템 프롬프트
            model: 모델명
            temperature: 생성 온도 (0.0-2.0)

        Yields:
            응답 텍스트 청크
        """
        ...  # pragma: no cover

    async def get_default_model(self) -> str:
        """기본 모델명 반환"""
        ...  # pragma: no cover

    def sanitize_error(self, error: Exception) -> str:
        """에러 메시지에서 민감 정보 마스킹

        Args:
            error: 원본 예외

        Returns:
            안전한 에러 메시지 문자열
        """
        ...  # pragma: no cover


class EchoBackend:
    """테스트/데모용 에코 백엔드 — 입력을 그대로 반환"""

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: str,
        model: str,
        temperature: float,
    ) -> AsyncIterator[str]:
        """마지막 사용자 메시지를 에코"""
        last_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_msg = m.get("content", "")
                break

        reply = f"**Echo** ({model}): {last_msg}"
        for word in reply.split():
            yield word + " "

    async def get_default_model(self) -> str:
        return "echo-v1"

    def sanitize_error(self, error: Exception) -> str:
        return str(error)
