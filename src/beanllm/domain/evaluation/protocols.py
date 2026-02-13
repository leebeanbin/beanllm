"""
Domain-layer protocols for evaluation metrics.

Metrics that need LLM calls depend on these protocols,
not on concrete Facade or Service implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Union, runtime_checkable


class LLMResponse(Protocol):
    """LLM 응답 프로토콜 — .content 속성만 필요"""

    @property
    def content(self) -> str: ...


@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    LLM 클라이언트 프로토콜

    평가 메트릭에서 LLM 호출이 필요할 때 사용합니다.
    Facade의 Client, IChatService 래퍼 등 어떤 구현체든
    이 프로토콜만 만족하면 됩니다.

    Example:
        >>> class MyLLMAdapter:
        ...     def chat(self, messages):
        ...         # 실제 LLM 호출
        ...         return response  # response.content 존재
        >>> metric = AnswerRelevanceMetric(client=MyLLMAdapter())
    """

    def chat(
        self,
        messages: Union[str, List[Dict[str, Any]]],
    ) -> LLMResponse: ...
