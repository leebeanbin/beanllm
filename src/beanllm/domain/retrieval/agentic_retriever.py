"""
AgenticRetriever
LLM이 스스로 검색 필요성을 판단하는 Agentic Retrieval 패턴 (2026 표준).

기존 RAG: 항상 검색 → 응답
Agentic:  [LLM 판단] → 필요 시에만 검색 → 응답

이 패턴은 불필요한 벡터 검색을 줄이고, 이미 충분한 컨텍스트가 있는 경우
지연 없이 응답할 수 있도록 한다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class DocumentProtocol(Protocol):
    """Vector store 문서의 최소 인터페이스."""

    page_content: str
    metadata: Dict[str, Any]


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Vector store의 최소 인터페이스."""

    def similarity_search(self, query: str, k: int = 4) -> List[Any]: ...


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """LLM provider의 최소 인터페이스."""

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any: ...


_DECISION_PROMPT = """\
You are deciding whether additional document retrieval is needed to answer a query.

Context already available:
{context}

Query: {query}

Respond with a JSON object containing exactly two keys:
- "needs_retrieval": true or false
- "reason": one-sentence explanation

JSON response:"""


class AgenticRetriever:
    """
    LLM이 검색 필요성을 판단하고 선택적으로 검색을 수행하는 Retriever.

    Example:
        ```python
        retriever = AgenticRetriever(
            vector_store=my_store,
            llm_provider=my_provider,
            model="gpt-4o-mini",
        )
        docs = await retriever.maybe_retrieve(
            query="What is the capital of France?",
            context="You are a geography expert.",
        )
        # docs is None when LLM decides no retrieval needed
        ```
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        llm_provider: LLMProviderProtocol,
        model: str = "gpt-4o-mini",
        k: int = 4,
        decision_max_tokens: int = 64,
    ) -> None:
        """
        Args:
            vector_store: 검색 대상 벡터 스토어
            llm_provider: 검색 필요성 판단에 사용할 LLM
            model: 판단에 사용할 모델명 (빠른 모델 권장)
            k: 검색 결과 개수
            decision_max_tokens: 판단 응답 최대 토큰
        """
        self._vector_store = vector_store
        self._llm = llm_provider
        self._model = model
        self._k = k
        self._decision_max_tokens = decision_max_tokens

    async def maybe_retrieve(
        self,
        query: str,
        context: str = "",
    ) -> Optional[List[Any]]:
        """
        LLM이 검색 필요성을 판단한 후 선택적으로 문서를 검색합니다.

        Args:
            query: 사용자 쿼리
            context: 이미 보유한 컨텍스트 (system prompt, conversation history 등)

        Returns:
            검색이 필요한 경우 Document 리스트, 불필요한 경우 None
        """
        needs = await self._decide(query, context)
        if not needs:
            return None
        return self._vector_store.similarity_search(query, k=self._k)

    async def _decide(self, query: str, context: str) -> bool:
        """LLM에게 검색 필요성 판단을 요청합니다."""
        prompt = _DECISION_PROMPT.format(
            context=context[:800] if context else "(none)",
            query=query,
        )
        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                max_tokens=self._decision_max_tokens,
                temperature=0.0,
            )
            text = response.content if hasattr(response, "content") else str(response)
            # Extract JSON — be lenient about surrounding text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(text[start:end])
                return bool(data.get("needs_retrieval", True))
        except Exception:
            pass
        # On any failure, default to retrieving (safe fallback)
        return True

    def retrieve(self, query: str) -> List[Any]:
        """동기 검색 (판단 없이 직접 검색)."""
        return self._vector_store.similarity_search(query, k=self._k)

    def __repr__(self) -> str:
        return (
            f"AgenticRetriever(model={self._model!r}, k={self._k}, "
            f"store={type(self._vector_store).__name__})"
        )
