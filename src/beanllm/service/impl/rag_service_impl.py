"""
RAGServiceImpl - RAG 서비스 구현체
SOLID 원칙:
- SRP: RAG 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
- OCP: Strategy 패턴으로 검색 방법 확장 가능
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, List, Optional

from ...dto.request.rag_request import RAGRequest
from ...dto.response.rag_response import RAGResponse
from ..rag_service import IRAGService
from .search_strategy import SearchStrategyFactory

if TYPE_CHECKING:
    from ...service.chat_service import IChatService
    from ...service.types import (
        DocumentLoaderProtocol,
        EmbeddingServiceProtocol,
        TextSplitterProtocol,
        VectorStoreProtocol,
    )


class RAGServiceImpl(IRAGService):
    """
    RAG 서비스 구현체

    책임:
    - RAG 비즈니스 로직만 (검색, 임베딩, LLM 호출)
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: RAG 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(
        self,
        vector_store: "VectorStoreProtocol",
        chat_service: "IChatService",
        embedding_service: Optional["EmbeddingServiceProtocol"] = None,
        document_loader: Optional["DocumentLoaderProtocol"] = None,
        text_splitter: Optional["TextSplitterProtocol"] = None,
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            vector_store: 벡터 스토어
            chat_service: 채팅 서비스
            embedding_service: 임베딩 서비스 (선택적)
            document_loader: 문서 로더 (선택적)
            text_splitter: 텍스트 분할기 (선택적)
        """
        self._vector_store = vector_store
        self._chat_service = chat_service
        self._embedding_service = embedding_service
        self._document_loader = document_loader
        self._text_splitter = text_splitter

    async def query(self, request: RAGRequest) -> RAGResponse:
        """
        RAG 질의 처리 (비즈니스 로직만)

        Args:
            request: RAG 요청 DTO

        Returns:
            RAGResponse: RAG 응답 DTO

        책임:
            - 검색 비즈니스 로직
            - LLM 호출 비즈니스 로직
            - 응답 생성 비즈니스 로직
            - if-else/try-catch 없음
        """
        # 1. 문서 검색 (비즈니스 로직)
        search_results = await self.retrieve(request)

        # 2. 컨텍스트 생성 (비즈니스 로직)
        context = self._build_context(search_results)

        # 3. 프롬프트 생성 (비즈니스 로직)
        prompt = self._build_prompt(request.query, context, request.prompt_template)

        # 4. LLM 호출 (비즈니스 로직)
        from ...dto.request.chat_request import ChatRequest

        chat_request = ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            model=request.llm_model,
        )
        chat_response = await self._chat_service.chat(chat_request)

        # 5. 응답 생성 (비즈니스 로직)
        return RAGResponse(
            answer=chat_response.content,
            sources=search_results,
            metadata={"model": request.llm_model, "k": request.k},
        )

    async def retrieve(self, request: RAGRequest) -> List[Any]:
        """
        문서 검색만 수행 (2단계 검색 지원)

        Args:
            request: RAG 요청 DTO

        Returns:
            검색 결과 리스트

        책임:
            - 검색 비즈니스 로직만
            - Strategy 패턴으로 if-else 제거
            - 2단계 검색: Broad search -> Rerank -> Dynamic selection
        """
        # 1단계: 넓은 검색 (broad_k)
        extra_params = getattr(request, "extra_params", {}) or {}
        broad_k = extra_params.get("broad_k", request.k * 2 if request.rerank else request.k)

        # 검색 전략 선택 (Strategy 패턴으로 if-else 제거)
        search_type = self._determine_search_type(request)
        strategy = SearchStrategyFactory.create(search_type)

        # 검색 수행 (비즈니스 로직)
        results = strategy.search(self._vector_store, request.query, broad_k)

        # 2단계: 재순위화 (선택적)
        if request.rerank:
            results = self._vector_store.rerank(request.query, results, top_k=request.k)

        # 3단계: 동적 패시지 선택 (토큰 제한 고려)
        max_tokens = extra_params.get("max_context_tokens", 4000)
        results = self._select_passages_dynamically(results, max_tokens)

        return results

    def _determine_search_type(self, request: RAGRequest) -> str:
        """
        검색 타입 결정 (비즈니스 로직)

        책임:
            - 검색 방법 결정만
            - if-else를 명확한 로직으로
        """
        if request.hybrid:
            return "hybrid"
        elif request.mmr:
            return "mmr"
        else:
            return "similarity"

    def _build_context(self, results: List[Any]) -> str:
        """검색 결과에서 컨텍스트 생성 (비즈니스 로직)"""
        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.document.content if hasattr(result, "document") else str(result)
            context_parts.append(f"[{i}] {content}")
        return "\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str, template: str = None) -> str:
        """프롬프트 생성 (비즈니스 로직)"""
        if template is None:
            template = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        return template.format(context=context, question=query)

    def _select_passages_dynamically(
        self,
        results: List[Any],
        max_tokens: int = 4000,
    ) -> List[Any]:
        """
        동적 패시지 선택: 토큰 제한 고려

        Args:
            results: 검색 결과 리스트
            max_tokens: 최대 토큰 수

        Returns:
            선택된 결과 리스트
        """
        selected = []
        current_tokens = 0

        for result in results:
            content = result.document.content if hasattr(result, "document") else str(result)
            passage_tokens = int(len(content.split()) * 1.3)  # 간단한 토큰 추정

            if current_tokens + passage_tokens > max_tokens:
                break

            selected.append(result)
            current_tokens += passage_tokens

        return selected

    async def stream_query(self, request: RAGRequest) -> AsyncIterator[str]:
        """
        RAG 스트리밍 질의 처리 (기존 rag_chain.py의 stream_query 정확히 마이그레이션)

        Args:
            request: RAG 요청 DTO

        Yields:
            str: 스트리밍 청크

        책임:
            - 스트리밍 RAG 비즈니스 로직만
            - if-else/try-catch 없음
        """
        # 1. 문서 검색 (기존과 동일)
        search_results = await self.retrieve(request)

        # 2. 컨텍스트 생성 (기존과 동일)
        context = self._build_context(search_results)

        # 3. 프롬프트 생성 (기존과 동일)
        prompt = self._build_prompt(request.query, context, request.prompt_template)

        # 4. 스트리밍 LLM 호출 (기존: llm.stream(prompt))
        from ...dto.request.chat_request import ChatRequest

        chat_request = ChatRequest(
            messages=[{"role": "user", "content": prompt}],
            model=request.llm_model,
        )

        # 스트리밍 호출 (기존: for chunk in llm.stream(prompt): yield chunk.content)
        async for chunk in self._chat_service.stream_chat(chat_request):
            yield chunk
