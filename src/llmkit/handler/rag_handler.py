"""
RAGHandler - RAG 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, Union

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.rag_request import RAGRequest
from ..dto.response.rag_response import RAGResponse
from ..service.rag_service import IRAGService


class RAGHandler:
    """
    RAG 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, rag_service: IRAGService) -> None:
        """
        의존성 주입

        Args:
            rag_service: RAG 서비스 (인터페이스에 의존 - DIP)
        """
        self._rag_service = rag_service

    @log_handler_call
    @handle_errors(error_message="RAG query failed")
    @validate_input(
        required_params=["query"],
        param_types={"query": str, "k": int},
        param_ranges={"k": (1, None)},
    )
    async def handle_query(
        self,
        query: str,
        source: Optional[Union[str, Path, List[Any]]] = None,
        vector_store: Optional[Any] = None,
        k: int = 4,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        llm_model: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """
        RAG 질의 처리 (모든 검증 및 에러 처리 포함)

        Args:
            query: 질문
            source: 문서 소스
            vector_store: 벡터 스토어
            k: 검색 결과 수
            rerank: 재순위화 여부
            mmr: MMR 사용 여부
            hybrid: Hybrid search 사용 여부
            llm_model: LLM 모델
            **kwargs: 추가 파라미터

        Returns:
            RAGResponse: RAG 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # 추가 검증: source 또는 vector_store 중 하나는 필수
        if not source and not vector_store:
            raise ValueError("Either source or vector_store must be provided")

        # DTO 생성
        request = RAGRequest(
            query=query,
            source=source,
            vector_store=vector_store,
            k=k,
            rerank=rerank,
            mmr=mmr,
            hybrid=hybrid,
            llm_model=llm_model,
            prompt_template=prompt_template,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._rag_service.query(request)

    async def handle_retrieve(
        self,
        query: str,
        vector_store: Optional[Any] = None,
        k: int = 4,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """
        문서 검색만 수행 (모든 검증 및 에러 처리 포함)

        Args:
            query: 검색 쿼리
            vector_store: 벡터 스토어
            k: 검색 결과 수
            rerank: 재순위화 여부
            mmr: MMR 사용 여부
            hybrid: Hybrid search 사용 여부
            **kwargs: 추가 파라미터

        Returns:
            검색 결과 리스트
        """
        # DTO 생성
        request = RAGRequest(
            query=query,
            vector_store=vector_store,
            k=k,
            rerank=rerank,
            mmr=mmr,
            hybrid=hybrid,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._rag_service.retrieve(request)

    @log_handler_call
    @handle_errors(error_message="RAG stream query failed")
    @validate_input(
        required_params=["query"],
        param_types={"query": str, "k": int},
        param_ranges={"k": (1, None)},
    )
    async def handle_stream_query(
        self,
        query: str,
        source: Optional[Union[str, Path, List[Any]]] = None,
        vector_store: Optional[Any] = None,
        k: int = 4,
        rerank: bool = False,
        mmr: bool = False,
        hybrid: bool = False,
        llm_model: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        RAG 스트리밍 질의 처리 (모든 검증 및 에러 처리 포함)

        Args:
            query: 질문
            source: 문서 소스
            vector_store: 벡터 스토어
            k: 검색 결과 수
            rerank: 재순위화 여부
            mmr: MMR 사용 여부
            hybrid: Hybrid search 사용 여부
            llm_model: LLM 모델
            prompt_template: 프롬프트 템플릿
            **kwargs: 추가 파라미터

        Yields:
            str: 스트리밍 청크
        """
        # 추가 검증: source 또는 vector_store 중 하나는 필수
        if not source and not vector_store:
            raise ValueError("Either source or vector_store must be provided")

        # DTO 생성
        request = RAGRequest(
            query=query,
            source=source,
            vector_store=vector_store,
            k=k,
            rerank=rerank,
            mmr=mmr,
            hybrid=hybrid,
            llm_model=llm_model,
            prompt_template=prompt_template,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        async for chunk in self._rag_service.stream_query(request):
            yield chunk
