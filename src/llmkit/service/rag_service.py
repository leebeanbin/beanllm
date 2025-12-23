"""
IRAGService - RAG 서비스 인터페이스
SOLID 원칙:
- ISP: RAG 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List

from ..dto.request.rag_request import RAGRequest
from ..dto.response.rag_response import RAGResponse


class IRAGService(ABC):
    """
    RAG 서비스 인터페이스

    책임:
    - RAG 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: RAG 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def query(self, request: RAGRequest) -> RAGResponse:
        """
        RAG 질의 처리

        Args:
            request: RAG 요청 DTO

        Returns:
            RAGResponse: RAG 응답 DTO

        책임:
            - RAG 비즈니스 로직만 (검색, 임베딩, LLM 호출 등)
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def retrieve(self, request: RAGRequest) -> List[Any]:
        """
        문서 검색만 수행 (LLM 호출 없음)

        Args:
            request: RAG 요청 DTO

        Returns:
            검색 결과 리스트

        책임:
            - 검색 비즈니스 로직만
            - 검증, 에러 처리 없음
        """
        pass

    @abstractmethod
    async def stream_query(self, request: RAGRequest) -> AsyncIterator[str]:
        """
        RAG 스트리밍 질의 처리

        Args:
            request: RAG 요청 DTO

        Yields:
            str: 스트리밍 청크

        책임:
            - 스트리밍 RAG 비즈니스 로직만
            - 검증, 에러 처리 없음
        """
        pass
