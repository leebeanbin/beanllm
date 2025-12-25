"""
IVisionRAGService - Vision RAG 서비스 인터페이스
SOLID 원칙:
- ISP: Vision RAG 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto.request.vision_rag_request import VisionRAGRequest
from ..dto.response.vision_rag_response import VisionRAGResponse


class IVisionRAGService(ABC):
    """
    Vision RAG 서비스 인터페이스

    책임:
    - Vision RAG 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: Vision RAG 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        이미지 검색

        Args:
            request: Vision RAG 요청 DTO

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (results 필드에 검색 결과 포함)

        책임:
            - 이미지 검색 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def query(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        질문에 답변 (이미지 포함)

        Args:
            request: Vision RAG 요청 DTO

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (answer, sources 필드 포함)

        책임:
            - Vision RAG 비즈니스 로직만 (검색, 컨텍스트 생성, LLM 호출 등)
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def batch_query(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        여러 질문에 대해 배치 답변

        Args:
            request: Vision RAG 요청 DTO (questions 필드 사용)

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO (answers 필드에 답변 리스트 포함)

        책임:
            - 배치 Vision RAG 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass
