"""
VisionRAGHandler - Vision RAG 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from typing import Any, List

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.vision_rag_request import VisionRAGRequest
from ..dto.response.vision_rag_response import VisionRAGResponse
from ..service.vision_rag_service import IVisionRAGService


class VisionRAGHandler:
    """
    Vision RAG 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, vision_rag_service: IVisionRAGService) -> None:
        """
        의존성 주입

        Args:
            vision_rag_service: Vision RAG 서비스 (인터페이스에 의존 - DIP)
        """
        self._vision_rag_service = vision_rag_service

    @log_handler_call
    @handle_errors(error_message="Vision RAG retrieve failed")
    @validate_input(
        required_params=["query"],
        param_types={"query": str, "k": int},
        param_ranges={"k": (1, None)},
    )
    async def handle_retrieve(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> VisionRAGResponse:
        """
        이미지 검색 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            query: 검색 쿼리 (텍스트)
            k: 반환할 결과 수
            **kwargs: 추가 파라미터

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = VisionRAGRequest(query=query, k=k, extra_params=kwargs)

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._vision_rag_service.retrieve(request)

    @log_handler_call
    @handle_errors(error_message="Vision RAG query failed")
    @validate_input(
        required_params=["question"],
        param_types={"question": str, "k": int, "include_sources": bool, "include_images": bool},
        param_ranges={"k": (1, None)},
    )
    async def handle_query(
        self,
        question: str,
        k: int = 4,
        include_sources: bool = False,
        include_images: bool = True,
        llm_model: str = "gpt-4o",
        **kwargs: Any,
    ) -> VisionRAGResponse:
        """
        질문에 답변 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            question: 질문
            k: 검색할 문서 수
            include_sources: 출처 포함 여부
            include_images: 이미지 포함 여부
            llm_model: LLM 모델
            **kwargs: 추가 파라미터

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = VisionRAGRequest(
            question=question,
            k=k,
            include_sources=include_sources,
            include_images=include_images,
            llm_model=llm_model,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._vision_rag_service.query(request)

    @log_handler_call
    @handle_errors(error_message="Vision RAG batch query failed")
    @validate_input(
        required_params=["questions"],
        param_types={"questions": list, "k": int},
        param_ranges={"k": (1, None)},
    )
    async def handle_batch_query(
        self,
        questions: List[str],
        k: int = 4,
        include_images: bool = True,
        llm_model: str = "gpt-4o",
        **kwargs: Any,
    ) -> VisionRAGResponse:
        """
        여러 질문에 대해 배치 답변 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            questions: 질문 리스트
            k: 검색할 문서 수
            include_images: 이미지 포함 여부
            llm_model: LLM 모델
            **kwargs: 추가 파라미터

        Returns:
            VisionRAGResponse: Vision RAG 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = VisionRAGRequest(
            questions=questions,
            k=k,
            include_images=include_images,
            llm_model=llm_model,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._vision_rag_service.batch_query(request)
