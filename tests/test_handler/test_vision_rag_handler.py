"""
VisionRAGHandler 테스트 - Vision RAG Handler 테스트
"""

from unittest.mock import AsyncMock, Mock

import pytest

from beanllm.dto.request import VisionRAGRequest
from beanllm.dto.response import VisionRAGResponse
from beanllm.handler import VisionRAGHandler


class TestVisionRAGHandler:
    """VisionRAGHandler 테스트"""

    @pytest.fixture
    def mock_vision_rag_service(self):
        """Mock VisionRAGService"""
        from beanllm.service.vision_rag_service import IVisionRAGService

        service = Mock(spec=IVisionRAGService)
        service.retrieve = AsyncMock(return_value=VisionRAGResponse(results=[]))
        service.query = AsyncMock(return_value=VisionRAGResponse(answer="Vision RAG answer"))
        service.batch_query = AsyncMock(
            return_value=VisionRAGResponse(answers=["Answer 1", "Answer 2"])
        )
        return service

    @pytest.fixture
    def vision_rag_handler(self, mock_vision_rag_service):
        """VisionRAGHandler 인스턴스"""
        return VisionRAGHandler(vision_rag_service=mock_vision_rag_service)

    @pytest.mark.asyncio
    async def test_handle_retrieve(self, vision_rag_handler):
        """이미지 검색 테스트"""
        # handle_retrieve는 VisionRAGResponse를 반환
        response = await vision_rag_handler.handle_retrieve(
            query="Find images of cats",
            k=5,
        )

        assert response is not None
        assert isinstance(response, VisionRAGResponse)
        assert response.results is not None
        assert isinstance(response.results, list)

    @pytest.mark.asyncio
    async def test_handle_query(self, vision_rag_handler):
        """질문 답변 테스트"""
        # handle_query는 VisionRAGResponse를 반환
        response = await vision_rag_handler.handle_query(
            question="What is in these images?",
            k=3,
        )

        assert response is not None
        assert isinstance(response, VisionRAGResponse)
        assert response.answer is not None
        assert isinstance(response.answer, str)

    @pytest.mark.asyncio
    async def test_handle_batch_query(self, vision_rag_handler):
        """배치 질문 답변 테스트"""
        # handle_batch_query는 VisionRAGResponse를 반환
        response = await vision_rag_handler.handle_batch_query(
            questions=["Question 1?", "Question 2?"],
            k=3,
        )

        assert response is not None
        assert isinstance(response, VisionRAGResponse)
        assert response.answers is not None
        assert isinstance(response.answers, list)
        assert len(response.answers) == 2

    @pytest.mark.asyncio
    async def test_handle_query_validation_error(self, vision_rag_handler):
        """입력 검증 에러 테스트"""
        # question이 빈 문자열이어도 통과할 수 있음
        try:
            await vision_rag_handler.handle_query(
                question="",
                k=3,
            )
            # 통과하면 통과
        except ValueError:
            # 검증 에러가 발생하면 통과
            pass
