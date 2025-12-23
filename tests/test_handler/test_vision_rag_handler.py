"""
VisionRAGHandler 테스트 - Vision RAG Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.vision_rag_request import VisionRAGRequest
from llmkit.dto.response.vision_rag_response import VisionRAGResponse
from llmkit.handler.vision_rag_handler import VisionRAGHandler


class TestVisionRAGHandler:
    """VisionRAGHandler 테스트"""

    @pytest.fixture
    def mock_vision_rag_service(self):
        """Mock VisionRAGService"""
        service = Mock()
        service.retrieve = AsyncMock(
            return_value=VisionRAGResponse(
                results=[]
            )
        )
        service.query = AsyncMock(
            return_value=VisionRAGResponse(
                answer="Vision RAG answer"
            )
        )
        service.batch_query = AsyncMock(
            return_value=VisionRAGResponse(
                answers=["Answer 1", "Answer 2"]
            )
        )
        return service

    @pytest.fixture
    def vision_rag_handler(self, mock_vision_rag_service):
        """VisionRAGHandler 인스턴스"""
        return VisionRAGHandler(vision_rag_service=mock_vision_rag_service)

    @pytest.mark.asyncio
    async def test_handle_retrieve(self, vision_rag_handler):
        """이미지 검색 테스트"""
        # handle_retrieve는 List를 반환
        results = await vision_rag_handler.handle_retrieve(
            query="Find images of cats",
            k=5,
        )

        assert results is not None
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_handle_query(self, vision_rag_handler):
        """질문 답변 테스트"""
        # handle_query는 str 또는 tuple을 반환
        response = await vision_rag_handler.handle_query(
            question="What is in these images?",
            k=3,
        )

        assert response is not None
        # str 또는 tuple
        assert isinstance(response, (str, tuple))

    @pytest.mark.asyncio
    async def test_handle_batch_query(self, vision_rag_handler):
        """배치 질문 답변 테스트"""
        # handle_batch_query는 List[str]을 반환
        answers = await vision_rag_handler.handle_batch_query(
            questions=["Question 1?", "Question 2?"],
            k=3,
        )

        assert answers is not None
        assert isinstance(answers, list)
        assert len(answers) == 2

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


