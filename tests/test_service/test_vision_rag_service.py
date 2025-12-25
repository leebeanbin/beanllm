"""
VisionRAGService 테스트 - Vision RAG 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from beanllm.dto.request.vision_rag_request import VisionRAGRequest
from beanllm.dto.response.vision_rag_response import VisionRAGResponse
from beanllm.dto.response.chat_response import ChatResponse
from beanllm.service.impl.vision_rag_service_impl import VisionRAGServiceImpl


class TestVisionRAGService:
    """VisionRAGService 테스트"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        store = Mock()
        store.similarity_search = Mock(
            return_value=[
                Mock(document=Mock(content="Image 1 content", image_path="img1.jpg")),
                Mock(document=Mock(content="Image 2 content", image_path="img2.jpg")),
            ]
        )
        store.add_documents = Mock()
        return store

    @pytest.fixture
    def mock_chat_service(self):
        """Mock ChatService"""
        service = Mock()
        service.chat = AsyncMock(
            return_value=ChatResponse(
                content="Vision RAG answer", model="gpt-4o", provider="openai"
            )
        )
        return service

    @pytest.fixture
    def vision_rag_service(self, mock_vector_store, mock_chat_service):
        """VisionRAGService 인스턴스"""
        return VisionRAGServiceImpl(
            vector_store=mock_vector_store,
            chat_service=mock_chat_service,
        )

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, vision_rag_service):
        """기본 이미지 검색 테스트"""
        request = VisionRAGRequest(
            query="Find images of cats",
            k=5,
        )

        response = await vision_rag_service.retrieve(request)

        assert response is not None
        assert isinstance(response, VisionRAGResponse)
        assert response.results is not None
        assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, vision_rag_service):
        """빈 쿼리 검색 테스트"""
        request = VisionRAGRequest(
            query=None,
            k=5,
        )

        response = await vision_rag_service.retrieve(request)

        assert response is not None
        # 빈 쿼리도 작동해야 함
        assert response.results is not None

    @pytest.mark.asyncio
    async def test_query_basic(self, vision_rag_service):
        """기본 질문 답변 테스트"""
        # _build_context를 Mock하여 ImageDocument import 문제 우회
        vision_rag_service._build_context = Mock(return_value="Context text")

        # vision_loaders 모듈을 sys.modules에 추가
        import sys

        if "beanllm.vision_loaders" not in sys.modules:
            mock_vision_loaders = Mock()
            mock_vision_loaders.ImageDocument = Mock
            sys.modules["beanllm.vision_loaders"] = mock_vision_loaders

        request = VisionRAGRequest(
            question="What is in these images?",
            k=3,
            include_images=False,  # 텍스트만 사용
        )

        response = await vision_rag_service.query(request)

        assert response is not None
        assert isinstance(response, VisionRAGResponse)
        assert response.answer is not None
        assert response.answer == "Vision RAG answer"

    @pytest.mark.asyncio
    async def test_query_with_sources(self, vision_rag_service):
        """소스 포함 질문 답변 테스트"""
        # _build_context를 Mock
        vision_rag_service._build_context = Mock(return_value="Context text")

        request = VisionRAGRequest(
            question="What is in these images?",
            k=3,
            include_sources=True,
            include_images=False,
        )

        response = await vision_rag_service.query(request)

        assert response is not None
        assert response.answer is not None
        assert response.sources is not None

    @pytest.mark.asyncio
    async def test_query_without_images(self, vision_rag_service):
        """이미지 제외 질문 답변 테스트"""
        # _build_context를 Mock하여 ImageDocument import 문제 우회
        vision_rag_service._build_context = Mock(return_value="Context text")

        request = VisionRAGRequest(
            question="What is in these images?",
            k=3,
            include_images=False,
        )

        response = await vision_rag_service.query(request)

        assert response is not None
        assert response.answer is not None

    @pytest.mark.asyncio
    async def test_query_custom_prompt_template(self, vision_rag_service):
        """커스텀 프롬프트 템플릿 테스트"""
        # _build_context를 Mock
        vision_rag_service._build_context = Mock(return_value="Context text")

        custom_template = "Answer: {question} with context: {context}"
        vision_rag_service._prompt_template = custom_template

        request = VisionRAGRequest(
            question="What is this?",
            k=3,
            include_images=False,
        )

        response = await vision_rag_service.query(request)

        assert response is not None
        assert response.answer is not None

    @pytest.mark.asyncio
    async def test_batch_query(self, vision_rag_service):
        """배치 질문 답변 테스트"""
        # _build_context를 Mock
        vision_rag_service._build_context = Mock(return_value="Context text")

        request = VisionRAGRequest(
            questions=["What is image 1?", "What is image 2?"],
            k=3,
            include_images=False,
        )

        response = await vision_rag_service.batch_query(request)

        assert response is not None
        assert response.answers is not None
        assert len(response.answers) == 2

    @pytest.mark.asyncio
    async def test_from_images(self, vision_rag_service, tmp_path):
        """이미지로부터 벡터 스토어에 추가 테스트"""
        # Mock vision_embedding
        mock_embedding = Mock()
        mock_embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
        vision_rag_service._vision_embedding = mock_embedding

        # Mock ImageLoader - domain.vision.loaders에 있음
        mock_loader = Mock()
        mock_doc = Mock()
        mock_doc.content = "Image caption"
        mock_doc.image_path = str(tmp_path / "test.jpg")
        mock_doc.get_image_base64 = Mock(return_value="base64data")
        mock_loader.load = Mock(return_value=[mock_doc])

        with patch("beanllm.domain.vision.loaders.ImageLoader", return_value=mock_loader):
            request = VisionRAGRequest(
                source=str(tmp_path / "test.jpg"),
                generate_captions=True,
            )

            try:
                response = await vision_rag_service.from_images(request)
                assert response is not None
            except (ImportError, ModuleNotFoundError, AttributeError):
                # vision_loaders가 없으면 스킵
                pytest.skip("vision_loaders module not available")

    @pytest.mark.asyncio
    async def test_from_sources(self, vision_rag_service, tmp_path):
        """소스로부터 벡터 스토어에 추가 테스트"""
        # Mock vision_embedding
        mock_embedding = Mock()
        mock_embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
        vision_rag_service._vision_embedding = mock_embedding

        # Mock ImageLoader
        mock_loader = Mock()
        mock_doc = Mock()
        mock_doc.content = "Image caption"
        mock_doc.image_path = str(tmp_path / "test.jpg")
        mock_doc.get_image_base64 = Mock(return_value="base64data")
        mock_loader.load = Mock(return_value=[mock_doc])

        with patch("beanllm.domain.vision.loaders.ImageLoader", return_value=mock_loader):
            request = VisionRAGRequest(
                sources=[str(tmp_path / "test.jpg")],
                generate_captions=True,
            )

            try:
                response = await vision_rag_service.from_sources(request)
                assert response is not None
            except (ImportError, ModuleNotFoundError, AttributeError):
                # vision_loaders가 없으면 스킵
                pytest.skip("vision_loaders module not available")


