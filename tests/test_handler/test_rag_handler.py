"""
RAGHandler 테스트 - RAG 핸들러 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

try:
    from llmkit.handler.rag_handler import RAGHandler
    from llmkit.service.rag_service import IRAGService
    from llmkit.dto.response.rag_response import RAGResponse
    from llmkit.domain.vector_stores.base import VectorSearchResult
    from llmkit.domain.loaders import Document
except ImportError:
    from src.llmkit.handler.rag_handler import RAGHandler
    from src.llmkit.service.rag_service import IRAGService
    from src.llmkit.dto.response.rag_response import RAGResponse
    from src.llmkit.domain.vector_stores.base import VectorSearchResult
    from src.llmkit.domain.loaders import Document


class TestRAGHandler:
    """RAGHandler 테스트"""

    @pytest.fixture
    def mock_rag_service(self):
        """Mock RAGService"""
        service = Mock(spec=IRAGService)
        search_results = [
            VectorSearchResult(
                document=Document(content="Doc 1", metadata={}), score=0.9, metadata={}
            ),
            VectorSearchResult(
                document=Document(content="Doc 2", metadata={}), score=0.8, metadata={}
            ),
        ]
        service.query = AsyncMock(
            return_value=RAGResponse(
                answer="Answer based on context",
                sources=search_results,
                metadata={"model": "gpt-4o-mini", "k": 2},
            )
        )
        service.retrieve = AsyncMock(return_value=search_results)
        # stream_query는 async generator로 설정
        async def mock_stream_query(*args, **kwargs):
            chunks = ["Answer", " ", "based", " ", "on", " ", "context"]
            for chunk in chunks:
                yield chunk
        service.stream_query = mock_stream_query
        return service

    @pytest.fixture
    def rag_handler(self, mock_rag_service):
        """RAGHandler 인스턴스"""
        return RAGHandler(rag_service=mock_rag_service)

    @pytest.mark.asyncio
    async def test_handle_query_basic(self, rag_handler):
        """기본 RAG 질의 처리 테스트"""
        response = await rag_handler.handle_query(
            query="What is this about?",
            vector_store=Mock(),
            k=2,
            llm_model="gpt-4o-mini",
        )

        assert response is not None
        assert isinstance(response, RAGResponse)
        assert response.answer == "Answer based on context"
        assert len(response.sources) == 2

    @pytest.mark.asyncio
    async def test_handle_query_with_search_options(self, rag_handler):
        """검색 옵션 포함 질의 처리 테스트"""
        response = await rag_handler.handle_query(
            query="What is this about?",
            vector_store=Mock(),
            k=2,
            rerank=True,
            mmr=True,
            hybrid=False,
            llm_model="gpt-4o-mini",
        )

        assert response is not None
        assert response.answer == "Answer based on context"

    @pytest.mark.asyncio
    async def test_handle_retrieve(self, rag_handler):
        """문서 검색 처리 테스트"""
        results = await rag_handler.handle_retrieve(
            query="What is this about?",
            vector_store=Mock(),
            k=2,
        )

        assert len(results) == 2
        assert rag_handler._rag_service.retrieve.called

    @pytest.mark.asyncio
    async def test_handle_query_dto_conversion(self, rag_handler):
        """DTO 변환 테스트"""
        mock_vector_store = Mock()
        response = await rag_handler.handle_query(
            query="What is this about?",
            vector_store=mock_vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            prompt_template="Custom: {context} {question}",
        )

        assert response is not None
        # RAGRequest가 올바르게 생성되었는지 확인
        call_args = rag_handler._rag_service.query.call_args[0][0]
        assert call_args.query == "What is this about?"
        assert call_args.vector_store == mock_vector_store
        assert call_args.k == 2
        assert call_args.llm_model == "gpt-4o-mini"
        assert call_args.prompt_template == "Custom: {context} {question}"

    @pytest.mark.asyncio
    async def test_handle_query_with_all_search_options(self, rag_handler):
        """모든 검색 옵션 포함 테스트"""
        response = await rag_handler.handle_query(
            query="What is this about?",
            vector_store=Mock(),
            k=5,
            llm_model="gpt-4o-mini",
            mmr=True,
            hybrid=True,
            rerank=True,
        )

        assert response is not None
        call_args = rag_handler._rag_service.query.call_args[0][0]
        assert call_args.mmr is True
        assert call_args.hybrid is True
        assert call_args.rerank is True

    @pytest.mark.asyncio
    async def test_handle_query_validation_missing_query(self, rag_handler):
        """입력 검증 - query 누락 테스트"""
        with pytest.raises((ValueError, TypeError)):
            await rag_handler.handle_query(
                query=None,  # 필수 파라미터 누락
                vector_store=Mock(),
                k=2,
                llm_model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_handle_query_validation_missing_vector_store(self, rag_handler):
        """입력 검증 - vector_store 누락 테스트"""
        with pytest.raises((ValueError, TypeError)):
            await rag_handler.handle_query(
                query="What is this?",
                vector_store=None,  # 필수 파라미터 누락
                k=2,
                llm_model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_handle_query_error_handling(self, rag_handler):
        """에러 처리 테스트"""
        # Service에서 에러 발생 시뮬레이션
        rag_handler._rag_service.query = AsyncMock(side_effect=ValueError("Service error"))

        with pytest.raises(ValueError):
            await rag_handler.handle_query(
                query="What is this?",
                vector_store=Mock(),
                k=2,
                llm_model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_handle_stream_query(self, rag_handler):
        """스트리밍 RAG 질의 처리 테스트"""
        # async generator로 설정
        async def mock_stream(*args, **kwargs):
            chunks = ["Answer", " ", "based", " ", "on", " ", "context"]
            for chunk in chunks:
                yield chunk

        # Service의 stream_query를 실제 async generator로 교체
        rag_handler._rag_service.stream_query = mock_stream

        chunks = []
        async for chunk in rag_handler.handle_stream_query(
            query="What is this about?",
            vector_store=Mock(),
            k=2,
            llm_model="gpt-4o-mini",
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks) == "Answer based on context"

    @pytest.mark.asyncio
    async def test_handle_retrieve_dto_conversion(self, rag_handler):
        """검색 DTO 변환 테스트"""
        mock_vector_store = Mock()
        results = await rag_handler.handle_retrieve(
            query="What is this about?",
            vector_store=mock_vector_store,
            k=3,
            mmr=True,
        )

        assert len(results) == 2
        # RAGRequest가 올바르게 생성되었는지 확인
        call_args = rag_handler._rag_service.retrieve.call_args[0][0]
        assert call_args.query == "What is this about?"
        assert call_args.vector_store == mock_vector_store
        assert call_args.k == 3
        assert call_args.mmr is True

    @pytest.mark.asyncio
    async def test_handle_query_default_values(self, rag_handler):
        """기본값 테스트"""
        response = await rag_handler.handle_query(
            query="What is this?",
            vector_store=Mock(),
            k=2,
            llm_model="gpt-4o-mini",
        )

        assert response is not None
        call_args = rag_handler._rag_service.query.call_args[0][0]
        # 기본값 확인
        assert call_args.mmr is False or call_args.mmr is None
        assert call_args.hybrid is False or call_args.hybrid is None
        assert call_args.rerank is False or call_args.rerank is None

    @pytest.mark.asyncio
    async def test_handle_query_custom_prompt_template(self, rag_handler):
        """커스텀 프롬프트 템플릿 테스트"""
        custom_template = "Context: {context}\nQuestion: {question}\nAnswer:"
        response = await rag_handler.handle_query(
            query="What is this?",
            vector_store=Mock(),
            k=2,
            llm_model="gpt-4o-mini",
            prompt_template=custom_template,
        )

        assert response is not None
        call_args = rag_handler._rag_service.query.call_args[0][0]
        assert call_args.prompt_template == custom_template

