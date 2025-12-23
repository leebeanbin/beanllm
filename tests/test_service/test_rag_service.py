"""
RAGService 테스트 - RAG 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.rag_request import RAGRequest
from llmkit.dto.response.rag_response import RAGResponse
from llmkit.service.impl.rag_service_impl import RAGServiceImpl


class TestRAGService:
    """RAGService 테스트"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        store = Mock()
        store.similarity_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.8),
            ]
        )
        return store

    @pytest.fixture
    def mock_chat_service(self):
        """Mock ChatService"""
        service = Mock()
        service.chat = AsyncMock(
            return_value=Mock(content="Answer based on context", model="gpt-4o-mini")
        )
        return service

    @pytest.fixture
    def rag_service(self, mock_vector_store, mock_chat_service):
        """RAGService 인스턴스"""
        return RAGServiceImpl(
            vector_store=mock_vector_store,
            chat_service=mock_chat_service,
        )

    @pytest.mark.asyncio
    async def test_query_basic(self, rag_service):
        """기본 RAG 질의 테스트"""
        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
        )

        response = await rag_service.query(request)

        assert response is not None
        assert isinstance(response, RAGResponse)
        assert response.answer == "Answer based on context"
        assert len(response.sources) == 2

    @pytest.mark.asyncio
    async def test_retrieve(self, rag_service):
        """문서 검색 테스트"""
        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
        )

        results = await rag_service.retrieve(request)

        assert len(results) == 2
        assert rag_service._vector_store.similarity_search.called

    @pytest.mark.asyncio
    async def test_query_with_custom_prompt(self, rag_service):
        """커스텀 프롬프트 포함 질의 테스트"""
        custom_template = "Custom template: {context}\nQuestion: {question}"
        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            prompt_template=custom_template,
        )

        response = await rag_service.query(request)

        assert response is not None
        assert response.answer == "Answer based on context"

    @pytest.mark.asyncio
    async def test_stream_query(self, rag_service):
        """스트리밍 RAG 질의 테스트"""

        # Mock 스트리밍 응답
        async def mock_stream(*args, **kwargs):
            chunks = ["Answer", " ", "based", " ", "on", " ", "context"]
            for chunk in chunks:
                yield chunk

        rag_service._chat_service.stream_chat = mock_stream

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
        )

        chunks = []
        async for chunk in rag_service.stream_query(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks) == "Answer based on context"

    @pytest.mark.asyncio
    async def test_query_with_mmr_search(self, rag_service):
        """MMR 검색 포함 RAG 질의 테스트"""
        # MMR 검색 Mock
        rag_service._vector_store.mmr_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.8),
            ]
        )

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            mmr=True,
        )

        response = await rag_service.query(request)

        assert response is not None
        assert rag_service._vector_store.mmr_search.called
        assert not rag_service._vector_store.similarity_search.called

    @pytest.mark.asyncio
    async def test_query_with_hybrid_search(self, rag_service):
        """하이브리드 검색 포함 RAG 질의 테스트"""
        # 하이브리드 검색 Mock
        rag_service._vector_store.hybrid_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.8),
            ]
        )

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            hybrid=True,
        )

        response = await rag_service.query(request)

        assert response is not None
        assert rag_service._vector_store.hybrid_search.called
        assert not rag_service._vector_store.similarity_search.called
        assert not rag_service._vector_store.mmr_search.called

    @pytest.mark.asyncio
    async def test_query_with_rerank(self, rag_service):
        """재순위화 포함 RAG 질의 테스트"""
        # 재순위화 Mock
        rag_service._vector_store.rerank = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.95),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.85),
            ]
        )

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            rerank=True,
        )

        response = await rag_service.query(request)

        assert response is not None
        # rerank=True일 때 k*2로 검색 후 재순위화
        assert rag_service._vector_store.similarity_search.called
        # k=2이므로 similarity_search는 k=4로 호출되어야 함
        call_kwargs = rag_service._vector_store.similarity_search.call_args[1]
        assert call_kwargs.get("k") == 4
        # 재순위화가 호출되었는지 확인
        assert rag_service._vector_store.rerank.called

    @pytest.mark.asyncio
    async def test_retrieve_with_mmr(self, rag_service):
        """MMR 검색으로 문서 검색 테스트"""
        rag_service._vector_store.mmr_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.8),
            ]
        )

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            mmr=True,
        )

        results = await rag_service.retrieve(request)

        assert len(results) == 2
        assert rag_service._vector_store.mmr_search.called

    @pytest.mark.asyncio
    async def test_retrieve_with_hybrid(self, rag_service):
        """하이브리드 검색으로 문서 검색 테스트"""
        rag_service._vector_store.hybrid_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.8),
            ]
        )

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            hybrid=True,
        )

        results = await rag_service.retrieve(request)

        assert len(results) == 2
        assert rag_service._vector_store.hybrid_search.called

    @pytest.mark.asyncio
    async def test_retrieve_with_rerank(self, rag_service):
        """재순위화 포함 문서 검색 테스트"""
        rag_service._vector_store.rerank = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.95),
            ]
        )

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=1,
            rerank=True,
        )

        results = await rag_service.retrieve(request)

        # rerank=True일 때 k*2로 검색 후 재순위화
        assert rag_service._vector_store.similarity_search.called
        call_kwargs = rag_service._vector_store.similarity_search.call_args[1]
        assert call_kwargs.get("k") == 2  # k=1 * 2
        assert rag_service._vector_store.rerank.called
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_query_empty_search_results(self, rag_service):
        """빈 검색 결과 테스트"""
        rag_service._vector_store.similarity_search = Mock(return_value=[])

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
        )

        response = await rag_service.query(request)

        assert response is not None
        assert response.answer == "Answer based on context"  # LLM은 여전히 응답
        assert len(response.sources) == 0

    @pytest.mark.asyncio
    async def test_query_context_building(self, rag_service):
        """컨텍스트 빌딩 테스트"""
        # 여러 문서 Mock
        search_results = [
            Mock(document=Mock(content="First document", metadata={}), score=0.9),
            Mock(document=Mock(content="Second document", metadata={}), score=0.8),
            Mock(document=Mock(content="Third document", metadata={}), score=0.7),
        ]
        rag_service._vector_store.similarity_search = Mock(return_value=search_results)

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=3,
            llm_model="gpt-4o-mini",
        )

        response = await rag_service.query(request)

        assert response is not None
        # ChatService가 호출되었는지 확인 (컨텍스트가 포함된 프롬프트로)
        rag_service._chat_service.chat.assert_called_once()
        call_args = rag_service._chat_service.chat.call_args[0][0]
        prompt = call_args.messages[0]["content"]
        # 컨텍스트가 포함되어 있는지 확인
        assert "First document" in prompt
        assert "Second document" in prompt
        assert "Third document" in prompt

    @pytest.mark.asyncio
    async def test_query_prompt_template(self, rag_service):
        """커스텀 프롬프트 템플릿 테스트"""
        custom_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            prompt_template=custom_template,
        )

        response = await rag_service.query(request)

        assert response is not None
        # ChatService가 커스텀 템플릿으로 호출되었는지 확인
        rag_service._chat_service.chat.assert_called_once()
        call_args = rag_service._chat_service.chat.call_args[0][0]
        prompt = call_args.messages[0]["content"]
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    @pytest.mark.asyncio
    async def test_query_default_prompt_template(self, rag_service):
        """기본 프롬프트 템플릿 테스트"""
        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            prompt_template=None,  # 기본 템플릿 사용
        )

        response = await rag_service.query(request)

        assert response is not None
        rag_service._chat_service.chat.assert_called_once()
        call_args = rag_service._chat_service.chat.call_args[0][0]
        prompt = call_args.messages[0]["content"]
        # 기본 템플릿 형식 확인
        assert "Based on the following context" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt

    @pytest.mark.asyncio
    async def test_query_response_metadata(self, rag_service):
        """응답 메타데이터 테스트"""
        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
        )

        response = await rag_service.query(request)

        assert response is not None
        assert response.metadata is not None
        assert response.metadata.get("model") == "gpt-4o-mini"
        assert response.metadata.get("k") == 2

    @pytest.mark.asyncio
    async def test_stream_query_with_mmr(self, rag_service):
        """MMR 검색 포함 스트리밍 RAG 질의 테스트"""
        rag_service._vector_store.mmr_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
                Mock(document=Mock(content="Doc 2", metadata={}), score=0.8),
            ]
        )

        async def mock_stream(*args, **kwargs):
            chunks = ["Answer", " ", "based", " ", "on", " ", "context"]
            for chunk in chunks:
                yield chunk

        rag_service._chat_service.stream_chat = mock_stream

        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            mmr=True,
        )

        chunks = []
        async for chunk in rag_service.stream_query(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert rag_service._vector_store.mmr_search.called

    @pytest.mark.asyncio
    async def test_retrieve_different_k_values(self, rag_service):
        """다양한 k 값으로 검색 테스트"""
        for k in [1, 3, 5, 10]:
            rag_service._vector_store.similarity_search = Mock(
                return_value=[
                    Mock(document=Mock(content=f"Doc {i}", metadata={}), score=0.9 - i * 0.1)
                    for i in range(k)
                ]
            )

            request = RAGRequest(
                query="What is this about?",
                vector_store=rag_service._vector_store,
                k=k,
            )

            results = await rag_service.retrieve(request)

            assert len(results) == k
            call_kwargs = rag_service._vector_store.similarity_search.call_args[1]
            assert call_kwargs.get("k") == k

    @pytest.mark.asyncio
    async def test_query_priority_hybrid_over_mmr(self, rag_service):
        """하이브리드가 MMR보다 우선순위가 높은지 테스트"""
        rag_service._vector_store.hybrid_search = Mock(
            return_value=[
                Mock(document=Mock(content="Doc 1", metadata={}), score=0.9),
            ]
        )

        # hybrid=True이고 mmr=True일 때 hybrid가 우선
        request = RAGRequest(
            query="What is this about?",
            vector_store=rag_service._vector_store,
            k=2,
            llm_model="gpt-4o-mini",
            hybrid=True,
            mmr=True,  # 둘 다 True
        )

        response = await rag_service.query(request)

        assert response is not None
        assert rag_service._vector_store.hybrid_search.called
        assert not rag_service._vector_store.mmr_search.called

