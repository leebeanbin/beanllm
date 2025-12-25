"""
RAG Debugger 테스트 - RAG 디버깅 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, patch

try:
    from beanllm.utils.rag_debug import (
        RAGDebugger,
        EmbeddingInfo,
        SimilarityInfo,
        inspect_embedding,
        compare_texts,
        validate_pipeline,
    )

    RAG_DEBUG_AVAILABLE = True
except ImportError:
    RAG_DEBUG_AVAILABLE = False


@pytest.mark.skipif(not RAG_DEBUG_AVAILABLE, reason="RAG Debugger not available")
class TestRAGDebugger:
    """RAGDebugger 테스트"""

    @pytest.fixture
    def rag_debugger(self):
        """RAGDebugger 인스턴스"""
        return RAGDebugger()

    def test_rag_debugger_inspect_embedding(self, rag_debugger):
        """임베딩 검사 테스트"""
        text = "Hello world"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        info = rag_debugger.inspect_embedding(text, embedding)

        assert isinstance(info, EmbeddingInfo)
        assert info.dimension == len(embedding)

    def test_rag_debugger_compare_texts(self, rag_debugger):
        """텍스트 비교 테스트"""
        text1 = "Hello world"
        text2 = "Hello there"

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        similarity = rag_debugger.compare_texts(text1, text2, mock_embedding_function)

        assert isinstance(similarity, SimilarityInfo)
        assert 0.0 <= similarity.cosine_similarity <= 1.0

    def test_rag_debugger_validate_rag_pipeline(self, rag_debugger):
        """파이프라인 검증 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1", metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_store.similarity_search = Mock(return_value=[])

        result = rag_debugger.validate_rag_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)

    def test_rag_debugger_compare_embeddings(self, rag_debugger):
        """임베딩 비교 테스트"""
        embeddings = [
            ("text1", [0.1, 0.2, 0.3]),
            ("text2", [0.4, 0.5, 0.6]),
            ("text3", [0.7, 0.8, 0.9]),
        ]

        rag_debugger.compare_embeddings(embeddings)

        # 출력만 확인 (반환값 없음)
        assert True

    def test_rag_debugger_inspect_chunks(self, rag_debugger):
        """청크 검사 테스트"""
        from beanllm.domain.loaders.types import Document

        chunks = [
            Document(content="chunk1 " * 10, metadata={}),
            Document(content="chunk2 " * 20, metadata={}),
            Document(content="chunk3 " * 15, metadata={}),
        ]

        stats = rag_debugger.inspect_chunks(chunks, show_samples=2)

        assert isinstance(stats, dict)
        assert "total_chunks" in stats
        assert stats["total_chunks"] == 3
        assert "avg_length" in stats

    def test_rag_debugger_inspect_chunks_empty(self, rag_debugger):
        """빈 청크 리스트 테스트"""
        stats = rag_debugger.inspect_chunks([])
        assert stats == {}

    def test_rag_debugger_inspect_vector_store(self, rag_debugger):
        """Vector Store 검사 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        mock_store = Mock()
        # VectorSearchResult를 직접 생성하지 않고 Mock 사용
        mock_result = Mock()
        mock_result.document = Document(content="Test content", metadata={})
        mock_result.score = 0.9
        mock_store.similarity_search = Mock(return_value=[mock_result])

        results = rag_debugger.inspect_vector_store(mock_store, ["test query"], k=3)

        assert isinstance(results, dict)
        assert "test query" in results
        assert len(results["test query"]) == 1

    def test_rag_debugger_validate_rag_pipeline_full(self, rag_debugger):
        """전체 RAG 파이프라인 검증 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1 " * 20, metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_result = Mock()
        mock_result.document = Document(content="Test", metadata={})
        mock_result.score = 0.8
        mock_store.similarity_search = Mock(return_value=[mock_result])

        result = rag_debugger.validate_rag_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "chunks" in result
        assert "embedding_dim" in result
        assert "search_results" in result
        assert "issues" in result


@pytest.mark.skipif(not RAG_DEBUG_AVAILABLE, reason="RAG Debugger not available")
class TestRAGDebuggerFunctions:
    """RAG Debugger 편의 함수 테스트"""

    def test_inspect_embedding_function(self):
        """inspect_embedding 편의 함수 테스트"""

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        info = inspect_embedding("Hello world", mock_embedding_function)

        assert isinstance(info, EmbeddingInfo)

    def test_compare_texts_function(self):
        """compare_texts 편의 함수 테스트"""

        # embedding_function이 필요함
        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        similarity = compare_texts("Hello", "Hi", mock_embedding_function)

        assert isinstance(similarity, SimilarityInfo)
        assert 0.0 <= similarity.cosine_similarity <= 1.0

    def test_validate_pipeline_function(self):
        """validate_pipeline 편의 함수 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1", metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_store.similarity_search = Mock(return_value=[])

        result = validate_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "chunks" in result

        mock_result = Mock()
        mock_result.document = Document(content="Test content", metadata={})
        mock_result.score = 0.9
        mock_store.similarity_search = Mock(return_value=[mock_result])

        results = rag_debugger.inspect_vector_store(mock_store, ["test query"], k=3)

        assert isinstance(results, dict)
        assert "test query" in results
        assert len(results["test query"]) == 1

    def test_rag_debugger_validate_rag_pipeline_full(self, rag_debugger):
        """전체 RAG 파이프라인 검증 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1 " * 20, metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_result = Mock()
        mock_result.document = Document(content="Test", metadata={})
        mock_result.score = 0.8
        mock_store.similarity_search = Mock(return_value=[mock_result])

        result = rag_debugger.validate_rag_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "chunks" in result
        assert "embedding_dim" in result
        assert "search_results" in result
        assert "issues" in result


@pytest.mark.skipif(not RAG_DEBUG_AVAILABLE, reason="RAG Debugger not available")
class TestRAGDebuggerFunctions:
    """RAG Debugger 편의 함수 테스트"""

    def test_inspect_embedding_function(self):
        """inspect_embedding 편의 함수 테스트"""

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        info = inspect_embedding("Hello world", mock_embedding_function)

        assert isinstance(info, EmbeddingInfo)

    def test_compare_texts_function(self):
        """compare_texts 편의 함수 테스트"""

        # embedding_function이 필요함
        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        similarity = compare_texts("Hello", "Hi", mock_embedding_function)

        assert isinstance(similarity, SimilarityInfo)
        assert 0.0 <= similarity.cosine_similarity <= 1.0

    def test_validate_pipeline_function(self):
        """validate_pipeline 편의 함수 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1", metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_store.similarity_search = Mock(return_value=[])

        result = validate_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "chunks" in result

        mock_result = Mock()
        mock_result.document = Document(content="Test content", metadata={})
        mock_result.score = 0.9
        mock_store.similarity_search = Mock(return_value=[mock_result])

        results = rag_debugger.inspect_vector_store(mock_store, ["test query"], k=3)

        assert isinstance(results, dict)
        assert "test query" in results
        assert len(results["test query"]) == 1

    def test_rag_debugger_validate_rag_pipeline_full(self, rag_debugger):
        """전체 RAG 파이프라인 검증 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1 " * 20, metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_result = Mock()
        mock_result.document = Document(content="Test", metadata={})
        mock_result.score = 0.8
        mock_store.similarity_search = Mock(return_value=[mock_result])

        result = rag_debugger.validate_rag_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "chunks" in result
        assert "embedding_dim" in result
        assert "search_results" in result
        assert "issues" in result


@pytest.mark.skipif(not RAG_DEBUG_AVAILABLE, reason="RAG Debugger not available")
class TestRAGDebuggerFunctions:
    """RAG Debugger 편의 함수 테스트"""

    def test_inspect_embedding_function(self):
        """inspect_embedding 편의 함수 테스트"""

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        info = inspect_embedding("Hello world", mock_embedding_function)

        assert isinstance(info, EmbeddingInfo)

    def test_compare_texts_function(self):
        """compare_texts 편의 함수 테스트"""

        # embedding_function이 필요함
        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        similarity = compare_texts("Hello", "Hi", mock_embedding_function)

        assert isinstance(similarity, SimilarityInfo)
        assert 0.0 <= similarity.cosine_similarity <= 1.0

    def test_validate_pipeline_function(self):
        """validate_pipeline 편의 함수 테스트"""
        from beanllm.domain.loaders.types import Document
        from unittest.mock import Mock

        documents = [Document(content="doc1", metadata={})]
        chunks = [Document(content="chunk1", metadata={})]

        def mock_embedding_function(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]

        mock_store = Mock()
        mock_store.similarity_search = Mock(return_value=[])

        result = validate_pipeline(
            documents=documents,
            chunks=chunks,
            embedding_function=mock_embedding_function,
            store=mock_store,
            test_queries=["test query"],
        )

        assert isinstance(result, dict)
        assert "documents" in result
        assert "chunks" in result
