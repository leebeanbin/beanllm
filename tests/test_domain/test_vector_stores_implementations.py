"""
Vector Store Implementations 테스트 - 실제 구현체 테스트
"""

import pytest
from unittest.mock import Mock, patch

from llmkit.domain.loaders import Document
from llmkit.domain.vector_stores.base import VectorSearchResult


class TestChromaVectorStore:
    """ChromaVectorStore 테스트"""

    @pytest.fixture
    def mock_embedding_function(self):
        """Mock 임베딩 함수"""
        return Mock(return_value=[[0.1, 0.2, 0.3]])

    def test_chroma_vector_store_initialization(self, mock_embedding_function):
        """ChromaVectorStore 초기화 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import ChromaVectorStore

            store = ChromaVectorStore(
                collection_name="test_collection",
                embedding_function=mock_embedding_function,
            )
            assert store is not None
            assert store.collection_name == "test_collection"
        except ImportError:
            pytest.skip("Chroma not installed")

    def test_chroma_add_documents(self, mock_embedding_function):
        """Chroma 문서 추가 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import ChromaVectorStore

            # Mock embedding_function이 각 텍스트마다 하나의 벡터를 반환하도록 설정
            def mock_embedding(texts):
                return [[0.1, 0.2, 0.3] for _ in texts]

            store = ChromaVectorStore(
                collection_name="test_collection",
                embedding_function=mock_embedding,
            )
            documents = [
                Document(content="Test 1", metadata={"id": 1}),
                Document(content="Test 2", metadata={"id": 2}),
            ]

            ids = store.add_documents(documents)

            assert isinstance(ids, list)
            assert len(ids) == 2
        except ImportError:
            pytest.skip("Chroma not installed")

    def test_chroma_similarity_search(self, mock_embedding_function):
        """Chroma 유사도 검색 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import ChromaVectorStore

            store = ChromaVectorStore(
                collection_name="test_collection",
                embedding_function=mock_embedding_function,
            )

            results = store.similarity_search("test query", k=5)

            assert isinstance(results, list)
        except ImportError:
            pytest.skip("Chroma not installed")


class TestPineconeVectorStore:
    """PineconeVectorStore 테스트"""

    @pytest.fixture
    def mock_embedding_function(self):
        """Mock 임베딩 함수"""
        return Mock(return_value=[[0.1, 0.2, 0.3]])

    def test_pinecone_vector_store_initialization(self, mock_embedding_function):
        """PineconeVectorStore 초기화 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import PineconeVectorStore

            store = PineconeVectorStore(
                index_name="test_index",
                embedding_function=mock_embedding_function,
            )
            assert store is not None
        except ImportError:
            pytest.skip("Pinecone not installed")


class TestFAISSVectorStore:
    """FAISSVectorStore 테스트"""

    @pytest.fixture
    def mock_embedding_function(self):
        """Mock 임베딩 함수"""
        return Mock(return_value=[[0.1, 0.2, 0.3]])

    def test_faiss_vector_store_initialization(self, mock_embedding_function):
        """FAISSVectorStore 초기화 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import FAISSVectorStore

            store = FAISSVectorStore(embedding_function=mock_embedding_function)
            assert store is not None
        except ImportError:
            pytest.skip("FAISS not installed")

    def test_faiss_add_documents(self, mock_embedding_function):
        """FAISS 문서 추가 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import FAISSVectorStore

            store = FAISSVectorStore(embedding_function=mock_embedding_function)
            documents = [
                Document(content="Test 1", metadata={"id": 1}),
                Document(content="Test 2", metadata={"id": 2}),
            ]

            ids = store.add_documents(documents)

            assert isinstance(ids, list)
            assert len(ids) == 2
        except ImportError:
            pytest.skip("FAISS not installed")

    def test_faiss_similarity_search(self, mock_embedding_function):
        """FAISS 유사도 검색 테스트"""
        try:
            from llmkit.domain.vector_stores.implementations import FAISSVectorStore

            store = FAISSVectorStore(embedding_function=mock_embedding_function)
            documents = [
                Document(content="Test 1", metadata={"id": 1}),
                Document(content="Test 2", metadata={"id": 2}),
            ]
            store.add_documents(documents)

            results = store.similarity_search("test query", k=5)

            assert isinstance(results, list)
        except ImportError:
            pytest.skip("FAISS not installed")


