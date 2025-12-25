"""
Vector Stores Search Algorithms 테스트
"""

import pytest
from unittest.mock import Mock

try:
    from beanllm.vector_stores.search import SearchAlgorithms
    from beanllm.vector_stores.base import VectorSearchResult
    from beanllm.domain.loaders.types import Document

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


@pytest.mark.skipif(not SEARCH_AVAILABLE, reason="Search algorithms not available")
class TestSearchAlgorithms:
    """SearchAlgorithms 테스트"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        store = Mock()
        doc1 = Document(content="Test 1", metadata={})
        doc2 = Document(content="Test 2", metadata={})
        store.similarity_search = Mock(
            return_value=[
                VectorSearchResult(document=doc1, score=0.9),
                VectorSearchResult(document=doc2, score=0.8),
            ]
        )
        return store

    def test_hybrid_search(self, mock_vector_store):
        """Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=0.5)
        assert isinstance(results, list)
        assert len(results) <= 2
        mock_vector_store.similarity_search.assert_called_once()

    def test_hybrid_search_alpha_zero(self, mock_vector_store):
        """Alpha=0 (키워드만) Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=0.0)
        assert isinstance(results, list)

    def test_hybrid_search_alpha_one(self, mock_vector_store):
        """Alpha=1 (벡터만) Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=1.0)
        assert isinstance(results, list)

    def test_keyword_search(self, mock_vector_store):
        """키워드 검색 테스트 (기본 구현은 빈 리스트)"""
        results = SearchAlgorithms._keyword_search(mock_vector_store, "test", k=5)
        assert isinstance(results, list)
        # 기본 구현은 빈 리스트 반환
        assert len(results) == 0

    def test_combine_results(self):
        """결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        doc2 = Document(content="Test 2", metadata={})
        doc3 = Document(content="Test 3", metadata={})

        vector_results = [
            VectorSearchResult(document=doc1, score=0.9),
            VectorSearchResult(document=doc2, score=0.8),
        ]
        keyword_results = [
            VectorSearchResult(document=doc2, score=0.7),
            VectorSearchResult(document=doc3, score=0.6),
        ]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=0.5)
        assert isinstance(combined, list)
        assert len(combined) == 3  # doc1, doc2, doc3

    def test_combine_results_alpha_zero(self):
        """Alpha=0 결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        vector_results = [VectorSearchResult(document=doc1, score=0.9)]
        keyword_results = [VectorSearchResult(document=doc1, score=0.7)]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=0.0)
        assert len(combined) == 1

    def test_combine_results_alpha_one(self):
        """Alpha=1 결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        vector_results = [VectorSearchResult(document=doc1, score=0.9)]
        keyword_results = [VectorSearchResult(document=doc1, score=0.7)]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=1.0)
        assert len(combined) == 1


"""
Vector Stores Search Algorithms 테스트
"""

import pytest
from unittest.mock import Mock

try:
    from beanllm.vector_stores.search import SearchAlgorithms
    from beanllm.vector_stores.base import VectorSearchResult
    from beanllm.domain.loaders.types import Document

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


@pytest.mark.skipif(not SEARCH_AVAILABLE, reason="Search algorithms not available")
class TestSearchAlgorithms:
    """SearchAlgorithms 테스트"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        store = Mock()
        doc1 = Document(content="Test 1", metadata={})
        doc2 = Document(content="Test 2", metadata={})
        store.similarity_search = Mock(
            return_value=[
                VectorSearchResult(document=doc1, score=0.9),
                VectorSearchResult(document=doc2, score=0.8),
            ]
        )
        return store

    def test_hybrid_search(self, mock_vector_store):
        """Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=0.5)
        assert isinstance(results, list)
        assert len(results) <= 2
        mock_vector_store.similarity_search.assert_called_once()

    def test_hybrid_search_alpha_zero(self, mock_vector_store):
        """Alpha=0 (키워드만) Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=0.0)
        assert isinstance(results, list)

    def test_hybrid_search_alpha_one(self, mock_vector_store):
        """Alpha=1 (벡터만) Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=1.0)
        assert isinstance(results, list)

    def test_keyword_search(self, mock_vector_store):
        """키워드 검색 테스트 (기본 구현은 빈 리스트)"""
        results = SearchAlgorithms._keyword_search(mock_vector_store, "test", k=5)
        assert isinstance(results, list)
        # 기본 구현은 빈 리스트 반환
        assert len(results) == 0

    def test_combine_results(self):
        """결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        doc2 = Document(content="Test 2", metadata={})
        doc3 = Document(content="Test 3", metadata={})

        vector_results = [
            VectorSearchResult(document=doc1, score=0.9),
            VectorSearchResult(document=doc2, score=0.8),
        ]
        keyword_results = [
            VectorSearchResult(document=doc2, score=0.7),
            VectorSearchResult(document=doc3, score=0.6),
        ]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=0.5)
        assert isinstance(combined, list)
        assert len(combined) == 3  # doc1, doc2, doc3

    def test_combine_results_alpha_zero(self):
        """Alpha=0 결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        vector_results = [VectorSearchResult(document=doc1, score=0.9)]
        keyword_results = [VectorSearchResult(document=doc1, score=0.7)]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=0.0)
        assert len(combined) == 1

    def test_combine_results_alpha_one(self):
        """Alpha=1 결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        vector_results = [VectorSearchResult(document=doc1, score=0.9)]
        keyword_results = [VectorSearchResult(document=doc1, score=0.7)]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=1.0)
        assert len(combined) == 1


"""
Vector Stores Search Algorithms 테스트
"""

import pytest
from unittest.mock import Mock

try:
    from beanllm.vector_stores.search import SearchAlgorithms
    from beanllm.vector_stores.base import VectorSearchResult
    from beanllm.domain.loaders.types import Document

    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


@pytest.mark.skipif(not SEARCH_AVAILABLE, reason="Search algorithms not available")
class TestSearchAlgorithms:
    """SearchAlgorithms 테스트"""

    @pytest.fixture
    def mock_vector_store(self):
        """Mock VectorStore"""
        store = Mock()
        doc1 = Document(content="Test 1", metadata={})
        doc2 = Document(content="Test 2", metadata={})
        store.similarity_search = Mock(
            return_value=[
                VectorSearchResult(document=doc1, score=0.9),
                VectorSearchResult(document=doc2, score=0.8),
            ]
        )
        return store

    def test_hybrid_search(self, mock_vector_store):
        """Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=0.5)
        assert isinstance(results, list)
        assert len(results) <= 2
        mock_vector_store.similarity_search.assert_called_once()

    def test_hybrid_search_alpha_zero(self, mock_vector_store):
        """Alpha=0 (키워드만) Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=0.0)
        assert isinstance(results, list)

    def test_hybrid_search_alpha_one(self, mock_vector_store):
        """Alpha=1 (벡터만) Hybrid Search 테스트"""
        results = SearchAlgorithms.hybrid_search(mock_vector_store, "test query", k=2, alpha=1.0)
        assert isinstance(results, list)

    def test_keyword_search(self, mock_vector_store):
        """키워드 검색 테스트 (기본 구현은 빈 리스트)"""
        results = SearchAlgorithms._keyword_search(mock_vector_store, "test", k=5)
        assert isinstance(results, list)
        # 기본 구현은 빈 리스트 반환
        assert len(results) == 0

    def test_combine_results(self):
        """결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        doc2 = Document(content="Test 2", metadata={})
        doc3 = Document(content="Test 3", metadata={})

        vector_results = [
            VectorSearchResult(document=doc1, score=0.9),
            VectorSearchResult(document=doc2, score=0.8),
        ]
        keyword_results = [
            VectorSearchResult(document=doc2, score=0.7),
            VectorSearchResult(document=doc3, score=0.6),
        ]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=0.5)
        assert isinstance(combined, list)
        assert len(combined) == 3  # doc1, doc2, doc3

    def test_combine_results_alpha_zero(self):
        """Alpha=0 결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        vector_results = [VectorSearchResult(document=doc1, score=0.9)]
        keyword_results = [VectorSearchResult(document=doc1, score=0.7)]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=0.0)
        assert len(combined) == 1

    def test_combine_results_alpha_one(self):
        """Alpha=1 결과 결합 테스트"""
        doc1 = Document(content="Test 1", metadata={})
        vector_results = [VectorSearchResult(document=doc1, score=0.9)]
        keyword_results = [VectorSearchResult(document=doc1, score=0.7)]

        combined = SearchAlgorithms._combine_results(vector_results, keyword_results, alpha=1.0)
        assert len(combined) == 1



