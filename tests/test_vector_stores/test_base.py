"""
Vector Stores Base 테스트
"""

import pytest
from unittest.mock import Mock

try:
    from beanllm.vector_stores.base import BaseVectorStore, VectorSearchResult
    from beanllm.domain.loaders.types import Document

    VECTOR_STORES_AVAILABLE = True
except ImportError:
    VECTOR_STORES_AVAILABLE = False


@pytest.mark.skipif(not VECTOR_STORES_AVAILABLE, reason="Vector stores not available")
class TestVectorSearchResult:
    """VectorSearchResult 테스트"""

    def test_vector_search_result_creation(self):
        """VectorSearchResult 생성 테스트"""
        doc = Document(content="Test", metadata={})
        result = VectorSearchResult(document=doc, score=0.9)
        assert result.document == doc
        assert result.score == 0.9
        assert result.metadata == {}

    def test_vector_search_result_with_metadata(self):
        """메타데이터 포함 VectorSearchResult 테스트"""
        doc = Document(content="Test", metadata={})
        result = VectorSearchResult(document=doc, score=0.9, metadata={"source": "test"})
        assert result.metadata == {"source": "test"}


@pytest.mark.skipif(not VECTOR_STORES_AVAILABLE, reason="Vector stores not available")
class TestBaseVectorStore:
    """BaseVectorStore 테스트"""

    def test_base_vector_store_abstract(self):
        """BaseVectorStore는 추상 클래스이므로 직접 인스턴스화 불가"""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_base_vector_store_implementation(self):
        """BaseVectorStore 구현체 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore(embedding_function=lambda x: [[0.1, 0.2]])
        assert store.embedding_function is not None

    def test_add_texts(self):
        """add_texts 메서드 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        result = store.add_texts(["Text 1", "Text 2"])
        assert len(result) == 2
        assert result[0] == "doc_0"
        assert result[1] == "doc_1"

    def test_add_texts_with_metadata(self):
        """메타데이터 포함 add_texts 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        result = store.add_texts(["Text 1", "Text 2"], metadatas=metadatas)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_asimilarity_search(self):
        """비동기 similarity_search 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                doc = Document(content="Test", metadata={})
                return [VectorSearchResult(document=doc, score=0.9)]

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        results = await store.asimilarity_search("test query", k=5)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_cosine_similarity(self):
        """코사인 유사도 계산 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert 0.0 <= similarity <= 1.0

    def test_cosine_similarity_identical(self):
        """동일한 벡터의 코사인 유사도 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_zero_norm(self):
        """영벡터의 코사인 유사도 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert similarity == 0.0


"""
Vector Stores Base 테스트
"""

import pytest
from unittest.mock import Mock

try:
    from beanllm.vector_stores.base import BaseVectorStore, VectorSearchResult
    from beanllm.domain.loaders.types import Document

    VECTOR_STORES_AVAILABLE = True
except ImportError:
    VECTOR_STORES_AVAILABLE = False


@pytest.mark.skipif(not VECTOR_STORES_AVAILABLE, reason="Vector stores not available")
class TestVectorSearchResult:
    """VectorSearchResult 테스트"""

    def test_vector_search_result_creation(self):
        """VectorSearchResult 생성 테스트"""
        doc = Document(content="Test", metadata={})
        result = VectorSearchResult(document=doc, score=0.9)
        assert result.document == doc
        assert result.score == 0.9
        assert result.metadata == {}

    def test_vector_search_result_with_metadata(self):
        """메타데이터 포함 VectorSearchResult 테스트"""
        doc = Document(content="Test", metadata={})
        result = VectorSearchResult(document=doc, score=0.9, metadata={"source": "test"})
        assert result.metadata == {"source": "test"}


@pytest.mark.skipif(not VECTOR_STORES_AVAILABLE, reason="Vector stores not available")
class TestBaseVectorStore:
    """BaseVectorStore 테스트"""

    def test_base_vector_store_abstract(self):
        """BaseVectorStore는 추상 클래스이므로 직접 인스턴스화 불가"""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_base_vector_store_implementation(self):
        """BaseVectorStore 구현체 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore(embedding_function=lambda x: [[0.1, 0.2]])
        assert store.embedding_function is not None

    def test_add_texts(self):
        """add_texts 메서드 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        result = store.add_texts(["Text 1", "Text 2"])
        assert len(result) == 2
        assert result[0] == "doc_0"
        assert result[1] == "doc_1"

    def test_add_texts_with_metadata(self):
        """메타데이터 포함 add_texts 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        result = store.add_texts(["Text 1", "Text 2"], metadatas=metadatas)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_asimilarity_search(self):
        """비동기 similarity_search 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                doc = Document(content="Test", metadata={})
                return [VectorSearchResult(document=doc, score=0.9)]

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        results = await store.asimilarity_search("test query", k=5)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_cosine_similarity(self):
        """코사인 유사도 계산 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert 0.0 <= similarity <= 1.0

    def test_cosine_similarity_identical(self):
        """동일한 벡터의 코사인 유사도 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_zero_norm(self):
        """영벡터의 코사인 유사도 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert similarity == 0.0


"""
Vector Stores Base 테스트
"""

import pytest
from unittest.mock import Mock

try:
    from beanllm.vector_stores.base import BaseVectorStore, VectorSearchResult
    from beanllm.domain.loaders.types import Document

    VECTOR_STORES_AVAILABLE = True
except ImportError:
    VECTOR_STORES_AVAILABLE = False


@pytest.mark.skipif(not VECTOR_STORES_AVAILABLE, reason="Vector stores not available")
class TestVectorSearchResult:
    """VectorSearchResult 테스트"""

    def test_vector_search_result_creation(self):
        """VectorSearchResult 생성 테스트"""
        doc = Document(content="Test", metadata={})
        result = VectorSearchResult(document=doc, score=0.9)
        assert result.document == doc
        assert result.score == 0.9
        assert result.metadata == {}

    def test_vector_search_result_with_metadata(self):
        """메타데이터 포함 VectorSearchResult 테스트"""
        doc = Document(content="Test", metadata={})
        result = VectorSearchResult(document=doc, score=0.9, metadata={"source": "test"})
        assert result.metadata == {"source": "test"}


@pytest.mark.skipif(not VECTOR_STORES_AVAILABLE, reason="Vector stores not available")
class TestBaseVectorStore:
    """BaseVectorStore 테스트"""

    def test_base_vector_store_abstract(self):
        """BaseVectorStore는 추상 클래스이므로 직접 인스턴스화 불가"""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_base_vector_store_implementation(self):
        """BaseVectorStore 구현체 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore(embedding_function=lambda x: [[0.1, 0.2]])
        assert store.embedding_function is not None

    def test_add_texts(self):
        """add_texts 메서드 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        result = store.add_texts(["Text 1", "Text 2"])
        assert len(result) == 2
        assert result[0] == "doc_0"
        assert result[1] == "doc_1"

    def test_add_texts_with_metadata(self):
        """메타데이터 포함 add_texts 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return [f"doc_{i}" for i in range(len(documents))]

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        result = store.add_texts(["Text 1", "Text 2"], metadatas=metadatas)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_asimilarity_search(self):
        """비동기 similarity_search 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                doc = Document(content="Test", metadata={})
                return [VectorSearchResult(document=doc, score=0.9)]

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        results = await store.asimilarity_search("test query", k=5)
        assert len(results) == 1
        assert results[0].score == 0.9

    def test_cosine_similarity(self):
        """코사인 유사도 계산 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert 0.0 <= similarity <= 1.0

    def test_cosine_similarity_identical(self):
        """동일한 벡터의 코사인 유사도 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [1.0, 0.0]
        vec2 = [1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_zero_norm(self):
        """영벡터의 코사인 유사도 테스트"""

        class MockVectorStore(BaseVectorStore):
            def add_documents(self, documents, **kwargs):
                return []

            def similarity_search(self, query: str, k: int = 4, **kwargs):
                return []

            def delete(self, ids, **kwargs):
                return True

        store = MockVectorStore()
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert similarity == 0.0



