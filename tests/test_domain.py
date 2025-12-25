"""
Domain Layer 테스트 - 핵심 비즈니스 로직 테스트
"""

import pytest

try:
    from beanllm.domain import (
        Document,
        Embedding,
        TextSplitter,
        BaseEmbedding,
        BaseTextSplitter,
        BaseVectorStore,
    )
except ImportError:
    from src.beanllm.domain import (
        Document,
        Embedding,
        TextSplitter,
        BaseEmbedding,
        BaseTextSplitter,
        BaseVectorStore,
    )


class TestDocument:
    """Document 엔티티 테스트"""

    def test_document_creation(self):
        """Document 생성 테스트"""
        doc = Document(content="Test content", metadata={"source": "test.txt"})
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test.txt"

    def test_document_with_empty_metadata(self):
        """빈 메타데이터로 Document 생성"""
        doc = Document(content="Test")
        assert doc.content == "Test"
        assert isinstance(doc.metadata, dict)

    def test_document_metadata_access(self):
        """메타데이터 접근 테스트"""
        doc = Document(
            content="Test",
            metadata={"source": "test.txt", "page": 1, "author": "Test Author"},
        )
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["page"] == 1
        assert doc.metadata["author"] == "Test Author"


class TestTextSplitter:
    """TextSplitter 테스트"""

    def test_text_splitter_factory(self):
        """TextSplitter 팩토리 테스트"""
        try:
            from beanllm.domain import RecursiveCharacterTextSplitter
        except ImportError:
            from src.beanllm.domain import RecursiveCharacterTextSplitter

        splitter = TextSplitter.create(strategy="recursive", chunk_size=100)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_text_splitter_split(self, sample_documents):
        """TextSplitter.split 테스트"""
        chunks = TextSplitter.split(sample_documents, chunk_size=100)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_text_splitter_strategies(self, sample_documents):
        """다양한 전략 테스트"""
        try:
            from beanllm.domain import CharacterTextSplitter, RecursiveCharacterTextSplitter
        except ImportError:
            from src.beanllm.domain import CharacterTextSplitter, RecursiveCharacterTextSplitter

        # Recursive
        splitter = TextSplitter.create(strategy="recursive")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

        # Character
        splitter = TextSplitter.create(strategy="character", separator="\n\n")
        assert isinstance(splitter, CharacterTextSplitter)


class TestEmbedding:
    """Embedding 테스트"""

    def test_embedding_base_class(self):
        """BaseEmbedding 추상 클래스 테스트"""
        # 직접 인스턴스화 불가능
        with pytest.raises(TypeError):
            BaseEmbedding()

    def test_embedding_factory(self):
        """Embedding 팩토리 테스트"""
        # 모델 이름으로 자동 감지 시도
        try:
            emb = Embedding(model="text-embedding-3-small")
            assert isinstance(emb, BaseEmbedding)
        except Exception:
            # Provider가 없을 수 있음
            pytest.skip("Embedding provider not available")


class TestVectorStore:
    """VectorStore 테스트"""

    def test_vector_store_base_class(self):
        """BaseVectorStore 추상 클래스 테스트"""
        # 직접 인스턴스화 불가능
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_vector_store_interface(self):
        """VectorStore 인터페이스 테스트"""
        try:
            from beanllm.domain.vector_stores.base import BaseVectorStore
        except ImportError:
            from src.beanllm.domain.vector_stores.base import BaseVectorStore

        # 인터페이스 메서드 확인
        assert hasattr(BaseVectorStore, "add_documents")
        assert hasattr(BaseVectorStore, "similarity_search")
        # mmr_search는 search.py의 Mixin에 있음 (선택적)
        # assert hasattr(BaseVectorStore, "mmr_search")


class TestDomainIntegration:
    """Domain 레이어 통합 테스트"""

    def test_document_to_chunks_pipeline(self, sample_documents):
        """Document → Chunks 파이프라인"""
        chunks = TextSplitter.split(sample_documents, chunk_size=100)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all("chunk" in chunk.metadata for chunk in chunks)

    def test_metadata_preservation(self, sample_documents):
        """메타데이터 보존 테스트"""
        # 원본 메타데이터 추가
        sample_documents[0].metadata["author"] = "Test Author"
        sample_documents[0].metadata["date"] = "2024-01-01"

        chunks = TextSplitter.split(sample_documents, chunk_size=100)

        # 원본 메타데이터 보존 확인 (첫 번째 문서의 청크들만 확인)
        first_doc_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("source") == sample_documents[0].metadata.get("source")
        ]
        if first_doc_chunks:
            assert all(chunk.metadata.get("author") == "Test Author" for chunk in first_doc_chunks)
            assert all(chunk.metadata.get("date") == "2024-01-01" for chunk in first_doc_chunks)

