"""
Text Splitters 테스트 - 텍스트 분할 테스트
"""

import pytest

from llmkit.domain.loaders import Document


class TestTextSplitter:
    """TextSplitter 테스트"""

    @pytest.fixture
    def sample_document(self):
        """샘플 문서"""
        return Document(
            content="This is a test document. " * 10,
            metadata={"source": "test.txt"},
        )

    def test_recursive_character_splitter(self, sample_document):
        """RecursiveCharacterTextSplitter 테스트"""
        try:
            from llmkit.domain.splitters.splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
            chunks = splitter.split_documents([sample_document])

            assert isinstance(chunks, list)
            assert len(chunks) > 0
            assert all(isinstance(chunk, Document) for chunk in chunks)
        except ImportError:
            pytest.skip("TextSplitter not available")

    def test_character_splitter(self, sample_document):
        """CharacterTextSplitter 테스트"""
        try:
            from llmkit.domain.splitters.splitters import CharacterTextSplitter

            splitter = CharacterTextSplitter(chunk_size=50, separator=" ")
            chunks = splitter.split_documents([sample_document])

            assert isinstance(chunks, list)
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("TextSplitter not available")

    def test_text_splitter_factory(self, sample_document):
        """TextSplitter 팩토리 테스트"""
        try:
            from llmkit.domain.splitters.factory import TextSplitter

            splitter = TextSplitter.create(strategy="recursive", chunk_size=50)
            chunks = splitter.split_documents([sample_document])

            assert isinstance(chunks, list)
            assert len(chunks) > 0
        except ImportError:
            pytest.skip("TextSplitter not available")


