"""
Tests for Text Splitters
"""
import pytest
from beanllm import (
    Document,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    TextSplitter,
    split_documents
)


@pytest.fixture
def sample_document():
    """샘플 문서"""
    text = """
# Introduction

This is a long document that needs to be split into smaller chunks.
It has multiple paragraphs.

## Section 1

First section content here.
Multiple lines of text.

## Section 2

Second section with more content.
This helps test the splitting.

# Conclusion

Final thoughts and summary.
    """.strip()

    return Document(content=text, metadata={"source": "test.md"})


@pytest.fixture
def long_text():
    """긴 텍스트"""
    return "AI is amazing. " * 100


class TestCharacterTextSplitter:
    """CharacterTextSplitter 테스트"""

    def test_basic_split(self):
        """기본 분할 테스트"""
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=100,
            chunk_overlap=20
        )

        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_documents(self, sample_document):
        """문서 분할 테스트"""
        splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=200
        )

        chunks = splitter.split_documents([sample_document])

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert chunks[0].metadata["source"] == "test.md"
        assert "chunk" in chunks[0].metadata


class TestRecursiveCharacterTextSplitter:
    """RecursiveCharacterTextSplitter 테스트"""

    def test_basic_split(self):
        """기본 분할 테스트"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )

        text = """
# Header

Paragraph 1 with some text.

Paragraph 2 with more text.

Final paragraph.
        """.strip()

        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        print(f"Generated {len(chunks)} chunks")

    def test_custom_separators(self):
        """커스텀 구분자 테스트"""
        splitter = RecursiveCharacterTextSplitter(
            separators=["###", "##", "#", "\n\n", "\n", " "],
            chunk_size=50,
            chunk_overlap=10
        )

        text = "# Big header\n## Smaller\n### Smallest\nContent here"
        chunks = splitter.split_text(text)

        assert len(chunks) > 0

    def test_split_documents(self, sample_document):
        """문서 분할 테스트"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=30
        )

        chunks = splitter.split_documents([sample_document])

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # 메타데이터 보존 확인
        assert chunks[0].metadata["source"] == "test.md"
        assert chunks[0].metadata["chunk"] == 0
        assert chunks[1].metadata["chunk"] == 1


class TestMarkdownHeaderTextSplitter:
    """MarkdownHeaderTextSplitter 테스트"""

    def test_split_by_headers(self):
        """헤더 기준 분할 테스트"""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
            ]
        )

        text = """
# Main Title

Introduction text.

## Section 1

Section 1 content.

### Subsection 1.1

Subsection content.

## Section 2

Section 2 content.
        """.strip()

        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # 메타데이터 확인
        has_h1 = any("H1" in chunk.metadata for chunk in chunks)
        has_h2 = any("H2" in chunk.metadata for chunk in chunks)

        assert has_h1 or has_h2

    def test_split_documents(self, sample_document):
        """문서 분할 테스트"""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header1"),
                ("##", "Header2"),
            ]
        )

        chunks = splitter.split_documents([sample_document])

        assert len(chunks) > 0

        # 원본 메타데이터 보존
        assert all(chunk.metadata.get("source") == "test.md" for chunk in chunks)


class TestTokenTextSplitter:
    """TokenTextSplitter 테스트"""

    def test_token_split(self, long_text):
        """토큰 기반 분할 테스트"""
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken not installed")

        splitter = TokenTextSplitter(
            encoding_name="cl100k_base",
            chunk_size=50,
            chunk_overlap=10
        )

        chunks = splitter.split_text(long_text)

        assert len(chunks) > 0
        print(f"Split into {len(chunks)} token-based chunks")

    def test_model_specific(self, long_text):
        """모델별 분할 테스트"""
        try:
            import tiktoken
        except ImportError:
            pytest.skip("tiktoken not installed")

        splitter = TokenTextSplitter(
            model_name="gpt-4",
            chunk_size=100,
            chunk_overlap=20
        )

        chunks = splitter.split_text(long_text)

        assert len(chunks) > 0


class TestTextSplitterFactory:
    """TextSplitter Factory 테스트"""

    def test_split_with_defaults(self, sample_document):
        """기본값 분할 테스트"""
        chunks = TextSplitter.split([sample_document])

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_split_with_strategy(self, sample_document):
        """전략별 분할 테스트"""
        # Recursive
        chunks_recursive = TextSplitter.split(
            [sample_document],
            strategy="recursive",
            chunk_size=100
        )
        assert len(chunks_recursive) > 0

        # Character
        chunks_char = TextSplitter.split(
            [sample_document],
            strategy="character",
            separator="\n\n",
            chunk_size=100
        )
        assert len(chunks_char) > 0

    def test_create_splitter(self):
        """Splitter 생성 테스트"""
        # Recursive
        splitter = TextSplitter.create(strategy="recursive", chunk_size=100)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

        # Character
        splitter2 = TextSplitter.create(strategy="character", chunk_size=100)
        assert isinstance(splitter2, CharacterTextSplitter)

    def test_unknown_strategy(self, sample_document):
        """알 수 없는 전략 테스트"""
        # Should fall back to recursive
        chunks = TextSplitter.split(
            [sample_document],
            strategy="unknown",
            chunk_size=100
        )
        assert len(chunks) > 0


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    def test_split_documents_function(self, sample_document):
        """split_documents 함수 테스트"""
        chunks = split_documents(
            [sample_document],
            chunk_size=150,
            chunk_overlap=30
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_split_documents_with_strategy(self, sample_document):
        """전략 지정 테스트"""
        chunks = split_documents(
            [sample_document],
            strategy="character",
            separator="\n\n",
            chunk_size=100
        )

        assert len(chunks) > 0


class TestIntegration:
    """통합 테스트"""

    def test_full_pipeline(self, tmp_path):
        """전체 파이프라인 테스트"""
        from beanllm import DocumentLoader, TextSplitter
        from pathlib import Path

        # 1. 문서 생성 (임시 디렉토리 사용)
        test_file = tmp_path / "test_integration.txt"
        test_file.write_text("""
AI and Machine Learning are transforming the world.

Deep learning uses neural networks with multiple layers.

Applications include computer vision and natural language processing.

The future of AI is exciting and full of possibilities.
        """.strip(), encoding="utf-8")

        try:
            # 2. 문서 로딩
            docs = DocumentLoader.load(test_file)
            assert len(docs) == 1

            # 3. 텍스트 분할
            chunks = TextSplitter.split(docs, chunk_size=100, chunk_overlap=20)
            assert len(chunks) > 0

            # 4. 메타데이터 확인
            assert all("source" in chunk.metadata for chunk in chunks)
            assert all("chunk" in chunk.metadata for chunk in chunks)

            print(f"\n✓ Full pipeline: {len(docs)} doc → {len(chunks)} chunks")

        finally:
            # 정리
            test_file.unlink()

    def test_metadata_preservation(self, sample_document):
        """메타데이터 보존 테스트"""
        # 추가 메타데이터
        sample_document.metadata["author"] = "Test Author"
        sample_document.metadata["date"] = "2024-01-01"

        chunks = TextSplitter.split([sample_document], chunk_size=100)

        # 원본 메타데이터 보존 확인
        assert all(chunk.metadata["source"] == "test.md" for chunk in chunks)
        assert all(chunk.metadata["author"] == "Test Author" for chunk in chunks)
        assert all(chunk.metadata["date"] == "2024-01-01" for chunk in chunks)

        # 청크 번호 추가 확인
        assert all("chunk" in chunk.metadata for chunk in chunks)


def test_smart_defaults():
    """스마트 기본값 테스트"""
    text = "Short text. " * 50
    doc = Document(content=text, metadata={})

    # 기본값으로 분할
    chunks = TextSplitter.split([doc])

    assert len(chunks) > 0
    print(f"\n✓ Smart defaults: {len(chunks)} chunks created")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
