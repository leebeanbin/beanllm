"""Tests for domain/splitters: base, splitters implementations, and TextSplitter factory."""

from unittest.mock import patch

import pytest

from beanllm.domain.loaders.types import Document
from beanllm.domain.splitters.base import BaseTextSplitter
from beanllm.domain.splitters.factory import TextSplitter, split_documents
from beanllm.domain.splitters.splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

try:
    import tiktoken  # noqa: F401

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(content: str, **meta) -> Document:
    return Document(content=content, metadata=meta)


# ---------------------------------------------------------------------------
# Concrete subclass for testing BaseTextSplitter internals
# ---------------------------------------------------------------------------


class _SimpleSplitter(BaseTextSplitter):
    def split_text(self, text: str):
        return text.split(" ")


# ---------------------------------------------------------------------------
# BaseTextSplitter._merge_splits
# ---------------------------------------------------------------------------


class TestMergeSplits:
    def setup_method(self):
        self.splitter = _SimpleSplitter(chunk_size=10, chunk_overlap=0)

    def test_small_splits_merged(self):
        splitter = _SimpleSplitter(chunk_size=20, chunk_overlap=0)
        splits = ["hello", "world"]
        result = splitter._merge_splits(splits, " ")
        assert len(result) == 1
        assert "hello" in result[0]

    def test_large_split_starts_new_chunk(self):
        splits = ["short", "a" * 15]
        result = self.splitter._merge_splits(splits, " ")
        assert len(result) >= 2

    def test_empty_splits_skipped(self):
        splits = ["", "hello", ""]
        result = self.splitter._merge_splits(splits, " ")
        assert all(r for r in result)

    def test_overlap_keeps_context(self):
        splitter = _SimpleSplitter(chunk_size=10, chunk_overlap=5)
        splits = ["abcde", "fghij", "klmno"]
        result = splitter._merge_splits(splits, "")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# BaseTextSplitter.create_documents / split_documents
# ---------------------------------------------------------------------------


class TestBaseTextSplitterDocuments:
    def setup_method(self):
        self.splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=0)

    def test_create_documents_basic(self):
        texts = ["paragraph one", "paragraph two"]
        docs = self.splitter.create_documents(texts)
        assert len(docs) >= 2
        assert all(isinstance(d, Document) for d in docs)

    def test_create_documents_with_metadata(self):
        texts = ["hello"]
        metas = [{"source": "test.txt"}]
        docs = self.splitter.create_documents(texts, metas)
        assert docs[0].metadata["source"] == "test.txt"
        assert "chunk" in docs[0].metadata

    def test_split_documents_uses_content(self):
        docs = [_doc("line1\n\nline2"), _doc("another")]
        result = self.splitter.split_documents(docs)
        assert len(result) >= 2

    def test_split_documents_preserves_metadata(self):
        docs = [_doc("text", author="alice")]
        result = self.splitter.split_documents(docs)
        assert result[0].metadata["author"] == "alice"


# ---------------------------------------------------------------------------
# CharacterTextSplitter
# ---------------------------------------------------------------------------


class TestCharacterTextSplitter:
    def test_splits_on_separator(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=0)
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_no_separator_splits_chars(self):
        splitter = CharacterTextSplitter(separator="", chunk_size=3, chunk_overlap=0)
        chunks = splitter.split_text("abc")
        assert len(chunks) >= 1

    def test_small_chunk_size(self):
        splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=0)
        text = "ab cd ef gh"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        assert all(len(c) <= 10 for c in chunks)

    def test_default_separator(self):
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        assert splitter.separator == "\n\n"

    def test_overlap_does_not_crash(self):
        splitter = CharacterTextSplitter(separator=" ", chunk_size=6, chunk_overlap=2)
        chunks = splitter.split_text("aaa bbb ccc ddd")
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------


class TestRecursiveCharacterTextSplitter:
    def test_splits_on_paragraph(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Para one is here.\n\nPara two is there.\n\nPara three."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_custom_separators(self):
        splitter = RecursiveCharacterTextSplitter(
            separators=["---", "\n"], chunk_size=20, chunk_overlap=0
        )
        text = "section1---section2---section3"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1

    def test_force_split_by_size_for_huge_chunk(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=0)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_by_size_helper(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=1)
        chunks = splitter._split_by_size("abcdefghij")
        assert all(len(c) <= 5 for c in chunks)
        assert len(chunks) >= 2

    def test_empty_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = splitter.split_text("")
        assert chunks == [] or chunks == [""]

    def test_default_separators_set(self):
        splitter = RecursiveCharacterTextSplitter()
        assert "\n\n" in splitter.separators
        assert "\n" in splitter.separators

    def test_keep_separator_false(self):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=50, chunk_overlap=0, keep_separator=False
        )
        chunks = splitter.split_text("a\n\nb\n\nc")
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# MarkdownHeaderTextSplitter
# ---------------------------------------------------------------------------


class TestMarkdownHeaderTextSplitter:
    def setup_method(self):
        self.splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2")])

    def test_splits_on_header(self):
        text = "# Title\n\nContent under title.\n\n## Section\n\nSection content."
        chunks = self.splitter.split_text(text)
        assert len(chunks) >= 1

    def test_metadata_has_header(self):
        text = "# My Title\n\nsome content"
        chunks = self.splitter.split_text(text)
        found = any(d.metadata.get("H1") == "My Title" for d in chunks)
        assert found

    def test_nested_headers_sets_multiple_metadata(self):
        text = "# H1 Title\n\ncontent\n\n## H2 Section\n\nmore content"
        chunks = self.splitter.split_text(text)
        assert any(d.metadata.get("H2") == "H2 Section" for d in chunks)

    def test_return_each_line(self):
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1")], return_each_line=True
        )
        text = "# Title\nline1\nline2\nline3"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2

    def test_split_documents(self):
        docs = [_doc("# Title\n\ncontent"), _doc("## Sec\n\nmore")]
        chunks = self.splitter.split_documents(docs)
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_split_documents_merges_original_metadata(self):
        docs = [_doc("# Title\n\ncontent", source="test.md")]
        chunks = self.splitter.split_documents(docs)
        assert any(c.metadata.get("source") == "test.md" for c in chunks)

    def test_no_headers_returns_full_content(self):
        text = "Just plain text\nwith no headers"
        chunks = self.splitter.split_text(text)
        assert len(chunks) == 1
        assert "plain text" in chunks[0].content


# ---------------------------------------------------------------------------
# TextSplitter factory — create()
# ---------------------------------------------------------------------------


class TestTextSplitterCreate:
    def test_create_recursive(self):
        splitter = TextSplitter.create(strategy="recursive", chunk_size=500)
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_character(self):
        splitter = TextSplitter.create(strategy="character", chunk_size=500)
        assert isinstance(splitter, CharacterTextSplitter)

    def test_create_markdown(self):
        splitter = TextSplitter.create(
            strategy="markdown",
            headers_to_split_on=[("#", "H1"), ("##", "H2")],
        )
        assert isinstance(splitter, MarkdownHeaderTextSplitter)

    def test_create_unknown_falls_back_to_recursive(self):
        splitter = TextSplitter.create(strategy="nonexistent")
        assert isinstance(splitter, RecursiveCharacterTextSplitter)

    def test_create_markdown_with_headers_kwargs(self):
        splitter = TextSplitter.create(
            strategy="markdown",
            headers_to_split_on=[("#", "Title")],
        )
        assert isinstance(splitter, MarkdownHeaderTextSplitter)
        assert splitter.headers_to_split_on == [("#", "Title")]

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_create_token(self):
        from beanllm.domain.splitters.splitters import TokenTextSplitter

        splitter = TextSplitter.create(strategy="token", chunk_size=100)
        assert isinstance(splitter, TokenTextSplitter)


# ---------------------------------------------------------------------------
# TextSplitter convenience class methods
# ---------------------------------------------------------------------------


class TestTextSplitterClassMethods:
    def test_recursive_returns_instance(self):
        s = TextSplitter.recursive(chunk_size=200)
        assert isinstance(s, RecursiveCharacterTextSplitter)
        assert s.chunk_size == 200

    def test_recursive_with_custom_separators(self):
        s = TextSplitter.recursive(separators=["---", "\n"])
        assert "---" in s.separators

    def test_character_returns_instance(self):
        s = TextSplitter.character(separator="\n", chunk_size=300)
        assert isinstance(s, CharacterTextSplitter)
        assert s.separator == "\n"

    def test_character_default_separator(self):
        s = TextSplitter.character()
        assert s.separator == "\n\n"

    def test_markdown_returns_instance_with_defaults(self):
        s = TextSplitter.markdown()
        assert isinstance(s, MarkdownHeaderTextSplitter)
        assert len(s.headers_to_split_on) == 3  # H1, H2, H3

    def test_markdown_custom_headers(self):
        s = TextSplitter.markdown(headers_to_split_on=[("#", "Title")])
        assert s.headers_to_split_on == [("#", "Title")]

    def test_markdown_return_each_line(self):
        s = TextSplitter.markdown(return_each_line=True)
        assert s.return_each_line is True

    @pytest.mark.skipif(not TIKTOKEN_AVAILABLE, reason="tiktoken not installed")
    def test_token_returns_instance(self):
        from beanllm.domain.splitters.splitters import TokenTextSplitter

        s = TextSplitter.token(chunk_size=100)
        assert isinstance(s, TokenTextSplitter)


# ---------------------------------------------------------------------------
# TextSplitter.split() — main entry point
# ---------------------------------------------------------------------------


class TestTextSplitterSplit:
    def _docs(self, *texts):
        return [_doc(t) for t in texts]

    def test_split_basic_recursive(self):
        docs = self._docs("Hello world. " * 20)
        result = TextSplitter.split(docs, chunk_size=50, chunk_overlap=0)
        assert len(result) >= 1
        assert all(isinstance(d, Document) for d in result)

    def test_single_separator_switches_to_character(self):
        docs = self._docs("a||b||c")
        result = TextSplitter.split(
            docs, strategy="recursive", separator="||", chunk_size=10, chunk_overlap=0
        )
        assert len(result) >= 1

    def test_separators_list_switches_character_to_recursive(self):
        docs = self._docs("a\n\nb\n\nc")
        result = TextSplitter.split(
            docs, strategy="character", separators=["\n\n"], chunk_size=10, chunk_overlap=0
        )
        assert len(result) >= 1

    def test_explicit_character_strategy(self):
        docs = self._docs("part1\n\npart2\n\npart3")
        result = TextSplitter.split(
            docs, strategy="character", separator="\n\n", chunk_size=100, chunk_overlap=0
        )
        assert len(result) >= 1

    def test_empty_docs_returns_empty(self):
        result = TextSplitter.split([], chunk_size=100)
        assert result == []

    def test_multiple_docs(self):
        docs = self._docs("doc one content", "doc two content")
        result = TextSplitter.split(docs, chunk_size=100, chunk_overlap=0)
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# split_documents convenience function
# ---------------------------------------------------------------------------


class TestSplitDocumentsFunction:
    def test_basic(self):
        docs = [_doc("hello world " * 10)]
        result = split_documents(docs, chunk_size=50, chunk_overlap=0)
        assert len(result) >= 1

    def test_with_separator(self):
        docs = [_doc("a\n\nb\n\nc")]
        result = split_documents(docs, separator="\n\n", chunk_size=100)
        assert len(result) >= 1

    def test_with_separators_list(self):
        docs = [_doc("a\n\nb\n\nc")]
        result = split_documents(docs, separators=["\n\n", "\n"], chunk_size=100)
        assert len(result) >= 1

    def test_delegates_to_text_splitter(self):
        docs = [_doc("test content")]
        with patch.object(TextSplitter, "split", return_value=docs) as mock_split:
            split_documents(docs, strategy="recursive")
            mock_split.assert_called_once()
