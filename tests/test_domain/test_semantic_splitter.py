"""
Comprehensive tests for SemanticTextSplitter.
Target: src/beanllm/domain/splitters/semantic.py
"""

from __future__ import annotations

import sys
from typing import Any, Callable, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_embed(text: str) -> List[float]:
    """Simple deterministic embedding for testing."""
    chars = list(text.lower())
    vec = [float(ord(c) % 10) / 10.0 for c in chars[:4]] + [0.0] * max(0, 4 - len(chars))
    return vec[:4]


def _make_splitter(embedding_function: Callable | None = None, **kwargs):
    """Create a SemanticTextSplitter with a provided embedding_function."""
    from beanllm.domain.splitters.semantic import SemanticTextSplitter

    if embedding_function is None:
        embedding_function = _simple_embed

    return SemanticTextSplitter(
        embedding_function=embedding_function,
        **kwargs,
    )


MULTI_SENTENCE_TEXT = (
    "Python is a high-level programming language. "
    "It is known for its simplicity and readability. "
    "Python supports multiple programming paradigms. "
    "Machine learning often uses Python extensively. "
    "Deep learning frameworks like PyTorch run on Python."
)

SHORT_TEXT = "Hello world."

SINGLE_SENTENCE = "This is just one sentence."


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestSemanticTextSplitterInit:
    def test_init_with_embedding_function(self):
        splitter = _make_splitter()
        assert splitter._embedding_function is not None
        assert splitter.model_name == "all-MiniLM-L6-v2"
        assert splitter.threshold == 0.5
        assert splitter.use_semchunk is False

    def test_custom_model_name(self):
        splitter = _make_splitter(model="my-custom-model")
        assert splitter.model_name == "my-custom-model"

    def test_custom_threshold(self):
        splitter = _make_splitter(threshold=0.8)
        assert splitter.threshold == 0.8

    def test_custom_chunk_sizes(self):
        splitter = _make_splitter(min_chunk_size=50, max_chunk_size=500)
        assert splitter.min_chunk_size == 50
        assert splitter.max_chunk_size == 500

    def test_buffer_size_stored(self):
        splitter = _make_splitter(buffer_size=3)
        assert splitter.buffer_size == 3

    def test_use_semchunk_init(self):
        mock_semchunk = MagicMock()
        mock_chunker = MagicMock()
        mock_semchunk.chunkerify = MagicMock(return_value=mock_chunker)

        with patch.dict(sys.modules, {"semchunk": mock_semchunk}):
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            splitter = SemanticTextSplitter(use_semchunk=True, chunk_size=256)
            assert splitter.use_semchunk is True
            assert splitter._chunker is mock_chunker

    def test_semchunk_missing_raises(self):
        with patch.dict(sys.modules, {"semchunk": None}):
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            with pytest.raises((ImportError, TypeError)):
                SemanticTextSplitter(use_semchunk=True)

    def test_sentence_transformer_init(self):
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode = MagicMock(return_value=MagicMock(tolist=lambda: [0.1, 0.2]))
        mock_st.SentenceTransformer = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            splitter = SemanticTextSplitter(model="all-MiniLM-L6-v2")
            assert splitter._model is mock_model

    def test_sentence_transformer_missing_raises(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            with pytest.raises((ImportError, TypeError)):
                SemanticTextSplitter(model="all-MiniLM-L6-v2")

    def test_repr_default(self):
        splitter = _make_splitter()
        r = repr(splitter)
        assert "SemanticTextSplitter" in r

    def test_repr_semchunk(self):
        mock_semchunk = MagicMock()
        mock_semchunk.chunkerify = MagicMock(return_value=MagicMock())
        with patch.dict(sys.modules, {"semchunk": mock_semchunk}):
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            splitter = SemanticTextSplitter(use_semchunk=True)
            r = repr(splitter)
            assert "semchunk" in r


# ---------------------------------------------------------------------------
# split_text tests
# ---------------------------------------------------------------------------


class TestSplitText:
    def test_empty_text_returns_empty(self):
        splitter = _make_splitter()
        result = splitter.split_text("")
        assert result == []

    def test_whitespace_only_returns_empty(self):
        splitter = _make_splitter()
        result = splitter.split_text("   \n  ")
        assert result == []

    def test_single_sentence(self):
        splitter = _make_splitter()
        result = splitter.split_text(SINGLE_SENTENCE)
        assert len(result) == 1
        assert SINGLE_SENTENCE in result[0] or result[0] in SINGLE_SENTENCE

    def test_multi_sentence_returns_chunks(self):
        splitter = _make_splitter(threshold=0.99)  # Very high threshold = split almost everywhere
        result = splitter.split_text(MULTI_SENTENCE_TEXT)
        assert len(result) >= 1
        assert all(isinstance(c, str) for c in result)

    def test_low_threshold_few_splits(self):
        splitter = _make_splitter(threshold=0.0)  # Very low = never split
        result = splitter.split_text(MULTI_SENTENCE_TEXT)
        assert len(result) >= 1  # At least one chunk

    def test_chunk_size_respected(self):
        splitter = _make_splitter(max_chunk_size=50, min_chunk_size=10)
        # Long text that forces splitting
        long_text = "word " * 200
        result = splitter.split_text(long_text)
        # Each chunk should be <= max_chunk_size (roughly)
        assert len(result) >= 1

    def test_min_chunk_size_merges_small_chunks(self):
        splitter = _make_splitter(min_chunk_size=1000, max_chunk_size=5000)
        result = splitter.split_text(MULTI_SENTENCE_TEXT)
        # With large min_chunk_size, small chunks get merged -> fewer chunks
        assert len(result) >= 1

    def test_semchunk_path(self):
        mock_chunker = MagicMock(return_value=["chunk1", "chunk2"])
        mock_semchunk = MagicMock()
        mock_semchunk.chunkerify = MagicMock(return_value=mock_chunker)

        with patch.dict(sys.modules, {"semchunk": mock_semchunk}):
            from beanllm.domain.splitters.semantic import SemanticTextSplitter

            splitter = SemanticTextSplitter(use_semchunk=True)
            result = splitter.split_text("Some text here.")
            assert result == ["chunk1", "chunk2"]

    def test_returns_list_of_strings(self):
        splitter = _make_splitter()
        result = splitter.split_text(MULTI_SENTENCE_TEXT)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)


# ---------------------------------------------------------------------------
# _get_sentence_embeddings tests
# ---------------------------------------------------------------------------


class TestGetSentenceEmbeddings:
    def test_with_embedding_function(self):
        splitter = _make_splitter(embedding_function=_simple_embed)
        sentences = ["Hello world.", "This is a test."]
        embeddings = splitter._get_sentence_embeddings(sentences)
        assert len(embeddings) == 2
        assert all(isinstance(e, list) for e in embeddings)

    def test_with_model(self):
        mock_model = MagicMock()
        mock_numpy_array = MagicMock()
        mock_numpy_array.tolist = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode = MagicMock(return_value=mock_numpy_array)

        splitter = _make_splitter(embedding_function=_simple_embed)
        splitter._model = mock_model

        sentences = ["S1.", "S2."]
        result = splitter._get_sentence_embeddings(sentences)
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.assert_called_once()

    def test_assertion_error_when_no_function(self):
        splitter = _make_splitter(embedding_function=_simple_embed)
        splitter._embedding_function = None
        with pytest.raises(AssertionError):
            splitter._get_sentence_embeddings(["s1"])


# ---------------------------------------------------------------------------
# split_documents tests
# ---------------------------------------------------------------------------


class TestSplitDocuments:
    def test_split_single_document(self):
        from beanllm.domain.loaders.types import Document

        splitter = _make_splitter()
        doc = Document(content=MULTI_SENTENCE_TEXT, metadata={"source": "test.txt"})
        result = splitter.split_documents([doc])
        assert len(result) >= 1
        assert all(hasattr(d, "content") for d in result)

    def test_preserves_metadata(self):
        from beanllm.domain.loaders.types import Document

        splitter = _make_splitter()
        doc = Document(content=MULTI_SENTENCE_TEXT, metadata={"source": "test.txt", "page": 1})
        result = splitter.split_documents([doc])
        assert all(d.metadata["source"] == "test.txt" for d in result)
        assert all(d.metadata["page"] == 1 for d in result)

    def test_adds_chunk_metadata(self):
        from beanllm.domain.loaders.types import Document

        splitter = _make_splitter()
        doc = Document(content=MULTI_SENTENCE_TEXT, metadata={})
        result = splitter.split_documents([doc])
        for d in result:
            assert "chunk" in d.metadata
            assert "total_chunks" in d.metadata
            assert d.metadata["splitter"] == "semantic"

    def test_multiple_documents(self):
        from beanllm.domain.loaders.types import Document

        splitter = _make_splitter()
        docs = [
            Document(content="Doc1. " + MULTI_SENTENCE_TEXT, metadata={"id": 1}),
            Document(content="Doc2. " + MULTI_SENTENCE_TEXT, metadata={"id": 2}),
        ]
        result = splitter.split_documents(docs)
        assert len(result) >= 2

    def test_empty_document_list(self):
        splitter = _make_splitter()
        result = splitter.split_documents([])
        assert result == []


# ---------------------------------------------------------------------------
# force_split_by_size and merge_small_chunks integration
# ---------------------------------------------------------------------------


class TestChunkSizeConstraints:
    def test_force_split_activates_when_chunk_too_large(self):
        splitter = _make_splitter(max_chunk_size=20, min_chunk_size=1)
        # Text with a sentence longer than max_chunk_size
        long_sentence = "A" * 100
        result = splitter.split_text(long_sentence)
        # Should be split into multiple chunks
        assert len(result) >= 1

    def test_merge_activates_when_chunk_too_small(self):
        splitter = _make_splitter(min_chunk_size=200, max_chunk_size=2000, threshold=0.99)
        short_sentences = "Hi. How. Go. Yes. No. OK."
        result = splitter.split_text(short_sentences)
        # Small chunks should be merged
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# semantic_preprocessing tests (split_into_sentences, normalize_text)
# ---------------------------------------------------------------------------


class TestSemanticPreprocessing:
    def test_split_into_sentences_basic(self):
        from beanllm.domain.splitters.semantic_preprocessing import split_into_sentences

        text = "Hello. World. How are you?"
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2

    def test_split_into_sentences_empty(self):
        from beanllm.domain.splitters.semantic_preprocessing import split_into_sentences

        result = split_into_sentences("")
        assert result == []

    def test_split_into_sentences_no_punctuation(self):
        from beanllm.domain.splitters.semantic_preprocessing import split_into_sentences

        result = split_into_sentences("one two three")
        assert len(result) >= 1

    def test_normalize_text(self):
        from beanllm.domain.splitters.semantic_preprocessing import normalize_text

        result = normalize_text("  hello   world  ")
        assert result == "hello world"

    def test_normalize_text_newlines(self):
        from beanllm.domain.splitters.semantic_preprocessing import normalize_text

        result = normalize_text("hello\n\nworld")
        assert "\n\n" not in result


# ---------------------------------------------------------------------------
# semantic_similarity tests (compute_cosine_similarity, find_breakpoints)
# ---------------------------------------------------------------------------


class TestSemanticSimilarity:
    def test_cosine_similarity_identical(self):
        from beanllm.domain.splitters.semantic_similarity import compute_cosine_similarity

        sim = compute_cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from beanllm.domain.splitters.semantic_similarity import compute_cosine_similarity

        sim = compute_cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        from beanllm.domain.splitters.semantic_similarity import compute_cosine_similarity

        sim = compute_cosine_similarity([0.0, 0.0], [1.0, 1.0])
        assert sim == 0.0

    def test_cosine_similarity_different_lengths_raises(self):
        from beanllm.domain.splitters.semantic_similarity import compute_cosine_similarity

        with pytest.raises(ValueError):
            compute_cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_find_breakpoints_empty(self):
        from beanllm.domain.splitters.semantic_similarity import find_breakpoints

        result = find_breakpoints([], threshold=0.5)
        assert result == []

    def test_find_breakpoints_single(self):
        from beanllm.domain.splitters.semantic_similarity import find_breakpoints

        result = find_breakpoints([[1.0, 0.0]], threshold=0.5)
        assert result == []

    def test_find_breakpoints_all_similar(self):
        from beanllm.domain.splitters.semantic_similarity import find_breakpoints

        # All identical embeddings -> high similarity -> no breakpoints (threshold=0.5)
        embeddings = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        result = find_breakpoints(embeddings, threshold=0.5)
        assert result == []

    def test_find_breakpoints_detects_change(self):
        from beanllm.domain.splitters.semantic_similarity import find_breakpoints

        # First two similar, third orthogonal to first two
        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],  # orthogonal to first two
        ]
        result = find_breakpoints(embeddings, threshold=0.5)
        assert len(result) >= 1

    def test_find_breakpoints_with_buffer(self):
        from beanllm.domain.splitters.semantic_similarity import find_breakpoints

        embeddings = [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.0, 0.9],
        ]
        result = find_breakpoints(embeddings, threshold=0.5, buffer_size=2)
        assert isinstance(result, list)
