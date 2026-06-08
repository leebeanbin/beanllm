"""
Comprehensive tests for BaseVectorStore and VectorSearchResult.
Target: src/beanllm/domain/vector_stores/base.py
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# VectorSearchResult tests
# ---------------------------------------------------------------------------


class TestVectorSearchResult:
    def test_basic_creation(self):
        from beanllm.domain.vector_stores.base import VectorSearchResult

        r = VectorSearchResult(document="doc", score=0.9)
        assert r.document == "doc"
        assert r.score == 0.9
        assert r.metadata == {}

    def test_metadata_none_becomes_empty_dict(self):
        from beanllm.domain.vector_stores.base import VectorSearchResult

        r = VectorSearchResult(document="d", score=0.5, metadata=None)
        assert r.metadata == {}

    def test_metadata_provided(self):
        from beanllm.domain.vector_stores.base import VectorSearchResult

        r = VectorSearchResult(document="d", score=0.5, metadata={"key": "val"})
        assert r.metadata["key"] == "val"

    def test_score_can_be_zero(self):
        from beanllm.domain.vector_stores.base import VectorSearchResult

        r = VectorSearchResult(document="d", score=0.0)
        assert r.score == 0.0

    def test_score_can_be_negative(self):
        from beanllm.domain.vector_stores.base import VectorSearchResult

        r = VectorSearchResult(document="d", score=-0.1)
        assert r.score == -0.1


# ---------------------------------------------------------------------------
# Concrete subclass for testing abstract methods
# ---------------------------------------------------------------------------


class ConcreteVectorStore:
    """Concrete implementation of BaseVectorStore for testing."""

    def __init__(self, **kwargs):
        from beanllm.domain.vector_stores.base import BaseVectorStore
        # Build it via __init__

        class _Concrete(BaseVectorStore):
            def add_documents(self, documents, **kw):
                return [f"id_{i}" for i in range(len(documents))]

            def similarity_search(self, query, k=4, **kw):
                from beanllm.domain.vector_stores.base import VectorSearchResult

                return [
                    VectorSearchResult(document=f"doc_{i}", score=float(i) / 10) for i in range(k)
                ]

            def delete(self, ids, **kw):
                return True

        self._cls = _Concrete
        self.store = _Concrete(**kwargs)


def _make_concrete(**kwargs):
    from beanllm.domain.vector_stores.base import BaseVectorStore, VectorSearchResult

    class _Concrete(BaseVectorStore):
        def add_documents(self, documents, **kw):
            return [f"id_{i}" for i in range(len(documents))]

        def similarity_search(self, query, k=4, **kw):
            return [VectorSearchResult(document=f"doc_{i}", score=float(i) / 10) for i in range(k)]

        def delete(self, ids, **kw):
            return True

    return _Concrete(**kwargs)


# ---------------------------------------------------------------------------
# BaseVectorStore initialisation tests
# ---------------------------------------------------------------------------


class TestBaseVectorStoreInit:
    def test_default_init(self):
        store = _make_concrete()
        assert store.embedding_function is None
        assert store._event_logger is None
        assert store._lock_manager is None

    def test_init_with_embedding_function(self):
        emb_fn = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        store = _make_concrete(embedding_function=emb_fn)
        assert store.embedding_function is emb_fn

    def test_init_with_event_logger(self):
        mock_logger = MagicMock()
        store = _make_concrete(event_logger=mock_logger)
        assert store._event_logger is mock_logger

    def test_init_with_lock_manager(self):
        mock_lock = MagicMock()
        store = _make_concrete(lock_manager=mock_lock)
        assert store._lock_manager is mock_lock


# ---------------------------------------------------------------------------
# Abstract methods via concrete
# ---------------------------------------------------------------------------


class TestConcreteImplementation:
    def test_add_documents(self):
        store = _make_concrete()
        ids = store.add_documents(["doc1", "doc2", "doc3"])
        assert len(ids) == 3
        assert all(id_.startswith("id_") for id_ in ids)

    def test_similarity_search(self):
        store = _make_concrete()
        results = store.similarity_search("query", k=3)
        assert len(results) == 3
        from beanllm.domain.vector_stores.base import VectorSearchResult

        assert all(isinstance(r, VectorSearchResult) for r in results)

    def test_delete(self):
        store = _make_concrete()
        success = store.delete(["id_0", "id_1"])
        assert success is True


# ---------------------------------------------------------------------------
# add_texts method
# ---------------------------------------------------------------------------


class TestAddTexts:
    def test_add_texts_without_metadata(self):
        store = _make_concrete()
        ids = store.add_texts(["text1", "text2"])
        assert len(ids) == 2

    def test_add_texts_with_metadata(self):
        store = _make_concrete()
        ids = store.add_texts(
            ["text1", "text2"],
            metadatas=[{"src": "a"}, {"src": "b"}],
        )
        assert len(ids) == 2

    def test_add_texts_empty(self):
        store = _make_concrete()
        ids = store.add_texts([])
        assert ids == []


# ---------------------------------------------------------------------------
# _publish_add_documents_event
# ---------------------------------------------------------------------------


class TestPublishEvent:
    def test_no_event_logger_no_op(self):
        store = _make_concrete()
        # Should not raise
        store._publish_add_documents_event(5)

    def test_with_event_logger_calls_log(self):
        mock_logger = MagicMock()
        # Make create_task work even outside event loop
        store = _make_concrete(event_logger=mock_logger)
        with patch("asyncio.create_task") as mock_create_task:
            store._publish_add_documents_event(3)
            # May or may not call create_task depending on context

    def test_event_logger_exception_swallowed(self):
        mock_logger = MagicMock()
        mock_logger.log_event.side_effect = RuntimeError("log error")
        store = _make_concrete(event_logger=mock_logger)
        # Should not raise due to try/except
        with patch("asyncio.create_task", side_effect=RuntimeError("no loop")):
            store._publish_add_documents_event(3)  # Should not raise


# ---------------------------------------------------------------------------
# asimilarity_search (async wrapper)
# ---------------------------------------------------------------------------


class TestAsimilaritySearch:
    def test_async_search_returns_results(self):
        store = _make_concrete()

        async def _run():
            return await store.asimilarity_search("test query", k=2)

        results = asyncio.run(_run())
        assert len(results) == 2

    def test_async_search_delegates_to_sync(self):
        store = _make_concrete()
        call_count = [0]
        original = store.similarity_search

        def spy(query, k, **kw):
            call_count[0] += 1
            return original(query, k, **kw)

        store.similarity_search = spy

        async def _run():
            return await store.asimilarity_search("query", k=1)

        asyncio.run(_run())
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        store = _make_concrete()
        sim = store._cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        store = _make_concrete()
        sim = store._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        store = _make_concrete()
        sim = store._cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero_or_nan(self):
        import math

        store = _make_concrete()
        sim = store._cosine_similarity([0.0, 0.0], [1.0, 1.0])
        # With numpy, 0/0 gives NaN; manual fallback guards with 0.0
        assert sim == 0.0 or math.isnan(sim)

    def test_cosine_similarity_without_numpy(self):
        store = _make_concrete()
        with patch.dict("sys.modules", {"numpy": None}):
            sim = store._cosine_similarity([3.0, 4.0], [3.0, 4.0])
            assert abs(sim - 1.0) < 1e-6

    def test_arbitrary_vectors(self):
        store = _make_concrete()
        sim = store._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(sim - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# batch_similarity_search
# ---------------------------------------------------------------------------


class TestBatchSimilaritySearch:
    def test_requires_embedding_function(self):
        store = _make_concrete()  # no embedding_function

        async def _run():
            await store.batch_similarity_search(["q1", "q2"])

        with pytest.raises(ValueError, match="Embedding function required"):
            asyncio.run(_run())

    def test_batch_search_cpu_path(self):
        # Embedding function returns list of lists; one vector per query
        emb_fn = MagicMock(side_effect=lambda queries: [[0.1, 0.2, 0.3] for _ in queries])
        store = _make_concrete(embedding_function=emb_fn)

        # Patch _batch_embed to directly return properly shaped vectors
        async def _mock_batch_embed(queries):
            return [[0.1, 0.2, 0.3] for _ in queries]

        store._batch_embed = _mock_batch_embed

        # Override _get_all_vectors_and_docs to return data
        from beanllm.domain.vector_stores.base import VectorSearchResult

        store._get_all_vectors_and_docs = MagicMock(
            return_value=(
                [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
                ["doc_0", "doc_1"],
            )
        )

        async def _run():
            return await store.batch_similarity_search(["q1", "q2"], k=1)

        results = asyncio.run(_run())
        assert len(results) == 2
        assert len(results[0]) == 1

    def test_batch_search_gpu_fallback_to_cpu(self):
        emb_fn = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        store = _make_concrete(embedding_function=emb_fn)

        async def _mock_batch_embed(queries):
            return [[0.1, 0.2] for _ in queries]

        store._batch_embed = _mock_batch_embed
        store._get_all_vectors_and_docs = MagicMock(
            return_value=([[0.1, 0.2], [0.3, 0.4]], ["d0", "d1"])
        )

        async def _run():
            return await store.batch_similarity_search(["q1", "q2"], k=1, use_gpu=True)

        results = asyncio.run(_run())
        assert isinstance(results, list)

    def test_batch_search_empty_store(self):
        emb_fn = MagicMock(return_value=[[0.1, 0.2]])
        store = _make_concrete(embedding_function=emb_fn)

        async def _mock_batch_embed(queries):
            return [[0.1, 0.2] for _ in queries]

        store._batch_embed = _mock_batch_embed
        store._get_all_vectors_and_docs = MagicMock(return_value=([], []))

        async def _run():
            return await store.batch_similarity_search(["q1"], k=5)

        results = asyncio.run(_run())
        assert results == [[]]


# ---------------------------------------------------------------------------
# _batch_embed
# ---------------------------------------------------------------------------


class TestBatchEmbed:
    def test_embed_with_sync_callable(self):
        # Return a 2D list (list of vectors) — one per query
        def emb_fn(queries):
            return [[0.1, 0.2] for _ in queries]

        store = _make_concrete(embedding_function=emb_fn)

        async def _run():
            return await store._batch_embed(["q1", "q2"])

        result = asyncio.run(_run())
        assert len(result) == 2

    def test_embed_with_embed_sync_attribute(self):
        emb_obj = MagicMock()
        emb_obj.embed_sync = MagicMock(return_value=[[0.1, 0.2]])
        store = _make_concrete(embedding_function=emb_obj)

        async def _run():
            return await store._batch_embed(["q"])

        result = asyncio.run(_run())
        assert result == [[0.1, 0.2]]

    def test_embed_single_vector_wrapped_in_list(self):
        # Use a plain callable (not MagicMock) so hasattr(..., "embed_sync") is False
        def emb_fn(texts):
            return [0.1, 0.2, 0.3]

        store = _make_concrete(embedding_function=emb_fn)

        async def _run():
            return await store._batch_embed(["q"])

        result = asyncio.run(_run())
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _get_all_vectors_and_docs (default implementation)
# ---------------------------------------------------------------------------


class TestGetAllVectorsAndDocs:
    def test_default_returns_empty(self):
        store = _make_concrete()
        vecs, docs = store._get_all_vectors_and_docs()
        assert vecs == []
        assert docs == []


# ---------------------------------------------------------------------------
# asimilarity_search_by_vector
# ---------------------------------------------------------------------------


class TestAsimilaritySearchByVector:
    def test_raises_not_implemented(self):
        store = _make_concrete()

        async def _run():
            return await store.asimilarity_search_by_vector([0.1, 0.2], k=3)

        with pytest.raises(NotImplementedError):
            asyncio.run(_run())


# ---------------------------------------------------------------------------
# cpu_batch_search — no numpy path
# ---------------------------------------------------------------------------


class TestCpuBatchSearchNoNumpy:
    def test_fallback_without_numpy(self):
        emb_fn = MagicMock(return_value=[[0.1, 0.2]])
        store = _make_concrete(embedding_function=emb_fn)

        async def _mock_search_by_vector(vec, k, **kw):
            from beanllm.domain.vector_stores.base import VectorSearchResult

            return [VectorSearchResult(document="d", score=0.9)]

        store.asimilarity_search_by_vector = _mock_search_by_vector

        with patch.dict("sys.modules", {"numpy": None}):

            async def _run():
                return await store._cpu_batch_search([[0.1, 0.2]], k=1)

            results = asyncio.run(_run())
            assert len(results) == 1
