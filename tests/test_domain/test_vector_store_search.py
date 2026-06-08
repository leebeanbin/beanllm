"""Tests for domain/vector_stores/search.py — SearchAlgorithms, AdvancedSearchMixin."""

from unittest.mock import MagicMock

import pytest

from beanllm.domain.vector_stores.base import VectorSearchResult
from beanllm.domain.vector_stores.search import AdvancedSearchMixin, SearchAlgorithms


def _make_result(content="doc", score=1.0):
    doc = MagicMock()
    doc.content = content
    return VectorSearchResult(document=doc, score=score)


# ---------------------------------------------------------------------------
# SearchAlgorithms._combine_results (pure Python, no external deps)
# ---------------------------------------------------------------------------


class TestCombineResults:
    def test_empty_both_returns_empty(self):
        result = SearchAlgorithms._combine_results([], [])
        assert result == []

    def test_only_vector_results(self):
        vr = [_make_result("doc1", 0.9), _make_result("doc2", 0.8)]
        result = SearchAlgorithms._combine_results(vr, [])
        assert len(result) == 2

    def test_only_keyword_results(self):
        kr = [_make_result("doc3", 0.7), _make_result("doc4", 0.6)]
        result = SearchAlgorithms._combine_results([], kr)
        assert len(result) == 2

    def test_sorted_by_score_descending(self):
        vr = [_make_result("a", 0.9), _make_result("b", 0.5)]
        result = SearchAlgorithms._combine_results(vr, [])
        assert result[0].score >= result[1].score

    def test_alpha_zero_only_keyword_weighted(self):
        vr = [_make_result("vdoc", 0.9)]
        kr = [_make_result("kdoc", 0.9)]
        result = SearchAlgorithms._combine_results(vr, kr, alpha=0.0)
        # keyword gets all weight, vector gets 0 weight
        assert len(result) >= 1

    def test_alpha_one_only_vector_weighted(self):
        vr = [_make_result("vdoc", 0.9)]
        kr = [_make_result("kdoc", 0.9)]
        result = SearchAlgorithms._combine_results(vr, kr, alpha=1.0)
        assert len(result) >= 1

    def test_returns_scored_results(self):
        vr = [_make_result("doc1", 0.9)]
        result = SearchAlgorithms._combine_results(vr, [])
        for r in result:
            assert r.score >= 0.0


# ---------------------------------------------------------------------------
# SearchAlgorithms._keyword_search (base impl returns empty)
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    def test_base_returns_empty_list(self):
        mock_store = MagicMock()
        result = SearchAlgorithms._keyword_search(mock_store, "query", k=5)
        assert result == []


# ---------------------------------------------------------------------------
# SearchAlgorithms.hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_calls_similarity_search(self):
        mock_store = MagicMock()
        vr = [_make_result("doc1", 0.9), _make_result("doc2", 0.8)]
        mock_store.similarity_search.return_value = vr
        result = SearchAlgorithms.hybrid_search(mock_store, "test query", k=2)
        mock_store.similarity_search.assert_called_once()
        assert len(result) <= 2

    def test_truncates_to_k(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [_make_result(f"doc{i}") for i in range(10)]
        result = SearchAlgorithms.hybrid_search(mock_store, "query", k=3)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# SearchAlgorithms.mmr_search
# ---------------------------------------------------------------------------


class TestMMRSearch:
    def test_returns_candidates_when_fewer_than_k(self):
        mock_store = MagicMock()
        candidates = [_make_result("a"), _make_result("b")]
        mock_store.similarity_search.return_value = candidates
        result = SearchAlgorithms.mmr_search(mock_store, "query", k=10)
        assert len(result) == 2

    def test_returns_empty_when_no_candidates(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        result = SearchAlgorithms.mmr_search(mock_store, "query", k=5)
        assert result == []

    def test_falls_back_without_embedding_function(self):
        mock_store = MagicMock()
        candidates = [_make_result(f"doc{i}") for i in range(10)]
        mock_store.similarity_search.return_value = candidates
        mock_store.embedding_function = None  # no embedding function
        result = SearchAlgorithms.mmr_search(mock_store, "query", k=3)
        assert len(result) == 3

    def test_runs_mmr_with_embedding_function(self):
        mock_store = MagicMock()
        docs = [_make_result(f"doc{i}") for i in range(8)]
        mock_store.similarity_search.return_value = docs
        mock_store.embedding_function = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        mock_store._cosine_similarity = MagicMock(return_value=0.5)
        result = SearchAlgorithms.mmr_search(mock_store, "query", k=3, fetch_k=8)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# SearchAlgorithms.rerank (requires sentence_transformers)
# ---------------------------------------------------------------------------


class TestRerank:
    def test_empty_results_returns_empty(self):
        result = SearchAlgorithms.rerank("query", [])
        assert result == []

    def test_raises_import_error_without_sentence_transformers(self):
        import sys

        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore
        try:
            with pytest.raises((ImportError, TypeError)):
                SearchAlgorithms.rerank("query", [_make_result("doc")])
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original


# ---------------------------------------------------------------------------
# AdvancedSearchMixin
# ---------------------------------------------------------------------------


class ConcreteStore(AdvancedSearchMixin):
    def similarity_search(self, query, k=4, **kwargs):
        return [_make_result(f"doc{i}") for i in range(min(k, 3))]

    @property
    def embedding_function(self):
        return None


class TestAdvancedSearchMixin:
    def test_hybrid_search_method(self):
        store = ConcreteStore()
        result = store.hybrid_search("query", k=2)
        assert isinstance(result, list)

    def test_mmr_search_method_fewer_than_k(self):
        store = ConcreteStore()
        result = store.mmr_search("query", k=10, fetch_k=20)
        assert len(result) <= 3  # ConcreteStore returns min(k, 3)

    def test_rerank_empty(self):
        store = ConcreteStore()
        result = store.rerank("query", [])
        assert result == []
