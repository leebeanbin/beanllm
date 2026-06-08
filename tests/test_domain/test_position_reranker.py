"""Tests for domain/retrieval/reranker_position.py (PositionEngineeringReranker)."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from beanllm.domain.retrieval.reranker_position import PositionEngineeringReranker
from beanllm.domain.retrieval.types import RerankResult


def _docs(n: int) -> List[str]:
    return [f"doc_{i}" for i in range(n)]


def _scores(n: int) -> List[float]:
    return [float(n - i) for i in range(n)]  # descending: n, n-1, ..., 1


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_strategy_is_head(self) -> None:
        r = PositionEngineeringReranker()
        assert r.strategy == "head"

    def test_accepts_all_valid_strategies(self) -> None:
        for strategy in ["head", "tail", "head_tail", "side"]:
            r = PositionEngineeringReranker(strategy=strategy)
            assert r.strategy == strategy

    def test_strategy_is_lowercased(self) -> None:
        r = PositionEngineeringReranker(strategy="HEAD")
        assert r.strategy == "head"

    def test_invalid_strategy_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid strategy"):
            PositionEngineeringReranker(strategy="unknown")

    def test_base_reranker_stored(self) -> None:
        mock_base = MagicMock()
        r = PositionEngineeringReranker(base_reranker=mock_base)
        assert r.base_reranker is mock_base


# ---------------------------------------------------------------------------
# rerank — with base_reranker
# ---------------------------------------------------------------------------


class TestRerankWithBaseReranker:
    def test_delegates_to_base_reranker(self) -> None:
        expected = [RerankResult(text="d0", score=0.9, index=0)]
        mock_base = MagicMock()
        mock_base.rerank.return_value = expected

        r = PositionEngineeringReranker(base_reranker=mock_base, strategy="head")
        result = r.rerank("query", _docs(3), top_k=1)

        mock_base.rerank.assert_called_once_with(query="query", documents=_docs(3), top_k=1)
        assert result == expected

    def test_base_reranker_result_goes_through_position_strategy(self) -> None:
        base_results = [
            RerankResult(text="d0", score=3.0, index=0),
            RerankResult(text="d1", score=2.0, index=1),
            RerankResult(text="d2", score=1.0, index=2),
        ]
        mock_base = MagicMock()
        mock_base.rerank.return_value = base_results

        r = PositionEngineeringReranker(base_reranker=mock_base, strategy="tail")
        result = r.rerank("query", _docs(3))
        # tail reverses: d2, d1, d0
        assert result[0].text == "d2"
        assert result[-1].text == "d0"


# ---------------------------------------------------------------------------
# rerank — with scores
# ---------------------------------------------------------------------------


class TestRerankWithScores:
    def test_uses_scores_for_ranking(self) -> None:
        docs = ["a", "b", "c"]
        scores = [0.1, 0.9, 0.5]

        r = PositionEngineeringReranker(strategy="head")
        result = r.rerank("q", docs, scores=scores)
        # head strategy keeps order → sorted by score: b(0.9), c(0.5), a(0.1)
        assert result[0].text == "b"
        assert result[1].text == "c"
        assert result[2].text == "a"

    def test_top_k_with_scores_heapq_path(self) -> None:
        docs = _docs(10)
        scores = [float(i) for i in range(10)]

        r = PositionEngineeringReranker(strategy="head")
        result = r.rerank("q", docs, top_k=3, scores=scores)
        assert len(result) == 3

    def test_top_k_larger_than_docs_returns_all(self) -> None:
        docs = _docs(3)
        scores = [1.0, 2.0, 3.0]

        r = PositionEngineeringReranker(strategy="head")
        result = r.rerank("q", docs, top_k=10, scores=scores)
        assert len(result) == 3

    def test_mismatched_scores_length_raises_value_error(self) -> None:
        r = PositionEngineeringReranker(strategy="head")
        with pytest.raises(ValueError, match="길이가 일치하지 않습니다"):
            r.rerank("q", _docs(3), scores=[1.0, 2.0])


# ---------------------------------------------------------------------------
# rerank — default (no base, no scores)
# ---------------------------------------------------------------------------


class TestRerankDefault:
    def test_default_assigns_reciprocal_scores(self) -> None:
        docs = ["first", "second", "third"]
        r = PositionEngineeringReranker(strategy="head")
        result = r.rerank("q", docs)
        assert result[0].score == pytest.approx(1.0 / 1)
        assert result[1].score == pytest.approx(1.0 / 2)
        assert result[2].score == pytest.approx(1.0 / 3)

    def test_default_top_k_truncates(self) -> None:
        r = PositionEngineeringReranker(strategy="head")
        result = r.rerank("q", _docs(5), top_k=2)
        assert len(result) == 2

    def test_default_returns_all_without_top_k(self) -> None:
        r = PositionEngineeringReranker(strategy="head")
        result = r.rerank("q", _docs(4))
        assert len(result) == 4


# ---------------------------------------------------------------------------
# _apply_position_engineering — all strategies
# ---------------------------------------------------------------------------


class TestPositionStrategies:
    def _make_results(self, n: int) -> List[RerankResult]:
        return [RerankResult(text=f"d{i}", score=float(n - i), index=i) for i in range(n)]

    def test_head_keeps_original_order(self) -> None:
        r = PositionEngineeringReranker(strategy="head")
        results = self._make_results(4)
        out = r._apply_position_engineering(results)
        assert [x.text for x in out] == ["d0", "d1", "d2", "d3"]

    def test_tail_reverses_order(self) -> None:
        r = PositionEngineeringReranker(strategy="tail")
        results = self._make_results(4)
        out = r._apply_position_engineering(results)
        assert [x.text for x in out] == ["d3", "d2", "d1", "d0"]

    def test_head_tail_interleaves_even_then_reversed_odd(self) -> None:
        # even indices: d0, d2, d4 → left
        # odd indices: d1, d3 → right (reversed: d3, d1)
        r = PositionEngineeringReranker(strategy="head_tail")
        results = self._make_results(5)
        out = r._apply_position_engineering(results)
        assert [x.text for x in out] == ["d0", "d2", "d4", "d3", "d1"]

    def test_side_alternates_front_back(self) -> None:
        # results: d0, d1, d2, d3
        # i=0 (even) → front[0] = d0
        # i=1 (odd)  → back[3] = d1
        # i=2 (even) → front[1] = d2
        # i=3 (odd)  → back[2] = d3
        # result: [d0, d2, d3, d1]
        r = PositionEngineeringReranker(strategy="side")
        results = self._make_results(4)
        out = r._apply_position_engineering(results)
        assert out[0].text == "d0"
        assert out[3].text == "d1"
        assert len(out) == 4

    def test_empty_list_returns_empty(self) -> None:
        r = PositionEngineeringReranker(strategy="head")
        assert r._apply_position_engineering([]) == []

    def test_single_element_all_strategies(self) -> None:
        result = [RerankResult(text="only", score=1.0, index=0)]
        for strategy in ["head", "tail", "head_tail", "side"]:
            r = PositionEngineeringReranker(strategy=strategy)
            out = r._apply_position_engineering(result)
            assert len(out) == 1
            assert out[0].text == "only"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_with_no_base_reranker(self) -> None:
        r = PositionEngineeringReranker(strategy="head")
        s = repr(r)
        assert "None" in s
        assert "head" in s

    def test_repr_with_base_reranker(self) -> None:
        mock_base = MagicMock()
        mock_base.__class__.__name__ = "CrossEncoderReranker"
        r = PositionEngineeringReranker(base_reranker=mock_base, strategy="side")
        s = repr(r)
        assert "CrossEncoderReranker" in s
        assert "side" in s
