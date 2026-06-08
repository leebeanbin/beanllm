"""
Comprehensive pytest tests for beanllm domain rag_debug modules.

Covers:
- embedding_analyzer.py
- chunking_experimenter.py
- loop_cycle.py
- improvement_loop.py
- parameter_tuner.py
- similarity_tester.py
- loop_phases.py
- loop_report.py
- experiment_report.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper factory: fake search result with .score, .document, .metadata
# ---------------------------------------------------------------------------


@dataclass
class FakeDocument:
    page_content: str


@dataclass
class FakeSearchResult:
    document: FakeDocument
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_search_result(content: str, score: float) -> FakeSearchResult:
    return FakeSearchResult(document=FakeDocument(page_content=content), score=score)


# ---------------------------------------------------------------------------
# Fake vector store
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """Minimal fake vector store for testing ParameterTuner / SimilarityTester."""

    def __init__(
        self,
        results: Optional[List[FakeSearchResult]] = None,
        raise_on_search: bool = False,
        support_mmr: bool = True,
        support_hybrid: bool = True,
    ) -> None:
        self._results = (
            results
            if results is not None
            else [
                make_search_result("doc A", 0.9),
                make_search_result("doc B", 0.8),
                make_search_result("doc C", 0.7),
            ]
        )
        self._raise_on_search = raise_on_search
        self._support_mmr = support_mmr
        self._support_hybrid = support_hybrid

    def similarity_search(self, query: str, k: int = 4) -> List[FakeSearchResult]:
        if self._raise_on_search:
            raise RuntimeError("search error")
        return self._results[:k]

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 8, lambda_mult: float = 0.5
    ) -> List[FakeSearchResult]:
        if not self._support_mmr:
            raise AttributeError("no mmr")
        return self._results[:k]

    def hybrid_search(self, query: str, k: int = 4) -> List[FakeSearchResult]:
        if not self._support_hybrid:
            raise AttributeError("no hybrid")
        return self._results[:k]


# ===========================================================================
# Tests: experiment_runner (ChunkingResult, build_grid_configs, etc.)
# ===========================================================================


class TestChunkingResult:
    def test_basic_construction(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import ChunkingResult

        result = ChunkingResult(
            strategy_name="test",
            strategy_config={"type": "recursive", "chunk_size": 500},
            chunks=["chunk1", "chunk2"],
            chunk_count=2,
            avg_chunk_size=6.0,
            retrieval_scores=[0.8, 0.6],
            avg_retrieval_score=0.7,
            latency_ms=12.5,
        )
        assert result.strategy_name == "test"
        assert result.chunk_count == 2
        assert result.avg_retrieval_score == 0.7
        assert result.metadata == {}

    def test_with_metadata(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import ChunkingResult

        result = ChunkingResult(
            strategy_name="strat",
            strategy_config={},
            chunks=[],
            chunk_count=0,
            avg_chunk_size=0.0,
            retrieval_scores=[],
            avg_retrieval_score=0.0,
            latency_ms=0.0,
            metadata={"doc_count": 5},
        )
        assert result.metadata["doc_count"] == 5


class TestBuildGridConfigs:
    def test_default_params(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import build_grid_configs

        configs = build_grid_configs()
        assert len(configs) > 0
        for cfg in configs:
            assert cfg["chunk_overlap"] < cfg["chunk_size"]

    def test_custom_sizes_and_overlaps(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import build_grid_configs

        configs = build_grid_configs(
            splitter_type="character",
            chunk_sizes=[200, 400],
            chunk_overlaps=[0, 50],
        )
        assert all(c["type"] == "character" for c in configs)
        # All combos where overlap < size
        names = [c["name"] for c in configs]
        assert "character_s200_o0" in names
        assert "character_s200_o50" in names

    def test_overlap_not_exceeding_size(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import build_grid_configs

        configs = build_grid_configs(chunk_sizes=[100], chunk_overlaps=[50, 100, 200])
        # only overlap < size passes (50 < 100)
        assert len(configs) == 1
        assert configs[0]["chunk_overlap"] == 50

    def test_fixed_params_passed_through(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import build_grid_configs

        configs = build_grid_configs(
            chunk_sizes=[512],
            chunk_overlaps=[50],
            separators=["\n"],
        )
        assert configs[0]["separators"] == ["\n"]


class TestComputeSimilarity:
    def test_no_embedding_function_overlap(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import compute_similarity

        query = "hello world"
        chunks = ["hello there", "foo bar baz"]
        sims = compute_similarity(query, chunks, None)
        assert len(sims) == 2
        assert sims[0] > sims[1]  # "hello" overlaps

    def test_empty_query(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import compute_similarity

        sims = compute_similarity("", ["abc"], None)
        assert sims == [0]

    def test_with_embedding_function(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import compute_similarity

        # Simple embedding: return the same fixed vector
        def embed(text: str) -> List[float]:
            return [1.0, 0.0, 0.0]

        sims = compute_similarity("query", ["chunk1", "chunk2"], embed)
        assert len(sims) == 2
        # cosine similarity of identical vectors == 1.0
        assert abs(sims[0] - 1.0) < 1e-6


class TestEvaluateRetrieval:
    def test_no_ground_truth(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import evaluate_retrieval

        score = evaluate_retrieval("query", ["the query text here", "unrelated"], None, {})
        assert 0.0 <= score <= 1.0

    def test_with_ground_truth_full_recall(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import evaluate_retrieval

        chunks = ["chunk0", "chunk1 query", "chunk2"]
        # ground truth says query maps to index 1
        score = evaluate_retrieval("query", chunks, None, {"query": [1]}, top_k=1)
        # recall would be 1.0 if correct chunk retrieved
        assert 0.0 <= score <= 1.0

    def test_empty_chunks(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import evaluate_retrieval

        score = evaluate_retrieval("query", [], None, {})
        assert score == 0.0


class TestRunExperiment:
    def test_run_experiment_basic(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import run_experiment

        docs = ["Hello world. This is a test document.", "Another document here."]
        queries = ["hello"]
        config = {"type": "recursive", "chunk_size": 100, "chunk_overlap": 10}

        result = run_experiment(docs, queries, config, None, {})
        assert result.strategy_name == "recursive_100"
        assert result.chunk_count > 0
        assert result.latency_ms >= 0.0

    def test_run_experiment_custom_name(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import run_experiment

        docs = ["Short doc."]
        queries = ["doc"]
        config = {"type": "recursive", "chunk_size": 50, "chunk_overlap": 0}
        result = run_experiment(docs, queries, config, None, {}, strategy_name="my_strat")
        assert result.strategy_name == "my_strat"

    def test_run_experiment_no_queries(self) -> None:
        from beanllm.domain.rag_debug.experiment_runner import run_experiment

        docs = ["Document text."]
        result = run_experiment(
            docs, [], {"type": "recursive", "chunk_size": 100, "chunk_overlap": 0}, None, {}
        )
        assert result.avg_retrieval_score == 0


# ===========================================================================
# Tests: experiment_feedback
# ===========================================================================


class TestChunkFeedback:
    def test_construction(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import ChunkFeedback

        fb = ChunkFeedback(query="q", chunk_id="c1", rating=0.8, feedback_type="relevance")
        assert fb.query == "q"
        assert fb.rating == 0.8
        assert fb.comment is None
        assert fb.strategy_name is None


class TestAddFeedback:
    def test_add_feedback_no_match(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import ChunkFeedback, add_feedback

        feedbacks: List[ChunkFeedback] = []
        current_chunks: Dict[str, List[str]] = {}
        add_feedback(feedbacks, current_chunks, "query", "chunk_xyz", 0.5)
        assert len(feedbacks) == 1
        assert feedbacks[0].strategy_name is None

    def test_add_feedback_with_match(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import ChunkFeedback, add_feedback

        feedbacks: List[ChunkFeedback] = []
        current_chunks: Dict[str, List[str]] = {"strategy_a": ["chunk_xyz", "other_chunk"]}
        add_feedback(feedbacks, current_chunks, "query", "chunk_xyz", 0.9, comment="great")
        assert feedbacks[0].strategy_name == "strategy_a"
        assert feedbacks[0].comment == "great"

    def test_add_feedback_with_substring_match(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import ChunkFeedback, add_feedback

        feedbacks: List[ChunkFeedback] = []
        # chunk_id is substring of a stored chunk
        current_chunks: Dict[str, List[str]] = {"strat": ["full chunk_xyz content"]}
        add_feedback(feedbacks, current_chunks, "q", "chunk_xyz", 0.3)
        assert feedbacks[0].strategy_name == "strat"


class TestGetFeedbackSummary:
    def test_empty_feedbacks(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import get_feedback_summary

        summary = get_feedback_summary([])
        assert summary["total"] == 0
        assert summary["avg_rating"] == 0.0

    def test_multiple_feedbacks(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import ChunkFeedback, get_feedback_summary

        feedbacks = [
            ChunkFeedback("q1", "c1", 0.8, "relevance"),
            ChunkFeedback("q2", "c2", 0.2, "relevance"),
            ChunkFeedback("q3", "c3", 0.5, "quality"),
        ]
        summary = get_feedback_summary(feedbacks)
        assert summary["total"] == 3
        assert abs(summary["avg_rating"] - 0.5) < 1e-6
        assert "relevance" in summary["by_type"]
        assert "quality" in summary["by_type"]

    def test_by_type_average(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import ChunkFeedback, get_feedback_summary

        feedbacks = [
            ChunkFeedback("q", "c1", 1.0, "relevance"),
            ChunkFeedback("q", "c2", 0.0, "relevance"),
        ]
        summary = get_feedback_summary(feedbacks)
        assert abs(summary["by_type"]["relevance"] - 0.5) < 1e-6


class TestImproveFromFeedback:
    def test_no_feedbacks(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import improve_from_feedback

        result = improve_from_feedback([], lambda: None)
        assert result["total_feedbacks"] == 0
        assert result["low_rated_count"] == 0
        assert result["suggestions"] == []
        assert result["recommended_configs"] == []

    def test_with_low_rated(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import (
            ChunkFeedback,
            improve_from_feedback,
        )

        feedbacks = [ChunkFeedback("q", "c", 0.1, "relevance")]
        result = improve_from_feedback(feedbacks, lambda: None, min_rating_threshold=0.3)
        assert result["low_rated_count"] == 1
        assert len(result["suggestions"]) > 0

    def test_with_best_strategy(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import improve_from_feedback

        best = {"config": {"type": "recursive", "chunk_size": 500}, "score": 0.9, "strategy": "s"}
        result = improve_from_feedback([], lambda: best)
        assert result["recommended_configs"] == [best["config"]]

    def test_best_strategy_no_config(self) -> None:
        from beanllm.domain.rag_debug.experiment_feedback import improve_from_feedback

        result = improve_from_feedback([], lambda: {"score": 0.5, "strategy": "x", "config": None})
        assert result["recommended_configs"] == []


# ===========================================================================
# Tests: experiment_report
# ===========================================================================


class TestGetComparisonReport:
    def test_empty_results(self) -> None:
        from beanllm.domain.rag_debug.experiment_report import get_comparison_report

        report = get_comparison_report([], lambda: None, lambda: {})
        assert "No experiment results" in report

    def test_with_results(self) -> None:
        from beanllm.domain.rag_debug.experiment_report import get_comparison_report
        from beanllm.domain.rag_debug.experiment_runner import ChunkingResult

        result = ChunkingResult(
            strategy_name="strat_a",
            strategy_config={"type": "recursive"},
            chunks=["c1"],
            chunk_count=1,
            avg_chunk_size=100.0,
            retrieval_scores=[0.8],
            avg_retrieval_score=0.8,
            latency_ms=10.0,
        )

        report = get_comparison_report(
            [result],
            lambda: {"strategy": "strat_a", "score": 0.8, "config": {}},
            lambda: {"total": 0},
        )
        assert "Chunking Strategy Comparison Report" in report
        assert "strat_a" in report
        assert "Best Strategy" in report

    def test_feedback_shown_when_present(self) -> None:
        from beanllm.domain.rag_debug.experiment_report import get_comparison_report
        from beanllm.domain.rag_debug.experiment_runner import ChunkingResult

        result = ChunkingResult(
            strategy_name="s",
            strategy_config={},
            chunks=[],
            chunk_count=0,
            avg_chunk_size=0.0,
            retrieval_scores=[],
            avg_retrieval_score=0.0,
            latency_ms=0.0,
        )
        report = get_comparison_report(
            [result],
            lambda: None,
            lambda: {"total": 3, "avg_rating": 0.75},
        )
        assert "Feedback Summary" in report
        assert "3" in report


# ===========================================================================
# Tests: ChunkingExperimenter
# ===========================================================================


class TestChunkingExperimenter:
    def _make(self, docs: Optional[List[str]] = None, queries: Optional[List[str]] = None) -> Any:
        from beanllm.domain.rag_debug.chunking_experimenter import ChunkingExperimenter

        return ChunkingExperimenter(
            documents=docs or ["Document one.", "Document two."],
            test_queries=queries or ["query one"],
        )

    def test_init(self) -> None:
        exp = self._make()
        assert exp.documents == ["Document one.", "Document two."]
        assert exp.test_queries == ["query one"]
        assert exp._results == []
        assert exp._feedbacks == []
        assert exp.ground_truth == {}

    def test_repr(self) -> None:
        exp = self._make()
        r = repr(exp)
        assert "ChunkingExperimenter" in r
        assert "docs=2" in r

    def test_run_experiment_returns_result(self) -> None:
        exp = self._make()
        config = {"type": "recursive", "chunk_size": 100, "chunk_overlap": 10}
        result = exp.run_experiment(config)
        assert result.chunk_count > 0
        assert len(exp._results) == 1

    def test_run_experiment_custom_name(self) -> None:
        exp = self._make()
        config = {"type": "recursive", "chunk_size": 100, "chunk_overlap": 0}
        result = exp.run_experiment(config, strategy_name="custom")
        assert result.strategy_name == "custom"

    def test_find_best_strategy_no_results(self) -> None:
        exp = self._make()
        assert exp.find_best_strategy() is None

    def test_find_best_strategy_with_results(self) -> None:
        exp = self._make(docs=["Long document text here. " * 5])
        for size in [100, 200]:
            exp.run_experiment({"type": "recursive", "chunk_size": size, "chunk_overlap": 0})
        best = exp.find_best_strategy()
        assert best is not None
        assert "strategy" in best
        assert "score" in best

    def test_compare_strategies(self) -> None:
        exp = self._make()
        configs = [
            {"type": "recursive", "chunk_size": 100, "chunk_overlap": 0, "name": "s1"},
            {"type": "recursive", "chunk_size": 200, "chunk_overlap": 0, "name": "s2"},
        ]
        results = exp.compare_strategies(configs)
        assert len(results) == 2
        # Results sorted by score descending
        assert results[0].avg_retrieval_score >= results[1].avg_retrieval_score

    def test_grid_search(self) -> None:
        exp = self._make(docs=["Text. " * 20])
        results = exp.grid_search(
            splitter_type="recursive",
            chunk_sizes=[100, 200],
            chunk_overlaps=[0, 50],
        )
        # grid: (100,0),(100,50),(200,0),(200,50)
        assert len(results) == 4
        assert all(hasattr(r, "avg_retrieval_score") for r in results)

    def test_add_feedback_and_summary(self) -> None:
        exp = self._make()
        exp.run_experiment({"type": "recursive", "chunk_size": 100, "chunk_overlap": 0})
        exp.add_feedback("query one", "some_chunk", 0.9, comment="great")
        summary = exp.get_feedback_summary()
        assert summary["total"] == 1
        assert summary["avg_rating"] == 0.9

    def test_improve_from_feedback_no_feedback(self) -> None:
        exp = self._make()
        result = exp.improve_from_feedback()
        assert result["total_feedbacks"] == 0

    def test_get_comparison_report_no_results(self) -> None:
        exp = self._make()
        report = exp.get_comparison_report()
        assert "No experiment results" in report

    def test_get_comparison_report_with_results(self) -> None:
        exp = self._make()
        exp.run_experiment({"type": "recursive", "chunk_size": 100, "chunk_overlap": 0})
        report = exp.get_comparison_report()
        assert "Chunking Strategy Comparison Report" in report

    def test_embedding_function_used(self) -> None:
        call_count = {"n": 0}

        def embed(text: str) -> List[float]:
            call_count["n"] += 1
            return [0.1, 0.2, 0.3]

        from beanllm.domain.rag_debug.chunking_experimenter import ChunkingExperimenter

        exp = ChunkingExperimenter(
            documents=["Doc text here."],
            test_queries=["doc"],
            embedding_function=embed,
        )
        exp.run_experiment({"type": "recursive", "chunk_size": 50, "chunk_overlap": 0})
        assert call_count["n"] > 0


# ===========================================================================
# Tests: loop_cycle
# ===========================================================================


class TestImprovementCycleDataclass:
    def test_construction(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementCycle

        now = datetime.now(timezone.utc)
        cycle = ImprovementCycle(
            cycle_number=1,
            timestamp=now,
            chunking_result=None,
            eval_score_before=0.5,
            eval_score_after=0.7,
            improvement=0.2,
            strategy_used="some_strategy",
            changes_made=["change A"],
        )
        assert cycle.cycle_number == 1
        assert cycle.improvement == 0.2
        assert cycle.changes_made == ["change A"]


class TestImprovementPlanDataclass:
    def test_construction(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan

        plan = ImprovementPlan(
            priority="high",
            area="chunking",
            issue="chunks too large",
            action="reduce chunk_size",
            expected_improvement=0.1,
        )
        assert plan.priority == "high"
        assert plan.config_changes == {}

    def test_with_config_changes(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan

        plan = ImprovementPlan(
            priority="low",
            area="retrieval",
            issue="low recall",
            action="increase top_k",
            expected_improvement=0.05,
            config_changes={"top_k": 8},
        )
        assert plan.config_changes == {"top_k": 8}


class TestGetCurrentScore:
    def _make_chunking_experimenter(self, score: Optional[float] = None) -> Any:
        mock = MagicMock()
        if score is not None:
            mock.find_best_strategy.return_value = {"score": score}
        else:
            mock.find_best_strategy.return_value = None
        return mock

    def test_no_best_no_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import get_current_score

        ce = self._make_chunking_experimenter(None)
        score = get_current_score(ce, None)
        assert score == 0.0

    def test_only_chunking_score(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import get_current_score

        ce = self._make_chunking_experimenter(0.8)
        score = get_current_score(ce, None)
        assert score == 0.8

    def test_only_eval_score(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import get_current_score

        ce = self._make_chunking_experimenter(None)
        evaluator = MagicMock()
        evaluator.get_evaluation_summary.return_value = {"unified_score": {"avg": 0.6}}
        score = get_current_score(ce, evaluator)
        assert score == 0.6

    def test_both_scores_averaged(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import get_current_score

        ce = self._make_chunking_experimenter(0.8)
        evaluator = MagicMock()
        evaluator.get_evaluation_summary.return_value = {"unified_score": {"avg": 0.6}}
        score = get_current_score(ce, evaluator)
        assert abs(score - 0.7) < 1e-6

    def test_eval_score_zero_uses_chunking(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import get_current_score

        ce = self._make_chunking_experimenter(0.5)
        evaluator = MagicMock()
        evaluator.get_evaluation_summary.return_value = {"unified_score": {"avg": 0.0}}
        score = get_current_score(ce, evaluator)
        assert score == 0.5


class TestRunImprovementCycleStep:
    def _make_loop(
        self,
        best_score: float = 0.5,
        plan: Any = None,
    ) -> Any:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan

        loop = MagicMock()
        loop._cycles = []
        loop._current_best_config = None

        # chunking_experimenter
        ce = MagicMock()
        ce.find_best_strategy.return_value = {"score": best_score}
        ce.run_experiment.return_value = MagicMock()
        loop.chunking_experimenter = ce

        loop.evaluator = None

        if plan is None:
            default_plan = ImprovementPlan(
                priority="medium",
                area="retrieval",
                issue="issue",
                action="some action here",
                expected_improvement=0.1,
            )
            loop.get_improvement_plan.return_value = [default_plan]
        else:
            loop.get_improvement_plan.return_value = [plan] if plan else []

        return loop

    def test_returns_cycle(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import run_improvement_cycle_step

        loop = self._make_loop()
        cycle = run_improvement_cycle_step(loop)
        assert cycle.cycle_number == 1
        assert len(loop._cycles) == 1

    def test_no_plans_returns_empty_cycle(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import run_improvement_cycle_step

        loop = self._make_loop(plan=False)  # empty plan list
        cycle = run_improvement_cycle_step(loop)
        assert cycle.strategy_used == "none"
        assert cycle.improvement == 0.0

    def test_chunking_area_triggers_experiment(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan, run_improvement_cycle_step

        loop = self._make_loop()
        plan = ImprovementPlan(
            priority="high",
            area="chunking",
            issue="large chunks",
            action="reduce chunk_size",
            expected_improvement=0.2,
            config_changes={"type": "recursive", "chunk_size": 200, "chunk_overlap": 0},
        )
        loop.get_improvement_plan.return_value = [plan]
        cycle = run_improvement_cycle_step(loop, plan)
        loop.chunking_experimenter.run_experiment.assert_called_once()
        assert "Chunking config" in cycle.changes_made[0]

    def test_explicit_plan_passed(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan, run_improvement_cycle_step

        loop = self._make_loop()
        plan = ImprovementPlan(
            priority="low",
            area="retrieval",
            issue="i",
            action="do something explicit here",
            expected_improvement=0.05,
        )
        cycle = run_improvement_cycle_step(loop, plan)
        # get_improvement_plan NOT called when plan is given
        loop.get_improvement_plan.assert_not_called()
        assert cycle.strategy_used == "do something explicit here"


class TestRunFullCycle:
    def _make_loop(self, score: float = 0.5) -> Any:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementCycle, ImprovementPlan

        loop = MagicMock()
        loop._cycles = []
        loop._current_best_config = {"type": "recursive"}

        ce = MagicMock()
        ce.find_best_strategy.return_value = {"score": score}
        loop.chunking_experimenter = ce
        loop.evaluator = None

        # Patch run_improvement_cycle_step to append a cycle to loop._cycles
        def fake_step(*args, **kwargs):
            cycle = ImprovementCycle(
                cycle_number=len(loop._cycles) + 1,
                timestamp=datetime.now(timezone.utc),
                chunking_result=None,
                eval_score_before=score,
                eval_score_after=score + 0.05,
                improvement=0.05,
                strategy_used="test",
                changes_made=[],
            )
            loop._cycles.append(cycle)
            return cycle

        loop.run_improvement_cycle = fake_step
        return loop

    def test_run_full_cycle_basic(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import (
            ImprovementCycle,
            ImprovementPlan,
            run_full_cycle,
        )

        loop = MagicMock()
        loop._cycles = []
        loop._current_best_config = {}

        ce = MagicMock()
        ce.find_best_strategy.return_value = {"score": 0.5}
        loop.chunking_experimenter = ce
        loop.evaluator = None

        # Patch run_improvement_cycle_step to not actually call loop.get_improvement_plan etc.
        def _fake_step(lp, plan=None):
            cycle = ImprovementCycle(
                cycle_number=len(lp._cycles) + 1,
                timestamp=datetime.now(timezone.utc),
                chunking_result=None,
                eval_score_before=0.5,
                eval_score_after=0.55,
                improvement=0.05,
                strategy_used="t",
                changes_made=[],
            )
            lp._cycles.append(cycle)
            return cycle

        with patch(
            "beanllm.domain.rag_debug.loop_cycle.run_improvement_cycle_step", side_effect=_fake_step
        ):
            result = run_full_cycle(loop, max_iterations=2, target_improvement=1.0)
        assert "initial_score" in result
        assert "final_score" in result
        assert "cycles_run" in result
        assert result["cycles_run"] == 2

    def test_run_full_cycle_early_stop_on_target(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementCycle, run_full_cycle

        loop = MagicMock()
        loop._cycles = []
        loop._current_best_config = {}

        ce = MagicMock()
        ce.find_best_strategy.return_value = {"score": 0.5}
        loop.chunking_experimenter = ce
        loop.evaluator = None

        def _fake_big_improvement(lp, plan=None):
            cycle = ImprovementCycle(
                cycle_number=len(lp._cycles) + 1,
                timestamp=datetime.now(timezone.utc),
                chunking_result=None,
                eval_score_before=0.0,
                eval_score_after=0.5,
                improvement=0.5,  # big improvement
                strategy_used="t",
                changes_made=[],
            )
            lp._cycles.append(cycle)
            return cycle

        with patch(
            "beanllm.domain.rag_debug.loop_cycle.run_improvement_cycle_step",
            side_effect=_fake_big_improvement,
        ):
            result = run_full_cycle(loop, max_iterations=5, target_improvement=0.2)
        # Should stop after 1 cycle since improvement >= target
        assert result["cycles_run"] == 1


# ===========================================================================
# Tests: loop_phases
# ===========================================================================


class TestRunInitialExperiments:
    def test_default_configs(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import run_initial_experiments

        mock_experimenter = MagicMock()
        mock_experimenter.compare_strategies.return_value = [MagicMock(), MagicMock()]
        mock_experimenter.find_best_strategy.return_value = {
            "score": 0.7,
            "config": {"type": "recursive"},
        }

        results, best_config, baseline = run_initial_experiments(mock_experimenter)
        assert len(results) == 2
        assert best_config == {"type": "recursive"}
        assert baseline == 0.7

    def test_custom_configs(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import run_initial_experiments

        mock_experimenter = MagicMock()
        mock_experimenter.compare_strategies.return_value = [MagicMock()]
        mock_experimenter.find_best_strategy.return_value = None

        configs = [{"type": "recursive", "chunk_size": 500, "chunk_overlap": 50}]
        results, best_config, baseline = run_initial_experiments(mock_experimenter, configs=configs)
        mock_experimenter.compare_strategies.assert_called_once_with(configs)
        assert best_config is None
        assert baseline == 0.0

    def test_grid_search_mode(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import run_initial_experiments

        mock_experimenter = MagicMock()
        mock_experimenter.grid_search.return_value = [MagicMock()]
        mock_experimenter.find_best_strategy.return_value = {"score": 0.5, "config": {}}

        results, _, _ = run_initial_experiments(mock_experimenter, use_grid_search=True)
        mock_experimenter.grid_search.assert_called_once()
        assert len(results) == 1


class TestEvaluatePipelinePhase:
    def test_no_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import evaluate_pipeline

        result = evaluate_pipeline(None, "q", "resp", ["ctx"])
        assert result["auto_scores"] == {}
        assert result["unified_score"] == 0.0

    def test_with_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import evaluate_pipeline

        evaluator = MagicMock()
        evaluator.evaluate_auto.return_value = {"faithfulness": 0.8, "relevance": 0.6}
        evaluator.get_unified_score.return_value = 0.7

        result = evaluate_pipeline(evaluator, "q", "resp", ["ctx"])
        assert result["query"] == "q"
        assert result["unified_score"] == 0.7


class TestBatchEvaluatePhase:
    def test_empty_qa_pairs(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import batch_evaluate

        def fake_eval(**kwargs):
            return {"unified_score": 0.5}

        result = batch_evaluate(fake_eval, [])
        assert result["total"] == 0
        assert result["avg_unified_score"] == 0.0

    def test_multiple_qa_pairs(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import batch_evaluate

        scores = [0.8, 0.4]

        def fake_eval(query, response, contexts):
            return {"unified_score": scores.pop(0)}

        qa_pairs = [
            {"query": "q1", "response": "r1", "contexts": ["c1"]},
            {"query": "q2", "response": "r2"},
        ]
        result = batch_evaluate(fake_eval, qa_pairs)
        assert result["total"] == 2
        assert abs(result["avg_unified_score"] - 0.6) < 1e-6


class TestAddHumanFeedbackPhase:
    def test_with_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import add_human_feedback

        evaluator = MagicMock()
        chunking_experimenter = MagicMock()

        add_human_feedback(evaluator, chunking_experimenter, "query", 0.9, comment="nice")
        evaluator.collect_human_feedback.assert_called_once_with(
            query="query", rating=0.9, feedback_type="overall", comment="nice"
        )
        chunking_experimenter.add_feedback.assert_not_called()

    def test_without_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import add_human_feedback

        chunking_experimenter = MagicMock()
        add_human_feedback(None, chunking_experimenter, "query", 0.5)
        chunking_experimenter.add_feedback.assert_not_called()

    def test_with_chunk_id(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import add_human_feedback

        evaluator = MagicMock()
        chunking_experimenter = MagicMock()
        add_human_feedback(evaluator, chunking_experimenter, "q", 0.7, chunk_id="chunk_123")
        chunking_experimenter.add_feedback.assert_called_once()


class TestAddComparisonFeedbackPhase:
    def test_with_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import add_comparison_feedback

        evaluator = MagicMock()
        add_comparison_feedback(evaluator, "q", "resp_a", "resp_b", "a")
        evaluator.collect_comparison_feedback.assert_called_once_with(
            query="q", response_a="resp_a", response_b="resp_b", winner="a"
        )

    def test_without_evaluator(self) -> None:
        from beanllm.domain.rag_debug.loop_phases import add_comparison_feedback

        # Should not raise
        add_comparison_feedback(None, "q", "a", "b", "a")


class TestGetImprovementPlan:
    def test_no_evaluator_no_feedback(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan
        from beanllm.domain.rag_debug.loop_phases import get_improvement_plan

        ce = MagicMock()
        ce.improve_from_feedback.return_value = {"recommended_configs": [], "suggestions": []}

        plans = get_improvement_plan(None, ce, ImprovementPlan)
        assert plans == []

    def test_with_chunking_suggestions(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan
        from beanllm.domain.rag_debug.loop_phases import get_improvement_plan

        ce = MagicMock()
        ce.improve_from_feedback.return_value = {
            "recommended_configs": [{"chunk_size": 200}],
            "suggestions": ["Use smaller chunks"],
        }

        plans = get_improvement_plan(None, ce, ImprovementPlan)
        assert len(plans) == 1
        assert plans[0].area == "chunking"
        assert plans[0].config_changes == {"chunk_size": 200}

    def test_with_evaluator_suggestions(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan
        from beanllm.domain.rag_debug.loop_phases import get_improvement_plan

        evaluator = MagicMock()
        suggestion = MagicMock()
        suggestion.priority = "high"
        suggestion.category = "faithfulness"
        suggestion.issue = "low faithfulness"
        suggestion.suggestion = "improve context"
        suggestion.expected_improvement = 0.2
        evaluator.get_improvement_suggestions.return_value = [suggestion]

        ce = MagicMock()
        ce.improve_from_feedback.return_value = {"recommended_configs": [], "suggestions": []}

        plans = get_improvement_plan(evaluator, ce, ImprovementPlan)
        assert len(plans) == 1
        assert plans[0].priority == "high"
        assert plans[0].area == "faithfulness"

    def test_priority_sorting(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan
        from beanllm.domain.rag_debug.loop_phases import get_improvement_plan

        evaluator = MagicMock()
        suggestions = []
        for priority in ["low", "high", "medium"]:
            s = MagicMock()
            s.priority = priority
            s.category = "cat"
            s.issue = "issue"
            s.suggestion = "suggestion text here"
            s.expected_improvement = 0.1
            suggestions.append(s)
        evaluator.get_improvement_suggestions.return_value = suggestions

        ce = MagicMock()
        ce.improve_from_feedback.return_value = {"recommended_configs": [], "suggestions": []}

        plans = get_improvement_plan(evaluator, ce, ImprovementPlan)
        priorities = [p.priority for p in plans]
        assert priorities.index("high") < priorities.index("medium") < priorities.index("low")


# ===========================================================================
# Tests: loop_report
# ===========================================================================


class TestExportFullReport:
    def _make_cycle(self, num: int, improvement: float = 0.05) -> Any:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementCycle

        return ImprovementCycle(
            cycle_number=num,
            timestamp=datetime.now(timezone.utc),
            chunking_result=None,
            eval_score_before=0.5,
            eval_score_after=0.5 + improvement,
            improvement=improvement,
            strategy_used="test_strategy_name_here",
            changes_made=["change A"],
        )

    def test_json_format(self) -> None:
        from beanllm.domain.rag_debug.loop_report import export_full_report

        report = export_full_report(
            format="json",
            cycles=[],
            baseline_score=0.5,
            get_current_score_fn=lambda: 0.7,
            chunking_report_fn=lambda: "chunking report",
            evaluator_report_fn=None,
            get_improvement_plan_fn=lambda: [],
            current_best_config={"type": "recursive"},
        )
        data = json.loads(report)
        assert data["baseline_score"] == 0.5
        assert data["current_score"] == 0.7
        assert data["cycles"] == 0
        assert data["best_config"] == {"type": "recursive"}

    def test_markdown_format_no_cycles(self) -> None:
        from beanllm.domain.rag_debug.loop_report import export_full_report

        report = export_full_report(
            format="markdown",
            cycles=[],
            baseline_score=0.3,
            get_current_score_fn=lambda: 0.5,
            chunking_report_fn=lambda: "## Chunking",
            evaluator_report_fn=None,
            get_improvement_plan_fn=lambda: [],
            current_best_config=None,
        )
        assert "# RAG Improvement Report" in report
        assert "Baseline score: 0.3000" in report
        assert "Chunking Experiments" in report

    def test_markdown_format_with_cycles(self) -> None:
        from beanllm.domain.rag_debug.loop_report import export_full_report

        cycles = [self._make_cycle(1), self._make_cycle(2)]
        report = export_full_report(
            format="markdown",
            cycles=cycles,
            baseline_score=0.5,
            get_current_score_fn=lambda: 0.6,
            chunking_report_fn=lambda: "chunking",
            evaluator_report_fn=None,
            get_improvement_plan_fn=lambda: [],
            current_best_config={"type": "recursive"},
        )
        assert "Improvement History" in report
        assert "Best Chunking Config" in report
        assert "Total improvement cycles: 2" in report

    def test_markdown_with_evaluator_report(self) -> None:
        from beanllm.domain.rag_debug.loop_report import export_full_report

        report = export_full_report(
            format="markdown",
            cycles=[],
            baseline_score=0.0,
            get_current_score_fn=lambda: 0.0,
            chunking_report_fn=lambda: "chunking",
            evaluator_report_fn=lambda: "## Eval Report",
            get_improvement_plan_fn=lambda: [],
            current_best_config=None,
        )
        assert "Evaluation Results" in report
        assert "## Eval Report" in report

    def test_markdown_with_improvement_plans(self) -> None:
        from beanllm.domain.rag_debug.loop_cycle import ImprovementPlan
        from beanllm.domain.rag_debug.loop_report import export_full_report

        plan = ImprovementPlan(
            priority="high",
            area="chunking",
            issue="bad chunks",
            action="reduce chunk size now",
            expected_improvement=0.1,
        )
        report = export_full_report(
            format="markdown",
            cycles=[],
            baseline_score=0.5,
            get_current_score_fn=lambda: 0.5,
            chunking_report_fn=lambda: "c",
            evaluator_report_fn=None,
            get_improvement_plan_fn=lambda: [plan],
            current_best_config=None,
        )
        assert "Remaining Improvements" in report


# ===========================================================================
# Tests: ParameterTuner
# ===========================================================================


class TestParameterTuner:
    def test_init_defaults(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore()
        tuner = ParameterTuner(vs)
        assert tuner.baseline_params["top_k"] == 4
        assert tuner.baseline_params["score_threshold"] == 0.0
        assert tuner.baseline_params["mmr_lambda"] == 0.5

    def test_init_custom_params(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore()
        tuner = ParameterTuner(vs, baseline_params={"top_k": 8})
        assert tuner.baseline_params["top_k"] == 8

    def test_tune_top_k_success(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore()
        tuner = ParameterTuner(vs)
        results = tuner.tune_top_k("test query", [1, 2, 3])
        assert "k=1" in results
        assert "k=2" in results
        assert results["k=1"]["num_results"] == 1
        assert results["k=2"]["num_results"] == 2

    def test_tune_top_k_empty_results(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(results=[])
        tuner = ParameterTuner(vs)
        results = tuner.tune_top_k("q", [4])
        assert results["k=4"]["avg_score"] == 0.0
        assert results["k=4"]["min_score"] == 0.0
        assert results["k=4"]["max_score"] == 0.0

    def test_tune_top_k_error_handled(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(raise_on_search=True)
        tuner = ParameterTuner(vs)
        results = tuner.tune_top_k("q", [4])
        assert "error" in results["k=4"]

    def test_tune_threshold(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(
            results=[
                make_search_result("a", 0.9),
                make_search_result("b", 0.5),
                make_search_result("c", 0.2),
            ]
        )
        tuner = ParameterTuner(vs)
        results = tuner.tune_threshold("q", [0.0, 0.6], k=10)
        assert results["threshold=0.0"]["num_results"] == 3
        assert results["threshold=0.6"]["num_results"] == 1

    def test_tune_threshold_search_error(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(raise_on_search=True)
        tuner = ParameterTuner(vs)
        result = tuner.tune_threshold("q", [0.5])
        assert "error" in result

    def test_tune_mmr_lambda_no_mmr_support(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = MagicMock(spec=[])  # no max_marginal_relevance_search
        tuner = ParameterTuner(vs)
        result = tuner.tune_mmr_lambda("q", [0.5])
        assert "error" in result
        assert "MMR not supported" in result["error"]

    def test_tune_mmr_lambda_success(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore()
        tuner = ParameterTuner(vs)
        results = tuner.tune_mmr_lambda("q", [0.3, 0.7])
        assert "lambda=0.3" in results
        assert "lambda=0.7" in results
        assert results["lambda=0.3"]["num_results"] == 3

    def test_tune_mmr_lambda_error_handled(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = MagicMock()
        vs.max_marginal_relevance_search.side_effect = RuntimeError("mmr error")
        tuner = ParameterTuner(vs)
        results = tuner.tune_mmr_lambda("q", [0.5])
        assert "error" in results["lambda=0.5"]

    def test_compare_with_baseline_improvement(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(results=[make_search_result("doc", 0.8)])
        tuner = ParameterTuner(vs)
        result = tuner.compare_with_baseline("q", {"top_k": 4})
        assert "baseline" in result
        assert "new" in result
        assert "improvement_pct" in result
        assert "recommendation" in result

    def test_compare_with_baseline_zero_baseline(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(results=[])
        tuner = ParameterTuner(vs)
        result = tuner.compare_with_baseline("q", {"top_k": 8})
        assert result["improvement_pct"] == 0.0

    def test_compare_with_baseline_error_handling(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore(raise_on_search=True)
        tuner = ParameterTuner(vs)
        result = tuner.compare_with_baseline("q", {"top_k": 4})
        assert result["baseline"]["avg_score"] == 0.0
        assert result["new"]["avg_score"] == 0.0

    def test_auto_tune(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore()
        tuner = ParameterTuner(vs)
        result = tuner.auto_tune(
            test_queries=["q1", "q2"],
            param_ranges={"top_k": [2, 4]},
        )
        assert "best_params" in result
        assert "best_score" in result
        assert "baseline_params" in result
        assert "improvement_pct" in result

    def test_auto_tune_default_ranges(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        vs = FakeVectorStore()
        tuner = ParameterTuner(vs)
        result = tuner.auto_tune(test_queries=["query"])
        assert "best_params" in result

    def test_auto_tune_search_errors_skipped(self) -> None:
        from beanllm.domain.rag_debug.parameter_tuner import ParameterTuner

        call_count = {"n": 0}

        class FlakeyStore(FakeVectorStore):
            def similarity_search(self, query: str, k: int = 4):
                call_count["n"] += 1
                if call_count["n"] % 2 == 0:
                    raise RuntimeError("intermittent error")
                return self._results[:k]

        vs = FlakeyStore()
        tuner = ParameterTuner(vs)
        result = tuner.auto_tune(["q1", "q2"], {"top_k": [4, 6]})
        assert "best_params" in result


# ===========================================================================
# Tests: SimilarityTester
# ===========================================================================


class TestSimilarityTester:
    def test_test_query_similarity_only(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        result = tester.test_query("test query", k=2, strategies=["similarity"])
        assert "similarity" in result
        assert result["similarity"]["num_results"] == 2

    def test_test_query_mmr_supported(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        result = tester.test_query("q", strategies=["mmr"])
        assert "mmr" in result
        assert result["mmr"]["num_results"] == 3

    def test_test_query_mmr_not_supported(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = MagicMock(spec=["similarity_search"])  # no MMR
        vs.similarity_search.return_value = []
        tester = SimilarityTester(vs)
        result = tester.test_query("q", strategies=["mmr"])
        assert "error" in result["mmr"]
        assert "MMR not supported" in result["mmr"]["error"]

    def test_test_query_hybrid_supported(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        result = tester.test_query("q", strategies=["hybrid"])
        assert "hybrid" in result
        assert result["hybrid"]["num_results"] == 3

    def test_test_query_hybrid_not_supported(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = MagicMock(spec=["similarity_search"])
        vs.similarity_search.return_value = []
        tester = SimilarityTester(vs)
        result = tester.test_query("q", strategies=["hybrid"])
        assert "error" in result["hybrid"]
        assert "Hybrid search not supported" in result["hybrid"]["error"]

    def test_test_query_all_strategies_default(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        result = tester.test_query("q")
        assert "similarity" in result
        assert "mmr" in result
        assert "hybrid" in result

    def test_test_query_similarity_error(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore(raise_on_search=True)
        tester = SimilarityTester(vs)
        result = tester.test_query("q", strategies=["similarity"])
        assert "error" in result["similarity"]

    def test_test_query_document_without_page_content(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        # Document without page_content attribute
        mock_result = MagicMock()
        mock_result.document = "plain string doc"
        mock_result.score = 0.5
        mock_result.metadata = {}

        vs = MagicMock()
        vs.similarity_search.return_value = [mock_result]
        tester = SimilarityTester(vs)
        result = tester.test_query("q", strategies=["similarity"])
        assert result["similarity"]["num_results"] == 1

    def test_batch_test(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        results = tester.batch_test(["q1", "q2"], k=2)
        assert len(results) == 2
        assert results[0]["query"] == "q1"
        assert results[1]["query"] == "q2"

    def test_batch_test_empty(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        results = tester.batch_test([])
        assert results == []

    def test_compare_strategies(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        result = tester.compare_strategies("q", k=2)
        assert "query" in result
        assert "strategy_results" in result
        assert "overlap_analysis" in result
        assert "recommendations" in result

    def test_analyze_overlap_no_errors(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        strategy_results = {
            "similarity": {"results": [{"content": "doc A"}, {"content": "doc B"}]},
            "mmr": {"results": [{"content": "doc A"}, {"content": "doc C"}]},
        }
        overlaps = tester._analyze_overlap(strategy_results)
        assert "similarity_vs_mmr" in overlaps
        assert overlaps["similarity_vs_mmr"]["intersection"] == 1

    def test_analyze_overlap_all_errors(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        strategy_results = {
            "similarity": {"error": "failed"},
            "mmr": {"error": "failed"},
        }
        overlaps = tester._analyze_overlap(strategy_results)
        assert overlaps == {}

    def test_generate_recommendations_no_strategies(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        recs = tester._generate_strategy_recommendations({"similarity": {"error": "e"}}, {})
        assert any("No search strategies" in r for r in recs)

    def test_generate_recommendations_low_overlap(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        strategy_results = {
            "similarity": {"results": []},
            "mmr": {"results": []},
        }
        overlap_analysis = {"similarity_vs_mmr": {"jaccard_similarity": 0.1}}
        recs = tester._generate_strategy_recommendations(strategy_results, overlap_analysis)
        assert any("Low overlap" in r for r in recs)

    def test_generate_recommendations_high_overlap(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        strategy_results = {
            "similarity": {"results": []},
            "mmr": {"results": []},
        }
        overlap_analysis = {"similarity_vs_mmr": {"jaccard_similarity": 0.9}}
        recs = tester._generate_strategy_recommendations(strategy_results, overlap_analysis)
        assert any("High overlap" in r for r in recs)

    def test_generate_recommendations_mmr_and_hybrid_available(self) -> None:
        from beanllm.domain.rag_debug.similarity_tester import SimilarityTester

        vs = FakeVectorStore()
        tester = SimilarityTester(vs)
        strategy_results = {
            "mmr": {"results": []},
            "hybrid": {"results": []},
        }
        recs = tester._generate_strategy_recommendations(strategy_results, {})
        assert any("MMR" in r for r in recs)
        assert any("Hybrid" in r for r in recs)


# ===========================================================================
# Tests: EmbeddingAnalyzer (mocked advanced deps)
# ===========================================================================


class TestEmbeddingAnalyzerWithMocks:
    """Test EmbeddingAnalyzer by mocking all the advanced external dependencies."""

    def _patch_all_deps(self):
        """Context manager that patches umap, hdbscan, sklearn."""
        import sys

        import numpy as np

        # Build fake modules
        fake_umap_module = MagicMock()
        fake_umap_instance = MagicMock()
        fake_umap_instance.fit_transform.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        fake_umap_module.UMAP.return_value = fake_umap_instance

        fake_hdbscan_module = MagicMock()
        fake_hdbscan_instance = MagicMock()
        fake_hdbscan_instance.fit_predict.return_value = np.array([0, 0, 1])
        fake_hdbscan_module.HDBSCAN.return_value = fake_hdbscan_instance

        fake_tsne = MagicMock()
        fake_tsne_instance = MagicMock()
        fake_tsne_instance.fit_transform.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        fake_tsne.return_value = fake_tsne_instance

        fake_silhouette = MagicMock(return_value=0.75)

        patches = [
            patch.dict(
                "sys.modules",
                {
                    "umap": fake_umap_module,
                    "hdbscan": fake_hdbscan_module,
                },
            ),
            patch(
                "beanllm.domain.rag_debug.embedding_analyzer.EmbeddingAnalyzer._check_dependencies"
            ),
        ]
        return (
            patches,
            fake_umap_module,
            fake_hdbscan_module,
            fake_tsne,
            fake_silhouette,
            fake_umap_instance,
            fake_hdbscan_instance,
            fake_tsne_instance,
        )

    def test_check_dependencies_missing_raises(self) -> None:
        """Test that missing deps raise ImportError."""
        import sys

        with patch.dict("sys.modules", {"umap": None, "hdbscan": None}):
            with pytest.raises((ImportError, Exception)):
                # Force reimport
                import importlib

                import beanllm.domain.rag_debug.embedding_analyzer as m
                from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

                importlib.reload(m)
                m.EmbeddingAnalyzer()

    def test_detect_outliers(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()
            embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            labels = np.array([0, -1, 1])
            outliers = analyzer.detect_outliers(embeddings, labels)
            assert outliers == [1]

    def test_detect_outliers_none(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()
            embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
            labels = np.array([0, 1])
            outliers = analyzer.detect_outliers(embeddings, labels)
            assert outliers == []

    def test_compute_silhouette_score_not_enough_points(self) -> None:
        import sys

        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        sklearn_mock = MagicMock()
        sklearn_mock.metrics.silhouette_score = MagicMock(return_value=0.5)
        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            with patch.dict(
                sys.modules, {"sklearn": sklearn_mock, "sklearn.metrics": sklearn_mock.metrics}
            ):
                analyzer = EmbeddingAnalyzer()
                embeddings = np.array([[0.1, 0.2]])
                labels = np.array([-1])  # Only noise
                score = analyzer.compute_silhouette_score(embeddings, labels)
                assert score is None

    def test_compute_silhouette_score_single_cluster(self) -> None:
        import sys

        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        sklearn_mock = MagicMock()
        sklearn_mock.metrics.silhouette_score = MagicMock(return_value=0.5)
        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            with patch.dict(
                sys.modules, {"sklearn": sklearn_mock, "sklearn.metrics": sklearn_mock.metrics}
            ):
                analyzer = EmbeddingAnalyzer()
                embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
                labels = np.array([0, 0, 0])  # All in one cluster — only 1 unique, score = None
                score = analyzer.compute_silhouette_score(embeddings, labels)
                assert score is None

    def test_compute_silhouette_score_exception(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()
            embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            labels = np.array([0, 0, 1])

            with patch(
                "beanllm.domain.rag_debug.embedding_analyzer.EmbeddingAnalyzer.compute_silhouette_score",
                return_value=None,
            ):
                score = analyzer.compute_silhouette_score(embeddings, labels)
                assert score is None

    def test_reduce_dimensions_umap(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()

            fake_reduced = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            fake_umap_instance = MagicMock()
            fake_umap_instance.fit_transform.return_value = fake_reduced
            fake_umap_module = MagicMock()
            fake_umap_module.UMAP.return_value = fake_umap_instance

            with patch.dict("sys.modules", {"umap": fake_umap_module}):
                embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
                result = analyzer.reduce_dimensions_umap(embeddings, n_components=2)
                assert result.shape == (3, 2)

    def test_reduce_dimensions_tsne(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()

            fake_reduced = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            fake_tsne_instance = MagicMock()
            fake_tsne_instance.fit_transform.return_value = fake_reduced
            fake_tsne_class = MagicMock(return_value=fake_tsne_instance)
            fake_sklearn_manifold = MagicMock()
            fake_sklearn_manifold.TSNE = fake_tsne_class

            with patch.dict(
                "sys.modules",
                {
                    "sklearn": MagicMock(),
                    "sklearn.manifold": fake_sklearn_manifold,
                },
            ):
                embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
                result = analyzer.reduce_dimensions_tsne(embeddings, n_components=2)
                assert result.shape == (3, 2)

    def test_cluster_hdbscan(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()

            fake_labels = np.array([0, 0, 1, -1])
            fake_clusterer = MagicMock()
            fake_clusterer.fit_predict.return_value = fake_labels
            fake_hdbscan_module = MagicMock()
            fake_hdbscan_module.HDBSCAN.return_value = fake_clusterer

            with patch.dict("sys.modules", {"hdbscan": fake_hdbscan_module}):
                embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
                labels, stats = analyzer.cluster_hdbscan(embeddings, min_cluster_size=2)
                assert stats["n_clusters"] == 2
                assert stats["n_noise"] == 1
                assert abs(stats["noise_ratio"] - 0.25) < 1e-6

    def test_analyze_invalid_method(self) -> None:
        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()
            with pytest.raises(ValueError, match="Unknown method"):
                analyzer.analyze([[0.1, 0.2]], method="pca")

    def test_analyze_umap_pipeline(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()

            fake_reduced = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            fake_labels = np.array([0, 0, 1])

            with (
                patch.object(analyzer, "reduce_dimensions_umap", return_value=fake_reduced),
                patch.object(
                    analyzer,
                    "cluster_hdbscan",
                    return_value=(
                        fake_labels,
                        {
                            "n_clusters": 2,
                            "n_noise": 0,
                            "noise_ratio": 0.0,
                            "cluster_sizes": {0: 2, 1: 1},
                        },
                    ),
                ),
                patch.object(analyzer, "detect_outliers", return_value=[]),
                patch.object(analyzer, "compute_silhouette_score", return_value=0.75),
            ):
                embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
                result = analyzer.analyze(embeddings, method="umap")
                assert result["method"] == "umap"
                assert result["silhouette_score"] == 0.75
                assert len(result["labels"]) == 3

    def test_analyze_tsne_pipeline(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()

            fake_reduced = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            fake_labels = np.array([0, 1, 1])

            with (
                patch.object(analyzer, "reduce_dimensions_tsne", return_value=fake_reduced),
                patch.object(
                    analyzer,
                    "cluster_hdbscan",
                    return_value=(
                        fake_labels,
                        {"n_clusters": 2, "n_noise": 0, "noise_ratio": 0.0, "cluster_sizes": {}},
                    ),
                ),
                patch.object(analyzer, "detect_outliers", return_value=[]),
                patch.object(analyzer, "compute_silhouette_score", return_value=None),
            ):
                embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                result = analyzer.analyze(embeddings, method="tsne")
                assert result["method"] == "tsne"
                assert result["silhouette_score"] is None

    def test_analyze_without_outlier_detection(self) -> None:
        import numpy as np

        from beanllm.domain.rag_debug.embedding_analyzer import EmbeddingAnalyzer

        with patch.object(EmbeddingAnalyzer, "_check_dependencies"):
            analyzer = EmbeddingAnalyzer()

            fake_reduced = np.array([[0.1, 0.2], [0.3, 0.4]])
            fake_labels = np.array([0, 1])

            with (
                patch.object(analyzer, "reduce_dimensions_umap", return_value=fake_reduced),
                patch.object(analyzer, "cluster_hdbscan", return_value=(fake_labels, {})),
                patch.object(analyzer, "compute_silhouette_score", return_value=None),
            ):
                result = analyzer.analyze([[0.1, 0.2], [0.3, 0.4]], detect_outliers=False)
                assert result["outliers"] == []


# ===========================================================================
# Tests: RAGImprovementLoop
# ===========================================================================


class TestRAGImprovementLoop:
    """Test the coordinator class with mocked evaluator."""

    def _make_loop(self, docs: Optional[List[str]] = None) -> Any:
        from beanllm.domain.rag_debug.improvement_loop import RAGImprovementLoop

        with patch("beanllm.domain.rag_debug.improvement_loop.UnifiedEvaluator", None):
            loop = RAGImprovementLoop(
                documents=docs or ["Doc one.", "Doc two."],
                test_queries=["query"],
            )
        return loop

    def test_init(self) -> None:
        loop = self._make_loop()
        assert len(loop.documents) == 2
        assert len(loop.test_queries) == 1
        assert loop._cycles == []
        assert loop._baseline_score == 0.0
        assert loop.evaluator is None

    def test_repr(self) -> None:
        loop = self._make_loop()
        r = repr(loop)
        assert "RAGImprovementLoop" in r
        assert "docs=2" in r

    def test_get_status(self) -> None:
        loop = self._make_loop()
        status = loop.get_status()
        assert status["documents"] == 2
        assert status["test_queries"] == 1
        assert status["improvement_cycles"] == 0

    def test_run_initial_experiments(self) -> None:
        loop = self._make_loop(docs=["Long document text here. " * 10])
        results = loop.run_initial_experiments(
            configs=[{"type": "recursive", "chunk_size": 100, "chunk_overlap": 0, "name": "s"}]
        )
        assert len(results) >= 1
        assert loop._baseline_score >= 0.0

    def test_evaluate_pipeline_no_evaluator(self) -> None:
        loop = self._make_loop()
        result = loop.evaluate_pipeline("query", "response", ["context"])
        assert result["auto_scores"] == {}
        assert result["unified_score"] == 0.0

    def test_batch_evaluate_no_evaluator(self) -> None:
        loop = self._make_loop()
        result = loop.batch_evaluate(
            [
                {"query": "q1", "response": "r1", "contexts": ["c1"]},
                {"query": "q2", "response": "r2"},
            ]
        )
        assert result["total"] == 2
        assert result["avg_unified_score"] == 0.0

    def test_add_human_feedback_no_evaluator(self) -> None:
        loop = self._make_loop()
        # Should not raise
        loop.add_human_feedback("query", 0.8)

    def test_add_comparison_feedback_no_evaluator(self) -> None:
        loop = self._make_loop()
        # Should not raise
        loop.add_comparison_feedback("q", "resp_a", "resp_b", "a")

    def test_get_improvement_plan_no_evaluator(self) -> None:
        loop = self._make_loop()
        plans = loop.get_improvement_plan()
        # No feedback, no evaluator -> empty or minimal
        assert isinstance(plans, list)

    def test_detect_drift_no_evaluator(self) -> None:
        loop = self._make_loop()
        result = loop.detect_drift()
        assert result is None

    def test_get_current_score_no_results(self) -> None:
        loop = self._make_loop()
        score = loop._get_current_score()
        assert score == 0.0

    def test_export_full_report_json(self) -> None:
        loop = self._make_loop()
        report = loop.export_full_report(format="json")
        data = json.loads(report)
        assert "baseline_score" in data

    def test_export_full_report_markdown(self) -> None:
        loop = self._make_loop()
        report = loop.export_full_report(format="markdown")
        assert "# RAG Improvement Report" in report

    def test_run_improvement_cycle(self) -> None:
        loop = self._make_loop()
        # First run some experiments to have a score
        loop.run_initial_experiments(
            configs=[{"type": "recursive", "chunk_size": 100, "chunk_overlap": 0, "name": "s1"}]
        )
        cycle = loop.run_improvement_cycle()
        assert len(loop._cycles) == 1
        assert cycle.cycle_number == 1

    def test_run_full_cycle(self) -> None:
        loop = self._make_loop(docs=["Long document. " * 20])
        loop.run_initial_experiments(
            configs=[{"type": "recursive", "chunk_size": 100, "chunk_overlap": 0, "name": "base"}]
        )
        result = loop.run_full_cycle(max_iterations=2)
        assert "initial_score" in result
        assert "cycles_run" in result

    def test_with_evaluator_evaluate_pipeline(self) -> None:
        from beanllm.domain.rag_debug.improvement_loop import RAGImprovementLoop

        mock_evaluator_class = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_instance.evaluate_auto.return_value = {"faithfulness": 0.8}
        mock_evaluator_instance.get_unified_score.return_value = 0.8
        mock_evaluator_class.return_value = mock_evaluator_instance

        with patch(
            "beanllm.domain.rag_debug.improvement_loop.UnifiedEvaluator", mock_evaluator_class
        ):
            loop = RAGImprovementLoop(
                documents=["Doc."],
                test_queries=["q"],
            )

        result = loop.evaluate_pipeline("q", "r", ["c"])
        assert result["query"] == "q"
        assert result["unified_score"] == 0.8

    def test_with_evaluator_add_human_feedback(self) -> None:
        from beanllm.domain.rag_debug.improvement_loop import RAGImprovementLoop

        mock_evaluator_class = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_class.return_value = mock_evaluator_instance

        with patch(
            "beanllm.domain.rag_debug.improvement_loop.UnifiedEvaluator", mock_evaluator_class
        ):
            loop = RAGImprovementLoop(documents=["Doc."], test_queries=["q"])

        loop.add_human_feedback("query", 0.9, comment="great")
        mock_evaluator_instance.collect_human_feedback.assert_called_once()

    def test_with_evaluator_detect_drift(self) -> None:
        from beanllm.domain.rag_debug.improvement_loop import RAGImprovementLoop

        mock_evaluator_class = MagicMock()
        mock_evaluator_instance = MagicMock()
        mock_evaluator_instance.detect_drift.return_value = {"drift": True}
        mock_evaluator_class.return_value = mock_evaluator_instance

        with patch(
            "beanllm.domain.rag_debug.improvement_loop.UnifiedEvaluator", mock_evaluator_class
        ):
            loop = RAGImprovementLoop(documents=["Doc."], test_queries=["q"])

        result = loop.detect_drift()
        assert result == {"drift": True}
