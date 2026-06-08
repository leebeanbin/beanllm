"""
Comprehensive tests for UnifiedEvaluator.
Target: src/beanllm/domain/evaluation/unified_evaluator.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evaluator(**kwargs):
    """Create a UnifiedEvaluator with no persist_path (in-memory)."""
    from beanllm.domain.evaluation.unified_evaluator import UnifiedEvaluator

    return UnifiedEvaluator(**kwargs)


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestUnifiedEvaluatorInit:
    def test_default_init(self):
        ev = _make_evaluator()
        assert ev.auto_metrics == ["faithfulness", "relevance"]
        assert abs(ev.human_weight - 0.5) < 1e-9
        assert abs(ev.auto_weight - 0.5) < 1e-9
        assert ev.persist_path is None
        assert ev.llm_judge is None
        assert ev.embedding_function is None

    def test_custom_metrics(self):
        ev = _make_evaluator(auto_metrics=["faithfulness", "coherence", "completeness"])
        assert ev.auto_metrics == ["faithfulness", "coherence", "completeness"]

    def test_weight_normalisation(self):
        ev = _make_evaluator(human_weight=0.6, auto_weight=0.4)
        assert abs(ev.human_weight - 0.6) < 1e-9
        assert abs(ev.auto_weight - 0.4) < 1e-9

    def test_unequal_weight_normalisation(self):
        # 3:1 ratio -> human=0.75, auto=0.25
        ev = _make_evaluator(human_weight=3.0, auto_weight=1.0)
        assert abs(ev.human_weight - 0.75) < 1e-9
        assert abs(ev.auto_weight - 0.25) < 1e-9

    def test_zero_weights_no_divide(self):
        # Both zero — should not raise
        ev = _make_evaluator(human_weight=0.0, auto_weight=0.0)
        assert ev.human_weight == 0.0
        assert ev.auto_weight == 0.0

    def test_persist_path_created(self, tmp_path):
        persist = tmp_path / "eval_data"
        ev = _make_evaluator(persist_path=str(persist))
        assert persist.exists()
        assert ev.persist_path == persist

    def test_repr(self):
        ev = _make_evaluator()
        r = repr(ev)
        assert "UnifiedEvaluator" in r
        assert "records=" in r

    def test_drift_threshold_stored(self):
        ev = _make_evaluator(drift_threshold=0.3)
        assert ev.drift_threshold == 0.3

    def test_supported_metrics_dict(self):
        from beanllm.domain.evaluation.unified_evaluator import UnifiedEvaluator

        assert "faithfulness" in UnifiedEvaluator.SUPPORTED_METRICS
        assert "relevance" in UnifiedEvaluator.SUPPORTED_METRICS
        assert "coherence" in UnifiedEvaluator.SUPPORTED_METRICS

    def test_kwargs_stored(self):
        ev = _make_evaluator(custom_option=42)
        assert ev.kwargs.get("custom_option") == 42


# ---------------------------------------------------------------------------
# Auto-evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluateAuto:
    def test_evaluate_auto_returns_dict(self):
        ev = _make_evaluator(auto_metrics=["faithfulness", "relevance"])
        scores = ev.evaluate_auto(
            query="What is RAG?",
            response="RAG is Retrieval-Augmented Generation.",
            contexts=["RAG combines retrieval with generation."],
        )
        assert isinstance(scores, dict)
        assert "faithfulness" in scores
        assert "relevance" in scores
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_evaluate_auto_unknown_metric_skipped(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        scores = ev.evaluate_auto(
            query="Q",
            response="A",
            contexts=["C"],
            metrics=["faithfulness", "totally_unknown_metric"],
        )
        assert "faithfulness" in scores
        assert "totally_unknown_metric" not in scores

    def test_evaluate_auto_stores_record(self):
        ev = _make_evaluator()
        ev.evaluate_auto(
            query="Hello?",
            response="Hi!",
            contexts=["Context."],
        )
        record = ev.get_record("Hello?")
        assert record is not None
        assert record.query == "Hello?"

    def test_evaluate_auto_updates_avg_score(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        ev.evaluate_auto(
            query="Test",
            response="Test response",
            contexts=["Test context"],
        )
        record = ev.get_record("Test")
        assert record.auto_avg_score >= 0.0

    def test_evaluate_auto_with_llm_judge(self):
        mock_judge = MagicMock(return_value=0.9)
        ev = _make_evaluator(auto_metrics=["faithfulness"], llm_judge=mock_judge)
        scores = ev.evaluate_auto(
            query="Q",
            response="Good response",
            contexts=["Context data"],
        )
        assert "faithfulness" in scores

    def test_evaluate_auto_with_embedding_function(self):
        def mock_embed(text):
            return [0.1, 0.2, 0.3]

        ev = _make_evaluator(auto_metrics=["relevance"], embedding_function=mock_embed)
        scores = ev.evaluate_auto(
            query="What is Python?",
            response="Python is a language.",
            contexts=["Python context"],
        )
        assert "relevance" in scores

    def test_evaluate_auto_all_metrics(self):
        ev = _make_evaluator(
            auto_metrics=[
                "faithfulness",
                "relevance",
                "context_precision",
                "context_recall",
                "coherence",
                "completeness",
            ]
        )
        scores = ev.evaluate_auto(
            query="Why is Python popular?",
            response="Python is popular because of simplicity.",
            contexts=["Python simplicity drives its popularity."],
        )
        assert len(scores) == 6

    def test_evaluate_auto_empty_contexts(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        scores = ev.evaluate_auto(query="Q", response="A", contexts=[])
        assert "faithfulness" in scores


# ---------------------------------------------------------------------------
# Human feedback tests
# ---------------------------------------------------------------------------


class TestCollectHumanFeedback:
    def test_collect_basic_feedback(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q", rating=0.8)
        record = ev.get_record("Q")
        assert record is not None
        assert record.human_feedback_count == 1
        assert record.human_avg_rating == 0.8

    def test_collect_multiple_feedbacks(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q", rating=0.6)
        ev.collect_human_feedback(query="Q", rating=0.8)
        record = ev.get_record("Q")
        assert record.human_feedback_count == 2
        assert abs(record.human_avg_rating - 0.7) < 1e-9

    def test_feedback_with_comment(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q", rating=0.9, comment="Great answer")
        record = ev.get_record("Q")
        assert "Great answer" in record.human_comments

    def test_feedback_updates_unified_score(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q", rating=0.7)
        score = ev.get_unified_score("Q")
        assert score is not None
        assert score > 0

    def test_collect_comparison_feedback_a_wins(self):
        ev = _make_evaluator()
        ev.collect_comparison_feedback(
            query="Q",
            response_a="Good answer",
            response_b="Bad answer",
            winner="A",
        )
        # Should not raise; record should exist
        record = ev.get_record("Q")
        assert record is not None

    def test_collect_comparison_feedback_b_wins(self):
        ev = _make_evaluator()
        ev.collect_comparison_feedback(
            query="Q",
            response_a="A",
            response_b="B",
            winner="B",
        )
        record = ev.get_record("Q")
        assert record is not None

    def test_collect_comparison_feedback_tie(self):
        ev = _make_evaluator()
        ev.collect_comparison_feedback(
            query="Q",
            response_a="A",
            response_b="B",
            winner="TIE",
        )
        record = ev.get_record("Q")
        assert record is not None


# ---------------------------------------------------------------------------
# Unified score tests
# ---------------------------------------------------------------------------


class TestUnifiedScore:
    def test_get_unified_score_nonexistent(self):
        ev = _make_evaluator()
        assert ev.get_unified_score("nonexistent") is None

    def test_unified_score_only_human(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q", rating=0.8)
        score = ev.get_unified_score("Q")
        assert abs(score - 0.8) < 1e-9

    def test_unified_score_only_auto(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        ev.evaluate_auto(query="Q", response="A", contexts=["C"])
        record = ev.get_record("Q")
        score = ev.get_unified_score("Q")
        assert score == record.auto_avg_score

    def test_unified_score_combined(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"], human_weight=0.5, auto_weight=0.5)
        ev.evaluate_auto(query="Q", response="A", contexts=["C"])
        ev.collect_human_feedback(query="Q", rating=1.0)
        score = ev.get_unified_score("Q")
        # Should be weighted average; non-negative and <=1
        assert 0.0 <= score <= 1.0

    def test_unified_score_zero_when_no_data(self):
        ev = _make_evaluator()
        # Create record but no scores
        ev._get_or_create_record("Q")
        score = ev.get_unified_score("Q")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Summary and records tests
# ---------------------------------------------------------------------------


class TestEvaluationSummary:
    def test_empty_summary(self):
        ev = _make_evaluator()
        summary = ev.get_evaluation_summary()
        assert summary["total"] == 0 or "No evaluations yet" in summary.get("message", "")

    def test_summary_with_records(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        ev.evaluate_auto(query="Q1", response="A1", contexts=["C1"])
        ev.collect_human_feedback(query="Q1", rating=0.8)
        ev.evaluate_auto(query="Q2", response="A2", contexts=["C2"])
        summary = ev.get_evaluation_summary()
        assert summary["total_records"] == 2
        assert summary["total_human_feedbacks"] >= 1

    def test_summary_score_structure(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        ev.evaluate_auto(query="Q", response="A", contexts=["C"])
        summary = ev.get_evaluation_summary()
        assert "unified_score" in summary
        assert "avg" in summary["unified_score"]

    def test_get_all_records(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q1", rating=0.5)
        ev.collect_human_feedback(query="Q2", rating=0.7)
        records = ev.get_all_records()
        assert len(records) == 2

    def test_get_record_returns_correct(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="specific_query", rating=0.9)
        record = ev.get_record("specific_query")
        assert record.query == "specific_query"


# ---------------------------------------------------------------------------
# Improvement suggestions and drift detection
# ---------------------------------------------------------------------------


class TestImprovementAndDrift:
    def test_improvement_suggestions_empty(self):
        ev = _make_evaluator()
        suggestions = ev.get_improvement_suggestions()
        assert isinstance(suggestions, list)

    def test_improvement_suggestions_with_low_scores(self):
        ev = _make_evaluator(auto_metrics=["faithfulness", "relevance"])
        # Force a record with very low scores
        ev.evaluate_auto(
            query="Why?",
            response="I don't know.",
            contexts=["Totally unrelated context about something else."],
        )
        suggestions = ev.get_improvement_suggestions()
        assert isinstance(suggestions, list)

    def test_detect_drift_empty(self):
        ev = _make_evaluator()
        drift = ev.detect_drift()
        # With no records, should return None or a dict
        assert drift is None or isinstance(drift, dict)

    def test_detect_drift_with_records(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        for i in range(5):
            ev.evaluate_auto(query=f"Q{i}", response="Answer", contexts=["Context"])
        drift = ev.detect_drift()
        assert drift is None or isinstance(drift, dict)


# ---------------------------------------------------------------------------
# Export report tests
# ---------------------------------------------------------------------------


class TestExportReport:
    def test_export_markdown(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        ev.evaluate_auto(query="Q", response="A", contexts=["C"])
        report = ev.export_report(format="markdown")
        assert "RAG Evaluation Report" in report
        assert "Scores" in report

    def test_export_other_format(self):
        ev = _make_evaluator()
        ev.collect_human_feedback(query="Q", rating=0.5)
        report = ev.export_report(format="json")
        assert report is not None
        assert len(report) > 0

    def test_export_with_suggestions(self):
        ev = _make_evaluator(auto_metrics=["faithfulness", "relevance"])
        ev.evaluate_auto(
            query="What is the meaning of life?",
            response="42",
            contexts=[""],
        )
        report = ev.export_report(format="markdown")
        assert isinstance(report, str)

    def test_export_with_drift_detected(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        # Patch drift detector to return detected drift
        ev._drift_detector.detect_drift = MagicMock(
            return_value={"detected": True, "message": "Performance dropped"}
        )
        ev.evaluate_auto(query="Q", response="A", contexts=["C"])
        report = ev.export_report(format="markdown")
        assert "Drift Detected" in report or "Performance dropped" in report


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_history(self, tmp_path):
        persist = tmp_path / "eval"
        ev1 = _make_evaluator(persist_path=str(persist))
        ev1.collect_human_feedback(query="Hello?", rating=0.9, comment="Good")
        ev1._save_history()

        # Load into new evaluator
        ev2 = _make_evaluator(persist_path=str(persist))
        assert len(ev2.get_all_records()) == 1
        r = ev2.get_record("Hello?")
        assert r is not None
        assert r.human_feedback_count == 1

    def test_persist_creates_json_file(self, tmp_path):
        persist = tmp_path / "data"
        ev = _make_evaluator(persist_path=str(persist))
        ev.collect_human_feedback(query="Test", rating=0.5)
        history_file = persist / "eval_history.json"
        assert history_file.exists()
        with open(history_file) as f:
            data = json.load(f)
        assert "records" in data

    def test_load_missing_file_ok(self, tmp_path):
        persist = tmp_path / "empty"
        persist.mkdir()
        # No history file exists, should not raise
        ev = _make_evaluator(persist_path=str(persist))
        assert len(ev.get_all_records()) == 0

    def test_load_corrupt_file_handled(self, tmp_path):
        persist = tmp_path / "corrupt"
        persist.mkdir()
        (persist / "eval_history.json").write_text("not valid json {{{{")
        # Should not raise, just log error
        ev = _make_evaluator(persist_path=str(persist))
        assert len(ev.get_all_records()) == 0


# ---------------------------------------------------------------------------
# Record ID generation
# ---------------------------------------------------------------------------


class TestRecordIdGeneration:
    def test_same_query_same_id(self):
        ev = _make_evaluator()
        id1 = ev._generate_record_id("What is Python?")
        id2 = ev._generate_record_id("What is Python?")
        assert id1 == id2

    def test_different_query_different_id(self):
        ev = _make_evaluator()
        id1 = ev._generate_record_id("What is Python?")
        id2 = ev._generate_record_id("What is Java?")
        assert id1 != id2

    def test_id_length(self):
        ev = _make_evaluator()
        rid = ev._generate_record_id("Test query")
        assert len(rid) == 12


# ---------------------------------------------------------------------------
# Low-scoring queries
# ---------------------------------------------------------------------------


class TestLowScoringQueries:
    def test_low_scoring_in_summary(self):
        ev = _make_evaluator(auto_metrics=["faithfulness"])
        # Force a record with zero unified score
        record = ev._get_or_create_record("bad_query", "bad_response", [])
        record.unified_score = 0.1
        summary = ev.get_evaluation_summary()
        assert "bad_query" in summary.get("low_scoring_queries", [])
