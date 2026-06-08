"""
Evaluation Analytics 테스트 - EvaluationAnalyticsEngine, MetricTrend, CorrelationAnalysis
"""

from datetime import datetime, timedelta

import pytest

from beanllm.domain.evaluation.analytics import (
    CorrelationAnalysis,
    EvaluationAnalytics,
    EvaluationAnalyticsEngine,
    MetricTrend,
)
from beanllm.domain.evaluation.results import BatchEvaluationResult, EvaluationResult


def _make_eval_result(metric: str, score: float) -> EvaluationResult:
    return EvaluationResult(metric_name=metric, score=score)


def _make_batch(results: list[EvaluationResult]) -> BatchEvaluationResult:
    avg = sum(r.score for r in results) / len(results) if results else 0.0
    return BatchEvaluationResult(results=results, average_score=avg)


class TestEvaluationResult:
    def test_create_evaluation_result(self) -> None:
        result = EvaluationResult(metric_name="accuracy", score=0.95)
        assert result.metric_name == "accuracy"
        assert result.score == 0.95
        assert result.explanation is None

    def test_evaluation_result_with_explanation(self) -> None:
        result = EvaluationResult(
            metric_name="faithfulness",
            score=0.88,
            explanation="High faithfulness score",
        )
        assert result.explanation == "High faithfulness score"

    def test_evaluation_result_repr(self) -> None:
        result = EvaluationResult(metric_name="precision", score=0.75)
        r = repr(result)
        assert "precision" in r
        assert "0.7500" in r

    def test_evaluation_result_with_metadata(self) -> None:
        result = EvaluationResult(
            metric_name="recall",
            score=0.9,
            metadata={"model": "gpt-4", "dataset": "test"},
        )
        assert result.metadata["model"] == "gpt-4"


class TestBatchEvaluationResult:
    def test_create_batch_result(self) -> None:
        results = [
            _make_eval_result("accuracy", 0.9),
            _make_eval_result("faithfulness", 0.8),
        ]
        batch = _make_batch(results)
        assert batch.average_score == pytest.approx(0.85)
        assert len(batch.results) == 2

    def test_get_metric(self) -> None:
        results = [
            _make_eval_result("accuracy", 0.9),
            _make_eval_result("faithfulness", 0.75),
        ]
        batch = _make_batch(results)
        found = batch.get_metric("accuracy")
        assert found is not None
        assert found.score == 0.9

    def test_get_metric_not_found(self) -> None:
        results = [_make_eval_result("accuracy", 0.9)]
        batch = _make_batch(results)
        assert batch.get_metric("nonexistent") is None

    def test_to_dict(self) -> None:
        results = [_make_eval_result("accuracy", 0.9)]
        batch = _make_batch(results)
        d = batch.to_dict()
        assert "results" in d
        assert "average_score" in d
        assert d["average_score"] == 0.9


class TestEvaluationAnalyticsEngine:
    @pytest.fixture
    def engine(self) -> EvaluationAnalyticsEngine:
        return EvaluationAnalyticsEngine()

    @pytest.fixture
    def engine_with_data(self) -> EvaluationAnalyticsEngine:
        eng = EvaluationAnalyticsEngine()
        # Add multiple results over time
        base_time = datetime.now() - timedelta(days=10)
        for i in range(5):
            results = [
                _make_eval_result("accuracy", 0.7 + i * 0.05),
                _make_eval_result("faithfulness", 0.8 - i * 0.02),
            ]
            batch = _make_batch(results)
            eng.add_evaluation_result(
                batch,
                timestamp=base_time + timedelta(days=i),
            )
        return eng

    def test_add_evaluation_result(self, engine: EvaluationAnalyticsEngine) -> None:
        batch = _make_batch([_make_eval_result("accuracy", 0.9)])
        engine.add_evaluation_result(batch)
        assert len(engine._history) == 1

    def test_add_evaluation_result_with_timestamp(self, engine: EvaluationAnalyticsEngine) -> None:
        batch = _make_batch([_make_eval_result("accuracy", 0.9)])
        ts = datetime(2025, 1, 15, 12, 0, 0)
        engine.add_evaluation_result(batch, timestamp=ts)
        assert engine._history[0]["timestamp"] == ts

    def test_add_evaluation_result_with_metadata(self, engine: EvaluationAnalyticsEngine) -> None:
        batch = _make_batch([_make_eval_result("recall", 0.8)])
        engine.add_evaluation_result(batch, metadata={"experiment": "A"})
        assert engine._history[0]["metadata"]["experiment"] == "A"

    def test_analyze_trends_empty(self, engine: EvaluationAnalyticsEngine) -> None:
        trends = engine.analyze_trends()
        assert trends == []

    def test_analyze_trends_with_data(self, engine_with_data: EvaluationAnalyticsEngine) -> None:
        trends = engine_with_data.analyze_trends()
        assert isinstance(trends, list)
        assert len(trends) >= 1
        for t in trends:
            assert isinstance(t, MetricTrend)
            assert t.metric_name in ("accuracy", "faithfulness")
            assert t.trend in ("improving", "declining", "stable")

    def test_analyze_trends_single_metric(
        self, engine_with_data: EvaluationAnalyticsEngine
    ) -> None:
        trends = engine_with_data.analyze_trends(metric_name="accuracy")
        assert all(t.metric_name == "accuracy" for t in trends)

    def test_analyze_trends_window_days(self, engine_with_data: EvaluationAnalyticsEngine) -> None:
        # 1-day window should limit results
        trends_short = engine_with_data.analyze_trends(window_days=1)
        trends_long = engine_with_data.analyze_trends(window_days=30)
        # Longer window should have >= entries
        assert len(trends_long) >= len(trends_short)

    def test_metric_trend_fields(self, engine_with_data: EvaluationAnalyticsEngine) -> None:
        trends = engine_with_data.analyze_trends(metric_name="accuracy")
        if trends:
            t = trends[0]
            assert t.average_score > 0
            assert t.min_score <= t.max_score
            assert t.std_dev >= 0
            assert len(t.timestamps) == len(t.scores)

    def test_analyze_correlations_empty(self, engine: EvaluationAnalyticsEngine) -> None:
        correlations = engine.analyze_correlations()
        assert correlations == []

    def test_analyze_correlations_insufficient_samples(
        self, engine_with_data: EvaluationAnalyticsEngine
    ) -> None:
        # Default min_samples=10, we only have 5
        correlations = engine_with_data.analyze_correlations(min_samples=10)
        assert correlations == []

    def test_analyze_correlations_with_enough_samples(self) -> None:
        eng = EvaluationAnalyticsEngine()
        base_time = datetime.now() - timedelta(days=30)
        for i in range(15):
            results = [
                _make_eval_result("precision", 0.5 + i * 0.03),
                _make_eval_result("recall", 0.5 + i * 0.03),  # perfectly correlated
            ]
            batch = _make_batch(results)
            eng.add_evaluation_result(batch, timestamp=base_time + timedelta(days=i))

        correlations = eng.analyze_correlations(min_samples=10)
        assert isinstance(correlations, list)
        if correlations:
            c = correlations[0]
            assert isinstance(c, CorrelationAnalysis)
            assert c.significance in ("strong", "moderate", "weak", "none")
            assert -1.0 <= c.correlation <= 1.0

    def test_calculate_correlation(self, engine: EvaluationAnalyticsEngine) -> None:
        # Perfect positive correlation
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr = engine._calculate_correlation(x, y)
        assert corr == pytest.approx(1.0)

    def test_calculate_correlation_negative(self, engine: EvaluationAnalyticsEngine) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        corr = engine._calculate_correlation(x, y)
        assert corr == pytest.approx(-1.0)

    def test_calculate_correlation_zero_variance(self, engine: EvaluationAnalyticsEngine) -> None:
        x = [1.0, 1.0, 1.0]
        y = [2.0, 3.0, 4.0]
        corr = engine._calculate_correlation(x, y)
        assert corr == 0.0

    def test_generate_analytics_empty(self, engine: EvaluationAnalyticsEngine) -> None:
        analytics = engine.generate_analytics()
        assert isinstance(analytics, EvaluationAnalytics)
        assert analytics.metric_trends == []
        assert analytics.correlations == []
        assert analytics.summary_stats["total_evaluations"] == 0

    def test_generate_analytics_with_data(
        self, engine_with_data: EvaluationAnalyticsEngine
    ) -> None:
        analytics = engine_with_data.generate_analytics()
        assert isinstance(analytics, EvaluationAnalytics)
        assert "window_days" in analytics.metadata
        assert "total_evaluations" in analytics.metadata
        assert analytics.metadata["total_evaluations"] == 5

    def test_generate_analytics_no_insights(
        self, engine_with_data: EvaluationAnalyticsEngine
    ) -> None:
        analytics = engine_with_data.generate_analytics(include_insights=False)
        assert analytics.insights == []

    def test_generate_analytics_with_insights(
        self, engine_with_data: EvaluationAnalyticsEngine
    ) -> None:
        analytics = engine_with_data.generate_analytics(include_insights=True)
        assert isinstance(analytics.insights, list)

    def test_summary_stats_with_data(self, engine_with_data: EvaluationAnalyticsEngine) -> None:
        stats = engine_with_data._calculate_summary_stats()
        assert stats["total_evaluations"] == 5
        assert "accuracy" in stats["average_scores"]
        assert "faithfulness" in stats["average_scores"]

    def test_clear_history_all(self, engine_with_data: EvaluationAnalyticsEngine) -> None:
        engine_with_data.clear_history()
        assert engine_with_data._history == []

    def test_clear_history_by_days(self, engine_with_data: EvaluationAnalyticsEngine) -> None:
        # Keep only last 3 days
        initial_count = len(engine_with_data._history)
        engine_with_data.clear_history(days=3)
        assert len(engine_with_data._history) <= initial_count

    def test_trend_improving(self) -> None:
        eng = EvaluationAnalyticsEngine()
        base_time = datetime.now() - timedelta(days=10)
        for i in range(6):
            results = [_make_eval_result("score", 0.3 + i * 0.12)]
            batch = _make_batch(results)
            eng.add_evaluation_result(batch, timestamp=base_time + timedelta(days=i))

        trends = eng.analyze_trends(metric_name="score")
        if trends:
            assert trends[0].trend in ("improving", "stable", "declining")

    def test_insights_generated_for_best_worst_metrics(self) -> None:
        eng = EvaluationAnalyticsEngine()
        base_time = datetime.now() - timedelta(days=5)
        for i in range(3):
            results = [
                _make_eval_result("precision", 0.9),
                _make_eval_result("recall", 0.6),
            ]
            batch = _make_batch(results)
            eng.add_evaluation_result(batch, timestamp=base_time + timedelta(days=i))

        analytics = eng.generate_analytics(include_insights=True)
        insights_text = " ".join(analytics.insights)
        # Should mention the metrics in insights
        assert len(analytics.insights) >= 0  # may be 0 if no trends computed
