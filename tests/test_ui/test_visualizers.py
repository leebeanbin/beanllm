"""
Tests for UI visualizer modules:
- MetricsVisualizer (all mixins)
- WorkflowViz
- EmbeddingViz
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from beanllm.ui.visualizers.metrics_viz import MetricsVisualizer


@pytest.fixture
def viz() -> MetricsVisualizer:
    console = MagicMock()
    return MetricsVisualizer(console=console)


# ── MetricsTablesMixin ────────────────────────────────────────────────────────


class TestShowSearchDashboard:
    def test_basic_metrics(self, viz):
        viz.show_search_dashboard(
            {"avg_score": 0.85, "avg_latency_ms": 120, "total_queries": 100, "top_k": 4}
        )
        viz.console.print.assert_called()

    def test_empty_metrics(self, viz):
        viz.show_search_dashboard({})
        viz.console.print.assert_called()

    def test_partial_metrics_score_only(self, viz):
        viz.show_search_dashboard({"avg_score": 0.9})
        viz.console.print.assert_called()

    def test_partial_metrics_latency_only(self, viz):
        viz.show_search_dashboard({"avg_latency_ms": 50})
        viz.console.print.assert_called()

    def test_custom_title(self, viz):
        viz.show_search_dashboard({"avg_score": 0.7}, title="My Dashboard")
        viz.console.print.assert_called()

    def test_high_score(self, viz):
        viz.show_search_dashboard({"avg_score": 0.99})
        viz.console.print.assert_called()

    def test_low_score(self, viz):
        viz.show_search_dashboard({"avg_score": 0.3})
        viz.console.print.assert_called()

    def test_high_latency(self, viz):
        viz.show_search_dashboard({"avg_latency_ms": 2000})
        viz.console.print.assert_called()

    def test_low_latency(self, viz):
        viz.show_search_dashboard({"avg_latency_ms": 10})
        viz.console.print.assert_called()


class TestGetScoreStatus:
    def test_excellent(self, viz):
        status = viz._get_score_status(0.95)
        assert isinstance(status, str)

    def test_good(self, viz):
        status = viz._get_score_status(0.75)
        assert isinstance(status, str)

    def test_poor(self, viz):
        status = viz._get_score_status(0.3)
        assert isinstance(status, str)


class TestGetLatencyStatus:
    def test_fast(self, viz):
        status = viz._get_latency_status(50)
        assert isinstance(status, str)

    def test_slow(self, viz):
        status = viz._get_latency_status(2000)
        assert isinstance(status, str)

    def test_medium(self, viz):
        status = viz._get_latency_status(300)
        assert isinstance(status, str)


class TestCompareParameters:
    def test_basic_comparison(self, viz):
        viz.compare_parameters(
            baseline={"top_k": 4, "score": 0.75},
            new={"top_k": 10, "score": 0.82},
        )
        viz.console.print.assert_called()

    def test_empty_dicts(self, viz):
        viz.compare_parameters(baseline={}, new={})
        viz.console.print.assert_called()

    def test_with_improvement(self, viz):
        viz.compare_parameters(
            baseline={"latency_ms": 200, "recall": 0.6},
            new={"latency_ms": 150, "recall": 0.8},
        )
        viz.console.print.assert_called()

    def test_with_regression(self, viz):
        viz.compare_parameters(
            baseline={"score": 0.9},
            new={"score": 0.5},
        )
        viz.console.print.assert_called()


class TestShowChunkStatistics:
    def test_basic(self, viz):
        viz.show_chunk_statistics(
            stats={"avg_size": 150, "total_chunks": 10, "min_size": 50, "max_size": 300}
        )
        viz.console.print.assert_called()

    def test_empty_chunks(self, viz):
        viz.show_chunk_statistics(stats={})
        viz.console.print.assert_called()


class TestShowTestResults:
    def test_basic(self, viz):
        viz.show_test_results(
            test_results=[
                {"query": "q1", "score": 0.9, "latency_ms": 100},
                {"query": "q2", "score": 0.7, "latency_ms": 200},
            ]
        )
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_test_results(test_results=[])
        viz.console.print.assert_called()


# ── MetricsSummaryMixin ───────────────────────────────────────────────────────


class TestShowRecommendations:
    def test_basic(self, viz):
        viz.show_recommendations(["Increase top_k", "Use BM25 hybrid"])
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_recommendations([])
        # empty list returns early without printing — no assertion on print

    def test_single(self, viz):
        viz.show_recommendations(["Only one tip"])
        viz.console.print.assert_called()


class TestShowProgressSummary:
    def test_basic(self, viz):
        viz.show_progress_summary(completed_steps=["step1", "step2"], total_steps=5)
        viz.console.print.assert_called()

    def test_target_met(self, viz):
        viz.show_progress_summary(completed_steps=["a", "b", "c"], total_steps=3)
        viz.console.print.assert_called()


class TestShowComparisonGrid:
    def test_basic(self, viz):
        viz.show_comparison_grid(
            strategies=["similarity", "mmr", "hybrid"],
            results={"q1": [0.8, 0.75, 0.82], "q2": [0.7, 0.65, 0.73]},
        )
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_comparison_grid(strategies=[], results={})
        viz.console.print.assert_called()


class TestShowErrorSummary:
    def test_basic(self, viz):
        viz.show_error_summary(
            errors=[{"type": "TimeoutError", "count": 3}, {"type": "ValueError", "count": 1}]
        )
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_error_summary(errors=[])
        viz.console.print.assert_called()


# ── OptimizerMetricsMixin ─────────────────────────────────────────────────────


class TestShowLatencyDistribution:
    def test_basic(self, viz):
        viz.show_latency_distribution(avg=0.2, p50=0.18, p95=0.45, p99=0.9)
        viz.console.print.assert_called()

    def test_all_equal(self, viz):
        viz.show_latency_distribution(avg=0.1, p50=0.1, p95=0.1, p99=0.1)
        viz.console.print.assert_called()


class TestShowComponentBreakdown:
    def test_basic(self, viz):
        viz.show_component_breakdown({"embedding": 40.0, "retrieval": 35.0, "generation": 25.0})
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_component_breakdown({})
        viz.console.print.assert_called()


class TestShowConvergence:
    def test_basic(self, viz):
        history = [{"trial": i, "score": 0.5 + i * 0.05, "params": {}} for i in range(5)]
        viz.show_convergence(history=history)
        viz.console.print.assert_called()

    def test_single(self, viz):
        viz.show_convergence(history=[{"trial": 1, "score": 0.8, "params": {}}])
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_convergence(history=[])
        viz.console.print.assert_called()


class TestShowParetoFrontier:
    def test_basic(self, viz):
        viz.show_pareto_frontier(
            pareto_solutions=[{"latency": 0.1, "quality": 0.7}, {"latency": 0.3, "quality": 0.9}],
            objectives=["latency", "quality"],
        )
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_pareto_frontier(pareto_solutions=[], objectives=["latency"])
        viz.console.print.assert_called()


class TestShowABComparison:
    def test_basic(self, viz):
        viz.show_ab_comparison(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.75,
            variant_b_mean=0.82,
            lift=0.09,
            is_significant=True,
        )
        viz.console.print.assert_called()

    def test_not_significant(self, viz):
        viz.show_ab_comparison(
            variant_a_name="A",
            variant_b_name="B",
            variant_a_mean=0.75,
            variant_b_mean=0.76,
            lift=0.01,
            is_significant=False,
        )
        viz.console.print.assert_called()


class TestShowPriorityDistribution:
    def test_basic(self, viz):
        viz.show_priority_distribution({"high": 3, "medium": 5, "low": 2})
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_priority_distribution({})
        viz.console.print.assert_called()


# ── GraphMetricsMixin ─────────────────────────────────────────────────────────


class TestShowGraphNetwork:
    def test_basic(self, viz):
        viz.show_graph_network(
            num_nodes=5, num_edges=3, density=0.6, num_components=1, avg_degree=1.2
        )
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_graph_network(
            num_nodes=0, num_edges=0, density=0.0, num_components=0, avg_degree=0.0
        )
        viz.console.print.assert_called()

    def test_large_graph(self, viz):
        viz.show_graph_network(
            num_nodes=1000, num_edges=5000, density=0.01, num_components=3, avg_degree=10.0
        )
        viz.console.print.assert_called()


class TestShowEntityDistribution:
    def test_basic(self, viz):
        viz.show_entity_distribution({"person": 5, "organization": 3, "location": 2})
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_entity_distribution({})
        viz.console.print.assert_called()


class TestShowRelationDistribution:
    def test_basic(self, viz):
        viz.show_relation_distribution({"works_for": 3, "located_in": 2})
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_relation_distribution({})
        viz.console.print.assert_called()


class TestShowPathVisualization:
    def test_basic(self, viz):
        viz.show_path_visualization(path=["entity1", "entity2", "entity3"])
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_path_visualization(path=[])
        viz.console.print.assert_called()

    def test_with_names(self, viz):
        viz.show_path_visualization(
            path=["e1", "e2"],
            entity_names={"e1": "Alice", "e2": "Apple"},
            relation_types=["works_for"],
        )
        viz.console.print.assert_called()


# ── MetricsChartsMixin ────────────────────────────────────────────────────────


class TestShowSizeDistribution:
    def test_basic(self, viz):
        viz.show_size_distribution({"0-500": 10, "500-1000": 50, "1000+": 5})
        viz.console.print.assert_called()

    def test_empty(self, viz):
        viz.show_size_distribution({})
        viz.console.print.assert_called()
