"""Tests for REPL display helper modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ── optimizer_display ──────────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.optimizer_display import (
        show_ab_test_results,
        show_benchmark_results,
        show_compare_results,
        show_optimize_parameter_table,
        show_optimize_results,
        show_profile_results,
        show_recommendations_panel,
        show_recommendations_result,
        show_string_recommendations,
    )

    OPTIMIZER_DISPLAY_AVAILABLE = True
except ImportError:
    OPTIMIZER_DISPLAY_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZER_DISPLAY_AVAILABLE, reason="optimizer_display not available")
class TestOptimizerDisplay:
    def _console(self):
        return MagicMock()

    def _benchmark_result(self, **kwargs):
        r = MagicMock()
        r.benchmark_id = "bench-001"
        r.num_queries = 50
        r.avg_latency = 0.2
        r.p50_latency = 0.18
        r.p95_latency = 0.45
        r.p99_latency = 0.9
        r.throughput = 5.0
        r.avg_score = 0.85
        r.min_score = 0.6
        r.max_score = 0.98
        r.total_duration = 10.0
        r.queries = ["q1", "q2", "q3"]
        for k, v in kwargs.items():
            setattr(r, k, v)
        return r

    def _optimize_result(self, **kwargs):
        r = MagicMock()
        r.optimization_id = "opt-001"
        r.best_score = 0.92
        r.n_trials = 20
        r.best_params = {"top_k": 5, "chunk_size": 512}
        r.convergence_data = [{"trial": 1, "score": 0.7}, {"trial": 2, "score": 0.85}]
        for k, v in kwargs.items():
            setattr(r, k, v)
        return r

    def _profile_result(self, **kwargs):
        r = MagicMock()
        r.profile_id = "prof-001"
        r.total_duration_ms = 250.0
        r.total_tokens = 1500
        r.total_cost = 0.003
        r.bottleneck = "embedding"
        r.components = [
            {"name": "embedding", "duration_ms": 120, "tokens": 800, "cost": 0.001},
            {"name": "retrieval", "duration_ms": 80, "tokens": 0, "cost": 0.0},
        ]
        r.breakdown = {"embedding": 48.0, "retrieval": 32.0}
        r.recommendations = ["Reduce chunk size", "Use caching"]
        for k, v in kwargs.items():
            setattr(r, k, v)
        return r

    def _ab_result(self, **kwargs):
        r = MagicMock()
        r.winner = "B"
        r.lift = 9.0
        r.p_value = 0.02
        r.is_significant = True
        r.confidence_level = 0.95
        r.variant_a_mean = 0.75
        r.variant_b_mean = 0.82
        for k, v in kwargs.items():
            setattr(r, k, v)
        return r

    def test_show_string_recommendations(self):
        c = self._console()
        show_string_recommendations(c, ["tip1", "tip2"])
        c.print.assert_called()

    def test_show_string_recommendations_empty(self):
        c = self._console()
        show_string_recommendations(c, [])
        # no error expected

    def test_show_recommendations_panel_basic(self):
        c = self._console()
        recs = [
            {"title": "Use caching", "description": "Add Redis cache", "priority": "high"},
            {"title": "Reduce chunk", "description": "Smaller chunks", "priority": "medium"},
        ]
        show_recommendations_panel(c, recs)
        c.print.assert_called()

    def test_show_recommendations_panel_empty(self):
        c = self._console()
        show_recommendations_panel(c, [])
        c.print.assert_called()

    def test_show_benchmark_results_basic(self):
        c = self._console()
        viz = MagicMock()
        show_benchmark_results(c, self._benchmark_result(), viz)
        c.print.assert_called()

    def test_show_benchmark_results_with_queries(self):
        c = self._console()
        viz = MagicMock()
        result = self._benchmark_result(queries=list(f"q{i}" for i in range(15)))
        show_benchmark_results(c, result, viz, show_queries=True)
        c.print.assert_called()

    def test_show_benchmark_results_no_queries(self):
        c = self._console()
        viz = MagicMock()
        result = self._benchmark_result(queries=[])
        show_benchmark_results(c, result, viz, show_queries=True)
        c.print.assert_called()

    def test_show_optimize_parameter_table_integer(self):
        c = self._console()
        params = [
            {"name": "top_k", "type": "integer", "low": 1, "high": 20},
            {"name": "chunk_size", "type": "float", "low": 128.0, "high": 2048.0},
            {"name": "strategy", "type": "categorical", "categories": ["bm25", "vector"]},
            {"name": "use_reranker", "type": "boolean"},
        ]
        show_optimize_parameter_table(c, params)
        c.print.assert_called()

    def test_show_optimize_results_basic(self):
        c = self._console()
        show_optimize_results(c, self._optimize_result())
        c.print.assert_called()

    def test_show_optimize_results_no_convergence(self):
        c = self._console()
        result = self._optimize_result(convergence_data=None, best_params=None)
        show_optimize_results(c, result)
        c.print.assert_called()

    def test_show_profile_results_basic(self):
        c = self._console()
        viz = MagicMock()
        show_profile_results(c, self._profile_result(), viz)
        c.print.assert_called()

    def test_show_profile_results_no_components(self):
        c = self._console()
        viz = MagicMock()
        result = self._profile_result(components=[], breakdown=None, recommendations=None)
        show_profile_results(c, result, viz)
        c.print.assert_called()

    def test_show_ab_test_results_significant_b_wins(self):
        c = self._console()
        show_ab_test_results(c, self._ab_result(), "baseline", "new")
        c.print.assert_called()

    def test_show_ab_test_results_significant_a_wins(self):
        c = self._console()
        result = self._ab_result(winner="A", lift=-5.0)
        show_ab_test_results(c, result, "baseline", "new")
        c.print.assert_called()

    def test_show_ab_test_results_not_significant(self):
        c = self._console()
        result = self._ab_result(is_significant=False, winner="tie", lift=0.5)
        show_ab_test_results(c, result, "A", "B")
        c.print.assert_called()

    def test_show_recommendations_result_basic(self):
        c = self._console()
        result = MagicMock()
        result.recommendations = ["tip1", "tip2"]
        result.summary = {"critical": 1, "high": 2, "medium": 3, "low": 1}
        recs = [{"title": "tip1", "description": "desc", "priority": "high"}]
        show_recommendations_result(c, "prof-001", result, recs)
        c.print.assert_called()

    def test_show_compare_results_basic(self):
        c = self._console()
        result = {
            "summary": {"total_configs": 3, "found": 3},
            "configs": {
                "opt1": {"type": "optimization", "best_score": 0.9, "n_trials": 10},
                "prof1": {
                    "type": "profile",
                    "total_duration_ms": 200,
                    "total_cost": 0.002,
                    "bottleneck": "embed",
                },
                "ab1": {"type": "ab_test", "winner": "B", "lift": 5.0, "is_significant": True},
                "unknown1": {"type": "other", "error": "not found"},
            },
        }
        show_compare_results(c, result, ["opt1", "prof1", "ab1", "unknown1"])
        c.print.assert_called()


# ── kg_display ──────────────────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.kg_display import show_query_results, show_quick_commands

    KG_DISPLAY_AVAILABLE = True
except ImportError:
    KG_DISPLAY_AVAILABLE = False


@pytest.mark.skipif(not KG_DISPLAY_AVAILABLE, reason="kg_display not available")
class TestKGDisplay:
    def test_find_entities_by_type(self):
        c = MagicMock()
        results = [{"id": "e1", "name": "Alice", "type": "person"}] * 25
        show_query_results(c, results, "find_entities_by_type")
        c.print.assert_called()

    def test_find_entities_by_name(self):
        c = MagicMock()
        show_query_results(
            c, [{"id": "e1", "name": "Alice", "type": "person"}], "find_entities_by_name"
        )
        c.print.assert_called()

    def test_find_related_entities(self):
        c = MagicMock()
        results = [{"id": "e2", "name": "Bob", "type": "person", "relation_type": "knows"}] * 25
        show_query_results(c, results, "find_related_entities")
        c.print.assert_called()

    def test_find_shortest_path(self):
        c = MagicMock()
        show_query_results(c, [{"path": ["Alice", "knows", "Bob"]}], "find_shortest_path")
        c.print.assert_called()

    def test_get_entity_details(self):
        c = MagicMock()
        details = {
            "id": "e1",
            "name": "Alice",
            "type": "person",
            "outgoing_relations": [1, 2],
            "incoming_relations": [3],
        }
        show_query_results(c, [details], "get_entity_details")
        c.print.assert_called()

    def test_unknown_query_type(self):
        c = MagicMock()
        show_query_results(c, [{"foo": "bar"}], "unknown_query_type")
        c.print.assert_called()

    def test_show_quick_commands(self):
        c = MagicMock()
        show_quick_commands(c, "graph-123")
        c.print.assert_called()


# ── kg_stats ─────────────────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.kg_stats import render_stats_tables

    KG_STATS_AVAILABLE = True
except ImportError:
    KG_STATS_AVAILABLE = False


@pytest.mark.skipif(not KG_STATS_AVAILABLE, reason="kg_stats not available")
class TestKGStats:
    def test_basic_stats_only(self):
        c = MagicMock()
        viz = MagicMock()
        viz._create_bar.return_value = "██░░"
        stats = {
            "graph_id": "g-001",
            "num_nodes": 100,
            "num_edges": 200,
            "density": 0.02,
            "average_degree": 4.0,
            "num_connected_components": 3,
        }
        render_stats_tables(c, stats, viz, show_distributions=False)
        c.print.assert_called()

    def test_with_distributions(self):
        c = MagicMock()
        viz = MagicMock()
        viz._create_bar.return_value = "██░░"
        stats = {
            "graph_id": "g-001",
            "num_nodes": 100,
            "num_edges": 200,
            "density": 0.02,
            "average_degree": 4.0,
            "num_connected_components": 1,
            "entity_type_counts": {"person": 60, "org": 40},
            "relation_type_counts": {"works_for": 30, "located_in": 20},
        }
        render_stats_tables(c, stats, viz, show_distributions=True)
        c.print.assert_called()


# ── orchestrator_display ───────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.orchestrator_display import (
        display_bottlenecks,
        display_node_results,
        format_analytics,
        format_execution_result,
        format_workflow_info,
        status_badge,
    )

    ORCH_DISPLAY_AVAILABLE = True
except ImportError:
    ORCH_DISPLAY_AVAILABLE = False


@pytest.mark.skipif(not ORCH_DISPLAY_AVAILABLE, reason="orchestrator_display not available")
class TestOrchestratorDisplay:
    def test_status_badge_completed(self):
        assert "COMPLETED" in status_badge("completed")

    def test_status_badge_failed(self):
        assert "FAILED" in status_badge("failed")

    def test_status_badge_running(self):
        assert "RUNNING" in status_badge("running")

    def test_status_badge_pending(self):
        assert "PENDING" in status_badge("pending")

    def test_status_badge_unknown(self):
        result = status_badge("custom_status")
        assert isinstance(result, str)

    def test_format_workflow_info(self):
        wf = MagicMock()
        wf.workflow_id = "wf-001"
        wf.workflow_name = "test-workflow"
        wf.strategy = "sequential"
        wf.num_nodes = 3
        wf.num_edges = 2
        wf.created_at = "2026-01-01"
        wf.metadata = None
        result = format_workflow_info(wf)
        assert "wf-001" in result

    def test_format_workflow_info_with_metadata(self):
        wf = MagicMock()
        wf.workflow_id = "wf-002"
        wf.workflow_name = "meta-workflow"
        wf.strategy = "parallel"
        wf.num_nodes = 2
        wf.num_edges = 1
        wf.created_at = "2026-01-02"
        wf.metadata = {"key": "value"}
        result = format_workflow_info(wf)
        assert "wf-002" in result

    def test_format_execution_result(self):
        r = MagicMock()
        r.execution_id = "exec-001"
        r.workflow_id = "wf-001"
        r.status = "completed"
        r.execution_time = 1.5
        r.result = None
        r.error = None
        result = format_execution_result(r)
        assert "exec-001" in result

    def test_format_execution_result_with_error(self):
        r = MagicMock()
        r.execution_id = "exec-002"
        r.workflow_id = "wf-001"
        r.status = "failed"
        r.execution_time = 0.5
        r.result = None
        r.error = "Something went wrong"
        result = format_execution_result(r)
        assert "exec-002" in result

    def test_format_analytics(self):
        a = MagicMock()
        a.total_executions = 10
        a.avg_execution_time = 2.3
        a.success_rate = 0.9
        a.bottlenecks = [{"node": "slow_node"}]
        a.agent_utilization = {"agent1": 0.8}
        result = format_analytics(a)
        assert "10" in result

    def test_format_analytics_no_utilization(self):
        a = MagicMock()
        a.total_executions = 5
        a.avg_execution_time = 1.0
        a.success_rate = 1.0
        a.bottlenecks = []
        a.agent_utilization = None
        result = format_analytics(a)
        assert isinstance(result, str)

    def test_display_node_results(self):
        c = MagicMock()
        nodes = [
            {"node_id": "n1", "status": "completed", "duration_ms": 100, "output": "ok"},
            {"node_id": "n2", "status": "failed", "duration_ms": 50, "output": "error"},
        ]
        display_node_results(c, nodes)
        c.print.assert_called()

    def test_display_bottlenecks(self):
        c = MagicMock()
        bottlenecks = [
            {"node_id": "n1", "duration_ms": 500, "percentage": 42.0, "recommendation": "cache it"}
        ]
        display_bottlenecks(c, bottlenecks)
        c.print.assert_called()


# ── orchestrator_monitor ───────────────────────────────────────────────────────

try:
    from beanllm.ui.repl.orchestrator_monitor import create_monitor_display

    ORCH_MONITOR_AVAILABLE = True
except ImportError:
    ORCH_MONITOR_AVAILABLE = False


@pytest.mark.skipif(not ORCH_MONITOR_AVAILABLE, reason="orchestrator_monitor not available")
class TestOrchestratorMonitor:
    def test_no_status(self):
        panel = create_monitor_display(None)
        assert panel is not None

    def test_with_status(self):
        status = MagicMock()
        status.execution_id = "exec-001"
        status.current_node = "node1"
        status.progress = 0.5
        status.nodes_completed = ["n1", "n2"]
        status.nodes_pending = ["n3"]
        status.elapsed_time = 5.0
        panel = create_monitor_display(status)
        assert panel is not None

    def test_complete_progress(self):
        status = MagicMock()
        status.execution_id = "exec-002"
        status.current_node = None
        status.progress = 1.0
        status.nodes_completed = ["n1", "n2", "n3"]
        status.nodes_pending = []
        status.elapsed_time = 10.0
        panel = create_monitor_display(status)
        assert panel is not None
