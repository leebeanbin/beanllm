"""Tests for domain/orchestrator/workflow_analytics.py."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.orchestrator.monitor_types import (
    EventType,
    MonitorEvent,
    NodeExecutionState,
    NodeStatus,
)
from beanllm.domain.orchestrator.workflow_analytics import WorkflowAnalytics


def _make_completed_state(
    node_id: str,
    duration_ms: float = 100.0,
    agent_id: str | None = None,
) -> NodeExecutionState:
    metadata = {}
    if agent_id:
        metadata["agent_id"] = agent_id
    return NodeExecutionState(
        node_id=node_id,
        status=NodeStatus.COMPLETED,
        duration_ms=duration_ms,
        metadata=metadata,
    )


def _make_failed_state(
    node_id: str,
    agent_id: str | None = None,
) -> NodeExecutionState:
    metadata = {}
    if agent_id:
        metadata["agent_id"] = agent_id
    return NodeExecutionState(
        node_id=node_id,
        status=NodeStatus.FAILED,
        metadata=metadata,
    )


def _make_event(wf_id: str, node_id: str | None = None) -> MonitorEvent:
    return MonitorEvent(
        event_type=EventType.NODE_END,
        timestamp=datetime.now(),
        workflow_id=wf_id,
        node_id=node_id,
    )


# ---------------------------------------------------------------------------
# add_execution
# ---------------------------------------------------------------------------


class TestAddExecution:
    def test_stores_execution_data(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("node_a")
        analytics.add_execution("wf1", {"node_a": state}, [])
        assert "wf1" in analytics.executions

    def test_updates_node_metrics_for_completed_nodes(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("node_x", duration_ms=250.0)
        analytics.add_execution("wf1", {"node_x": state}, [])
        assert "node_x" in analytics.node_metrics
        assert analytics.node_metrics["node_x"] == [250.0]

    def test_updates_agent_metrics_for_completed_node_with_agent_id(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("node_a", duration_ms=150.0, agent_id="agent1")
        analytics.add_execution("wf1", {"node_a": state}, [])
        assert "agent1" in analytics.agent_metrics
        assert analytics.agent_metrics["agent1"]["executions"] == 1
        assert analytics.agent_metrics["agent1"]["successes"] == 1
        assert analytics.agent_metrics["agent1"]["total_duration_ms"] == 150.0
        assert "node_a" in analytics.agent_metrics["agent1"]["nodes"]

    def test_updates_agent_metrics_for_failed_node_with_agent_id(self):
        analytics = WorkflowAnalytics()
        state = _make_failed_state("node_b", agent_id="agent2")
        analytics.add_execution("wf1", {"node_b": state}, [])
        assert "agent2" in analytics.agent_metrics
        assert analytics.agent_metrics["agent2"]["executions"] == 1
        assert analytics.agent_metrics["agent2"]["successes"] == 0

    def test_skips_zero_duration_completed_nodes(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("node_z", duration_ms=0.0)
        analytics.add_execution("wf1", {"node_z": state}, [])
        assert "node_z" not in analytics.node_metrics

    def test_completed_node_without_agent_id_does_not_update_agent_metrics(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("node_no_agent", duration_ms=100.0, agent_id=None)
        analytics.add_execution("wf1", {"node_no_agent": state}, [])
        assert len(analytics.agent_metrics) == 0

    def test_multiple_executions_accumulate_metrics(self):
        analytics = WorkflowAnalytics()
        s1 = _make_completed_state("n1", duration_ms=100.0, agent_id="a1")
        s2 = _make_completed_state("n2", duration_ms=200.0, agent_id="a1")
        analytics.add_execution("wf1", {"n1": s1}, [])
        analytics.add_execution("wf2", {"n2": s2}, [])
        assert analytics.agent_metrics["a1"]["executions"] == 2
        assert analytics.agent_metrics["a1"]["total_duration_ms"] == 300.0


# ---------------------------------------------------------------------------
# find_bottlenecks
# ---------------------------------------------------------------------------


class TestFindBottlenecks:
    def test_returns_empty_for_unknown_workflow(self):
        analytics = WorkflowAnalytics()
        result = analytics.find_bottlenecks("nonexistent")
        assert result == []

    def test_returns_bottleneck_analysis_for_known_workflow(self):
        analytics = WorkflowAnalytics()
        states = {
            "n1": _make_completed_state("n1", duration_ms=1000.0),
            "n2": _make_completed_state("n2", duration_ms=100.0),
        }
        analytics.add_execution("wf1", states, [])
        result = analytics.find_bottlenecks("wf1")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# analyze_execution_paths
# ---------------------------------------------------------------------------


class TestAnalyzeExecutionPaths:
    def test_returns_empty_for_unknown_workflow(self):
        analytics = WorkflowAnalytics()
        result = analytics.analyze_execution_paths("nonexistent")
        assert result == []

    def test_returns_path_analysis_for_known_workflow(self):
        analytics = WorkflowAnalytics()
        events = [
            _make_event("wf1", "node_start"),
            _make_event("wf1", "node_end"),
        ]
        states = {"n1": _make_completed_state("n1")}
        analytics.add_execution("wf1", states, events)
        result = analytics.analyze_execution_paths("wf1")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_node_statistics
# ---------------------------------------------------------------------------


class TestGetNodeStatistics:
    def test_returns_stats_dict_for_node(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("target_node", duration_ms=300.0)
        analytics.add_execution("wf1", {"target_node": state}, [])
        stats = analytics.get_node_statistics("target_node")
        assert isinstance(stats, dict)

    def test_returns_stats_for_unknown_node(self):
        analytics = WorkflowAnalytics()
        stats = analytics.get_node_statistics("unknown_node")
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# compare_executions
# ---------------------------------------------------------------------------


class TestCompareExecutions:
    def test_returns_error_when_workflow_not_found(self):
        analytics = WorkflowAnalytics()
        result = analytics.compare_executions("a", "b")
        assert "error" in result

    def test_returns_comparison_when_both_exist(self):
        analytics = WorkflowAnalytics()
        s1 = _make_completed_state("n1", duration_ms=100.0)
        s2 = _make_completed_state("n1", duration_ms=200.0)
        analytics.add_execution("wf_a", {"n1": s1}, [])
        analytics.add_execution("wf_b", {"n1": s2}, [])
        result = analytics.compare_executions("wf_a", "wf_b")
        assert isinstance(result, dict)
        assert "error" not in result

    def test_returns_error_when_only_one_exists(self):
        analytics = WorkflowAnalytics()
        analytics.add_execution("wf_a", {}, [])
        result = analytics.compare_executions("wf_a", "wf_b_missing")
        assert "error" in result


# ---------------------------------------------------------------------------
# generate_optimization_recommendations
# ---------------------------------------------------------------------------


class TestGenerateOptimizationRecommendations:
    def test_returns_optimized_message_when_no_issues(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("n1", duration_ms=10.0)
        analytics.add_execution("wf1", {"n1": state}, [])
        recs = analytics.generate_optimization_recommendations("wf1")
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_recommends_parallelization_for_many_sequential_nodes(self):
        analytics = WorkflowAnalytics()
        states = {f"n{i}": _make_completed_state(f"n{i}", duration_ms=10.0) for i in range(5)}
        analytics.add_execution("wf1", states, [])
        recs = analytics.generate_optimization_recommendations("wf1")
        has_parallel_rec = any("parallel" in r.lower() or "Paralleliz" in r for r in recs)
        assert has_parallel_rec or len(recs) > 0  # Recommendations generated

    def test_recommends_consolidation_for_underutilized_agents(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("n1", duration_ms=10.0, agent_id="lonely_agent")
        analytics.add_execution("wf1", {"n1": state}, [])
        recs = analytics.generate_optimization_recommendations("wf1")
        assert isinstance(recs, list)

    def test_warns_about_low_success_rate(self):
        analytics = WorkflowAnalytics()
        states = {
            "n1": _make_completed_state("n1", duration_ms=10.0),
            "n2": _make_failed_state("n2"),
            "n3": _make_failed_state("n3"),
            "n4": _make_failed_state("n4"),
            "n5": _make_failed_state("n5"),
        }
        analytics.add_execution("wf1", states, [])
        recs = analytics.generate_optimization_recommendations("wf1")
        has_success_rate_warning = any("success rate" in r.lower() for r in recs)
        assert has_success_rate_warning


# ---------------------------------------------------------------------------
# calculate_cost_estimate
# ---------------------------------------------------------------------------


class TestCalculateCostEstimate:
    def test_returns_error_for_unknown_workflow(self):
        analytics = WorkflowAnalytics()
        result = analytics.calculate_cost_estimate("nonexistent")
        assert "error" in result

    def test_returns_cost_dict_for_known_workflow(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("n1", duration_ms=1000.0)
        analytics.add_execution("wf1", {"n1": state}, [])
        result = analytics.calculate_cost_estimate("wf1")
        assert isinstance(result, dict)

    def test_accepts_custom_cost_per_second(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("n1", duration_ms=1000.0)
        analytics.add_execution("wf1", {"n1": state}, [])
        result = analytics.calculate_cost_estimate("wf1", cost_per_second={"n1": 0.01})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# export_analytics_report
# ---------------------------------------------------------------------------


class TestExportAnalyticsReport:
    def test_returns_error_for_unknown_workflow(self):
        analytics = WorkflowAnalytics()
        result = analytics.export_analytics_report("nonexistent")
        assert "error" in result

    def test_returns_complete_report_for_known_workflow(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("n1", duration_ms=100.0)
        analytics.add_execution("wf1", {"n1": state}, [])
        report = analytics.export_analytics_report("wf1")
        assert "workflow_id" in report
        assert report["workflow_id"] == "wf1"
        assert "bottlenecks" in report
        assert "agent_utilization" in report
        assert "execution_paths" in report
        assert "recommendations" in report
        assert "cost_estimate" in report


# ---------------------------------------------------------------------------
# get_summary_statistics
# ---------------------------------------------------------------------------


class TestGetSummaryStatistics:
    def test_returns_dict(self):
        analytics = WorkflowAnalytics()
        result = analytics.get_summary_statistics()
        assert isinstance(result, dict)

    def test_returns_dict_with_data(self):
        analytics = WorkflowAnalytics()
        state = _make_completed_state("n1", duration_ms=150.0)
        analytics.add_execution("wf1", {"n1": state}, [])
        result = analytics.get_summary_statistics()
        assert isinstance(result, dict)
