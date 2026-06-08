"""
Tests for beanllm.ui.visualizers.workflow_viz
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from beanllm.ui.visualizers.workflow_viz import (
        WorkflowVisualizer,
        show_execution_progress,
        show_workflow_analytics,
        show_workflow_diagram,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.visualizers.workflow_viz not available")
class TestWorkflowVisualizer:
    """Tests for WorkflowVisualizer class."""

    def setup_method(self):
        self.console = MagicMock()
        self.viz = WorkflowVisualizer(console=self.console)

    # ------------------------------------------------------------------ #
    # show_diagram
    # ------------------------------------------------------------------ #

    def test_show_diagram_basic(self):
        self.viz.show_diagram("A -> B -> C")
        self.console.print.assert_called_once()

    def test_show_diagram_custom_title(self):
        self.viz.show_diagram("A -> B", title="My Workflow")
        self.console.print.assert_called_once()

    def test_show_diagram_custom_border(self):
        self.viz.show_diagram("A -> B", border_style="red")
        self.console.print.assert_called_once()

    def test_show_diagram_empty_string(self):
        self.viz.show_diagram("")
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_progress
    # ------------------------------------------------------------------ #

    def test_show_progress_basic(self):
        self.viz.show_progress(
            workflow_id="wf-001",
            total_nodes=5,
            nodes_completed=["n1", "n2"],
            nodes_running=["n3"],
            nodes_pending=["n4", "n5"],
        )
        self.console.print.assert_called_once()

    def test_show_progress_with_failed_nodes(self):
        self.viz.show_progress(
            workflow_id="wf-002",
            total_nodes=5,
            nodes_completed=["n1"],
            nodes_running=[],
            nodes_pending=["n4"],
            nodes_failed=["n2", "n3"],
        )
        self.console.print.assert_called_once()

    def test_show_progress_empty_lists(self):
        self.viz.show_progress(
            workflow_id="wf-003",
            total_nodes=0,
            nodes_completed=[],
            nodes_running=[],
            nodes_pending=[],
        )
        self.console.print.assert_called_once()

    def test_show_progress_many_completed_nodes(self):
        """More than 3 completed nodes triggers the '+N more' branch."""
        self.viz.show_progress(
            workflow_id="wf-004",
            total_nodes=10,
            nodes_completed=["n1", "n2", "n3", "n4", "n5"],
            nodes_running=["n6"],
            nodes_pending=["n7", "n8", "n9", "n10"],
            elapsed_time=12.5,
        )
        self.console.print.assert_called_once()

    def test_show_progress_many_pending_nodes(self):
        """More than 3 pending nodes triggers the '+N more' branch."""
        self.viz.show_progress(
            workflow_id="wf-005",
            total_nodes=8,
            nodes_completed=[],
            nodes_running=[],
            nodes_pending=["n1", "n2", "n3", "n4", "n5"],
        )
        self.console.print.assert_called_once()

    def test_show_progress_zero_total_nodes(self):
        """Edge case: total_nodes=0 should not raise ZeroDivisionError."""
        self.viz.show_progress(
            workflow_id="wf-006",
            total_nodes=0,
            nodes_completed=[],
            nodes_running=[],
            nodes_pending=[],
        )
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_node_states
    # ------------------------------------------------------------------ #

    def test_show_node_states_basic(self):
        node_states = {
            "node1": {"status": "completed", "duration_ms": 150.0},
            "node2": {"status": "running"},
        }
        self.viz.show_node_states(node_states)
        self.console.print.assert_called_once()

    def test_show_node_states_all_status_types(self):
        node_states = {
            "n_completed": {"status": "completed"},
            "n_failed": {"status": "failed", "error": "Timeout"},
            "n_running": {"status": "running"},
            "n_pending": {"status": "pending"},
            "n_skipped": {"status": "skipped"},
            "n_unknown": {"status": "unknown_status"},
        }
        self.viz.show_node_states(node_states)
        self.console.print.assert_called_once()

    def test_show_node_states_with_all_fields(self):
        node_states = {
            "node1": {
                "status": "completed",
                "start_time": "2024-01-01 10:00:00",
                "end_time": "2024-01-01 10:00:05",
                "duration_ms": 5000.0,
                "error": None,
                "output": {"result": "some output data"},
            }
        }
        self.viz.show_node_states(node_states)
        self.console.print.assert_called_once()

    def test_show_node_states_with_error(self):
        node_states = {
            "node_err": {
                "status": "failed",
                "error": "Connection refused",
            }
        }
        self.viz.show_node_states(node_states)
        self.console.print.assert_called_once()

    def test_show_node_states_with_output(self):
        node_states = {
            "node_out": {
                "status": "completed",
                "output": "A" * 60,  # longer than 50 chars to trigger truncation
            }
        }
        self.viz.show_node_states(node_states)
        self.console.print.assert_called_once()

    def test_show_node_states_empty(self):
        self.viz.show_node_states({})
        self.console.print.assert_called_once()

    def test_show_node_states_custom_title(self):
        self.viz.show_node_states({"node1": {"status": "running"}}, title="Custom Title")
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_execution_timeline
    # ------------------------------------------------------------------ #

    def test_show_execution_timeline_basic(self):
        events = [
            {"timestamp": "10:00:00", "event_type": "workflow_start", "node_id": "N/A", "data": {}},
            {
                "timestamp": "10:00:01",
                "event_type": "node_start",
                "node_id": "node1",
                "data": {"x": 1},
            },
            {"timestamp": "10:00:02", "event_type": "node_end", "node_id": "node1", "data": {}},
        ]
        self.viz.show_execution_timeline(events)
        self.console.print.assert_called_once()

    def test_show_execution_timeline_empty(self):
        self.viz.show_execution_timeline([])
        self.console.print.assert_called_once()

    def test_show_execution_timeline_many_events_truncated(self):
        """More than max_events events triggers slice to last N."""
        events = [
            {
                "timestamp": f"10:00:{i:02d}",
                "event_type": "node_start",
                "node_id": f"n{i}",
                "data": {},
            }
            for i in range(30)
        ]
        self.viz.show_execution_timeline(events, max_events=10)
        self.console.print.assert_called_once()

    def test_show_execution_timeline_all_event_types(self):
        event_types = [
            "workflow_start",
            "workflow_end",
            "node_start",
            "node_end",
            "node_error",
            "edge_traversed",
            "state_changed",
            "custom_event",
        ]
        events = [
            {"timestamp": "T", "event_type": et, "node_id": "n1", "data": {"k": "v"}}
            for et in event_types
        ]
        self.viz.show_execution_timeline(events)
        self.console.print.assert_called_once()

    def test_show_execution_timeline_missing_keys(self):
        """Events with missing keys should not raise."""
        events = [{}]
        self.viz.show_execution_timeline(events)
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_bottlenecks
    # ------------------------------------------------------------------ #

    def test_show_bottlenecks_basic(self):
        bottlenecks = [
            {
                "node_id": "node1",
                "duration_ms": 1500.0,
                "percentage": 60.0,
                "recommendation": "Cache results",
            },
            {"node_id": "node2", "duration_ms": 500.0, "percentage": 20.0, "recommendation": ""},
        ]
        self.viz.show_bottlenecks(bottlenecks)
        self.console.print.assert_called_once()

    def test_show_bottlenecks_empty(self):
        self.viz.show_bottlenecks([])
        self.console.print.assert_called_once()

    def test_show_bottlenecks_custom_title(self):
        bottlenecks = [
            {"node_id": "n", "duration_ms": 100, "percentage": 50, "recommendation": "Optimize"}
        ]
        self.viz.show_bottlenecks(bottlenecks, title="Custom Bottlenecks")
        self.console.print.assert_called_once()

    def test_show_bottlenecks_missing_fields(self):
        bottlenecks = [{}]
        self.viz.show_bottlenecks(bottlenecks)
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_agent_utilization
    # ------------------------------------------------------------------ #

    def test_show_agent_utilization_basic(self):
        agent_utilization = {
            "agent-a": 0.95,
            "agent-b": 0.72,
            "agent-c": 0.45,
        }
        self.viz.show_agent_utilization(agent_utilization)
        self.console.print.assert_called_once()

    def test_show_agent_utilization_empty(self):
        self.viz.show_agent_utilization({})
        self.console.print.assert_called_once()

    def test_show_agent_utilization_all_threshold_levels(self):
        """Test >=90%, >=70%, and <70% rate badges."""
        agent_utilization = {
            "high-agent": 0.95,  # >= 90% → green
            "mid-agent": 0.75,  # >= 70% → yellow
            "low-agent": 0.50,  # < 70% → red
        }
        self.viz.show_agent_utilization(agent_utilization)
        self.console.print.assert_called_once()

    def test_show_agent_utilization_custom_title(self):
        self.viz.show_agent_utilization({"a": 1.0}, title="My Utilization")
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_cost_breakdown
    # ------------------------------------------------------------------ #

    def test_show_cost_breakdown_basic(self):
        cost_breakdown = {
            "node1": 0.05,
            "node2": 0.02,
            "node3": 0.001,
        }
        self.viz.show_cost_breakdown(cost_breakdown)
        self.console.print.assert_called_once()

    def test_show_cost_breakdown_empty(self):
        self.viz.show_cost_breakdown({})
        self.console.print.assert_called_once()

    def test_show_cost_breakdown_single_item(self):
        self.viz.show_cost_breakdown({"only-node": 0.10})
        self.console.print.assert_called_once()

    def test_show_cost_breakdown_zero_costs(self):
        self.viz.show_cost_breakdown({"n1": 0.0, "n2": 0.0})
        self.console.print.assert_called_once()

    def test_show_cost_breakdown_custom_title(self):
        self.viz.show_cost_breakdown({"n": 1.0}, title="Custom Cost")
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # show_workflow_summary
    # ------------------------------------------------------------------ #

    def test_show_workflow_summary_basic(self):
        self.viz.show_workflow_summary(
            workflow_id="wf-123",
            workflow_name="My Workflow",
            num_nodes=10,
            num_edges=12,
            strategy="sequential",
        )
        self.console.print.assert_called_once()

    def test_show_workflow_summary_with_metadata(self):
        self.viz.show_workflow_summary(
            workflow_id="wf-456",
            workflow_name="Complex Workflow",
            num_nodes=5,
            num_edges=7,
            strategy="parallel",
            metadata={"start_nodes": ["start"], "end_nodes": ["end"]},
        )
        self.console.print.assert_called_once()

    def test_show_workflow_summary_no_metadata(self):
        self.viz.show_workflow_summary("id", "name", 1, 1, "dag", metadata=None)
        self.console.print.assert_called_once()

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def test_get_status_icon_all(self):
        viz = WorkflowVisualizer(console=MagicMock())
        for status in ("completed", "failed", "running", "pending", "skipped", "unknown"):
            icon = viz._get_status_icon(status)
            assert isinstance(icon, str)

    def test_get_event_icon_all(self):
        viz = WorkflowVisualizer(console=MagicMock())
        for event_type in (
            "workflow_start",
            "workflow_end",
            "node_start",
            "node_end",
            "node_error",
            "edge_traversed",
            "state_changed",
            "unknown_event",
        ):
            icon = viz._get_event_icon(event_type)
            assert isinstance(icon, str)


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.visualizers.workflow_viz not available")
class TestWorkflowVizModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_show_workflow_diagram(self):
        mock_console = MagicMock()
        show_workflow_diagram("A -> B", console=mock_console)
        mock_console.print.assert_called_once()

    def test_show_workflow_diagram_custom_title(self):
        mock_console = MagicMock()
        show_workflow_diagram("A -> B", title="Custom", console=mock_console)
        mock_console.print.assert_called_once()

    def test_show_workflow_diagram_no_console(self):
        """Should not raise when no console is provided (uses default)."""
        with patch("beanllm.ui.visualizers.workflow_viz.get_console") as mock_gc:
            mock_gc.return_value = MagicMock()
            show_workflow_diagram("A -> B")

    def test_show_execution_progress(self):
        mock_console = MagicMock()
        show_execution_progress(
            workflow_id="wf-1",
            total_nodes=3,
            nodes_completed=["n1"],
            nodes_running=["n2"],
            nodes_pending=["n3"],
            elapsed_time=2.5,
            console=mock_console,
        )
        mock_console.print.assert_called_once()

    def test_show_execution_progress_no_console(self):
        with patch("beanllm.ui.visualizers.workflow_viz.get_console") as mock_gc:
            mock_gc.return_value = MagicMock()
            show_execution_progress(
                workflow_id="wf-x",
                total_nodes=1,
                nodes_completed=[],
                nodes_running=[],
                nodes_pending=["n1"],
            )

    def test_show_workflow_analytics_all_data(self):
        mock_console = MagicMock()
        show_workflow_analytics(
            bottlenecks=[
                {"node_id": "n1", "duration_ms": 500, "percentage": 50, "recommendation": "fix"}
            ],
            agent_utilization={"agent-a": 0.8},
            cost_breakdown={"n1": 0.05},
            console=mock_console,
        )
        # 3 tables + 2 separating console.print() calls
        assert mock_console.print.call_count >= 3

    def test_show_workflow_analytics_empty_data(self):
        mock_console = MagicMock()
        show_workflow_analytics(
            bottlenecks=[],
            agent_utilization={},
            cost_breakdown={},
            console=mock_console,
        )
        assert mock_console.print.call_count >= 3

    def test_show_workflow_analytics_no_console(self):
        with patch("beanllm.ui.visualizers.workflow_viz.get_console") as mock_gc:
            mock_gc.return_value = MagicMock()
            show_workflow_analytics(
                bottlenecks=[],
                agent_utilization={},
                cost_breakdown={},
            )
