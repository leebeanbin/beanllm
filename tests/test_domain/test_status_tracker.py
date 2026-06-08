"""Tests for domain/orchestrator/monitoring/status_tracker.py — StatusTracker."""

from datetime import datetime

import pytest

from beanllm.domain.orchestrator.monitor_types import NodeStatus
from beanllm.domain.orchestrator.monitoring.status_tracker import StatusTracker


class TestStatusTrackerInit:
    def test_init_stores_workflow_id(self):
        tracker = StatusTracker("wf-001", total_nodes=5)
        assert tracker.workflow_id == "wf-001"

    def test_init_stores_total_nodes(self):
        tracker = StatusTracker("wf-001", total_nodes=5)
        assert tracker.total_nodes == 5

    def test_init_empty_node_states(self):
        tracker = StatusTracker("wf-001")
        assert tracker.node_states == {}

    def test_init_start_time_none(self):
        tracker = StatusTracker("wf-001")
        assert tracker.start_time is None

    def test_init_end_time_none(self):
        tracker = StatusTracker("wf-001")
        assert tracker.end_time is None

    def test_init_stats_zeroed(self):
        tracker = StatusTracker("wf-001")
        assert tracker.stats["nodes_completed"] == 0
        assert tracker.stats["nodes_failed"] == 0
        assert tracker.stats["nodes_running"] == 0
        assert tracker.stats["nodes_pending"] == 0


class TestGetStatus:
    def test_get_status_workflow_id(self):
        tracker = StatusTracker("wf-xyz")
        status = tracker.get_status()
        assert status["workflow_id"] == "wf-xyz"

    def test_get_status_not_running_initially(self):
        tracker = StatusTracker("wf-001")
        status = tracker.get_status()
        assert status["is_running"] is False

    def test_get_status_progress_zero_with_no_nodes(self):
        tracker = StatusTracker("wf-001", total_nodes=0)
        status = tracker.get_status()
        assert status["progress_percent"] == 0.0

    def test_get_status_progress_after_completion(self):
        tracker = StatusTracker("wf-001", total_nodes=4)
        tracker.stats["nodes_completed"] = 2
        status = tracker.get_status()
        assert status["progress_percent"] == 50.0

    def test_get_status_running_when_started(self):
        tracker = StatusTracker("wf-001")
        tracker.start_time = datetime.now()
        status = tracker.get_status()
        assert status["is_running"] is True

    def test_get_status_not_running_when_ended(self):
        tracker = StatusTracker("wf-001")
        tracker.start_time = datetime.now()
        tracker.end_time = datetime.now()
        status = tracker.get_status()
        assert status["is_running"] is False

    def test_get_status_elapsed_ms_nonzero_when_started(self):
        import time

        tracker = StatusTracker("wf-001")
        tracker.start_time = datetime.now()
        time.sleep(0.01)
        status = tracker.get_status()
        assert status["elapsed_ms"] > 0


class TestGetNodeState:
    def test_returns_none_for_unknown_node(self):
        tracker = StatusTracker("wf-001")
        assert tracker.get_node_state("nonexistent") is None

    def test_returns_state_after_node_started(self):
        tracker = StatusTracker("wf-001", total_nodes=2)
        tracker.node_started("node-A")
        state = tracker.get_node_state("node-A")
        assert state is not None
        assert state.status == NodeStatus.RUNNING


class TestGetAllNodeStates:
    def test_returns_empty_dict_initially(self):
        tracker = StatusTracker("wf-001")
        states = tracker.get_all_node_states()
        assert states == {}

    def test_returns_copy_not_original(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        states = tracker.get_all_node_states()
        states["new-key"] = None
        assert "new-key" not in tracker.node_states


class TestNodeStarted:
    def test_returns_monitor_event(self):
        from beanllm.domain.orchestrator.monitor_types import MonitorEvent

        tracker = StatusTracker("wf-001", total_nodes=2)
        event = tracker.node_started("node-A")
        assert isinstance(event, MonitorEvent)

    def test_sets_node_to_running(self):
        tracker = StatusTracker("wf-001", total_nodes=2)
        tracker.node_started("node-A")
        assert tracker.node_states["node-A"].status == NodeStatus.RUNNING

    def test_increments_nodes_running_stat(self):
        tracker = StatusTracker("wf-001")
        tracker.node_started("node-A")
        assert tracker.stats["nodes_running"] == 1

    def test_second_attempt_increments_attempts(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_states["node-A"].status = NodeStatus.RUNNING
        tracker.node_started("node-A")
        assert tracker.node_states["node-A"].attempts == 2

    def test_node_started_with_metadata(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A", metadata={"agent": "llm-agent"})
        assert tracker.node_states["node-A"].metadata["agent"] == "llm-agent"


class TestNodeCompleted:
    def test_returns_monitor_event(self):
        from beanllm.domain.orchestrator.monitor_types import MonitorEvent

        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        event = tracker.node_completed("node-A", output="result")
        assert isinstance(event, MonitorEvent)

    def test_sets_node_to_completed(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_completed("node-A")
        assert tracker.node_states["node-A"].status == NodeStatus.COMPLETED

    def test_increments_nodes_completed_stat(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_completed("node-A")
        assert tracker.stats["nodes_completed"] == 1

    def test_decrements_nodes_running_stat(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_completed("node-A")
        assert tracker.stats["nodes_running"] == 0

    def test_completed_without_start_creates_state(self):
        tracker = StatusTracker("wf-001")
        tracker.node_completed("node-X")
        assert "node-X" in tracker.node_states
        assert tracker.node_states["node-X"].status == NodeStatus.COMPLETED

    def test_completed_stores_output(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_completed("node-A", output="my_output")
        assert tracker.node_states["node-A"].output == "my_output"

    def test_completed_calculates_duration(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_completed("node-A")
        assert tracker.node_states["node-A"].duration_ms >= 0

    def test_completed_with_metadata(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_completed("node-A", metadata={"result_count": 5})
        assert tracker.node_states["node-A"].metadata["result_count"] == 5


class TestNodeFailed:
    def test_returns_monitor_event(self):
        from beanllm.domain.orchestrator.monitor_types import MonitorEvent

        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        event = tracker.node_failed("node-A", error="Timeout")
        assert isinstance(event, MonitorEvent)

    def test_sets_node_to_failed(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_failed("node-A", error="Error!")
        assert tracker.node_states["node-A"].status == NodeStatus.FAILED

    def test_increments_nodes_failed_stat(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_failed("node-A", error="err")
        assert tracker.stats["nodes_failed"] == 1

    def test_failed_without_start_creates_state(self):
        tracker = StatusTracker("wf-001")
        tracker.node_failed("node-X", error="unexpected")
        assert "node-X" in tracker.node_states

    def test_failed_stores_error(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_failed("node-A", error="TimeoutError")
        assert tracker.node_states["node-A"].error == "TimeoutError"


class TestNodeSkipped:
    def test_returns_monitor_event(self):
        from beanllm.domain.orchestrator.monitor_types import MonitorEvent

        tracker = StatusTracker("wf-001")
        event = tracker.node_skipped("node-A")
        assert isinstance(event, MonitorEvent)

    def test_sets_node_to_skipped(self):
        tracker = StatusTracker("wf-001")
        tracker.node_skipped("node-A")
        assert tracker.node_states["node-A"].status == NodeStatus.SKIPPED

    def test_skipped_existing_node_updates_status(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.node_skipped("node-A")
        assert tracker.node_states["node-A"].status == NodeStatus.SKIPPED

    def test_skipped_with_custom_reason(self):
        from beanllm.domain.orchestrator.monitor_types import EventType

        tracker = StatusTracker("wf-001")
        event = tracker.node_skipped("node-B", reason="Dependency failed")
        assert event.data["reason"] == "Dependency failed"


class TestReset:
    def test_reset_clears_node_states(self):
        tracker = StatusTracker("wf-001", total_nodes=1)
        tracker.node_started("node-A")
        tracker.reset()
        assert tracker.node_states == {}

    def test_reset_clears_times(self):
        tracker = StatusTracker("wf-001")
        tracker.start_time = datetime.now()
        tracker.end_time = datetime.now()
        tracker.reset()
        assert tracker.start_time is None
        assert tracker.end_time is None

    def test_reset_zeroes_stats(self):
        tracker = StatusTracker("wf-001", total_nodes=2)
        tracker.node_started("node-A")
        tracker.node_completed("node-A")
        tracker.reset()
        assert tracker.stats["nodes_completed"] == 0
        assert tracker.stats["nodes_running"] == 0
