"""
Orchestrator Domain 타입 테스트 - WorkflowNode, WorkflowEdge, ExecutionResult,
MonitorEvent, NodeExecutionState, StatisticsAnalyzer
"""

from datetime import datetime

import pytest

from beanllm.domain.orchestrator.analytics.statistics import StatisticsAnalyzer
from beanllm.domain.orchestrator.monitor_types import (
    EventType,
    MonitorEvent,
    NodeExecutionState,
    NodeStatus,
)
from beanllm.domain.orchestrator.workflow_types import (
    EdgeCondition,
    ExecutionResult,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)


class TestNodeStatus:
    def test_node_status_values(self) -> None:
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.COMPLETED.value == "completed"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.SKIPPED.value == "skipped"


class TestEventType:
    def test_event_type_values(self) -> None:
        assert EventType.WORKFLOW_START.value == "workflow_start"
        assert EventType.WORKFLOW_END.value == "workflow_end"
        assert EventType.NODE_START.value == "node_start"
        assert EventType.NODE_END.value == "node_end"
        assert EventType.NODE_ERROR.value == "node_error"
        assert EventType.EDGE_TRAVERSED.value == "edge_traversed"


class TestMonitorEvent:
    def test_create_monitor_event(self) -> None:
        ts = datetime(2025, 1, 1, 12, 0, 0)
        event = MonitorEvent(
            event_type=EventType.WORKFLOW_START,
            timestamp=ts,
            workflow_id="wf-1",
        )
        assert event.event_type == EventType.WORKFLOW_START
        assert event.workflow_id == "wf-1"
        assert event.node_id is None

    def test_monitor_event_with_node(self) -> None:
        event = MonitorEvent(
            event_type=EventType.NODE_START,
            timestamp=datetime.now(),
            workflow_id="wf-1",
            node_id="node-1",
            data={"input": "test"},
        )
        assert event.node_id == "node-1"
        assert event.data["input"] == "test"

    def test_monitor_event_to_dict(self) -> None:
        ts = datetime(2025, 6, 1, 10, 0, 0)
        event = MonitorEvent(
            event_type=EventType.NODE_END,
            timestamp=ts,
            workflow_id="wf-2",
            node_id="n1",
            data={"result": "ok"},
        )
        d = event.to_dict()
        assert d["event_type"] == "node_end"
        assert d["workflow_id"] == "wf-2"
        assert d["node_id"] == "n1"
        assert "timestamp" in d


class TestNodeExecutionState:
    def test_create_node_state(self) -> None:
        state = NodeExecutionState(
            node_id="n1",
            status=NodeStatus.COMPLETED,
            duration_ms=150.0,
        )
        assert state.node_id == "n1"
        assert state.status == NodeStatus.COMPLETED
        assert state.duration_ms == 150.0

    def test_node_state_to_dict(self) -> None:
        state = NodeExecutionState(
            node_id="n2",
            status=NodeStatus.FAILED,
            error="Connection timeout",
            attempts=3,
        )
        d = state.to_dict()
        assert d["node_id"] == "n2"
        assert d["status"] == "failed"
        assert d["error"] == "Connection timeout"
        assert d["attempts"] == 3

    def test_node_state_with_times(self) -> None:
        start = datetime(2025, 1, 1, 12, 0, 0)
        end = datetime(2025, 1, 1, 12, 0, 1)
        state = NodeExecutionState(
            node_id="n3",
            status=NodeStatus.COMPLETED,
            start_time=start,
            end_time=end,
            duration_ms=1000.0,
        )
        d = state.to_dict()
        assert d["start_time"] is not None
        assert d["end_time"] is not None


class TestNodeType:
    def test_node_type_values(self) -> None:
        assert NodeType.AGENT.value == "agent"
        assert NodeType.TOOL.value == "tool"
        assert NodeType.DECISION.value == "decision"
        assert NodeType.PARALLEL.value == "parallel"
        assert NodeType.START.value == "start"
        assert NodeType.END.value == "end"


class TestWorkflowNode:
    def test_create_workflow_node(self) -> None:
        node = WorkflowNode(
            node_id="n1",
            node_type=NodeType.AGENT,
            name="Agent Node",
        )
        assert node.node_id == "n1"
        assert node.node_type == NodeType.AGENT
        assert node.name == "Agent Node"

    def test_workflow_node_to_dict(self) -> None:
        node = WorkflowNode(
            node_id="n2",
            node_type=NodeType.TOOL,
            name="Tool Node",
            config={"tool": "search"},
            position=(10, 20),
        )
        d = node.to_dict()
        assert d["node_id"] == "n2"
        assert d["node_type"] == "tool"
        assert d["config"] == {"tool": "search"}
        assert d["position"] == (10, 20)

    def test_workflow_node_from_dict(self) -> None:
        data = {
            "node_id": "n3",
            "node_type": "agent",
            "name": "Agent",
            "config": {},
            "position": (0, 0),
            "metadata": {},
        }
        node = WorkflowNode.from_dict(data)
        assert node.node_id == "n3"
        assert node.node_type == NodeType.AGENT


class TestWorkflowEdge:
    def test_create_workflow_edge(self) -> None:
        edge = WorkflowEdge(
            edge_id="e1",
            source="n1",
            target="n2",
        )
        assert edge.edge_id == "e1"
        assert edge.source == "n1"
        assert edge.target == "n2"
        assert edge.condition == EdgeCondition.ALWAYS

    def test_edge_should_execute_always(self) -> None:
        edge = WorkflowEdge(edge_id="e1", source="n1", target="n2", condition=EdgeCondition.ALWAYS)
        assert edge.should_execute({"success": False}) is True

    def test_edge_should_execute_on_success_true(self) -> None:
        edge = WorkflowEdge(
            edge_id="e2", source="n1", target="n2", condition=EdgeCondition.ON_SUCCESS
        )
        assert edge.should_execute({"success": True}) is True

    def test_edge_should_execute_on_success_false(self) -> None:
        edge = WorkflowEdge(
            edge_id="e3", source="n1", target="n2", condition=EdgeCondition.ON_SUCCESS
        )
        assert edge.should_execute({"success": False}) is False

    def test_edge_should_execute_on_failure(self) -> None:
        edge = WorkflowEdge(
            edge_id="e4", source="n1", target="n2", condition=EdgeCondition.ON_FAILURE
        )
        assert edge.should_execute({"success": False}) is True
        assert edge.should_execute({"success": True}) is False

    def test_edge_should_execute_conditional(self) -> None:
        edge = WorkflowEdge(
            edge_id="e5",
            source="n1",
            target="n2",
            condition=EdgeCondition.CONDITIONAL,
            condition_func=lambda ctx: ctx.get("value", 0) > 5,
        )
        assert edge.should_execute({"value": 10}) is True
        assert edge.should_execute({"value": 3}) is False

    def test_edge_should_execute_conditional_no_func(self) -> None:
        edge = WorkflowEdge(
            edge_id="e6",
            source="n1",
            target="n2",
            condition=EdgeCondition.CONDITIONAL,
        )
        assert edge.should_execute({}) is True

    def test_edge_to_dict(self) -> None:
        edge = WorkflowEdge(
            edge_id="e7",
            source="n1",
            target="n2",
            condition=EdgeCondition.ON_SUCCESS,
        )
        d = edge.to_dict()
        assert d["edge_id"] == "e7"
        assert d["source"] == "n1"
        assert d["target"] == "n2"
        assert d["condition"] == "on_success"


class TestExecutionResult:
    def test_create_execution_result(self) -> None:
        result = ExecutionResult(
            node_id="n1",
            success=True,
            output="result text",
        )
        assert result.node_id == "n1"
        assert result.success is True
        assert result.output == "result text"

    def test_execution_result_to_dict(self) -> None:
        result = ExecutionResult(
            node_id="n2",
            success=False,
            output={"data": "val"},
            error="Timeout",
            duration_ms=500.0,
        )
        d = result.to_dict()
        assert d["node_id"] == "n2"
        assert d["success"] is False
        assert d["error"] == "Timeout"
        assert d["duration_ms"] == 500.0


class TestStatisticsAnalyzer:
    def test_get_node_statistics_empty(self) -> None:
        stats = StatisticsAnalyzer.get_node_statistics("node-1", {})
        assert stats["node_id"] == "node-1"
        assert stats["executions"] == 0

    def test_get_node_statistics_with_data(self) -> None:
        node_metrics = {"node-1": [100.0, 200.0, 150.0]}
        stats = StatisticsAnalyzer.get_node_statistics("node-1", node_metrics)
        assert stats["executions"] == 3
        assert stats["avg_duration_ms"] == pytest.approx(150.0)
        assert stats["min_duration_ms"] == 100.0
        assert stats["max_duration_ms"] == 200.0
        assert stats["total_duration_ms"] == 450.0

    def test_compare_executions(self) -> None:
        state_a = NodeExecutionState(node_id="n1", status=NodeStatus.COMPLETED, duration_ms=100.0)
        state_b = NodeExecutionState(node_id="n2", status=NodeStatus.COMPLETED, duration_ms=200.0)

        # Workflow A has faster execution
        states_a = {"n1": state_a}
        states_b = {"n2": state_b}

        result = StatisticsAnalyzer.compare_executions(states_a, states_b, "wf-a", "wf-b")

        assert "workflow_a" in result
        assert "workflow_b" in result
        assert "comparison" in result
        assert result["workflow_a"]["workflow_id"] == "wf-a"
        assert result["comparison"]["faster"] == "wf-a"

    def test_compare_executions_empty(self) -> None:
        result = StatisticsAnalyzer.compare_executions({}, {}, "wf-a", "wf-b")
        assert result["workflow_a"]["success_rate"] == 0.0
        assert result["workflow_b"]["success_rate"] == 0.0

    def test_get_summary_statistics_empty(self) -> None:
        stats = StatisticsAnalyzer.get_summary_statistics({}, {}, {})
        assert stats["total_executions"] == 0

    def test_get_summary_statistics_with_data(self) -> None:
        completed = NodeExecutionState(node_id="n1", status=NodeStatus.COMPLETED, duration_ms=100.0)
        failed = NodeExecutionState(node_id="n2", status=NodeStatus.FAILED, duration_ms=0.0)

        executions = {
            "exec-1": {"node_states": {"n1": completed, "n2": failed}},
            "exec-2": {"node_states": {"n1": completed}},
        }

        stats = StatisticsAnalyzer.get_summary_statistics(executions, {}, {})
        assert stats["total_executions"] == 2
        assert stats["avg_duration_ms"] >= 0

    def test_edge_condition_values(self) -> None:
        assert EdgeCondition.ALWAYS.value == "always"
        assert EdgeCondition.ON_SUCCESS.value == "on_success"
        assert EdgeCondition.ON_FAILURE.value == "on_failure"
        assert EdgeCondition.CONDITIONAL.value == "conditional"
