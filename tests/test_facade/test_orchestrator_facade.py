"""
Orchestrator Facade 테스트
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.dto.response.advanced.orchestrator_response import (
    AnalyticsResponse,
    CreateWorkflowResponse,
    ExecuteWorkflowResponse,
    MonitorWorkflowResponse,
)
from beanllm.facade.advanced.orchestrator_facade import Orchestrator
from beanllm.handler.advanced.orchestrator_handler import OrchestratorHandler


def _make_handler() -> MagicMock:
    handler = AsyncMock(spec=OrchestratorHandler)
    handler.handle_create_workflow.return_value = CreateWorkflowResponse(
        workflow_id="wf-1",
        workflow_name="test",
        num_nodes=2,
        num_edges=1,
        strategy="sequential",
        visualization="[A] --> [B]",
        created_at="2026-01-01T00:00:00",
    )
    handler.handle_execute_workflow.return_value = ExecuteWorkflowResponse(
        execution_id="exec-1",
        workflow_id="wf-1",
        status="completed",
        result="Done",
        execution_time=1.5,
    )
    handler.handle_monitor_workflow.return_value = MonitorWorkflowResponse(
        execution_id="exec-1",
        workflow_id="wf-1",
        current_node="node-2",
        progress=0.75,
    )
    handler.handle_get_analytics.return_value = AnalyticsResponse(
        workflow_id="wf-1",
        total_executions=5,
        avg_execution_time=1.2,
        success_rate=1.0,
        bottlenecks=[],
        agent_utilization={"agent-1": 0.8},
        cost_breakdown={"total": 0.01},
    )
    handler.handle_visualize_workflow.return_value = "[A] --> [B]"
    handler.handle_get_templates.return_value = {"sequential": {}, "parallel": {}}
    return handler


class TestOrchestratorFacade:
    @pytest.fixture
    def orchestrator(self) -> Orchestrator:
        orch = object.__new__(Orchestrator)
        orch._handler = _make_handler()
        return orch

    async def test_create_workflow_sequential(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.create_workflow(
            name="Research Pipeline",
            strategy="sequential",
            nodes=[{"id": "n1"}, {"id": "n2"}],
            edges=[{"from": "n1", "to": "n2"}],
        )
        assert isinstance(result, CreateWorkflowResponse)
        assert result.workflow_id == "wf-1"

    async def test_create_workflow_with_config(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.create_workflow(
            name="Research & Write",
            strategy="research_write",
            config={"researcher_id": "r1", "writer_id": "w1"},
        )
        assert isinstance(result, CreateWorkflowResponse)

    async def test_execute(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.execute(
            workflow_id="wf-1",
            agents={"agent-1": MagicMock()},
            task="Research AI trends",
        )
        assert isinstance(result, ExecuteWorkflowResponse)
        assert result.status == "completed"

    async def test_monitor(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.monitor(
            workflow_id="wf-1",
            execution_id="exec-1",
        )
        assert isinstance(result, MonitorWorkflowResponse)
        assert result.progress == 0.75

    async def test_analyze(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.analyze("wf-1")
        assert isinstance(result, AnalyticsResponse)
        assert result.total_executions == 5
        assert result.success_rate == 1.0

    async def test_visualize(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.visualize("wf-1")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_get_templates(self, orchestrator: Orchestrator) -> None:
        result = await orchestrator.get_templates()
        assert isinstance(result, dict)
        assert "sequential" in result

    async def test_orchestrator_has_handler(self, orchestrator: Orchestrator) -> None:
        assert orchestrator._handler is not None
