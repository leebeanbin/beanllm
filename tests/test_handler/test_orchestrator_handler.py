"""
OrchestratorHandler 테스트
"""

from unittest.mock import AsyncMock

import pytest

from beanllm.dto.request.advanced.orchestrator_request import (
    CreateWorkflowRequest,
    ExecuteWorkflowRequest,
    MonitorWorkflowRequest,
)
from beanllm.dto.response.advanced.orchestrator_response import (
    AnalyticsResponse,
    CreateWorkflowResponse,
    ExecuteWorkflowResponse,
    MonitorWorkflowResponse,
)
from beanllm.handler.advanced.orchestrator_handler import OrchestratorHandler
from beanllm.service.orchestrator_service import IOrchestratorService


def _make_create_response(workflow_id: str = "wf-1") -> CreateWorkflowResponse:
    return CreateWorkflowResponse(
        workflow_id=workflow_id,
        workflow_name="test-workflow",
        num_nodes=2,
        num_edges=1,
        strategy="sequential",
        visualization="[A] --> [B]",
        created_at="2026-01-01T00:00:00",
    )


def _make_execute_response(execution_id: str = "exec-1") -> ExecuteWorkflowResponse:
    return ExecuteWorkflowResponse(
        execution_id=execution_id,
        workflow_id="wf-1",
        status="completed",
        result="Task result",
        execution_time=1.5,
    )


def _make_monitor_response() -> MonitorWorkflowResponse:
    return MonitorWorkflowResponse(
        execution_id="exec-1",
        workflow_id="wf-1",
        current_node="node-2",
        progress=0.5,
    )


def _make_analytics_response() -> AnalyticsResponse:
    return AnalyticsResponse(
        workflow_id="wf-1",
        total_executions=10,
        avg_execution_time=1.5,
        success_rate=0.9,
        bottlenecks=[],
        agent_utilization={"agent-1": 0.8},
        cost_breakdown={"total": 0.05},
    )


class TestOrchestratorHandler:
    @pytest.fixture
    def mock_service(self) -> AsyncMock:
        service = AsyncMock(spec=IOrchestratorService)
        service.create_workflow.return_value = _make_create_response()
        service.execute_workflow.return_value = _make_execute_response()
        service.monitor_workflow.return_value = _make_monitor_response()
        service.get_analytics.return_value = _make_analytics_response()
        service.get_templates.return_value = {"sequential": {}, "parallel": {}, "debate": {}}
        service.visualize_workflow.return_value = "[A] --> [B]"
        return service

    @pytest.fixture
    def handler(self, mock_service: AsyncMock) -> OrchestratorHandler:
        return OrchestratorHandler(service=mock_service)

    async def test_handle_create_workflow_sequential(self, handler: OrchestratorHandler) -> None:
        request = CreateWorkflowRequest(
            workflow_name="Research Pipeline",
            nodes=[{"id": "n1"}, {"id": "n2"}],
            edges=[{"from": "n1", "to": "n2"}],
            strategy="sequential",
        )
        result = await handler.handle_create_workflow(request)
        assert isinstance(result, CreateWorkflowResponse)
        assert result.workflow_id == "wf-1"

    async def test_handle_create_workflow_no_name_raises(
        self, handler: OrchestratorHandler
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="",
            nodes=[{"id": "n1"}],
            edges=[],
        )
        with pytest.raises(Exception):
            await handler.handle_create_workflow(request)

    async def test_handle_create_workflow_custom_no_nodes_raises(
        self, handler: OrchestratorHandler
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="Custom",
            nodes=[],
            edges=[{"from": "n1", "to": "n2"}],
            strategy="custom",
        )
        with pytest.raises(Exception):
            await handler.handle_create_workflow(request)

    async def test_handle_create_workflow_custom_no_edges_raises(
        self, handler: OrchestratorHandler
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="Custom",
            nodes=[{"id": "n1"}],
            edges=[],
            strategy="custom",
        )
        with pytest.raises(Exception):
            await handler.handle_create_workflow(request)

    async def test_handle_execute_workflow(self, handler: OrchestratorHandler) -> None:
        request = ExecuteWorkflowRequest(
            workflow_id="wf-1",
            input_data={"task": "Research AI trends"},
        )
        result = await handler.handle_execute_workflow(request)
        assert isinstance(result, ExecuteWorkflowResponse)
        assert result.status == "completed"

    async def test_handle_execute_workflow_no_id_raises(self, handler: OrchestratorHandler) -> None:
        request = ExecuteWorkflowRequest(
            workflow_id="",
            input_data={"task": "test"},
        )
        with pytest.raises(Exception):
            await handler.handle_execute_workflow(request)

    async def test_handle_execute_workflow_no_input_raises(
        self, handler: OrchestratorHandler
    ) -> None:
        request = ExecuteWorkflowRequest(
            workflow_id="wf-1",
            input_data={},
        )
        with pytest.raises(Exception):
            await handler.handle_execute_workflow(request)

    async def test_handle_monitor_workflow(self, handler: OrchestratorHandler) -> None:
        request = MonitorWorkflowRequest(
            workflow_id="wf-1",
            execution_id="exec-1",
        )
        result = await handler.handle_monitor_workflow(request)
        assert isinstance(result, MonitorWorkflowResponse)
        assert result.progress == 0.5

    async def test_handle_monitor_no_workflow_id_raises(self, handler: OrchestratorHandler) -> None:
        request = MonitorWorkflowRequest(workflow_id="", execution_id="exec-1")
        with pytest.raises(Exception):
            await handler.handle_monitor_workflow(request)

    async def test_handle_monitor_no_execution_id_raises(
        self, handler: OrchestratorHandler
    ) -> None:
        request = MonitorWorkflowRequest(workflow_id="wf-1", execution_id="")
        with pytest.raises(Exception):
            await handler.handle_monitor_workflow(request)

    async def test_handle_get_analytics(self, handler: OrchestratorHandler) -> None:
        result = await handler.handle_get_analytics("wf-1")
        assert isinstance(result, AnalyticsResponse)
        assert result.total_executions == 10

    async def test_handle_get_analytics_no_id_raises(self, handler: OrchestratorHandler) -> None:
        with pytest.raises(Exception):
            await handler.handle_get_analytics("")

    async def test_handle_get_templates(self, handler: OrchestratorHandler) -> None:
        result = await handler.handle_get_templates()
        assert isinstance(result, dict)
        assert len(result) > 0

    async def test_handle_visualize_workflow(self, handler: OrchestratorHandler) -> None:
        result = await handler.handle_visualize_workflow("wf-1")
        assert isinstance(result, str)

    async def test_handle_visualize_workflow_no_id_raises(
        self, handler: OrchestratorHandler
    ) -> None:
        with pytest.raises(Exception):
            await handler.handle_visualize_workflow("")
