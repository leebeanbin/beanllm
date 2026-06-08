"""
OrchestratorService 테스트 - 워크플로우 생성, 실행, 모니터링, 분석
"""

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
from beanllm.service.impl.advanced.orchestrator_service_impl import OrchestratorServiceImpl


@pytest.fixture
def service() -> OrchestratorServiceImpl:
    return OrchestratorServiceImpl()


@pytest.fixture
def simple_workflow_request() -> CreateWorkflowRequest:
    return CreateWorkflowRequest(
        workflow_name="test-workflow",
        strategy="custom",
        nodes=[
            {"name": "start", "type": "start"},
            {"name": "agent1", "type": "agent"},
            {"name": "end", "type": "end"},
        ],
        edges=[
            {"from": "start", "to": "agent1"},
            {"from": "agent1", "to": "end"},
        ],
    )


class TestCreateWorkflow:
    @pytest.mark.asyncio
    async def test_create_custom_workflow(
        self, service: OrchestratorServiceImpl, simple_workflow_request: CreateWorkflowRequest
    ) -> None:
        response = await service.create_workflow(simple_workflow_request)
        assert isinstance(response, CreateWorkflowResponse)
        assert response.workflow_id is not None
        assert response.workflow_name == "test-workflow"

    @pytest.mark.asyncio
    async def test_create_workflow_from_template_sequential(
        self, service: OrchestratorServiceImpl
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="sequential-wf",
            strategy="research_write",
            nodes=[],
            edges=[],
        )
        response = await service.create_workflow(request)
        assert response.workflow_id is not None

    @pytest.mark.asyncio
    async def test_create_workflow_from_template_parallel(
        self, service: OrchestratorServiceImpl
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="parallel-wf",
            strategy="parallel",
            nodes=[],
            edges=[],
        )
        response = await service.create_workflow(request)
        assert response.workflow_id is not None

    @pytest.mark.asyncio
    async def test_create_workflow_from_template_hierarchical(
        self, service: OrchestratorServiceImpl
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="hier-wf",
            strategy="hierarchical",
            nodes=[],
            edges=[],
        )
        response = await service.create_workflow(request)
        assert response.workflow_id is not None

    @pytest.mark.asyncio
    async def test_create_workflow_from_template_debate(
        self, service: OrchestratorServiceImpl
    ) -> None:
        request = CreateWorkflowRequest(
            workflow_name="debate-wf",
            strategy="debate",
            nodes=[],
            edges=[],
        )
        response = await service.create_workflow(request)
        assert response.workflow_id is not None

    @pytest.mark.asyncio
    async def test_created_workflow_stored(
        self, service: OrchestratorServiceImpl, simple_workflow_request: CreateWorkflowRequest
    ) -> None:
        response = await service.create_workflow(simple_workflow_request)
        assert response.workflow_id in service._workflows

    @pytest.mark.asyncio
    async def test_create_workflow_returns_visualization(
        self, service: OrchestratorServiceImpl, simple_workflow_request: CreateWorkflowRequest
    ) -> None:
        response = await service.create_workflow(simple_workflow_request)
        assert response.visualization is not None


class TestExecuteWorkflow:
    @pytest.mark.asyncio
    async def test_execute_workflow_basic(self, service: OrchestratorServiceImpl) -> None:
        create_req = CreateWorkflowRequest(
            workflow_name="exec-wf",
            strategy="research_write",
            nodes=[],
            edges=[],
        )
        create_resp = await service.create_workflow(create_req)

        exec_req = ExecuteWorkflowRequest(
            workflow_id=create_resp.workflow_id,
            input_data={"task": "Summarize AI trends"},
        )
        response = await service.execute_workflow(exec_req)
        assert isinstance(response, ExecuteWorkflowResponse)
        assert response.execution_id is not None
        assert response.workflow_id == create_resp.workflow_id

    @pytest.mark.asyncio
    async def test_execute_nonexistent_workflow_raises(
        self, service: OrchestratorServiceImpl
    ) -> None:
        exec_req = ExecuteWorkflowRequest(
            workflow_id="nonexistent-id",
            input_data={"task": "test"},
        )
        with pytest.raises((ValueError, KeyError)):
            await service.execute_workflow(exec_req)

    @pytest.mark.asyncio
    async def test_execute_workflow_creates_monitor(self, service: OrchestratorServiceImpl) -> None:
        create_req = CreateWorkflowRequest(
            workflow_name="mon-wf",
            strategy="parallel",
            nodes=[],
            edges=[],
        )
        create_resp = await service.create_workflow(create_req)

        exec_req = ExecuteWorkflowRequest(
            workflow_id=create_resp.workflow_id,
            input_data={"task": "test task"},
        )
        exec_resp = await service.execute_workflow(exec_req)
        assert exec_resp.execution_id in service._monitors


class TestMonitorWorkflow:
    @pytest.mark.asyncio
    async def test_monitor_running_execution(self, service: OrchestratorServiceImpl) -> None:
        create_req = CreateWorkflowRequest(
            workflow_name="mon-wf",
            strategy="sequential",
            nodes=[],
            edges=[],
        )
        create_resp = await service.create_workflow(create_req)
        exec_resp = await service.execute_workflow(
            ExecuteWorkflowRequest(
                workflow_id=create_resp.workflow_id,
                input_data={"task": "monitor test"},
            )
        )

        monitor_req = MonitorWorkflowRequest(
            workflow_id=create_resp.workflow_id,
            execution_id=exec_resp.execution_id,
        )
        response = await service.monitor_workflow(monitor_req)
        assert isinstance(response, MonitorWorkflowResponse)
        assert response.execution_id == exec_resp.execution_id

    @pytest.mark.asyncio
    async def test_monitor_nonexistent_execution_raises(
        self, service: OrchestratorServiceImpl
    ) -> None:
        monitor_req = MonitorWorkflowRequest(
            workflow_id="wf-id",
            execution_id="exec-nonexistent",
        )
        with pytest.raises((ValueError, KeyError)):
            await service.monitor_workflow(monitor_req)


class TestGetAnalytics:
    @pytest.mark.asyncio
    async def test_analytics_for_workflow(self, service: OrchestratorServiceImpl) -> None:
        create_req = CreateWorkflowRequest(
            workflow_name="analytics-wf",
            strategy="research_write",
            nodes=[],
            edges=[],
        )
        create_resp = await service.create_workflow(create_req)
        await service.execute_workflow(
            ExecuteWorkflowRequest(
                workflow_id=create_resp.workflow_id,
                input_data={"task": "analytics test"},
            )
        )

        response = await service.get_analytics(create_resp.workflow_id)
        assert isinstance(response, AnalyticsResponse)
        assert response.workflow_id == create_resp.workflow_id

    @pytest.mark.asyncio
    async def test_analytics_nonexistent_workflow_raises(
        self, service: OrchestratorServiceImpl
    ) -> None:
        with pytest.raises((ValueError, KeyError)):
            await service.get_analytics("nonexistent-wf")


class TestVisualizeWorkflow:
    @pytest.mark.asyncio
    async def test_visualize_returns_string(self, service: OrchestratorServiceImpl) -> None:
        create_req = CreateWorkflowRequest(
            workflow_name="vis-wf",
            strategy="parallel",
            nodes=[],
            edges=[],
        )
        create_resp = await service.create_workflow(create_req)
        result = await service.visualize_workflow(create_resp.workflow_id)
        assert isinstance(result, str)
        assert len(result) > 0


class TestGetTemplates:
    @pytest.mark.asyncio
    async def test_get_templates_returns_dict(self, service: OrchestratorServiceImpl) -> None:
        templates = await service.get_templates()
        assert isinstance(templates, dict)

    @pytest.mark.asyncio
    async def test_templates_not_empty(self, service: OrchestratorServiceImpl) -> None:
        templates = await service.get_templates()
        assert len(templates) > 0
