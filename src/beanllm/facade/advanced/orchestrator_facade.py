"""
Orchestrator Facade - Multi-Agent 워크플로우 오케스트레이션을 위한 간단한 공개 API
책임: 사용하기 쉬운 인터페이스 제공, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
- 편의 메서드는 OrchestratorConvenienceMixin으로 분리
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
from beanllm.facade.advanced.orchestrator_convenience_mixin import (
    OrchestratorConvenienceMixin,
)
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.handler.advanced.orchestrator_handler import OrchestratorHandler

logger = get_logger(__name__)


class Orchestrator(OrchestratorConvenienceMixin):
    """
    Multi-Agent 워크플로우 오케스트레이터 Facade

    핵심 API (create_workflow, execute, monitor, analyze, visualize, get_templates)를
    제공하며, 편의 메서드(quick_*, create_and_execute, run_full_workflow)는
    OrchestratorConvenienceMixin 에서 상속받습니다.

    Example:
        ```python
        orchestrator = Orchestrator()

        workflow = await orchestrator.create_workflow(
            name="Research Pipeline",
            strategy="research_write",
            config={"researcher_id": "r1", "writer_id": "w1"}
        )

        result = await orchestrator.execute(
            workflow_id=workflow.workflow_id,
            agents=agents_dict,
            task="Research AI trends in 2025"
        )
        ```
    """

    def __init__(self) -> None:
        """Orchestrator 초기화 (Handler는 DI Container로부터 생성)"""
        self._handler: Optional["OrchestratorHandler"] = None
        self._init_handler()

    def _init_handler(self) -> None:
        """Handler 초기화 (DI Container 사용)"""
        from beanllm.utils.di_container import get_container

        container = get_container()
        service_factory = container.get_service_factory()
        handler_factory = container.get_handler_factory(service_factory)
        self._handler = handler_factory.create_orchestrator_handler()

    def _ensure_handler(self) -> "OrchestratorHandler":
        """Handler 존재 확인 및 반환"""
        assert self._handler is not None, "OrchestratorHandler not initialized"
        return self._handler

    async def create_workflow(
        self,
        name: str,
        strategy: str = "custom",
        config: Optional[Dict[str, Any]] = None,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
    ) -> CreateWorkflowResponse:
        """
        워크플로우 생성

        Args:
            name: 워크플로우 이름
            strategy: 전략 ("research_write", "parallel", "hierarchical", "debate", "pipeline", "custom")
            config: 전략별 설정
            nodes: 노드 정의 (strategy == "custom" 일 때 필수)
            edges: 엣지 정의 (strategy == "custom" 일 때 필수)

        Returns:
            CreateWorkflowResponse: 생성된 워크플로우 정보
        """
        logger.info(f"Creating workflow: {name}, strategy={strategy}")

        request = CreateWorkflowRequest(
            workflow_name=name,
            strategy=strategy,
            config=config or {},
            nodes=nodes or [],
            edges=edges or [],
        )

        response = await self._ensure_handler().handle_create_workflow(request)

        logger.info(
            f"Workflow created: {response.workflow_id}, "
            f"{response.num_nodes} nodes, {response.num_edges} edges"
        )

        return response

    async def execute(
        self,
        workflow_id: str,
        agents: Dict[str, Any],
        task: str,
        tools: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> ExecuteWorkflowResponse:
        """
        워크플로우 실행

        Args:
            workflow_id: 워크플로우 ID
            agents: Agent 인스턴스 딕셔너리
            task: 실행할 태스크
            tools: 사용 가능한 도구 (optional)
            stream: 스트리밍 모드 (optional)

        Returns:
            ExecuteWorkflowResponse: 실행 결과
        """
        logger.info(f"Executing workflow: {workflow_id}")

        request = ExecuteWorkflowRequest(
            workflow_id=workflow_id,
            input_data={
                "task": task,
                "agents": agents,
                "tools": tools or {},
            },
            stream=stream,
        )

        response = await self._ensure_handler().handle_execute_workflow(request)

        logger.info(
            f"Workflow execution {response.status}: {response.execution_id}, "
            f"time={response.execution_time:.2f}s"
        )

        return response

    async def monitor(
        self,
        workflow_id: str,
        execution_id: str,
        real_time: bool = False,
    ) -> MonitorWorkflowResponse:
        """
        워크플로우 실시간 모니터링

        Args:
            workflow_id: 워크플로우 ID
            execution_id: 실행 ID
            real_time: 실시간 업데이트 여부

        Returns:
            MonitorWorkflowResponse: 모니터링 데이터
        """
        logger.debug(f"Monitoring workflow: {workflow_id}, execution={execution_id}")

        request = MonitorWorkflowRequest(
            workflow_id=workflow_id,
            execution_id=execution_id,
            real_time=real_time,
        )

        return await self._ensure_handler().handle_monitor_workflow(request)

    async def analyze(self, workflow_id: str) -> AnalyticsResponse:
        """
        워크플로우 성능 분석

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            AnalyticsResponse: 분석 결과
        """
        logger.info(f"Analyzing workflow: {workflow_id}")

        response = await self._ensure_handler().handle_get_analytics(workflow_id)

        logger.info(
            f"Analytics generated: {response.total_executions} executions, "
            f"success_rate={response.success_rate:.2%}"
        )

        return response

    async def visualize(
        self,
        workflow_id: str,
        style: str = "box",
    ) -> str:
        """
        워크플로우 시각화 (ASCII 다이어그램)

        Args:
            workflow_id: 워크플로우 ID
            style: 다이어그램 스타일 ("box", "simple", "compact")

        Returns:
            str: ASCII 다이어그램
        """
        logger.debug(f"Visualizing workflow: {workflow_id}")
        return await self._ensure_handler().handle_visualize_workflow(workflow_id)

    async def get_templates(self) -> Dict[str, Any]:
        """
        사전 정의된 워크플로우 템플릿 목록 조회

        Returns:
            Dict[str, Any]: 템플릿 목록
        """
        logger.debug("Fetching workflow templates")
        return await self._ensure_handler().handle_get_templates()
