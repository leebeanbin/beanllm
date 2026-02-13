"""
Orchestrator Facade - Multi-Agent 워크플로우 오케스트레이션을 위한 간단한 공개 API
책임: 사용하기 쉬운 인터페이스 제공, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

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

    사용하기 쉬운 공개 API를 제공하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        # Orchestrator 생성
        orchestrator = Orchestrator()

        # 템플릿으로 워크플로우 생성
        workflow = await orchestrator.create_workflow(
            name="Research Pipeline",
            strategy="research_write",
            config={"researcher_id": "r1", "writer_id": "w1"}
        )

        # 워크플로우 실행
        result = await orchestrator.execute(
            workflow_id=workflow.workflow_id,
            agents=agents_dict,
            task="Research AI trends in 2025"
        )

        # 실시간 모니터링
        status = await orchestrator.monitor(
            workflow_id=workflow.workflow_id,
            execution_id=result.execution_id
        )

        # 성능 분석
        analytics = await orchestrator.analyze(workflow.workflow_id)

        # 워크플로우 시각화
        diagram = await orchestrator.visualize(workflow.workflow_id)
        print(diagram)

        # 원스톱: 생성 + 실행
        result = await orchestrator.create_and_execute(
            name="Quick Research",
            strategy="research_write",
            agents=agents_dict,
            task="Analyze the impact of AI"
        )
        ```
    """

    def __init__(self) -> None:
        """Orchestrator 초기화 (Handler는 DI Container로부터 생성)"""
        self._handler: Optional["OrchestratorHandler"] = None
        self._init_handler()

    def _init_handler(self) -> None:
        """Handler 초기화 (DI Container 사용)"""
        from beanllm.utils.core.di_container import get_container

        container = get_container()
        service_factory = container.get_service_factory()
        handler_factory = container.get_handler_factory(service_factory)

        # OrchestratorHandler 생성
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
            config: 전략별 설정 (strategy != "custom" 일 때)
            nodes: 노드 정의 (strategy == "custom" 일 때 필수)
            edges: 엣지 정의 (strategy == "custom" 일 때 필수)

        Returns:
            CreateWorkflowResponse: 생성된 워크플로우 정보

        Raises:
            ValueError: Validation 실패 시
            RuntimeError: 생성 실패 시

        Example:
            ```python
            # 템플릿 사용
            workflow = await orchestrator.create_workflow(
                name="Research & Write",
                strategy="research_write",
                config={
                    "researcher_id": "researcher_agent",
                    "writer_id": "writer_agent",
                    "reviewer_id": "reviewer_agent"  # optional
                }
            )

            # 커스텀 워크플로우
            workflow = await orchestrator.create_workflow(
                name="Custom Pipeline",
                strategy="custom",
                nodes=[
                    {"type": "agent", "name": "agent1", "config": {...}},
                    {"type": "agent", "name": "agent2", "config": {...}}
                ],
                edges=[
                    {"from": "agent1", "to": "agent2"}
                ]
            )
            ```
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

        return cast(CreateWorkflowResponse, response)

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
            agents: Agent 인스턴스 딕셔너리 {agent_id: agent_instance}
            task: 실행할 태스크
            tools: 사용 가능한 도구 딕셔너리 (optional)
            stream: 스트리밍 모드 (optional)

        Returns:
            ExecuteWorkflowResponse: 실행 결과

        Raises:
            ValueError: workflow_id가 없거나 agents가 비어있을 때
            RuntimeError: 실행 실패 시

        Example:
            ```python
            result = await orchestrator.execute(
                workflow_id="wf-123",
                agents={
                    "researcher": researcher_agent,
                    "writer": writer_agent
                },
                task="Research and write about quantum computing",
                tools={"search": search_tool}
            )

            if result.status == "completed":
                print(f"Result: {result.result}")
                print(f"Execution time: {result.execution_time}s")
            ```
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

        return cast(ExecuteWorkflowResponse, response)

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

        Raises:
            ValueError: workflow_id 또는 execution_id가 없을 때
            RuntimeError: 모니터링 실패 시

        Example:
            ```python
            status = await orchestrator.monitor(
                workflow_id="wf-123",
                execution_id="exec-456"
            )

            print(f"Current node: {status.current_node}")
            print(f"Progress: {status.progress * 100}%")
            print(f"Completed: {len(status.nodes_completed)} nodes")
            print(f"Pending: {len(status.nodes_pending)} nodes")
            ```
        """
        logger.debug(f"Monitoring workflow: {workflow_id}, execution={execution_id}")

        request = MonitorWorkflowRequest(
            workflow_id=workflow_id,
            execution_id=execution_id,
            real_time=real_time,
        )

        response = await self._ensure_handler().handle_monitor_workflow(request)

        return cast(MonitorWorkflowResponse, response)

    async def analyze(self, workflow_id: str) -> AnalyticsResponse:
        """
        워크플로우 성능 분석

        Args:
            workflow_id: 워크플로우 ID

        Returns:
            AnalyticsResponse: 분석 결과

        Raises:
            ValueError: workflow_id가 없을 때
            RuntimeError: 분석 실패 시

        Example:
            ```python
            analytics = await orchestrator.analyze("wf-123")

            print(f"Total executions: {analytics.total_executions}")
            print(f"Avg execution time: {analytics.avg_execution_time}s")
            print(f"Success rate: {analytics.success_rate * 100}%")

            # Bottlenecks
            for bottleneck in analytics.bottlenecks:
                print(f"Bottleneck: {bottleneck['node_id']}, "
                      f"{bottleneck['duration_ms']}ms, "
                      f"{bottleneck['recommendation']}")

            # Recommendations
            for rec in analytics.recommendations:
                print(f"- {rec}")
            ```
        """
        logger.info(f"Analyzing workflow: {workflow_id}")

        response = await self._ensure_handler().handle_get_analytics(workflow_id)

        logger.info(
            f"Analytics generated: {response.total_executions} executions, "
            f"success_rate={response.success_rate:.2%}"
        )

        return cast(AnalyticsResponse, response)

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

        Raises:
            ValueError: workflow_id가 없을 때
            RuntimeError: 시각화 실패 시

        Example:
            ```python
            diagram = await orchestrator.visualize("wf-123")
            print(diagram)

            # Output:
            # ┌─────────────┐
            # │  START      │
            # └──────┬──────┘
            #        ▼
            # ┌─────────────┐
            # │ Researcher  │
            # └──────┬──────┘
            #        ▼
            # ┌─────────────┐
            # │   Writer    │
            # └──────┬──────┘
            #        ▼
            # ┌─────────────┐
            # │    END      │
            # └─────────────┘
            ```
        """
        logger.debug(f"Visualizing workflow: {workflow_id}")

        diagram = await self._ensure_handler().handle_visualize_workflow(workflow_id)

        return cast(str, diagram)

    async def get_templates(self) -> Dict[str, Any]:
        """
        사전 정의된 워크플로우 템플릿 목록 조회

        Returns:
            Dict[str, Any]: 템플릿 목록

        Example:
            ```python
            templates = await orchestrator.get_templates()

            for name, info in templates.items():
                print(f"{name}: {info['description']}")
                print(f"  Params: {info['params']}")

            # Output:
            # research_write: Researcher → Writer → [Reviewer]
            #   Params: ['researcher_id', 'writer_id', 'reviewer_id (optional)']
            # parallel: Multiple agents execute in parallel and aggregate results
            #   Params: ['agent_ids', 'aggregation (vote/consensus)']
            # ...
            ```
        """
        logger.debug("Fetching workflow templates")

        templates = await self._ensure_handler().handle_get_templates()

        return cast(Dict[str, Any], templates)

    # ==================== Convenience Methods ====================

    async def create_and_execute(
        self,
        name: str,
        strategy: str,
        agents: Dict[str, Any],
        task: str,
        config: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        워크플로우 생성 및 실행 (원스톱)

        Args:
            name: 워크플로우 이름
            strategy: 전략
            agents: Agent 인스턴스 딕셔너리
            task: 실행할 태스크
            config: 전략별 설정
            tools: 사용 가능한 도구

        Returns:
            Dict: 생성 및 실행 결과

        Example:
            ```python
            result = await orchestrator.create_and_execute(
                name="Quick Research",
                strategy="research_write",
                agents={"researcher": r_agent, "writer": w_agent},
                task="Research quantum computing",
                config={"researcher_id": "researcher", "writer_id": "writer"}
            )

            print(f"Workflow ID: {result['workflow'].workflow_id}")
            print(f"Execution status: {result['execution'].status}")
            print(f"Result: {result['execution'].result}")
            ```
        """
        logger.info(f"Creating and executing workflow: {name}")

        # Create workflow
        workflow = await self.create_workflow(
            name=name,
            strategy=strategy,
            config=config,
        )

        # Execute workflow
        execution = await self.execute(
            workflow_id=workflow.workflow_id,
            agents=agents,
            task=task,
            tools=tools,
        )

        return {
            "workflow": workflow,
            "execution": execution,
        }

    async def run_full_workflow(
        self,
        workflow_id: str,
        agents: Dict[str, Any],
        task: str,
        tools: Optional[Dict[str, Any]] = None,
        monitor: bool = True,
        analyze: bool = True,
    ) -> Dict[str, Any]:
        """
        전체 워크플로우 실행 (실행 + 모니터링 + 분석)

        Args:
            workflow_id: 워크플로우 ID
            agents: Agent 인스턴스 딕셔너리
            task: 실행할 태스크
            tools: 사용 가능한 도구
            monitor: 모니터링 실행 여부
            analyze: 분석 실행 여부

        Returns:
            Dict: 실행, 모니터링, 분석 결과

        Example:
            ```python
            results = await orchestrator.run_full_workflow(
                workflow_id="wf-123",
                agents=agents_dict,
                task="Complex analysis task",
                monitor=True,
                analyze=True
            )

            print(f"Execution: {results['execution'].status}")
            print(f"Monitor: {results['monitor'].progress}")
            print(f"Analytics: {results['analytics'].success_rate}")
            ```
        """
        logger.info(f"Running full workflow: {workflow_id}")

        # Execute
        execution = await self.execute(
            workflow_id=workflow_id,
            agents=agents,
            task=task,
            tools=tools,
        )

        results: Dict[
            str,
            Union[
                ExecuteWorkflowResponse,
                MonitorWorkflowResponse,
                AnalyticsResponse,
            ],
        ] = {"execution": execution}

        # Monitor
        if monitor and execution.execution_id:
            results["monitor"] = await self.monitor(
                workflow_id=workflow_id,
                execution_id=execution.execution_id,
            )

        # Analyze
        if analyze:
            results["analytics"] = await self.analyze(workflow_id)

        logger.info("Full workflow completed")

        return cast(Dict[str, Any], results)
