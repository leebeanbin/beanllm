"""
OrchestratorServiceImpl - Multi-Agent 오케스트레이터 서비스 구현체
SOLID 원칙:
- SRP: 오케스트레이션 비즈니스 로직만 담당
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from beanllm.domain.orchestrator import (
    NodeType,
    VisualBuilder,
    WorkflowAnalytics,
    WorkflowGraph,
    WorkflowMonitor,
    WorkflowTemplates,
)
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
from beanllm.utils.logging import get_logger

from ...orchestrator_service import IOrchestratorService

logger = get_logger(__name__)


class OrchestratorServiceImpl(IOrchestratorService):
    """
    Multi-Agent 오케스트레이터 서비스 구현체

    책임:
    - 워크플로우 생성, 저장, 관리
    - 실행 오케스트레이션
    - 모니터링 데이터 수집
    - 분석 데이터 제공
    """

    def __init__(self) -> None:
        """Initialize service with storage"""
        # Workflow storage: workflow_id -> WorkflowGraph
        self._workflows: Dict[str, WorkflowGraph] = {}

        # Monitor storage: execution_id -> WorkflowMonitor
        self._monitors: Dict[str, WorkflowMonitor] = {}

        # Analytics engine
        self._analytics = WorkflowAnalytics()

        logger.info("OrchestratorService initialized")

    async def create_workflow(
        self, request: CreateWorkflowRequest
    ) -> CreateWorkflowResponse:
        """워크플로우 생성"""
        logger.info(f"Creating workflow: {request.workflow_name}")

        # Check if using template
        if request.strategy in ["research_write", "parallel", "hierarchical", "debate"]:
            workflow = self._create_from_template(request)
        else:
            # Create custom workflow
            workflow = WorkflowGraph(name=request.workflow_name)

            # Add nodes
            node_id_map = {}
            for node_def in request.nodes:
                node_id = workflow.add_node(
                    node_type=NodeType(node_def.get("type", "agent")),
                    name=node_def.get("name", "node"),
                    config=node_def.get("config", {}),
                    position=node_def.get("position", (0, 0)),
                )
                node_id_map[node_def.get("name")] = node_id

            # Add edges
            for edge_def in request.edges:
                source_name = edge_def.get("from")
                target_name = edge_def.get("to")

                if source_name in node_id_map and target_name in node_id_map:
                    workflow.add_edge(
                        source=node_id_map[source_name],
                        target=node_id_map[target_name],
                    )

        # Store workflow
        self._workflows[workflow.workflow_id] = workflow

        # Generate visualization
        builder = VisualBuilder(workflow)
        diagram = builder.build_diagram(style="box")

        # Create response
        response = CreateWorkflowResponse(
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            num_nodes=len(workflow.nodes),
            num_edges=len(workflow.edges),
            strategy=request.strategy,
            visualization=diagram,
            created_at=datetime.now().isoformat(),
            metadata={
                "start_nodes": len(workflow.get_start_nodes()),
                "end_nodes": len(workflow.get_end_nodes()),
            },
        )

        logger.info(f"Workflow created: {workflow.workflow_id}")
        return response

    def _create_from_template(self, request: CreateWorkflowRequest) -> WorkflowGraph:
        """템플릿으로부터 워크플로우 생성"""
        config = request.config or {}

        if request.strategy == "research_write":
            workflow = WorkflowTemplates.research_and_write(
                researcher_id=config.get("researcher_id", "researcher"),
                writer_id=config.get("writer_id", "writer"),
                reviewer_id=config.get("reviewer_id"),
            )
        elif request.strategy == "parallel":
            workflow = WorkflowTemplates.parallel_consensus(
                agent_ids=config.get("agent_ids", ["agent1", "agent2"]),
                aggregation=config.get("aggregation", "vote"),
            )
        elif request.strategy == "hierarchical":
            workflow = WorkflowTemplates.hierarchical_delegation(
                manager_id=config.get("manager_id", "manager"),
                worker_ids=config.get("worker_ids", ["worker1", "worker2"]),
            )
        elif request.strategy == "debate":
            workflow = WorkflowTemplates.debate_and_judge(
                debater_ids=config.get("debater_ids", ["debater1", "debater2"]),
                judge_id=config.get("judge_id", "judge"),
                rounds=config.get("rounds", 3),
            )
        else:
            # Default: simple pipeline
            workflow = WorkflowTemplates.pipeline(
                stages=config.get("stages", ["stage1", "stage2"]),
                agent_ids=config.get("agent_ids"),
            )

        return workflow

    async def execute_workflow(
        self, request: ExecuteWorkflowRequest
    ) -> ExecuteWorkflowResponse:
        """워크플로우 실행"""
        logger.info(f"Executing workflow: {request.workflow_id}")

        # Get workflow
        workflow = self._workflows.get(request.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {request.workflow_id}")

        # Create execution ID
        execution_id = str(uuid.uuid4())

        # Create monitor
        monitor = WorkflowMonitor(
            workflow_id=request.workflow_id,
            total_nodes=len(workflow.nodes),
        )
        self._monitors[execution_id] = monitor

        # Start monitoring
        await monitor.start()

        try:
            # Execute workflow
            task = request.input_data.get("task", "")
            agents = request.input_data.get("agents", {})
            tools = request.input_data.get("tools", {})

            result = await workflow.execute(agents=agents, task=task, tools=tools)

            # Update analytics
            self._analytics.add_execution(
                workflow_id=request.workflow_id,
                node_states=monitor.get_all_node_states(),
                events=monitor.event_history,
            )

            # End monitoring
            await monitor.end(success=result.get("success", False))

            # Create response
            execution_time = monitor.get_status().get("elapsed_ms", 0) / 1000

            response = ExecuteWorkflowResponse(
                execution_id=execution_id,
                workflow_id=request.workflow_id,
                status="completed" if result.get("success") else "failed",
                result=result.get("final_outputs"),
                node_results=result.get("execution_history", []),
                execution_time=execution_time,
                metadata={"total_nodes": len(workflow.nodes), "stats": monitor.stats},
            )

            logger.info(f"Workflow executed: {execution_id} in {execution_time:.2f}s")
            return response

        except Exception as e:
            await monitor.end(success=False)
            logger.error(f"Workflow execution failed: {e}")

            return ExecuteWorkflowResponse(
                execution_id=execution_id,
                workflow_id=request.workflow_id,
                status="failed",
                error=str(e),
                metadata={},
            )

    async def monitor_workflow(
        self, request: MonitorWorkflowRequest
    ) -> MonitorWorkflowResponse:
        """워크플로우 실시간 모니터링"""
        logger.debug(f"Monitoring workflow execution: {request.execution_id}")

        # Get monitor
        monitor = self._monitors.get(request.execution_id)
        if not monitor:
            raise ValueError(f"Execution not found: {request.execution_id}")

        # Get current status
        status = monitor.get_status()
        node_states = monitor.get_all_node_states()

        # Find current node (running)
        current_node = None
        for node_id, state in node_states.items():
            if state.status.value == "running":
                current_node = node_id
                break

        # Nodes completed and pending
        nodes_completed = [
            nid for nid, state in node_states.items() if state.status.value == "completed"
        ]
        nodes_pending = [
            nid for nid, state in node_states.items() if state.status.value == "pending"
        ]

        # Create response
        response = MonitorWorkflowResponse(
            execution_id=request.execution_id,
            workflow_id=request.workflow_id,
            current_node=current_node,
            progress=status.get("progress_percent", 0.0) / 100,
            nodes_completed=nodes_completed,
            nodes_pending=nodes_pending,
            elapsed_time=status.get("elapsed_ms", 0) / 1000,
            metadata=status,
        )

        return response

    async def get_analytics(self, workflow_id: str) -> AnalyticsResponse:
        """워크플로우 분석"""
        logger.info(f"Generating analytics for workflow: {workflow_id}")

        # Get workflow
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Check if executions exist
        if workflow_id not in self._analytics.executions:
            return AnalyticsResponse(
                workflow_id=workflow_id,
                total_executions=0,
                avg_execution_time=0.0,
                success_rate=0.0,
                bottlenecks=[],
                agent_utilization={},
                cost_breakdown={},
            )

        # Find bottlenecks
        bottlenecks_analysis = self._analytics.find_bottlenecks(workflow_id)
        bottlenecks = [
            {
                "node_id": bn.node_id,
                "duration_ms": bn.duration_ms,
                "percentage": bn.percentage_of_total,
                "recommendation": bn.recommendation,
            }
            for bn in bottlenecks_analysis
        ]

        # Agent utilization
        utilization_stats = self._analytics.analyze_agent_utilization()
        agent_utilization = {
            agent_id: stats.success_rate for agent_id, stats in utilization_stats.items()
        }

        # Cost estimate
        cost_data = self._analytics.calculate_cost_estimate(workflow_id)
        cost_breakdown = cost_data.get("node_costs", {})

        # Recommendations
        recommendations = self._analytics.generate_optimization_recommendations(workflow_id)

        # Summary stats
        summary = self._analytics.get_summary_statistics()

        # Create response
        response = AnalyticsResponse(
            workflow_id=workflow_id,
            total_executions=summary.get("total_executions", 0),
            avg_execution_time=summary.get("avg_duration_ms", 0.0) / 1000,
            success_rate=summary.get("avg_success_rate", 0.0),
            bottlenecks=bottlenecks,
            agent_utilization=agent_utilization,
            cost_breakdown=cost_breakdown,
            recommendations=recommendations,
        )

        logger.info(f"Analytics generated for workflow: {workflow_id}")
        return response

    async def visualize_workflow(self, workflow_id: str) -> str:
        """워크플로우 시각화"""
        logger.debug(f"Visualizing workflow: {workflow_id}")

        # Get workflow
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        # Generate visualization
        builder = VisualBuilder(workflow)
        diagram = builder.build_diagram(style="box", show_config=True)

        return diagram

    async def get_templates(self) -> Dict[str, Any]:
        """사전 정의된 워크플로우 템플릿 목록"""
        templates = {
            "research_write": {
                "name": "Research & Write",
                "description": "Researcher → Writer → [Reviewer]",
                "params": ["researcher_id", "writer_id", "reviewer_id (optional)"],
            },
            "parallel": {
                "name": "Parallel Consensus",
                "description": "Multiple agents execute in parallel and aggregate results",
                "params": ["agent_ids", "aggregation (vote/consensus)"],
            },
            "hierarchical": {
                "name": "Hierarchical Delegation",
                "description": "Manager decomposes task → Workers execute → Manager synthesizes",
                "params": ["manager_id", "worker_ids"],
            },
            "debate": {
                "name": "Debate & Judge",
                "description": "Agents debate over multiple rounds, judge decides",
                "params": ["debater_ids", "judge_id", "rounds"],
            },
            "pipeline": {
                "name": "Sequential Pipeline",
                "description": "Sequential execution through multiple stages",
                "params": ["stages", "agent_ids"],
            },
        }

        return templates
