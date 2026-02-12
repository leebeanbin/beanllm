"""
WorkflowGraph - 노드 기반 워크플로우 그래프
SOLID 원칙:
- SRP: 워크플로우 구조 정의 및 실행만 담당
- OCP: 새로운 노드 타입 추가 가능 (workflow_executors.py)

타입/모델: workflow_types.py
실행 전략: workflow_executors.py
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from beanllm.domain.orchestrator.workflow_executors import (
    execute_debate_node,
    execute_decision_node,
    execute_hierarchical_node,
    execute_merge_node,
    execute_sequential_node,
)
from beanllm.domain.orchestrator.workflow_types import (
    EdgeCondition,
    ExecutionResult,
    NodeType,
    WorkflowEdge,
    WorkflowNode,
)
from beanllm.utils.logging import get_logger

# Re-export for backward compatibility
__all__ = [
    "NodeType",
    "EdgeCondition",
    "WorkflowNode",
    "WorkflowEdge",
    "ExecutionResult",
    "WorkflowGraph",
]

logger = get_logger(__name__)


class WorkflowGraph:
    """
    워크플로우 그래프

    노드와 엣지로 구성된 방향성 그래프 (DAG)

    Example:
        ```python
        workflow = WorkflowGraph(name="Research Pipeline")
        start = workflow.add_node(NodeType.START, "start")
        research = workflow.add_node(NodeType.AGENT, "researcher",
                                     config={"agent_id": "researcher"})
        end = workflow.add_node(NodeType.END, "end")
        workflow.add_edge(start, research)
        workflow.add_edge(research, end)
        result = await workflow.execute(
            agents={"researcher": agent1}, task="Research AI trends"
        )
        ```
    """

    def __init__(
        self,
        name: str = "Workflow",
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            name: 워크플로우 이름
            workflow_id: 고유 ID (None이면 자동 생성)
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, WorkflowEdge] = {}
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        self.reverse_adjacency: Dict[str, List[str]] = {}  # target -> [edge_ids]

        # Execution state
        self.execution_history: List[ExecutionResult] = []
        self.current_state: Dict[str, Any] = {}

    def add_node(
        self,
        node_type: NodeType,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        position: Optional[tuple[int, int]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        노드 추가

        Args:
            node_type: 노드 타입
            name: 노드 이름
            config: 노드 설정
            position: 시각화 위치 (x, y)
            node_id: 노드 ID (None이면 자동 생성)

        Returns:
            str: 노드 ID
        """
        node_id = node_id or f"{name}_{str(uuid.uuid4())[:8]}"

        node = WorkflowNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            config=config or {},
            position=position or (0, 0),
        )

        self.nodes[node_id] = node
        self.adjacency[node_id] = []
        self.reverse_adjacency[node_id] = []

        logger.debug(f"Added node: {node_id} ({node_type.value})")

        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        condition: EdgeCondition = EdgeCondition.ALWAYS,
        condition_func: Optional[Callable[[Any], bool]] = None,
        edge_id: Optional[str] = None,
    ) -> str:
        """
        엣지 추가

        Args:
            source: 소스 노드 ID
            target: 타겟 노드 ID
            condition: 엣지 조건
            condition_func: 조건 함수 (condition=CONDITIONAL일 때)
            edge_id: 엣지 ID (None이면 자동 생성)

        Returns:
            str: 엣지 ID

        Raises:
            ValueError: 노드가 존재하지 않거나 사이클이 생성될 경우
        """
        if source not in self.nodes:
            raise ValueError(f"Source node not found: {source}")
        if target not in self.nodes:
            raise ValueError(f"Target node not found: {target}")

        edge_id = edge_id or f"{source}_to_{target}_{str(uuid.uuid4())[:8]}"

        edge = WorkflowEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            condition=condition,
            condition_func=condition_func,
        )

        self.edges[edge_id] = edge
        self.adjacency[source].append(edge_id)
        self.reverse_adjacency[target].append(edge_id)

        # Check for cycles
        if self._has_cycle():
            # Rollback
            del self.edges[edge_id]
            self.adjacency[source].remove(edge_id)
            self.reverse_adjacency[target].remove(edge_id)
            raise ValueError(f"Adding edge {source} -> {target} creates a cycle")

        logger.debug(f"Added edge: {source} -> {target}")

        return edge_id

    def _has_cycle(self) -> bool:
        """사이클 감지 (DFS)"""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for edge_id in self.adjacency.get(node_id, []):
                edge = self.edges[edge_id]
                neighbor = edge.target

                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def get_start_nodes(self) -> List[str]:
        """시작 노드 찾기 (incoming edge가 없는 노드)"""
        start_nodes = []
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.START:
                start_nodes.append(node_id)
            elif not self.reverse_adjacency.get(node_id):
                start_nodes.append(node_id)

        return start_nodes

    def get_end_nodes(self) -> List[str]:
        """종료 노드 찾기 (outgoing edge가 없는 노드)"""
        end_nodes = []
        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.END:
                end_nodes.append(node_id)
            elif not self.adjacency.get(node_id):
                end_nodes.append(node_id)

        return end_nodes

    def get_topological_order(self) -> List[str]:
        """위상 정렬 (Topological Sort)"""
        in_degree = {node_id: 0 for node_id in self.nodes}

        for node_id in self.nodes:
            for edge_id in self.adjacency[node_id]:
                edge = self.edges[edge_id]
                in_degree[edge.target] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result: List[str] = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for edge_id in self.adjacency[node_id]:
                edge = self.edges[edge_id]
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle")

        return result

    async def execute(
        self,
        agents: Dict[str, Any],
        task: str,
        tools: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        워크플로우 실행

        Args:
            agents: Agent 딕셔너리 {agent_id: Agent}
            task: 초기 작업
            tools: Tool 딕셔너리 {tool_id: Tool}
            **kwargs: 추가 파라미터

        Returns:
            Dict: 실행 결과
        """
        logger.info(f"Executing workflow: {self.name}")

        self.execution_history = []
        self.current_state = {"task": task, "agents": agents, "tools": tools or {}}

        # Topological order로 실행
        try:
            order = self.get_topological_order()
        except ValueError as e:
            return {"success": False, "error": str(e)}

        # Execute nodes in order
        for node_id in order:
            node = self.nodes[node_id]

            # Check if all prerequisites are met
            can_execute = True
            for edge_id in self.reverse_adjacency.get(node_id, []):
                edge = self.edges[edge_id]
                if not edge.should_execute(self.current_state):
                    can_execute = False
                    break

            if not can_execute:
                continue

            # Execute node
            result = await self._execute_node(node)
            self.execution_history.append(result)

            # Update state
            self.current_state[node_id] = result.output
            self.current_state["success"] = result.success

            if not result.success and node.config.get("fail_fast", False):
                logger.error(f"Node {node_id} failed, stopping execution")
                break

        # Compile final result
        end_nodes = self.get_end_nodes()
        final_outputs = [self.current_state.get(nid) for nid in end_nodes]

        return {
            "success": all(r.success for r in self.execution_history),
            "final_outputs": final_outputs,
            "execution_history": [r.to_dict() for r in self.execution_history],
            "workflow_id": self.workflow_id,
            "workflow_name": self.name,
        }

    async def _execute_node(self, node: WorkflowNode) -> ExecutionResult:
        """노드 실행 (타입별 실행 전략 위임)"""
        start_time = datetime.now()
        logger.debug(f"Executing node: {node.node_id} ({node.node_type.value})")

        try:
            if node.node_type == NodeType.START:
                output = self.current_state.get("task")
                success = True

            elif node.node_type == NodeType.END:
                output = self.current_state.get("last_output", "Workflow completed")
                success = True

            elif node.node_type == NodeType.AGENT:
                agent_id = node.config.get("agent_id")
                agent = self.current_state["agents"].get(agent_id)
                if not agent:
                    raise ValueError(f"Agent not found: {agent_id}")
                input_data = self.current_state.get("task")
                result = await agent.run(input_data)
                output = result.answer
                success = True

            elif node.node_type == NodeType.TOOL:
                tool_id = node.config.get("tool_id")
                tool = self.current_state["tools"].get(tool_id)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_id}")
                input_data = node.config.get("input", {})
                output = await tool.execute(**input_data)
                success = True

            elif node.node_type == NodeType.PARALLEL:
                agent_ids = node.config.get("agent_ids", [])
                agents = [self.current_state["agents"][aid] for aid in agent_ids]
                tasks = [agent.run(self.current_state.get("task")) for agent in agents]
                results = await asyncio.gather(*tasks)
                output = [r.answer for r in results]
                success = True

            elif node.node_type == NodeType.SEQUENTIAL:
                output = await execute_sequential_node(
                    node,
                    self.nodes,
                    self.current_state,
                    self._execute_node,
                    self.execution_history,
                )
                success = True

            elif node.node_type == NodeType.DECISION:
                output = await execute_decision_node(node, self.current_state)
                success = True

            elif node.node_type == NodeType.HIERARCHICAL:
                output = await execute_hierarchical_node(node, self.current_state)
                success = True

            elif node.node_type == NodeType.DEBATE:
                output = await execute_debate_node(node, self.current_state)
                success = True

            elif node.node_type == NodeType.MERGE:
                output = await execute_merge_node(node, self.current_state)
                success = True

            else:
                output = f"Node type {node.node_type.value} not yet implemented"
                success = False

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            return ExecutionResult(
                node_id=node.node_id,
                success=success,
                output=output,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
            )

        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            logger.error(f"Node {node.node_id} failed: {e}")

            return ExecutionResult(
                node_id=node.node_id,
                success=False,
                output=None,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
            )

    def to_dict(self) -> Dict[str, Any]:
        """워크플로우를 딕셔너리로 변환"""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {eid: edge.to_dict() for eid, edge in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowGraph":
        """딕셔너리에서 워크플로우 생성"""
        workflow = cls(
            name=data["name"],
            workflow_id=data["workflow_id"],
        )

        # Add nodes
        for node_data in data["nodes"].values():
            node = WorkflowNode.from_dict(node_data)
            workflow.nodes[node.node_id] = node
            workflow.adjacency[node.node_id] = []
            workflow.reverse_adjacency[node.node_id] = []

        # Add edges
        for edge_data in data["edges"].values():
            edge = WorkflowEdge(
                edge_id=edge_data["edge_id"],
                source=edge_data["source"],
                target=edge_data["target"],
                condition=EdgeCondition(edge_data["condition"]),
                metadata=edge_data.get("metadata", {}),
            )
            workflow.edges[edge.edge_id] = edge
            workflow.adjacency[edge.source].append(edge.edge_id)
            workflow.reverse_adjacency[edge.target].append(edge.edge_id)

        return workflow
