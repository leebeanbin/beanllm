"""
WorkflowGraph - 노드 기반 워크플로우 그래프
SOLID 원칙:
- SRP: 워크플로우 구조 정의 및 실행만 담당
- OCP: 새로운 노드 타입 추가 가능
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class NodeType(Enum):
    """워크플로우 노드 타입"""

    AGENT = "agent"  # 단일 Agent 실행
    TOOL = "tool"  # Tool 실행
    DECISION = "decision"  # 조건부 분기
    PARALLEL = "parallel"  # 병렬 실행
    SEQUENTIAL = "sequential"  # 순차 실행 그룹
    HIERARCHICAL = "hierarchical"  # 계층적 실행
    DEBATE = "debate"  # 토론
    MERGE = "merge"  # 결과 병합
    START = "start"  # 시작 노드
    END = "end"  # 종료 노드


class EdgeCondition(Enum):
    """엣지 조건"""

    ALWAYS = "always"  # 항상 실행
    ON_SUCCESS = "on_success"  # 성공 시
    ON_FAILURE = "on_failure"  # 실패 시
    CONDITIONAL = "conditional"  # 조건부 (함수 평가)


@dataclass
class WorkflowNode:
    """
    워크플로우 노드

    각 노드는 워크플로우의 한 단계를 나타냅니다.
    """

    node_id: str
    node_type: NodeType
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: tuple[int, int] = (0, 0)  # (x, y) for visualization
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "config": self.config,
            "position": self.position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        """딕셔너리에서 생성"""
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            config=data.get("config", {}),
            position=tuple(data.get("position", (0, 0))),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkflowEdge:
    """
    워크플로우 엣지 (노드 간 연결)
    """

    edge_id: str
    source: str  # source node_id
    target: str  # target node_id
    condition: EdgeCondition = EdgeCondition.ALWAYS
    condition_func: Optional[Callable[[Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_execute(self, context: Dict[str, Any]) -> bool:
        """
        엣지를 따라 실행할지 결정

        Args:
            context: 실행 컨텍스트 (이전 노드의 결과 등)

        Returns:
            bool: 실행 여부
        """
        if self.condition == EdgeCondition.ALWAYS:
            return True

        elif self.condition == EdgeCondition.ON_SUCCESS:
            return context.get("success", True)

        elif self.condition == EdgeCondition.ON_FAILURE:
            return not context.get("success", True)

        elif self.condition == EdgeCondition.CONDITIONAL:
            if self.condition_func:
                return self.condition_func(context)
            return True

        return True

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "condition": self.condition.value,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """노드 실행 결과"""

    node_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "node_id": self.node_id,
            "success": self.success,
            "output": str(self.output),
            "error": self.error,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class WorkflowGraph:
    """
    워크플로우 그래프

    노드와 엣지로 구성된 방향성 그래프 (DAG)

    Example:
        ```python
        # Create workflow
        workflow = WorkflowGraph(name="Research Pipeline")

        # Add nodes
        start = workflow.add_node(NodeType.START, "start")
        research = workflow.add_node(
            NodeType.AGENT,
            "researcher",
            config={"agent_id": "researcher"}
        )
        analyze = workflow.add_node(
            NodeType.AGENT,
            "analyzer",
            config={"agent_id": "analyzer"}
        )
        end = workflow.add_node(NodeType.END, "end")

        # Add edges
        workflow.add_edge(start, research)
        workflow.add_edge(research, analyze)
        workflow.add_edge(analyze, end)

        # Execute
        result = await workflow.execute(
            agents={"researcher": agent1, "analyzer": agent2},
            task="Research AI trends"
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
        visited = set()
        rec_stack = set()

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
        result = []

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
        **kwargs,
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
        """
        노드 실행

        Args:
            node: 실행할 노드

        Returns:
            ExecutionResult: 실행 결과
        """
        start_time = datetime.now()
        logger.debug(f"Executing node: {node.node_id} ({node.node_type.value})")

        try:
            if node.node_type == NodeType.START:
                output = self.current_state.get("task")
                success = True

            elif node.node_type == NodeType.END:
                # Get last output
                output = self.current_state.get("last_output", "Workflow completed")
                success = True

            elif node.node_type == NodeType.AGENT:
                # Execute agent
                agent_id = node.config.get("agent_id")
                agent = self.current_state["agents"].get(agent_id)

                if not agent:
                    raise ValueError(f"Agent not found: {agent_id}")

                input_data = self.current_state.get("task")
                result = await agent.run(input_data)
                output = result.answer
                success = True

            elif node.node_type == NodeType.TOOL:
                # Execute tool
                tool_id = node.config.get("tool_id")
                tool = self.current_state["tools"].get(tool_id)

                if not tool:
                    raise ValueError(f"Tool not found: {tool_id}")

                input_data = node.config.get("input", {})
                output = await tool.execute(**input_data)
                success = True

            elif node.node_type == NodeType.PARALLEL:
                # Parallel execution
                agent_ids = node.config.get("agent_ids", [])
                agents = [self.current_state["agents"][aid] for aid in agent_ids]

                tasks = [agent.run(self.current_state.get("task")) for agent in agents]
                results = await asyncio.gather(*tasks)
                output = [r.answer for r in results]
                success = True

            elif node.node_type == NodeType.SEQUENTIAL:
                # Sequential execution group
                output = await self._execute_sequential_node(node)
                success = True

            elif node.node_type == NodeType.DECISION:
                # Conditional branching
                output = await self._execute_decision_node(node)
                success = True

            elif node.node_type == NodeType.HIERARCHICAL:
                # Hierarchical execution (manager-worker pattern)
                output = await self._execute_hierarchical_node(node)
                success = True

            elif node.node_type == NodeType.DEBATE:
                # Multi-agent debate
                output = await self._execute_debate_node(node)
                success = True

            elif node.node_type == NodeType.MERGE:
                # Result merging
                output = await self._execute_merge_node(node)
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

    async def _execute_sequential_node(self, node: WorkflowNode) -> Any:
        """
        순차 실행 노드

        child_nodes에 지정된 노드들을 순차적으로 실행합니다.
        이전 노드의 출력이 다음 노드의 입력으로 전달됩니다.

        Config:
            child_nodes: List[str] - 순차 실행할 노드 ID 목록
            pass_output: bool - 이전 출력을 다음 입력으로 전달 (기본: True)

        Returns:
            최종 노드의 출력
        """
        child_node_ids = node.config.get("child_nodes", [])
        pass_output = node.config.get("pass_output", True)

        current_input = self.current_state.get("task")
        last_output = None

        for child_id in child_node_ids:
            child_node = self.nodes.get(child_id)
            if not child_node:
                logger.warning(f"Sequential: child node not found: {child_id}")
                continue

            # 이전 출력을 현재 task로 설정
            if pass_output and last_output is not None:
                self.current_state["task"] = last_output

            result = await self._execute_node(child_node)
            self.execution_history.append(result)

            if not result.success:
                logger.error(f"Sequential: child node {child_id} failed")
                return {"error": result.error, "failed_at": child_id}

            last_output = result.output
            self.current_state[child_id] = result.output

        # 원래 task 복원
        self.current_state["task"] = current_input

        return last_output

    async def _execute_decision_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """
        조건부 분기 노드

        조건 함수 또는 키워드 매칭을 통해 다음 실행 경로를 결정합니다.

        Config:
            condition: Callable[[Any], str] - 분기 키를 반환하는 함수
            condition_key: str - current_state에서 조건 값을 가져올 키
            branches: Dict[str, str] - {분기_키: 타겟_노드_ID}
            default: str - 기본 분기 노드 ID

        Returns:
            Dict with decision and next_node
        """
        condition_fn = node.config.get("condition")
        condition_key = node.config.get("condition_key")
        branches = node.config.get("branches", {})
        default_branch = node.config.get("default")

        # 조건 평가
        branch_key = None

        if callable(condition_fn):
            branch_key = condition_fn(self.current_state)
        elif condition_key:
            branch_key = str(self.current_state.get(condition_key, ""))
        else:
            # 마지막 출력에서 키워드 매칭
            last_output = str(self.current_state.get("last_output", ""))
            for key in branches:
                if key.lower() in last_output.lower():
                    branch_key = key
                    break

        # 분기 결정
        target_node_id = branches.get(branch_key) or default_branch

        logger.debug(f"Decision: branch_key={branch_key}, target={target_node_id}")

        return {
            "decision": branch_key,
            "next_node": target_node_id,
            "available_branches": list(branches.keys()),
        }

    async def _execute_hierarchical_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """
        계층적 실행 노드 (Manager-Worker 패턴)

        Manager가 task를 분해하고, Worker들이 병렬 실행 후,
        Manager가 결과를 종합합니다.

        Config:
            manager_id: str - Manager agent ID
            worker_ids: List[str] - Worker agent ID 목록
            decompose_prompt: str - Task 분해 프롬프트 템플릿
            synthesize_prompt: str - 결과 종합 프롬프트 템플릿
            max_workers: int - 최대 동시 실행 Worker 수 (기본: 전체)

        Returns:
            Dict with subtasks, worker_results, final_result
        """
        manager_id = node.config.get("manager_id")
        worker_ids = node.config.get("worker_ids", [])
        decompose_prompt = node.config.get(
            "decompose_prompt",
            "Break down this task into {n} subtasks:\n{task}",
        )
        synthesize_prompt = node.config.get(
            "synthesize_prompt",
            "Synthesize these results into a final answer:\n{results}",
        )
        max_workers = node.config.get("max_workers", len(worker_ids))

        manager = self.current_state["agents"].get(manager_id)
        if not manager:
            raise ValueError(f"Hierarchical: manager not found: {manager_id}")

        workers = [self.current_state["agents"].get(wid) for wid in worker_ids[:max_workers]]
        workers = [w for w in workers if w is not None]

        if not workers:
            raise ValueError("Hierarchical: no workers found")

        task = self.current_state.get("task", "")

        # 1. Manager가 task decomposition
        decompose_task = decompose_prompt.format(n=len(workers), task=task)
        decompose_result = await manager.run(decompose_task)
        subtasks_text = decompose_result.answer

        # 간단한 파싱: 줄바꿈으로 분리
        subtasks = [
            line.strip()
            for line in subtasks_text.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # subtask 수와 worker 수 맞추기
        if len(subtasks) < len(workers):
            subtasks.extend([subtasks[-1]] * (len(workers) - len(subtasks)))
        subtasks = subtasks[: len(workers)]

        logger.debug(f"Hierarchical: {len(subtasks)} subtasks for {len(workers)} workers")

        # 2. Workers 병렬 실행
        worker_tasks = [worker.run(subtask) for worker, subtask in zip(workers, subtasks)]
        worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        # 에러 처리
        processed_results = []
        for i, result in enumerate(worker_results):
            if isinstance(result, Exception):
                processed_results.append(f"Worker {i} failed: {result!s}")
            else:
                processed_results.append(result.answer)

        # 3. Manager가 synthesis
        results_text = "\n\n".join(f"[Worker {i+1}]\n{r}" for i, r in enumerate(processed_results))
        synthesize_task = synthesize_prompt.format(results=results_text)
        final_result = await manager.run(synthesize_task)

        return {
            "subtasks": subtasks,
            "worker_results": processed_results,
            "final_result": final_result.answer,
        }

    async def _execute_debate_node(self, node: WorkflowNode) -> Dict[str, Any]:
        """
        토론 노드 (Multi-Agent Debate)

        여러 에이전트가 N라운드 토론을 진행합니다.
        각 라운드에서 이전 라운드 결과를 참조합니다.

        Config:
            agent_ids: List[str] - 토론 참가 에이전트 ID 목록
            rounds: int - 토론 라운드 수 (기본: 3)
            judge_id: str - 판사 에이전트 ID (선택)
            debate_prompt: str - 토론 프롬프트 템플릿
            judge_prompt: str - 판정 프롬프트 템플릿

        Returns:
            Dict with rounds_history, final_consensus, judge_verdict
        """
        agent_ids = node.config.get("agent_ids", [])
        rounds = node.config.get("rounds", 3)
        judge_id = node.config.get("judge_id")
        debate_prompt = node.config.get(
            "debate_prompt",
            "Topic: {task}\n\nPrevious arguments:\n{history}\n\nProvide your argument:",
        )
        judge_prompt = node.config.get(
            "judge_prompt",
            "Based on this debate, provide a final verdict:\n{debate}",
        )

        agents = [self.current_state["agents"].get(aid) for aid in agent_ids]
        agents = [a for a in agents if a is not None]

        if len(agents) < 2:
            raise ValueError("Debate: at least 2 agents required")

        task = self.current_state.get("task", "")
        debate_history: List[List[str]] = []

        # N라운드 토론
        for round_num in range(rounds):
            # 이전 라운드 히스토리 포맷
            history_text = ""
            if debate_history:
                for prev_round, round_results in enumerate(debate_history):
                    history_text += f"\n--- Round {prev_round + 1} ---\n"
                    for i, result in enumerate(round_results):
                        history_text += f"Agent {i + 1}: {result}\n"

            # 현재 라운드 실행
            round_prompt = debate_prompt.format(task=task, history=history_text)
            round_tasks = [agent.run(round_prompt) for agent in agents]
            round_results = await asyncio.gather(*round_tasks, return_exceptions=True)

            # 결과 처리
            processed_results = []
            for result in round_results:
                if isinstance(result, Exception):
                    processed_results.append(f"(error: {result!s})")
                else:
                    processed_results.append(result.answer)

            debate_history.append(processed_results)
            logger.debug(f"Debate round {round_num + 1}/{rounds} completed")

        # 최종 합의 (마지막 라운드 결과)
        final_round = debate_history[-1] if debate_history else []

        # 판사가 있으면 판정
        judge_verdict = None
        if judge_id:
            judge = self.current_state["agents"].get(judge_id)
            if judge:
                full_debate = ""
                for round_num, round_results in enumerate(debate_history):
                    full_debate += f"\n=== Round {round_num + 1} ===\n"
                    for i, result in enumerate(round_results):
                        full_debate += f"Agent {i + 1}: {result}\n"

                judge_task = judge_prompt.format(debate=full_debate)
                judge_result = await judge.run(judge_task)
                judge_verdict = judge_result.answer

        return {
            "rounds": len(debate_history),
            "rounds_history": debate_history,
            "final_arguments": final_round,
            "judge_verdict": judge_verdict,
        }

    async def _execute_merge_node(self, node: WorkflowNode) -> Any:
        """
        병합 노드

        병렬 실행된 결과들을 다양한 전략으로 집계합니다.

        Config:
            input_nodes: List[str] - 입력 노드 ID 목록
            strategy: str - 집계 전략 (vote, consensus, first, all, custom)
            custom_merge: Callable - 커스텀 병합 함수 (strategy=custom일 때)
            weights: Dict[str, float] - 노드별 가중치 (vote 전략)

        Returns:
            병합된 결과
        """
        input_nodes = node.config.get("input_nodes", [])
        strategy = node.config.get("strategy", "all")
        custom_merge = node.config.get("custom_merge")
        weights = node.config.get("weights", {})

        # 입력 수집
        collected = []
        for input_id in input_nodes:
            if input_id in self.current_state:
                collected.append(self.current_state[input_id])

        if not collected:
            logger.warning("Merge: no inputs collected")
            return None

        # 전략별 병합
        if strategy == "vote":
            # 가중치 투표
            vote_counts: Counter[str] = Counter()
            for i, item in enumerate(collected):
                node_id = input_nodes[i] if i < len(input_nodes) else str(i)
                weight = weights.get(node_id, 1.0)
                # 문자열 해시 또는 직접 사용
                key = str(item) if not isinstance(item, str) else item
                vote_counts[key] += weight

            if vote_counts:
                winner = vote_counts.most_common(1)[0][0]
                return {
                    "strategy": "vote",
                    "winner": winner,
                    "vote_counts": dict(vote_counts),
                }
            return None

        elif strategy == "consensus":
            # 합의: 모든 결과가 같아야 함
            unique = set(str(item) for item in collected)
            if len(unique) == 1:
                return {
                    "strategy": "consensus",
                    "agreed": True,
                    "result": collected[0],
                }
            return {
                "strategy": "consensus",
                "agreed": False,
                "results": collected,
                "unique_count": len(unique),
            }

        elif strategy == "first":
            # 첫 번째 성공 결과
            for item in collected:
                if item is not None:
                    return {
                        "strategy": "first",
                        "result": item,
                    }
            return None

        elif strategy == "custom" and callable(custom_merge):
            # 커스텀 병합 함수
            return {
                "strategy": "custom",
                "result": custom_merge(collected),
            }

        else:  # "all" (기본)
            return {
                "strategy": "all",
                "results": collected,
                "count": len(collected),
            }

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
