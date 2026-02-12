"""
Workflow Executors - 노드 타입별 실행 전략

각 노드 타입(Sequential, Decision, Hierarchical, Debate, Merge)의
실행 로직을 독립 함수로 분리하여 OCP(Open-Closed Principle) 준수.
새 노드 타입 추가 시 이 파일에 함수만 추가하면 됨.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Dict, List

from beanllm.domain.orchestrator.workflow_types import (
    ExecutionResult,
    WorkflowNode,
)
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


async def execute_sequential_node(
    node: WorkflowNode,
    nodes: Dict[str, WorkflowNode],
    current_state: Dict[str, Any],
    execute_node_fn: Any,
    execution_history: List[ExecutionResult],
) -> Any:
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

    current_input = current_state.get("task")
    last_output = None

    for child_id in child_node_ids:
        child_node = nodes.get(child_id)
        if not child_node:
            logger.warning(f"Sequential: child node not found: {child_id}")
            continue

        # 이전 출력을 현재 task로 설정
        if pass_output and last_output is not None:
            current_state["task"] = last_output

        result = await execute_node_fn(child_node)
        execution_history.append(result)

        if not result.success:
            logger.error(f"Sequential: child node {child_id} failed")
            return {"error": result.error, "failed_at": child_id}

        last_output = result.output
        current_state[child_id] = result.output

    # 원래 task 복원
    current_state["task"] = current_input

    return last_output


async def execute_decision_node(
    node: WorkflowNode,
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
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
        branch_key = condition_fn(current_state)
    elif condition_key:
        branch_key = str(current_state.get(condition_key, ""))
    else:
        # 마지막 출력에서 키워드 매칭
        last_output = str(current_state.get("last_output", ""))
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


async def execute_hierarchical_node(
    node: WorkflowNode,
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
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

    manager = current_state["agents"].get(manager_id)
    if not manager:
        raise ValueError(f"Hierarchical: manager not found: {manager_id}")

    workers = [current_state["agents"].get(wid) for wid in worker_ids[:max_workers]]
    workers = [w for w in workers if w is not None]

    if not workers:
        raise ValueError("Hierarchical: no workers found")

    task = current_state.get("task", "")

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
        if isinstance(result, BaseException):
            processed_results.append(f"Worker {i} failed: {result!s}")
        else:
            processed_results.append(result.answer)

    # 3. Manager가 synthesis
    results_text = "\n\n".join(f"[Worker {i + 1}]\n{r}" for i, r in enumerate(processed_results))
    synthesize_task = synthesize_prompt.format(results=results_text)
    final_result = await manager.run(synthesize_task)

    return {
        "subtasks": subtasks,
        "worker_results": processed_results,
        "final_result": final_result.answer,
    }


async def execute_debate_node(
    node: WorkflowNode,
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
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

    agents = [current_state["agents"].get(aid) for aid in agent_ids]
    agents = [a for a in agents if a is not None]

    if len(agents) < 2:
        raise ValueError("Debate: at least 2 agents required")

    task = current_state.get("task", "")
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
        judge = current_state["agents"].get(judge_id)
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


async def execute_merge_node(
    node: WorkflowNode,
    current_state: Dict[str, Any],
) -> Any:
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
        if input_id in current_state:
            collected.append(current_state[input_id])

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
