"""
Coordination Strategies - Agent 조정 전략들
"""

import asyncio
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.protocols import AgentProtocol, AgentResultProtocol

logger = get_logger(__name__)

# 프롬프트 토큰 폭증 방지용 상수
_MAX_ANSWER_CHARS = 2_000  # 에이전트 답변 하나당 최대 문자 수
_MAX_CONTEXT_CHARS = 6_000  # AutonomousPlanning 누적 컨텍스트 최대 문자 수
_MAX_SYNTHESIS_CHARS = 8_000  # Hierarchical 종합 프롬프트 최대 문자 수


def _truncate(text: str, max_chars: int) -> str:
    """문자열을 *max_chars* 이하로 잘라 반환. 초과 시 '...[truncated]' 표기."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"


class CoordinationStrategy(ABC):
    """조정 전략 베이스 클래스"""

    @abstractmethod
    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """전략 실행"""
        pass


class SequentialStrategy(CoordinationStrategy):
    """
    순차 실행 전략

    Mathematical Foundation:
        Function composition:
        result = fₙ ∘ fₙ₋₁ ∘ ... ∘ f₂ ∘ f₁(task)

        Time Complexity: O(Σ Tᵢ) - 모든 agent 시간의 합
    """

    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """순차 실행. 에이전트 실패 시 즉시 전파."""
        run_id = str(uuid.uuid4())
        results = []
        timings: List[float] = []
        current_input = task

        for i, agent in enumerate(agents):
            logger.info(f"Sequential[{run_id}]: Agent {i + 1}/{len(agents)} executing")
            t0 = time.monotonic()
            result = await agent.run(current_input)
            timings.append(time.monotonic() - t0)

            if not result.success:
                raise RuntimeError(
                    f"Sequential step {i + 1}/{len(agents)} failed "
                    f"(agent={getattr(agent, 'name', i)}): {result.error}"
                )

            if result.answer is None:
                raise RuntimeError(
                    f"Sequential step {i + 1}/{len(agents)} returned None answer "
                    f"(agent={getattr(agent, 'name', i)})"
                )

            results.append(result)
            current_input = result.answer

        return {
            "final_result": results[-1].answer if results else None,
            "intermediate_results": [r.answer for r in results],
            "all_steps": results,
            "strategy": "sequential",
            "run_id": run_id,
            "step_timings_s": timings,
            "total_time_s": sum(timings),
        }


class ParallelStrategy(CoordinationStrategy):
    """
    병렬 실행 전략

    Mathematical Foundation:
        Parallel execution:
        result = {f₁(task), f₂(task), ..., fₙ(task)} executed concurrently

        Speedup: S = T_sequential / T_parallel
        Ideal: S = n (number of agents)

        Time Complexity: O(max(T₁, T₂, ..., Tₙ))
    """

    def __init__(self, aggregation: str = "vote") -> None:
        """
        Args:
            aggregation: 결과 집계 방법
                - "vote": 투표 (다수결)
                - "consensus": 합의 (모두 동의)
                - "first": 첫 번째 완료
                - "all": 모든 결과 반환
        """
        self.aggregation = aggregation

    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """병렬 실행. 부분 실패를 허용하고 성공한 결과로 집계."""
        run_id = str(uuid.uuid4())
        logger.info(f"Parallel[{run_id}]: Executing {len(agents)} agents concurrently")

        tasks = [asyncio.create_task(agent.run(task)) for agent in agents]
        t0 = time.monotonic()

        if self.aggregation == "first":
            return await self._execute_first(tasks, run_id, len(agents), t0)

        return await self._execute_all(tasks, run_id, t0)

    async def _execute_first(
        self,
        tasks: List[asyncio.Task],  # type: ignore[type-arg]
        run_id: str,
        total: int,
        t0: float,
    ) -> Dict[str, Any]:
        """첫 번째로 *성공*한 에이전트의 결과를 반환."""
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # 취소 후 반드시 cleanup — 누수 방지
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        completed_task = list(done)[0]
        exc = completed_task.exception()
        if exc is not None:
            # 첫 완료 태스크가 예외 → 나머지 중 성공한 것이 없으므로 raise
            raise RuntimeError(f"Parallel-first: completed task raised {exc!r}") from exc

        result = completed_task.result()
        return {
            "final_result": result.answer,
            "strategy": "parallel-first",
            "completed": 1,
            "total": total,
            "total_time_s": time.monotonic() - t0,
            "run_id": run_id,
        }

    async def _execute_all(
        self,
        tasks: List[asyncio.Task],  # type: ignore[type-arg]
        run_id: str,
        t0: float,
    ) -> Dict[str, Any]:
        """모든 에이전트를 실행하고 성공한 결과로 집계."""
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        successes: List[Any] = []
        failures: List[BaseException] = []
        for r in raw:
            if isinstance(r, BaseException):
                failures.append(r)
            else:
                successes.append(r)

        if failures:
            logger.warning(
                f"Parallel[{run_id}]: {len(failures)}/{len(raw)} agents failed — "
                f"proceeding with {len(successes)} results. "
                f"First error: {failures[0]!r}"
            )

        if not successes:
            raise RuntimeError(
                f"Parallel[{run_id}]: All {len(raw)} agents failed. "
                f"First error: {failures[0]!r}"
            ) from failures[0]

        answers = [r.answer for r in successes]

        base = {
            "all_answers": answers,
            "run_id": run_id,
            "total_time_s": time.monotonic() - t0,
            "agents_failed": len(failures),
            "agents_succeeded": len(successes),
        }

        if self.aggregation == "vote":
            vote_counts = Counter(answers)
            final_answer = vote_counts.most_common(1)[0][0]
            return {
                **base,
                "final_result": final_answer,
                "vote_counts": dict(vote_counts),
                "strategy": "parallel-vote",
                "agreement_rate": vote_counts[final_answer] / len(answers),
            }

        if self.aggregation == "consensus":
            consensus = len(set(answers)) == 1
            return {
                **base,
                "final_result": answers[0] if consensus else None,
                "consensus": consensus,
                "strategy": "parallel-consensus",
            }

        # "all"
        return {
            **base,
            "final_result": answers,
            "all_results": successes,
            "strategy": "parallel-all",
        }


class HierarchicalStrategy(CoordinationStrategy):
    """
    계층적 실행 전략

    Mathematical Foundation:
        Tree structure:
        - Root: Manager agent
        - Leaves: Worker agents

        manager ─┬─ worker₁
                 ├─ worker₂
                 └─ worker₃

        Time: O(d × T_max) where d=depth, T_max=max agent time
    """

    def __init__(self, manager_agent: "AgentProtocol") -> None:
        self.manager = manager_agent

    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """계층적 실행. 워커 부분 실패를 허용."""
        run_id = str(uuid.uuid4())
        logger.info(f"Hierarchical[{run_id}]: Manager delegating to {len(agents)} workers")
        t0 = time.monotonic()

        # 1. Manager가 작업 분해
        delegation_prompt = (
            f"You are a manager. Break down this task into subtasks for {len(agents)} workers.\n\n"
            f"Task: {task}\n\n"
            f'Return a JSON list of subtasks:\n{{"subtasks": ["subtask1", "subtask2", ...]}}'
        )

        delegation_result = await self.manager.run(delegation_prompt)
        json_match = re.search(r"\{.*\}", delegation_result.answer, re.DOTALL)
        if json_match:
            try:
                subtasks_data = json.loads(json_match.group())
                subtasks = subtasks_data.get("subtasks", [])
            except json.JSONDecodeError:
                logger.warning(
                    f"Hierarchical[{run_id}]: JSON parse failed for subtasks; using task as-is for all workers"
                )
                subtasks = [task] * len(agents)
        else:
            logger.warning(
                f"Hierarchical[{run_id}]: No JSON found in delegation response; using task as-is for all workers"
            )
            subtasks = [task] * len(agents)

        # 2. Workers 병렬 실행 (부분 실패 허용)
        worker_coros = [agent.run(subtask) for agent, subtask in zip(agents, subtasks)]
        raw_results = await asyncio.gather(*worker_coros, return_exceptions=True)

        worker_answers: List[str] = []
        failed_workers: List[int] = []
        for idx, r in enumerate(raw_results):
            if isinstance(r, BaseException):
                logger.warning(f"Hierarchical[{run_id}]: Worker {idx + 1} failed: {r!r}")
                failed_workers.append(idx)
                worker_answers.append(f"[Worker {idx + 1} failed: {r}]")
            else:
                worker_answers.append(r.answer)

        # 3. Manager가 결과 종합 (토큰 폭증 방지: 각 답변 자름)
        truncated_answers = [_truncate(ans, _MAX_ANSWER_CHARS) for ans in worker_answers]
        joined = "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(truncated_answers))
        joined = _truncate(joined, _MAX_SYNTHESIS_CHARS)

        synthesis_prompt = (
            "You are a manager. Synthesize the results from your workers into a final answer.\n\n"
            f"Original Task: {task}\n\nWorker Results:\n{joined}\n\nProvide a comprehensive final answer:"
        )

        final_result = await self.manager.run(synthesis_prompt)

        return {
            "final_result": final_result.answer,
            "subtasks": subtasks,
            "worker_results": worker_answers,
            "strategy": "hierarchical",
            "run_id": run_id,
            "manager_steps": len(delegation_result.steps) + len(final_result.steps),
            "total_workers": len(agents),
            "failed_workers": failed_workers,
            "total_time_s": time.monotonic() - t0,
        }


class AutonomousPlanningStrategy(CoordinationStrategy):
    """
    자율 계획 전략 (Plan-and-Execute + Self-Healing)

    1. Planner가 작업을 분석하고 단계별 계획(Tasks) 수립
    2. 사용자 승인 대기 (선택 사항)
    3. 각 단계를 적절한 Agent에게 할당하여 실행
    4. 실패 시 Planner가 재계획 (max_replans 회 제한)
    """

    def __init__(
        self,
        planner_agent: "AgentProtocol",
        on_plan_ready: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    ) -> None:
        """
        Args:
            planner_agent: 계획을 수립할 에이전트
            on_plan_ready: 계획 수립 완료 시 호출될 콜백 (승인/수정용)
        """
        self.planner = planner_agent
        self.on_plan_ready = on_plan_ready

    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """계획 수립 및 실행."""
        run_id = str(uuid.uuid4())
        logger.info(f"AutonomousPlanning[{run_id}]: Generating plan...")
        t0 = time.monotonic()

        # 1. 계획 수립
        agent_names = [getattr(a, "name", f"agent_{i}") for i, a in enumerate(agents)]
        plan_prompt = (
            "You are a Strategic Planner. Break down the following task into a structured plan.\n"
            "Each step should have a 'title', 'description', and 'assigned_agent_id'.\n\n"
            f"Task: {task}\n\n"
            f"Available Agents:\n{chr(10).join(f'- {n}' for n in agent_names)}\n\n"
            "Return a JSON object:\n"
            '{\n  "plan": [\n    {"title": "Step 1", "description": "...", "assigned_agent_id": "..."},\n    ...\n  ]\n}'
        )

        plan_result = await self.planner.run(plan_prompt)
        json_match = re.search(r"\{.*\}", plan_result.answer, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group()).get("plan", [])
            except json.JSONDecodeError:
                plan = [
                    {"title": "Execution", "description": task, "assigned_agent_id": agent_names[0]}
                ]
        else:
            plan = [
                {"title": "Execution", "description": task, "assigned_agent_id": agent_names[0]}
            ]

        # 2. 콜백 호출 (승인 대기 등) — CancelledError는 그대로 전파
        if self.on_plan_ready:
            try:
                if asyncio.iscoroutinefunction(self.on_plan_ready):
                    await self.on_plan_ready(plan)
                else:
                    self.on_plan_ready(plan)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"AutonomousPlanning[{run_id}]: on_plan_ready callback failed: {e!r}"
                )

        # 3. 계획 실행 (Self-Healing 루프)
        results: List[Dict[str, Any]] = []
        agent_map = {n: a for n, a in zip(agent_names, agents)}
        max_replans = kwargs.get("max_replans", 2)
        replan_count = 0
        aborted = False
        current_step_idx = 0

        while current_step_idx < len(plan):
            step = plan[current_step_idx]
            title = step.get("title", "Untitled Step")
            description = step.get("description", "")
            agent_id = step.get("assigned_agent_id", agent_names[0])
            agent = agent_map.get(agent_id) or agents[0]

            logger.info(
                f"AutonomousPlanning[{run_id}]: Step {current_step_idx + 1} '{title}' → {agent_id}"
            )

            # 누적 컨텍스트 (토큰 폭증 방지: 최근 결과만 유지)
            context_parts = [
                f"- {r['title']}: {_truncate(r['result'], _MAX_ANSWER_CHARS)}" for r in results
            ]
            context_str = _truncate("\n".join(context_parts), _MAX_CONTEXT_CHARS)
            step_prompt = (
                f"Original Task: {task}\n\n"
                f"Previous Results:\n{context_str or '(none yet)'}\n\n"
                f"Current Step: {title}\nDescription: {description}\n\nExecute this step."
            )

            try:
                result = await agent.run(step_prompt)
                if not result.success:
                    raise RuntimeError(result.error or "Agent reported failure")

                results.append(
                    {
                        "title": title,
                        "agent_id": agent_id,
                        "result": result.answer,
                        "full_result": result,
                    }
                )
                current_step_idx += 1

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"AutonomousPlanning[{run_id}]: Step '{title}' failed: {e!r}")

                if replan_count < max_replans:
                    replan_count += 1
                    logger.info(
                        f"AutonomousPlanning[{run_id}]: Re-planning "
                        f"(attempt {replan_count}/{max_replans})..."
                    )
                    replan_prompt = (
                        f"Task failure detected!\n"
                        f"Original Task: {task}\n"
                        f"Completed so far: {[r['title'] for r in results]}\n"
                        f"Failed Step: {title}\nError: {e}\n\n"
                        f"Current Plan: {plan}\n\n"
                        "Revise the REMAINING plan to recover. "
                        'Return JSON: {"plan": [...]}'
                    )
                    try:
                        replan_result = await self.planner.run(replan_prompt)
                        re_match = re.search(r"\{.*\}", replan_result.answer, re.DOTALL)
                        if re_match:
                            new_steps = json.loads(re_match.group()).get("plan", [])
                            if new_steps:
                                plan = plan[:current_step_idx] + new_steps
                                logger.info(
                                    f"AutonomousPlanning[{run_id}]: Plan revised — "
                                    f"{[s['title'] for s in new_steps]}"
                                )
                                continue
                    except asyncio.CancelledError:
                        raise
                    except Exception as replan_err:
                        logger.error(
                            f"AutonomousPlanning[{run_id}]: Replanning itself failed: {replan_err!r}"
                        )

                logger.error(
                    f"AutonomousPlanning[{run_id}]: Max replans reached or replanning failed. Aborting."
                )
                aborted = True
                break

        # 4. 최종 요약
        exec_summary = "\n".join(
            f"- {r['title']}: {_truncate(r['result'], _MAX_ANSWER_CHARS)}" for r in results
        )
        exec_summary = _truncate(exec_summary, _MAX_SYNTHESIS_CHARS)
        summary_prompt = (
            "As the Strategic Planner, summarize the final outcome.\n\n"
            f"Original Task: {task}\n\nExecution Results:\n{exec_summary}\n\n"
            "Provide a comprehensive final answer:"
        )
        final_summary = await self.planner.run(summary_prompt)

        return {
            "final_result": final_summary.answer,
            "plan": plan,
            "execution_results": results,
            "strategy": "autonomous_planning",
            "run_id": run_id,
            "replan_count": replan_count,
            "completed_steps": len(results),
            "total_steps": len(plan),
            "aborted": aborted,
            "total_time_s": time.monotonic() - t0,
        }


class ReflectiveStrategy(CoordinationStrategy):
    """
    자기 성찰 및 상호 검증 전략 (Reflective & Self-Correction)

    1. Agent A가 답변 생성
    2. Agent B(Reviewer)가 답변 검토 및 피드백 제공
    3. Agent A가 피드백을 반영하여 답변 수정
    4. 만족할 때까지 반복 (최대 rounds)
    """

    def __init__(self, reviewer_agent: "AgentProtocol", rounds: int = 2) -> None:
        self.reviewer = reviewer_agent
        self.rounds = rounds

    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """성찰 기반 실행."""
        run_id = str(uuid.uuid4())
        primary_agent = agents[0]
        history: List[Dict[str, Any]] = []
        t0 = time.monotonic()

        logger.info(f"Reflective[{run_id}]: Generating initial answer...")
        current_answer = await primary_agent.run(task)
        history.append({"role": "primary", "content": current_answer.answer, "round": 0})

        for r in range(1, self.rounds + 1):
            logger.info(f"Reflective[{run_id}] Round {r}/{self.rounds}: Reviewing...")

            review_prompt = (
                f"Task: {task}\n"
                f"Answer to Review: {_truncate(current_answer.answer, _MAX_ANSWER_CHARS)}\n\n"
                "Critique the answer. Find errors, missing points, or areas for improvement.\n"
                "If the answer is perfect, start your response with 'PASSED'."
            )
            review_result = await self.reviewer.run(review_prompt)
            history.append({"role": "reviewer", "content": review_result.answer, "round": r})

            if review_result.answer.strip().startswith("PASSED"):
                logger.info(f"Reflective[{run_id}]: Quality check passed at round {r}.")
                break

            logger.info(f"Reflective[{run_id}] Round {r}/{self.rounds}: Refining...")
            refine_prompt = (
                f"Task: {task}\n"
                f"Your Previous Answer: {_truncate(current_answer.answer, _MAX_ANSWER_CHARS)}\n"
                f"Feedback from Reviewer: {_truncate(review_result.answer, _MAX_ANSWER_CHARS)}\n\n"
                "Refine your answer based on the feedback."
            )
            current_answer = await primary_agent.run(refine_prompt)
            history.append({"role": "primary", "content": current_answer.answer, "round": r})

        return {
            "final_result": current_answer.answer,
            "reflection_history": history,
            "rounds_completed": len([h for h in history if h["role"] == "reviewer"]),
            "strategy": "reflective",
            "run_id": run_id,
            "total_time_s": time.monotonic() - t0,
        }


class DebateStrategy(CoordinationStrategy):
    """
    토론 전략

    Mathematical Foundation:
        Iterative refinement:
        xₙ₊₁ = f(xₙ, feedback)

        Convergence:
        lim(n→∞) d(xₙ, x*) = 0

        Nash Equilibrium:
        Each agent's strategy is optimal given others' strategies
    """

    def __init__(self, rounds: int = 3, judge_agent: "Optional[AgentProtocol]" = None) -> None:
        self.rounds = rounds
        self.judge = judge_agent

    async def execute(
        self, agents: "List[AgentProtocol]", task: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """토론 실행. 초기 답변 수집 및 각 라운드 내 에이전트 응답을 병렬화."""
        run_id = str(uuid.uuid4())
        logger.info(f"Debate[{run_id}]: {len(agents)} agents, {self.rounds} rounds")
        t0 = time.monotonic()

        # 초기 답변 병렬 수집
        init_results_raw = await asyncio.gather(
            *[agent.run(task) for agent in agents], return_exceptions=True
        )

        current_answers: Dict[str, str] = {}
        for i, r in enumerate(init_results_raw):
            if isinstance(r, BaseException):
                logger.warning(f"Debate[{run_id}]: Agent {i} failed on initial answer: {r!r}")
                current_answers[f"agent_{i}"] = f"[Agent {i} failed: {r}]"
            else:
                current_answers[f"agent_{i}"] = r.answer

        debate_history = [{"round": 0, "answers": current_answers.copy()}]

        # 토론 라운드 — 라운드 내 에이전트 응답 병렬화
        for round_num in range(1, self.rounds + 1):
            logger.info(f"Debate[{run_id}] Round {round_num}/{self.rounds}")

            async def _get_refined_answer(
                agent: "AgentProtocol",
                agent_key: str,
                answers_snapshot: Dict[str, str],
            ) -> tuple[str, str]:
                other_answers = "\n".join(
                    _truncate(f"Agent {k}: {v}", _MAX_ANSWER_CHARS)
                    for k, v in answers_snapshot.items()
                    if k != agent_key
                )
                prompt = (
                    f"Task: {task}\n\n"
                    f"Your previous answer:\n{_truncate(answers_snapshot[agent_key], _MAX_ANSWER_CHARS)}\n\n"
                    f"Other agents' answers:\n{_truncate(other_answers, _MAX_ANSWER_CHARS * 3)}\n\n"
                    "Consider the other answers and refine yours. "
                    "Stick with it if confident, or incorporate good points from others.\n\n"
                    "Your refined answer:"
                )
                result = await agent.run(prompt)
                return agent_key, result.answer

            snapshot = current_answers.copy()
            round_results_raw = await asyncio.gather(
                *[
                    _get_refined_answer(agent, f"agent_{i}", snapshot)
                    for i, agent in enumerate(agents)
                ],
                return_exceptions=True,
            )

            new_answers: Dict[str, str] = {}
            for item in round_results_raw:
                if isinstance(item, BaseException):
                    logger.warning(f"Debate[{run_id}] Round {round_num}: agent failed: {item!r}")
                else:
                    key, answer = item
                    new_answers[key] = answer

            # 실패한 에이전트는 이전 답변 유지
            for k in current_answers:
                if k not in new_answers:
                    new_answers[k] = current_answers[k]

            current_answers = new_answers
            debate_history.append({"round": round_num, "answers": current_answers.copy()})

        # 최종 판정
        if self.judge:
            joined = "\n".join(
                _truncate(f"Agent {i}: {ans}", _MAX_ANSWER_CHARS)
                for i, ans in enumerate(current_answers.values())
            )
            judge_prompt = (
                f"Task: {task}\n\n"
                f"After {self.rounds} rounds of debate, here are the final answers:\n\n"
                f"{joined}\n\n"
                "As a judge, determine the best answer and explain why:"
            )
            judge_result = await self.judge.run(judge_prompt)
            final_answer = judge_result.answer
            decision_method = "judge"
        else:
            vote_counts = Counter(current_answers.values())
            final_answer = vote_counts.most_common(1)[0][0]
            decision_method = "vote"

        return {
            "final_result": final_answer,
            "debate_history": debate_history,
            "rounds": self.rounds,
            "decision_method": decision_method,
            "strategy": "debate",
            "run_id": run_id,
            "total_time_s": time.monotonic() - t0,
        }
