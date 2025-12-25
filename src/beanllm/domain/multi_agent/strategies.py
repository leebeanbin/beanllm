"""
Coordination Strategies - Agent 조정 전략들
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional

from ...utils.logger import get_logger

logger = get_logger(__name__)


class CoordinationStrategy(ABC):
    """조정 전략 베이스 클래스"""

    @abstractmethod
    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
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

    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """순차 실행"""
        results = []
        current_input = task

        for i, agent in enumerate(agents):
            logger.info(f"Sequential: Agent {i + 1}/{len(agents)} executing")

            result = await agent.run(current_input)
            results.append(result)

            # 다음 agent의 입력은 이전 agent의 출력
            current_input = result.answer

        return {
            "final_result": results[-1].answer if results else None,
            "intermediate_results": [r.answer for r in results],
            "all_steps": results,
            "strategy": "sequential",
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

    def __init__(self, aggregation: str = "vote"):
        """
        Args:
            aggregation: 결과 집계 방법
                - "vote": 투표 (다수결)
                - "consensus": 합의 (모두 동의)
                - "first": 첫 번째 완료
                - "all": 모든 결과 반환
        """
        self.aggregation = aggregation

    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """병렬 실행"""
        logger.info(f"Parallel: Executing {len(agents)} agents concurrently")

        # 모든 agent를 병렬 실행
        # asyncio.wait는 Task를 받아야 하므로 coroutine을 Task로 변환
        tasks = [asyncio.create_task(agent.run(task)) for agent in agents]

        if self.aggregation == "first":
            # 첫 번째 완료된 것만 사용
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # 나머지 취소
            for t in pending:
                t.cancel()

            result = list(done)[0].result()
            return {
                "final_result": result.answer,
                "strategy": "parallel-first",
                "completed": 1,
                "total": len(agents),
            }

        else:
            # 모든 agent 완료 대기
            results = await asyncio.gather(*tasks)
            answers = [r.answer for r in results]

            if self.aggregation == "vote":
                # 투표: 가장 많이 나온 답 선택
                vote_counts = Counter(answers)
                final_answer = vote_counts.most_common(1)[0][0]

                return {
                    "final_result": final_answer,
                    "all_answers": answers,
                    "vote_counts": dict(vote_counts),
                    "strategy": "parallel-vote",
                    "agreement_rate": vote_counts[final_answer] / len(answers),
                }

            elif self.aggregation == "consensus":
                # 합의: 모두 같은 답이어야 함
                if len(set(answers)) == 1:
                    return {
                        "final_result": answers[0],
                        "consensus": True,
                        "strategy": "parallel-consensus",
                    }
                else:
                    return {
                        "final_result": None,
                        "consensus": False,
                        "all_answers": answers,
                        "strategy": "parallel-consensus",
                    }

            else:  # "all"
                return {"final_result": answers, "all_results": results, "strategy": "parallel-all"}


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

    def __init__(self, manager_agent: Any):  # Agent
        """
        Args:
            manager_agent: 매니저 역할 agent
        """
        self.manager = manager_agent

    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:  # Workers
        """계층적 실행"""
        logger.info(f"Hierarchical: Manager delegating to {len(agents)} workers")

        # 1. Manager가 작업 분해
        delegation_prompt = f"""You are a manager. Break down this task into subtasks for {len(agents)} workers.

Task: {task}

Return a JSON list of subtasks:
{{"subtasks": ["subtask1", "subtask2", ...]}}
"""

        delegation_result = await self.manager.run(delegation_prompt)

        # JSON 파싱
        json_match = re.search(r"\{.*\}", delegation_result.answer, re.DOTALL)
        if json_match:
            subtasks_data = json.loads(json_match.group())
            subtasks = subtasks_data.get("subtasks", [])
        else:
            # 파싱 실패시 단순 분할
            subtasks = [task] * len(agents)

        # 2. Workers 병렬 실행
        worker_tasks = []
        for i, (agent, subtask) in enumerate(zip(agents, subtasks)):
            logger.info(f"Worker {i + 1}: {subtask[:50]}...")
            worker_tasks.append(agent.run(subtask))

        worker_results = await asyncio.gather(*worker_tasks)
        worker_answers = [r.answer for r in worker_results]

        # 3. Manager가 결과 종합
        synthesis_prompt = f"""You are a manager. Synthesize the results from your workers into a final answer.

Original Task: {task}

Worker Results:
{chr(10).join(f"{i + 1}. {ans}" for i, ans in enumerate(worker_answers))}

Provide a comprehensive final answer:
"""

        final_result = await self.manager.run(synthesis_prompt)

        return {
            "final_result": final_result.answer,
            "subtasks": subtasks,
            "worker_results": worker_answers,
            "strategy": "hierarchical",
            "manager_steps": len(delegation_result.steps) + len(final_result.steps),
            "total_workers": len(agents),
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

    def __init__(self, rounds: int = 3, judge_agent: Optional[Any] = None):  # Agent
        """
        Args:
            rounds: 토론 라운드 수
            judge_agent: 판정 agent (None이면 투표)
        """
        self.rounds = rounds
        self.judge = judge_agent

    async def execute(self, agents: List[Any], task: str, **kwargs) -> Dict[str, Any]:
        """토론 실행"""
        logger.info(f"Debate: {len(agents)} agents, {self.rounds} rounds")

        debate_history = []
        current_answers = {}

        # 초기 답변
        for i, agent in enumerate(agents):
            result = await agent.run(task)
            current_answers[f"agent_{i}"] = result.answer

        debate_history.append({"round": 0, "answers": current_answers.copy()})

        # 토론 라운드
        for round_num in range(1, self.rounds + 1):
            logger.info(f"Debate Round {round_num}/{self.rounds}")

            new_answers = {}

            for i, agent in enumerate(agents):
                # 다른 agents의 답변 보여주기
                other_answers = "\n".join(
                    [
                        f"Agent {j}: {ans}"
                        for j, ans in enumerate(current_answers.values())
                        if j != i
                    ]
                )

                debate_prompt = f"""Task: {task}

Your previous answer:
{current_answers[f"agent_{i}"]}

Other agents' answers:
{other_answers}

Consider the other answers and refine your answer. You can:
- Stick with your answer if you're confident
- Incorporate good points from others
- Point out flaws in other answers

Your refined answer:
"""

                result = await agent.run(debate_prompt)
                new_answers[f"agent_{i}"] = result.answer

            current_answers = new_answers
            debate_history.append({"round": round_num, "answers": current_answers.copy()})

        # 최종 판정
        if self.judge:
            # Judge가 판정
            judge_prompt = f"""Task: {task}

After {self.rounds} rounds of debate, here are the final answers:

{chr(10).join(f"Agent {i}: {ans}" for i, ans in enumerate(current_answers.values()))}

As a judge, determine the best answer and explain why:
"""

            judge_result = await self.judge.run(judge_prompt)
            final_answer = judge_result.answer
            decision_method = "judge"

        else:
            # 투표로 결정
            vote_counts = Counter(current_answers.values())
            final_answer = vote_counts.most_common(1)[0][0]
            decision_method = "vote"

        return {
            "final_result": final_answer,
            "debate_history": debate_history,
            "rounds": self.rounds,
            "decision_method": decision_method,
            "strategy": "debate",
        }
