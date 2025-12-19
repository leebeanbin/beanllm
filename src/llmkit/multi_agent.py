"""
Multi-Agent System - Agent Collaboration & Coordination
여러 에이전트의 협업, 통신, 조정 시스템

Mathematical Foundation:
    - Message Passing: Communication between processes
    - Consensus Algorithms: Achieving agreement in distributed systems
    - Game Theory: Strategic interaction between agents
    - Distributed Systems: Coordination patterns
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Callable, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from .agent import Agent, AgentResult, AgentStep
from .client import Client
from .utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Message Passing System
# =============================================================================

class MessageType(Enum):
    """메시지 타입"""
    REQUEST = "request"           # 작업 요청
    RESPONSE = "response"         # 작업 응답
    BROADCAST = "broadcast"       # 전체 공지
    QUERY = "query"              # 정보 요청
    INFORM = "inform"            # 정보 전달
    DELEGATE = "delegate"        # 작업 위임
    VOTE = "vote"                # 투표
    CONSENSUS = "consensus"       # 합의


@dataclass
class AgentMessage:
    """
    Agent 간 메시지

    Mathematical Foundation:
        Message Passing Model에서 메시지는 튜플로 표현됩니다:
        m = (sender, receiver, content, timestamp)

        Channel capacity:
        C = max I(X; Y) where X: input, Y: output
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""                          # 송신자 agent ID
    receiver: Optional[str] = None            # 수신자 (None이면 broadcast)
    message_type: MessageType = MessageType.INFORM
    content: Any = None                       # 메시지 내용
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None           # 답장하는 메시지 ID

    def reply(
        self,
        content: Any,
        message_type: MessageType = MessageType.RESPONSE
    ) -> 'AgentMessage':
        """이 메시지에 대한 답장 생성"""
        return AgentMessage(
            sender=self.receiver,
            receiver=self.sender,
            message_type=message_type,
            content=content,
            reply_to=self.id
        )


class CommunicationBus:
    """
    Agent 간 통신 버스

    Publish-Subscribe 패턴 구현

    Mathematical Foundation:
        Event-driven architecture:
        - Publisher: P → {e₁, e₂, ..., eₙ}
        - Subscriber: S ← {e ∈ E | filter(e)}
        - Delivery guarantee: At-most-once, At-least-once, Exactly-once
    """

    def __init__(self, delivery_guarantee: str = "at-most-once"):
        """
        Args:
            delivery_guarantee: 전송 보장 수준
                - "at-most-once": 최대 1번 (빠름, 손실 가능)
                - "at-least-once": 최소 1번 (중복 가능)
                - "exactly-once": 정확히 1번 (느림, 보장)
        """
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[Callable]] = {}  # agent_id -> [callbacks]
        self.delivery_guarantee = delivery_guarantee
        self.delivered_messages: set = set()  # For exactly-once

    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]):
        """메시지 구독"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        logger.debug(f"Agent {agent_id} subscribed to bus")

    def unsubscribe(self, agent_id: str, callback: Optional[Callable] = None):
        """구독 취소"""
        if agent_id in self.subscribers:
            if callback:
                self.subscribers[agent_id].remove(callback)
            else:
                del self.subscribers[agent_id]

    async def publish(self, message: AgentMessage):
        """
        메시지 발행

        Time Complexity: O(n) where n = number of subscribers
        """
        self.messages.append(message)

        # Exactly-once: 중복 방지
        if self.delivery_guarantee == "exactly-once":
            if message.id in self.delivered_messages:
                logger.debug(f"Message {message.id} already delivered, skipping")
                return
            self.delivered_messages.add(message.id)

        # 수신자에게 전달
        if message.receiver:
            # Unicast (1:1)
            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
        else:
            # Broadcast (1:N)
            for agent_id, callbacks in self.subscribers.items():
                # 자기 자신은 제외
                if agent_id == message.sender:
                    continue

                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in callback for {agent_id}: {e}")

    def get_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """메시지 히스토리 조회"""
        if agent_id:
            filtered = [
                m for m in self.messages
                if m.sender == agent_id or m.receiver == agent_id
            ]
            return filtered[-limit:]
        return self.messages[-limit:]


# =============================================================================
# Coordination Strategies
# =============================================================================

class CoordinationStrategy(ABC):
    """조정 전략 베이스 클래스"""

    @abstractmethod
    async def execute(
        self,
        agents: List[Agent],
        task: str,
        **kwargs
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
        self,
        agents: List[Agent],
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        """순차 실행"""
        results = []
        current_input = task

        for i, agent in enumerate(agents):
            logger.info(f"Sequential: Agent {i+1}/{len(agents)} executing")

            result = await agent.run(current_input)
            results.append(result)

            # 다음 agent의 입력은 이전 agent의 출력
            current_input = result.answer

        return {
            "final_result": results[-1].answer if results else None,
            "intermediate_results": [r.answer for r in results],
            "all_steps": results,
            "strategy": "sequential"
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

    async def execute(
        self,
        agents: List[Agent],
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        """병렬 실행"""
        logger.info(f"Parallel: Executing {len(agents)} agents concurrently")

        # 모든 agent를 병렬 실행
        tasks = [agent.run(task) for agent in agents]

        if self.aggregation == "first":
            # 첫 번째 완료된 것만 사용
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # 나머지 취소
            for t in pending:
                t.cancel()

            result = list(done)[0].result()
            return {
                "final_result": result.answer,
                "strategy": "parallel-first",
                "completed": 1,
                "total": len(agents)
            }

        else:
            # 모든 agent 완료 대기
            results = await asyncio.gather(*tasks)
            answers = [r.answer for r in results]

            if self.aggregation == "vote":
                # 투표: 가장 많이 나온 답 선택
                from collections import Counter
                vote_counts = Counter(answers)
                final_answer = vote_counts.most_common(1)[0][0]

                return {
                    "final_result": final_answer,
                    "all_answers": answers,
                    "vote_counts": dict(vote_counts),
                    "strategy": "parallel-vote",
                    "agreement_rate": vote_counts[final_answer] / len(answers)
                }

            elif self.aggregation == "consensus":
                # 합의: 모두 같은 답이어야 함
                if len(set(answers)) == 1:
                    return {
                        "final_result": answers[0],
                        "consensus": True,
                        "strategy": "parallel-consensus"
                    }
                else:
                    return {
                        "final_result": None,
                        "consensus": False,
                        "all_answers": answers,
                        "strategy": "parallel-consensus"
                    }

            else:  # "all"
                return {
                    "final_result": answers,
                    "all_results": results,
                    "strategy": "parallel-all"
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

    def __init__(self, manager_agent: Agent):
        """
        Args:
            manager_agent: 매니저 역할 agent
        """
        self.manager = manager_agent

    async def execute(
        self,
        agents: List[Agent],  # Workers
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
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
        import json
        import re

        json_match = re.search(r'\{.*\}', delegation_result.answer, re.DOTALL)
        if json_match:
            subtasks_data = json.loads(json_match.group())
            subtasks = subtasks_data.get("subtasks", [])
        else:
            # 파싱 실패시 단순 분할
            subtasks = [task] * len(agents)

        # 2. Workers 병렬 실행
        worker_tasks = []
        for i, (agent, subtask) in enumerate(zip(agents, subtasks)):
            logger.info(f"Worker {i+1}: {subtask[:50]}...")
            worker_tasks.append(agent.run(subtask))

        worker_results = await asyncio.gather(*worker_tasks)
        worker_answers = [r.answer for r in worker_results]

        # 3. Manager가 결과 종합
        synthesis_prompt = f"""You are a manager. Synthesize the results from your workers into a final answer.

Original Task: {task}

Worker Results:
{chr(10).join(f'{i+1}. {ans}' for i, ans in enumerate(worker_answers))}

Provide a comprehensive final answer:
"""

        final_result = await self.manager.run(synthesis_prompt)

        return {
            "final_result": final_result.answer,
            "subtasks": subtasks,
            "worker_results": worker_answers,
            "strategy": "hierarchical",
            "manager_steps": len(delegation_result.steps) + len(final_result.steps),
            "total_workers": len(agents)
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

    def __init__(self, rounds: int = 3, judge_agent: Optional[Agent] = None):
        """
        Args:
            rounds: 토론 라운드 수
            judge_agent: 판정 agent (None이면 투표)
        """
        self.rounds = rounds
        self.judge = judge_agent

    async def execute(
        self,
        agents: List[Agent],
        task: str,
        **kwargs
    ) -> Dict[str, Any]:
        """토론 실행"""
        logger.info(f"Debate: {len(agents)} agents, {self.rounds} rounds")

        debate_history = []
        current_answers = {}

        # 초기 답변
        for i, agent in enumerate(agents):
            result = await agent.run(task)
            current_answers[f"agent_{i}"] = result.answer

        debate_history.append({
            "round": 0,
            "answers": current_answers.copy()
        })

        # 토론 라운드
        for round_num in range(1, self.rounds + 1):
            logger.info(f"Debate Round {round_num}/{self.rounds}")

            new_answers = {}

            for i, agent in enumerate(agents):
                # 다른 agents의 답변 보여주기
                other_answers = "\n".join([
                    f"Agent {j}: {ans}"
                    for j, ans in enumerate(current_answers.values())
                    if j != i
                ])

                debate_prompt = f"""Task: {task}

Your previous answer:
{current_answers[f'agent_{i}']}

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
            debate_history.append({
                "round": round_num,
                "answers": current_answers.copy()
            })

        # 최종 판정
        if self.judge:
            # Judge가 판정
            judge_prompt = f"""Task: {task}

After {self.rounds} rounds of debate, here are the final answers:

{chr(10).join(f'Agent {i}: {ans}' for i, ans in enumerate(current_answers.values()))}

As a judge, determine the best answer and explain why:
"""

            judge_result = await self.judge.run(judge_prompt)
            final_answer = judge_result.answer
            decision_method = "judge"

        else:
            # 투표로 결정
            from collections import Counter
            vote_counts = Counter(current_answers.values())
            final_answer = vote_counts.most_common(1)[0][0]
            decision_method = "vote"

        return {
            "final_result": final_answer,
            "debate_history": debate_history,
            "rounds": self.rounds,
            "decision_method": decision_method,
            "strategy": "debate"
        }


# =============================================================================
# Multi-Agent Coordinator
# =============================================================================

class MultiAgentCoordinator:
    """
    Multi-Agent 조정자

    여러 agent를 조정하고 협업시키는 시스템

    Mathematical Foundation:
        Coordinator as a controller:
        - State: S = {s₁, s₂, ..., sₙ} (각 agent의 상태)
        - Action: A = {coordinate, delegate, aggregate}
        - Transition: s' = δ(s, a)

    Example:
        ```python
        from llmkit import Agent, MultiAgentCoordinator

        # Agents 생성
        researcher = Agent(model="gpt-4o", tools=[search_tool])
        writer = Agent(model="gpt-4o", tools=[])

        # Coordinator
        coordinator = MultiAgentCoordinator(
            agents={"researcher": researcher, "writer": writer}
        )

        # 순차 실행
        result = await coordinator.execute_sequential(
            task="Research AI and write a summary",
            agent_order=["researcher", "writer"]
        )

        # 병렬 실행
        result = await coordinator.execute_parallel(
            task="What is the capital of France?",
            agents=["agent1", "agent2", "agent3"],
            aggregation="vote"
        )
        ```
    """

    def __init__(
        self,
        agents: Dict[str, Agent],
        communication_bus: Optional[CommunicationBus] = None
    ):
        """
        Args:
            agents: Agent 딕셔너리 {agent_id: Agent}
            communication_bus: 통신 버스 (None이면 자동 생성)
        """
        self.agents = agents
        self.bus = communication_bus or CommunicationBus()

        # 각 agent를 bus에 구독
        for agent_id in agents:
            self.bus.subscribe(agent_id, self._on_message)

    def _on_message(self, message: AgentMessage):
        """메시지 수신 핸들러"""
        logger.debug(f"Message received: {message.sender} → {message.receiver}")

    def add_agent(self, agent_id: str, agent: Agent):
        """Agent 추가"""
        self.agents[agent_id] = agent
        self.bus.subscribe(agent_id, self._on_message)

    def remove_agent(self, agent_id: str):
        """Agent 제거"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.bus.unsubscribe(agent_id)

    async def execute_sequential(
        self,
        task: str,
        agent_order: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        순차 실행

        Args:
            task: 작업
            agent_order: Agent 실행 순서 (agent_id 리스트)
        """
        agents = [self.agents[aid] for aid in agent_order]
        strategy = SequentialStrategy()
        return await strategy.execute(agents, task, **kwargs)

    async def execute_parallel(
        self,
        task: str,
        agent_ids: Optional[List[str]] = None,
        aggregation: str = "vote",
        **kwargs
    ) -> Dict[str, Any]:
        """
        병렬 실행

        Args:
            task: 작업
            agent_ids: 사용할 agent IDs (None이면 전체)
            aggregation: 집계 방법 (vote, consensus, first, all)
        """
        if agent_ids is None:
            agent_ids = list(self.agents.keys())

        agents = [self.agents[aid] for aid in agent_ids]
        strategy = ParallelStrategy(aggregation=aggregation)
        return await strategy.execute(agents, task, **kwargs)

    async def execute_hierarchical(
        self,
        task: str,
        manager_id: str,
        worker_ids: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        계층적 실행

        Args:
            task: 작업
            manager_id: 매니저 agent ID
            worker_ids: 워커 agent IDs
        """
        manager = self.agents[manager_id]
        workers = [self.agents[wid] for wid in worker_ids]

        strategy = HierarchicalStrategy(manager_agent=manager)
        return await strategy.execute(workers, task, **kwargs)

    async def execute_debate(
        self,
        task: str,
        agent_ids: Optional[List[str]] = None,
        rounds: int = 3,
        judge_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        토론 실행

        Args:
            task: 작업
            agent_ids: 토론 참여 agent IDs
            rounds: 토론 라운드 수
            judge_id: 판정자 agent ID (None이면 투표)
        """
        if agent_ids is None:
            agent_ids = list(self.agents.keys())

        agents = [self.agents[aid] for aid in agent_ids]
        judge = self.agents[judge_id] if judge_id else None

        strategy = DebateStrategy(rounds=rounds, judge_agent=judge)
        return await strategy.execute(agents, task, **kwargs)

    async def send_message(
        self,
        sender: str,
        receiver: Optional[str],
        content: Any,
        message_type: MessageType = MessageType.INFORM
    ):
        """메시지 전송"""
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content
        )
        await self.bus.publish(message)

    def get_communication_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """통신 히스토리 조회"""
        return self.bus.get_history(agent_id, limit)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_coordinator(
    agent_configs: List[Dict[str, Any]],
    **kwargs
) -> MultiAgentCoordinator:
    """
    Coordinator 빠르게 생성

    Args:
        agent_configs: Agent 설정 리스트
            [{"id": "agent1", "model": "gpt-4o", "tools": [...]}, ...]

    Returns:
        MultiAgentCoordinator
    """
    agents = {}

    for config in agent_configs:
        agent_id = config.pop("id")
        agents[agent_id] = Agent(**config)

    return MultiAgentCoordinator(agents=agents, **kwargs)


async def quick_debate(
    task: str,
    num_agents: int = 3,
    rounds: int = 2,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    빠른 토론 실행

    Args:
        task: 토론 주제
        num_agents: Agent 수
        rounds: 토론 라운드
        model: 사용할 모델

    Returns:
        토론 결과
    """
    # Agents 생성
    agents = {
        f"agent_{i}": Agent(model=model)
        for i in range(num_agents)
    }

    coordinator = MultiAgentCoordinator(agents=agents)

    return await coordinator.execute_debate(
        task=task,
        rounds=rounds
    )
