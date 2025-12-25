"""
Multi-Agent Facade - 기존 Multi-Agent API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..domain.multi_agent import AgentMessage, CommunicationBus, MessageType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultiAgentCoordinator:
    """
    Multi-Agent 조정자 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm import Agent, MultiAgentCoordinator

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
        self, agents: Dict[str, Any], communication_bus: Optional[CommunicationBus] = None
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

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory
        self._multi_agent_handler = handler_factory.create_multi_agent_handler()

    def _on_message(self, message: AgentMessage):
        """메시지 수신 핸들러"""
        logger.debug(f"Message received: {message.sender} → {message.receiver}")

    def add_agent(self, agent_id: str, agent: Any):  # Agent
        """Agent 추가"""
        self.agents[agent_id] = agent
        self.bus.subscribe(agent_id, self._on_message)

    def remove_agent(self, agent_id: str):
        """Agent 제거"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.bus.unsubscribe(agent_id)

    async def execute_sequential(
        self, task: str, agent_order: List[str], **kwargs
    ) -> Dict[str, Any]:
        """
        순차 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            task: 작업
            agent_order: Agent 실행 순서 (agent_id 리스트)
        """
        # Agent 리스트 생성
        agents = [self.agents[aid] for aid in agent_order]

        # Handler를 통한 처리
        response = await self._multi_agent_handler.handle_execute(
            strategy="sequential",
            task=task,
            agents=agents,
            agent_order=agent_order,
            **kwargs,
        )

        # MultiAgentResponse를 Dict로 변환 (기존 API 유지)
        return {
            "final_result": response.final_result,
            "intermediate_results": response.intermediate_results,
            "all_steps": response.all_steps,
            "strategy": response.strategy,
            **response.metadata,
        }

    async def execute_parallel(
        self, task: str, agent_ids: Optional[List[str]] = None, aggregation: str = "vote", **kwargs
    ) -> Dict[str, Any]:
        """
        병렬 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            task: 작업
            agent_ids: 사용할 agent IDs (None이면 전체)
            aggregation: 집계 방법 (vote, consensus, first, all)
        """
        if agent_ids is None:
            agent_ids = list(self.agents.keys())

        # Agent 리스트 생성
        agents = [self.agents[aid] for aid in agent_ids]

        # Handler를 통한 처리
        response = await self._multi_agent_handler.handle_execute(
            strategy="parallel",
            task=task,
            agents=agents,
            agent_ids=agent_ids,
            aggregation=aggregation,
            **kwargs,
        )

        # MultiAgentResponse를 Dict로 변환 (기존 API 유지)
        return {
            "final_result": response.final_result,
            "strategy": response.strategy,
            **response.metadata,
        }

    async def execute_hierarchical(
        self, task: str, manager_id: str, worker_ids: List[str], **kwargs
    ) -> Dict[str, Any]:
        """
        계층적 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            task: 작업
            manager_id: 매니저 agent ID
            worker_ids: 워커 agent IDs
        """
        # Agent 리스트 생성 (manager + workers) - 첫 번째가 manager
        manager = self.agents[manager_id]
        workers = [self.agents[wid] for wid in worker_ids]
        agents = [manager] + workers  # manager가 첫 번째

        # Handler를 통한 처리
        response = await self._multi_agent_handler.handle_execute(
            strategy="hierarchical",
            task=task,
            agents=agents,
            manager_id=manager_id,
            worker_ids=worker_ids,
            **kwargs,
        )

        # MultiAgentResponse를 Dict로 변환 (기존 API 유지)
        return {
            "final_result": response.final_result,
            "strategy": response.strategy,
            **response.metadata,
        }

    async def execute_debate(
        self,
        task: str,
        agent_ids: Optional[List[str]] = None,
        rounds: int = 3,
        judge_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        토론 실행

        내부적으로 Handler를 사용하여 처리 (기존 multi_agent.py의 execute_debate() 정확히 마이그레이션)

        Args:
            task: 작업
            agent_ids: 토론 참여 agent IDs
            rounds: 토론 라운드 수
            judge_id: 판정자 agent ID (None이면 투표)
        """
        if agent_ids is None:
            agent_ids = list(self.agents.keys())

        # Agent 리스트 생성 (토론 참여 agents만) - 기존과 동일
        agents = [self.agents[aid] for aid in agent_ids]

        # Judge agent 찾기 (기존 multi_agent.py와 동일)
        self.agents[judge_id] if judge_id else None

        # Handler를 통한 처리
        # judge를 agents_dict로 전달하여 handler에서 찾을 수 있도록 함
        response = await self._multi_agent_handler.handle_execute(
            strategy="debate",
            task=task,
            agents=agents,
            agent_ids=agent_ids,
            rounds=rounds,
            judge_id=judge_id,
            agents_dict=self.agents,  # judge를 찾기 위한 딕셔너리 전달
            **kwargs,
        )

        # MultiAgentResponse를 Dict로 변환 (기존 API 유지)
        return {
            "final_result": response.final_result,
            "strategy": response.strategy,
            **response.metadata,
        }

    async def send_message(
        self,
        sender: str,
        receiver: Optional[str],
        content: Any,
        message_type: MessageType = MessageType.INFORM,
    ):
        """메시지 전송"""
        message = AgentMessage(
            sender=sender, receiver=receiver, message_type=message_type, content=content
        )
        await self.bus.publish(message)

    def get_communication_history(
        self, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[AgentMessage]:
        """통신 히스토리 조회"""
        return self.bus.get_history(agent_id, limit)


# 편의 함수
def create_coordinator(agent_configs: List[Dict[str, Any]], **kwargs) -> MultiAgentCoordinator:
    """
    Coordinator 빠르게 생성

    Args:
        agent_configs: Agent 설정 리스트
            [{"id": "agent1", "model": "gpt-4o", "tools": [...]}, ...]

    Returns:
        MultiAgentCoordinator
    """
    from ..facade.agent_facade import Agent

    agents = {}

    for config in agent_configs:
        agent_id = config.pop("id")
        agents[agent_id] = Agent(**config)

    return MultiAgentCoordinator(agents=agents, **kwargs)


async def quick_debate(
    task: str, num_agents: int = 3, rounds: int = 2, model: str = "gpt-4o-mini"
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
    from ..facade.agent_facade import Agent

    # Agents 생성
    agents = {f"agent_{i}": Agent(model=model) for i in range(num_agents)}

    coordinator = MultiAgentCoordinator(agents=agents)

    return await coordinator.execute_debate(task=task, rounds=rounds)
