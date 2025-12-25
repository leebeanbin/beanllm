"""
MultiAgentServiceImpl - Multi-Agent 서비스 구현체
SOLID 원칙:
- SRP: Multi-Agent 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...domain.multi_agent.strategies import (
    DebateStrategy,
    HierarchicalStrategy,
    ParallelStrategy,
    SequentialStrategy,
)
from ...dto.request.multi_agent_request import MultiAgentRequest
from ...dto.response.multi_agent_response import MultiAgentResponse
from ...utils.logger import get_logger
from ..multi_agent_service import IMultiAgentService

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class MultiAgentServiceImpl(IMultiAgentService):
    """
    Multi-Agent 서비스 구현체

    책임:
    - Multi-Agent 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: Multi-Agent 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(self) -> None:
        """의존성 주입을 통한 생성자"""
        pass

    async def execute_sequential(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        순차 실행 (기존 multi_agent.py의 SequentialStrategy.execute() 정확히 마이그레이션)

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        # 기존 multi_agent.py의 SequentialStrategy.execute() 로직 정확히 마이그레이션
        strategy = SequentialStrategy()
        result = await strategy.execute(request.agents or [], request.task, **request.extra_params)

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=result.get("strategy", "sequential"),
            intermediate_results=result.get("intermediate_results"),
            all_steps=result.get("all_steps"),
            metadata=result,
        )

    async def execute_parallel(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        병렬 실행 (기존 multi_agent.py의 ParallelStrategy.execute() 정확히 마이그레이션)

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        # 기존 multi_agent.py의 ParallelStrategy.execute() 로직 정확히 마이그레이션
        strategy = ParallelStrategy(aggregation=request.aggregation)
        result = await strategy.execute(request.agents or [], request.task, **request.extra_params)

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=result.get("strategy", "parallel"),
            metadata=result,
        )

    async def execute_hierarchical(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        계층적 실행 (기존 multi_agent.py의 HierarchicalStrategy.execute() 정확히 마이그레이션)

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO

        Note: request.agents는 [manager, worker1, worker2, ...] 순서로 전달되어야 함
        """
        if not request.agents or len(request.agents) < 2:
            raise ValueError(
                "At least manager and one worker are required for hierarchical strategy"
            )

        # 첫 번째 agent가 manager, 나머지가 workers (기존 multi_agent.py와 동일한 구조)
        manager_agent = request.agents[0]
        workers = request.agents[1:]

        # 기존 multi_agent.py의 HierarchicalStrategy.execute() 로직 정확히 마이그레이션
        strategy = HierarchicalStrategy(manager_agent=manager_agent)
        result = await strategy.execute(workers, request.task, **request.extra_params)

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=result.get("strategy", "hierarchical"),
            metadata=result,
        )

    async def execute_debate(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        토론 실행 (기존 multi_agent.py의 DebateStrategy.execute() 정확히 마이그레이션)

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO

        Note:
        - request.agents는 토론 참여 agents만 포함
        - request.judge_agent가 있으면 사용, 없으면 None (기존 multi_agent.py와 동일)
        """
        debate_agents = request.agents or []

        # Judge agent (기존 multi_agent.py: judge = self.agents[judge_id] if judge_id else None)
        # 새로운 구조: request.judge_agent로 직접 전달
        judge_agent = request.judge_agent

        # 기존 multi_agent.py의 DebateStrategy.execute() 로직 정확히 마이그레이션
        strategy = DebateStrategy(rounds=request.rounds, judge_agent=judge_agent)
        result = await strategy.execute(debate_agents, request.task, **request.extra_params)

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=result.get("strategy", "debate"),
            metadata=result,
        )
