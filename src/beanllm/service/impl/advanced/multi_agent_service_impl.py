"""
MultiAgentServiceImpl - Multi-Agent 서비스 구현체
SOLID 원칙:
- SRP: Multi-Agent 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, Optional, cast

from beanllm.domain.multi_agent.strategies import (
    AutonomousPlanningStrategy,
    DebateStrategy,
    HierarchicalStrategy,
    ParallelStrategy,
    ReflectiveStrategy,
    SequentialStrategy,
)
from beanllm.domain.protocols import IPlanRepository
from beanllm.dto.request.advanced.multi_agent_request import MultiAgentRequest
from beanllm.dto.response.advanced.multi_agent_response import MultiAgentResponse
from beanllm.infrastructure.distributed.pipeline_decorators import with_distributed_features
from beanllm.service.multi_agent_service import IMultiAgentService
from beanllm.utils.logging import get_logger

# 환경변수로 분산 모드 활성화 여부 확인
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"

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

    def __init__(self, plan_repository: Optional[IPlanRepository] = None) -> None:
        """
        Args:
            plan_repository: 사용자 수정 계획을 조회하는 리포지토리.
                Playground 등 외부 레이어에서 주입. None이면 사용자 수정 계획 기능 비활성화.
        """
        self._plan_repository = plan_repository

    @with_distributed_features(
        pipeline_type="multi_agent",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        enable_distributed_lock=True,
        cache_key_prefix="multi_agent:sequential",
        rate_limit_key="multi_agent:execute",
        lock_key=lambda self, args, kwargs: (
            f"multi_agent:sequential:{hash(args[0].task.encode()) if args and hasattr(args[0], 'task') else 'default'}"
        ),
        event_type="multi_agent.sequential",
    )
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

    @with_distributed_features(
        pipeline_type="multi_agent",
        enable_rate_limiting=True,
        enable_event_streaming=True,
        rate_limit_key="multi_agent:execute",
        event_type="multi_agent.parallel",
    )
    async def execute_parallel(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        병렬 실행 (기존 multi_agent.py의 ParallelStrategy.execute() 정확히 마이그레이션)

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        # 분산 모드: Task Queue 사용
        if USE_DISTRIBUTED and len(request.agents or []) > 3:
            try:
                from beanllm.infrastructure.distributed import BatchProcessor, ConcurrencyController

                batch_processor = BatchProcessor(task_type="multi_agent.tasks", max_concurrent=10)
                concurrency_controller = ConcurrencyController()

                async def process_agent(agent: Any, task: str) -> Any:
                    """단일 Agent 실행 (동시성 제어)"""
                    cm = await concurrency_controller.with_concurrency_control(
                        "multi_agent.parallel",
                        max_concurrent=10,
                        rate_limit_key="multi_agent:execute",
                    )
                    async with cm:
                        return await agent.run(task)

                # 배치 처리
                tasks_data = [
                    {"agent": agent, "task": request.task or ""} for agent in (request.agents or [])
                ]

                async def run_agent(task_data: Any) -> Any:
                    return await process_agent(task_data["agent"], cast(str, task_data["task"]))

                results = await batch_processor.process_batch(
                    task_name="execute",
                    tasks_data=tasks_data,
                    handler=run_agent,
                    priority=0,
                )

                # 결과 집계
                strategy = ParallelStrategy(aggregation=request.aggregation)
                answers = [r.answer if hasattr(r, "answer") else str(r) for r in results]
                if request.aggregation == "vote":
                    from collections import Counter

                    vote_counts = Counter(answers)
                    final_answer = vote_counts.most_common(1)[0][0]
                    result = {
                        "final_result": final_answer,
                        "all_answers": answers,
                        "vote_counts": dict(vote_counts),
                        "strategy": "parallel-vote",
                        "agreement_rate": vote_counts[final_answer] / len(answers),
                    }
                elif request.aggregation == "consensus":
                    if len(set(answers)) == 1:
                        result = {
                            "final_result": answers[0],
                            "consensus": True,
                            "strategy": "parallel-consensus",
                        }
                    else:
                        result = {
                            "final_result": None,
                            "consensus": False,
                            "all_answers": answers,
                            "strategy": "parallel-consensus",
                        }
                else:  # "all"
                    result = {
                        "final_result": answers,
                        "all_results": results,
                        "strategy": "parallel-all",
                    }

                return MultiAgentResponse(
                    final_result=result.get("final_result"),
                    strategy=cast(str, result.get("strategy", "parallel")),
                    metadata=result,
                )
            except Exception as e:
                logger.warning(
                    f"Distributed parallel execution failed: {e!r}. "
                    "Falling back to local ParallelStrategy."
                )
                # metadata에 degraded_mode 플래그를 포함해 호출자가 인지할 수 있게 함

        # 로컬 ParallelStrategy 실행 (분산 모드 비활성화 또는 fallback)
        strategy = ParallelStrategy(aggregation=request.aggregation)
        result = await strategy.execute(request.agents or [], request.task, **request.extra_params)
        result["degraded_mode"] = USE_DISTRIBUTED  # 분산 요청이었으나 로컬 실행됐음을 표시

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=cast(str, result.get("strategy", "parallel")),
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

    async def execute_autonomous_planning(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        자율 계획 및 실행

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        if not request.agents:
            raise ValueError("No agents provided for autonomous planning")

        # 첫 번째 agent가 planner
        planner_agent = request.agents[0]
        worker_agents = request.agents[1:] if len(request.agents) > 1 else request.agents

        # 텔레메트리 버스 가져오기
        telemetry_bus = request.extra_params.get("telemetry_bus")
        wait_for_approval = request.extra_params.get("wait_for_approval", False)

        async def plan_ready_callback(plan):
            if telemetry_bus:
                from beanllm.domain.multi_agent.communication import (
                    TelemetryEvent,
                    TelemetryEventType,
                )

                await telemetry_bus.emit_telemetry(
                    TelemetryEvent(
                        event_type=TelemetryEventType.PLAN_READY,
                        agent_id=request.planner_id or "planner",
                        content=plan,
                        metadata={"wait_for_approval": wait_for_approval},
                    )
                )

            if wait_for_approval:
                # 승인 대기 로직: request.extra_params의 이벤트를 기다림
                resume_event = request.extra_params.get("resume_event")
                if resume_event and isinstance(resume_event, asyncio.Event):
                    logger.info("AutonomousPlanning: Waiting for user approval...")
                    await resume_event.wait()
                    logger.info("AutonomousPlanning: Approval received, resuming...")

                    # 주입된 plan_repository를 통해 사용자 수정 계획 조회
                    # (sys.path 조작 금지 — playground 모듈은 DI로만 접근)
                    execution_id = request.extra_params.get("execution_id")
                    if execution_id and self._plan_repository is not None:
                        modified_plan = self._plan_repository.get_modified_plan(execution_id)
                        if modified_plan:
                            logger.info(
                                f"AutonomousPlanning: Applying user-modified plan "
                                f"({len(modified_plan)} steps)"
                            )
                            plan.clear()
                            plan.extend(modified_plan)

        strategy = AutonomousPlanningStrategy(
            planner_agent=planner_agent, on_plan_ready=plan_ready_callback
        )
        result = await strategy.execute(worker_agents, request.task, **request.extra_params)

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=result.get("strategy", "autonomous_planning"),
            metadata=result,
        )

    async def execute_reflective(self, request: MultiAgentRequest) -> MultiAgentResponse:
        """
        자기 성찰 및 상호 검증

        Args:
            request: Multi-Agent 요청 DTO

        Returns:
            MultiAgentResponse: Multi-Agent 응답 DTO
        """
        if not request.agents or len(request.agents) < 2:
            raise ValueError("At least two agents are required for reflective strategy")

        # 첫 번째가 실행 주체, 두 번째가 리뷰어
        primary_agent = request.agents[0]
        reviewer_agent = request.agents[1]

        # 기존 multi_agent.py의 ReflectiveStrategy.execute() 로직 적용
        strategy = ReflectiveStrategy(reviewer_agent=reviewer_agent, rounds=request.rounds)
        result = await strategy.execute([primary_agent], request.task, **request.extra_params)

        return MultiAgentResponse(
            final_result=result.get("final_result"),
            strategy=result.get("strategy", "reflective"),
            metadata=result,
        )
