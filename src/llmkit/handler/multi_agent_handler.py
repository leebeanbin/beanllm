"""
MultiAgentHandler - Multi-Agent 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..dto.request.multi_agent_request import MultiAgentRequest
from ..dto.response.multi_agent_response import MultiAgentResponse
from ..service.multi_agent_service import IMultiAgentService


class MultiAgentHandler:
    """
    Multi-Agent 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, multi_agent_service: IMultiAgentService) -> None:
        """
        의존성 주입

        Args:
            multi_agent_service: Multi-Agent 서비스 (인터페이스에 의존 - DIP)
        """
        self._multi_agent_service = multi_agent_service

    @log_handler_call
    @handle_errors(error_message="Multi-Agent execution failed")
    @validate_input(
        required_params=["strategy", "task"],
        param_types={"strategy": str, "task": str, "agents": list},
    )
    async def handle_execute(
        self,
        strategy: str,
        task: str,
        agents: Optional[List[Any]] = None,
        agent_order: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None,
        manager_id: Optional[str] = None,
        worker_ids: Optional[List[str]] = None,
        aggregation: str = "vote",
        rounds: int = 3,
        judge_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiAgentResponse:
        """
        Multi-Agent 실행 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            strategy: 전략 (sequential, parallel, hierarchical, debate)
            task: 작업
            agents: Agent 리스트
            agent_order: 순차 실행용 순서
            agent_ids: 병렬/토론 실행용 agent IDs
            manager_id: 계층적 실행용 매니저 ID
            worker_ids: 계층적 실행용 워커 IDs
            aggregation: 병렬 실행용 집계 방법
            rounds: 토론 실행용 라운드 수
            judge_id: 토론 실행용 판정자 ID
            **kwargs: 추가 파라미터

        Returns:
            MultiAgentResponse: Multi-Agent 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # judge_id가 있으면 judge_agent를 찾아서 전달
        judge_agent = None
        if judge_id and "agents_dict" in kwargs:
            # agents_dict에서 judge 찾기 (facade에서 전달)
            agents_dict = kwargs.pop("agents_dict")
            if judge_id in agents_dict:
                judge_agent = agents_dict[judge_id]

        # DTO 생성
        request = MultiAgentRequest(
            strategy=strategy,
            task=task,
            agents=agents or [],
            agent_order=agent_order or [],
            agent_ids=agent_ids or [],
            manager_id=manager_id,
            worker_ids=worker_ids or [],
            aggregation=aggregation,
            rounds=rounds,
            judge_id=judge_id,
            judge_agent=judge_agent,
            extra_params=kwargs,
        )

        # Service 호출 (Strategy 패턴 적용 - 통합 execute 메서드 사용)
        return await self._multi_agent_service.execute(request)
