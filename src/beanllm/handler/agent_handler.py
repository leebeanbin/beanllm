"""
AgentHandler - 에이전트 요청 처리 (Controller 역할)
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
from ..dto.request.agent_request import AgentRequest
from ..dto.response.agent_response import AgentResponse
from ..service.agent_service import IAgentService


class AgentHandler:
    """
    에이전트 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, agent_service: IAgentService) -> None:
        """
        의존성 주입

        Args:
            agent_service: 에이전트 서비스 (인터페이스에 의존 - DIP)
        """
        self._agent_service = agent_service

    @log_handler_call
    @handle_errors(error_message="Agent task failed")
    @validate_input(
        required_params=["task", "model"],
        param_types={"task": str, "model": str, "max_steps": int},
        param_ranges={"temperature": (0, 2), "max_steps": (1, None)},
    )
    async def handle_run(
        self,
        task: str,
        model: str,
        tools: Optional[List[Any]] = None,
        max_steps: int = 10,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        tool_registry: Optional[Any] = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """
        에이전트 실행 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            task: 작업 설명
            model: 모델 이름
            tools: 도구 리스트 (tool_registry가 없을 때 사용)
            max_steps: 최대 단계 수
            temperature: 온도
            system_prompt: 시스템 프롬프트
            tool_registry: 도구 레지스트리 (선택적, 없으면 tools로부터 생성)
            **kwargs: 추가 파라미터

        Returns:
            AgentResponse: 에이전트 응답

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # ToolRegistry 생성 (기존 agent.py와 동일한 로직)
        from beanllm.domain.tools import ToolRegistry

        registry = tool_registry or ToolRegistry()
        if tools:
            for tool in tools:
                registry.add_tool(tool)

        # DTO 생성 (tool_registry 포함)
        request = AgentRequest(
            task=task,
            model=model,
            tools=tools or [],
            tool_registry=registry,  # DTO에 포함하여 전달
            max_steps=max_steps,
            temperature=temperature,
            system_prompt=system_prompt,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        # tool_registry는 request를 통해 전달됨
        return await self._agent_service.run(request)
