"""
Agent Facade - 기존 Agent API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..domain.tools import Tool, ToolRegistry


@dataclass
class AgentStep:
    """에이전트 단계 (기존 API 유지)"""

    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


@dataclass
class AgentResult:
    """에이전트 실행 결과 (기존 API 유지)"""

    answer: str
    steps: List[AgentStep]
    total_steps: int
    success: bool = True
    error: Optional[str] = None


class Agent:
    """
    ReAct 에이전트 (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        ```python
        from beanllm import Agent, Tool

        # 도구 정의
        def search(query: str) -> str:
            return f"Results for {query}"

        # 에이전트 생성
        agent = Agent(
            model="gpt-4o-mini",
            tools=[Tool.from_function(search)]
        )

        # 실행
        result = await agent.run("서울 인구는?")
        print(result.answer)
        ```
    """

    def __init__(
        self,
        model: str,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        provider: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            model: 모델 ID
            tools: 도구 목록
            max_iterations: 최대 반복 횟수
            provider: Provider 이름
            verbose: 상세 로그 출력
        """
        self.model = model
        self.provider = provider
        self.max_iterations = max_iterations
        self.verbose = verbose

        # ToolRegistry 생성
        self.registry = ToolRegistry()
        if tools:
            for tool in tools:
                self.registry.add_tool(tool)

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..utils.di_container import get_container

        container = get_container()
        handler_factory = container.handler_factory

        # AgentHandler 생성
        self._agent_handler = handler_factory.create_agent_handler()

    async def run(self, task: str) -> AgentResult:
        """
        에이전트 실행

        내부적으로 Handler를 사용하여 처리

        Args:
            task: 수행할 작업

        Returns:
            AgentResult: 실행 결과
        """
        # Handler를 통한 처리 (기존 agent.py와 동일)
        # 기존: tools는 registry.get_all()로 가져옴
        tools_list = (
            self.registry.get_all()
            if hasattr(self.registry, "get_all")
            else (
                list(self.registry.get_all_tools().values())
                if hasattr(self.registry, "get_all_tools")
                else []
            )
        )

        response = await self._agent_handler.handle_run(
            task=task,
            model=self.model,
            tools=tools_list,
            max_steps=self.max_iterations,
            tool_registry=self.registry,  # ToolRegistry 전달 (기존 구조 유지)
            provider=self.provider,
        )

        # AgentResponse를 AgentResult로 변환 (기존 API 유지)
        steps = [
            AgentStep(
                step_number=step.get("step_number", i + 1),
                thought=step.get("thought", ""),
                action=step.get("action"),
                action_input=step.get("action_input"),  # action_input 파싱 완료
                observation=step.get("observation", ""),
                is_final=step.get("is_final", False) or (i == len(response.steps) - 1),
                final_answer=step.get("final_answer")
                or (step.get("observation") if i == len(response.steps) - 1 else None),
            )
            for i, step in enumerate(response.steps)
        ]

        return AgentResult(
            answer=response.answer,
            steps=steps,
            total_steps=response.total_steps,
            success=response.success,
            error=response.error,
        )

    def add_tool(self, tool: Tool) -> None:
        """
        도구 추가 (기존 API 유지)

        Args:
            tool: 추가할 도구
        """
        self.registry.add_tool(tool)


# 편의 함수 (기존 API 유지)
def create_agent(
    model: str,
    tools: Optional[List[Tool]] = None,
    max_iterations: int = 10,
    provider: Optional[str] = None,
) -> Agent:
    """
    Agent 생성 (편의 함수)

    Example:
        ```python
        agent = create_agent("gpt-4o-mini", tools=[...])
        ```
    """
    return Agent(model=model, tools=tools, max_iterations=max_iterations, provider=provider)
