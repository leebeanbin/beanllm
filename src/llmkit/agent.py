"""
Agent - ReAct Pattern Implementation
생각(Reasoning)하고 행동(Acting)하는 AI 에이전트
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .client import Client
from .tools import Tool, ToolRegistry
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentStep:
    """에이전트 단계"""

    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None


@dataclass
class AgentResult:
    """에이전트 실행 결과"""

    answer: str
    steps: List[AgentStep]
    total_steps: int
    success: bool = True
    error: Optional[str] = None


class Agent:
    """
    ReAct 에이전트

    생각(Thought) → 행동(Action) → 관찰(Observation) 반복

    Example:
        ```python
        from llmkit import Agent, Tool

        # 도구 정의
        def search(query: str) -> str:
            return f"Results for {query}"

        def calculator(a: float, b: float) -> float:
            return a + b

        # 에이전트 생성
        agent = Agent(
            model="gpt-4o-mini",
            tools=[
                Tool.from_function(search),
                Tool.from_function(calculator)
            ]
        )

        # 실행
        result = await agent.run("서울 인구는? 그리고 2를 곱해줘")
        print(result.answer)
        print(f"Steps: {result.total_steps}")
        ```
    """

    REACT_PROMPT = """You are a helpful AI assistant with access to tools.

To solve the task, you should follow the ReAct (Reasoning + Acting) pattern:
1. **Thought**: Think about what to do next
2. **Action**: Choose a tool to use
3. **Observation**: See the result
4. Repeat until you have the final answer

Available tools:
{tools_description}

Format:
Thought: [your reasoning]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}
Observation: [tool result]
... (repeat as needed)
Thought: I now know the final answer
Final Answer: [your final answer]

Important:
- Always start with "Thought:"
- Use "Action:" to call a tool
- Use "Action Input:" as valid JSON
- Use "Final Answer:" when you have the answer
- Be concise and clear

Task: {task}

Let's begin!
"""

    def __init__(
        self,
        model: str,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        provider: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            model: 모델 ID
            tools: 도구 목록
            max_iterations: 최대 반복 횟수
            provider: Provider 이름
            verbose: 상세 로그 출력
        """
        self.client = Client(model=model, provider=provider)
        self.registry = ToolRegistry()

        # 도구 등록
        if tools:
            for tool in tools:
                self.registry.add_tool(tool)

        self.max_iterations = max_iterations
        self.verbose = verbose

    async def run(self, task: str) -> AgentResult:
        """
        에이전트 실행

        Args:
            task: 수행할 작업

        Returns:
            AgentResult: 실행 결과
        """
        steps = []
        step_number = 0

        # 도구 설명 생성
        tools_description = self._format_tools()

        # 초기 프롬프트
        prompt = self.REACT_PROMPT.format(tools_description=tools_description, task=task)

        messages = [{"role": "user", "content": prompt}]
        conversation_history = prompt

        try:
            while step_number < self.max_iterations:
                step_number += 1

                if self.verbose:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Step {step_number}")
                    logger.info(f"{'='*60}")

                # LLM 호출
                response = await self.client.chat(messages, temperature=0.0)
                content = response.content

                if self.verbose:
                    logger.info(f"LLM Response:\n{content}")

                # 응답 파싱
                step = self._parse_response(content, step_number)
                steps.append(step)

                # 최종 답변인 경우
                if step.is_final and step.final_answer:
                    return AgentResult(
                        answer=step.final_answer, steps=steps, total_steps=step_number, success=True
                    )

                # 도구 실행
                if step.action and step.action_input:
                    observation = self._execute_tool(step.action, step.action_input)
                    step.observation = observation

                    if self.verbose:
                        logger.info(f"Observation: {observation}")

                    # 대화 히스토리 업데이트
                    conversation_history += f"\n\n{content}\nObservation: {observation}"
                    messages = [
                        {"role": "user", "content": conversation_history + "\n\nContinue..."}
                    ]

            # 최대 반복 도달
            return AgentResult(
                answer="Maximum iterations reached without final answer",
                steps=steps,
                total_steps=step_number,
                success=False,
                error="Max iterations exceeded",
            )

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return AgentResult(
                answer="", steps=steps, total_steps=step_number, success=False, error=str(e)
            )

    def _format_tools(self) -> str:
        """도구 목록을 문자열로 포맷"""
        tools = self.registry.get_all()
        if not tools:
            return "No tools available"

        lines = []
        for tool in tools:
            params = ", ".join(f"{p.name}: {p.type}" for p in tool.parameters)
            lines.append(f"- {tool.name}({params}): {tool.description}")

        return "\n".join(lines)

    def _parse_response(self, content: str, step_number: int) -> AgentStep:
        """LLM 응답 파싱"""
        step = AgentStep(step_number=step_number, thought="")

        # Thought 추출
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|$)", content, re.DOTALL
        )
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Final Answer 체크
        final_match = re.search(r"Final Answer:\s*(.+?)$", content, re.DOTALL)
        if final_match:
            step.is_final = True
            step.final_answer = final_match.group(1).strip()
            return step

        # Action 추출
        action_match = re.search(r"Action:\s*(\w+)", content)
        if action_match:
            step.action = action_match.group(1).strip()

        # Action Input 추출 (JSON)
        input_match = re.search(r"Action Input:\s*(\{.+?\})", content, re.DOTALL)
        if input_match:
            try:
                step.action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse action input: {e}")
                step.action_input = {}

        return step

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """도구 실행"""
        try:
            result = self.registry.execute(tool_name, arguments)
            return str(result)
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}"
            logger.error(error_msg)
            return error_msg

    def add_tool(self, tool: Tool):
        """도구 추가"""
        self.registry.add_tool(tool)


# 편의 함수
async def create_agent(model: str, tools: Optional[List[Tool]] = None, **kwargs) -> Agent:
    """Agent 생성"""
    return Agent(model=model, tools=tools, **kwargs)
