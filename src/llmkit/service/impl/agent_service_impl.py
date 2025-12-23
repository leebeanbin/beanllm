"""
AgentServiceImpl - 에이전트 서비스 구현체
SOLID 원칙:
- SRP: 에이전트 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...dto.request.agent_request import AgentRequest
from ...dto.response.agent_response import AgentResponse
from ...utils.logger import get_logger
from ..agent_service import IAgentService

if TYPE_CHECKING:
    from ...service.chat_service import IChatService
    from ...service.types import ToolRegistryProtocol

logger = get_logger(__name__)


class AgentServiceImpl(IAgentService):
    """
    에이전트 서비스 구현체

    책임:
    - 에이전트 비즈니스 로직만 (ReAct 패턴 실행)
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: 에이전트 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    """

    def __init__(
        self,
        chat_service: "IChatService",
        tool_registry: Optional["ToolRegistryProtocol"] = None,
    ) -> None:
        """
        의존성 주입을 통한 생성자

        Args:
            chat_service: 채팅 서비스
            tool_registry: 도구 레지스트리 (선택적)
        """
        self._chat_service = chat_service
        self._tool_registry = tool_registry

    async def run(self, request: AgentRequest) -> AgentResponse:
        """
        에이전트 실행 (비즈니스 로직만)

        기존 agent.py의 run() 메서드를 정확히 마이그레이션

        Args:
            request: 에이전트 요청 DTO

        Returns:
            AgentResponse: 에이전트 응답 DTO

        책임:
            - ReAct 패턴 실행 비즈니스 로직
            - 도구 호출 비즈니스 로직
            - if-else/try-catch 없음 (Handler에서 처리)
        """
        from ...dto.request.chat_request import ChatRequest

        # 기존 agent.py의 run() 로직을 정확히 마이그레이션
        steps: List[Dict[str, Any]] = []
        step_number = 0

        # tool_registry 우선순위: request.tool_registry > self._tool_registry
        tool_registry = request.tool_registry or self._tool_registry

        # 도구 설명 생성 (기존: self._format_tools())
        tools_description = self._format_tools(tool_registry)

        # 초기 프롬프트 (기존: self.REACT_PROMPT.format(...))
        prompt = self.REACT_PROMPT.format(tools_description=tools_description, task=request.task)

        # messages 배열 관리 (기존과 동일)
        messages = [{"role": "user", "content": prompt}]
        conversation_history = prompt

        # 기존 while 루프 로직 정확히 마이그레이션
        while step_number < request.max_steps:
            step_number += 1

            # LLM 호출 (기존: await self.client.chat(messages, temperature=0.0))
            chat_request = ChatRequest(
                messages=messages,
                model=request.model,
                system=request.system_prompt,
                temperature=request.temperature or 0.0,
            )
            response = await self._chat_service.chat(chat_request)
            content = response.content

            # 응답 파싱 (기존: self._parse_response(content, step_number))
            parsed_step = self._parse_response(content, step_number)
            steps.append(parsed_step)

            # 최종 답변인 경우 (기존: if step.is_final and step.final_answer)
            if parsed_step.get("is_final") and parsed_step.get("final_answer"):
                return AgentResponse(
                    answer=parsed_step["final_answer"],
                    steps=steps,
                    total_steps=step_number,
                    success=True,
                )

            # 도구 실행 (기존: if step.action and step.action_input)
            action_name = parsed_step.get("action")
            action_input = parsed_step.get("action_input")
            if action_name and action_input:
                observation = self._execute_tool(action_name, action_input, tool_registry)
                parsed_step["observation"] = observation

                # 대화 히스토리 업데이트 (기존과 정확히 동일)
                conversation_history += f"\n\n{content}\nObservation: {observation}"
                messages = [{"role": "user", "content": conversation_history + "\n\nContinue..."}]

        # 최대 반복 도달 (기존과 동일)
        return AgentResponse(
            answer="Maximum iterations reached without final answer",
            steps=steps,
            total_steps=step_number,
            success=False,
            error="Max iterations exceeded",
        )

    # ReAct 프롬프트 템플릿 (기존 agent.py에서 정확히 복사)
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

    def _format_tools(self, tool_registry: Optional["ToolRegistryProtocol"] = None) -> str:
        """도구 목록을 문자열로 포맷 (기존 agent.py와 정확히 동일)"""
        # tool_registry 우선순위: 인자 > self._tool_registry
        registry = tool_registry or self._tool_registry
        
        # 기존: tools = self.registry.get_all()
        if not registry:
            return "No tools available"

        # ToolRegistry의 get_all() 메서드 사용 (기존과 동일)
        if hasattr(registry, "get_all"):
            tools = registry.get_all()
        elif hasattr(registry, "get_all_tools"):
            # Protocol의 get_all_tools()는 Dict를 반환하므로 values() 사용
            tools_dict = registry.get_all_tools()
            tools = list(tools_dict.values()) if isinstance(tools_dict, dict) else []
        else:
            return "No tools available"

        if not tools:
            return "No tools available"

        # 기존 로직 정확히 동일
        lines = []
        for tool in tools:
            params = ", ".join(f"{p.name}: {p.type}" for p in tool.parameters)
            lines.append(f"- {tool.name}({params}): {tool.description}")

        return "\n".join(lines)

    def _parse_response(self, content: str, step_number: int) -> Dict[str, Any]:
        """LLM 응답 파싱 (기존 agent.py와 정확히 동일한 로직)"""
        # 기존: step = AgentStep(step_number=step_number, thought="")
        # Dict로 변환하여 반환
        step: Dict[str, Any] = {
            "step_number": step_number,
            "thought": "",
            "action": None,
            "action_input": None,
            "observation": None,
            "is_final": False,
            "final_answer": None,
        }

        # Thought 추출 (기존과 정확히 동일)
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|$)", content, re.DOTALL
        )
        if thought_match:
            step["thought"] = thought_match.group(1).strip()

        # Final Answer 체크 (기존과 정확히 동일)
        final_match = re.search(r"Final Answer:\s*(.+?)$", content, re.DOTALL)
        if final_match:
            step["is_final"] = True
            step["final_answer"] = final_match.group(1).strip()
            return step

        # Action 추출 (기존과 정확히 동일)
        action_match = re.search(r"Action:\s*(\w+)", content)
        if action_match:
            step["action"] = action_match.group(1).strip()

        # Action Input 추출 (JSON) (기존과 정확히 동일)
        input_match = re.search(r"Action Input:\s*(\{.+?\})", content, re.DOTALL)
        if input_match:
            try:
                step["action_input"] = json.loads(input_match.group(1))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse action input: {e}")
                step["action_input"] = {}

        return step

    def _execute_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        tool_registry: Optional["ToolRegistryProtocol"] = None
    ) -> str:
        """도구 실행 (기존 agent.py와 정확히 동일한 로직)"""
        # tool_registry 우선순위: 인자 > self._tool_registry
        registry = tool_registry or self._tool_registry
        
        if not registry:
            return f"Tool registry not available. Cannot execute tool '{tool_name}'"

        # 기존: result = self.registry.execute(tool_name, arguments)
        try:
            result = registry.execute(tool_name, arguments)
            return str(result)
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {e}"
            logger.error(error_msg)
            return error_msg
