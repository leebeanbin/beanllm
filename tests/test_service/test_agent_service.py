"""
AgentService 테스트 - 에이전트 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.agent_request import AgentRequest
from llmkit.dto.response.agent_response import AgentResponse
from llmkit.service.impl.agent_service_impl import AgentServiceImpl


class TestAgentService:
    """AgentService 테스트"""

    @pytest.fixture
    def mock_chat_service(self):
        """Mock ChatService"""
        service = Mock()
        return service

    @pytest.fixture
    def mock_tool_registry(self):
        """Mock ToolRegistry"""
        registry = Mock()
        return registry

    @pytest.fixture
    def agent_service(self, mock_chat_service, mock_tool_registry):
        """AgentService 인스턴스"""
        return AgentServiceImpl(
            chat_service=mock_chat_service,
            tool_registry=mock_tool_registry,
        )

    @pytest.mark.asyncio
    async def test_run_basic(self, agent_service):
        """기본 에이전트 실행 테스트"""
        # Mock LLM 응답 - 최종 답변 포함
        from llmkit.dto.response.chat_response import ChatResponse

        final_answer_response = ChatResponse(
            content="""
Thought: I need to answer this question.
Final Answer: The answer is 42.
""",
            model="gpt-4o-mini",
            provider="openai",
        )
        # tool_registry가 None이어도 작동하도록
        agent_service._tool_registry = None
        agent_service._chat_service.chat = AsyncMock(return_value=final_answer_response)

        request = AgentRequest(
            task="What is the answer to life?",
            model="gpt-4o-mini",
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        assert isinstance(response, AgentResponse)
        assert response.success is True
        assert response.answer == "The answer is 42."
        assert response.total_steps == 1
        assert len(response.steps) == 1

    @pytest.mark.asyncio
    async def test_run_with_tool_execution(self, agent_service):
        """도구 실행 포함 에이전트 실행 테스트"""
        # Mock 도구
        mock_tool = Mock()
        mock_tool.name = "calculator"
        mock_tool.description = "Calculate math expressions"
        mock_tool.parameters = []

        agent_service._tool_registry.get_all = Mock(return_value=[mock_tool])
        agent_service._tool_registry.execute = Mock(return_value="15")

        # Step 1: Action 요청
        action_response = Mock()
        action_response.content = """
Thought: I need to calculate 5 * 3.
Action: calculator
Action Input: {"expression": "5 * 3"}
"""
        # Step 2: Final Answer
        final_response = Mock()
        final_response.content = """
Thought: I got the result, now I can answer.
Final Answer: The result is 15.
"""

        agent_service._chat_service.chat = AsyncMock(side_effect=[action_response, final_response])

        request = AgentRequest(
            task="Calculate 5 * 3",
            model="gpt-4o-mini",
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        assert response.success is True
        assert response.answer == "The result is 15."
        assert response.total_steps == 2
        assert len(response.steps) == 2
        # 도구가 실행되었는지 확인
        agent_service._tool_registry.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_max_steps_reached(self, agent_service):
        """최대 반복 횟수 도달 테스트"""
        # 최종 답변 없이 계속 Action만 반환
        action_response = Mock()
        action_response.content = """
Thought: I need to do something.
Action: calculator
Action Input: {"expression": "1 + 1"}
"""

        agent_service._chat_service.chat = AsyncMock(return_value=action_response)
        agent_service._tool_registry.get_all = Mock(return_value=[])
        agent_service._tool_registry.execute = Mock(return_value="2")

        request = AgentRequest(
            task="Complex task",
            model="gpt-4o-mini",
            max_steps=3,  # 3번만 반복
        )

        response = await agent_service.run(request)

        assert response is not None
        assert response.success is False
        assert "Maximum iterations reached" in response.answer
        assert response.total_steps == 3
        assert response.error == "Max iterations exceeded"
        # ChatService가 max_steps만큼 호출되었는지 확인
        assert agent_service._chat_service.chat.call_count == 3

    @pytest.mark.asyncio
    async def test_run_no_tools(self, agent_service):
        """도구가 없는 경우 테스트"""
        agent_service._tool_registry = None

        final_answer_response = Mock()
        final_answer_response.content = """
Thought: I can answer without tools.
Final Answer: The answer is simple.
"""
        agent_service._chat_service.chat = AsyncMock(return_value=final_answer_response)

        request = AgentRequest(
            task="Simple question",
            model="gpt-4o-mini",
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        assert response.success is True
        # 도구 없이도 실행되어야 함

    @pytest.mark.asyncio
    async def test_run_with_system_prompt(self, agent_service):
        """시스템 프롬프트 포함 에이전트 실행 테스트"""
        from llmkit.dto.response.chat_response import ChatResponse

        final_answer_response = ChatResponse(
            content="""
Thought: I should follow the system prompt.
Final Answer: Following instructions.
""",
            model="gpt-4o-mini",
            provider="openai",
        )
        agent_service._tool_registry = None
        agent_service._chat_service.chat = AsyncMock(return_value=final_answer_response)

        request = AgentRequest(
            task="Follow instructions",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant",
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        # ChatService가 system_prompt로 호출되었는지 확인
        call_args = agent_service._chat_service.chat.call_args[0][0]
        assert call_args.system == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_run_with_temperature(self, agent_service):
        """Temperature 파라미터 포함 에이전트 실행 테스트"""
        from llmkit.dto.response.chat_response import ChatResponse

        final_answer_response = ChatResponse(
            content="""
Thought: I should be creative.
Final Answer: Creative answer.
""",
            model="gpt-4o-mini",
            provider="openai",
        )
        agent_service._tool_registry = None
        agent_service._chat_service.chat = AsyncMock(return_value=final_answer_response)

        request = AgentRequest(
            task="Creative task",
            model="gpt-4o-mini",
            temperature=0.7,
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        # ChatService가 temperature로 호출되었는지 확인
        call_args = agent_service._chat_service.chat.call_args[0][0]
        assert call_args.temperature == 0.7

    @pytest.mark.asyncio
    async def test_run_react_pattern(self, agent_service):
        """ReAct 패턴 테스트 (Thought -> Action -> Observation -> Final Answer)"""
        # Mock 도구
        mock_tool = Mock()
        mock_tool.name = "search"
        mock_tool.description = "Search the web"
        mock_tool.parameters = []

        agent_service._tool_registry.get_all = Mock(return_value=[mock_tool])
        agent_service._tool_registry.execute = Mock(return_value="Search results")

        # Step 1: Thought + Action
        step1_response = Mock()
        step1_response.content = """
Thought: I need to search for information.
Action: search
Action Input: {"query": "test"}
"""
        # Step 2: Observation 후 Final Answer
        step2_response = Mock()
        step2_response.content = """
Thought: I found the information, now I can answer.
Final Answer: Based on the search, the answer is X.
"""

        agent_service._chat_service.chat = AsyncMock(side_effect=[step1_response, step2_response])

        request = AgentRequest(
            task="Search and answer",
            model="gpt-4o-mini",
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        assert response.success is True
        assert response.total_steps == 2
        # Step 1에 observation이 포함되어 있는지 확인
        assert len(response.steps) == 2
        assert response.steps[0].get("observation") == "Search results"

    def test_parse_response_final_answer(self, agent_service):
        """응답 파싱 - 최종 답변 테스트"""
        content = """
Thought: I have the answer.
Final Answer: This is the final answer.
"""
        parsed = agent_service._parse_response(content, step_number=1)

        assert parsed["is_final"] is True
        assert parsed["final_answer"] == "This is the final answer."
        assert parsed["thought"] == "I have the answer."

    def test_parse_response_action(self, agent_service):
        """응답 파싱 - Action 포함 테스트"""
        content = """
Thought: I need to use a tool.
Action: calculator
Action Input: {"expression": "2 + 2"}
"""
        parsed = agent_service._parse_response(content, step_number=1)

        assert parsed["is_final"] is False
        assert parsed["action"] == "calculator"
        assert parsed["action_input"] == {"expression": "2 + 2"}
        assert parsed["thought"] == "I need to use a tool."

    def test_parse_response_invalid_json(self, agent_service):
        """응답 파싱 - 잘못된 JSON 테스트"""
        content = """
Thought: I need to use a tool.
Action: calculator
Action Input: {invalid json}
"""
        parsed = agent_service._parse_response(content, step_number=1)

        assert parsed["action"] == "calculator"
        # 잘못된 JSON은 빈 dict로 처리
        assert parsed["action_input"] == {}

    def test_execute_tool_success(self, agent_service):
        """도구 실행 성공 테스트"""
        agent_service._tool_registry.execute = Mock(return_value="Result")

        result = agent_service._execute_tool("calculator", {"expression": "1 + 1"})

        assert result == "Result"
        agent_service._tool_registry.execute.assert_called_once_with(
            "calculator", {"expression": "1 + 1"}
        )

    def test_execute_tool_no_registry(self, agent_service):
        """도구 레지스트리가 없는 경우 테스트"""
        agent_service._tool_registry = None

        result = agent_service._execute_tool("calculator", {"expression": "1 + 1"})

        assert "Tool registry not available" in result

    def test_execute_tool_error(self, agent_service):
        """도구 실행 에러 테스트"""
        agent_service._tool_registry.execute = Mock(side_effect=ValueError("Tool error"))

        result = agent_service._execute_tool("calculator", {"expression": "1 + 1"})

        assert "Error executing tool" in result
        assert "Tool error" in result

    def test_format_tools_with_tools(self, agent_service):
        """도구 포맷팅 - 도구가 있는 경우"""
        mock_tool = Mock()
        mock_tool.name = "calculator"
        mock_tool.description = "Calculate expressions"
        mock_param = Mock()
        mock_param.name = "expression"
        mock_param.type = "str"
        mock_tool.parameters = [mock_param]

        agent_service._tool_registry.get_all = Mock(return_value=[mock_tool])

        formatted = agent_service._format_tools()

        assert "calculator" in formatted
        assert "Calculate expressions" in formatted
        assert "expression: str" in formatted

    def test_format_tools_no_tools(self, agent_service):
        """도구 포맷팅 - 도구가 없는 경우"""
        agent_service._tool_registry.get_all = Mock(return_value=[])

        formatted = agent_service._format_tools()

        assert formatted == "No tools available"

    def test_format_tools_no_registry(self, agent_service):
        """도구 포맷팅 - 레지스트리가 없는 경우"""
        agent_service._tool_registry = None

        formatted = agent_service._format_tools()

        assert formatted == "No tools available"

    def test_format_tools_get_all_tools(self, agent_service):
        """도구 포맷팅 - get_all_tools() 메서드 사용"""
        mock_tool = Mock()
        mock_tool.name = "search"
        mock_tool.description = "Search tool"
        mock_tool.parameters = []

        # get_all()이 없고 get_all_tools()만 있는 경우
        del agent_service._tool_registry.get_all
        agent_service._tool_registry.get_all_tools = Mock(return_value={"search": mock_tool})

        formatted = agent_service._format_tools()

        assert "search" in formatted
        assert "Search tool" in formatted

    @pytest.mark.asyncio
    async def test_run_multiple_tool_calls(self, agent_service):
        """여러 도구 호출 테스트"""
        # Mock 도구들
        tool1 = Mock()
        tool1.name = "search"
        tool1.description = "Search"
        tool1.parameters = []

        tool2 = Mock()
        tool2.name = "calculator"
        tool2.description = "Calculate"
        tool2.parameters = []

        agent_service._tool_registry.get_all = Mock(return_value=[tool1, tool2])
        agent_service._tool_registry.execute = Mock(side_effect=["Search result", "4"])

        # Step 1: 첫 번째 도구
        step1 = Mock()
        step1.content = """
Thought: I need to search first.
Action: search
Action Input: {"query": "test"}
"""
        # Step 2: 두 번째 도구
        step2 = Mock()
        step2.content = """
Thought: Now I need to calculate.
Action: calculator
Action Input: {"expression": "2 + 2"}
"""
        # Step 3: Final Answer
        step3 = Mock()
        step3.content = """
Thought: I have all the information.
Final Answer: The answer is 4.
"""

        agent_service._chat_service.chat = AsyncMock(side_effect=[step1, step2, step3])

        request = AgentRequest(
            task="Search and calculate",
            model="gpt-4o-mini",
            max_steps=10,
        )

        response = await agent_service.run(request)

        assert response is not None
        assert response.success is True
        assert response.total_steps == 3
        # 두 도구가 모두 실행되었는지 확인
        assert agent_service._tool_registry.execute.call_count == 2

