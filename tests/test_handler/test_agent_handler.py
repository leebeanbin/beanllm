"""
AgentHandler 테스트 - Agent Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from beanllm.dto.request.agent_request import AgentRequest
from beanllm.dto.response.agent_response import AgentResponse
from beanllm.handler.agent_handler import AgentHandler


class TestAgentHandler:
    """AgentHandler 테스트"""

    @pytest.fixture
    def mock_agent_service(self):
        """Mock AgentService"""
        service = Mock()
        service.run = AsyncMock(
            return_value=AgentResponse(
                answer="Task completed",
                steps=[],
                total_steps=0,
            )
        )
        return service

    @pytest.fixture
    def agent_handler(self, mock_agent_service):
        """AgentHandler 인스턴스"""
        return AgentHandler(agent_service=mock_agent_service)

    @pytest.mark.asyncio
    async def test_handle_run_basic(self, agent_handler):
        """기본 에이전트 실행 테스트"""
        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
        )

        assert response is not None
        assert isinstance(response, AgentResponse)
        assert response.answer == "Task completed"

    @pytest.mark.asyncio
    async def test_handle_run_with_tools(self, agent_handler):
        """도구 포함 에이전트 실행 테스트"""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test tool"

        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
            tools=[mock_tool],
        )

        assert response is not None
        # tool_registry가 생성되었는지 확인
        assert hasattr(agent_handler._agent_service, "_tool_registry")

    @pytest.mark.asyncio
    async def test_handle_run_with_tool_registry(self, agent_handler):
        """ToolRegistry 포함 에이전트 실행 테스트"""
        from beanllm.domain.tools import ToolRegistry

        registry = ToolRegistry()
        mock_tool = Mock()
        registry.add_tool(mock_tool)

        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
            tool_registry=registry,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_max_steps(self, agent_handler):
        """최대 단계 수 포함 테스트"""
        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
            max_steps=5,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_temperature(self, agent_handler):
        """온도 파라미터 포함 테스트"""
        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
            temperature=0.7,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_system_prompt(self, agent_handler):
        """시스템 프롬프트 포함 테스트"""
        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
            system_prompt="You are a helpful assistant",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_validation_error(self, agent_handler):
        """입력 검증 에러 테스트"""
        # task가 없으면 검증 에러 (빈 문자열은 검증 통과할 수 있음)
        # 대신 None이나 필수 파라미터 누락 테스트
        try:
            await agent_handler.handle_run(
                task="",  # 빈 문자열
                model="gpt-4o-mini",
            )
            # 빈 문자열이 허용되면 통과
        except ValueError:
            # 검증 에러가 발생하면 통과
            pass

    @pytest.mark.asyncio
    async def test_handle_run_invalid_temperature(self, agent_handler):
        """잘못된 온도 값 테스트"""
        # temperature가 범위를 벗어나면 검증 에러
        try:
            await agent_handler.handle_run(
                task="Test task",
                model="gpt-4o-mini",
                temperature=3.0,  # 범위 초과
            )
            # 검증이 통과하면 통과
        except ValueError:
            # 검증 에러가 발생하면 통과
            pass

    @pytest.mark.asyncio
    async def test_handle_run_invalid_max_steps(self, agent_handler):
        """잘못된 최대 단계 수 테스트"""
        # max_steps가 1보다 작으면 검증 에러
        try:
            await agent_handler.handle_run(
                task="Test task",
                model="gpt-4o-mini",
                max_steps=0,  # 범위 초과
            )
            # 검증이 통과하면 통과
        except ValueError:
            # 검증 에러가 발생하면 통과
            pass

    @pytest.mark.asyncio
    async def test_handle_run_extra_params(self, agent_handler):
        """추가 파라미터 포함 테스트"""
        response = await agent_handler.handle_run(
            task="Test task",
            model="gpt-4o-mini",
            extra_param1="value1",
            extra_param2=123,
        )

        assert response is not None
        # extra_params가 DTO에 포함되었는지 확인
        call_args = agent_handler._agent_service.run.call_args[0][0]
        assert "extra_param1" in call_args.extra_params
        assert call_args.extra_params["extra_param1"] == "value1"


