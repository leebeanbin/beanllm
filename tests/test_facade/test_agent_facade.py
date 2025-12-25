"""
Agent Facade 테스트 - Agent 인터페이스 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from beanllm.facade.agent_facade import Agent, AgentResult

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Agent not available")
class TestAgentFacade:
    """Agent Facade 테스트"""

    @pytest.fixture
    def agent(self):
        """Agent 인스턴스 (Handler를 Mock으로 교체)"""
        with patch("beanllm.utils.di_container.get_container") as mock_get_container:
            mock_handler = MagicMock()
            mock_response = Mock()
            mock_response.answer = "Agent response"
            mock_response.steps = [{"step_number": 1, "thought": "test"}]
            mock_response.total_steps = 1
            mock_response.success = True
            mock_response.error = None

            async def mock_handle_run(*args, **kwargs):
                return mock_response

            mock_handler.handle_run = MagicMock(side_effect=mock_handle_run)

            mock_handler_factory = Mock()
            mock_handler_factory.create_agent_handler.return_value = mock_handler

            mock_container = Mock()
            mock_container.handler_factory = mock_handler_factory
            mock_get_container.return_value = mock_container

            agent = Agent(model="gpt-4o-mini")
            return agent

    @pytest.mark.asyncio
    async def test_run(self, agent):
        """Agent 실행 테스트"""
        result = await agent.run("Solve this problem")

        assert isinstance(result, AgentResult)
        assert result.answer == "Agent response"
        assert result.total_steps == 1
        assert agent._agent_handler.handle_run.called


