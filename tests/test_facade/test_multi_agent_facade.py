"""
Multi-Agent Facade 테스트 - Multi-Agent 인터페이스 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from beanllm.facade.multi_agent_facade import MultiAgentCoordinator
    from beanllm.facade.agent_facade import Agent

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="MultiAgentCoordinator not available")
class TestMultiAgentFacade:
    """MultiAgentCoordinator Facade 테스트"""

    @pytest.fixture
    def coordinator(self):
        """MultiAgentCoordinator 인스턴스 (Handler를 Mock으로 교체)"""
        with patch("beanllm.utils.di_container.get_container") as mock_get_container:
            mock_handler = MagicMock()
            mock_response = Mock()
            mock_response.final_result = "Multi-agent result"
            mock_response.strategy = "sequential"
            mock_response.intermediate_results = []
            mock_response.all_steps = []
            mock_response.metadata = {}

            async def mock_handle_execute(*args, **kwargs):
                return mock_response

            mock_handler.handle_execute = MagicMock(side_effect=mock_handle_execute)

            mock_handler_factory = Mock()
            mock_handler_factory.create_multi_agent_handler.return_value = mock_handler

            mock_container = Mock()
            mock_container.handler_factory = mock_handler_factory
            mock_get_container.return_value = mock_container

            agents = {"agent1": Agent(model="gpt-4o-mini")}
            coordinator = MultiAgentCoordinator(agents=agents)
            return coordinator

    @pytest.mark.asyncio
    async def test_execute_sequential(self, coordinator):
        """순차 실행 테스트"""
        result = await coordinator.execute_sequential(
            task="Collaborative task",
            agent_order=["agent1"],
        )

        assert isinstance(result, dict)
        assert result["final_result"] == "Multi-agent result"
        assert result["strategy"] == "sequential"
        assert coordinator._multi_agent_handler.handle_execute.called


