"""
Chain Facade 테스트 - Chain 인터페이스 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from beanllm.facade.chain_facade import Chain, ChainResult
    from beanllm.facade.client_facade import Client

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Chain not available")
class TestChainFacade:
    """Chain Facade 테스트"""

    @pytest.fixture
    def mock_client(self):
        """Mock Client"""
        client = Mock(spec=Client)
        client.model = "gpt-4o-mini"
        return client

    @pytest.fixture
    def chain(self, mock_client):
        """Chain 인스턴스 (Handler를 Mock으로 교체)"""
        with patch("beanllm.utils.di_container.get_container") as mock_get_container:
            mock_handler = MagicMock()
            mock_response = Mock()
            mock_response.output = "Chain output"
            mock_response.steps = []
            mock_response.metadata = {}
            mock_response.success = True
            mock_response.error = None

            async def mock_handle_run(*args, **kwargs):
                return mock_response

            mock_handler.handle_run = MagicMock(side_effect=mock_handle_run)

            mock_handler_factory = Mock()
            mock_handler_factory.create_chain_handler.return_value = mock_handler

            mock_container = Mock()
            mock_container.handler_factory = mock_handler_factory
            mock_get_container.return_value = mock_container

            chain = Chain(mock_client)
            return chain

    @pytest.mark.asyncio
    async def test_run(self, chain):
        """Chain 실행 테스트"""
        result = await chain.run("Test input")

        assert isinstance(result, ChainResult)
        assert result.output == "Chain output"
        assert chain._chain_handler.handle_run.called


