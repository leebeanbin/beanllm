"""
Graph Facade 테스트 - Graph 인터페이스 테스트
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.graph_facade import Graph
    from llmkit.domain.graph import GraphState

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Graph not available")
class TestGraphFacade:
    """Graph Facade 테스트"""

    @pytest.fixture
    def graph(self):
        """Graph 인스턴스 (Handler를 Mock으로 교체)"""
        with patch("llmkit.facade.graph_facade.HandlerFactory") as mock_factory:
            mock_handler = MagicMock()
            mock_response = Mock()
            mock_response.final_state = {"result": "Graph result"}
            mock_response.visited_nodes = ["node1", "node2"]
            mock_response.metadata = {}

            async def mock_handle_run(*args, **kwargs):
                return mock_response

            mock_handler.handle_run = MagicMock(side_effect=mock_handle_run)

            mock_handler_factory = Mock()
            mock_handler_factory.create_graph_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory

            graph = Graph()
            graph._graph_handler = mock_handler
            return graph

    @pytest.mark.asyncio
    async def test_run(self, graph):
        """Graph 실행 테스트"""
        initial_state = {"input": "test"}
        result = await graph.run(initial_state)

        assert isinstance(result, GraphState)
        assert result.data["result"] == "Graph result"
        assert graph._graph_handler.handle_run.called


