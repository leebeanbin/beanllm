"""
StateGraph Facade 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.state_graph_facade import StateGraph
    from llmkit.domain.state_graph import END
    from llmkit.domain.graph.graph_state import GraphState
    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="StateGraph Facade not available")
class TestStateGraph:
    @pytest.fixture
    def graph(self):
        with patch("llmkit.utils.di_container.get_container") as mock_get_container:
            from unittest.mock import AsyncMock
            mock_handler = MagicMock()
            mock_response = Mock()
            mock_response.final_state = GraphState(data={"result": "test"})
            mock_response.visited_nodes = ["node1"]
            mock_response.metadata = {}
            async def mock_handle_invoke(*args, **kwargs):
                return mock_response
            mock_handler.handle_invoke = AsyncMock(side_effect=mock_handle_invoke)

            # stream mock - generator 함수 (node_name, state) 튜플 반환
            def mock_handle_stream(*args, **kwargs):
                yield ("node1", {"step": 1})  # state는 Dict
                yield ("node2", {"step": 2})
            mock_handler.handle_stream = MagicMock(return_value=mock_handle_stream())

            mock_handler_factory = Mock()
            mock_handler_factory.create_state_graph_handler.return_value = mock_handler

            mock_container = Mock()
            mock_container.handler_factory = mock_handler_factory
            mock_get_container.return_value = mock_container

            graph = StateGraph()
            # 노드와 엣지 설정
            graph.nodes["node1"] = lambda state: state
            graph.entry_point = "node1"
            return graph

    @pytest.mark.asyncio
    async def test_invoke(self, graph):
        result = await graph.invoke({"input": "test"})
        # response.final_state가 GraphState일 수도 있으므로 둘 다 확인
        if isinstance(result, dict):
            assert result == {"result": "test"}
        else:
            # GraphState 객체인 경우
            assert hasattr(result, 'data')
            assert result.data == {"result": "test"}
        assert graph._state_graph_handler.handle_invoke.called

    def test_stream(self, graph):
        results = list(graph.stream({"input": "test"}))
        assert len(results) == 2
        # stream은 (node_name, state) 튜플을 반환
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
        assert graph._state_graph_handler.handle_stream.called

    def test_add_node(self, graph):
        def new_node(state):
            return state
        graph.add_node("node2", new_node)
        assert "node2" in graph.nodes

    def test_add_edge(self, graph):
        graph.add_node("node2", lambda state: state)
        graph.add_edge("node1", "node2")
        assert graph.edges["node1"] == "node2"

    def test_add_edge_to_end(self, graph):
        graph.add_edge("node1", END)
        assert graph.edges["node1"] == END

    def test_set_entry_point(self, graph):
        graph.add_node("node2", lambda state: state)
        graph.set_entry_point("node2")
        assert graph.entry_point == "node2"


