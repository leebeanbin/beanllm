"""
StateGraph Facade 테스트
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

try:
    from beanllm.domain.graph.graph_state import GraphState
    from beanllm.domain.state_graph import END
    from beanllm.facade.state_graph_facade import StateGraph

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="StateGraph Facade not available")
class TestStateGraph:
    @pytest.fixture
    def graph(self):
        with patch("beanllm.utils.di_container.get_container") as mock_get_container:
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
            assert hasattr(result, "data")
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


# ---------------------------------------------------------------------------
# Tests using correct import paths
# ---------------------------------------------------------------------------


def _make_sg(**kwargs):
    from beanllm.facade.advanced.state_graph_facade import StateGraph

    mock_handler = MagicMock()
    mock_handler_factory = MagicMock()
    mock_handler_factory.create_state_graph_handler.return_value = mock_handler
    mock_container = MagicMock()
    mock_container.handler_factory = mock_handler_factory

    with patch("beanllm.utils.core.di_container.get_container", return_value=mock_container):
        graph = StateGraph(**kwargs)
    return graph, mock_handler


class TestStateGraphNodes:
    def test_add_node_stores_function(self):
        graph, _ = _make_sg()
        fn = lambda s: s
        graph.add_node("step1", fn)
        assert graph.nodes["step1"] is fn

    def test_add_duplicate_node_raises(self):
        graph, _ = _make_sg()
        graph.add_node("step1", lambda s: s)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node("step1", lambda s: s)

    def test_add_edge_to_end(self):
        from beanllm.domain.state_graph import END

        graph, _ = _make_sg()
        graph.add_node("step1", lambda s: s)
        graph.add_edge("step1", END)
        assert graph.edges["step1"] is END

    def test_add_edge_nonexistent_raises(self):
        graph, _ = _make_sg()
        with pytest.raises(ValueError, match="not found"):
            graph.add_edge("ghost", END)

    def test_set_entry_point(self):
        graph, _ = _make_sg()
        graph.add_node("start", lambda s: s)
        graph.set_entry_point("start")
        assert graph.entry_point == "start"

    def test_set_entry_point_nonexistent_raises(self):
        graph, _ = _make_sg()
        with pytest.raises(ValueError, match="not found"):
            graph.set_entry_point("nonexistent")


class TestStateGraphConditionalEdge:
    def test_add_conditional_edge_stores_func(self):
        graph, _ = _make_sg()
        graph.add_node("router", lambda s: s)
        graph.add_node("pathA", lambda s: s)
        fn = lambda s: "pathA"
        mapping = {"pathA": "pathA"}
        graph.add_conditional_edge("router", fn, mapping)
        stored_fn, stored_map = graph.conditional_edges["router"]
        assert stored_fn is fn
        assert stored_map == mapping

    def test_add_conditional_edge_nonexistent_raises(self):
        graph, _ = _make_sg()
        with pytest.raises(ValueError, match="not found"):
            graph.add_conditional_edge("ghost", lambda s: "next")


class TestStateGraphInvoke:
    async def test_invoke_returns_final_state(self):
        from unittest.mock import AsyncMock

        graph, mock_handler = _make_sg()
        graph.add_node("step1", lambda s: s)
        graph.set_entry_point("step1")

        expected = {"result": "ok"}
        mock_resp = MagicMock()
        mock_resp.final_state = expected
        mock_handler.handle_invoke = AsyncMock(return_value=mock_resp)

        result = await graph.invoke({"input": "test"})
        assert result == expected


class TestStateGraphStream:
    def test_stream_yields_tuples(self):
        graph, mock_handler = _make_sg()
        mock_handler.handle_stream.return_value = [
            ("step1", {"v": 1}),
            ("step2", {"v": 2}),
        ]
        items = list(graph.stream({"v": 0}))
        assert items == [("step1", {"v": 1}), ("step2", {"v": 2})]


class TestStateGraphVisualize:
    def test_visualize_contains_nodes(self):
        from beanllm.domain.state_graph import END

        graph, _ = _make_sg()
        graph.add_node("process", lambda s: s)
        graph.add_edge("process", END)
        graph.set_entry_point("process")
        viz = graph.visualize()
        assert "process" in viz
        assert "END" in viz

    def test_visualize_conditional_edges(self):
        graph, _ = _make_sg()
        graph.add_node("router", lambda s: s)
        graph.add_node("pathA", lambda s: s)
        graph.add_conditional_edge("router", lambda s: "pathA", {"pathA": "pathA"})
        viz = graph.visualize()
        assert "conditional" in viz


class TestCreateStateGraph:
    def test_creates_state_graph_with_config(self):
        from beanllm.facade.advanced.state_graph_facade import StateGraph, create_state_graph

        mock_handler = MagicMock()
        mock_handler_factory = MagicMock()
        mock_handler_factory.create_state_graph_handler.return_value = mock_handler
        mock_container = MagicMock()
        mock_container.handler_factory = mock_handler_factory

        with patch("beanllm.utils.core.di_container.get_container", return_value=mock_container):
            graph = create_state_graph(state_schema=dict, debug=True)

        assert isinstance(graph, StateGraph)
        assert graph.config.debug is True
