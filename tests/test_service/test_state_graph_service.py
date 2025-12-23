"""
StateGraphService 테스트 - StateGraph 서비스 구현체 테스트
"""

import pytest
from unittest.mock import Mock
from pathlib import Path

from llmkit.dto.request.state_graph_request import StateGraphRequest
from llmkit.dto.response.state_graph_response import StateGraphResponse
from llmkit.domain.state_graph import END
from llmkit.service.impl.state_graph_service_impl import StateGraphServiceImpl


class TestStateGraphService:
    """StateGraphService 테스트"""

    @pytest.fixture
    def state_graph_service(self):
        """StateGraphService 인스턴스"""
        return StateGraphServiceImpl()

    @pytest.fixture
    def simple_nodes(self):
        """간단한 노드 함수들"""

        def node_a(state):
            state["value"] = state.get("value", 0) + 1
            state["path"] = state.get("path", []) + ["A"]
            return state

        def node_b(state):
            state["value"] = state.get("value", 0) + 2
            state["path"] = state.get("path", []) + ["B"]
            return state

        return {"A": node_a, "B": node_b}

    @pytest.mark.asyncio
    async def test_invoke_basic(self, state_graph_service, simple_nodes):
        """기본 StateGraph 실행 테스트"""
        request = StateGraphRequest(
            initial_state={"value": 0, "path": []},
            nodes=simple_nodes,
            edges={"A": "B", "B": END},
            entry_point="A",
        )

        response = await state_graph_service.invoke(request)

        assert response is not None
        assert isinstance(response, StateGraphResponse)
        assert response.final_state["value"] == 3  # A(+1) + B(+2)
        assert response.final_state["path"] == ["A", "B"]
        assert len(response.nodes_executed) == 2

    @pytest.mark.asyncio
    async def test_invoke_no_entry_point(self, state_graph_service):
        """Entry point 없이 실행 테스트"""
        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes={},
            entry_point=None,
        )

        with pytest.raises(ValueError, match="Entry point not set"):
            await state_graph_service.invoke(request)

    @pytest.mark.asyncio
    async def test_invoke_conditional_edges(self, state_graph_service):
        """조건부 엣지 테스트"""

        def node_start(state):
            state["count"] = state.get("count", 0) + 1
            return state

        def node_even(state):
            state["result"] = "even"
            return state

        def node_odd(state):
            state["result"] = "odd"
            return state

        def is_even(state):
            return state.get("count", 0) % 2 == 0

        nodes = {"start": node_start, "even": node_even, "odd": node_odd}
        # conditional_edges 형식: (condition_func, edge_mapping)
        # edge_mapping은 조건 결과를 키로 사용
        conditional_edges = {
            "start": (is_even, {True: "even", False: "odd"}),
        }
        edges = {"even": END, "odd": END}

        request = StateGraphRequest(
            initial_state={"count": 2},  # 짝수
            nodes=nodes,
            edges=edges,
            conditional_edges=conditional_edges,
            entry_point="start",
        )

        response = await state_graph_service.invoke(request)

        assert response is not None
        # start 노드 실행 후 조건에 따라 even 또는 odd로 이동
        # count가 2(짝수)이므로 even으로 이동
        assert "result" in response.final_state
        assert response.final_state["result"] in ["even", "odd"]
        assert "even" in response.nodes_executed or "odd" in response.nodes_executed

    @pytest.mark.asyncio
    async def test_invoke_max_iterations(self, state_graph_service):
        """최대 반복 횟수 테스트"""

        def loop_node(state):
            state["count"] = state.get("count", 0) + 1
            return state

        nodes = {"loop": loop_node}
        edges = {"loop": "loop"}  # 무한 루프

        request = StateGraphRequest(
            initial_state={"count": 0},
            nodes=nodes,
            edges=edges,
            entry_point="loop",
            max_iterations=5,
        )

        # max_iterations에 도달하면 RuntimeError 발생
        with pytest.raises(RuntimeError, match="Max iterations"):
            await state_graph_service.invoke(request)

    @pytest.mark.asyncio
    async def test_invoke_with_execution_id(self, state_graph_service, simple_nodes):
        """Execution ID 지정 테스트"""
        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes=simple_nodes,
            edges={"A": END},
            entry_point="A",
            execution_id="custom_exec_123",
        )

        response = await state_graph_service.invoke(request)

        assert response is not None
        assert response.execution_id == "custom_exec_123"

    @pytest.mark.asyncio
    async def test_invoke_node_error(self, state_graph_service):
        """노드 실행 에러 테스트"""

        def error_node(state):
            raise ValueError("Node error")
            return state

        nodes = {"error": error_node}
        edges = {"error": END}

        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes=nodes,
            edges=edges,
            entry_point="error",
        )

        with pytest.raises(ValueError, match="Node error"):
            await state_graph_service.invoke(request)

    @pytest.mark.asyncio
    def test_stream_basic(self, state_graph_service, simple_nodes):
        """기본 StateGraph 스트리밍 테스트"""
        request = StateGraphRequest(
            initial_state={"value": 0, "path": []},
            nodes=simple_nodes,
            edges={"A": "B", "B": END},
            entry_point="A",
        )

        results = []
        for node_name, state in state_graph_service.stream(request):
            results.append((node_name, state.copy()))

        assert len(results) == 2
        assert results[0][0] == "A"
        assert results[1][0] == "B"
        assert results[1][1]["value"] == 3

    @pytest.mark.asyncio
    def test_stream_no_entry_point(self, state_graph_service):
        """Entry point 없이 스트리밍 테스트"""
        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes={},
            entry_point=None,
        )

        with pytest.raises(ValueError, match="Entry point not set"):
            list(state_graph_service.stream(request))

    @pytest.mark.asyncio
    def test_stream_max_iterations(self, state_graph_service):
        """스트리밍 최대 반복 횟수 테스트"""

        def loop_node(state):
            state["count"] = state.get("count", 0) + 1
            return state

        nodes = {"loop": loop_node}
        edges = {"loop": "loop"}  # 무한 루프

        request = StateGraphRequest(
            initial_state={"count": 0},
            nodes=nodes,
            edges=edges,
            entry_point="loop",
            max_iterations=3,
        )

        # max_iterations에 도달하면 RuntimeError 발생
        with pytest.raises(RuntimeError, match="Max iterations"):
            list(state_graph_service.stream(request))

    @pytest.mark.asyncio
    async def test_invoke_with_checkpointing(self, state_graph_service, tmp_path):
        """체크포인트 포함 실행 테스트"""

        def node_a(state):
            state["value"] = 1
            return state

        nodes = {"A": node_a}
        edges = {"A": END}

        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes=nodes,
            edges=edges,
            entry_point="A",
            enable_checkpointing=True,
            checkpoint_dir=tmp_path,
        )

        response = await state_graph_service.invoke(request)

        assert response is not None
        assert response.final_state["value"] == 1


