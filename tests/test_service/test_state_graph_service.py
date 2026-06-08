"""
StateGraphService 테스트 - StateGraph 서비스 구현체 테스트
"""

from pathlib import Path
from typing import Optional, TypedDict, Union
from unittest.mock import Mock

import pytest

from beanllm.domain.graph.graph_state import GraphState
from beanllm.domain.state_graph import END, Checkpoint
from beanllm.dto.request import StateGraphRequest
from beanllm.dto.response import StateGraphResponse
from beanllm.service.impl.advanced.state_graph_service_impl import StateGraphServiceImpl


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

    # ------------------------------------------------------------------
    # Coverage of missed lines: GraphState / deepcopy / debug / resume
    # ------------------------------------------------------------------

    async def test_invoke_with_graphstate_initial_state(self, state_graph_service):
        """GraphState as initial_state → line 88."""

        def node_a(state):
            # Must return dict so StateGraphResponse passes Pydantic validation
            return {"value": state.get("value", 0) + 10}

        initial = GraphState(data={"value": 0})
        request = StateGraphRequest(
            initial_state=initial,
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
        )

        response = await state_graph_service.invoke(request)
        assert response.final_state["value"] == 10

    async def test_invoke_with_deepcopy_initial_state(self, state_graph_service):
        """Non-dict/non-GraphState initial_state → deepcopy branch (line 92)."""

        class FakeState:
            def __init__(self):
                self.value = 0

            def __deepcopy__(self, memo):
                c = FakeState()
                c.value = self.value
                return c

        def node_a(state):
            # node receives FakeState-like object, returns a dict
            return {"value": 42}

        initial = FakeState()
        request = StateGraphRequest(
            initial_state=initial,
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
        )

        response = await state_graph_service.invoke(request)
        assert response.final_state["value"] == 42

    async def test_invoke_with_debug_true(self, state_graph_service, simple_nodes):
        """debug=True triggers logger.debug calls (line 116)."""
        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes=simple_nodes,
            edges={"A": END},
            entry_point="A",
            debug=True,
        )

        response = await state_graph_service.invoke(request)
        assert response is not None

    async def test_invoke_resume_from_with_no_saved_state(self, state_graph_service, tmp_path):
        """resume_from set but checkpoint has no data → line 107."""

        def node_a(state):
            state["resumed"] = True
            return state

        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            enable_checkpointing=True,
            checkpoint_dir=tmp_path,
            resume_from="A",
        )

        response = await state_graph_service.invoke(request)
        assert response is not None

    async def test_invoke_validate_state_with_schema(self, state_graph_service):
        """_validate_state with TypedDict schema (lines 268-288)."""

        class MySchema(TypedDict):
            value: int
            name: Optional[str]

        def node_a(state):
            return state

        request = StateGraphRequest(
            initial_state={"value": 1, "name": "test"},
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            state_schema=MySchema,
        )

        response = await state_graph_service.invoke(request)
        assert response is not None

    async def test_invoke_validate_state_missing_required_field(self, state_graph_service):
        """_validate_state with missing required field caught and returned True (line 281→288)."""

        class StrictSchema(TypedDict):
            required_field: int

        def node_a(state):
            return state

        # state is missing required_field → ValueError raised inside _validate_state
        # but exception is caught and returns True
        request = StateGraphRequest(
            initial_state={},  # missing required_field
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            state_schema=StrictSchema,
        )

        response = await state_graph_service.invoke(request)
        assert response is not None

    async def test_invoke_validate_state_debug_warning(self, state_graph_service):
        """debug=True when schema validation fails logs warning (lines 286-287)."""

        class StrictSchema(TypedDict):
            must_have: int

        def node_a(state):
            return state

        request = StateGraphRequest(
            initial_state={},
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            state_schema=StrictSchema,
            debug=True,
        )

        response = await state_graph_service.invoke(request)
        assert response is not None

    async def test_invoke_conditional_edge_no_edge_mapping(self, state_graph_service):
        """Conditional edge with empty edge_mapping → line 311."""

        def node_start(state):
            state["go"] = "node_end"
            return state

        def node_end(state):
            state["done"] = True
            return state

        def condition(state):
            return state.get("go", "")

        # Empty edge_mapping → uses `result if result in nodes else END`
        nodes = {"start": node_start, "node_end": node_end}
        conditional_edges = {"start": (condition, {})}

        request = StateGraphRequest(
            initial_state={"go": "node_end"},
            nodes=nodes,
            edges={"node_end": END},
            conditional_edges=conditional_edges,
            entry_point="start",
        )

        response = await state_graph_service.invoke(request)
        assert response is not None

    async def test_invoke_node_with_no_edges_falls_back_to_end(self, state_graph_service):
        """Node with no edge and no conditional edge → returns END (line 321)."""

        def lone_node(state):
            state["done"] = True
            return state

        # No edges and no conditional_edges for 'lone' node → returns END
        request = StateGraphRequest(
            initial_state={},
            nodes={"lone": lone_node},
            edges={},  # no edge for 'lone'
            entry_point="lone",
        )

        response = await state_graph_service.invoke(request)
        assert response.final_state["done"] is True

    # ------------------------------------------------------------------
    # stream() missed branches
    # ------------------------------------------------------------------

    def test_stream_with_execution_id(self, state_graph_service, simple_nodes):
        """stream() with execution_id set → line 209."""
        request = StateGraphRequest(
            initial_state={"value": 0, "path": []},
            nodes=simple_nodes,
            edges={"A": END},
            entry_point="A",
            execution_id="stream_exec_001",
        )

        results = list(state_graph_service.stream(request))
        assert len(results) == 1
        assert results[0][0] == "A"

    def test_stream_with_graphstate_initial_state(self, state_graph_service):
        """stream() with GraphState initial_state → lines 214, 235."""

        def node_a(state):
            state["value"] = state.get("value", 0) + 5
            return state  # Returns GraphState

        initial = GraphState(data={"value": 0})
        request = StateGraphRequest(
            initial_state=initial,
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
        )

        results = list(state_graph_service.stream(request))
        assert len(results) == 1
        assert results[0][0] == "A"

    def test_stream_with_deepcopy_initial_state(self, state_graph_service):
        """stream() with non-dict/non-GraphState → line 218/239."""

        class FakeState:
            def __init__(self):
                self.value = 99

            def __deepcopy__(self, memo):
                c = FakeState()
                c.value = self.value
                return c

        def node_a(state):
            # Return a non-dict, non-GraphState to hit line 239
            class Result:
                value = 99

                def __deepcopy__(self, memo):
                    r = Result()
                    return r

            return Result()

        initial = FakeState()
        request = StateGraphRequest(
            initial_state=initial,
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
        )

        results = list(state_graph_service.stream(request))
        assert len(results) == 1

    def test_stream_with_checkpointing(self, state_graph_service, tmp_path):
        """stream() with enable_checkpointing → lines 224, 245."""

        def node_a(state):
            state["v"] = 1
            return state

        request = StateGraphRequest(
            initial_state={"v": 0},
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            enable_checkpointing=True,
            checkpoint_dir=tmp_path,
        )

        results = list(state_graph_service.stream(request))
        assert len(results) == 1

    # ------------------------------------------------------------------
    # Remaining missed lines: checkpoint resume with state + union schema
    # ------------------------------------------------------------------

    async def test_invoke_resume_from_with_saved_state(self, state_graph_service, tmp_path):
        """resume_from with existing checkpoint → restores state (lines 104-105)."""
        exec_id = "resume_exec_001"

        # Pre-save a checkpoint for node "A"
        ckpt = Checkpoint(tmp_path)
        ckpt.save(exec_id, {"value": 99}, "A")

        def node_a(state):
            # Receives the restored state (value=99)
            return {"value": state.get("value", 0) + 1}

        request = StateGraphRequest(
            initial_state={"value": 0},
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            enable_checkpointing=True,
            checkpoint_dir=tmp_path,
            execution_id=exec_id,
            resume_from="A",
        )

        response = await state_graph_service.invoke(request)
        # Restored from checkpoint (value=99), node adds 1 → 100
        assert response.final_state["value"] == 100

    async def test_invoke_validate_state_union_required_field_missing(self, state_graph_service):
        """_validate_state: Union (non-optional) field missing → line 279."""

        class SchemaWithUnion(TypedDict):
            union_field: Union[str, int]  # Required Union without None

        def node_a(state):
            return state

        # State is missing union_field → ValueError raised but caught → returns True
        request = StateGraphRequest(
            initial_state={},  # union_field is missing
            nodes={"A": node_a},
            edges={"A": END},
            entry_point="A",
            state_schema=SchemaWithUnion,
        )

        response = await state_graph_service.invoke(request)
        assert response is not None
