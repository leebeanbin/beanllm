"""
StateGraphHandler 테스트 - StateGraph Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.state_graph_request import StateGraphRequest
from llmkit.dto.response.state_graph_response import StateGraphResponse
from llmkit.domain.state_graph import END
from llmkit.handler.state_graph_handler import StateGraphHandler


class TestStateGraphHandler:
    """StateGraphHandler 테스트"""

    @pytest.fixture
    def mock_state_graph_service(self):
        """Mock StateGraphService"""
        from llmkit.service.state_graph_service import IStateGraphService

        service = Mock(spec=IStateGraphService)
        service.invoke = AsyncMock(
            return_value=StateGraphResponse(
                final_state={"result": "completed"},
                execution_id="exec_123",
                nodes_executed=["node1", "node2"],
            )
        )

        def mock_stream(request):
            yield ("node1", {"value": 1})
            yield ("node2", {"value": 2})

        # Use side_effect to create a new generator for each call
        service.stream = Mock(side_effect=lambda request: mock_stream(request))
        return service

    @pytest.fixture
    def state_graph_handler(self, mock_state_graph_service):
        """StateGraphHandler 인스턴스"""
        return StateGraphHandler(state_graph_service=mock_state_graph_service)

    @pytest.fixture
    def simple_nodes(self):
        """간단한 노드 함수들"""

        def node_a(state):
            state["value"] = 1
            return state

        def node_b(state):
            state["value"] = 2
            return state

        return {"A": node_a, "B": node_b}

    @pytest.mark.asyncio
    async def test_handle_invoke_basic(self, state_graph_handler, simple_nodes):
        """기본 StateGraph 실행 테스트"""
        response = await state_graph_handler.handle_invoke(
            initial_state={"value": 0},
            nodes=simple_nodes,
            edges={"A": "B", "B": END},
            entry_point="A",
        )

        assert response is not None
        assert isinstance(response, StateGraphResponse)
        assert response.final_state is not None

    @pytest.mark.asyncio
    async def test_handle_invoke_with_execution_id(self, state_graph_handler, simple_nodes):
        """Execution ID 포함 테스트"""
        # execution_id가 request에 포함되면 그대로 사용
        response = await state_graph_handler.handle_invoke(
            initial_state={"value": 0},
            nodes=simple_nodes,
            edges={"A": END},
            entry_point="A",
            execution_id="custom_exec",
        )

        assert response is not None
        # execution_id는 Service에서 생성되거나 request에서 가져옴
        assert response.execution_id is not None

    @pytest.mark.asyncio
    async def test_handle_invoke_with_checkpointing(
        self, state_graph_handler, simple_nodes, tmp_path
    ):
        """체크포인트 포함 테스트"""
        response = await state_graph_handler.handle_invoke(
            initial_state={"value": 0},
            nodes=simple_nodes,
            edges={"A": END},
            entry_point="A",
            enable_checkpointing=True,
            checkpoint_dir=tmp_path,
        )

        assert response is not None

    def test_handle_stream(self, state_graph_handler, simple_nodes):
        """StateGraph 스트리밍 테스트"""
        # handle_stream은 동기 generator를 반환 (decorator가 동기 generator 지원)
        results = list(
            state_graph_handler.handle_stream(
                initial_state={"value": 0},
                nodes=simple_nodes,
                edges={"A": "B", "B": END},
                entry_point="A",
            )
        )
        assert len(results) > 0
        # stream은 (node_name, state) 튜플을 반환
        assert all(isinstance(item, tuple) and len(item) == 2 for item in results)

    @pytest.mark.asyncio
    async def test_handle_invoke_validation_error(self, state_graph_handler):
        """입력 검증 에러 테스트"""
        # initial_state가 없으면 검증 에러
        with pytest.raises(ValueError):
            await state_graph_handler.handle_invoke(
                initial_state=None,
                nodes={},
                entry_point="A",
            )
