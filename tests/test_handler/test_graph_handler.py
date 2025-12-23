"""
GraphHandler 테스트 - Graph Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.graph_request import GraphRequest
from llmkit.dto.response.graph_response import GraphResponse
from llmkit.handler.graph_handler import GraphHandler


class TestGraphHandler:
    """GraphHandler 테스트"""

    @pytest.fixture
    def mock_graph_service(self):
        """Mock GraphService"""
        service = Mock()
        service.run_graph = AsyncMock(
            return_value=GraphResponse(
                final_state={"result": "completed"},
                visited_nodes=["node1", "node2"],
            )
        )
        return service

    @pytest.fixture
    def graph_handler(self, mock_graph_service):
        """GraphHandler 인스턴스"""
        return GraphHandler(graph_service=mock_graph_service)

    @pytest.fixture
    def simple_node(self):
        """간단한 노드"""
        node = Mock()
        node.name = "test_node"
        node.execute = Mock(return_value={"result": "test"})
        return node

    @pytest.mark.asyncio
    async def test_handle_run_basic(self, graph_handler):
        """기본 Graph 실행 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
        )

        assert response is not None
        assert isinstance(response, GraphResponse)
        assert response.final_state is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_nodes(self, graph_handler, simple_node):
        """노드 포함 Graph 실행 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            nodes=[simple_node],
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_edges(self, graph_handler):
        """엣지 포함 Graph 실행 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            edges={"node1": ["node2"]},
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_conditional_edges(self, graph_handler):
        """조건부 엣지 포함 Graph 실행 테스트"""
        def condition(state):
            return state.get("value", 0) > 0

        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            conditional_edges={"node1": condition},
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_entry_point(self, graph_handler):
        """Entry point 포함 Graph 실행 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            entry_point="start_node",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_cache(self, graph_handler):
        """캐싱 옵션 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            enable_cache=False,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_verbose(self, graph_handler):
        """Verbose 옵션 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            verbose=True,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_max_iterations(self, graph_handler):
        """최대 반복 횟수 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            max_iterations=50,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_validation_error(self, graph_handler):
        """입력 검증 에러 테스트"""
        # initial_state가 없으면 검증 에러
        with pytest.raises(ValueError):
            await graph_handler.handle_run(
                initial_state=None,  # None은 검증 실패
            )

    @pytest.mark.asyncio
    async def test_handle_run_extra_params(self, graph_handler):
        """추가 파라미터 포함 테스트"""
        response = await graph_handler.handle_run(
            initial_state={"value": 0},
            extra_param="value",
        )

        assert response is not None
        # extra_params가 DTO에 포함되었는지 확인
        call_args = graph_handler._graph_service.run_graph.call_args[0][0]
        assert "extra_param" in call_args.extra_params
