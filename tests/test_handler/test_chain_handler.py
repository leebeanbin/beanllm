"""
ChainHandler 테스트 - Chain Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.chain_request import ChainRequest
from llmkit.dto.response.chain_response import ChainResponse
from llmkit.handler.chain_handler import ChainHandler


class TestChainHandler:
    """ChainHandler 테스트"""

    @pytest.fixture
    def mock_chain_service(self):
        """Mock ChainService"""
        from llmkit.service.chain_service import IChainService

        service = Mock(spec=IChainService)

        # Mock execute method which is the actual method called by handler
        async def mock_execute(request):
            if request.chain_type == "basic":
                return ChainResponse(output="Chain output")
            elif request.chain_type == "prompt":
                return ChainResponse(output="Prompt chain output")
            elif request.chain_type == "sequential":
                return ChainResponse(output="Sequential chain output")
            elif request.chain_type == "parallel":
                return ChainResponse(output="Parallel chain output")
            else:
                raise ValueError(f"Unknown chain type: {request.chain_type}")

        service.execute = AsyncMock(side_effect=mock_execute)
        return service

    @pytest.fixture
    def chain_handler(self, mock_chain_service):
        """ChainHandler 인스턴스"""
        return ChainHandler(chain_service=mock_chain_service)

    @pytest.mark.asyncio
    async def test_handle_run_basic(self, chain_handler):
        """기본 Chain 실행 테스트"""
        response = await chain_handler.handle_run(
            chain_type="basic",
            user_input="Hello",
        )

        assert response is not None
        assert isinstance(response, ChainResponse)
        assert response.output == "Chain output"
        chain_handler._chain_service.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_run_prompt(self, chain_handler):
        """Prompt Chain 실행 테스트"""
        response = await chain_handler.handle_run(
            chain_type="prompt",
            template="Hello {name}",
            template_vars={"name": "World"},
        )

        assert response is not None
        assert response.output == "Prompt chain output"
        chain_handler._chain_service.execute.assert_called()

    @pytest.mark.asyncio
    async def test_handle_run_sequential(self, chain_handler):
        """Sequential Chain 실행 테스트"""
        mock_chain = Mock()
        response = await chain_handler.handle_run(
            chain_type="sequential",
            chains=[mock_chain],
        )

        assert response is not None
        assert response.output == "Sequential chain output"
        chain_handler._chain_service.execute.assert_called()

    @pytest.mark.asyncio
    async def test_handle_run_parallel(self, chain_handler):
        """Parallel Chain 실행 테스트"""
        mock_chain = Mock()
        response = await chain_handler.handle_run(
            chain_type="parallel",
            chains=[mock_chain],
        )

        assert response is not None
        assert response.output == "Parallel chain output"
        chain_handler._chain_service.execute.assert_called()

    @pytest.mark.asyncio
    async def test_handle_run_unknown_type(self, chain_handler):
        """알 수 없는 Chain 타입 에러 테스트"""
        with pytest.raises(ValueError, match="Unknown chain type"):
            await chain_handler.handle_run(
                chain_type="unknown",
            )

    @pytest.mark.asyncio
    async def test_handle_run_with_model(self, chain_handler):
        """모델 파라미터 포함 테스트"""
        response = await chain_handler.handle_run(
            chain_type="basic",
            user_input="Hello",
            model="gpt-4",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_memory(self, chain_handler):
        """메모리 설정 포함 테스트"""
        response = await chain_handler.handle_run(
            chain_type="basic",
            user_input="Hello",
            memory_type="buffer",
            memory_config={"max_size": 10},
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_tools(self, chain_handler):
        """도구 포함 테스트"""
        mock_tool = Mock()
        response = await chain_handler.handle_run(
            chain_type="basic",
            user_input="Hello",
            tools=[mock_tool],
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_with_verbose(self, chain_handler):
        """Verbose 옵션 테스트"""
        response = await chain_handler.handle_run(
            chain_type="basic",
            user_input="Hello",
            verbose=True,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_handle_run_extra_params(self, chain_handler):
        """추가 파라미터 포함 테스트"""
        response = await chain_handler.handle_run(
            chain_type="basic",
            user_input="Hello",
            extra_param="value",
        )

        assert response is not None
        # extra_params가 DTO에 포함되었는지 확인
        call_args = chain_handler._chain_service.execute.call_args[0][0]
        assert "extra_param" in call_args.extra_params


