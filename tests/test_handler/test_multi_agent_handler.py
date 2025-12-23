"""
MultiAgentHandler 테스트 - Multi-Agent Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from llmkit.dto.request.multi_agent_request import MultiAgentRequest
from llmkit.dto.response.multi_agent_response import MultiAgentResponse
from llmkit.handler.multi_agent_handler import MultiAgentHandler


class TestMultiAgentHandler:
    """MultiAgentHandler 테스트"""

    @pytest.fixture
    def mock_multi_agent_service(self):
        """Mock MultiAgentService"""
        service = Mock()
        service.execute_sequential = AsyncMock(
            return_value=MultiAgentResponse(
                final_result="Sequential result",
                strategy="sequential",
            )
        )
        service.execute_parallel = AsyncMock(
            return_value=MultiAgentResponse(
                final_result="Parallel result",
                strategy="parallel",
            )
        )
        service.execute_hierarchical = AsyncMock(
            return_value=MultiAgentResponse(
                final_result="Hierarchical result",
                strategy="hierarchical",
            )
        )
        service.execute_debate = AsyncMock(
            return_value=MultiAgentResponse(
                final_result="Debate result",
                strategy="debate",
            )
        )
        return service

    @pytest.fixture
    def multi_agent_handler(self, mock_multi_agent_service):
        """MultiAgentHandler 인스턴스"""
        return MultiAgentHandler(multi_agent_service=mock_multi_agent_service)

    @pytest.fixture
    def mock_agent(self):
        """Mock Agent"""
        agent = Mock()
        agent.id = "agent_1"
        agent.run = AsyncMock(return_value=Mock(result="Agent result"))
        return agent

    @pytest.mark.asyncio
    async def test_handle_execute_sequential(self, multi_agent_handler, mock_agent):
        """Sequential 전략 실행 테스트"""
        response = await multi_agent_handler.handle_execute(
            strategy="sequential",
            task="Test task",
            agents=[mock_agent],
            agent_order=["agent_1"],
        )

        assert response is not None
        assert isinstance(response, MultiAgentResponse)
        assert response.strategy == "sequential"
        multi_agent_handler._multi_agent_service.execute_sequential.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_execute_parallel(self, multi_agent_handler, mock_agent):
        """Parallel 전략 실행 테스트"""
        response = await multi_agent_handler.handle_execute(
            strategy="parallel",
            task="Test task",
            agents=[mock_agent],
            agent_ids=["agent_1"],
            aggregation="vote",
        )

        assert response is not None
        assert response.strategy == "parallel"
        multi_agent_handler._multi_agent_service.execute_parallel.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_execute_hierarchical(self, multi_agent_handler, mock_agent):
        """Hierarchical 전략 실행 테스트"""
        response = await multi_agent_handler.handle_execute(
            strategy="hierarchical",
            task="Test task",
            agents=[mock_agent],
            manager_id="manager_1",
            worker_ids=["worker_1", "worker_2"],
        )

        assert response is not None
        assert response.strategy == "hierarchical"
        multi_agent_handler._multi_agent_service.execute_hierarchical.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_execute_debate(self, multi_agent_handler, mock_agent):
        """Debate 전략 실행 테스트"""
        judge_agent = Mock()
        judge_agent.id = "judge_1"

        response = await multi_agent_handler.handle_execute(
            strategy="debate",
            task="Test task",
            agents=[mock_agent],
            agent_ids=["agent_1"],
            rounds=3,
            judge_id="judge_1",
            agents_dict={"judge_1": judge_agent},
        )

        assert response is not None
        assert response.strategy == "debate"
        multi_agent_handler._multi_agent_service.execute_debate.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_execute_unknown_strategy(self, multi_agent_handler):
        """알 수 없는 전략 에러 테스트"""
        with pytest.raises(ValueError, match="Unknown strategy"):
            await multi_agent_handler.handle_execute(
                strategy="unknown",
                task="Test task",
            )

    @pytest.mark.asyncio
    async def test_handle_execute_validation_error(self, multi_agent_handler):
        """입력 검증 에러 테스트"""
        # strategy가 없으면 검증 에러
        with pytest.raises(ValueError):
            await multi_agent_handler.handle_execute(
                strategy="",  # 빈 문자열
                task="Test task",
            )

    @pytest.mark.asyncio
    async def test_handle_execute_extra_params(self, multi_agent_handler, mock_agent):
        """추가 파라미터 포함 테스트"""
        response = await multi_agent_handler.handle_execute(
            strategy="sequential",
            task="Test task",
            agents=[mock_agent],
            extra_param="value",
        )

        assert response is not None
        # extra_params가 DTO에 포함되었는지 확인
        call_args = multi_agent_handler._multi_agent_service.execute_sequential.call_args[0][0]
        assert "extra_param" in call_args.extra_params


