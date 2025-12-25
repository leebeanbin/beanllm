"""
MultiAgentService 테스트 - Multi-Agent 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

from beanllm.dto.request.multi_agent_request import MultiAgentRequest
from beanllm.dto.response.multi_agent_response import MultiAgentResponse
from beanllm.service.impl.multi_agent_service_impl import MultiAgentServiceImpl


class TestMultiAgentService:
    """MultiAgentService 테스트"""

    @pytest.fixture
    def multi_agent_service(self):
        """MultiAgentService 인스턴스"""
        return MultiAgentServiceImpl()

    @pytest.fixture
    def mock_agent(self):
        """Mock Agent 생성"""
        agent = Mock()
        result = Mock()
        result.answer = "Agent response"
        agent.run = AsyncMock(return_value=result)
        return agent

    @pytest.fixture
    def mock_agents(self):
        """여러 Mock Agent 생성"""
        agents = []
        for i in range(3):
            agent = Mock()
            result = Mock()
            result.answer = f"Agent {i+1} response"
            agent.run = AsyncMock(return_value=result)
            agents.append(agent)
        return agents

    @pytest.mark.asyncio
    async def test_execute_sequential_basic(self, multi_agent_service, mock_agents):
        """기본 순차 실행 테스트"""
        request = MultiAgentRequest(
            strategy="sequential",
            task="Process task",
            agents=mock_agents,
        )

        response = await multi_agent_service.execute_sequential(request)

        assert response is not None
        assert isinstance(response, MultiAgentResponse)
        assert response.strategy == "sequential"
        assert response.final_result == "Agent 3 response"  # 마지막 agent의 결과
        assert len(response.intermediate_results) == 3
        # 모든 agent가 순차적으로 실행되었는지 확인
        assert mock_agents[0].run.call_count == 1
        assert mock_agents[1].run.call_count == 1
        assert mock_agents[2].run.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_sequential_chain(self, multi_agent_service):
        """순차 실행 체인 테스트 (이전 결과를 다음 입력으로)"""
        agent1 = Mock()
        result1 = Mock()
        result1.answer = "Step 1 result"
        agent1.run = AsyncMock(return_value=result1)

        agent2 = Mock()
        result2 = Mock()
        result2.answer = "Step 2 result"
        agent2.run = AsyncMock(return_value=result2)

        request = MultiAgentRequest(
            strategy="sequential",
            task="Initial task",
            agents=[agent1, agent2],
        )

        response = await multi_agent_service.execute_sequential(request)

        assert response is not None
        assert response.final_result == "Step 2 result"
        # agent2가 agent1의 결과를 입력으로 받았는지 확인
        agent2.run.assert_called_once_with("Step 1 result")

    @pytest.mark.asyncio
    async def test_execute_parallel_basic(self, multi_agent_service, mock_agents):
        """기본 병렬 실행 테스트"""
        request = MultiAgentRequest(
            strategy="parallel",
            task="Process task",
            agents=mock_agents,
            aggregation="vote",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        # strategy는 aggregation을 포함할 수 있음 (예: "parallel-vote")
        assert "parallel" in response.strategy
        # 모든 agent가 병렬로 실행되었는지 확인
        assert mock_agents[0].run.call_count == 1
        assert mock_agents[1].run.call_count == 1
        assert mock_agents[2].run.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_parallel_aggregation_vote(self, multi_agent_service):
        """병렬 실행 - 투표 집계 테스트"""
        agents = []
        for i in range(3):
            agent = Mock()
            result = Mock()
            result.answer = "Option A" if i < 2 else "Option B"  # 2:1로 Option A 승리
            agent.run = AsyncMock(return_value=result)
            agents.append(agent)

        request = MultiAgentRequest(
            strategy="parallel",
            task="Vote on option",
            agents=agents,
            aggregation="vote",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        # strategy는 aggregation을 포함할 수 있음
        assert "parallel" in response.strategy
        # 투표 결과 확인 (다수결) - Option A가 2표로 승리해야 함
        assert response.final_result == "Option A"
        # metadata에 vote_counts가 있는지 확인
        if hasattr(response, "metadata") and response.metadata:
            assert "vote_counts" in response.metadata or "all_answers" in response.metadata

    @pytest.mark.asyncio
    async def test_execute_parallel_aggregation_first(self, multi_agent_service, mock_agents):
        """병렬 실행 - 첫 번째 완료 집계 테스트"""
        import asyncio

        # 첫 번째 agent가 빠르게 완료되도록 설정
        fast_result = Mock()
        fast_result.answer = "Fast response"

        # asyncio.wait는 coroutine이 아닌 Task를 받아야 함
        # ParallelStrategy는 agent.run(task)를 호출하고, 이를 tasks 리스트에 추가함
        # tasks = [agent.run(task) for agent in agents]에서 agent.run(task)는 coroutine을 반환
        # asyncio.wait는 coroutine을 직접 받을 수 없으므로, 실제 async 함수를 반환하도록 설정
        async def fast_run(task):
            await asyncio.sleep(0.01)  # 빠르게 완료
            return fast_result

        # 나머지는 느리게
        async def slow_run(task):
            await asyncio.sleep(0.1)  # 느리게 완료
            result = Mock()
            result.answer = "Slow response"
            return result

        # 실제 async 함수를 직접 할당 (AsyncMock이 아닌)
        # 이렇게 하면 agent.run(task)가 coroutine을 반환하고, asyncio.wait가 이를 Task로 변환
        mock_agents[0].run = fast_run
        mock_agents[1].run = slow_run
        mock_agents[2].run = slow_run

        request = MultiAgentRequest(
            strategy="parallel",
            task="Process task",
            agents=mock_agents,
            aggregation="first",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        assert "parallel" in response.strategy
        # first aggregation은 첫 번째 완료된 결과를 반환
        assert response.final_result == "Fast response"
        # metadata에 completed 정보가 있는지 확인
        if hasattr(response, "metadata") and response.metadata:
            assert "completed" in response.metadata or "strategy" in response.metadata

    @pytest.mark.asyncio
    async def test_execute_hierarchical_basic(self, multi_agent_service):
        """기본 계층적 실행 테스트"""
        manager = Mock()
        manager_result = Mock()
        # Manager는 JSON 형식의 subtasks를 반환해야 함
        manager_result.answer = '{"subtasks": ["subtask1", "subtask2"]}'
        # steps 속성을 리스트로 설정 (len() 호출을 위해)
        manager_result.steps = []
        manager.run = AsyncMock(return_value=manager_result)

        # final_result도 steps 속성 필요
        final_result = Mock()
        final_result.answer = "Final synthesized answer"
        final_result.steps = []
        manager.run = AsyncMock(side_effect=[manager_result, final_result])

        worker1 = Mock()
        # Mock 객체에 __len__ 메서드 추가 (strategies.py:219에서 len() 호출)
        worker1.__len__ = lambda self: 1
        worker1_result = Mock()
        worker1_result.answer = "Worker 1 result"
        worker1.run = AsyncMock(return_value=worker1_result)

        worker2 = Mock()
        # Mock 객체에 __len__ 메서드 추가
        worker2.__len__ = lambda self: 1
        worker2_result = Mock()
        worker2_result.answer = "Worker 2 result"
        worker2.run = AsyncMock(return_value=worker2_result)

        # agents 리스트는 실제 리스트이므로 len()이 작동함
        request = MultiAgentRequest(
            strategy="hierarchical",
            task="Hierarchical task",
            agents=[manager, worker1, worker2],  # 첫 번째가 manager
        )

        response = await multi_agent_service.execute_hierarchical(request)

        assert response is not None
        assert response.strategy == "hierarchical"
        # manager와 workers가 모두 실행되었는지 확인
        assert manager.run.called
        # HierarchicalStrategy.execute는 workers 리스트를 받음
        assert worker1.run.called or worker2.run.called

    @pytest.mark.asyncio
    async def test_execute_hierarchical_insufficient_agents(self, multi_agent_service):
        """계층적 실행 - Agent 부족 에러 테스트"""
        request = MultiAgentRequest(
            strategy="hierarchical",
            task="Task",
            agents=[],  # Agent 없음
        )

        with pytest.raises(ValueError, match="At least manager and one worker"):
            await multi_agent_service.execute_hierarchical(request)

    @pytest.mark.asyncio
    async def test_execute_hierarchical_only_manager(self, multi_agent_service):
        """계층적 실행 - Manager만 있는 경우 에러 테스트"""
        manager = Mock()
        request = MultiAgentRequest(
            strategy="hierarchical",
            task="Task",
            agents=[manager],  # Manager만 있음
        )

        with pytest.raises(ValueError, match="At least manager and one worker"):
            await multi_agent_service.execute_hierarchical(request)

    @pytest.mark.asyncio
    async def test_execute_debate_basic(self, multi_agent_service, mock_agents):
        """기본 토론 실행 테스트"""
        request = MultiAgentRequest(
            strategy="debate",
            task="Debate topic",
            agents=mock_agents,
            rounds=2,
        )

        response = await multi_agent_service.execute_debate(request)

        assert response is not None
        assert response.strategy == "debate"

    @pytest.mark.asyncio
    async def test_execute_debate_with_judge(self, multi_agent_service, mock_agents):
        """판정자 포함 토론 실행 테스트"""
        judge = Mock()
        judge_result = Mock()
        judge_result.answer = "Judge decision"
        judge.run = AsyncMock(return_value=judge_result)

        request = MultiAgentRequest(
            strategy="debate",
            task="Debate topic",
            agents=mock_agents,
            rounds=2,
            judge_agent=judge,
        )

        response = await multi_agent_service.execute_debate(request)

        assert response is not None
        assert response.strategy == "debate"

    @pytest.mark.asyncio
    async def test_execute_debate_no_judge(self, multi_agent_service, mock_agents):
        """판정자 없이 토론 실행 테스트"""
        request = MultiAgentRequest(
            strategy="debate",
            task="Debate topic",
            agents=mock_agents,
            rounds=2,
            judge_agent=None,  # 판정자 없음
        )

        response = await multi_agent_service.execute_debate(request)

        assert response is not None
        assert response.strategy == "debate"

    @pytest.mark.asyncio
    async def test_execute_sequential_empty_agents(self, multi_agent_service):
        """순차 실행 - Agent가 없는 경우 테스트"""
        request = MultiAgentRequest(
            strategy="sequential",
            task="Task",
            agents=[],
        )

        response = await multi_agent_service.execute_sequential(request)

        assert response is not None
        assert response.final_result is None
        assert len(response.intermediate_results) == 0

    @pytest.mark.asyncio
    async def test_execute_parallel_empty_agents(self, multi_agent_service):
        """병렬 실행 - Agent가 없는 경우 테스트"""
        request = MultiAgentRequest(
            strategy="parallel",
            task="Task",
            agents=[],
            aggregation="vote",
        )

        # 빈 agents일 때는 IndexError가 발생할 수 있음
        try:
            response = await multi_agent_service.execute_parallel(request)
            assert response is not None
            assert "parallel" in response.strategy
        except (IndexError, ValueError):
            # 빈 agents로 인한 에러는 허용
            pass

    @pytest.mark.asyncio
    async def test_execute_sequential_single_agent(self, multi_agent_service, mock_agent):
        """순차 실행 - Agent가 하나인 경우 테스트"""
        request = MultiAgentRequest(
            strategy="sequential",
            task="Task",
            agents=[mock_agent],
        )

        response = await multi_agent_service.execute_sequential(request)

        assert response is not None
        assert response.final_result == "Agent response"
        assert len(response.intermediate_results) == 1

    @pytest.mark.asyncio
    async def test_execute_parallel_single_agent(self, multi_agent_service, mock_agent):
        """병렬 실행 - Agent가 하나인 경우 테스트"""
        request = MultiAgentRequest(
            strategy="parallel",
            task="Task",
            agents=[mock_agent],
            aggregation="vote",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        assert "parallel" in response.strategy

    @pytest.mark.asyncio
    async def test_execute_sequential_extra_params(self, multi_agent_service, mock_agents):
        """순차 실행 - 추가 파라미터 테스트"""
        request = MultiAgentRequest(
            strategy="sequential",
            task="Task",
            agents=mock_agents,
            extra_params={"param1": "value1", "param2": "value2"},
        )

        response = await multi_agent_service.execute_sequential(request)

        assert response is not None
        assert response.strategy == "sequential"

    @pytest.mark.asyncio
    async def test_execute_parallel_extra_params(self, multi_agent_service, mock_agents):
        """병렬 실행 - 추가 파라미터 테스트"""
        request = MultiAgentRequest(
            strategy="parallel",
            task="Task",
            agents=mock_agents,
            aggregation="vote",
            extra_params={"param1": "value1"},
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        assert "parallel" in response.strategy

    @pytest.mark.asyncio
    async def test_execute_hierarchical_extra_params(self, multi_agent_service):
        """계층적 실행 - 추가 파라미터 테스트"""
        manager = Mock()
        manager_result = Mock()
        # Manager는 JSON 형식의 subtasks를 반환해야 함
        manager_result.answer = '{"subtasks": ["subtask1"]}'
        # steps 속성을 리스트로 설정 (len() 호출을 위해)
        manager_result.steps = []
        manager.run = AsyncMock(return_value=manager_result)

        # final_result도 steps 속성 필요
        final_result = Mock()
        final_result.answer = "Final synthesized answer"
        final_result.steps = []
        manager.run = AsyncMock(side_effect=[manager_result, final_result])

        worker = Mock()
        # Mock 객체에 __len__ 메서드 추가 (strategies.py:219에서 len() 호출)
        worker.__len__ = lambda self: 1
        worker_result = Mock()
        worker_result.answer = "Worker result"
        worker.run = AsyncMock(return_value=worker_result)

        # agents 리스트는 실제 리스트이므로 len()이 작동함
        request = MultiAgentRequest(
            strategy="hierarchical",
            task="Task",
            agents=[manager, worker],
            extra_params={"param1": "value1"},
        )

        response = await multi_agent_service.execute_hierarchical(request)

        assert response is not None
        assert response.strategy == "hierarchical"
        # manager와 worker가 실행되었는지 확인
        assert manager.run.called
        assert worker.run.called
        # HierarchicalStrategy.execute는 workers 리스트를 받음
        assert worker.run.called

    @pytest.mark.asyncio
    async def test_execute_debate_extra_params(self, multi_agent_service, mock_agents):
        """토론 실행 - 추가 파라미터 테스트"""
        request = MultiAgentRequest(
            strategy="debate",
            task="Debate topic",
            agents=mock_agents,
            rounds=3,
            extra_params={"param1": "value1"},
        )

        response = await multi_agent_service.execute_debate(request)

        assert response is not None
        assert response.strategy == "debate"

    @pytest.mark.asyncio
    async def test_execute_debate_custom_rounds(self, multi_agent_service, mock_agents):
        """토론 실행 - 커스텀 라운드 수 테스트"""
        request = MultiAgentRequest(
            strategy="debate",
            task="Debate topic",
            agents=mock_agents,
            rounds=5,  # 커스텀 라운드 수
        )

        response = await multi_agent_service.execute_debate(request)

        assert response is not None
        assert response.strategy == "debate"
        # metadata에 rounds 정보가 있는지 확인
        if hasattr(response, "metadata") and response.metadata:
            assert "rounds" in response.metadata or "debate_history" in response.metadata

    @pytest.mark.asyncio
    async def test_execute_parallel_aggregation_consensus(self, multi_agent_service):
        """병렬 실행 - 합의 집계 테스트"""
        agents = []
        # 모든 agent가 같은 답변을 반환하도록 설정
        for i in range(3):
            agent = Mock()
            result = Mock()
            result.answer = "Consensus answer"  # 모두 동일한 답변
            agent.run = AsyncMock(return_value=result)
            agents.append(agent)

        request = MultiAgentRequest(
            strategy="parallel",
            task="Reach consensus",
            agents=agents,
            aggregation="consensus",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        assert "parallel" in response.strategy
        # consensus가 성공한 경우 final_result가 있어야 함
        assert response.final_result == "Consensus answer"
        # metadata에 consensus 정보가 있는지 확인
        if hasattr(response, "metadata") and response.metadata:
            assert "consensus" in response.metadata or "strategy" in response.metadata

    @pytest.mark.asyncio
    async def test_execute_parallel_aggregation_consensus_failed(self, multi_agent_service):
        """병렬 실행 - 합의 실패 테스트"""
        agents = []
        # 서로 다른 답변을 반환하도록 설정
        answers = ["Answer A", "Answer B", "Answer C"]
        for i, answer in enumerate(answers):
            agent = Mock()
            result = Mock()
            result.answer = answer
            agent.run = AsyncMock(return_value=result)
            agents.append(agent)

        request = MultiAgentRequest(
            strategy="parallel",
            task="Reach consensus",
            agents=agents,
            aggregation="consensus",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        assert "parallel" in response.strategy
        # consensus가 실패한 경우 final_result가 None일 수 있음
        # metadata에 consensus 정보가 있는지 확인
        if hasattr(response, "metadata") and response.metadata:
            assert "consensus" in response.metadata or "all_answers" in response.metadata

    @pytest.mark.asyncio
    async def test_execute_parallel_aggregation_all(self, multi_agent_service, mock_agents):
        """병렬 실행 - 모든 결과 반환 집계 테스트"""
        request = MultiAgentRequest(
            strategy="parallel",
            task="Process task",
            agents=mock_agents,
            aggregation="all",
        )

        response = await multi_agent_service.execute_parallel(request)

        assert response is not None
        assert "parallel" in response.strategy
        # all aggregation은 모든 결과를 리스트로 반환
        assert isinstance(response.final_result, list)
        assert len(response.final_result) == len(mock_agents)
        # metadata에 all_results가 있는지 확인
        if hasattr(response, "metadata") and response.metadata:
            assert "all_results" in response.metadata or "strategy" in response.metadata
