"""
Parallel Features 테스트 - ParallelChain, ParallelStrategy, Parallel Graph 실행
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.multi_agent.strategies import ParallelStrategy
from beanllm.facade.core.chain_facade import ChainResult, ParallelChain


def _make_chain_result(output: str, success: bool = True) -> MagicMock:
    r = MagicMock(spec=ChainResult)
    r.output = output
    r.steps = []
    r.metadata = {}
    r.success = success
    r.error = None
    return r


class TestParallelChain:
    @pytest.fixture
    def mock_chains(self):
        chain1 = MagicMock()
        chain1.run = AsyncMock(return_value=_make_chain_result("Result from chain 1"))
        chain2 = MagicMock()
        chain2.run = AsyncMock(return_value=_make_chain_result("Result from chain 2"))
        chain3 = MagicMock()
        chain3.run = AsyncMock(return_value=_make_chain_result("Result from chain 3"))
        return [chain1, chain2, chain3]

    @pytest.fixture
    def parallel_chain(self, mock_chains):
        with patch("beanllm.utils.core.di_container.get_container") as mock_get:
            mock_handler = MagicMock()
            mock_factory = MagicMock()
            mock_factory.create_chain_handler.return_value = mock_handler
            mock_container = MagicMock()
            mock_container.handler_factory = mock_factory
            mock_get.return_value = mock_container

            return ParallelChain(chains=mock_chains)

    async def test_parallel_chain_runs_all_chains(
        self, parallel_chain: ParallelChain, mock_chains
    ) -> None:
        result = await parallel_chain.run(user_input="Test task")

        assert isinstance(result, ChainResult)
        # All chains should be called
        for chain in mock_chains:
            chain.run.assert_called_once()

    async def test_parallel_chain_combines_outputs(self, parallel_chain: ParallelChain) -> None:
        result = await parallel_chain.run(user_input="Test task")

        assert "Result from chain 1" in result.output
        assert "Result from chain 2" in result.output
        assert "Result from chain 3" in result.output

    async def test_parallel_chain_success_flag(self, parallel_chain: ParallelChain) -> None:
        result = await parallel_chain.run(user_input="Test task")
        assert result.success is True

    async def test_parallel_chain_with_one_failure(self, mock_chains) -> None:
        mock_chains[1].run = AsyncMock(return_value=_make_chain_result("", success=False))

        with patch("beanllm.utils.core.di_container.get_container") as mock_get:
            mock_handler = MagicMock()
            mock_factory = MagicMock()
            mock_factory.create_chain_handler.return_value = mock_handler
            mock_container = MagicMock()
            mock_container.handler_factory = mock_factory
            mock_get.return_value = mock_container

            parallel_chain = ParallelChain(chains=mock_chains)
            result = await parallel_chain.run(user_input="Test task")

        assert result.success is False

    async def test_parallel_chain_metadata_contains_outputs(
        self, parallel_chain: ParallelChain
    ) -> None:
        result = await parallel_chain.run(user_input="Test task")

        assert "outputs" in result.metadata
        assert result.metadata["count"] == 3

    async def test_parallel_chain_runs_concurrently(self) -> None:
        """Verify chains run concurrently (not sequentially)"""
        delays = []

        async def slow_chain_run(**kwargs):
            await asyncio.sleep(0.05)
            delays.append(asyncio.get_event_loop().time())
            return _make_chain_result("slow result")

        with patch("beanllm.utils.core.di_container.get_container") as mock_get:
            mock_handler = MagicMock()
            mock_factory = MagicMock()
            mock_factory.create_chain_handler.return_value = mock_handler
            mock_container = MagicMock()
            mock_container.handler_factory = mock_factory
            mock_get.return_value = mock_container

            chains = []
            for _ in range(3):
                chain = MagicMock()
                chain.run = slow_chain_run
                chains.append(chain)

            parallel_chain = ParallelChain(chains=chains)

        import time

        start = time.time()
        result = await parallel_chain.run(user_input="concurrent test")
        elapsed = time.time() - start

        # If running concurrently, should complete in ~0.05s, not ~0.15s
        assert elapsed < 0.15
        assert result.success is True


class TestParallelStrategyAdvanced:
    async def test_parallel_strategy_all_agents_called(self) -> None:
        strategy = ParallelStrategy(aggregation="all")
        call_count = [0]

        async def count_and_respond(task: str, **kwargs):
            call_count[0] += 1
            r = MagicMock()
            r.answer = f"response {call_count[0]}"
            return r

        agents = []
        for _ in range(4):
            agent = MagicMock()
            agent.run = count_and_respond
            agents.append(agent)

        result = await strategy.execute(agents=agents, task="parallel task")
        assert call_count[0] == 4
        assert isinstance(result, dict)

    async def test_parallel_strategy_vote_aggregation(self) -> None:
        strategy = ParallelStrategy(aggregation="vote")

        # 3 agents vote yes, 1 votes no
        responses = ["yes", "yes", "yes", "no"]
        agents = []
        for resp in responses:
            agent = MagicMock()
            r = MagicMock()
            r.answer = resp
            agent.run = AsyncMock(return_value=r)
            agents.append(agent)

        result = await strategy.execute(agents=agents, task="Should we proceed?")
        assert isinstance(result, dict)
        # Result should reflect majority "yes"

    async def test_parallel_strategy_executes_concurrently(self) -> None:
        """Verify agents run in parallel"""
        start_times = []

        async def timed_agent(task: str, **kwargs):
            start_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.03)
            r = MagicMock()
            r.answer = "result"
            return r

        strategy = ParallelStrategy()
        agents = []
        for _ in range(3):
            agent = MagicMock()
            agent.run = timed_agent
            agents.append(agent)

        import time

        start = time.time()
        await strategy.execute(agents=agents, task="concurrent parallel test")
        elapsed = time.time() - start

        # Should complete in ~0.03s concurrently, not ~0.09s sequentially
        assert elapsed < 0.09
