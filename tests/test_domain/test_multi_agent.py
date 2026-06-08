"""
Multi-Agent Domain 테스트 - Communication Bus 및 전략 패턴
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.domain.multi_agent.communication import (
    AgentMessage,
    CommunicationBus,
    MessageType,
)
from beanllm.domain.multi_agent.strategies import (
    DebateStrategy,
    HierarchicalStrategy,
    ParallelStrategy,
    SequentialStrategy,
)


class TestAgentMessage:
    def test_create_message(self) -> None:
        msg = AgentMessage(
            sender="agent-1",
            receiver="agent-2",
            content="Hello!",
            message_type=MessageType.REQUEST,
        )
        assert msg.sender == "agent-1"
        assert msg.receiver == "agent-2"
        assert msg.content == "Hello!"
        assert msg.message_type == MessageType.REQUEST

    def test_message_has_id(self) -> None:
        msg = AgentMessage(sender="a1", content="test")
        assert msg.id is not None
        assert len(msg.id) > 0

    def test_message_has_timestamp(self) -> None:
        msg = AgentMessage(sender="a1", content="test")
        assert msg.timestamp is not None

    def test_message_reply(self) -> None:
        original = AgentMessage(
            sender="agent-1",
            receiver="agent-2",
            content="Question?",
            message_type=MessageType.REQUEST,
        )
        reply = original.reply("Answer!", message_type=MessageType.RESPONSE)
        assert reply.content == "Answer!"
        assert reply.message_type == MessageType.RESPONSE
        assert reply.reply_to == original.id

    def test_broadcast_message_no_receiver(self) -> None:
        msg = AgentMessage(
            sender="coordinator",
            receiver=None,
            content="Start task",
            message_type=MessageType.BROADCAST,
        )
        assert msg.receiver is None

    def test_message_type_enum_values(self) -> None:
        assert MessageType.REQUEST.value == "request"
        assert MessageType.RESPONSE.value == "response"
        assert MessageType.BROADCAST.value == "broadcast"


class TestCommunicationBus:
    @pytest.fixture
    def bus(self) -> CommunicationBus:
        return CommunicationBus(delivery_guarantee="at-most-once")

    def test_subscribe(self, bus: CommunicationBus) -> None:
        callback = MagicMock()
        bus.subscribe("agent-1", callback)
        assert "agent-1" in bus.subscribers

    def test_unsubscribe(self, bus: CommunicationBus) -> None:
        callback = MagicMock()
        bus.subscribe("agent-1", callback)
        bus.unsubscribe("agent-1")
        assert "agent-1" not in bus.subscribers

    async def test_publish_calls_subscriber(self, bus: CommunicationBus) -> None:
        received = []

        def callback(msg: AgentMessage) -> None:
            received.append(msg)

        bus.subscribe("agent-1", callback)

        msg = AgentMessage(
            sender="agent-0",
            receiver="agent-1",
            content="task",
            message_type=MessageType.REQUEST,
        )
        await bus.publish(msg)

        assert len(received) == 1
        assert received[0].content == "task"

    async def test_publish_broadcast_calls_all(self, bus: CommunicationBus) -> None:
        received_a = []
        received_b = []

        bus.subscribe("agent-a", lambda m: received_a.append(m))
        bus.subscribe("agent-b", lambda m: received_b.append(m))

        msg = AgentMessage(
            sender="coordinator",
            receiver=None,  # broadcast
            content="start",
            message_type=MessageType.BROADCAST,
        )
        await bus.publish(msg)

        assert len(received_a) == 1
        assert len(received_b) == 1

    async def test_publish_no_broadcast_skips_others(self, bus: CommunicationBus) -> None:
        received_a = []
        received_b = []

        bus.subscribe("agent-a", lambda m: received_a.append(m))
        bus.subscribe("agent-b", lambda m: received_b.append(m))

        msg = AgentMessage(
            sender="sender",
            receiver="agent-a",  # only to agent-a
            content="private",
            message_type=MessageType.REQUEST,
        )
        await bus.publish(msg)

        assert len(received_a) == 1
        assert len(received_b) == 0

    def test_get_history(self, bus: CommunicationBus) -> None:
        history = bus.get_history()
        assert isinstance(history, list)

    def test_delivery_guarantee_set(self) -> None:
        bus = CommunicationBus(delivery_guarantee="exactly-once")
        assert bus.delivery_guarantee == "exactly-once"


def _make_agent_response(answer: str) -> MagicMock:
    """Agents return objects with .answer attribute"""
    r = MagicMock()
    r.answer = answer
    r.content = answer
    r.__str__ = lambda self: answer
    return r


class TestSequentialStrategy:
    @pytest.fixture
    def strategy(self) -> SequentialStrategy:
        return SequentialStrategy()

    async def test_execute_sequential(self, strategy: SequentialStrategy) -> None:
        agent1 = AsyncMock()
        agent1.run.return_value = _make_agent_response("result from agent1")
        agent2 = AsyncMock()
        agent2.run.return_value = _make_agent_response("result from agent2")

        result = await strategy.execute(
            agents=[agent1, agent2],
            task="test task",
        )
        assert isinstance(result, dict)
        assert "final_result" in result or len(result) > 0

    async def test_execute_single_agent(self, strategy: SequentialStrategy) -> None:
        agent = AsyncMock()
        agent.run.return_value = _make_agent_response("single agent result")

        result = await strategy.execute(agents=[agent], task="simple task")
        assert isinstance(result, dict)
        assert result.get("final_result") == "single agent result"


class TestParallelStrategy:
    @pytest.fixture
    def strategy(self) -> ParallelStrategy:
        return ParallelStrategy(aggregation="vote")

    async def test_execute_parallel(self, strategy: ParallelStrategy) -> None:
        agent1 = AsyncMock()
        agent1.run.return_value = _make_agent_response("vote: yes")
        agent2 = AsyncMock()
        agent2.run.return_value = _make_agent_response("vote: yes")
        agent3 = AsyncMock()
        agent3.run.return_value = _make_agent_response("vote: no")

        result = await strategy.execute(
            agents=[agent1, agent2, agent3],
            task="Should we do X?",
        )
        assert isinstance(result, dict)

    async def test_execute_parallel_aggregation_types(self) -> None:
        for aggregation in ["vote", "all", "first"]:
            strategy = ParallelStrategy(aggregation=aggregation)
            agent = AsyncMock()
            agent.run.return_value = _make_agent_response("result")

            result = await strategy.execute(agents=[agent], task="task")
            assert isinstance(result, dict)


class TestDebateStrategy:
    async def test_execute_debate(self) -> None:
        strategy = DebateStrategy(rounds=1)

        agent1 = AsyncMock()
        agent1.run.return_value = _make_agent_response("Argument for position A")
        agent2 = AsyncMock()
        agent2.run.return_value = _make_agent_response("Counter-argument for position B")

        result = await strategy.execute(
            agents=[agent1, agent2],
            task="Debate: Is Python better than Java?",
        )
        assert isinstance(result, dict)

    async def test_execute_debate_with_judge(self) -> None:
        judge = AsyncMock()
        judge.run.return_value = _make_agent_response("Winner: agent1 with stronger arguments")

        strategy = DebateStrategy(rounds=1, judge_agent=judge)

        agent1 = AsyncMock()
        agent1.run.return_value = _make_agent_response("My argument")
        agent2 = AsyncMock()
        agent2.run.return_value = _make_agent_response("Counter argument")

        result = await strategy.execute(
            agents=[agent1, agent2],
            task="Topic: AI safety",
        )
        assert isinstance(result, dict)
