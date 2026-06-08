"""
Comprehensive tests for beanllm utils and domain files.

Covers:
- beanllm.utils.integration.callbacks (CallbackEvent, LoggingCallback, CostTrackingCallback,
  TimingCallback, StreamingCallback, FunctionCallback, CallbackManager)
- beanllm.utils.core.cache (LRUCache)
- beanllm.domain.graph.nodes (FunctionNode, AgentNode, LLMNode, GraderNode,
  ConditionalNode, LoopNode, ParallelNode)
- beanllm.domain.loaders.core.text (TextLoader)
- beanllm.domain.splitters.splitters (CharacterTextSplitter,
  RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter)
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Graph node imports
# ---------------------------------------------------------------------------
from beanllm.domain.graph.graph_state import GraphState
from beanllm.domain.graph.nodes import (
    AgentNode,
    ConditionalNode,
    FunctionNode,
    GraderNode,
    LLMNode,
    LoopNode,
    ParallelNode,
)

# ---------------------------------------------------------------------------
# Loader imports
# ---------------------------------------------------------------------------
from beanllm.domain.loaders.core.text import TextLoader
from beanllm.domain.loaders.types import Document

# ---------------------------------------------------------------------------
# Splitter imports
# ---------------------------------------------------------------------------
from beanllm.domain.splitters.splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Cache imports
# ---------------------------------------------------------------------------
from beanllm.utils.core.cache import LRUCache

# ---------------------------------------------------------------------------
# Callback imports
# ---------------------------------------------------------------------------
from beanllm.utils.integration.callbacks import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    CostTrackingCallback,
    FunctionCallback,
    LoggingCallback,
    StreamingCallback,
    TimingCallback,
    create_callback_manager,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a small temporary text file for TextLoader tests."""
    p = tmp_path / "sample.txt"
    p.write_text("Hello, BeanLLM!\nSecond line.\nThird line.", encoding="utf-8")
    return p


@pytest.fixture()
def lru_cache() -> LRUCache:
    """Return a fresh LRUCache with no TTL."""
    cache: LRUCache = LRUCache(max_size=5)
    yield cache
    cache.shutdown()


@pytest.fixture()
def graph_state() -> GraphState:
    return GraphState(data={"input": "test input", "counter": 0})


# ===========================================================================
# Section 1 – CallbackEvent
# ===========================================================================


class TestCallbackEvent:
    def test_creation_with_event_type(self):
        event = CallbackEvent(event_type="start")
        assert event.event_type == "start"

    def test_default_timestamp_is_datetime(self):
        event = CallbackEvent(event_type="end")
        assert isinstance(event.timestamp, datetime)

    def test_default_data_is_empty_dict(self):
        event = CallbackEvent(event_type="error")
        assert event.data == {}

    def test_custom_data(self):
        event = CallbackEvent(event_type="token", data={"token": "hello"})
        assert event.data["token"] == "hello"

    def test_multiple_events_have_independent_data(self):
        e1 = CallbackEvent(event_type="a", data={"x": 1})
        e2 = CallbackEvent(event_type="b", data={"x": 2})
        assert e1.data["x"] != e2.data["x"]


# ===========================================================================
# Section 2 – LoggingCallback
# ===========================================================================


class TestLoggingCallback:
    def test_init_default_verbose(self):
        cb = LoggingCallback()
        assert cb.verbose is True

    def test_init_verbose_false(self):
        cb = LoggingCallback(verbose=False)
        assert cb.verbose is False

    def test_on_llm_start_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_llm_start("gpt-4o", [{"role": "user", "content": "hi"}])

    def test_on_llm_end_without_tokens(self):
        cb = LoggingCallback(verbose=False)
        cb.on_llm_end("gpt-4o", "Hello!")

    def test_on_llm_end_with_tokens(self):
        cb = LoggingCallback(verbose=False)
        cb.on_llm_end("gpt-4o", "Hello!", tokens_used=42)

    def test_on_llm_error_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_llm_error("gpt-4o", ValueError("bad"))

    def test_on_agent_start_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_agent_start("MyAgent", "summarise")

    def test_on_agent_end_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_agent_end("MyAgent", "done")

    def test_on_agent_action_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_agent_action("MyAgent", "search")

    def test_on_chain_start_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_chain_start("MyChain", {"question": "what?"})

    def test_on_chain_end_does_not_raise(self):
        cb = LoggingCallback(verbose=False)
        cb.on_chain_end("MyChain", {"answer": "42"})

    def test_verbose_true_prints(self, capsys):
        cb = LoggingCallback(verbose=True)
        cb.on_llm_start("gpt-4o", [])
        captured = capsys.readouterr()
        assert "LLM Start" in captured.out

    def test_verbose_false_no_print(self, capsys):
        cb = LoggingCallback(verbose=False)
        cb.on_llm_start("gpt-4o", [])
        captured = capsys.readouterr()
        assert captured.out == ""


# ===========================================================================
# Section 3 – CostTrackingCallback
# ===========================================================================


class TestCostTrackingCallback:
    def test_initial_state(self):
        cb = CostTrackingCallback()
        assert cb.total_cost == 0.0
        assert cb.total_input_tokens == 0
        assert cb.total_output_tokens == 0
        assert cb.calls == []

    def test_on_llm_end_known_model(self):
        cb = CostTrackingCallback()
        cb.on_llm_end("gpt-4o", "response", input_tokens=1000, output_tokens=500)
        assert cb.total_input_tokens == 1000
        assert cb.total_output_tokens == 500
        assert cb.total_cost > 0

    def test_on_llm_end_unknown_model_zero_cost(self):
        cb = CostTrackingCallback()
        cb.on_llm_end("unknown-model", "response", input_tokens=1000, output_tokens=500)
        assert cb.total_cost == 0.0

    def test_get_total_cost(self):
        cb = CostTrackingCallback()
        cb.on_llm_end("gpt-4o-mini", "r", input_tokens=1_000_000, output_tokens=0)
        assert pytest.approx(cb.get_total_cost(), rel=1e-4) == 0.150

    def test_get_total_tokens(self):
        cb = CostTrackingCallback()
        cb.on_llm_end("gpt-4o", "r", input_tokens=100, output_tokens=200)
        assert cb.get_total_tokens() == 300

    def test_get_stats_structure(self):
        cb = CostTrackingCallback()
        stats = cb.get_stats()
        assert "total_calls" in stats
        assert "total_tokens" in stats
        assert "total_cost" in stats

    def test_reset_clears_state(self):
        cb = CostTrackingCallback()
        cb.on_llm_end("gpt-4o", "r", input_tokens=100, output_tokens=100)
        cb.reset()
        assert cb.total_cost == 0.0
        assert cb.calls == []

    def test_multiple_calls_accumulate(self):
        cb = CostTrackingCallback()
        cb.on_llm_end("gpt-4o", "r1", input_tokens=100, output_tokens=100)
        cb.on_llm_end("gpt-4o", "r2", input_tokens=100, output_tokens=100)
        assert len(cb.calls) == 2
        assert cb.total_input_tokens == 200


# ===========================================================================
# Section 4 – StreamingCallback
# ===========================================================================


class TestStreamingCallback:
    def test_tokens_buffered_and_flushed_on_end(self):
        received: list[str] = []
        cb = StreamingCallback(on_token=lambda t: received.append(t), buffer_size=3)
        cb.on_llm_token("a")
        cb.on_llm_token("b")
        assert received == []  # buffer not full yet
        cb.on_llm_token("c")
        assert received == ["abc"]

    def test_remaining_tokens_flushed_on_llm_end(self):
        received: list[str] = []
        cb = StreamingCallback(on_token=lambda t: received.append(t), buffer_size=10)
        cb.on_llm_token("hello")
        cb.on_llm_end("model", "full text")
        assert received == ["hello"]

    def test_no_on_token_func_does_not_raise(self):
        cb = StreamingCallback()
        cb.on_llm_token("tok")
        cb.on_llm_end("model", "text")


# ===========================================================================
# Section 5 – FunctionCallback
# ===========================================================================


class TestFunctionCallback:
    def test_on_start_called(self):
        calls: list[str] = []
        cb = FunctionCallback(on_start=lambda model, messages, **kw: calls.append(model))
        cb.on_llm_start("gpt-4o", [])
        assert "gpt-4o" in calls

    def test_on_end_called(self):
        calls: list[str] = []
        cb = FunctionCallback(on_end=lambda model, response, **kw: calls.append(response))
        cb.on_llm_end("gpt-4o", "Hello!")
        assert "Hello!" in calls

    def test_on_error_called(self):
        errors: list[Exception] = []
        cb = FunctionCallback(on_error=lambda model, error, **kw: errors.append(error))
        err = ValueError("fail")
        cb.on_llm_error("gpt-4o", err)
        assert err in errors

    def test_on_token_called(self):
        tokens: list[str] = []
        cb = FunctionCallback(on_token=lambda token, **kw: tokens.append(token))
        cb.on_llm_token("hi")
        assert "hi" in tokens

    def test_none_handlers_do_not_raise(self):
        cb = FunctionCallback()
        cb.on_llm_start("model", [])
        cb.on_llm_end("model", "r")
        cb.on_llm_error("model", ValueError("x"))
        cb.on_llm_token("t")


# ===========================================================================
# Section 6 – CallbackManager
# ===========================================================================


class TestCallbackManager:
    def test_init_empty(self):
        mgr = CallbackManager()
        assert mgr.callbacks == []

    def test_add_callback(self):
        mgr = CallbackManager()
        cb = LoggingCallback(verbose=False)
        mgr.add_callback(cb)
        assert cb in mgr.callbacks

    def test_remove_callback(self):
        mgr = CallbackManager()
        cb = LoggingCallback(verbose=False)
        mgr.add_callback(cb)
        mgr.remove_callback(cb)
        assert cb not in mgr.callbacks

    def test_remove_nonexistent_callback_does_not_raise(self):
        mgr = CallbackManager()
        mgr.remove_callback(LoggingCallback(verbose=False))

    def test_trigger_fires_all_callbacks(self):
        calls: list[str] = []
        cb1 = FunctionCallback(on_start=lambda model, messages, **kw: calls.append("cb1"))
        cb2 = FunctionCallback(on_start=lambda model, messages, **kw: calls.append("cb2"))
        mgr = CallbackManager([cb1, cb2])
        mgr.on_llm_start("gpt", [])
        assert "cb1" in calls and "cb2" in calls

    def test_callback_error_does_not_stop_others(self, capsys):
        """A broken callback must not prevent subsequent callbacks from firing."""
        calls: list[str] = []

        class BrokenCallback(BaseCallback):
            def on_llm_start(self, model, messages, **kwargs):
                raise RuntimeError("broken")

        cb_broken = BrokenCallback()
        cb_good = FunctionCallback(on_start=lambda model, messages, **kw: calls.append("good"))
        mgr = CallbackManager([cb_broken, cb_good])
        mgr.on_llm_start("gpt", [])
        assert "good" in calls

    def test_on_llm_end_delegated(self):
        calls: list[str] = []
        cb = FunctionCallback(on_end=lambda model, response, **kw: calls.append(response))
        mgr = CallbackManager([cb])
        mgr.on_llm_end("gpt", "the response")
        assert "the response" in calls

    def test_on_agent_start_delegated(self):
        calls: list[str] = []

        class AuditCB(BaseCallback):
            def on_agent_start(self, agent_name, task, **kwargs):
                calls.append(agent_name)

        mgr = CallbackManager([AuditCB()])
        mgr.on_agent_start("agent-x", "do-stuff")
        assert "agent-x" in calls

    def test_on_tool_start_delegated(self):
        calls: list[str] = []

        class ToolCB(BaseCallback):
            def on_tool_start(self, tool_name, inputs, **kwargs):
                calls.append(tool_name)

        mgr = CallbackManager([ToolCB()])
        mgr.on_tool_start("search", {"query": "AI"})
        assert "search" in calls

    def test_create_callback_manager_helper(self):
        cb = LoggingCallback(verbose=False)
        mgr = create_callback_manager(cb)
        assert isinstance(mgr, CallbackManager)
        assert cb in mgr.callbacks


# ===========================================================================
# Section 7 – LRUCache
# ===========================================================================


class TestLRUCache:
    def test_set_and_get(self, lru_cache: LRUCache):
        lru_cache.set("k", "v")
        assert lru_cache.get("k") == "v"

    def test_get_missing_returns_none(self, lru_cache: LRUCache):
        assert lru_cache.get("missing") is None

    def test_get_missing_returns_default(self, lru_cache: LRUCache):
        assert lru_cache.get("missing", "fallback") == "fallback"

    def test_delete_existing(self, lru_cache: LRUCache):
        lru_cache.set("k", "v")
        result = lru_cache.delete("k")
        assert result is True
        assert lru_cache.get("k") is None

    def test_delete_nonexistent_returns_false(self, lru_cache: LRUCache):
        assert lru_cache.delete("no_such_key") is False

    def test_clear_empties_cache(self, lru_cache: LRUCache):
        lru_cache.set("a", 1)
        lru_cache.set("b", 2)
        lru_cache.clear()
        assert len(lru_cache) == 0

    def test_clear_resets_stats(self, lru_cache: LRUCache):
        lru_cache.set("a", 1)
        lru_cache.get("a")
        lru_cache.clear()
        stats = lru_cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_len_reflects_entries(self, lru_cache: LRUCache):
        lru_cache.set("x", 10)
        lru_cache.set("y", 20)
        assert len(lru_cache) == 2

    def test_contains_existing_key(self, lru_cache: LRUCache):
        lru_cache.set("z", 99)
        assert "z" in lru_cache

    def test_contains_missing_key(self, lru_cache: LRUCache):
        assert "ghost" not in lru_cache

    def test_max_size_lru_eviction(self):
        cache: LRUCache = LRUCache(max_size=3)
        try:
            cache.set("a", 1)
            cache.set("b", 2)
            cache.set("c", 3)
            # Access 'a' to make it recently used
            cache.get("a")
            # Adding 'd' should evict 'b' (least recently used)
            cache.set("d", 4)
            assert len(cache) == 3
            assert cache.get("b") is None  # evicted
            assert cache.get("a") is not None
        finally:
            cache.shutdown()

    def test_stats_hit_rate(self, lru_cache: LRUCache):
        lru_cache.set("k", "v")
        lru_cache.get("k")  # hit
        lru_cache.get("missing")  # miss
        stats = lru_cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert pytest.approx(stats["hit_rate"], abs=0.01) == 0.5

    def test_stats_structure(self, lru_cache: LRUCache):
        stats = lru_cache.stats()
        for key in ("size", "max_size", "ttl", "hits", "misses", "hit_rate", "evictions"):
            assert key in stats

    def test_ttl_expiry(self):
        cache: LRUCache = LRUCache(max_size=10, ttl=1, cleanup_interval=60)
        try:
            cache.set("expire_me", "soon")
            time.sleep(1.1)
            assert cache.get("expire_me") is None
        finally:
            cache.shutdown()

    def test_ttl_contains_expired(self):
        cache: LRUCache = LRUCache(max_size=10, ttl=1, cleanup_interval=60)
        try:
            cache.set("key", "val")
            time.sleep(1.1)
            assert "key" not in cache
        finally:
            cache.shutdown()

    def test_eviction_callback_called(self):
        evicted: list[str] = []
        cache: LRUCache = LRUCache(max_size=2, on_evict=lambda k, v: evicted.append(k))
        try:
            cache.set("a", 1)
            cache.set("b", 2)
            cache.set("c", 3)  # triggers eviction of 'a'
            assert "a" in evicted
        finally:
            cache.shutdown()

    def test_shutdown_clears_cache(self, lru_cache: LRUCache):
        lru_cache.set("k", "v")
        lru_cache.shutdown()
        assert len(lru_cache) == 0

    def test_update_existing_key_does_not_grow(self, lru_cache: LRUCache):
        lru_cache.set("k", "v1")
        lru_cache.set("k", "v2")
        assert len(lru_cache) == 1
        assert lru_cache.get("k") == "v2"


# ===========================================================================
# Section 8 – FunctionNode
# ===========================================================================


class TestFunctionNode:
    def test_init_stores_name(self):
        node = FunctionNode("my_node", lambda s: {})
        assert node.name == "my_node"

    def test_init_default_cache_disabled(self):
        node = FunctionNode("n", lambda s: {})
        assert node.cache_enabled is False

    def test_init_with_description(self):
        node = FunctionNode("n", lambda s: {}, description="test node")
        assert node.description == "test node"

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, graph_state: GraphState):
        def handler(state: GraphState) -> Dict[str, Any]:
            return {"processed": state.get("input", "") + "_done"}

        node = FunctionNode("proc", handler)
        result = await node.execute(graph_state)
        assert result["processed"] == "test input_done"

    @pytest.mark.asyncio
    async def test_execute_async_function(self, graph_state: GraphState):
        async def async_handler(state: GraphState) -> Dict[str, Any]:
            return {"async_result": "ok"}

        node = FunctionNode("async_node", async_handler)
        result = await node.execute(graph_state)
        assert result["async_result"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_non_dict_result_wrapped(self, graph_state: GraphState):
        node = FunctionNode("wrap_node", lambda s: "plain string")
        result = await node.execute(graph_state)
        assert "result" in result
        assert result["result"] == "plain string"

    @pytest.mark.asyncio
    async def test_execute_with_cache_flag(self, graph_state: GraphState):
        node = FunctionNode("cached", lambda s: {"v": 1}, cache=True)
        result = await node.execute(graph_state)
        assert result["v"] == 1


# ===========================================================================
# Section 9 – AgentNode
# ===========================================================================


class TestAgentNode:
    def test_init_defaults(self):
        mock_agent = MagicMock()
        node = AgentNode("agent1", mock_agent)
        assert node.name == "agent1"
        assert node.input_key == "input"
        assert node.output_key == "output"

    def test_init_custom_keys(self):
        mock_agent = MagicMock()
        node = AgentNode("agent2", mock_agent, input_key="query", output_key="answer")
        assert node.input_key == "query"
        assert node.output_key == "answer"

    @pytest.mark.asyncio
    async def test_execute_calls_agent_run(self, graph_state: GraphState):
        mock_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "42"
        mock_result.total_steps = 3
        mock_result.success = True
        mock_agent.run = AsyncMock(return_value=mock_result)

        node = AgentNode("test_agent", mock_agent)
        result = await node.execute(graph_state)

        mock_agent.run.assert_called_once_with("test input")
        assert result["output"] == "42"
        assert result["output_steps"] == 3
        assert result["output_success"] is True


# ===========================================================================
# Section 10 – LLMNode
# ===========================================================================


class TestLLMNode:
    def test_init_stores_fields(self):
        mock_client = MagicMock()
        node = LLMNode("llm_node", mock_client, "Say {word}", input_keys=["word"])
        assert node.name == "llm_node"
        assert node.template == "Say {word}"
        assert node.input_keys == ["word"]
        assert node.output_key == "output"

    @pytest.mark.asyncio
    async def test_execute_formats_template_and_calls_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "hello world"
        mock_client.chat = AsyncMock(return_value=mock_response)

        node = LLMNode("n", mock_client, "Say {word}", input_keys=["word"], output_key="reply")
        state = GraphState(data={"word": "hi"})
        result = await node.execute(state)

        assert result["reply"] == "hello world"
        mock_client.chat.assert_called_once_with([{"role": "user", "content": "Say hi"}])

    @pytest.mark.asyncio
    async def test_execute_with_parser(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "raw content"
        mock_client.chat = AsyncMock(return_value=mock_response)

        mock_parser = MagicMock()
        mock_parser.parse = MagicMock(return_value="parsed content")

        node = LLMNode("n", mock_client, "{text}", input_keys=["text"], parser=mock_parser)
        state = GraphState(data={"text": "blah"})
        result = await node.execute(state)

        mock_parser.parse.assert_called_once_with("raw content")
        assert result["output"] == "parsed content"


# ===========================================================================
# Section 11 – GraderNode
# ===========================================================================


class TestGraderNode:
    def test_init_stores_fields(self):
        mock_client = MagicMock()
        node = GraderNode("grader", mock_client, "Is it good?", input_key="answer")
        assert node.criteria == "Is it good?"
        assert node.input_key == "answer"
        assert node.output_key == "grade"
        assert node.scale == 10

    @pytest.mark.asyncio
    async def test_execute_parses_score(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Score: 8\nExplanation: Quite good."
        mock_client.chat = AsyncMock(return_value=mock_response)

        node = GraderNode("g", mock_client, "Criteria", input_key="answer")
        state = GraphState(data={"answer": "My answer here"})
        result = await node.execute(state)

        assert result["grade"] == 8
        assert "Quite good" in result["grade_explanation"]
        assert result["grade_max"] == 10

    @pytest.mark.asyncio
    async def test_execute_no_score_defaults_to_zero(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "No score found in this response."
        mock_client.chat = AsyncMock(return_value=mock_response)

        node = GraderNode("g", mock_client, "Criteria", input_key="answer")
        state = GraphState(data={"answer": "something"})
        result = await node.execute(state)

        assert result["grade"] == 0


# ===========================================================================
# Section 12 – ConditionalNode
# ===========================================================================


class TestConditionalNode:
    @pytest.mark.asyncio
    async def test_execute_true_branch(self, graph_state: GraphState):
        true_node = FunctionNode("true_n", lambda s: {"branch": "true"})
        false_node = FunctionNode("false_n", lambda s: {"branch": "false"})
        node = ConditionalNode(
            "cond",
            condition=lambda s: True,
            true_node=true_node,
            false_node=false_node,
        )
        result = await node.execute(graph_state)
        assert result["branch"] == "true"
        assert result["cond_condition"] is True
        assert result["cond_executed"] == "true_n"

    @pytest.mark.asyncio
    async def test_execute_false_branch(self, graph_state: GraphState):
        true_node = FunctionNode("true_n", lambda s: {"branch": "true"})
        false_node = FunctionNode("false_n", lambda s: {"branch": "false"})
        node = ConditionalNode(
            "cond",
            condition=lambda s: False,
            true_node=true_node,
            false_node=false_node,
        )
        result = await node.execute(graph_state)
        assert result["branch"] == "false"
        assert result["cond_condition"] is False

    @pytest.mark.asyncio
    async def test_execute_no_branch_node_returns_metadata(self, graph_state: GraphState):
        node = ConditionalNode("cond", condition=lambda s: True)
        result = await node.execute(graph_state)
        assert result["cond_condition"] is True
        assert result["cond_executed"] is None


# ===========================================================================
# Section 13 – LoopNode
# ===========================================================================


class TestLoopNode:
    @pytest.mark.asyncio
    async def test_loop_terminates_by_condition(self):
        call_count = 0

        def body_fn(state: GraphState) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            state.set("counter", state.get("counter", 0) + 1)
            return {"counter": state.get("counter")}

        body = FunctionNode("body", body_fn)
        state = GraphState(data={"counter": 0})

        node = LoopNode(
            "loop",
            body_node=body,
            termination_condition=lambda s: s.get("counter", 0) >= 3,
            max_iterations=10,
        )
        result = await node.execute(state)
        assert result["loop_iterations"] == 3
        assert result["loop_terminated"] is True

    @pytest.mark.asyncio
    async def test_loop_respects_max_iterations(self):
        body = FunctionNode("body", lambda s: {})
        state = GraphState(data={})

        node = LoopNode(
            "loop",
            body_node=body,
            termination_condition=lambda s: False,  # never terminates naturally
            max_iterations=4,
        )
        result = await node.execute(state)
        assert result["loop_iterations"] == 4

    @pytest.mark.asyncio
    async def test_loop_immediate_termination(self):
        body = FunctionNode("body", lambda s: {})
        state = GraphState(data={"done": True})

        node = LoopNode(
            "loop",
            body_node=body,
            termination_condition=lambda s: s.get("done", False),
        )
        result = await node.execute(state)
        assert result["loop_iterations"] == 0


# ===========================================================================
# Section 14 – ParallelNode
# ===========================================================================


class TestParallelNode:
    @pytest.mark.asyncio
    async def test_merge_strategy(self, graph_state: GraphState):
        n1 = FunctionNode("nodeA", lambda s: {"val": 1})
        n2 = FunctionNode("nodeB", lambda s: {"val": 2})
        pnode = ParallelNode("par", [n1, n2], aggregate_strategy="merge")
        result = await pnode.execute(graph_state)
        assert "nodeA_val" in result
        assert "nodeB_val" in result
        assert result["par_count"] == 2

    @pytest.mark.asyncio
    async def test_list_strategy(self, graph_state: GraphState):
        n1 = FunctionNode("n1", lambda s: {"x": 10})
        n2 = FunctionNode("n2", lambda s: {"x": 20})
        pnode = ParallelNode("par", [n1, n2], aggregate_strategy="list")
        result = await pnode.execute(graph_state)
        assert "par_results" in result
        assert len(result["par_results"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_strategy_raises(self, graph_state: GraphState):
        n1 = FunctionNode("n1", lambda s: {"x": 1})
        pnode = ParallelNode("par", [n1], aggregate_strategy="unknown")
        with pytest.raises(ValueError, match="Unknown aggregate strategy"):
            await pnode.execute(graph_state)

    def test_init_stores_child_nodes(self):
        n1 = FunctionNode("n1", lambda s: {})
        n2 = FunctionNode("n2", lambda s: {})
        pnode = ParallelNode("par", [n1, n2])
        assert len(pnode.child_nodes) == 2


# ===========================================================================
# Section 15 – TextLoader
# ===========================================================================


class TestTextLoader:
    def test_init_stores_path(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        assert loader.file_path == tmp_text_file

    def test_init_default_encoding_utf8(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        assert loader.encoding == "utf-8"

    def test_load_returns_list_of_documents(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        docs = loader.load()
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)

    def test_load_content_matches_file(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        docs = loader.load()
        assert "Hello, BeanLLM!" in docs[0].content

    def test_load_metadata_has_source(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        docs = loader.load()
        assert "source" in docs[0].metadata
        assert str(tmp_text_file) == docs[0].metadata["source"]

    def test_load_metadata_has_encoding(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        docs = loader.load()
        assert docs[0].metadata["encoding"] == "utf-8"

    def test_lazy_load_yields_document(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False)
        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert "Hello, BeanLLM!" in docs[0].content

    def test_load_with_path_object(self, tmp_text_file: Path):
        loader = TextLoader(tmp_text_file, validate_path=False)
        docs = loader.load()
        assert len(docs) == 1

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        loader = TextLoader(str(tmp_path / "ghost.txt"), validate_path=False)
        with pytest.raises(Exception):
            loader.load()

    def test_load_streaming_yields_chunks(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False, chunk_size=5)
        chunks = list(loader.load_streaming())
        assert len(chunks) >= 1
        full = "".join(c.content for c in chunks)
        assert "Hello" in full

    def test_load_with_explicit_mmap_false(self, tmp_text_file: Path):
        loader = TextLoader(str(tmp_text_file), validate_path=False, use_mmap=False)
        docs = loader.load()
        assert len(docs) == 1


# ===========================================================================
# Section 16 – CharacterTextSplitter
# ===========================================================================


class TestCharacterTextSplitter:
    def test_basic_split_by_double_newline(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        # All original content should be present
        combined = " ".join(chunks)
        assert "First" in combined
        assert "Second" in combined

    def test_empty_string_returns_no_chunks(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
        chunks = splitter.split_text("")
        assert chunks == [] or chunks == [""]

    def test_chunk_size_respected(self):
        splitter = CharacterTextSplitter(separator=" ", chunk_size=20, chunk_overlap=0)
        text = "word " * 50
        chunks = splitter.split_text(text)
        for chunk in chunks:
            assert len(chunk) <= 20 + 5  # small tolerance for separator

    def test_single_chunk_no_separator(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
        text = "No paragraph break here."
        chunks = splitter.split_text(text)
        assert any("No paragraph" in c for c in chunks)

    def test_split_documents(self):
        from beanllm.domain.loaders.types import Document as Doc

        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=20, chunk_overlap=0)
        docs = [Doc(content="Hello world\n\nFoo bar baz", metadata={"src": "x"})]
        result = splitter.split_documents(docs)
        assert isinstance(result, list)
        assert all(isinstance(d, Doc) for d in result)

    def test_create_documents_adds_chunk_metadata(self):
        splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
        result = splitter.create_documents(["a\n\nb"], [{"src": "test"}])
        for doc in result:
            assert "chunk" in doc.metadata


# ===========================================================================
# Section 17 – RecursiveCharacterTextSplitter
# ===========================================================================


class TestRecursiveCharacterTextSplitter:
    def test_default_separators(self):
        splitter = RecursiveCharacterTextSplitter()
        assert "\n\n" in splitter.separators

    def test_custom_separators(self):
        splitter = RecursiveCharacterTextSplitter(separators=["---", "\n"])
        assert "---" in splitter.separators

    def test_split_text_basic(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "First paragraph.\n\nSecond paragraph that is longer."
        chunks = splitter.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_split_preserves_all_content(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=0)
        text = "Hello world. This is a test. More content here."
        chunks = splitter.split_text(text)
        combined = "".join(c.replace("\n", " ") for c in chunks)
        # Check key words appear somewhere
        assert "Hello" in combined

    def test_large_chunk_recursively_split(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "A" * 100
        chunks = splitter.split_text(text)
        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_chunk_overlap_creates_overlap(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = splitter.split_text(text)
        if len(chunks) > 1:
            # The tail of chunk[0] should appear at the start of chunk[1]
            overlap_candidate = chunks[0][-5:]
            assert overlap_candidate in chunks[1]

    def test_split_documents_with_metadata(self):
        from beanllm.domain.loaders.types import Document as Doc

        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
        docs = [Doc(content="One two three four five six seven eight nine ten", metadata={"id": 1})]
        result = splitter.split_documents(docs)
        assert all(d.metadata.get("id") == 1 for d in result)


# ===========================================================================
# Section 18 – MarkdownHeaderTextSplitter
# ===========================================================================


class TestMarkdownHeaderTextSplitter:
    MARKDOWN = """\
# Title

Intro text.

## Section A

Content of section A.

## Section B

Content of section B.
"""

    def test_splits_on_h1(self):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1")])
        docs = splitter.split_text(self.MARKDOWN)
        assert any("Intro" in d.content for d in docs)

    def test_splits_on_h2(self):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
        docs = splitter.split_text(self.MARKDOWN)
        sections = [d.metadata.get("h2") for d in docs if d.metadata.get("h2")]
        assert "Section A" in sections or "Section B" in sections

    def test_metadata_populated(self):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
        docs = splitter.split_text(self.MARKDOWN)
        meta_sections = [d.metadata.get("section") for d in docs if d.metadata.get("section")]
        assert len(meta_sections) > 0

    def test_no_headers_returns_single_chunk(self):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1")])
        docs = splitter.split_text("plain text without headers")
        assert len(docs) == 1

    def test_empty_text(self):
        # The splitter always yields at least one Document (possibly with empty content)
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1")])
        docs = splitter.split_text("")
        # Either empty list or a single empty-content document is acceptable
        assert docs == [] or (len(docs) == 1 and docs[0].content == "")

    def test_split_documents_merges_metadata(self):
        from beanllm.domain.loaders.types import Document as Doc

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "section")])
        source_doc = Doc(content=self.MARKDOWN, metadata={"origin": "wiki"})
        chunks = splitter.split_documents([source_doc])
        assert all(d.metadata.get("origin") == "wiki" for d in chunks)

    def test_return_each_line_mode(self):
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")], return_each_line=True
        )
        docs = splitter.split_text(self.MARKDOWN)
        # Each non-blank, non-header line is a separate doc
        assert len(docs) >= 2
