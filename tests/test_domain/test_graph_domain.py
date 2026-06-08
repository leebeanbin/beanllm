"""Tests for domain/graph: GraphState, BaseNode, and node implementations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from beanllm.domain.graph.base_node import BaseNode
from beanllm.domain.graph.graph_state import GraphState
from beanllm.domain.graph.nodes import (
    AgentNode,
    ConditionalNode,
    FunctionNode,
    LLMNode,
    LoopNode,
    ParallelNode,
)

# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------


class TestGraphState:
    def test_default_empty(self):
        s = GraphState()
        assert s.data == {}
        assert s.metadata == {}

    def test_get_existing(self):
        s = GraphState(data={"x": 42})
        assert s.get("x") == 42

    def test_get_missing_returns_default(self):
        s = GraphState()
        assert s.get("missing") is None
        assert s.get("missing", "def") == "def"

    def test_set(self):
        s = GraphState()
        s.set("k", "v")
        assert s.data["k"] == "v"

    def test_update(self):
        s = GraphState()
        s.update({"a": 1, "b": 2})
        assert s.data["a"] == 1
        assert s.data["b"] == 2

    def test_getitem(self):
        s = GraphState(data={"key": "val"})
        assert s["key"] == "val"

    def test_setitem(self):
        s = GraphState()
        s["foo"] = "bar"
        assert s.data["foo"] == "bar"

    def test_contains(self):
        s = GraphState(data={"p": 1})
        assert "p" in s
        assert "q" not in s

    def test_copy_is_shallow(self):
        s = GraphState(data={"a": [1, 2]})
        c = s.copy()
        assert c.data["a"] is s.data["a"]  # same list object (shallow)
        assert c is not s

    def test_copy_isolation(self):
        s = GraphState(data={"x": 1})
        c = s.copy()
        c.set("x", 99)
        assert s.get("x") == 1  # original unchanged

    def test_deepcopy(self):
        s = GraphState(data={"lst": [1, 2, 3]})
        d = s.deepcopy()
        d.data["lst"].append(4)
        assert len(s.data["lst"]) == 3  # original unchanged

    def test_metadata_preserved_in_copy(self):
        s = GraphState(data={}, metadata={"run_id": "abc"})
        c = s.copy()
        assert c.metadata["run_id"] == "abc"


# ---------------------------------------------------------------------------
# BaseNode (abstract) - via concrete subclass
# ---------------------------------------------------------------------------


class _SimpleNode(BaseNode):
    async def execute(self, state: GraphState):
        return {"done": True}


class TestBaseNode:
    def test_init_defaults(self):
        n = _SimpleNode("my_node")
        assert n.name == "my_node"
        assert n.cache_enabled is False
        assert n.description == ""

    def test_init_custom(self):
        n = _SimpleNode("n", cache=True, description="desc")
        assert n.cache_enabled is True
        assert n.description == "desc"

    async def test_execute(self):
        n = _SimpleNode("n")
        result = await n.execute(GraphState())
        assert result == {"done": True}


# ---------------------------------------------------------------------------
# FunctionNode
# ---------------------------------------------------------------------------


class TestFunctionNode:
    async def test_sync_function(self):
        def my_fn(state):
            return {"value": state.get("x", 0) + 1}

        node = FunctionNode("add", my_fn)
        state = GraphState(data={"x": 5})
        result = await node.execute(state)
        assert result["value"] == 6

    async def test_async_function(self):
        async def my_async_fn(state):
            return {"async_result": "ok"}

        node = FunctionNode("async_node", my_async_fn)
        result = await node.execute(GraphState())
        assert result["async_result"] == "ok"

    async def test_non_dict_result_wrapped(self):
        def returns_string(state):
            return "hello"

        node = FunctionNode("str_fn", returns_string)
        result = await node.execute(GraphState())
        assert result == {"result": "hello"}

    def test_init_with_cache(self):
        node = FunctionNode("n", lambda s: {}, cache=True, description="d")
        assert node.cache_enabled is True
        assert node.description == "d"


# ---------------------------------------------------------------------------
# AgentNode
# ---------------------------------------------------------------------------


class TestAgentNode:
    async def test_execute_calls_agent_run(self):
        mock_agent = MagicMock()
        mock_result = MagicMock(answer="answer_text", total_steps=3, success=True)
        mock_agent.run = AsyncMock(return_value=mock_result)

        node = AgentNode("agent_node", mock_agent, input_key="query", output_key="answer")
        state = GraphState(data={"query": "What is AI?"})
        result = await node.execute(state)

        mock_agent.run.assert_called_once_with("What is AI?")
        assert result["answer"] == "answer_text"
        assert result["answer_steps"] == 3
        assert result["answer_success"] is True

    async def test_default_input_output_keys(self):
        mock_agent = MagicMock()
        mock_result = MagicMock(answer="out", total_steps=1, success=True)
        mock_agent.run = AsyncMock(return_value=mock_result)

        node = AgentNode("n", mock_agent)
        state = GraphState(data={"input": "hello"})
        result = await node.execute(state)
        assert "output" in result


# ---------------------------------------------------------------------------
# LLMNode
# ---------------------------------------------------------------------------


class TestLLMNode:
    def _make_client(self, content="response"):
        client = MagicMock()
        client.chat = AsyncMock(return_value=MagicMock(content=content))
        return client

    async def test_execute_renders_template(self):
        client = self._make_client("summary text")
        node = LLMNode("llm", client, template="Summarize: {text}", input_keys=["text"])
        state = GraphState(data={"text": "long document"})
        result = await node.execute(state)
        assert result["output"] == "summary text"
        client.chat.assert_called_once()
        call_args = client.chat.call_args[0][0]
        assert "long document" in call_args[0]["content"]

    async def test_execute_with_parser(self):
        client = self._make_client("raw")
        mock_parser = MagicMock()
        mock_parser.parse.return_value = {"parsed": True}
        node = LLMNode("llm", client, template="{x}", input_keys=["x"], parser=mock_parser)
        state = GraphState(data={"x": "val"})
        result = await node.execute(state)
        assert result["output"] == {"parsed": True}
        mock_parser.parse.assert_called_once_with("raw")

    async def test_custom_output_key(self):
        client = self._make_client("result")
        node = LLMNode("n", client, template="{q}", input_keys=["q"], output_key="answer")
        state = GraphState(data={"q": "hi"})
        result = await node.execute(state)
        assert "answer" in result

    async def test_missing_input_key_uses_empty_string(self):
        client = self._make_client("ok")
        node = LLMNode("n", client, template="[{missing}]", input_keys=["missing"])
        result = await node.execute(GraphState())
        # Should not raise - missing key yields empty string
        assert "output" in result


# ---------------------------------------------------------------------------
# ConditionalNode
# ---------------------------------------------------------------------------


class TestConditionalNode:
    async def test_true_branch_executed(self):
        true_node = FunctionNode("true_fn", lambda s: {"branch": "true"})
        false_node = FunctionNode("false_fn", lambda s: {"branch": "false"})

        cond = ConditionalNode(
            "cond",
            condition=lambda s: s.get("flag") is True,
            true_node=true_node,
            false_node=false_node,
        )
        state = GraphState(data={"flag": True})
        result = await cond.execute(state)
        assert result["branch"] == "true"
        assert result["cond_condition"] is True
        assert result["cond_executed"] == "true_fn"

    async def test_false_branch_executed(self):
        true_node = FunctionNode("t", lambda s: {"branch": "true"})
        false_node = FunctionNode("f", lambda s: {"branch": "false"})

        cond = ConditionalNode("cond", lambda s: False, true_node=true_node, false_node=false_node)
        result = await cond.execute(GraphState())
        assert result["branch"] == "false"
        assert result["cond_condition"] is False

    async def test_no_branch_nodes(self):
        cond = ConditionalNode("cond", lambda s: True)
        result = await cond.execute(GraphState())
        assert result["cond_condition"] is True
        assert result["cond_executed"] is None


# ---------------------------------------------------------------------------
# LoopNode
# ---------------------------------------------------------------------------


class TestLoopNode:
    async def test_loop_runs_until_condition(self):
        counter = {"n": 0}

        def body_fn(state):
            counter["n"] += 1
            return {"count": counter["n"]}

        body = FunctionNode("body", body_fn)
        loop = LoopNode("loop", body, termination_condition=lambda s: s.get("count", 0) >= 3)

        state = GraphState()
        result = await loop.execute(state)
        assert result["loop_iterations"] == 3
        assert result["loop_terminated"] is True

    async def test_max_iterations_respected(self):
        body = FunctionNode("body", lambda s: {})
        loop = LoopNode("loop", body, termination_condition=lambda s: False, max_iterations=2)

        result = await loop.execute(GraphState())
        assert result["loop_iterations"] == 2

    async def test_loop_results_collected(self):
        body = FunctionNode("b", lambda s: {"x": 1})
        loop = LoopNode("loop", body, termination_condition=lambda s: False, max_iterations=3)

        result = await loop.execute(GraphState())
        assert len(result["loop_results"]) == 3


# ---------------------------------------------------------------------------
# ParallelNode
# ---------------------------------------------------------------------------


class TestParallelNode:
    async def test_merge_strategy(self):
        n1 = FunctionNode("n1", lambda s: {"a": 1})
        n2 = FunctionNode("n2", lambda s: {"b": 2})
        par = ParallelNode("par", [n1, n2], aggregate_strategy="merge")

        result = await par.execute(GraphState())
        # merge strategy prefixes keys with node name
        assert result["n1_a"] == 1
        assert result["n2_b"] == 2
        assert result["par_count"] == 2

    async def test_list_strategy(self):
        n1 = FunctionNode("n1", lambda s: {"x": 1})
        n2 = FunctionNode("n2", lambda s: {"x": 2})
        par = ParallelNode("par", [n1, n2], aggregate_strategy="list")

        result = await par.execute(GraphState())
        assert "par_results" in result
        assert result["par_count"] == 2

    async def test_empty_parallel_node(self):
        par = ParallelNode("par", [], aggregate_strategy="list")
        result = await par.execute(GraphState())
        assert result["par_count"] == 0

    async def test_init_defaults(self):
        n = ParallelNode("p", [])
        assert n.aggregate_strategy == "merge"
