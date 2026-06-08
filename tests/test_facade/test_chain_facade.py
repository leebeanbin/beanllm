"""Tests for facade/core/chain_facade.py — Chain, PromptChain, SequentialChain, etc."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from beanllm.dto.response.core.chain_response import ChainResponse
from beanllm.facade.core.chain_facade import (
    Chain,
    ChainBuilder,
    ChainResult,
    ParallelChain,
    PromptChain,
    SequentialChain,
    create_chain,
)
from beanllm.facade.core.client_facade import Client


def _make_response(output="answer", success=True, error=None, steps=None):
    return ChainResponse(output=output, steps=steps or [], success=success, error=error)


def _mock_container(handler):
    container = Mock()
    factory = Mock()
    factory.create_chain_handler.return_value = handler
    container.handler_factory = factory
    return container


# ---------------------------------------------------------------------------
# ChainResult
# ---------------------------------------------------------------------------


class TestChainResult:
    def test_basic_creation(self):
        r = ChainResult(output="hello")
        assert r.output == "hello"
        assert r.success is True
        assert r.error is None
        assert r.steps == []
        assert r.metadata == {}

    def test_with_error(self):
        r = ChainResult(output="", success=False, error="something went wrong")
        assert r.success is False
        assert r.error == "something went wrong"

    def test_with_steps_and_metadata(self):
        r = ChainResult(output="x", steps=[{"step": 1}], metadata={"key": "val"})
        assert len(r.steps) == 1
        assert r.metadata["key"] == "val"

    def test_frozen(self):
        r = ChainResult(output="x")
        with pytest.raises((AttributeError, TypeError)):
            r.output = "y"  # type: ignore


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class TestChain:
    def _make_chain(self, response=None):
        handler = MagicMock()
        resp = response or _make_response("Chain output")
        handler.handle_run = AsyncMock(return_value=resp)
        container = _mock_container(handler)

        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            client = Mock(spec=Client)
            client.model = "gpt-4o-mini"
            chain = Chain(client)
        chain._chain_handler = handler
        return chain

    async def test_run_returns_chain_result(self):
        chain = self._make_chain()
        result = await chain.run("hello")
        assert isinstance(result, ChainResult)
        assert result.output == "Chain output"

    async def test_run_calls_handle_run(self):
        chain = self._make_chain()
        await chain.run("hello")
        chain._chain_handler.handle_run.assert_awaited_once()

    async def test_run_passes_model(self):
        chain = self._make_chain()
        await chain.run("test input")
        call_kwargs = chain._chain_handler.handle_run.await_args.kwargs
        assert call_kwargs.get("model") == "gpt-4o-mini"

    async def test_run_with_verbose_false(self):
        handler = MagicMock()
        handler.handle_run = AsyncMock(return_value=_make_response("ok"))
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            client = Mock(spec=Client)
            client.model = "gpt-4o-mini"
            chain = Chain(client, verbose=False)
        chain._chain_handler = handler
        result = await chain.run("q")
        assert result.success is True

    async def test_run_passes_success_false(self):
        chain = self._make_chain(_make_response("", success=False, error="fail"))
        result = await chain.run("x")
        assert result.success is False
        assert result.error == "fail"


# ---------------------------------------------------------------------------
# PromptChain
# ---------------------------------------------------------------------------


class TestPromptChain:
    def _make_prompt_chain(self, template="Answer {q}", response=None):
        handler = MagicMock()
        resp = response or _make_response("Prompt answer")
        handler.handle_run = AsyncMock(return_value=resp)
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            client = Mock(spec=Client)
            client.model = "gpt-4o-mini"
            pc = PromptChain(client, template)
        pc._chain_handler = handler
        return pc

    async def test_run_returns_chain_result(self):
        pc = self._make_prompt_chain()
        result = await pc.run(q="What is AI?")
        assert isinstance(result, ChainResult)
        assert result.output == "Prompt answer"

    async def test_run_passes_template_vars(self):
        pc = self._make_prompt_chain()
        await pc.run(q="test", context="some context")
        call_kwargs = pc._chain_handler.handle_run.await_args.kwargs
        assert call_kwargs.get("chain_type") == "prompt"
        assert "q" in call_kwargs.get("template_vars", {})

    async def test_run_passes_template(self):
        pc = self._make_prompt_chain("My template {x}")
        await pc.run(x="value")
        call_kwargs = pc._chain_handler.handle_run.await_args.kwargs
        assert call_kwargs.get("template") == "My template {x}"

    async def test_run_without_memory(self):
        pc = self._make_prompt_chain()
        assert pc.memory is None
        result = await pc.run()
        assert result.output == "Prompt answer"


# ---------------------------------------------------------------------------
# SequentialChain
# ---------------------------------------------------------------------------


class TestSequentialChain:
    def _make_sequential(self, chain_outputs):
        results = [_make_response(o) for o in chain_outputs]
        chains = []
        for resp in results:
            c = MagicMock()
            c.run = AsyncMock(return_value=ChainResult(output=resp.output, steps=resp.steps))
            chains.append(c)

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            seq = SequentialChain(chains)
        return seq, chains

    async def test_single_chain(self):
        seq, _ = self._make_sequential(["result1"])
        result = await seq.run(input="start")
        assert result.output == "result1"
        assert result.success is True

    async def test_two_chains_passes_output(self):
        seq, chains = self._make_sequential(["first", "second"])
        result = await seq.run(input="start")
        assert result.output == "second"
        chains[1].run.assert_awaited_once()

    async def test_stops_on_failure(self):
        fail_result = ChainResult(output="", success=False, error="step1 failed")
        chains = []
        c0 = MagicMock()
        c0.run = AsyncMock(return_value=fail_result)
        c1 = MagicMock()
        c1.run = AsyncMock(return_value=ChainResult(output="should not run"))
        chains = [c0, c1]

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            seq = SequentialChain(chains)
        result = await seq.run(input="x")
        assert result.success is False
        c1.run.assert_not_awaited()

    async def test_exception_returns_error_result(self):
        chains = [MagicMock()]
        chains[0].run = AsyncMock(side_effect=RuntimeError("boom"))

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            seq = SequentialChain(chains)
        result = await seq.run(input="x")
        assert result.success is False
        assert "boom" in result.error

    async def test_prompt_chain_gets_input_kwarg(self):
        """PromptChain in position 2+ receives input= kwarg."""
        c0 = MagicMock(spec=[])
        c0.run = AsyncMock(return_value=ChainResult(output="first"))

        c1 = MagicMock(spec=PromptChain)
        c1.run = AsyncMock(return_value=ChainResult(output="second"))

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            seq = SequentialChain([c0, c1])
        result = await seq.run(input="start")
        c1.run.assert_awaited_once_with(input="first")
        assert result.output == "second"


# ---------------------------------------------------------------------------
# ParallelChain
# ---------------------------------------------------------------------------


class TestParallelChain:
    def _make_parallel(self, chain_outputs):
        chains = []
        for output in chain_outputs:
            c = MagicMock()
            c.run = AsyncMock(return_value=ChainResult(output=output))
            chains.append(c)

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            par = ParallelChain(chains)
        return par, chains

    async def test_runs_all_chains(self):
        par, chains = self._make_parallel(["out1", "out2", "out3"])
        result = await par.run(input="x")
        for c in chains:
            c.run.assert_awaited_once()
        assert result.success is True

    async def test_combines_outputs_with_separator(self):
        par, _ = self._make_parallel(["A", "B"])
        result = await par.run()
        assert "A" in result.output
        assert "B" in result.output
        assert "---" in result.output

    async def test_metadata_contains_outputs(self):
        par, _ = self._make_parallel(["x", "y"])
        result = await par.run()
        assert result.metadata.get("count") == 2
        assert "outputs" in result.metadata

    async def test_one_failure_marks_result_failed(self):
        c0 = MagicMock()
        c0.run = AsyncMock(return_value=ChainResult(output="ok"))
        c1 = MagicMock()
        c1.run = AsyncMock(return_value=ChainResult(output="", success=False, error="fail"))

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            par = ParallelChain([c0, c1])
        result = await par.run()
        assert result.success is False

    async def test_exception_returns_error_result(self):
        c0 = MagicMock()
        c0.run = AsyncMock(side_effect=RuntimeError("parallel crash"))

        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            par = ParallelChain([c0])
        result = await par.run()
        assert result.success is False
        assert "parallel crash" in result.error


# ---------------------------------------------------------------------------
# ChainBuilder
# ---------------------------------------------------------------------------


class TestChainBuilder:
    def _make_client(self):
        client = Mock(spec=Client)
        client.model = "gpt-4o-mini"
        return client

    def test_with_template_sets_template(self):
        client = self._make_client()
        builder = ChainBuilder(client).with_template("Hello {name}")
        assert builder._template == "Hello {name}"

    def test_with_tools_sets_tools(self):
        client = self._make_client()
        tools = [Mock(), Mock()]
        builder = ChainBuilder(client).with_tools(tools)
        assert len(builder._tools) == 2

    def test_verbose_sets_flag(self):
        client = self._make_client()
        builder = ChainBuilder(client).verbose(True)
        assert builder._verbose is True

    def test_verbose_default_true(self):
        client = self._make_client()
        builder = ChainBuilder(client).verbose()
        assert builder._verbose is True

    def test_with_memory_buffer(self):
        client = self._make_client()
        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            builder = ChainBuilder(client).with_memory("buffer")
        assert builder._memory is not None

    def test_build_without_template_returns_chain(self):
        client = self._make_client()
        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            chain = ChainBuilder(client).build()
        assert isinstance(chain, Chain)

    def test_build_with_template_returns_prompt_chain(self):
        client = self._make_client()
        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            chain = ChainBuilder(client).with_template("T {x}").build()
        assert isinstance(chain, PromptChain)

    def test_fluent_chaining_returns_builder(self):
        client = self._make_client()
        builder = ChainBuilder(client)
        result = builder.with_template("t").with_tools([]).verbose(False)
        assert result is builder

    async def test_run_with_template(self):
        client = self._make_client()
        handler = MagicMock()
        resp = _make_response("builder response")
        handler.handle_run = AsyncMock(return_value=resp)
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            result = await ChainBuilder(client).with_template("Hello {name}").run(name="Alice")
        assert result.output == "builder response"

    async def test_run_without_template_uses_input(self):
        client = self._make_client()
        handler = MagicMock()
        resp = _make_response("basic response")
        handler.handle_run = AsyncMock(return_value=resp)
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            result = await ChainBuilder(client).run(input="my question")
        assert result.output == "basic response"

    async def test_run_without_template_uses_question(self):
        client = self._make_client()
        handler = MagicMock()
        resp = _make_response("q response")
        handler.handle_run = AsyncMock(return_value=resp)
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            result = await ChainBuilder(client).run(question="my question")
        assert result.output == "q response"


# ---------------------------------------------------------------------------
# create_chain
# ---------------------------------------------------------------------------


class TestCreateChain:
    def _client(self):
        c = Mock(spec=Client)
        c.model = "gpt-4o-mini"
        return c

    def test_create_basic_chain(self):
        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            chain = create_chain(self._client(), "basic")
        assert isinstance(chain, Chain)

    def test_create_prompt_chain(self):
        handler = MagicMock()
        container = _mock_container(handler)
        with patch("beanllm.utils.core.di_container.get_container", return_value=container):
            chain = create_chain(self._client(), "prompt", template="T {x}")
        assert isinstance(chain, PromptChain)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown chain type"):
            create_chain(self._client(), "unknown_type")
