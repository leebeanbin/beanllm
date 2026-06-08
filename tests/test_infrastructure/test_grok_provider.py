"""Tests for GrokProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

MODULE = "beanllm.providers.grok_provider"


def _make_openai_response(content: str = "hello", model: str = "grok-4"):
    resp = MagicMock()
    resp.model = model
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 3
    resp.usage.total_tokens = 8
    return resp


def _make_grok(api_key: str = "xai-test-key", **kwargs):
    with (
        patch(f"{MODULE}.EnvConfig") as mock_cfg,
        patch(f"{MODULE}.AsyncOpenAI"),
    ):
        mock_cfg.XAI_API_KEY = api_key
        from beanllm.providers.grok_provider import GrokProvider

        return GrokProvider(**kwargs)


class TestGrokProviderInit:
    def test_raises_without_api_key(self):
        with patch(f"{MODULE}.EnvConfig") as mock_cfg:
            mock_cfg.XAI_API_KEY = None
            with patch(f"{MODULE}.AsyncOpenAI"):
                from beanllm.providers.grok_provider import GrokProvider

                with pytest.raises(ValueError, match="XAI_API_KEY"):
                    GrokProvider()

    def test_raises_without_openai_package(self):
        with patch(f"{MODULE}.AsyncOpenAI", None):
            from beanllm.providers.grok_provider import GrokProvider

            with pytest.raises(ImportError, match="openai"):
                GrokProvider()

    def test_default_model(self):
        provider = _make_grok()
        assert provider.default_model == "grok-4"

    def test_repr(self):
        provider = _make_grok()
        assert "GrokProvider" in repr(provider)
        assert "grok-4" in repr(provider)


class TestGrokProviderIsAvailable:
    def test_available_with_key(self):
        with patch(f"{MODULE}.EnvConfig") as mock_cfg:
            mock_cfg.XAI_API_KEY = "xai-test"
            from beanllm.providers.grok_provider import GrokProvider

            p = object.__new__(GrokProvider)
            assert p.is_available() is True

    def test_not_available_without_key(self):
        with patch(f"{MODULE}.EnvConfig") as mock_cfg:
            mock_cfg.XAI_API_KEY = None
            from beanllm.providers.grok_provider import GrokProvider

            p = object.__new__(GrokProvider)
            assert p.is_available() is False


class TestGrokProviderListModels:
    async def test_list_models_returns_list(self):
        provider = _make_grok()
        models = await provider.list_models()
        assert isinstance(models, list)
        assert any("grok" in m for m in models)


class TestGrokProviderChat:
    async def test_chat_returns_llm_response(self):
        provider = _make_grok()
        fake_resp = _make_openai_response("Hi there")
        provider.client.chat.completions.create = AsyncMock(return_value=fake_resp)
        provider._acquire_rate_limit = AsyncMock()

        from beanllm.providers.base_provider import LLMResponse

        result = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="grok-4",
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there"
        assert result.model == "grok-4"

    async def test_chat_passes_max_tokens(self):
        provider = _make_grok()
        fake_resp = _make_openai_response()
        provider.client.chat.completions.create = AsyncMock(return_value=fake_resp)
        provider._acquire_rate_limit = AsyncMock()

        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="grok-4",
            max_tokens=100,
        )
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    async def test_chat_passes_system_message(self):
        provider = _make_grok()
        fake_resp = _make_openai_response()
        provider.client.chat.completions.create = AsyncMock(return_value=fake_resp)
        provider._acquire_rate_limit = AsyncMock()

        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="grok-4",
            system="You are helpful",
        )
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful"


class TestGrokProviderStreamChat:
    async def test_stream_chat_yields_chunks(self):
        provider = _make_grok()
        provider._acquire_rate_limit = AsyncMock()

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        async def fake_stream():
            yield chunk1
            yield chunk2

        provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            model="grok-4",
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    async def test_stream_chat_skips_empty_content(self):
        provider = _make_grok()
        provider._acquire_rate_limit = AsyncMock()

        empty_chunk = MagicMock()
        empty_chunk.choices = [MagicMock()]
        empty_chunk.choices[0].delta.content = None

        async def fake_stream():
            yield empty_chunk

        provider.client.chat.completions.create = AsyncMock(return_value=fake_stream())

        chunks = [c async for c in provider.stream_chat([], model="grok-4")]
        assert chunks == []


class TestGrokHealthCheck:
    async def test_health_check_true_on_success(self):
        provider = _make_grok()
        fake_resp = _make_openai_response("ok")
        provider.client.chat.completions.create = AsyncMock(return_value=fake_resp)
        provider._acquire_rate_limit = AsyncMock()

        result = await provider.health_check()
        assert result is True

    async def test_health_check_false_on_exception(self):
        provider = _make_grok()
        provider.client.chat.completions.create = AsyncMock(side_effect=Exception("down"))
        provider._acquire_rate_limit = AsyncMock()

        result = await provider.health_check()
        assert result is False
