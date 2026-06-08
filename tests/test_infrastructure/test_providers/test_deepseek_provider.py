"""Tests for providers/deepseek_provider.py — DeepSeekProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider():
    """Create DeepSeekProvider with mocked API key and AsyncOpenAI client."""
    mock_client = MagicMock()

    with (
        patch("beanllm.utils.config.EnvConfig.DEEPSEEK_API_KEY", new="test-key"),
        patch("openai.AsyncOpenAI", return_value=mock_client),
    ):
        from beanllm.providers.deepseek_provider import DeepSeekProvider

        provider = DeepSeekProvider()
    provider.client = mock_client
    return provider, mock_client


# ---------------------------------------------------------------------------
# DeepSeekProvider.__init__
# ---------------------------------------------------------------------------


class TestDeepSeekProviderInit:
    def test_init_sets_default_model(self):
        provider, _ = _make_provider()
        assert provider.default_model == "deepseek-chat"

    def test_init_sets_available_models(self):
        provider, _ = _make_provider()
        assert "deepseek-chat" in provider._available_models
        assert "deepseek-reasoner" in provider._available_models

    def test_init_raises_without_api_key(self):
        with patch("beanllm.utils.config.EnvConfig.DEEPSEEK_API_KEY", new=None):
            from beanllm.providers.deepseek_provider import DeepSeekProvider

            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                DeepSeekProvider()

    def test_repr_contains_model(self):
        provider, _ = _make_provider()
        assert "deepseek-chat" in repr(provider)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestDeepSeekIsAvailable:
    def test_returns_true_with_key(self):
        provider, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.DEEPSEEK_API_KEY", new="test-key"):
            assert provider.is_available() is True

    def test_returns_false_without_key(self):
        provider, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.DEEPSEEK_API_KEY", new=None):
            assert provider.is_available() is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


class TestDeepSeekListModels:
    async def test_returns_list(self):
        provider, _ = _make_provider()
        models = await provider.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    async def test_includes_deepseek_chat(self):
        provider, _ = _make_provider()
        models = await provider.list_models()
        assert "deepseek-chat" in models


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


class TestDeepSeekChat:
    async def test_chat_calls_client(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from DeepSeek!"
        mock_response.model = "deepseek-chat"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        from beanllm.providers.base_provider import LLMResponse

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="deepseek-chat",
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from DeepSeek!"

    async def test_chat_with_system_message(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I am an assistant."
        mock_response.model = "deepseek-chat"
        mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=5, total_tokens=25)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Who are you?"}],
            model="deepseek-chat",
            system="You are a helpful assistant.",
        )
        # system message should be prepended
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"

    async def test_chat_with_max_tokens(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Short response."
        mock_response.model = "deepseek-chat"
        mock_response.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Say something."}],
            model="deepseek-chat",
            max_tokens=50,
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 50

    async def test_chat_without_max_tokens_omits_param(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response."
        mock_response.model = "deepseek-chat"
        mock_response.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="deepseek-chat",
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs


# ---------------------------------------------------------------------------
# stream_chat
# ---------------------------------------------------------------------------


class TestDeepSeekStreamChat:
    async def test_stream_chat_yields_chunks(self):
        provider, mock_client = _make_provider()

        async def mock_async_gen():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hello"
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = " world"
            yield chunk1
            yield chunk2

        mock_client.chat.completions.create = AsyncMock(return_value=mock_async_gen())

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="deepseek-chat",
        ):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    async def test_stream_chat_skips_empty_content(self):
        provider, mock_client = _make_provider()

        async def mock_async_gen():
            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = None
            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = "World"
            yield chunk1
            yield chunk2

        mock_client.chat.completions.create = AsyncMock(return_value=mock_async_gen())

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="deepseek-chat",
        ):
            chunks.append(chunk)

        assert chunks == ["World"]


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestDeepSeekHealthCheck:
    async def test_health_check_returns_true_on_success(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hi"
        mock_response.model = "deepseek-chat"
        mock_response.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await provider.health_check()
        assert result is True

    async def test_health_check_returns_false_on_error(self):
        provider, mock_client = _make_provider()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API error"))

        result = await provider.health_check()
        assert result is False
