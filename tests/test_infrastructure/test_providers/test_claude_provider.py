"""Tests for providers/claude_provider.py — ClaudeProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_provider():
    """Create ClaudeProvider with mocked AsyncAnthropic and API key."""
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    with (
        patch("beanllm.providers.claude_provider.AsyncAnthropic", mock_anthropic),
        patch("beanllm.utils.config.EnvConfig.ANTHROPIC_API_KEY", new="test-anthropic-key"),
    ):
        from beanllm.providers.claude_provider import ClaudeProvider

        provider = ClaudeProvider()
    provider.client = mock_client
    return provider, mock_client


class TestClaudeProviderInit:
    def test_init_sets_default_model(self):
        provider, _ = _make_provider()
        assert "claude" in provider.default_model

    def test_init_creates_client(self):
        provider, mock_client = _make_provider()
        assert provider.client is mock_client

    def test_init_raises_without_anthropic(self):
        with (
            patch("beanllm.providers.claude_provider.AsyncAnthropic", None),
            patch("beanllm.utils.config.EnvConfig.ANTHROPIC_API_KEY", new="test-key"),
        ):
            from beanllm.providers.claude_provider import ClaudeProvider

            with pytest.raises(ImportError, match="anthropic"):
                ClaudeProvider()

    def test_init_raises_without_api_key(self):
        mock_anthropic = MagicMock()
        with (
            patch("beanllm.providers.claude_provider.AsyncAnthropic", mock_anthropic),
            patch("beanllm.utils.config.EnvConfig.ANTHROPIC_API_KEY", new=None),
        ):
            from beanllm.providers.claude_provider import ClaudeProvider

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                ClaudeProvider()


class TestClaudeIsAvailable:
    def test_returns_true_with_key(self):
        provider, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.ANTHROPIC_API_KEY", new="key"):
            assert provider.is_available() is True

    def test_returns_false_without_key(self):
        provider, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.ANTHROPIC_API_KEY", new=None):
            assert provider.is_available() is False


class TestClaudeListModels:
    async def test_returns_list_of_strings(self):
        provider, _ = _make_provider()
        models = await provider.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    async def test_includes_claude_3_5(self):
        provider, _ = _make_provider()
        models = await provider.list_models()
        assert any("claude-3-5" in m for m in models)


class TestClaudeChat:
    async def test_chat_returns_llm_response(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Claude says hello!")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        from beanllm.providers.base_provider import LLMResponse

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-5-sonnet-20241022",
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Claude says hello!"

    async def test_chat_with_system_message(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I am an assistant.")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = MagicMock(input_tokens=20, output_tokens=5)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Who are you?"}],
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant.",
        )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "You are a helpful assistant."

    async def test_chat_with_max_tokens(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Short.")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=1)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
        )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 50

    async def test_chat_includes_usage(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = MagicMock(input_tokens=15, output_tokens=7)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-5-sonnet-20241022",
        )
        assert result.usage["input_tokens"] == 15
        assert result.usage["output_tokens"] == 7

    async def test_chat_filters_non_user_assistant_roles(self):
        """Only 'user' and 'assistant' roles should pass through."""
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Sure")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = MagicMock(input_tokens=5, output_tokens=2)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            model="claude-3-5-sonnet-20241022",
        )
        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages_sent = call_kwargs["messages"]
        # system role should be filtered out
        assert all(m["role"] in ("user", "assistant") for m in messages_sent)


class TestClaudeStreamChat:
    async def test_stream_chat_yields_text(self):
        provider, mock_client = _make_provider()

        async def _text_stream():
            yield "Hello"
            yield " Claude"

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stream_ctx.text_stream = _text_stream()
        mock_client.messages.stream.return_value = mock_stream_ctx

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-5-sonnet-20241022",
        ):
            chunks.append(chunk)
        assert chunks == ["Hello", " Claude"]

    async def test_stream_chat_with_system(self):
        provider, mock_client = _make_provider()

        async def _text_stream():
            yield "Content"

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_stream_ctx.text_stream = _text_stream()
        mock_client.messages.stream.return_value = mock_stream_ctx

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3-5-sonnet-20241022",
            system="Be brief.",
        ):
            chunks.append(chunk)
        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert call_kwargs.get("system") == "Be brief."


class TestClaudeHealthCheck:
    async def test_returns_true_on_success(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="OK")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        assert await provider.health_check() is True

    async def test_returns_false_on_error(self):
        provider, mock_client = _make_provider()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        assert await provider.health_check() is False
