"""Tests for providers/perplexity_provider.py — PerplexityProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_provider():
    mock_client = MagicMock()
    with (
        patch("beanllm.utils.config.EnvConfig.PERPLEXITY_API_KEY", new="pplx-test-key"),
        patch("openai.AsyncOpenAI", return_value=mock_client),
    ):
        from beanllm.providers.perplexity_provider import PerplexityProvider

        provider = PerplexityProvider()
    provider.client = mock_client
    return provider, mock_client


class TestPerplexityProviderInit:
    def test_init_sets_default_model(self):
        provider, _ = _make_provider()
        assert "sonar" in provider.default_model

    def test_init_sets_available_models(self):
        provider, _ = _make_provider()
        assert any("sonar" in m for m in provider._available_models)

    def test_init_raises_without_api_key(self):
        with patch("beanllm.utils.config.EnvConfig.PERPLEXITY_API_KEY", new=None):
            from beanllm.providers.perplexity_provider import PerplexityProvider

            with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
                PerplexityProvider()

    def test_repr_contains_model(self):
        provider, _ = _make_provider()
        assert "sonar" in repr(provider).lower() or "Perplexity" in repr(provider)


class TestPerplexityIsAvailable:
    def test_returns_true_with_key(self):
        provider, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.PERPLEXITY_API_KEY", new="key"):
            assert provider.is_available() is True

    def test_returns_false_without_key(self):
        provider, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.PERPLEXITY_API_KEY", new=None):
            assert provider.is_available() is False


class TestPerplexityListModels:
    async def test_returns_list(self):
        provider, _ = _make_provider()
        models = await provider.list_models()
        assert isinstance(models, list)
        assert len(models) > 0


class TestPerplexityChat:
    async def test_chat_returns_llm_response(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Perplexity answer"
        mock_response.model = "sonar"
        mock_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        from beanllm.providers.base_provider import LLMResponse

        result = await provider.chat(
            messages=[{"role": "user", "content": "What is the weather?"}],
            model="sonar",
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Perplexity answer"

    async def test_chat_with_system_prepends_system_message(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "sonar"
        mock_response.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="sonar",
            system="Be helpful.",
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "Be helpful."

    async def test_chat_with_max_tokens(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer."
        mock_response.model = "sonar"
        mock_response.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="sonar",
            max_tokens=100,
        )
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("max_tokens") == 100


class TestPerplexityStreamChat:
    async def test_stream_chat_yields_chunks(self):
        provider, mock_client = _make_provider()

        async def mock_gen():
            c1 = MagicMock()
            c1.choices = [MagicMock()]
            c1.choices[0].delta.content = "Answer "
            c2 = MagicMock()
            c2.choices = [MagicMock()]
            c2.choices[0].delta.content = "here."
            yield c1
            yield c2

        mock_client.chat.completions.create = AsyncMock(return_value=mock_gen())
        chunks = []
        async for c in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="sonar",
        ):
            chunks.append(c)
        assert chunks == ["Answer ", "here."]

    async def test_stream_chat_skips_none_content(self):
        provider, mock_client = _make_provider()

        async def mock_gen():
            c1 = MagicMock()
            c1.choices = [MagicMock()]
            c1.choices[0].delta.content = None
            c2 = MagicMock()
            c2.choices = [MagicMock()]
            c2.choices[0].delta.content = "Content"
            yield c1
            yield c2

        mock_client.chat.completions.create = AsyncMock(return_value=mock_gen())
        chunks = []
        async for c in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="sonar",
        ):
            chunks.append(c)
        assert chunks == ["Content"]


class TestPerplexityHealthCheck:
    async def test_returns_true_on_success(self):
        provider, mock_client = _make_provider()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hi"
        mock_response.model = "sonar"
        mock_response.usage = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        assert await provider.health_check() is True

    async def test_returns_false_on_error(self):
        provider, mock_client = _make_provider()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("error"))
        assert await provider.health_check() is False
