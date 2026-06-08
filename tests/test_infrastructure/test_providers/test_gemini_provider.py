"""Tests for providers/gemini_provider.py — GeminiProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_provider():
    """Create GeminiProvider with mocked genai and API key."""
    mock_genai = MagicMock()
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    with (
        patch("beanllm.providers.gemini_provider.genai", mock_genai),
        patch("beanllm.utils.config.EnvConfig.GEMINI_API_KEY", new="test-gemini-key"),
    ):
        from beanllm.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider()
    provider.client = mock_client
    return provider, mock_client, mock_genai


class TestGeminiProviderInit:
    def test_init_sets_default_model(self):
        provider, _, _ = _make_provider()
        assert "gemini" in provider.default_model

    def test_init_creates_client(self):
        provider, mock_client, _ = _make_provider()
        assert provider.client is mock_client

    def test_init_raises_without_genai(self):
        with (
            patch("beanllm.providers.gemini_provider.genai", None),
            patch("beanllm.utils.config.EnvConfig.GEMINI_API_KEY", new="test-key"),
        ):
            from beanllm.providers.gemini_provider import GeminiProvider

            with pytest.raises(ImportError, match="google-generativeai"):
                GeminiProvider()

    def test_init_raises_without_api_key(self):
        mock_genai = MagicMock()
        with (
            patch("beanllm.providers.gemini_provider.genai", mock_genai),
            patch("beanllm.utils.config.EnvConfig.GEMINI_API_KEY", new=None),
        ):
            from beanllm.providers.gemini_provider import GeminiProvider

            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                GeminiProvider()


class TestGeminiIsAvailable:
    def test_returns_true_with_key(self):
        provider, _, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.GEMINI_API_KEY", new="key"):
            assert provider.is_available() is True

    def test_returns_false_without_key(self):
        provider, _, _ = _make_provider()
        with patch("beanllm.utils.config.EnvConfig.GEMINI_API_KEY", new=None):
            assert provider.is_available() is False


class TestGeminiListModels:
    async def test_returns_list_of_strings(self):
        provider, _, _ = _make_provider()
        models = await provider.list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    async def test_includes_gemini_2(self):
        provider, _, _ = _make_provider()
        models = await provider.list_models()
        assert any("gemini-2" in m for m in models)


class TestGeminiChat:
    async def test_chat_returns_llm_response(self):
        provider, mock_client, _ = _make_provider()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        from beanllm.providers.base_provider import LLMResponse

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.0-flash",
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Gemini!"

    async def test_chat_with_system_prepends_to_contents(self):
        provider, mock_client, _ = _make_provider()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.0-flash",
            system="Be concise.",
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        assert "Be concise." in contents

    async def test_chat_with_assistant_message(self):
        provider, mock_client, _ = _make_provider()
        mock_response = MagicMock()
        mock_response.text = "I remember saying that."
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "What did you say?"},
            ],
            model="gemini-2.0-flash",
        )
        assert result.content == "I remember saying that."

    async def test_chat_with_max_tokens(self):
        provider, mock_client, _ = _make_provider()
        mock_response = MagicMock()
        mock_response.text = "Short."
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.0-flash",
            max_tokens=100,
        )
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs.get("max_output_tokens") == 100

    async def test_chat_response_without_text_attr_uses_str(self):
        provider, mock_client, _ = _make_provider()
        mock_response = MagicMock(spec=[])  # No .text attribute
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.0-flash",
        )
        assert isinstance(result.content, str)


class TestGeminiStreamChat:
    async def test_stream_chat_yields_chunks(self):
        provider, mock_client, _ = _make_provider()

        async def _mock_stream():
            c1 = MagicMock()
            c1.text = "Hello"
            c2 = MagicMock()
            c2.text = " world"
            yield c1
            yield c2

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_mock_stream())

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.0-flash",
        ):
            chunks.append(chunk)
        assert chunks == ["Hello", " world"]

    async def test_stream_chat_skips_none_text_chunks(self):
        provider, mock_client, _ = _make_provider()

        async def _mock_stream():
            c1 = MagicMock()
            c1.text = None
            c2 = MagicMock()
            c2.text = "Content"
            yield c1
            yield c2

        mock_client.aio.models.generate_content_stream = AsyncMock(return_value=_mock_stream())

        chunks = []
        async for chunk in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemini-2.0-flash",
        ):
            chunks.append(chunk)
        assert chunks == ["Content"]


class TestGeminiHealthCheck:
    async def test_returns_true_on_success(self):
        provider, mock_client, _ = _make_provider()
        mock_response = MagicMock()
        mock_response.text = "hi"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        assert await provider.health_check() is True

    async def test_returns_false_on_error(self):
        provider, mock_client, _ = _make_provider()
        mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("Gemini error"))

        assert await provider.health_check() is False
