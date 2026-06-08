"""
Tests for ollama_provider.py

Mocks the ollama library to test OllamaProvider without a running Ollama server.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Build the ollama mock before importing the provider
# ---------------------------------------------------------------------------


def _make_ollama_mock():
    """Return a minimal mock for the ollama package."""
    mock = MagicMock()
    mock.AsyncClient = MagicMock
    return mock


_OLLAMA_MOCK = _make_ollama_mock()


@pytest.fixture(autouse=True)
def patch_ollama():
    """Ensure ollama is always mocked."""
    sys.modules["ollama"] = _OLLAMA_MOCK
    yield
    # Do not remove; leave for subsequent imports to avoid reload issues


@pytest.fixture
def mock_async_client():
    """A mock ollama.AsyncClient instance."""
    client = MagicMock()
    client.chat = AsyncMock()
    client.list = AsyncMock()
    client.host = "http://localhost:11434"
    return client


@pytest.fixture
def provider(mock_async_client):
    """An OllamaProvider with a mocked AsyncClient."""
    _OLLAMA_MOCK.AsyncClient = MagicMock(return_value=mock_async_client)

    # Patch EnvConfig to avoid env variable issues
    with patch("beanllm.providers.ollama_provider.EnvConfig") as mock_cfg:
        mock_cfg.OLLAMA_HOST = "http://localhost:11434"
        from beanllm.providers.ollama_provider import OllamaProvider

        prov = OllamaProvider.__new__(OllamaProvider)
        from beanllm.providers.base_provider import BaseLLMProvider

        BaseLLMProvider.__init__(prov, {})
        prov.client = mock_async_client
        prov.default_model = "qwen2.5:7b"
    return prov


# ---------------------------------------------------------------------------
# chat tests
# ---------------------------------------------------------------------------


class TestChat:
    async def test_chat_returns_llm_response(self, provider, mock_async_client):
        """chat() should return LLMResponse with content and model."""
        from beanllm.providers.base_provider import LLMResponse

        mock_async_client.chat.return_value = {
            "message": {"content": "Hello from Ollama!"},
            "model": "qwen2.5:7b",
        }

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="qwen2.5:7b",
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Ollama!"
        assert result.model == "qwen2.5:7b"

    async def test_chat_prepends_system_message(self, provider, mock_async_client):
        """chat() with system param should prepend system message."""
        mock_async_client.chat.return_value = {
            "message": {"content": "OK"},
            "model": "qwen2.5:7b",
        }

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="qwen2.5:7b",
            system="You are a pirate.",
        )

        call_kwargs = mock_async_client.chat.call_args[1]
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "You are a pirate."

    async def test_chat_uses_default_model_when_empty(self, provider, mock_async_client):
        """chat() with empty model string should use default_model."""
        mock_async_client.chat.return_value = {
            "message": {"content": "default response"},
            "model": "qwen2.5:7b",
        }

        result = await provider.chat(
            messages=[{"role": "user", "content": "test"}],
            model="",
        )

        call_kwargs = mock_async_client.chat.call_args[1]
        assert call_kwargs["model"] == "qwen2.5:7b"

    async def test_chat_passes_max_tokens_as_num_predict(self, provider, mock_async_client):
        """chat() should pass max_tokens as num_predict via options."""
        mock_async_client.chat.return_value = {
            "message": {"content": "Short"},
            "model": "qwen2.5:7b",
        }

        await provider.chat(
            messages=[{"role": "user", "content": "brief"}],
            model="qwen2.5:7b",
            max_tokens=50,
        )

        call_kwargs = mock_async_client.chat.call_args[1]
        assert call_kwargs["options"]["num_predict"] == 50

    async def test_chat_passes_kwargs_temperature(self, provider, mock_async_client):
        """chat() kwargs temperature should override default."""
        mock_async_client.chat.return_value = {
            "message": {"content": "hot"},
            "model": "qwen2.5:7b",
        }

        await provider.chat(
            messages=[{"role": "user", "content": "hot"}],
            model="qwen2.5:7b",
            temperature=0.9,
        )

        call_kwargs = mock_async_client.chat.call_args[1]
        assert call_kwargs["options"]["temperature"] == 0.9

    async def test_chat_raises_on_exception(self, provider, mock_async_client):
        """chat() should propagate exceptions."""
        mock_async_client.chat.side_effect = ConnectionError("Connection refused")

        with pytest.raises(Exception):
            await provider.chat(
                messages=[{"role": "user", "content": "hi"}],
                model="qwen2.5:7b",
            )

    async def test_chat_uses_stream_false(self, provider, mock_async_client):
        """chat() should call client.chat with stream=False."""
        mock_async_client.chat.return_value = {
            "message": {"content": "not streaming"},
            "model": "qwen2.5:7b",
        }

        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="qwen2.5:7b",
        )

        call_kwargs = mock_async_client.chat.call_args[1]
        assert call_kwargs["stream"] is False


# ---------------------------------------------------------------------------
# stream_chat tests
# ---------------------------------------------------------------------------


class TestStreamChat:
    async def test_stream_chat_yields_content(self, provider, mock_async_client):
        """stream_chat should yield content from stream parts."""
        parts = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " World"}},
            {"message": {"content": ""}},  # Empty content should be skipped
        ]

        async def fake_stream():
            for part in parts:
                yield part

        mock_async_client.chat.return_value = fake_stream()

        results = []
        async for token in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="qwen2.5:7b",
        ):
            results.append(token)

        assert results == ["Hello", " World"]

    async def test_stream_chat_with_system(self, provider, mock_async_client):
        """stream_chat with system param should prepend system message."""

        async def fake_stream():
            yield {"message": {"content": "ok"}}

        mock_async_client.chat.return_value = fake_stream()

        results = []
        async for token in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            model="qwen2.5:7b",
            system="Be brief.",
        ):
            results.append(token)

        call_kwargs = mock_async_client.chat.call_args[1]
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "Be brief."

    async def test_stream_chat_raises_on_exception(self, provider, mock_async_client):
        """stream_chat should propagate exceptions."""
        mock_async_client.chat.side_effect = Exception("Stream failed")

        with pytest.raises(Exception, match="Stream failed"):
            async for _ in provider.stream_chat(
                messages=[{"role": "user", "content": "hi"}],
                model="qwen2.5:7b",
            ):
                pass

    async def test_stream_chat_uses_stream_true(self, provider, mock_async_client):
        """stream_chat should call client.chat with stream=True."""

        async def fake_stream():
            yield {"message": {"content": "x"}}

        mock_async_client.chat.return_value = fake_stream()

        async for _ in provider.stream_chat(
            messages=[{"role": "user", "content": "hi"}],
            model="qwen2.5:7b",
        ):
            pass

        call_kwargs = mock_async_client.chat.call_args[1]
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# list_models tests
# ---------------------------------------------------------------------------


class TestListModels:
    async def test_list_models_from_list_response_object(self, provider, mock_async_client):
        """list_models should handle ListResponse objects with .models attribute."""
        mock_model = MagicMock()
        mock_model.model = "phi3.5:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        # Remove dict and list behavior to force hasattr("models") path
        del mock_response.__iter__
        mock_async_client.list.return_value = mock_response

        result = await provider.list_models()
        assert "phi3.5:latest" in result

    async def test_list_models_from_dict_response(self, provider, mock_async_client):
        """list_models should handle dict response."""
        mock_async_client.list.return_value = {
            "models": [
                {"name": "llama3.2:3b"},
                {"name": "mistral:7b"},
            ]
        }

        result = await provider.list_models()
        assert "llama3.2:3b" in result
        assert "mistral:7b" in result

    async def test_list_models_from_list_response(self, provider, mock_async_client):
        """list_models should handle list response."""
        mock_async_client.list.return_value = ["model-a", "model-b"]

        result = await provider.list_models()
        assert "model-a" in result
        assert "model-b" in result

    async def test_list_models_dict_model_key(self, provider, mock_async_client):
        """list_models should extract 'model' key from dict items if 'name' absent."""
        mock_async_client.list.return_value = {"models": [{"model": "qwen2.5:7b"}]}

        result = await provider.list_models()
        assert "qwen2.5:7b" in result

    async def test_list_models_returns_empty_on_connection_error(self, provider, mock_async_client):
        """list_models should return [] on ConnectionError."""
        mock_async_client.list.side_effect = ConnectionError("Server down")

        result = await provider.list_models()
        assert result == []

    async def test_list_models_returns_empty_on_generic_error(self, provider, mock_async_client):
        """list_models should return [] on generic errors."""
        mock_async_client.list.side_effect = RuntimeError("Unknown error")

        result = await provider.list_models()
        assert result == []

    async def test_list_models_skips_none_items(self, provider, mock_async_client):
        """list_models should skip None items gracefully."""
        mock_async_client.list.return_value = {"models": [None, {"name": "valid-model"}]}

        result = await provider.list_models()
        assert "valid-model" in result

    async def test_list_models_handles_unexpected_response_type(self, provider, mock_async_client):
        """list_models should handle unexpected response types gracefully."""
        mock_response = MagicMock(spec=[])  # no .models, not dict, not list
        mock_async_client.list.return_value = mock_response

        result = await provider.list_models()
        assert result == []

    async def test_list_models_model_object_with_model_attr(self, provider, mock_async_client):
        """list_models should extract name from Model-like objects in list."""
        m = MagicMock()
        m.model = "ollama-model:latest"
        # Use a plain list (not a ListResponse)
        mock_async_client.list.return_value = [m]

        result = await provider.list_models()
        assert "ollama-model:latest" in result


# ---------------------------------------------------------------------------
# is_available / health_check / close
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_is_available_always_true(self, provider):
        """Ollama doesn't require API key, always available."""
        assert provider.is_available() is True


class TestHealthCheck:
    async def test_health_check_returns_true_on_success(self, provider, mock_async_client):
        mock_async_client.list.return_value = {"models": []}

        result = await provider.health_check()
        assert result is True

    async def test_health_check_returns_false_on_error(self, provider, mock_async_client):
        mock_async_client.list.side_effect = Exception("Connection refused")

        result = await provider.health_check()
        assert result is False


class TestClose:
    async def test_close_does_not_raise(self, provider):
        """close() should complete without errors."""
        await provider.close()
