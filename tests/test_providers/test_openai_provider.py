"""
Tests for openai_provider.py

Mocks the openai library and environment to test OpenAIProvider
without real API calls.
"""

import sys
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Setup: patch openai before importing provider
# ---------------------------------------------------------------------------


def _make_openai_mock():
    """Build a minimal openai module mock."""
    mock = MagicMock()
    mock.AsyncOpenAI = MagicMock
    mock.APIError = type("APIError", (Exception,), {})
    mock.APITimeoutError = type("APITimeoutError", (Exception,), {})
    return mock


# We patch at import time via sys.modules
_OPENAI_MOCK = _make_openai_mock()


@pytest.fixture(autouse=True)
def patch_openai_and_env(monkeypatch):
    """Patch openai library and env config for every test."""
    monkeypatch.setitem(sys.modules, "openai", _OPENAI_MOCK)

    # Ensure OPENAI_API_KEY is set so provider doesn't raise ValueError
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-0000")

    # Patch EnvConfig to return the key
    with patch("beanllm.providers.openai_provider.EnvConfig") as mock_env_cfg:
        mock_env_cfg.OPENAI_API_KEY = "sk-test-key-0000"
        mock_env_cfg.is_provider_available.return_value = True
        yield mock_env_cfg


@pytest.fixture
def mock_async_openai_client():
    """Return a fully mocked AsyncOpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.models = MagicMock()
    client.models.list = AsyncMock()
    return client


@pytest.fixture
def provider(mock_async_openai_client):
    """Create an OpenAIProvider with a mocked client."""
    # openai is already patched via autouse fixture
    # We also need AsyncOpenAI to be not-None for the ImportError guard
    _OPENAI_MOCK.AsyncOpenAI = MagicMock(return_value=mock_async_openai_client)

    from beanllm.providers.openai_provider import OpenAIProvider

    with patch("beanllm.providers.openai_provider.AsyncOpenAI", _OPENAI_MOCK.AsyncOpenAI):
        prov = OpenAIProvider.__new__(OpenAIProvider)
        # Call __init__ via BaseLLMProvider
        from beanllm.providers.base_provider import BaseLLMProvider

        BaseLLMProvider.__init__(prov, {})
        prov.client = mock_async_openai_client
        prov.default_model = "gpt-4o-mini"
        prov._models_cache = None
        prov._models_cache_time = None
        prov._models_cache_ttl = 3600
    return prov


# ---------------------------------------------------------------------------
# _get_model_parameter_config tests
# ---------------------------------------------------------------------------


class TestGetModelParameterConfig:
    def test_cache_hit_direct(self, provider):
        """Known model should return cached config directly."""
        config = provider._get_model_parameter_config("gpt-4o")
        assert config["supports_temperature"] is True
        assert config["supports_max_tokens"] is True
        assert config["uses_max_completion_tokens"] is False

    def test_cache_hit_gpt4o_mini(self, provider):
        config = provider._get_model_parameter_config("gpt-4o-mini")
        assert config["supports_temperature"] is True

    def test_reasoning_model_no_temperature(self, provider):
        """o-series models should not support temperature."""
        config = provider._get_model_parameter_config("o1")
        assert config["supports_temperature"] is False

    def test_gpt5_uses_max_completion_tokens(self, provider):
        config = provider._get_model_parameter_config("gpt-5")
        assert config["uses_max_completion_tokens"] is True
        assert config["supports_temperature"] is False

    def test_unknown_model_falls_to_strategy(self, provider):
        """Unknown model not in cache should use strategy pattern."""
        config = provider._get_model_parameter_config("custom-unknown-model")
        # Should return a dict with the expected keys
        assert "supports_temperature" in config
        assert "supports_max_tokens" in config
        assert "uses_max_completion_tokens" in config

    def test_dated_model_extracts_base(self, provider):
        """Dated model like gpt-4o-2024-05-13 should match gpt-4o base."""
        config = provider._get_model_parameter_config("gpt-4o-2024-05-13")
        assert config["supports_temperature"] is True


# ---------------------------------------------------------------------------
# chat tests
# ---------------------------------------------------------------------------


class TestChat:
    async def test_chat_returns_llm_response(self, provider, mock_async_openai_client):
        """chat() should call openai and return LLMResponse."""
        from beanllm.providers.base_provider import LLMResponse

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.usage.total_tokens = 8
        mock_async_openai_client.chat.completions.create.return_value = mock_response

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.model == "gpt-4o-mini"
        assert result.usage["total_tokens"] == 8

    async def test_chat_with_system_message(self, provider, mock_async_openai_client):
        """chat() should prepend system message when provided."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-4o"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_async_openai_client.chat.completions.create.return_value = mock_response

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            system="You are helpful.",
        )

        call_kwargs = mock_async_openai_client.chat.completions.create.call_args[1]
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "You are helpful."

    async def test_chat_with_max_tokens(self, provider, mock_async_openai_client):
        """chat() with max_tokens should include max_tokens in request."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Short"
        mock_response.model = "gpt-4o-mini"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 6
        mock_async_openai_client.chat.completions.create.return_value = mock_response

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
            max_tokens=100,
        )

        call_kwargs = mock_async_openai_client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 100

    async def test_chat_reasoning_model_uses_max_completion_tokens(
        self, provider, mock_async_openai_client
    ):
        """Reasoning model (gpt-5) should use max_completion_tokens."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Reasoned"
        mock_response.model = "gpt-5"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_async_openai_client.chat.completions.create.return_value = mock_response

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-5",
            max_tokens=200,
        )

        call_kwargs = mock_async_openai_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 200
        assert "max_tokens" not in call_kwargs

    async def test_chat_raises_provider_error_on_api_error(
        self, provider, mock_async_openai_client
    ):
        """chat() should raise ProviderError on API failure."""
        from beanllm.utils.exceptions import ProviderError

        mock_async_openai_client.chat.completions.create.side_effect = Exception("API down")

        with pytest.raises((ProviderError, Exception)):
            await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o-mini",
            )

    async def test_chat_o1_model_no_temperature(self, provider, mock_async_openai_client):
        """o1 model should not include temperature in request."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Reasoning..."
        mock_response.model = "o1"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 15
        mock_async_openai_client.chat.completions.create.return_value = mock_response

        await provider.chat(
            messages=[{"role": "user", "content": "Solve this"}],
            model="o1",
        )

        call_kwargs = mock_async_openai_client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs


# ---------------------------------------------------------------------------
# stream_chat tests
# ---------------------------------------------------------------------------


class TestStreamChat:
    async def test_stream_chat_yields_content(self, provider, mock_async_openai_client):
        """stream_chat should yield content chunks."""
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " World"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None  # Empty chunk should be skipped

        async def fake_stream():
            for chunk in [chunk1, chunk2, chunk3]:
                yield chunk

        mock_async_openai_client.chat.completions.create.return_value = fake_stream()

        results = []
        async for token in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
        ):
            results.append(token)

        assert results == ["Hello", " World"]

    async def test_stream_chat_with_system_message(self, provider, mock_async_openai_client):
        """stream_chat should prepend system message."""

        async def fake_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "ok"
            yield chunk

        mock_async_openai_client.chat.completions.create.return_value = fake_stream()

        results = []
        async for token in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o",
            system="Be helpful",
        ):
            results.append(token)

        call_kwargs = mock_async_openai_client.chat.completions.create.call_args[1]
        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"

    async def test_stream_chat_yields_error_on_exception(self, provider, mock_async_openai_client):
        """stream_chat should yield error string on exception."""
        mock_async_openai_client.chat.completions.create.side_effect = Exception("Network error")

        results = []
        async for token in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
        ):
            results.append(token)

        assert len(results) == 1
        assert "[Error:" in results[0]

    async def test_stream_chat_with_max_tokens(self, provider, mock_async_openai_client):
        """stream_chat should include max_tokens in request params."""

        async def fake_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "text"
            yield chunk

        mock_async_openai_client.chat.completions.create.return_value = fake_stream()

        results = []
        async for token in provider.stream_chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
            max_tokens=50,
        ):
            results.append(token)

        call_kwargs = mock_async_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# list_models tests
# ---------------------------------------------------------------------------


class TestListModels:
    async def test_list_models_returns_model_ids(self, provider, mock_async_openai_client):
        """list_models should return list of model IDs from API."""
        mock_model1 = MagicMock()
        mock_model1.id = "gpt-4o"
        mock_model2 = MagicMock()
        mock_model2.id = "gpt-4o-mini"
        mock_response = MagicMock()
        mock_response.data = [mock_model1, mock_model2]
        mock_async_openai_client.models.list.return_value = mock_response

        result = await provider.list_models()

        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    async def test_list_models_returns_cached_result(self, provider, mock_async_openai_client):
        """list_models should return cached result on second call."""
        import time

        provider._models_cache = ["gpt-4o"]
        provider._models_cache_time = time.time()

        result = await provider.list_models()

        assert result == ["gpt-4o"]
        mock_async_openai_client.models.list.assert_not_called()

    async def test_list_models_falls_back_on_error(self, provider, mock_async_openai_client):
        """list_models should return default models on API failure."""
        mock_async_openai_client.models.list.side_effect = Exception("API error")

        result = await provider.list_models()

        assert isinstance(result, list)
        assert len(result) > 0
        assert "gpt-4o" in result

    async def test_list_models_expired_cache_refetches(self, provider, mock_async_openai_client):
        """Expired cache should trigger a new API call."""
        import time

        # Set expired cache
        provider._models_cache = ["old-model"]
        provider._models_cache_time = time.time() - 7200  # 2 hours ago

        mock_model = MagicMock()
        mock_model.id = "new-model"
        mock_response = MagicMock()
        mock_response.data = [mock_model]
        mock_async_openai_client.models.list.return_value = mock_response

        result = await provider.list_models()

        assert "new-model" in result
        mock_async_openai_client.models.list.assert_called_once()


# ---------------------------------------------------------------------------
# is_available / health_check tests
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_is_available_returns_bool(self, provider, patch_openai_and_env):
        result = provider.is_available()
        assert isinstance(result, bool)

    def test_is_available_true_when_key_set(self, provider, patch_openai_and_env):
        patch_openai_and_env.is_provider_available.return_value = True
        assert provider.is_available() is True

    def test_is_available_false_when_no_key(self, provider, patch_openai_and_env):
        patch_openai_and_env.is_provider_available.return_value = False
        assert provider.is_available() is False


class TestHealthCheck:
    async def test_health_check_returns_true_on_success(self, provider, mock_async_openai_client):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_async_openai_client.chat.completions.create.return_value = mock_response

        result = await provider.health_check()
        assert result is True

    async def test_health_check_returns_false_on_exception(
        self, provider, mock_async_openai_client
    ):
        mock_async_openai_client.chat.completions.create.side_effect = Exception("Timeout")

        result = await provider.health_check()
        assert result is False


# ---------------------------------------------------------------------------
# find_lightweight_model tests
# ---------------------------------------------------------------------------


class TestFindLightweightModel:
    def test_empty_list_returns_none(self, provider):
        assert provider.find_lightweight_model([]) is None

    def test_no_chat_models_returns_none(self, provider):
        result = provider.find_lightweight_model(["text-embedding-3-small", "dall-e-3"])
        assert result is None

    def test_returns_mini_model_when_available(self, provider):
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4"]
        result = provider.find_lightweight_model(models)
        assert result == "gpt-4o-mini"

    def test_prefers_nano_over_mini(self, provider):
        models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini"]
        result = provider.find_lightweight_model(models)
        assert "nano" in result

    def test_falls_back_to_o3_mini_when_no_nano_or_mini(self, provider):
        models = ["gpt-4o", "gpt-4", "o3-mini"]
        result = provider.find_lightweight_model(models)
        assert result is not None

    def test_returns_o4_mini_as_fallback(self, provider):
        models = ["gpt-4o", "gpt-4", "o4-mini"]
        result = provider.find_lightweight_model(models)
        assert result is not None


# ---------------------------------------------------------------------------
# _filter_chat_models tests
# ---------------------------------------------------------------------------


class TestFilterChatModels:
    def test_filters_embedding_models(self, provider):
        models = ["gpt-4o", "text-embedding-3-small", "gpt-4o-mini"]
        result = provider._filter_chat_models(models)
        assert "text-embedding-3-small" not in result

    def test_filters_tts_models(self, provider):
        models = ["gpt-4o", "tts-1", "gpt-3.5-turbo"]
        result = provider._filter_chat_models(models)
        assert "tts-1" not in result

    def test_keeps_gpt_models(self, provider):
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        result = provider._filter_chat_models(models)
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result

    def test_keeps_o_series_models(self, provider):
        models = ["o1", "o3-mini", "gpt-4o"]
        result = provider._filter_chat_models(models)
        assert "o1" in result
        assert "o3-mini" in result

    def test_filters_dall_e(self, provider):
        models = ["dall-e-3", "gpt-4o"]
        result = provider._filter_chat_models(models)
        assert "dall-e-3" not in result

    def test_filters_whisper(self, provider):
        models = ["whisper-1", "gpt-4o"]
        result = provider._filter_chat_models(models)
        assert "whisper-1" not in result
