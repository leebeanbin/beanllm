"""Tests for providers/base_provider.py and providers/provider_factory.py."""

from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.providers.base_provider import BaseLLMProvider, LLMResponse

# ---------------------------------------------------------------------------
# Concrete implementation for testing abstract BaseLLMProvider
# ---------------------------------------------------------------------------


class ConcreteProvider(BaseLLMProvider):
    """Minimal concrete provider for testing BaseLLMProvider utility methods."""

    def stream_chat(
        self, messages, model, system=None, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        async def _gen():
            yield "chunk"

        return _gen()

    async def chat(
        self, messages, model, system=None, temperature=0.7, max_tokens=None
    ) -> LLMResponse:
        return LLMResponse(content="response", model=model)

    async def list_models(self) -> List[str]:
        return ["model-1"]

    def is_available(self) -> bool:
        return True

    async def health_check(self) -> bool:
        return True


def _make_provider():
    return ConcreteProvider(config={"key": "test"})


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------


class TestLLMResponse:
    def test_creates_with_content_and_model(self):
        r = LLMResponse(content="Hello", model="gpt-4o")
        assert r.content == "Hello"
        assert r.model == "gpt-4o"

    def test_usage_defaults_to_none(self):
        r = LLMResponse(content="Hi", model="gpt-4o")
        assert r.usage is None

    def test_creates_with_usage(self):
        r = LLMResponse(content="Hi", model="gpt-4o", usage={"total_tokens": 10})
        assert r.usage["total_tokens"] == 10


# ---------------------------------------------------------------------------
# BaseLLMProvider._handle_provider_error
# ---------------------------------------------------------------------------


class TestHandleProviderError:
    def test_returns_provider_error_instance(self):
        from beanllm.utils.exceptions import ProviderError

        provider = _make_provider()
        err = provider._handle_provider_error(ValueError("oops"), "chat")
        assert isinstance(err, ProviderError)

    def test_error_message_contains_operation(self):
        from beanllm.utils.exceptions import ProviderError

        provider = _make_provider()
        err = provider._handle_provider_error(ValueError("oops"), "stream_chat")
        assert "stream_chat" in str(err)

    def test_custom_fallback_message_used(self):
        from beanllm.utils.exceptions import ProviderError

        provider = _make_provider()
        err = provider._handle_provider_error(
            ValueError("api fail"), "chat", fallback_message="Custom error"
        )
        assert "Custom error" in str(err)

    def test_original_error_included_in_message(self):
        from beanllm.utils.exceptions import ProviderError

        provider = _make_provider()
        err = provider._handle_provider_error(ValueError("unique-detail-xyz"), "chat")
        assert "unique-detail-xyz" in str(err)


# ---------------------------------------------------------------------------
# BaseLLMProvider._extract_params
# ---------------------------------------------------------------------------


class TestExtractParams:
    def test_uses_explicit_temperature_and_max_tokens(self):
        provider = _make_provider()
        params = provider._extract_params({}, temperature=0.5, max_tokens=100)
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100

    def test_kwargs_override_defaults(self):
        provider = _make_provider()
        params = provider._extract_params(
            {"temperature": 0.9, "max_tokens": 50}, temperature=0.1, max_tokens=10
        )
        assert params["temperature"] == 0.9
        assert params["max_tokens"] == 50

    def test_none_defaults_when_no_kwargs(self):
        provider = _make_provider()
        params = provider._extract_params({})
        assert params["temperature"] is None
        assert params["max_tokens"] is None


# ---------------------------------------------------------------------------
# BaseLLMProvider._prepare_openai_messages
# ---------------------------------------------------------------------------


class TestPrepareOpenAIMessages:
    def test_no_system_returns_same_messages(self):
        provider = _make_provider()
        msgs = [{"role": "user", "content": "Hello"}]
        result = provider._prepare_openai_messages(msgs)
        assert result == msgs

    def test_system_prepended_when_provided(self):
        provider = _make_provider()
        msgs = [{"role": "user", "content": "Hello"}]
        result = provider._prepare_openai_messages(msgs, system="Be helpful.")
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful."
        assert result[1] == msgs[0]

    def test_does_not_mutate_original_list(self):
        provider = _make_provider()
        msgs = [{"role": "user", "content": "Hi"}]
        provider._prepare_openai_messages(msgs, system="Prompt")
        assert len(msgs) == 1  # original unchanged


# ---------------------------------------------------------------------------
# BaseLLMProvider._extract_openai_usage
# ---------------------------------------------------------------------------


class TestExtractOpenAIUsage:
    def test_returns_none_when_no_usage_attr(self):
        provider = _make_provider()
        mock_resp = MagicMock(spec=[])  # no 'usage' attr
        assert provider._extract_openai_usage(mock_resp) is None

    def test_returns_none_when_usage_is_falsy(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.usage = None
        assert provider._extract_openai_usage(mock_resp) is None

    def test_extracts_usage_tokens(self):
        provider = _make_provider()
        mock_resp = MagicMock()
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 20
        mock_resp.usage.total_tokens = 30
        result = provider._extract_openai_usage(mock_resp)
        assert result == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


# ---------------------------------------------------------------------------
# BaseLLMProvider._safe_health_check
# ---------------------------------------------------------------------------


class TestSafeHealthCheck:
    async def test_returns_true_when_check_succeeds(self):
        provider = _make_provider()
        result = await provider._safe_health_check(AsyncMock(return_value=True))
        assert result is True

    async def test_returns_false_when_check_fails(self):
        provider = _make_provider()
        result = await provider._safe_health_check(AsyncMock(side_effect=Exception("failed")))
        assert result is False

    async def test_returns_false_not_true_on_exception(self):
        provider = _make_provider()
        check = AsyncMock(side_effect=RuntimeError("boom"))
        result = await provider._safe_health_check(check)
        assert result is False


# ---------------------------------------------------------------------------
# BaseLLMProvider._safe_is_available
# ---------------------------------------------------------------------------


class TestSafeIsAvailable:
    def test_returns_true_when_check_returns_true(self):
        provider = _make_provider()
        assert provider._safe_is_available(lambda: True) is True

    def test_returns_false_when_check_returns_false(self):
        provider = _make_provider()
        assert provider._safe_is_available(lambda: False) is False

    def test_returns_false_on_exception(self):
        provider = _make_provider()

        def raiser():
            raise RuntimeError("env var missing")

        assert provider._safe_is_available(raiser) is False


# ---------------------------------------------------------------------------
# ProviderFactory
# ---------------------------------------------------------------------------


class TestProviderFactoryGetAvailableProviders:
    def test_returns_list(self):
        from beanllm.providers.provider_factory import ProviderFactory

        with patch(
            "beanllm.providers.provider_factory.ProviderFactory._get_provider_priority",
            return_value=[],
        ):
            result = ProviderFactory.get_available_providers()
        assert isinstance(result, list)

    def test_returns_available_provider_name(self):
        from beanllm.providers.provider_factory import ProviderFactory

        mock_cls = MagicMock()
        mock_cls.return_value.is_available.return_value = True

        with (
            patch(
                "beanllm.providers.provider_factory.ProviderFactory._get_provider_priority",
                return_value=[("test-provider", mock_cls, "TEST_KEY")],
            ),
            patch(
                "beanllm.providers.provider_registry.is_provider_env_available", return_value=True
            ),
        ):
            result = ProviderFactory.get_available_providers()
        assert "test-provider" in result


class TestProviderFactoryClearCache:
    def test_clear_cache_empties_instances(self):
        from beanllm.providers.provider_factory import ProviderFactory

        ProviderFactory._instances["test"] = MagicMock()
        ProviderFactory.clear_cache()
        assert "test" not in ProviderFactory._instances

    def test_clear_cache_calls_close_if_available(self):
        from beanllm.providers.provider_factory import ProviderFactory

        mock_provider = MagicMock()

        async def close():
            pass

        mock_provider.close = close
        ProviderFactory._instances["mock"] = mock_provider
        ProviderFactory.clear_cache()
        assert len(ProviderFactory._instances) == 0


class TestProviderFactoryGetProvider:
    def test_raises_for_unknown_provider(self):
        from beanllm.providers.provider_factory import ProviderFactory

        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.get_provider("completely-unknown-xyz")

    def test_raises_when_no_providers_available(self):
        from beanllm.providers.provider_factory import ProviderFactory

        ProviderFactory._instances.clear()
        with (
            patch(
                "beanllm.providers.provider_factory.ProviderFactory._get_provider_priority",
                return_value=[],
            ),
        ):
            with pytest.raises(ValueError):
                ProviderFactory.get_provider()

    def test_returns_cached_instance(self):
        from beanllm.providers.provider_factory import ProviderFactory

        mock_provider = MagicMock(spec=BaseLLMProvider)
        ProviderFactory._instances["cached-provider"] = mock_provider

        result = ProviderFactory.get_provider("cached-provider")
        assert result is mock_provider
        ProviderFactory._instances.pop("cached-provider", None)

    def test_get_default_provider_calls_get_provider(self):
        from beanllm.providers.provider_factory import ProviderFactory

        mock_provider = MagicMock(spec=BaseLLMProvider)
        with patch.object(ProviderFactory, "get_provider", return_value=mock_provider):
            result = ProviderFactory.get_default_provider()
        assert result is mock_provider
