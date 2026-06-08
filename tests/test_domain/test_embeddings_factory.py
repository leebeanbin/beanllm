"""Tests for domain/embeddings factory and base classes."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.embeddings.factory import Embedding, embed, embed_sync
from beanllm.domain.graph.graph_state import GraphState

# ---------------------------------------------------------------------------
# Embedding._detect_provider
# ---------------------------------------------------------------------------


class TestEmbeddingDetectProvider:
    def test_openai_model(self):
        assert Embedding._detect_provider("text-embedding-3-small") == "openai"
        assert Embedding._detect_provider("text-embedding-3-large") == "openai"
        assert Embedding._detect_provider("text-embedding-ada-002") == "openai"

    def test_gemini_model(self):
        assert Embedding._detect_provider("models/embedding-001") == "gemini"
        assert Embedding._detect_provider("text-embedding-004") == "gemini"

    def test_ollama_model(self):
        assert Embedding._detect_provider("nomic-embed-text") == "ollama"
        assert Embedding._detect_provider("mxbai-embed-large") == "ollama"

    def test_voyage_model(self):
        assert Embedding._detect_provider("voyage-2") == "voyage"
        assert Embedding._detect_provider("voyage-code-2") == "voyage"

    def test_jina_model(self):
        assert Embedding._detect_provider("jina-embeddings-v2-base-en") == "jina"

    def test_mistral_model(self):
        assert Embedding._detect_provider("mistral-embed") == "mistral"

    def test_cohere_model(self):
        assert Embedding._detect_provider("embed-english-v3.0") == "cohere"
        assert Embedding._detect_provider("embed-multilingual-v3.0") == "cohere"

    def test_unknown_model_returns_none(self):
        assert Embedding._detect_provider("unknown-random-model-xyz") is None


# ---------------------------------------------------------------------------
# Embedding.__new__ (factory method)
# ---------------------------------------------------------------------------


class TestEmbeddingFactory:
    def _make_mock_cls(self):
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls, mock_instance

    def test_autodetect_openai(self):
        mock_cls, mock_inst = self._make_mock_cls()
        with patch.dict(Embedding.PROVIDERS, {"openai": mock_cls}):
            result = Embedding(model="text-embedding-3-small")
        mock_cls.assert_called_once_with(model="text-embedding-3-small")
        assert result is mock_inst

    def test_explicit_provider(self):
        mock_cls, mock_inst = self._make_mock_cls()
        with patch.dict(Embedding.PROVIDERS, {"ollama": mock_cls}):
            result = Embedding(model="nomic-embed-text", provider="ollama")
        mock_cls.assert_called_once_with(model="nomic-embed-text")
        assert result is mock_inst

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            Embedding(model="x", provider="nonexistent_provider")

    def test_unknown_model_defaults_to_openai(self):
        mock_cls, mock_inst = self._make_mock_cls()
        with patch.dict(Embedding.PROVIDERS, {"openai": mock_cls}):
            result = Embedding(model="some-unknown-model-xyz")
        mock_cls.assert_called_once()
        assert result is mock_inst

    def test_kwargs_passed_through(self):
        mock_cls, mock_inst = self._make_mock_cls()
        with patch.dict(Embedding.PROVIDERS, {"openai": mock_cls}):
            Embedding(model="text-embedding-3-small", api_key="sk-test", batch_size=16)
        mock_cls.assert_called_once_with(
            model="text-embedding-3-small", api_key="sk-test", batch_size=16
        )


# ---------------------------------------------------------------------------
# Embedding.list_available_providers
# ---------------------------------------------------------------------------


class TestEmbeddingListAvailableProviders:
    def test_ollama_always_available(self):
        providers = Embedding.list_available_providers()
        assert "ollama" in providers

    def test_openai_available_with_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            providers = Embedding.list_available_providers()
        assert "openai" in providers

    def test_openai_not_available_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            providers = Embedding.list_available_providers()
        assert "openai" not in providers

    def test_gemini_available_with_google_key(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "key"}):
            providers = Embedding.list_available_providers()
        assert "gemini" in providers

    def test_gemini_available_with_gemini_key(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GOOGLE_API_KEY", "GEMINI_API_KEY")}
        env["GEMINI_API_KEY"] = "gkey"
        with patch.dict(os.environ, env, clear=True):
            providers = Embedding.list_available_providers()
        assert "gemini" in providers

    def test_returns_list(self):
        providers = Embedding.list_available_providers()
        assert isinstance(providers, list)


# ---------------------------------------------------------------------------
# Embedding.get_default_provider
# ---------------------------------------------------------------------------


class TestEmbeddingGetDefaultProvider:
    def test_returns_openai_when_key_set(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            provider = Embedding.get_default_provider()
        assert provider == "openai"

    def test_returns_ollama_when_no_api_keys(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(k.startswith(p) for p in ["OPENAI", "GOOGLE", "GEMINI", "VOYAGE", "COHERE"])
        }
        with patch.dict(os.environ, env, clear=True):
            provider = Embedding.get_default_provider()
        assert provider == "ollama"

    def test_returns_none_if_truly_none_available(self):
        with patch.object(Embedding, "list_available_providers", return_value=[]):
            provider = Embedding.get_default_provider()
        assert provider is None


# ---------------------------------------------------------------------------
# Convenience functions embed / embed_sync
# ---------------------------------------------------------------------------


class TestEmbedConvenienceFunctions:
    async def test_embed_single_string(self):
        mock_instance = MagicMock()
        mock_instance.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as MockEmbedding:
            MockEmbedding.return_value = mock_instance
            result = await embed("hello world")

        # Single string should be wrapped in list
        mock_instance.embed.assert_called_once_with(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]

    async def test_embed_list_of_strings(self):
        mock_instance = MagicMock()
        mock_instance.embed = AsyncMock(return_value=[[0.1], [0.2]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as MockEmbedding:
            MockEmbedding.return_value = mock_instance
            result = await embed(["text1", "text2"])

        mock_instance.embed.assert_called_once_with(["text1", "text2"])
        assert len(result) == 2

    def test_embed_sync_single_string(self):
        mock_instance = MagicMock()
        mock_instance.embed_sync = MagicMock(return_value=[[0.5, 0.6]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as MockEmbedding:
            MockEmbedding.return_value = mock_instance
            result = embed_sync("hello")

        mock_instance.embed_sync.assert_called_once_with(["hello"])
        assert result == [[0.5, 0.6]]

    def test_embed_sync_list(self):
        mock_instance = MagicMock()
        mock_instance.embed_sync = MagicMock(return_value=[[0.1], [0.2], [0.3]])

        with patch("beanllm.domain.embeddings.factory.Embedding") as MockEmbedding:
            MockEmbedding.return_value = mock_instance
            result = embed_sync(["a", "b", "c"])

        mock_instance.embed_sync.assert_called_once_with(["a", "b", "c"])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Embedding providers dict keys
# ---------------------------------------------------------------------------


class TestEmbeddingProvidersDict:
    def test_all_expected_providers_present(self):
        expected = {"openai", "gemini", "ollama", "voyage", "jina", "mistral", "cohere"}
        assert set(Embedding.PROVIDERS.keys()) == expected

    def test_provider_patterns_keys_match_providers(self):
        pattern_keys = set(Embedding.PROVIDER_PATTERNS.keys())
        provider_keys = set(Embedding.PROVIDERS.keys())
        assert pattern_keys == provider_keys

    def test_provider_env_vars_keys_match_providers(self):
        env_keys = set(Embedding.PROVIDER_ENV_VARS.keys())
        provider_keys = set(Embedding.PROVIDERS.keys())
        assert env_keys == provider_keys
