"""Tests for domain/embeddings/api/api_embeddings.py — API-based embedding providers."""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.embeddings.api.api_embeddings import OpenAIEmbedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_embedding(model="text-embedding-3-small"):
    """Create OpenAIEmbedding with mocked clients."""
    mock_async_client = MagicMock()
    mock_sync_client = MagicMock()
    with (
        patch("openai.AsyncOpenAI", return_value=mock_async_client),
        patch("openai.OpenAI", return_value=mock_sync_client),
    ):
        emb = OpenAIEmbedding(model=model, api_key="test-key")
    emb.async_client = mock_async_client
    emb.sync_client = mock_sync_client
    return emb, mock_async_client, mock_sync_client


def _make_embedding_response(vectors: List[List[float]]):
    """Create a mock OpenAI embeddings response."""
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=v) for v in vectors]
    mock_resp.usage = MagicMock(total_tokens=10)
    return mock_resp


# ---------------------------------------------------------------------------
# OpenAIEmbedding
# ---------------------------------------------------------------------------


class TestOpenAIEmbeddingInit:
    def test_stores_model(self):
        emb, _, _ = _make_openai_embedding()
        assert emb.model == "text-embedding-3-small"

    def test_stores_api_key(self):
        emb, _, _ = _make_openai_embedding()
        assert emb.api_key == "test-key"

    def test_raises_without_api_key(self):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("openai.AsyncOpenAI"),
            patch("openai.OpenAI"),
        ):
            import os

            old = os.environ.pop("OPENAI_API_KEY", None)
            try:
                with pytest.raises(ValueError, match="OpenAI"):
                    OpenAIEmbedding(model="text-embedding-3-small")
            finally:
                if old:
                    os.environ["OPENAI_API_KEY"] = old


class TestOpenAIEmbeddingEmbed:
    async def test_embed_returns_vectors(self):
        emb, mock_async, _ = _make_openai_embedding()
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_async.embeddings.create = AsyncMock(return_value=_make_embedding_response(vectors))
        result = await emb.embed(["hello", "world"])
        assert result == vectors

    async def test_embed_calls_with_correct_model(self):
        emb, mock_async, _ = _make_openai_embedding("text-embedding-3-large")
        mock_async.embeddings.create = AsyncMock(return_value=_make_embedding_response([[0.1]]))
        await emb.embed(["text"])
        call_kwargs = mock_async.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-large"

    async def test_embed_raises_on_error(self):
        emb, mock_async, _ = _make_openai_embedding()
        mock_async.embeddings.create = AsyncMock(side_effect=Exception("API error"))
        with pytest.raises(Exception, match="API error"):
            await emb.embed(["text"])


class TestOpenAIEmbeddingEmbedSync:
    def test_embed_sync_returns_vectors(self):
        emb, _, mock_sync = _make_openai_embedding()
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        mock_sync.embeddings.create.return_value = _make_embedding_response(vectors)
        result = emb.embed_sync(["hello", "world"])
        assert result == vectors

    def test_embed_sync_calls_with_model(self):
        emb, _, mock_sync = _make_openai_embedding("text-embedding-3-small")
        mock_sync.embeddings.create.return_value = _make_embedding_response([[0.1]])
        emb.embed_sync(["text"])
        call_kwargs = mock_sync.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"

    def test_embed_sync_raises_on_error(self):
        emb, _, mock_sync = _make_openai_embedding()
        mock_sync.embeddings.create.side_effect = Exception("API error")
        with pytest.raises(Exception):
            emb.embed_sync(["text"])


# ---------------------------------------------------------------------------
# GeminiEmbedding
# ---------------------------------------------------------------------------


def _make_gemini_embedding():
    mock_genai = MagicMock()
    with (
        patch.dict("sys.modules", {"google.generativeai": mock_genai}),
        patch("beanllm.domain.embeddings.api.api_embeddings.GeminiEmbedding._validate_import"),
        patch(
            "beanllm.domain.embeddings.api.api_embeddings.GeminiEmbedding._get_api_key",
            return_value="test-key",
        ),
    ):
        from beanllm.domain.embeddings.api.api_embeddings import GeminiEmbedding

        emb = GeminiEmbedding(api_key="test-key")
    emb.genai = mock_genai
    return emb, mock_genai


class TestGeminiEmbeddingEmbedSync:
    def test_embed_sync_batch_success(self):
        emb, mock_genai = _make_gemini_embedding()
        mock_genai.embed_content.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        result = emb.embed_sync(["text1", "text2"])
        assert len(result) == 2

    def test_embed_sync_single_embedding_in_batch(self):
        emb, mock_genai = _make_gemini_embedding()
        mock_genai.embed_content.return_value = {"embedding": [0.1, 0.2]}
        result = emb.embed_sync(["text1"])
        assert result == [[0.1, 0.2]]

    def test_embed_sync_list_response(self):
        emb, mock_genai = _make_gemini_embedding()
        mock_genai.embed_content.return_value = [[0.1, 0.2], [0.3, 0.4]]
        result = emb.embed_sync(["text1", "text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_sync_batch_fallback_to_sequential(self):
        emb, mock_genai = _make_gemini_embedding()
        # First call raises ValueError (unexpected format), trigger fallback
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Unexpected batch response format")
            return {"embedding": [0.1, 0.2]}

        mock_genai.embed_content.side_effect = side_effect
        result = emb.embed_sync(["text1"])
        # Should have called once for batch (failed), once for sequential
        assert call_count[0] == 2

    def test_embed_sync_raises_on_outer_exception(self):
        emb, mock_genai = _make_gemini_embedding()
        mock_genai.embed_content.side_effect = RuntimeError("Connection error")
        with pytest.raises(RuntimeError):
            emb.embed_sync(["text"])


# ---------------------------------------------------------------------------
# OllamaEmbedding
# ---------------------------------------------------------------------------


def _make_ollama_embedding():
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_ollama.Client.return_value = mock_client
    with (
        patch.dict("sys.modules", {"ollama": mock_ollama}),
        patch("beanllm.domain.embeddings.api.api_embeddings.OllamaEmbedding._validate_import"),
    ):
        from beanllm.domain.embeddings.api.api_embeddings import OllamaEmbedding

        emb = OllamaEmbedding(model="nomic-embed-text")
    emb.client = mock_client
    return emb, mock_client


class TestOllamaEmbeddingEmbedSync:
    def test_embed_sync_with_dict_embeddings_response(self):
        emb, mock_client = _make_ollama_embedding()
        mock_client.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        result = emb.embed_sync(["text1", "text2"])
        assert len(result) == 2

    def test_embed_sync_with_list_response(self):
        emb, mock_client = _make_ollama_embedding()
        mock_client.embed.return_value = [[0.1, 0.2]]
        result = emb.embed_sync(["text1"])
        assert result == [[0.1, 0.2]]

    def test_embed_sync_raises_on_error(self):
        emb, mock_client = _make_ollama_embedding()
        mock_client.embed.side_effect = Exception("Ollama error")
        with pytest.raises((Exception, RuntimeError)):
            emb.embed_sync(["text"])


# ---------------------------------------------------------------------------
# VoyageEmbedding
# ---------------------------------------------------------------------------


def _make_voyage_embedding():
    mock_voyage = MagicMock()
    mock_client = MagicMock()
    mock_voyage.Client.return_value = mock_client
    with (
        patch.dict("sys.modules", {"voyageai": mock_voyage}),
        patch("beanllm.domain.embeddings.api.api_embeddings.VoyageEmbedding._validate_import"),
        patch(
            "beanllm.domain.embeddings.api.api_embeddings.VoyageEmbedding._get_api_key",
            return_value="voyage-key",
        ),
    ):
        from beanllm.domain.embeddings.api.api_embeddings import VoyageEmbedding

        emb = VoyageEmbedding(api_key="voyage-key")
    emb.client = mock_client
    return emb, mock_client


class TestVoyageEmbeddingEmbedSync:
    def test_embed_sync_returns_embeddings(self):
        emb, mock_client = _make_voyage_embedding()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_result
        result = emb.embed_sync(["text1", "text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_sync_raises_on_error(self):
        emb, mock_client = _make_voyage_embedding()
        mock_client.embed.side_effect = Exception("Voyage error")
        with pytest.raises(Exception):
            emb.embed_sync(["text"])


# ---------------------------------------------------------------------------
# JinaEmbedding
# ---------------------------------------------------------------------------


def _make_jina_embedding():
    with patch(
        "beanllm.domain.embeddings.api.api_embeddings.JinaEmbedding._get_api_key",
        return_value="jina-key",
    ):
        from beanllm.domain.embeddings.api.api_embeddings import JinaEmbedding

        emb = JinaEmbedding(api_key="jina-key")
    return emb


class TestJinaEmbeddingEmbedSync:
    def test_embed_sync_returns_embeddings(self):
        emb = _make_jina_embedding()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
        }
        with patch("httpx.post", return_value=mock_response):
            result = emb.embed_sync(["text1", "text2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_sync_raises_on_error(self):
        emb = _make_jina_embedding()
        with patch("httpx.post", side_effect=Exception("Jina error")):
            with pytest.raises(Exception):
                emb.embed_sync(["text"])
