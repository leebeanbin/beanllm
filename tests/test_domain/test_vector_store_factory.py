"""Tests for domain/vector_stores/factory.py — VectorStore, VectorStoreBuilder, convenience fns."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.vector_stores.factory import (
    VectorStore,
    VectorStoreBuilder,
    create_vector_store,
    from_documents,
)


def _mock_store_class():
    """Return a mock vector store class that creates a valid mock instance."""
    instance = MagicMock()
    instance.add_documents = MagicMock()
    cls = MagicMock(return_value=instance)
    return cls, instance


# ---------------------------------------------------------------------------
# VectorStore.list_available_providers
# ---------------------------------------------------------------------------


class TestListAvailableProviders:
    def test_chroma_always_available(self):
        providers = VectorStore.list_available_providers()
        assert "chroma" in providers

    def test_faiss_always_available(self):
        providers = VectorStore.list_available_providers()
        assert "faiss" in providers

    def test_pinecone_available_when_key_set(self):
        with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}):
            providers = VectorStore.list_available_providers()
        assert "pinecone" in providers

    def test_pinecone_not_available_when_no_key(self):
        env = {k: v for k, v in os.environ.items() if k != "PINECONE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            providers = VectorStore.list_available_providers()
        assert "pinecone" not in providers

    def test_returns_list(self):
        providers = VectorStore.list_available_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 1


# ---------------------------------------------------------------------------
# VectorStore.get_default_provider
# ---------------------------------------------------------------------------


class TestGetDefaultProvider:
    def test_returns_string(self):
        provider = VectorStore.get_default_provider()
        assert isinstance(provider, str)
        assert len(provider) > 0

    def test_returns_from_priority_list(self):
        provider = VectorStore.get_default_provider()
        assert provider in ["chroma", "faiss", "qdrant", "pinecone", "weaviate"]

    def test_chroma_is_first_priority(self):
        # When chroma is available (always), it should be selected
        provider = VectorStore.get_default_provider()
        assert provider == "chroma"  # chroma has no env var requirement

    def test_fallback_when_no_providers(self):
        # Even if nothing is available (it won't happen since chroma/faiss are local), returns chroma
        with patch.object(VectorStore, "list_available_providers", return_value=[]):
            provider = VectorStore.get_default_provider()
        assert provider == "chroma"


# ---------------------------------------------------------------------------
# VectorStore.__new__
# ---------------------------------------------------------------------------


class TestVectorStoreNew:
    def test_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            VectorStore(provider="totally_unknown_provider")

    def test_creates_instance_for_known_provider(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            result = VectorStore(provider="chroma", embedding_function=None)
        assert result is mock_instance

    def test_auto_detects_provider_when_none(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        with (
            patch.object(VectorStore, "PROVIDERS", mock_providers),
            patch.object(VectorStore, "PROVIDER_ENV_VARS", {"chroma": None}),
        ):
            result = VectorStore(provider=None, embedding_function=None)
        assert result is mock_instance

    def test_passes_kwargs_to_store_class(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            VectorStore(provider="chroma", embedding_function="fn", collection_name="test")
        mock_cls.assert_called_once_with(embedding_function="fn", collection_name="test")

    def test_classmethod_chroma(self):
        mock_cls, mock_instance = _mock_store_class()
        with patch("beanllm.domain.vector_stores.factory.ChromaVectorStore", mock_cls):
            result = VectorStore.chroma(embedding_function=None)
        assert result is mock_instance

    def test_classmethod_faiss(self):
        mock_cls, mock_instance = _mock_store_class()
        with patch("beanllm.domain.vector_stores.factory.FAISSVectorStore", mock_cls):
            result = VectorStore.faiss(embedding_function=None)
        assert result is mock_instance

    def test_classmethod_pinecone(self):
        mock_cls, mock_instance = _mock_store_class()
        with patch("beanllm.domain.vector_stores.factory.PineconeVectorStore", mock_cls):
            result = VectorStore.pinecone(embedding_function=None)
        assert result is mock_instance

    def test_classmethod_qdrant(self):
        mock_cls, mock_instance = _mock_store_class()
        with patch("beanllm.domain.vector_stores.factory.QdrantVectorStore", mock_cls):
            result = VectorStore.qdrant(embedding_function=None)
        assert result is mock_instance

    def test_classmethod_weaviate(self):
        mock_cls, mock_instance = _mock_store_class()
        with patch("beanllm.domain.vector_stores.factory.WeaviateVectorStore", mock_cls):
            result = VectorStore.weaviate(embedding_function=None)
        assert result is mock_instance


# ---------------------------------------------------------------------------
# VectorStoreBuilder
# ---------------------------------------------------------------------------


class TestVectorStoreBuilder:
    def test_default_provider_is_chroma(self):
        builder = VectorStoreBuilder()
        assert builder.provider == "chroma"

    def test_use_chroma_sets_provider(self):
        builder = VectorStoreBuilder().use_chroma()
        assert builder.provider == "chroma"

    def test_use_faiss_sets_provider(self):
        builder = VectorStoreBuilder().use_faiss()
        assert builder.provider == "faiss"

    def test_use_pinecone_sets_provider(self):
        builder = VectorStoreBuilder().use_pinecone()
        assert builder.provider == "pinecone"

    def test_use_qdrant_sets_provider(self):
        builder = VectorStoreBuilder().use_qdrant()
        assert builder.provider == "qdrant"

    def test_use_weaviate_sets_provider(self):
        builder = VectorStoreBuilder().use_weaviate()
        assert builder.provider == "weaviate"

    def test_with_embedding_sets_function(self):
        embed_fn = lambda texts: [[0.1] * 3] * len(texts)
        builder = VectorStoreBuilder().with_embedding(embed_fn)
        assert builder.embedding_function is embed_fn

    def test_with_collection_sets_kwarg(self):
        builder = VectorStoreBuilder().with_collection("my_collection")
        assert builder.kwargs.get("collection_name") == "my_collection"

    def test_use_chroma_passes_kwargs(self):
        builder = VectorStoreBuilder().use_chroma(persist_directory="/tmp/test")
        assert builder.kwargs.get("persist_directory") == "/tmp/test"

    def test_fluent_chaining_returns_builder(self):
        builder = VectorStoreBuilder()
        result = builder.use_faiss().with_collection("c")
        assert result is builder

    def test_build_creates_vector_store(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        builder = VectorStoreBuilder().use_chroma()
        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            result = builder.build()
        assert result is mock_instance

    def test_build_passes_embedding_to_store(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"faiss": mock_cls}
        embed_fn = lambda t: [[0.1]] * len(t)
        builder = VectorStoreBuilder().use_faiss().with_embedding(embed_fn)
        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            builder.build()
        mock_cls.assert_called_once_with(embedding_function=embed_fn)


# ---------------------------------------------------------------------------
# create_vector_store
# ---------------------------------------------------------------------------


class TestCreateVectorStore:
    def test_creates_store_with_provider(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            result = create_vector_store("chroma", embedding_function=None)
        assert result is mock_instance

    def test_creates_store_without_provider(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        with (
            patch.object(VectorStore, "PROVIDERS", mock_providers),
            patch.object(VectorStore, "PROVIDER_ENV_VARS", {"chroma": None}),
        ):
            result = create_vector_store(embedding_function=None)
        assert result is mock_instance


# ---------------------------------------------------------------------------
# from_documents
# ---------------------------------------------------------------------------


class TestFromDocuments:
    def test_creates_store_and_adds_documents(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        docs = [MagicMock(), MagicMock()]
        embed_fn = MagicMock()

        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            result = from_documents(docs, embed_fn, provider="chroma")

        mock_instance.add_documents.assert_called_once_with(docs)
        assert result is mock_instance

    def test_without_event_logger_no_logging(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        docs = [MagicMock()]
        embed_fn = MagicMock()

        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            result = from_documents(docs, embed_fn, provider="chroma", event_logger=None)
        assert result is not None

    def test_with_event_logger_logs_events(self):
        # from_documents is sync; uses asyncio.run() internally for event logging.
        # We verify log_event is called (as a coroutine factory) at least once.
        mock_cls, mock_instance = _mock_store_class()
        mock_providers = {"chroma": mock_cls}
        docs = [MagicMock()]
        embed_fn = MagicMock()

        logged_events = []

        async def fake_log(event, data):
            logged_events.append(event)

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: fake_log(e, d))

        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            result = from_documents(docs, embed_fn, provider="chroma", event_logger=mock_logger)

        assert result is mock_instance
        assert mock_logger.log_event.call_count >= 1

    def test_raises_on_exception_from_add_documents(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_instance.add_documents.side_effect = RuntimeError("storage error")
        mock_providers = {"chroma": mock_cls}
        docs = [MagicMock()]
        embed_fn = MagicMock()

        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            with pytest.raises(RuntimeError, match="storage error"):
                from_documents(docs, embed_fn, provider="chroma")

    def test_with_event_logger_logs_error_on_exception(self):
        mock_cls, mock_instance = _mock_store_class()
        mock_instance.add_documents.side_effect = RuntimeError("add failed")
        mock_providers = {"chroma": mock_cls}
        docs = [MagicMock()]
        embed_fn = MagicMock()

        async def fake_log(event, data):
            pass

        mock_logger = MagicMock()
        mock_logger.log_event = MagicMock(side_effect=lambda e, d: fake_log(e, d))

        with patch.object(VectorStore, "PROVIDERS", mock_providers):
            with pytest.raises(RuntimeError):
                from_documents(docs, embed_fn, provider="chroma", event_logger=mock_logger)

        event_calls = [call.args[0] for call in mock_logger.log_event.call_args_list]
        assert any("error" in ev for ev in event_calls)
