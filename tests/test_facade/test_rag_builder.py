"""Tests for facade/core/rag_builder.py — RAGBuilder and create_rag."""

from unittest.mock import MagicMock, patch

import pytest

from beanllm.facade.core.rag_builder import RAGBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_documents(n: int = 2):
    docs = []
    for i in range(n):
        doc = MagicMock()
        doc.page_content = f"Document content {i}"
        doc.metadata = {}
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# RAGBuilder init
# ---------------------------------------------------------------------------


class TestRAGBuilderInit:
    def test_documents_none_initially(self):
        b = RAGBuilder()
        assert b.documents is None

    def test_embedding_none_initially(self):
        b = RAGBuilder()
        assert b.embedding is None

    def test_vector_store_none_initially(self):
        b = RAGBuilder()
        assert b.vector_store is None

    def test_llm_client_none_initially(self):
        b = RAGBuilder()
        assert b.llm_client is None

    def test_retriever_config_empty_dict(self):
        b = RAGBuilder()
        assert b.retriever_config == {}


# ---------------------------------------------------------------------------
# Fluent methods
# ---------------------------------------------------------------------------


class TestRAGBuilderFluentMethods:
    def test_load_documents_from_list_stores_directly(self):
        b = RAGBuilder()
        docs = _make_documents(3)
        result = b.load_documents(docs)
        assert b.documents == docs
        assert result is b  # fluent

    def test_split_text_returns_self(self):
        b = RAGBuilder()
        result = b.split_text(chunk_size=500, chunk_overlap=50)
        assert result is b
        assert b.chunk_size == 500
        assert b.chunk_overlap == 50

    def test_embed_with_sets_embedding(self):
        b = RAGBuilder()
        mock_emb = MagicMock()
        result = b.embed_with(mock_emb)
        assert b.embedding is mock_emb
        assert result is b

    def test_store_in_sets_vector_store(self):
        b = RAGBuilder()
        mock_vs = MagicMock()
        result = b.store_in(mock_vs)
        assert b.vector_store is mock_vs
        assert result is b

    def test_use_llm_sets_client(self):
        b = RAGBuilder()
        mock_client = MagicMock()
        result = b.use_llm(mock_client)
        assert b.llm_client is mock_client
        assert result is b

    def test_with_prompt_sets_template(self):
        b = RAGBuilder()
        result = b.with_prompt("Answer using context: {context}")
        assert b.prompt_template == "Answer using context: {context}"
        assert result is b

    def test_with_retriever_config_updates_dict(self):
        b = RAGBuilder()
        result = b.with_retriever_config(k=5, search_type="mmr")
        assert b.retriever_config["k"] == 5
        assert b.retriever_config["search_type"] == "mmr"
        assert result is b

    def test_with_retriever_config_merges_multiple_calls(self):
        b = RAGBuilder()
        b.with_retriever_config(k=5)
        b.with_retriever_config(search_type="mmr")
        assert b.retriever_config["k"] == 5
        assert b.retriever_config["search_type"] == "mmr"

    def test_load_documents_from_path_calls_loader(self):
        b = RAGBuilder()
        mock_docs = _make_documents(2)
        with patch("beanllm.domain.loaders.DocumentLoader.load", return_value=mock_docs):
            b.load_documents("/tmp/test_docs")
        assert b.documents == mock_docs


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class TestRAGBuilderBuild:
    def test_build_raises_if_no_documents(self):
        b = RAGBuilder()
        with pytest.raises(ValueError, match="Documents not loaded"):
            b.build()

    def test_build_raises_if_embedding_missing_embed_sync(self):
        b = RAGBuilder()
        b.documents = _make_documents(1)
        mock_emb = MagicMock(spec=[])  # no embed_sync attr
        b.embedding = mock_emb

        mock_chunks = _make_documents(1)
        with (
            patch("beanllm.domain.splitters.TextSplitter.split", return_value=mock_chunks),
            pytest.raises(ValueError, match="embed_sync"),
        ):
            b.build()

    def test_build_succeeds_with_mock_components(self):
        b = RAGBuilder()
        docs = _make_documents(2)
        b.documents = docs

        mock_chunks = _make_documents(2)
        mock_vs = MagicMock()
        mock_rag = MagicMock()
        mock_emb = MagicMock()
        mock_emb.embed_sync = MagicMock(return_value=[[0.1, 0.2]])

        with (
            patch("beanllm.domain.splitters.TextSplitter.split", return_value=mock_chunks),
            patch("beanllm.domain.vector_stores.from_documents", return_value=mock_vs),
            patch("beanllm.facade.core.rag_facade.RAGChain", return_value=mock_rag),
            patch("beanllm.utils.core.di_container.get_container"),
        ):
            b.embedding = mock_emb
            b.llm_client = MagicMock()
            result = b.build()
        assert result is mock_rag

    def test_build_uses_provided_vector_store(self):
        b = RAGBuilder()
        b.documents = _make_documents(2)
        b.chunks = _make_documents(2)  # skip split step

        mock_vs = MagicMock()
        b.vector_store = mock_vs

        mock_emb = MagicMock()
        mock_emb.embed_sync = MagicMock(return_value=[[0.1]])
        b.embedding = mock_emb
        b.llm_client = MagicMock()

        mock_rag = MagicMock()
        with (
            patch("beanllm.facade.core.rag_facade.RAGChain", return_value=mock_rag),
        ):
            b.build()
        mock_vs.add_documents.assert_called_once()

    def test_build_auto_selects_ollama_embedding_for_ollama_llm(self):
        b = RAGBuilder()
        b.documents = _make_documents(1)
        b.chunks = _make_documents(1)

        mock_client = MagicMock()
        mock_client.provider = "ollama"
        mock_client.model = "llama3"
        b.llm_client = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_sync = MagicMock(return_value=[[0.1]])
        mock_vs = MagicMock()
        mock_rag = MagicMock()

        with (
            patch("beanllm.domain.embeddings.Embedding", return_value=mock_emb) as mock_emb_cls,
            patch("beanllm.domain.vector_stores.from_documents", return_value=mock_vs),
            patch("beanllm.facade.core.rag_facade.RAGChain", return_value=mock_rag),
        ):
            b.build()
        mock_emb_cls.assert_called_once_with(model="nomic-embed-text")

    def test_build_auto_selects_ollama_embedding_for_llama_model(self):
        b = RAGBuilder()
        b.documents = _make_documents(1)
        b.chunks = _make_documents(1)

        mock_client = MagicMock()
        mock_client.provider = "openai"
        mock_client.model = "llama3:8b"
        b.llm_client = mock_client

        mock_emb = MagicMock()
        mock_emb.embed_sync = MagicMock(return_value=[[0.1]])
        mock_vs = MagicMock()
        mock_rag = MagicMock()

        with (
            patch("beanllm.domain.embeddings.Embedding", return_value=mock_emb) as mock_emb_cls,
            patch("beanllm.domain.vector_stores.from_documents", return_value=mock_vs),
            patch("beanllm.facade.core.rag_facade.RAGChain", return_value=mock_rag),
        ):
            b.build()
        mock_emb_cls.assert_called_once_with(model="nomic-embed-text")
