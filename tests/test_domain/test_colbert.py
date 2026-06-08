"""
Comprehensive tests for ColBERTRetriever and ColPaliRetriever.
Target: src/beanllm/domain/retrieval/colbert.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.retrieval.types import SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(content: str, score: float, rank: int = 1, doc_id: str = "doc_0"):
    return {
        "content": content,
        "score": score,
        "rank": rank,
        "document_id": doc_id,
    }


def _make_rag_mock(search_results=None):
    """Build a mock RAGPretrainedModel."""
    mock_rag = MagicMock()
    if search_results is None:
        search_results = [_make_search_result("Result content", 0.95)]
    mock_rag.search = MagicMock(return_value=search_results)
    mock_rag.index = MagicMock()
    mock_rag.add_to_index = MagicMock()
    return mock_rag


def _make_ragatouille_mock(mock_rag=None):
    if mock_rag is None:
        mock_rag = _make_rag_mock()
    mock_ragatouille = MagicMock()
    mock_ragatouille.RAGPretrainedModel = MagicMock()
    mock_ragatouille.RAGPretrainedModel.from_pretrained = MagicMock(return_value=mock_rag)
    mock_ragatouille.RAGPretrainedModel.from_index = MagicMock(return_value=mock_rag)
    return mock_ragatouille, mock_rag


def _make_retriever_with_docs(documents=None, mock_rag=None):
    """Create a ColBERTRetriever with mocked ragatouille.

    Returns (retriever, mock_rag_inst, patcher) where patcher is a
    context-manager that keeps ragatouille mocked.  Tests must call the
    returned retriever's methods while `patcher` is active.
    """
    documents = documents or ["Doc 1: Python.", "Doc 2: Java.", "Doc 3: C++."]
    mock_ragatouille, mock_rag_inst = _make_ragatouille_mock(mock_rag)
    patcher = patch.dict(sys.modules, {"ragatouille": mock_ragatouille})
    patcher.start()

    from beanllm.domain.retrieval.colbert import ColBERTRetriever

    retriever = ColBERTRetriever(
        model="colbert-ir/colbertv2.0",
        documents=documents,
    )

    return retriever, mock_rag_inst, patcher


# ---------------------------------------------------------------------------
# SearchResult type tests (from types.py)
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_basic_creation(self):
        r = SearchResult(text="hello", score=0.9)
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.metadata is None

    def test_with_metadata(self):
        r = SearchResult(text="t", score=0.5, metadata={"source": "x"})
        assert r.metadata["source"] == "x"

    def test_repr(self):
        r = SearchResult(text="hello world test", score=0.75)
        assert "0.7500" in repr(r)


# ---------------------------------------------------------------------------
# ColBERTRetriever — init
# ---------------------------------------------------------------------------


class TestColBERTRetrieverInit:
    def test_init_no_documents(self):
        mock_ragatouille, _ = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            assert r.model_name == "colbert-ir/colbertv2.0"
            assert r._indexed is False
            assert r._documents == []

    def test_init_with_documents_triggers_indexing(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever(documents=["doc1", "doc2"])
            assert r._indexed is True
            mock_rag.index.assert_called_once()

    def test_init_stores_parameters(self):
        mock_ragatouille, _ = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever(
                model="custom-model",
                index_name="my_index",
                use_gpu=False,
            )
            assert r.model_name == "custom-model"
            assert r.index_name == "my_index"
            assert r.use_gpu is False

    def test_init_with_document_ids(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever(
                documents=["d1", "d2"],
                document_ids=["id_a", "id_b"],
            )
            assert r._document_ids == ["id_a", "id_b"]

    def test_init_with_metadatas(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            metas = [{"source": "a"}, {"source": "b"}]
            r = ColBERTRetriever(documents=["d1", "d2"], document_metadatas=metas)
            assert r._document_metadatas == metas

    def test_repr(self):
        mock_ragatouille, _ = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            rep = repr(r)
            assert "ColBERTRetriever" in rep

    def test_init_ragatouille_missing_raises(self):
        with patch.dict(sys.modules, {"ragatouille": None}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever.__new__(ColBERTRetriever)
            r.model_name = "test"
            r.kwargs = {}
            r._rag = None
            with pytest.raises((ImportError, TypeError)):
                r._init_ragatouille()


# ---------------------------------------------------------------------------
# search tests
# ---------------------------------------------------------------------------


class TestColBERTSearch:
    def test_search_not_indexed_raises(self):
        mock_ragatouille, _ = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            with pytest.raises(ValueError, match="No documents indexed"):
                r.search("query")

    def test_search_returns_results(self):
        mock_rag = _make_rag_mock(
            [
                _make_search_result("Python is great", 0.95, 1, "doc_0"),
                _make_search_result("Java is popular", 0.85, 2, "doc_1"),
            ]
        )
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search("programming language", k=2)
            assert len(results) == 2
            assert all(isinstance(r, SearchResult) for r in results)
        finally:
            patcher.stop()

    def test_search_score_correct(self):
        mock_rag = _make_rag_mock([_make_search_result("content", 0.95, 1, "doc_0")])
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search("query")
            assert results[0].score == 0.95
        finally:
            patcher.stop()

    def test_search_text_correct(self):
        mock_rag = _make_rag_mock([_make_search_result("Found document text", 0.9, 1, "doc_0")])
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search("query")
            assert results[0].text == "Found document text"
        finally:
            patcher.stop()

    def test_search_includes_rank_in_metadata(self):
        mock_rag = _make_rag_mock([_make_search_result("content", 0.9, 3, "doc_0")])
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search("query")
            assert results[0].metadata["rank"] == 3
        finally:
            patcher.stop()

    def test_search_includes_doc_metadata(self):
        docs = ["doc1", "doc2"]
        metas = [{"source": "file1.txt"}, {"source": "file2.txt"}]
        mock_rag = _make_rag_mock([_make_search_result("doc1 content", 0.9, 1, "doc_0")])
        mock_ragatouille, _ = _make_ragatouille_mock(mock_rag)

        patcher = patch.dict(sys.modules, {"ragatouille": mock_ragatouille})
        patcher.start()
        try:
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            retriever = ColBERTRetriever(documents=docs, document_metadatas=metas)
            results = retriever.search("query")
            assert results[0].metadata.get("source") == "file1.txt"
        finally:
            patcher.stop()

    def test_search_result_without_doc_id(self):
        mock_rag = _make_rag_mock([{"content": "content", "score": 0.9, "rank": 1}])
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search("query")
            assert len(results) == 1
        finally:
            patcher.stop()

    def test_search_empty_result(self):
        mock_rag = _make_rag_mock([])
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search("query")
            assert results == []
        finally:
            patcher.stop()

    def test_search_calls_rag_with_k(self):
        mock_rag = _make_rag_mock([])
        retriever, mock_rag_inst, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            retriever.search("test", k=5)
            mock_rag_inst.search.assert_called_once_with(query="test", k=5)
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# add_documents tests
# ---------------------------------------------------------------------------


class TestAddDocuments:
    def test_add_documents_to_empty_retriever(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            r.add_documents(["new doc1", "new doc2"])
            assert r._indexed is True
            mock_rag.index.assert_called_once()

    def test_add_documents_to_indexed_retriever(self):
        retriever, mock_rag, patcher = _make_retriever_with_docs()
        try:
            retriever.add_documents(["new doc"])
            mock_rag.add_to_index.assert_called_once()
        finally:
            patcher.stop()

    def test_add_documents_updates_doc_list(self):
        retriever, _, patcher = _make_retriever_with_docs(["d1"])
        try:
            initial_count = len(retriever._documents)
            retriever.add_documents(["new_doc"])
            assert len(retriever._documents) == initial_count + 1
        finally:
            patcher.stop()

    def test_add_documents_generates_ids_when_none(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            r.add_documents(["d1", "d2"])
            assert r._document_ids == ["doc_0", "doc_1"]

    def test_add_documents_with_explicit_ids(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            r.add_documents(["d1"], document_ids=["my_id"])
            assert "my_id" in r._document_ids

    def test_add_documents_second_time_auto_ids_offset(self):
        retriever, mock_rag, patcher = _make_retriever_with_docs(["d1", "d2"])
        try:
            initial_len = len(retriever._documents)
            retriever.add_documents(["d3", "d4"])
            assert f"doc_{initial_len}" in retriever._document_ids
        finally:
            patcher.stop()

    def test_add_documents_updates_id_to_index_cache(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever()
            r.add_documents(["d1"], document_ids=["abc"])
            assert "abc" in r._doc_id_to_index


# ---------------------------------------------------------------------------
# search_batch tests
# ---------------------------------------------------------------------------


class TestSearchBatch:
    def test_search_batch_returns_list(self):
        mock_rag = _make_rag_mock([_make_search_result("res", 0.9)])
        retriever, _, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            results = retriever.search_batch(["q1", "q2"], k=3)
            assert len(results) == 2
            assert all(isinstance(r, list) for r in results)
        finally:
            patcher.stop()

    def test_search_batch_calls_search_per_query(self):
        mock_rag = _make_rag_mock([])
        retriever, mock_rag_inst, patcher = _make_retriever_with_docs(mock_rag=mock_rag)
        try:
            retriever.search_batch(["q1", "q2", "q3"])
            assert mock_rag_inst.search.call_count == 3
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# save_index tests
# ---------------------------------------------------------------------------


class TestSaveIndex:
    def test_save_index_does_not_raise(self, tmp_path):
        retriever, _, _ = _make_retriever_with_docs()
        retriever.save_index(str(tmp_path / "index"))
        # Should not raise

    def test_save_index_logs_path(self, tmp_path):
        retriever, _, _ = _make_retriever_with_docs()
        retriever.index_path = str(tmp_path / "idx")
        retriever.save_index(str(tmp_path / "other"))
        # Should not raise


# ---------------------------------------------------------------------------
# load_index classmethod tests
# ---------------------------------------------------------------------------


class TestLoadIndex:
    def test_load_index_creates_retriever(self, tmp_path):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            retriever = ColBERTRetriever.load_index(str(tmp_path / "index"))
            assert retriever._indexed is True
            assert retriever._rag is mock_rag

    def test_load_index_sets_path(self, tmp_path):
        path = str(tmp_path / "my_index")
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            retriever = ColBERTRetriever.load_index(path)
            assert retriever.index_path == path

    def test_load_index_raises_when_ragatouille_missing(self, tmp_path):
        with patch.dict(sys.modules, {"ragatouille": None}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            with pytest.raises((ImportError, TypeError)):
                ColBERTRetriever.load_index(str(tmp_path))


# ---------------------------------------------------------------------------
# _init_and_index internals
# ---------------------------------------------------------------------------


class TestInitAndIndex:
    def test_doc_id_to_index_cache_built(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever(
                documents=["d1", "d2"],
                document_ids=["a", "b"],
            )
            assert r._doc_id_to_index == {"a": 0, "b": 1}

    def test_auto_generated_ids_cached(self):
        mock_ragatouille, mock_rag = _make_ragatouille_mock()
        with patch.dict(sys.modules, {"ragatouille": mock_ragatouille}):
            from beanllm.domain.retrieval.colbert import ColBERTRetriever

            r = ColBERTRetriever(documents=["d1", "d2"])
            assert "doc_0" in r._doc_id_to_index
            assert "doc_1" in r._doc_id_to_index


# ---------------------------------------------------------------------------
# ColPaliRetriever tests
# ---------------------------------------------------------------------------


class TestColPaliRetriever:
    def _make_colpali(self, **kwargs):
        from beanllm.domain.retrieval.colbert import ColPaliRetriever

        return ColPaliRetriever(**kwargs)

    def test_init(self):
        r = self._make_colpali()
        assert r.model_name == "vidore/colpali"
        assert r._indexed is False
        assert r._model is None

    def test_repr(self):
        r = self._make_colpali()
        rep = repr(r)
        assert "ColPaliRetriever" in rep

    def test_search_not_indexed_raises(self):
        r = self._make_colpali()
        with pytest.raises(ValueError, match="No images indexed"):
            r.search("query")

    def test_init_model_raises_when_byaldi_missing(self):
        with patch.dict(sys.modules, {"byaldi": None}):
            r = self._make_colpali()
            with pytest.raises((ImportError, TypeError)):
                r._init_model()

    def test_add_images(self):
        mock_model = MagicMock()
        mock_model.index = MagicMock()
        mock_byaldi = MagicMock()
        mock_byaldi.RAGMultiModalModel = MagicMock()
        mock_byaldi.RAGMultiModalModel.from_pretrained = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"byaldi": mock_byaldi}):
            r = self._make_colpali()
            r.add_images(["doc1.pdf", "doc2.png"])
            assert r._indexed is True
            mock_model.index.assert_called_once()

    def test_search_returns_results(self):
        mock_results = [
            {"doc_id": "doc_0", "score": 0.9, "page_num": 1, "base64": "abc"},
        ]
        mock_model = MagicMock()
        mock_model.index = MagicMock()
        mock_model.search = MagicMock(return_value=mock_results)
        mock_byaldi = MagicMock()
        mock_byaldi.RAGMultiModalModel = MagicMock()
        mock_byaldi.RAGMultiModalModel.from_pretrained = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"byaldi": mock_byaldi}):
            r = self._make_colpali()
            r.add_images(["doc1.pdf"])
            results = r.search("find charts", k=3)
            assert len(results) == 1
            assert results[0].score == 0.9
            assert results[0].text == "doc_0"

    def test_search_result_metadata_keys(self):
        mock_results = [
            {"doc_id": "d", "score": 0.5, "page_num": 2, "base64": "xyz"},
        ]
        mock_model = MagicMock()
        mock_model.index = MagicMock()
        mock_model.search = MagicMock(return_value=mock_results)
        mock_byaldi = MagicMock()
        mock_byaldi.RAGMultiModalModel = MagicMock()
        mock_byaldi.RAGMultiModalModel.from_pretrained = MagicMock(return_value=mock_model)

        with patch.dict(sys.modules, {"byaldi": mock_byaldi}):
            r = self._make_colpali()
            r.add_images(["doc.pdf"])
            results = r.search("query")
            assert results[0].metadata["page_num"] == 2
            assert results[0].metadata["base64"] == "xyz"
