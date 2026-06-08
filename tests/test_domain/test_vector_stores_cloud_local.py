"""Tests for Milvus, Weaviate, LanceDB, and Qdrant vector stores via sys.modules mocking."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_doc(content: str = "hello", metadata: dict | None = None):
    from beanllm.domain.loaders import Document

    return Document(content=content, metadata=metadata or {})


def _embed_fn(texts: List[str]) -> List[List[float]]:
    return [[0.1 * (i + 1)] * 4 for i in range(len(texts))]


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------


class _MilvusMocks:
    """Context manager that injects mock pymilvus into sys.modules."""

    def __init__(self, collection_exists: bool = False):
        self._collection_exists = collection_exists
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        mock_pymilvus = ModuleType("pymilvus")
        mock_utility = MagicMock()
        mock_utility.has_collection.return_value = self._collection_exists

        mock_connections = MagicMock()
        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]  # empty hits by default

        mock_pymilvus.connections = mock_connections
        mock_pymilvus.Collection = MagicMock(return_value=mock_collection)
        mock_pymilvus.CollectionSchema = MagicMock()
        mock_pymilvus.FieldSchema = MagicMock()
        mock_pymilvus.DataType = MagicMock()
        mock_pymilvus.utility = mock_utility

        for key in list(sys.modules.keys()):
            if "pymilvus" in key:
                self._saved[key] = sys.modules.pop(key)

        sys.modules["pymilvus"] = mock_pymilvus
        self.mock_collection = mock_collection
        self.mock_connections = mock_connections
        self.mock_utility = mock_utility
        return self

    def __exit__(self, *args):
        sys.modules.pop("pymilvus", None)
        sys.modules.update(self._saved)


class TestMilvusVectorStore:
    def test_import_error_when_pymilvus_missing(self):
        saved = sys.modules.pop("pymilvus", None)
        # Make import fail
        sys.modules["pymilvus"] = None  # type: ignore[assignment]
        try:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            with pytest.raises(ImportError, match="pymilvus"):
                MilvusVectorStore()
        finally:
            if saved is None:
                sys.modules.pop("pymilvus", None)
            else:
                sys.modules["pymilvus"] = saved

    def test_constructor_without_token(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(collection_name="test", uri="http://localhost:19530")
            m.mock_connections.connect.assert_called_once()
            assert store.collection_name == "test"

    def test_constructor_with_token_uses_token(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(
                collection_name="col",
                uri="https://cloud.milvus.io",
                token="secret_token",
            )
            call_kwargs = m.mock_connections.connect.call_args
            assert call_kwargs is not None

    def test_constructor_creates_new_collection(self):
        with _MilvusMocks(collection_exists=False) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(collection_name="new_col")
            # When collection doesn't exist, create_index should be called
            m.mock_collection.create_index.assert_called_once()

    def test_constructor_loads_existing_collection(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(collection_name="existing")
            # load() should always be called
            m.mock_collection.load.assert_called_once()

    def test_add_documents_requires_embedding_function(self):
        with _MilvusMocks(collection_exists=True):
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.add_documents([_make_doc("a")])

    def test_add_documents_inserts_to_collection(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(embedding_function=_embed_fn)
            docs = [_make_doc("doc1"), _make_doc("doc2")]
            ids = store.add_documents(docs)
            assert len(ids) == 2
            m.mock_collection.insert.assert_called_once()
            m.mock_collection.flush.assert_called_once()

    def test_similarity_search_requires_embedding_function(self):
        with _MilvusMocks(collection_exists=True):
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.similarity_search("query")

    def test_similarity_search_returns_results(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            # Build fake hit
            hit = MagicMock()
            hit.entity.get.side_effect = lambda key, default=None: {
                "text": "hello",
                "metadata": {"src": "test"},
            }.get(key, default)
            hit.distance = 0.2  # COSINE: score = 1 - 0.2 = 0.8

            m.mock_collection.search.return_value = [[hit]]

            store = MilvusVectorStore(embedding_function=_embed_fn, metric_type="COSINE")
            results = store.similarity_search("test query", k=1)

            assert len(results) == 1
            assert results[0].score == pytest.approx(0.8)

    def test_similarity_search_cosine_score_conversion(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            hit = MagicMock()
            hit.entity.get.side_effect = lambda k, d=None: {"text": "t", "metadata": {}}.get(k, d)
            hit.distance = 0.5  # score = 1 - 0.5 = 0.5

            m.mock_collection.search.return_value = [[hit]]
            store = MilvusVectorStore(embedding_function=_embed_fn, metric_type="COSINE")
            results = store.similarity_search("q")
            assert results[0].score == pytest.approx(0.5)

    def test_similarity_search_non_cosine_no_conversion(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            hit = MagicMock()
            hit.entity.get.side_effect = lambda k, d=None: {"text": "t", "metadata": {}}.get(k, d)
            hit.distance = 0.7

            m.mock_collection.search.return_value = [[hit]]
            store = MilvusVectorStore(embedding_function=_embed_fn, metric_type="L2")
            results = store.similarity_search("q")
            assert results[0].score == pytest.approx(0.7)  # no conversion for L2

    def test_get_all_vectors_and_docs_empty_on_exception(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            m.mock_collection.query.side_effect = RuntimeError("query failed")
            store = MilvusVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert vectors == []
            assert docs == []

    def test_get_all_vectors_and_docs_success(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            m.mock_collection.query.return_value = [
                {"embedding": [0.1, 0.2], "text": "t1", "metadata": {}},
                {"embedding": [0.3, 0.4], "text": "t2", "metadata": {"k": "v"}},
            ]
            store = MilvusVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert len(vectors) == 2
            assert len(docs) == 2

    async def test_asimilarity_search_by_vector(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            hit = MagicMock()
            hit.entity.get.side_effect = lambda k, d=None: {"text": "v", "metadata": {}}.get(k, d)
            hit.distance = 0.1
            m.mock_collection.search.return_value = [[hit]]

            store = MilvusVectorStore(embedding_function=_embed_fn, metric_type="COSINE")
            results = await store.asimilarity_search_by_vector([0.1, 0.2], k=1)
            assert len(results) == 1

    def test_delete_builds_expr_and_calls_delete(self):
        with _MilvusMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.milvus import MilvusVectorStore

            store = MilvusVectorStore(embedding_function=_embed_fn)
            result = store.delete(["id1", "id2"])
            assert result is True
            m.mock_collection.delete.assert_called_once()
            m.mock_collection.flush.assert_called()


# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------


class _WeaviateMocks:
    def __init__(self, class_exists: bool = True):
        self._class_exists = class_exists
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        mock_weaviate = ModuleType("weaviate")
        mock_client = MagicMock()
        mock_client.schema.exists.return_value = self._class_exists

        # query chain: .get().with_near_vector().with_limit().with_additional().do()
        query_chain = MagicMock()
        query_chain.do.return_value = {"data": {"Get": {"LlmkitDocument": []}}}
        mock_client.query.get.return_value.with_near_vector.return_value.with_limit.return_value.with_additional.return_value = query_chain

        mock_weaviate.Client = MagicMock(return_value=mock_client)
        mock_weaviate.AuthApiKey = MagicMock()

        for key in list(sys.modules.keys()):
            if "weaviate" in key:
                self._saved[key] = sys.modules.pop(key)

        sys.modules["weaviate"] = mock_weaviate
        self.mock_client = mock_client
        self.mock_weaviate = mock_weaviate
        self.query_chain = query_chain
        return self

    def __exit__(self, *args):
        sys.modules.pop("weaviate", None)
        sys.modules.update(self._saved)


class TestWeaviateVectorStore:
    def test_import_error_when_weaviate_missing(self):
        saved = sys.modules.pop("weaviate", None)
        sys.modules["weaviate"] = None  # type: ignore[assignment]
        try:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            with pytest.raises(ImportError, match="Weaviate"):
                WeaviateVectorStore()
        finally:
            if saved is None:
                sys.modules.pop("weaviate", None)
            else:
                sys.modules["weaviate"] = saved

    def test_constructor_without_api_key(self):
        with _WeaviateMocks(class_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            store = WeaviateVectorStore(class_name="MyClass", url="http://localhost:8080")
            assert store.class_name == "MyClass"
            # schema.create_class not called since class exists
            m.mock_client.schema.create_class.assert_not_called()

    def test_constructor_with_api_key(self):
        with _WeaviateMocks(class_exists=True) as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            store = WeaviateVectorStore(api_key="secret_key")
            # AuthApiKey should have been used
            m.mock_weaviate.AuthApiKey.assert_called_once_with(api_key="secret_key")

    def test_constructor_creates_schema_if_not_exists(self):
        with _WeaviateMocks(class_exists=False) as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            store = WeaviateVectorStore(class_name="NewClass")
            m.mock_client.schema.create_class.assert_called_once()

    def test_add_documents_requires_embedding_function(self):
        with _WeaviateMocks():
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            store = WeaviateVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.add_documents([_make_doc("a")])

    def test_add_documents_inserts_via_batch(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            mock_batch = MagicMock()
            mock_batch.__enter__ = MagicMock(return_value=mock_batch)
            mock_batch.__exit__ = MagicMock(return_value=False)
            mock_batch.add_data_object.return_value = "uuid-123"
            m.mock_client.batch = mock_batch

            store = WeaviateVectorStore(embedding_function=_embed_fn)
            ids = store.add_documents([_make_doc("hello"), _make_doc("world")])
            assert len(ids) == 2
            assert mock_batch.add_data_object.call_count == 2

    def test_similarity_search_requires_embedding_function(self):
        with _WeaviateMocks():
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            store = WeaviateVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.similarity_search("query")

    def test_similarity_search_returns_empty_when_no_results(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            m.query_chain.do.return_value = {"data": {"Get": {"LlmkitDocument": []}}}
            store = WeaviateVectorStore(embedding_function=_embed_fn)
            results = store.similarity_search("q")
            assert results == []

    def test_similarity_search_returns_results(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            m.query_chain.do.return_value = {
                "data": {
                    "Get": {
                        "LlmkitDocument": [
                            {
                                "text": "hello",
                                "metadata": {},
                                "_additional": {"distance": 0.0},
                            }
                        ]
                    }
                }
            }
            store = WeaviateVectorStore(embedding_function=_embed_fn)
            results = store.similarity_search("q")
            assert len(results) == 1
            assert results[0].score == pytest.approx(1.0)  # 1/(1+0) = 1.0

    def test_get_all_vectors_and_docs_empty_on_exception(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            m.mock_client.query.get.side_effect = RuntimeError("fail")
            store = WeaviateVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert vectors == []

    def test_get_all_vectors_and_docs_returns_vectors(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            all_chain = MagicMock()
            all_chain.do.return_value = {
                "data": {
                    "Get": {
                        "LlmkitDocument": [
                            {"text": "t", "metadata": {}, "_additional": {"vector": [0.1, 0.2]}},
                        ]
                    }
                }
            }
            m.mock_client.query.get.return_value.with_additional.return_value.with_limit.return_value = all_chain
            store = WeaviateVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert len(vectors) == 1

    async def test_asimilarity_search_by_vector(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            m.query_chain.do.return_value = {
                "data": {
                    "Get": {
                        "LlmkitDocument": [
                            {
                                "text": "t",
                                "metadata": {},
                                "_additional": {"certainty": 0.9, "distance": 0.1},
                            }
                        ]
                    }
                }
            }
            store = WeaviateVectorStore(embedding_function=_embed_fn)
            results = await store.asimilarity_search_by_vector([0.1, 0.2])
            assert len(results) == 1
            assert results[0].score == pytest.approx(0.9)

    def test_delete_calls_data_object_delete(self):
        with _WeaviateMocks() as m:
            from beanllm.domain.vector_stores.cloud.weaviate import WeaviateVectorStore

            store = WeaviateVectorStore(embedding_function=_embed_fn)
            result = store.delete(["uuid-1", "uuid-2"])
            assert result is True
            assert m.mock_client.data_object.delete.call_count == 2


# ---------------------------------------------------------------------------
# LanceDB
# ---------------------------------------------------------------------------


class _LanceDBMocks:
    def __init__(self, table_exists: bool = True):
        self._table_exists = table_exists
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        mock_lancedb = ModuleType("lancedb")
        mock_db = MagicMock()
        mock_table = MagicMock()

        if self._table_exists:
            mock_db.open_table.return_value = mock_table
        else:
            mock_db.open_table.side_effect = Exception("Table not found")

        mock_db.create_table.return_value = mock_table
        mock_lancedb.connect = MagicMock(return_value=mock_db)

        for key in list(sys.modules.keys()):
            if "lancedb" in key:
                self._saved[key] = sys.modules.pop(key)

        sys.modules["lancedb"] = mock_lancedb
        self.mock_db = mock_db
        self.mock_table = mock_table
        return self

    def __exit__(self, *args):
        sys.modules.pop("lancedb", None)
        sys.modules.update(self._saved)


class TestLanceDBVectorStore:
    def test_import_error_when_lancedb_missing(self):
        saved = sys.modules.pop("lancedb", None)
        sys.modules["lancedb"] = None  # type: ignore[assignment]
        try:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            with pytest.raises(ImportError, match="lancedb"):
                LanceDBVectorStore()
        finally:
            if saved is None:
                sys.modules.pop("lancedb", None)
            else:
                sys.modules["lancedb"] = saved

    def test_constructor_opens_existing_table(self):
        with _LanceDBMocks(table_exists=True) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(table_name="docs", uri="./data")
            assert store.table is m.mock_table
            assert store.table_name == "docs"

    def test_constructor_sets_table_none_when_not_found(self):
        with _LanceDBMocks(table_exists=False) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(table_name="new_table")
            assert store.table is None

    def test_add_documents_requires_embedding_function(self):
        with _LanceDBMocks():
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.add_documents([_make_doc("x")])

    def test_add_documents_creates_table_when_none(self):
        with _LanceDBMocks(table_exists=False) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            assert store.table is None
            ids = store.add_documents([_make_doc("hello")])
            # create_table should have been called
            m.mock_db.create_table.assert_called_once()
            assert len(ids) == 1

    def test_add_documents_adds_to_existing_table(self):
        with _LanceDBMocks(table_exists=True) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            ids = store.add_documents([_make_doc("a"), _make_doc("b")])
            m.mock_table.add.assert_called_once()
            assert len(ids) == 2

    def test_similarity_search_requires_embedding_function(self):
        with _LanceDBMocks():
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.similarity_search("q")

    def test_similarity_search_returns_empty_when_table_none(self):
        with _LanceDBMocks(table_exists=False):
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            results = store.similarity_search("q")
            assert results == []

    def test_similarity_search_returns_results(self):
        with _LanceDBMocks(table_exists=True) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            m.mock_table.search.return_value.limit.return_value.to_list.return_value = [
                {"text": "hello", "metadata": {}, "_distance": 0.2},
                {"text": "world", "metadata": {"k": "v"}, "_distance": 0.5},
            ]
            store = LanceDBVectorStore(embedding_function=_embed_fn)
            results = store.similarity_search("q", k=2)
            assert len(results) == 2
            assert results[0].score == pytest.approx(0.8)  # 1 - 0.2
            assert results[1].score == pytest.approx(0.5)  # 1 - 0.5

    def test_get_all_vectors_and_docs_none_table(self):
        with _LanceDBMocks(table_exists=False):
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert vectors == []

    def test_get_all_vectors_and_docs_exception(self):
        with _LanceDBMocks(table_exists=True) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            m.mock_table.to_pandas.side_effect = RuntimeError("fail")
            store = LanceDBVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert vectors == []

    async def test_asimilarity_search_by_vector_none_table(self):
        with _LanceDBMocks(table_exists=False):
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            results = await store.asimilarity_search_by_vector([0.1, 0.2])
            assert results == []

    async def test_asimilarity_search_by_vector_returns_results(self):
        with _LanceDBMocks(table_exists=True) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            m.mock_table.search.return_value.limit.return_value.to_list.return_value = [
                {"text": "t", "metadata": {}, "_distance": 0.3},
            ]
            store = LanceDBVectorStore(embedding_function=_embed_fn)
            results = await store.asimilarity_search_by_vector([0.1, 0.2])
            assert len(results) == 1
            assert results[0].score == pytest.approx(0.7)

    def test_delete_returns_false_when_table_none(self):
        with _LanceDBMocks(table_exists=False):
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            result = store.delete(["id1"])
            assert result is False

    def test_delete_calls_delete_on_table(self):
        with _LanceDBMocks(table_exists=True) as m:
            from beanllm.domain.vector_stores.local.lancedb import LanceDBVectorStore

            store = LanceDBVectorStore(embedding_function=_embed_fn)
            result = store.delete(["id1", "id2"])
            assert result is True
            assert m.mock_table.delete.call_count == 2


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


class _QdrantMocks:
    def __init__(self, collection_exists: bool = True):
        self._collection_exists = collection_exists
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        mock_qdrant_client = ModuleType("qdrant_client")
        mock_models = ModuleType("qdrant_client.models")

        mock_client_inst = MagicMock()
        if not self._collection_exists:
            mock_client_inst.get_collection.side_effect = Exception("Collection not found")
        else:
            mock_client_inst.get_collection.return_value = MagicMock()

        mock_qdrant_client.QdrantClient = MagicMock(return_value=mock_client_inst)

        mock_point_struct = MagicMock()
        mock_models.PointStruct = mock_point_struct
        mock_models.Distance = MagicMock()
        mock_models.VectorParams = MagicMock()

        for key in list(sys.modules.keys()):
            if "qdrant_client" in key:
                self._saved[key] = sys.modules.pop(key)

        sys.modules["qdrant_client"] = mock_qdrant_client
        sys.modules["qdrant_client.models"] = mock_models
        self.mock_client = mock_client_inst
        self.mock_qdrant = mock_qdrant_client
        self.mock_point_struct = mock_point_struct
        return self

    def __exit__(self, *args):
        for key in list(sys.modules.keys()):
            if "qdrant_client" in key:
                sys.modules.pop(key)
        sys.modules.update(self._saved)


class TestQdrantVectorStore:
    def test_import_error_when_qdrant_missing(self):
        saved_keys = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if "qdrant" in k}
        sys.modules["qdrant_client"] = None  # type: ignore[assignment]
        try:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            with pytest.raises(ImportError, match="Qdrant"):
                QdrantVectorStore()
        finally:
            sys.modules.pop("qdrant_client", None)
            sys.modules.update(saved_keys)

    def test_constructor_without_api_key(self):
        with _QdrantMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(collection_name="test", url="http://localhost:6333")
            assert store.collection_name == "test"
            assert store.dimension == 1536

    def test_constructor_with_api_key(self):
        with _QdrantMocks(collection_exists=True) as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(api_key="mykey")
            call_kwargs = m.mock_qdrant.QdrantClient.call_args
            assert "api_key" in str(call_kwargs)

    def test_constructor_creates_collection_when_missing(self):
        with _QdrantMocks(collection_exists=False) as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(collection_name="new_col")
            m.mock_client.create_collection.assert_called_once()

    def test_add_documents_requires_embedding_function(self):
        with _QdrantMocks():
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.add_documents([_make_doc("x")])

    def test_add_documents_upserts_points(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(embedding_function=_embed_fn)
            ids = store.add_documents([_make_doc("a"), _make_doc("b")])
            m.mock_client.upsert.assert_called_once()
            assert len(ids) == 2

    def test_similarity_search_requires_embedding_function(self):
        with _QdrantMocks():
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(embedding_function=None)
            with pytest.raises(ValueError, match="Embedding function"):
                store.similarity_search("q")

    def test_similarity_search_returns_results(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            hit = MagicMock()
            hit.payload = {"text": "result", "src": "test"}
            hit.score = 0.95

            m.mock_client.search.return_value = [hit]
            store = QdrantVectorStore(embedding_function=_embed_fn)
            results = store.similarity_search("q", k=1)

            assert len(results) == 1
            assert results[0].score == pytest.approx(0.95)

    def test_similarity_search_pops_text_from_payload(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            hit = MagicMock()
            hit.payload = {"text": "my text", "extra": "val"}
            hit.score = 0.8

            m.mock_client.search.return_value = [hit]
            store = QdrantVectorStore(embedding_function=_embed_fn)
            results = store.similarity_search("q")
            assert results[0].document.content == "my text"

    def test_get_all_vectors_and_docs_exception(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            m.mock_client.scroll.side_effect = RuntimeError("fail")
            store = QdrantVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert vectors == []

    def test_get_all_vectors_and_docs_returns_data(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            point = MagicMock()
            point.vector = [0.1, 0.2]
            point.payload = {"text": "t", "k": "v"}
            m.mock_client.scroll.return_value = ([point], None)

            store = QdrantVectorStore(embedding_function=_embed_fn)
            vectors, docs = store._get_all_vectors_and_docs()
            assert len(vectors) == 1
            assert docs[0].content == "t"

    async def test_asimilarity_search_by_vector_returns_results(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            hit = MagicMock()
            hit.payload = {"text": "vec result", "extra": "x"}
            hit.score = 0.77
            m.mock_client.search.return_value = [hit]

            store = QdrantVectorStore(embedding_function=_embed_fn)
            results = await store.asimilarity_search_by_vector([0.1, 0.2])
            assert len(results) == 1
            assert results[0].score == pytest.approx(0.77)

    def test_delete_calls_client_delete(self):
        with _QdrantMocks() as m:
            from beanllm.domain.vector_stores.local.qdrant import QdrantVectorStore

            store = QdrantVectorStore(embedding_function=_embed_fn)
            result = store.delete(["id1", "id2"])
            assert result is True
            m.mock_client.delete.assert_called_once()
