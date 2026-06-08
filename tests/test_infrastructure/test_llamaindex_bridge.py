"""
Tests for infrastructure/integrations/llamaindex/bridge.py

Mocks llama_index to test LlamaIndexBridge without the actual library.
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Build llama_index mocks
# ---------------------------------------------------------------------------


def _build_llamaindex_mock():
    """Build a minimal llama_index mock hierarchy."""
    li = MagicMock()

    # llama_index.core.Document
    class FakeLlamaDocument:
        def __init__(self, text="", metadata=None, doc_id=None):
            self.text = text
            self.metadata = metadata or {}
            self.doc_id = doc_id

    li.core = MagicMock()
    li.core.Document = FakeLlamaDocument

    # llama_index.core.embeddings.BaseEmbedding
    class FakeBaseEmbedding:
        def __init__(self):
            self.model_name = "fake"

    li.core.embeddings = MagicMock()
    li.core.embeddings.BaseEmbedding = FakeBaseEmbedding

    # llama_index.core.llms
    class FakeCompletionResponse:
        def __init__(self, text=""):
            self.text = text

    class FakeCustomLLM:
        def __init__(self):
            pass

    def fake_llm_completion_callback():
        def decorator(fn):
            return fn

        return decorator

    li.core.llms = MagicMock()
    li.core.llms.CompletionResponse = FakeCompletionResponse
    li.core.llms.CustomLLM = FakeCustomLLM
    li.core.llms.callbacks = MagicMock()
    li.core.llms.callbacks.llm_completion_callback = fake_llm_completion_callback

    return li, FakeLlamaDocument, FakeBaseEmbedding, FakeCompletionResponse, FakeCustomLLM


_LI_MOCK, _FakeLlamaDoc, _FakeBaseEmb, _FakeCompResp, _FakeCustomLLM = _build_llamaindex_mock()


@pytest.fixture(autouse=True)
def patch_llamaindex():
    sys.modules["llama_index"] = _LI_MOCK
    sys.modules["llama_index.core"] = _LI_MOCK.core
    sys.modules["llama_index.core.embeddings"] = _LI_MOCK.core.embeddings
    sys.modules["llama_index.core.llms"] = _LI_MOCK.core.llms
    sys.modules["llama_index.core.llms.callbacks"] = _LI_MOCK.core.llms.callbacks
    yield


from beanllm.infrastructure.integrations.llamaindex.bridge import LlamaIndexBridge

# ---------------------------------------------------------------------------
# Helper: fake beanLLM document
# ---------------------------------------------------------------------------


class FakeBeanDoc:
    def __init__(self, content="", metadata=None):
        self.content = content
        self.metadata = metadata or {}


# ---------------------------------------------------------------------------
# LlamaIndexBridge.convert_documents tests
# ---------------------------------------------------------------------------


class TestConvertDocuments:
    def test_returns_list_of_same_length(self):
        bean_docs = [FakeBeanDoc("doc1"), FakeBeanDoc("doc2"), FakeBeanDoc("doc3")]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        assert len(result) == 3

    def test_converts_content_to_text(self):
        bean_docs = [FakeBeanDoc(content="Hello world")]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        assert result[0].text == "Hello world"

    def test_preserves_metadata(self):
        bean_docs = [FakeBeanDoc(content="test", metadata={"source": "doc.pdf", "page": 1})]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        assert result[0].metadata == {"source": "doc.pdf", "page": 1}

    def test_uses_id_from_metadata(self):
        bean_docs = [FakeBeanDoc(content="text", metadata={"id": "doc-123"})]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        assert result[0].doc_id == "doc-123"

    def test_doc_id_none_when_no_id_in_metadata(self):
        bean_docs = [FakeBeanDoc(content="text", metadata={"source": "x"})]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        assert result[0].doc_id is None

    def test_empty_metadata_is_empty_dict(self):
        bean_docs = [FakeBeanDoc(content="text")]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        assert result[0].metadata == {}

    def test_empty_list_returns_empty_list(self):
        result = LlamaIndexBridge.convert_documents([])
        assert result == []

    def test_multiple_documents_all_converted(self):
        bean_docs = [FakeBeanDoc(content=f"doc {i}", metadata={"index": i}) for i in range(5)]
        result = LlamaIndexBridge.convert_documents(bean_docs)
        for i, doc in enumerate(result):
            assert doc.text == f"doc {i}"

    def test_raises_import_error_when_llama_index_missing(self):
        orig = sys.modules.pop("llama_index.core", None)
        try:
            # Simulate ImportError inside the method
            import importlib

            with patch_missing_llamaindex():
                with pytest.raises(ImportError, match="llama-index"):
                    # Force the method to try importing again
                    # We temporarily break the import
                    bean_docs = [FakeBeanDoc(content="x")]
                    _test_convert_documents_import_error(bean_docs)
        finally:
            if orig is not None:
                sys.modules["llama_index.core"] = orig


def _test_convert_documents_import_error(docs):
    """Helper to test import error path by mocking inside the function."""
    import importlib

    # This tests the ImportError branch in convert_documents
    # We patch the import inside the method
    from unittest.mock import patch

    with patch.dict(
        "sys.modules",
        {"llama_index.core": None},  # type: ignore
    ):
        importlib.invalidate_caches()
        LlamaIndexBridge.convert_documents(docs)


from contextlib import contextmanager


@contextmanager
def patch_missing_llamaindex():
    """Context manager that makes llama_index.core import fail."""
    from unittest.mock import patch

    with patch.dict("sys.modules", {"llama_index.core": None}):  # type: ignore
        yield


# ---------------------------------------------------------------------------
# LlamaIndexBridge.convert_to_bean_documents tests
# ---------------------------------------------------------------------------


class TestConvertToBeanDocuments:
    def test_converts_llama_docs_to_bean_format(self):
        llama_docs = [_FakeLlamaDoc(text="hello", metadata={"source": "test.txt"})]
        result = LlamaIndexBridge.convert_to_bean_documents(llama_docs)
        assert len(result) == 1

    def test_preserves_text_as_content(self):
        llama_docs = [_FakeLlamaDoc(text="This is content")]
        result = LlamaIndexBridge.convert_to_bean_documents(llama_docs)
        assert result[0].content == "This is content"

    def test_preserves_metadata(self):
        llama_docs = [_FakeLlamaDoc(text="t", metadata={"source": "doc.txt", "page": 5})]
        result = LlamaIndexBridge.convert_to_bean_documents(llama_docs)
        assert result[0].metadata["page"] == 5

    def test_adds_default_source_when_missing(self):
        llama_docs = [_FakeLlamaDoc(text="t", metadata={})]
        result = LlamaIndexBridge.convert_to_bean_documents(llama_docs)
        assert result[0].metadata.get("source") == "llamaindex"

    def test_keeps_existing_source(self):
        llama_docs = [_FakeLlamaDoc(text="t", metadata={"source": "my_source.pdf"})]
        result = LlamaIndexBridge.convert_to_bean_documents(llama_docs)
        assert result[0].metadata["source"] == "my_source.pdf"

    def test_empty_list_returns_empty_list(self):
        result = LlamaIndexBridge.convert_to_bean_documents([])
        assert result == []

    def test_multiple_docs_converted(self):
        llama_docs = [_FakeLlamaDoc(text=f"text {i}", metadata={"page": i}) for i in range(3)]
        result = LlamaIndexBridge.convert_to_bean_documents(llama_docs)
        assert len(result) == 3
        for i, doc in enumerate(result):
            assert doc.content == f"text {i}"


# ---------------------------------------------------------------------------
# LlamaIndexBridge.wrap_embeddings tests
# ---------------------------------------------------------------------------


class TestWrapEmbeddings:
    def _make_embedding_fn(self, value=None):
        if value is None:
            value = [0.1, 0.2, 0.3]

        def embed_fn(text: str):
            return value

        return embed_fn

    def test_returns_an_object(self):
        embed_fn = self._make_embedding_fn()
        result = LlamaIndexBridge.wrap_embeddings(embed_fn)
        assert result is not None

    def test_wrapped_object_has_model_name(self):
        embed_fn = self._make_embedding_fn()
        result = LlamaIndexBridge.wrap_embeddings(embed_fn, model_name="test-model")
        assert result.model_name == "test-model"

    def test_default_model_name(self):
        embed_fn = self._make_embedding_fn()
        result = LlamaIndexBridge.wrap_embeddings(embed_fn)
        assert result.model_name == "beanllm-custom"

    def test_query_embedding_calls_function(self):
        expected = [1.0, 2.0, 3.0]
        embed_fn = self._make_embedding_fn(expected)
        wrapper = LlamaIndexBridge.wrap_embeddings(embed_fn)
        result = wrapper._get_query_embedding("test query")
        assert result == expected

    def test_text_embedding_calls_function(self):
        expected = [0.5, 0.6]
        embed_fn = self._make_embedding_fn(expected)
        wrapper = LlamaIndexBridge.wrap_embeddings(embed_fn)
        result = wrapper._get_text_embedding("some text")
        assert result == expected

    async def test_async_query_embedding(self):
        expected = [0.1, 0.2]
        embed_fn = self._make_embedding_fn(expected)
        wrapper = LlamaIndexBridge.wrap_embeddings(embed_fn)
        result = await wrapper._aget_query_embedding("async query")
        assert result == expected

    async def test_async_text_embedding(self):
        expected = [0.3, 0.4]
        embed_fn = self._make_embedding_fn(expected)
        wrapper = LlamaIndexBridge.wrap_embeddings(embed_fn)
        result = await wrapper._aget_text_embedding("async text")
        assert result == expected

    def test_wrapped_inherits_from_base_embedding(self):
        embed_fn = self._make_embedding_fn()
        wrapper = LlamaIndexBridge.wrap_embeddings(embed_fn)
        assert isinstance(wrapper, _FakeBaseEmb)


# ---------------------------------------------------------------------------
# LlamaIndexBridge.wrap_llm tests
# ---------------------------------------------------------------------------


class TestWrapLLM:
    def _make_mock_client(self):
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "I am a response"
        client.chat = MagicMock(return_value=mock_response)
        return client

    def test_returns_an_object(self):
        client = self._make_mock_client()
        result = LlamaIndexBridge.wrap_llm(client)
        assert result is not None

    def test_wrapped_llm_has_model_name(self):
        client = self._make_mock_client()
        result = LlamaIndexBridge.wrap_llm(client, model_name="gpt-4o-mini")
        assert result.model_name == "gpt-4o-mini"

    def test_default_model_name(self):
        client = self._make_mock_client()
        result = LlamaIndexBridge.wrap_llm(client)
        assert result.model_name == "beanllm-custom"

    def test_metadata_contains_model_name(self):
        client = self._make_mock_client()
        wrapper = LlamaIndexBridge.wrap_llm(client, model_name="gpt-4o")
        meta = wrapper.metadata
        assert meta["model_name"] == "gpt-4o"
        assert meta["is_chat_model"] is True

    def test_complete_calls_client_chat(self):
        client = self._make_mock_client()
        wrapper = LlamaIndexBridge.wrap_llm(client, model_name="gpt-4o-mini")
        response = wrapper.complete("What is 2+2?")
        client.chat.assert_called_once()

    def test_complete_returns_completion_response(self):
        client = self._make_mock_client()
        wrapper = LlamaIndexBridge.wrap_llm(client, model_name="gpt-4o-mini")
        response = wrapper.complete("Hello")
        assert hasattr(response, "text")
        assert response.text == "I am a response"

    def test_stream_complete_raises_not_implemented(self):
        client = self._make_mock_client()
        wrapper = LlamaIndexBridge.wrap_llm(client)
        with pytest.raises(NotImplementedError, match="Streaming not supported"):
            wrapper.stream_complete("prompt")

    def test_inherits_from_custom_llm(self):
        client = self._make_mock_client()
        wrapper = LlamaIndexBridge.wrap_llm(client)
        assert isinstance(wrapper, _FakeCustomLLM)

    def test_complete_passes_prompt_as_message(self):
        client = self._make_mock_client()
        wrapper = LlamaIndexBridge.wrap_llm(client)
        wrapper.complete("Tell me a joke")
        call_args = client.chat.call_args
        messages = call_args[1]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Tell me a joke"
