"""
Tests for local embedding implementations.

Strategy: inject mock sentence_transformers / transformers into sys.modules
before importing the embedding classes, so GPU/model loading is never triggered.
Sophisticated tests: batch processing, cosine similarity, dimension validation,
fp16 path, device selection, and error handling.
"""

from __future__ import annotations

import sys
from typing import Any, List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sentence_transformer_mock(dim: int = 384, n_texts: int = None) -> MagicMock:
    """Create a mock SentenceTransformer that returns deterministic embeddings."""
    mock_model = MagicMock()
    mock_model.max_seq_length = 512

    def encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
    ):
        n = len(texts) if hasattr(texts, "__len__") else 1
        vecs = np.random.default_rng(42).random((n, dim)).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
        return vecs

    mock_model.encode.side_effect = encode
    mock_model.half.return_value = mock_model
    mock_model.eval.return_value = mock_model
    return mock_model


def _make_st_module_mock(dim: int = 384) -> MagicMock:
    mock_module = MagicMock()
    mock_model_instance = _make_sentence_transformer_mock(dim)
    mock_module.SentenceTransformer.return_value = mock_model_instance
    return mock_module


def _make_transformers_mock(dim: int = 768) -> MagicMock:
    """Mock transformers (AutoModel + AutoTokenizer) module."""
    mock_module = MagicMock()

    # AutoTokenizer
    mock_tokenizer = MagicMock()

    def tokenize(texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        batch_size = len(texts)
        mock_encoding = MagicMock()
        # input_ids shape: (batch, seq_len)
        import torch as _torch

        # We can't import real torch, but we mock the tensor
        mock_tensor = MagicMock()
        mock_tensor.__len__ = lambda self: batch_size
        mock_encoding.__iter__ = lambda self: iter({"input_ids": mock_tensor})
        mock_encoding.items.return_value = [
            ("input_ids", mock_tensor),
            ("attention_mask", mock_tensor),
        ]
        return mock_encoding

    mock_tokenizer.side_effect = tokenize
    mock_module.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    # AutoModel
    mock_model = MagicMock()
    mock_module.AutoModel.from_pretrained.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    # model output: [[batch, seq_len, hidden_dim]] tensor-like
    mock_output = MagicMock()
    mock_tensor_out = MagicMock()
    mock_tensor_out.__getitem__.return_value = MagicMock()
    mock_output.__getitem__.return_value = mock_tensor_out
    mock_model.return_value = [mock_output]

    return mock_module


# ---------------------------------------------------------------------------
# HuggingFaceEmbedding tests
# ---------------------------------------------------------------------------


class TestHuggingFaceEmbedding:
    @pytest.fixture(autouse=True)
    def inject_st_mock(self):
        """Inject sentence_transformers mock before each test."""
        dim = 384
        mock_st = _make_st_module_mock(dim)
        with patch.dict(
            sys.modules,
            {
                "sentence_transformers": mock_st,
                "torch": MagicMock(),
            },
        ):
            yield mock_st

    def test_embed_sync_returns_list_of_vectors(self) -> None:
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False, use_fp16=False)
        result = emb.embed_sync(["hello", "world", "test"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, list) for v in result)

    def test_embed_sync_vector_dimensions_consistent(self) -> None:
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False)
        result = emb.embed_sync(["a", "b", "c", "d"])
        dims = [len(v) for v in result]
        assert len(set(dims)) == 1  # All same dimension

    def test_embed_sync_normalized_vectors_unit_length(self) -> None:
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False, normalize=True)
        result = emb.embed_sync(["normalize this text"])
        vec = np.array(result[0])
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_embed_sync_same_text_same_vector(self) -> None:
        """Same input → same output (deterministic mock)."""
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False)
        r1 = emb.embed_sync(["consistent text"])
        r2 = emb.embed_sync(["consistent text"])
        assert r1 == r2  # Mock returns same deterministic result

    def test_embed_sync_batch_size_passed_to_encode(self, inject_st_mock) -> None:
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False, batch_size=16)
        texts = [f"doc_{i}" for i in range(5)]
        emb.embed_sync(texts)

        # The mock model's encode was called with batch_size=16
        mock_model = inject_st_mock.SentenceTransformer.return_value
        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args.kwargs
        assert call_kwargs.get("batch_size") == 16

    def test_fp16_path_calls_half_on_model(self, inject_st_mock) -> None:
        """Covers: FP16 path when device is cuda."""
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        # Force device to "cuda"
        with patch(
            "beanllm.domain.embeddings.base.BaseLocalEmbedding._get_device", return_value="cuda"
        ):
            # Also mock torch.cuda to avoid actual GPU check
            mock_torch = sys.modules["torch"]
            mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

            emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=True, use_fp16=True)
            emb.embed_sync(["cuda fp16 test"])

            # half() should have been called on the model
            mock_model = inject_st_mock.SentenceTransformer.return_value
            mock_model.half.assert_called_once()

    def test_max_seq_length_set_from_kwargs(self, inject_st_mock) -> None:
        """Covers: max_seq_length kwarg path (line 133-134)."""
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False, max_seq_length=128)
        emb.embed_sync(["test"])

        mock_model = inject_st_mock.SentenceTransformer.return_value
        assert mock_model.max_seq_length == 128

    def test_model_loaded_only_once(self, inject_st_mock) -> None:
        """Model loading is lazy — SentenceTransformer called only once across multiple embed calls."""
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False)
        emb.embed_sync(["first call"])
        emb.embed_sync(["second call"])
        assert inject_st_mock.SentenceTransformer.call_count == 1

    def test_cosine_similarity_unrelated_texts_below_one(self) -> None:
        """Embedding quality: cosine similarity between random distinct texts < 1.0."""
        from beanllm.domain.embeddings.local.huggingface_embeddings import HuggingFaceEmbedding

        emb = HuggingFaceEmbedding(model="all-MiniLM-L6-v2", use_gpu=False, normalize=True)
        result = emb.embed_sync(["quantum physics research", "cooking pasta recipe"])
        # With the deterministic mock, the two vectors are different
        v1, v2 = np.array(result[0]), np.array(result[1])
        cosine = float(np.dot(v1, v2))
        # They should NOT be identical (different positions in batch)
        assert cosine != pytest.approx(1.0)  # distinct embeddings


# ---------------------------------------------------------------------------
# NVEmbedEmbedding tests
# ---------------------------------------------------------------------------


class TestNVEmbedEmbedding:
    @pytest.fixture(autouse=True)
    def inject_st_mock(self):
        mock_st = _make_st_module_mock(dim=4096)
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            yield mock_st

    def test_embed_sync_with_prefix(self) -> None:
        from beanllm.domain.embeddings.local.nvembed_embeddings import NVEmbedEmbedding

        emb = NVEmbedEmbedding(model="nvidia/NV-Embed-v2", prefix="passage", use_gpu=False)
        result = emb.embed_sync(["some document text"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_prefix_prepended_to_texts(self, inject_st_mock) -> None:
        """Covers: _prepare_texts prefix path (line 132)."""
        from beanllm.domain.embeddings.local.nvembed_embeddings import NVEmbedEmbedding

        emb = NVEmbedEmbedding(model="nvidia/NV-Embed-v2", prefix="query", use_gpu=False)
        emb.embed_sync(["find relevant docs"])

        # Verify encode was called with "query: find relevant docs"
        mock_model = inject_st_mock.SentenceTransformer.return_value
        call_args = mock_model.encode.call_args.args
        assert call_args[0][0].startswith("query:")

    def test_instruction_mode_prepends_instruction(self, inject_st_mock) -> None:
        """Covers: _prepare_texts instruction path (lines 128-129)."""
        from beanllm.domain.embeddings.local.nvembed_embeddings import NVEmbedEmbedding

        emb = NVEmbedEmbedding(
            model="nvidia/NV-Embed-v2",
            instruction="Given a web search query, retrieve relevant passages",
            use_gpu=False,
        )
        emb.embed_sync(["what is machine learning"])

        mock_model = inject_st_mock.SentenceTransformer.return_value
        call_args = mock_model.encode.call_args.args
        text = call_args[0][0]
        assert text.startswith("Instruct:")
        assert "Query:" in text

    def test_cpu_warning_logged(self, inject_st_mock) -> None:
        """Covers: CPU warning (line 106) — just checks no exception."""
        from beanllm.domain.embeddings.local.nvembed_embeddings import NVEmbedEmbedding

        with patch(
            "beanllm.domain.embeddings.base.BaseLocalEmbedding._get_device", return_value="cpu"
        ):
            emb = NVEmbedEmbedding(model="nvidia/NV-Embed-v2", use_gpu=False)
            result = emb.embed_sync(["cpu test"])
            assert isinstance(result, list)

    def test_vector_dimensions_consistent(self) -> None:
        from beanllm.domain.embeddings.local.nvembed_embeddings import NVEmbedEmbedding

        emb = NVEmbedEmbedding(model="nvidia/NV-Embed-v2", use_gpu=False)
        result = emb.embed_sync(["doc1", "doc2", "doc3"])
        dims = [len(v) for v in result]
        assert len(set(dims)) == 1


# ---------------------------------------------------------------------------
# BatchProcessor.process_batch tests (task_processor.py lines 233-279)
# ---------------------------------------------------------------------------


class TestBatchProcessorBatch:
    def _make_batch_processor(self):
        """Create a BatchProcessor with mocked dependencies."""
        from beanllm.infrastructure.distributed.task_processor import BatchProcessor

        bp = BatchProcessor(task_type="test.tasks", max_concurrent=5)
        bp.concurrency_controller = None  # No concurrency controller

        # Mock the TaskProcessor internals
        bp.task_processor.batch_enqueue = MagicMock()
        bp.task_processor.error_handler = MagicMock()
        bp.task_processor.error_handler.handle_error = MagicMock(return_value=None)
        return bp

    async def test_process_batch_async_handler(self) -> None:
        """Covers: async handler path (line 247-248, 253-254)."""
        bp = self._make_batch_processor()

        task_ids = ["tid1", "tid2", "tid3"]
        tasks_data = [{"n": 1}, {"n": 2}, {"n": 3}]

        async def fake_batch_enqueue(name, data, priority=0):
            return task_ids

        bp.task_processor.batch_enqueue = fake_batch_enqueue

        async def handler(data):
            return data["n"] * 2

        results = await bp.process_batch("test.task", tasks_data, handler=handler)
        assert results == [2, 4, 6]

    async def test_process_batch_sync_handler(self) -> None:
        """Covers: sync handler via run_in_executor (lines 250-251, 255-257)."""
        bp = self._make_batch_processor()

        async def fake_batch_enqueue(name, data, priority=0):
            return [f"tid{i}" for i in range(len(data))]

        bp.task_processor.batch_enqueue = fake_batch_enqueue

        def sync_handler(data):
            return data["val"] + 10

        tasks_data = [{"val": 1}, {"val": 2}, {"val": 3}]
        results = await bp.process_batch("test.task", tasks_data, handler=sync_handler)
        assert results == [11, 12, 13]

    async def test_process_batch_error_is_captured(self) -> None:
        """Covers: exception path (lines 268-275)."""
        bp = self._make_batch_processor()
        bp.task_processor.error_handler.handle_error = MagicMock()

        async def fake_batch_enqueue(name, data, priority=0):
            return [f"tid{i}" for i in range(len(data))]

        bp.task_processor.batch_enqueue = fake_batch_enqueue

        # Make handle_error async
        async def async_handle_error(**kwargs):
            pass

        bp.task_processor.error_handler.handle_error = async_handle_error

        async def failing_handler(data):
            raise ValueError(f"handler failed for {data}")

        tasks_data = [{"x": 1}, {"x": 2}]
        results = await bp.process_batch("test.task", tasks_data, handler=failing_handler)
        # Both errors → captured as {"error": ...}
        assert all("error" in r for r in results)

    async def test_process_batch_empty_input(self) -> None:
        """Empty tasks_data → empty result."""
        bp = self._make_batch_processor()

        async def fake_batch_enqueue(name, data, priority=0):
            return []

        bp.task_processor.batch_enqueue = fake_batch_enqueue

        async def handler(data):
            return data

        results = await bp.process_batch("test.task", [], handler=handler)
        assert results == []

    async def test_process_batch_with_concurrency_controller(self) -> None:
        """Covers: concurrency_controller path (lines 242-251)."""
        bp = self._make_batch_processor()

        # Build an async context manager that can be awaited
        class AsyncCM:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_cc = MagicMock()

        async def with_cc(key, max_concurrent):
            return AsyncCM()

        mock_cc.with_concurrency_control = with_cc
        bp.concurrency_controller = mock_cc

        async def fake_batch_enqueue(name, data, priority=0):
            return [f"tid{i}" for i in range(len(data))]

        bp.task_processor.batch_enqueue = fake_batch_enqueue
        bp.task_processor.task_type = "test"

        async def handler(data):
            return data

        tasks_data = [{"a": 1}]
        results = await bp.process_batch("test.task", tasks_data, handler=handler)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# BatchProcessor.process_items tests (lines 281-340)
# ---------------------------------------------------------------------------


class TestBatchProcessorItems:
    def _make_batch_processor(self):
        from beanllm.infrastructure.distributed.task_processor import BatchProcessor

        bp = BatchProcessor(task_type="test.tasks", max_concurrent=5)
        bp.concurrency_controller = None

        async def async_handle_error(**kwargs):
            pass

        bp.task_processor.error_handler.handle_error = async_handle_error
        bp.task_processor.task_type = "test"
        return bp

    async def test_process_items_async_handler(self) -> None:
        bp = self._make_batch_processor()

        async def handler(item):
            return item * 2

        results = await bp.process_items([1, 2, 3], handler=handler)
        assert results == [2, 4, 6]

    async def test_process_items_sync_handler(self) -> None:
        bp = self._make_batch_processor()

        def sync_handler(item):
            return str(item)

        results = await bp.process_items([10, 20], handler=sync_handler)
        assert results == ["10", "20"]

    async def test_process_items_exception_becomes_none(self) -> None:
        """Covers: error in process_items → None (line 336)."""
        bp = self._make_batch_processor()

        async def failing_handler(item):
            raise RuntimeError("item failed")

        results = await bp.process_items([1, 2], handler=failing_handler)
        assert results == [None, None]

    async def test_process_items_with_max_concurrent_override(self) -> None:
        """Covers: max_concurrent parameter (line 298)."""
        bp = self._make_batch_processor()

        async def handler(item):
            return item

        results = await bp.process_items([1, 2, 3], handler=handler, max_concurrent=2)
        assert results == [1, 2, 3]

    async def test_process_items_empty_list(self) -> None:
        bp = self._make_batch_processor()

        async def handler(item):
            return item

        results = await bp.process_items([], handler=handler)
        assert results == []

    async def test_process_items_with_concurrency_controller(self) -> None:
        """Covers: concurrency_controller path in process_items (lines 306-315)."""
        bp = self._make_batch_processor()

        class AsyncCM:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        mock_cc = MagicMock()

        async def with_cc(key, max_concurrent):
            return AsyncCM()

        mock_cc.with_concurrency_control = with_cc
        bp.concurrency_controller = mock_cc

        async def handler(item):
            return item + 100

        results = await bp.process_items([1, 2], handler=handler)
        assert len(results) == 2
