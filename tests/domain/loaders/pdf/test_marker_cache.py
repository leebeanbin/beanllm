"""Tests for domain/loaders/pdf/engines/marker_cache.py (MarkerCacheMixin)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Concrete implementation of MarkerCacheMixin for testing
# ---------------------------------------------------------------------------
from beanllm.domain.loaders.pdf.engines.marker_cache import MarkerCacheMixin  # noqa: E402


class ConcreteMarkerEngine(MarkerCacheMixin):
    """Minimal concrete class that satisfies MarkerCacheMixin's type contract."""

    enable_cache: bool = True

    def __init__(
        self,
        cache_size: int = 5,
        use_gpu: bool = False,
        batch_size: int = 2,
    ):
        self.max_pages = None
        self.cache_size = cache_size
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._result_cache: Dict[str, Dict[str, Any]] = {}
        self._model_cache: Optional[Any] = None

    def extract(self, pdf_path, config):
        """Stub — overridden in tests via mock."""
        raise NotImplementedError


@pytest.fixture
def engine():
    return ConcreteMarkerEngine(cache_size=3, use_gpu=False, batch_size=2)


@pytest.fixture
def gpu_engine():
    return ConcreteMarkerEngine(cache_size=3, use_gpu=True, batch_size=2)


# ---------------------------------------------------------------------------
# _get_cache_key
# ---------------------------------------------------------------------------


class TestGetCacheKey:
    def test_returns_string(self, engine, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 ...")
        key = engine._get_cache_key(pdf, {})
        assert isinstance(key, str)
        assert len(key) == 64  # sha256 hex

    def test_same_file_same_config_same_key(self, engine, tmp_path):
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")
        config = {"to_markdown": True, "max_pages": 10}
        k1 = engine._get_cache_key(pdf, config)
        k2 = engine._get_cache_key(pdf, config)
        assert k1 == k2

    def test_different_config_different_key(self, engine, tmp_path):
        pdf = tmp_path / "b.pdf"
        pdf.write_bytes(b"%PDF")
        k1 = engine._get_cache_key(pdf, {"to_markdown": True})
        k2 = engine._get_cache_key(pdf, {"to_markdown": False})
        assert k1 != k2

    def test_includes_max_pages_from_config(self, engine, tmp_path):
        pdf = tmp_path / "c.pdf"
        pdf.write_bytes(b"%PDF")
        k1 = engine._get_cache_key(pdf, {"max_pages": 5})
        k2 = engine._get_cache_key(pdf, {"max_pages": 10})
        assert k1 != k2

    def test_falls_back_to_self_max_pages(self, engine, tmp_path):
        engine.max_pages = 20
        pdf = tmp_path / "d.pdf"
        pdf.write_bytes(b"%PDF")
        k = engine._get_cache_key(pdf, {})
        assert isinstance(k, str)


# ---------------------------------------------------------------------------
# _cache_result
# ---------------------------------------------------------------------------


class TestCacheResult:
    def test_stores_result(self, engine):
        engine._cache_result("key1", {"data": "hello"})
        assert "key1" in engine._result_cache

    def test_stores_copy_not_reference(self, engine):
        original = {"data": "hello"}
        engine._cache_result("k", original)
        original["data"] = "modified"
        assert engine._result_cache["k"]["data"] == "hello"

    def test_evicts_oldest_when_full(self, engine):
        # cache_size = 3
        engine._cache_result("k1", {"v": 1})
        engine._cache_result("k2", {"v": 2})
        engine._cache_result("k3", {"v": 3})
        # Cache is full; adding k4 should evict k1 (oldest)
        engine._cache_result("k4", {"v": 4})
        assert "k1" not in engine._result_cache
        assert "k4" in engine._result_cache

    def test_cache_size_respected(self, engine):
        for i in range(10):
            engine._cache_result(f"k{i}", {"v": i})
        assert len(engine._result_cache) <= engine.cache_size


# ---------------------------------------------------------------------------
# _load_models_cached
# ---------------------------------------------------------------------------


class TestLoadModelsCached:
    def test_loads_models_on_first_call(self, engine):
        mock_models = [MagicMock(), MagicMock()]
        mock_marker = MagicMock()
        mock_marker.models.load_all_models.return_value = mock_models

        with patch.dict(sys.modules, {"marker": mock_marker, "marker.models": mock_marker.models}):
            result = engine._load_models_cached()

        assert result is mock_models
        assert engine._model_cache is mock_models

    def test_returns_cached_on_second_call(self, engine):
        cached = MagicMock()
        engine._model_cache = cached

        result = engine._load_models_cached()
        assert result is cached

    def test_does_not_call_load_all_models_when_cached(self, engine):
        engine._model_cache = "already_loaded"

        mock_marker = MagicMock()
        with patch.dict(sys.modules, {"marker": mock_marker, "marker.models": mock_marker.models}):
            engine._load_models_cached()

        mock_marker.models.load_all_models.assert_not_called()


# ---------------------------------------------------------------------------
# _cleanup_gpu_memory
# ---------------------------------------------------------------------------


class TestCleanupGPUMemory:
    def test_calls_cuda_empty_cache_when_available(self, engine):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            engine._cleanup_gpu_memory()

        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()

    def test_skips_cuda_when_not_available(self, engine):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            engine._cleanup_gpu_memory()

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_handles_import_error_gracefully(self, engine):
        saved = sys.modules.pop("torch", None)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            engine._cleanup_gpu_memory()  # Should not raise
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

    def test_handles_unexpected_exception_gracefully(self, engine):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            engine._cleanup_gpu_memory()  # Should not raise


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clears_result_cache(self, engine):
        engine._result_cache["k1"] = {"v": 1}
        engine._result_cache["k2"] = {"v": 2}
        engine.clear_cache()
        assert engine._result_cache == {}

    def test_clears_model_cache(self, engine):
        engine._model_cache = MagicMock()
        engine.clear_cache()
        assert engine._model_cache is None

    def test_does_nothing_with_model_cache_when_none(self, engine):
        engine._model_cache = None
        engine.clear_cache()  # Should not raise

    def test_calls_cleanup_gpu_when_use_gpu(self, gpu_engine):
        gpu_engine._model_cache = MagicMock()
        with patch.object(gpu_engine, "_cleanup_gpu_memory") as mock_cleanup:
            gpu_engine.clear_cache()
        mock_cleanup.assert_called_once()

    def test_skips_gpu_cleanup_when_no_model(self, gpu_engine):
        gpu_engine._model_cache = None
        with patch.object(gpu_engine, "_cleanup_gpu_memory") as mock_cleanup:
            gpu_engine.clear_cache()
        mock_cleanup.assert_not_called()


# ---------------------------------------------------------------------------
# get_cache_stats
# ---------------------------------------------------------------------------


class TestGetCacheStats:
    def test_returns_dict(self, engine):
        stats = engine.get_cache_stats()
        assert isinstance(stats, dict)

    def test_cache_enabled_field(self, engine):
        stats = engine.get_cache_stats()
        assert "cache_enabled" in stats
        assert stats["cache_enabled"] is True

    def test_cache_size_reflects_current(self, engine):
        engine._result_cache["k"] = {"v": 1}
        stats = engine.get_cache_stats()
        assert stats["cache_size"] == 1

    def test_cache_limit_field(self, engine):
        stats = engine.get_cache_stats()
        assert stats["cache_limit"] == engine.cache_size

    def test_model_cached_false_when_none(self, engine):
        engine._model_cache = None
        stats = engine.get_cache_stats()
        assert stats["model_cached"] is False

    def test_model_cached_true_when_set(self, engine):
        engine._model_cache = MagicMock()
        stats = engine.get_cache_stats()
        assert stats["model_cached"] is True

    def test_use_gpu_field(self, engine, gpu_engine):
        assert engine.get_cache_stats()["use_gpu"] is False
        assert gpu_engine.get_cache_stats()["use_gpu"] is True


# ---------------------------------------------------------------------------
# extract_batch
# ---------------------------------------------------------------------------


class TestExtractBatch:
    def test_returns_list_of_results(self, engine, tmp_path):
        pdfs = [tmp_path / f"p{i}.pdf" for i in range(3)]
        for p in pdfs:
            p.write_bytes(b"%PDF")

        expected = [{"pages": [{"page": i}]} for i in range(3)]
        engine.extract = MagicMock(side_effect=expected)

        results = engine.extract_batch(pdfs, {})
        assert len(results) == 3
        assert results[0] == expected[0]

    def test_returns_none_for_failed_pdf(self, engine, tmp_path):
        pdfs = [tmp_path / f"p{i}.pdf" for i in range(2)]
        for p in pdfs:
            p.write_bytes(b"%PDF")

        engine.extract = MagicMock(side_effect=[{"ok": True}, RuntimeError("fail")])
        results = engine.extract_batch(pdfs, {})
        assert results[0] == {"ok": True}
        assert results[1] is None

    def test_empty_list_returns_empty(self, engine):
        results = engine.extract_batch([], {})
        assert results == []

    def test_calls_cleanup_every_batch_size_for_gpu(self, gpu_engine, tmp_path):
        pdfs = [tmp_path / f"p{i}.pdf" for i in range(4)]
        for p in pdfs:
            p.write_bytes(b"%PDF")

        gpu_engine.extract = MagicMock(return_value={"ok": True})

        with patch.object(gpu_engine, "_cleanup_gpu_memory") as mock_cleanup:
            gpu_engine.extract_batch(pdfs, {})

        # batch_size=2, so cleanup called after index 2 and 4
        assert mock_cleanup.call_count == 2

    def test_processes_all_pdfs(self, engine, tmp_path):
        n = 5
        pdfs = [tmp_path / f"p{i}.pdf" for i in range(n)]
        for p in pdfs:
            p.write_bytes(b"%PDF")

        engine.extract = MagicMock(return_value={"ok": True})
        results = engine.extract_batch(pdfs, {})
        assert len(results) == n
        assert engine.extract.call_count == n
