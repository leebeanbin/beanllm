"""Extra coverage for domain/ocr/bean_ocr.py — covers cache/lock/hash branches."""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from PIL import Image

MODULE = "beanllm.domain.ocr.bean_ocr"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ocr_result():
    from beanllm.domain.ocr.models import BoundingBox, OCRResult, OCRTextLine

    bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
    line = OCRTextLine(text="text", bbox=bbox, confidence=0.9)
    return OCRResult(
        text="text",
        lines=[line],
        language="en",
        confidence=0.9,
        engine="paddleocr",
        processing_time=0.1,
    )


def _make_mock_engine():
    """Return a mock BaseOCREngine that returns a suitable raw dict."""
    from beanllm.domain.ocr.engines.base import BaseOCREngine
    from beanllm.domain.ocr.models import BoundingBox, OCRTextLine

    class _MockEngine(BaseOCREngine):
        def __init__(self):
            super().__init__(name="mock")

        def recognize(self, image, config):
            bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=20.0)
            line = OCRTextLine(text="text", bbox=bbox, confidence=0.9)
            return {
                "text": "text",
                "lines": [line],
                "confidence": 0.9,
                "language": config.language,
                "metadata": {},
            }

    return _MockEngine()


@pytest.fixture
def mock_engine():
    return _make_mock_engine()


def _make_ocr_instance(
    mock_engine, cache=None, lock_manager=None, rate_limiter=None, event_logger=None, **extra_config
):
    """Create a beanOCR with preprocessing disabled, passing protocols directly."""
    from beanllm.domain.ocr.bean_ocr import beanOCR
    from beanllm.domain.ocr.models import OCRConfig

    config = OCRConfig(engine="paddleocr", enable_preprocessing=False, **extra_config)
    with patch(f"{MODULE}.create_ocr_engine", return_value=mock_engine):
        return beanOCR(
            config=config,
            cache=cache,
            lock_manager=lock_manager,
            rate_limiter=rate_limiter,
            event_logger=event_logger,
        )


@pytest.fixture
def ocr(mock_engine):
    return _make_ocr_instance(mock_engine)


# ---------------------------------------------------------------------------
# Constructor paths
# ---------------------------------------------------------------------------


class TestConstructorPaths:
    def test_init_stores_config(self, mock_engine):
        from beanllm.domain.ocr.models import OCRConfig

        config = OCRConfig(engine="paddleocr", enable_preprocessing=False)
        with patch(f"{MODULE}.create_ocr_engine", return_value=mock_engine):
            from beanllm.domain.ocr.bean_ocr import beanOCR

            o = beanOCR(config=config)
            assert o.config is config

    def test_init_with_kwargs(self, mock_engine):
        o = _make_ocr_instance(mock_engine, language="ko")
        assert o.config.language == "ko"

    def test_enable_preprocessing_creates_preprocessor(self, mock_engine):
        mock_pp = MagicMock()
        with (
            patch(f"{MODULE}.create_ocr_engine", return_value=mock_engine),
            patch("beanllm.domain.ocr.bean_ocr.ImagePreprocessor", mock_pp, create=True),
        ):
            from beanllm.domain.ocr.bean_ocr import beanOCR
            from beanllm.domain.ocr.models import OCRConfig

            # Patch the lazy import inside _init_components
            with patch.dict(
                "sys.modules",
                {"beanllm.domain.ocr.preprocessing": MagicMock(ImagePreprocessor=mock_pp)},
            ):
                config = OCRConfig(engine="paddleocr", enable_preprocessing=True)
                o = beanOCR(config=config)
                assert o._preprocessor is not None

    def test_enable_llm_postprocessing_creates_postprocessor(self, mock_engine):
        mock_pp = MagicMock()
        with (
            patch(f"{MODULE}.create_ocr_engine", return_value=mock_engine),
            patch.dict(
                "sys.modules",
                {"beanllm.domain.ocr.postprocessing": MagicMock(LLMPostprocessor=mock_pp)},
            ),
        ):
            from beanllm.domain.ocr.bean_ocr import beanOCR
            from beanllm.domain.ocr.models import OCRConfig

            config = OCRConfig(
                engine="paddleocr",
                enable_preprocessing=False,
                enable_llm_postprocessing=True,
                llm_model="gpt-4o-mini",
            )
            o = beanOCR(config=config)
            assert o._postprocessor is not None

    def test_init_without_optional_protocols(self, mock_engine):
        o = _make_ocr_instance(mock_engine)
        assert o._cache is None
        assert o._rate_limiter is None
        assert o._event_logger is None
        assert o._lock_manager is None


# ---------------------------------------------------------------------------
# recognize() — image hash computation branches
# ---------------------------------------------------------------------------


class TestRecognizeHashBranches:
    def test_hash_for_path_string(self, mock_engine, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (10, 10)).save(str(img_path))
        o = _make_ocr_instance(mock_engine)
        result = o.recognize(str(img_path))
        assert result is not None

    def test_hash_for_path_object(self, mock_engine, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (10, 10)).save(str(img_path))
        o = _make_ocr_instance(mock_engine)
        result = o.recognize(img_path)
        assert result is not None

    def test_hash_for_nonexistent_path(self, mock_engine):
        o = _make_ocr_instance(mock_engine)
        # pipeline_load_image raises FileNotFoundError for missing files
        with pytest.raises(FileNotFoundError):
            o.recognize("/nonexistent/path.jpg")

    def test_hash_for_numpy_array(self, mock_engine):
        o = _make_ocr_instance(mock_engine)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = o.recognize(image)
        assert result is not None

    def test_hash_for_pil_image(self, mock_engine):
        o = _make_ocr_instance(mock_engine)
        pil_img = Image.new("RGB", (10, 10))
        result = o.recognize(pil_img)
        assert result is not None


# ---------------------------------------------------------------------------
# recognize() — cache paths
# ---------------------------------------------------------------------------


class TestRecognizeCachePath:
    def test_cache_hit_returns_cached_result(self, mock_engine):
        cached_result = _make_ocr_result()
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=cached_result)

        o = _make_ocr_instance(mock_engine, cache=mock_cache)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = o.recognize(image)
        assert result is cached_result

    def test_cache_miss_runs_recognition(self, mock_engine):
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        o = _make_ocr_instance(mock_engine, cache=mock_cache)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = o.recognize(image)
        assert result is not None

    def test_cache_exception_is_swallowed(self, mock_engine):
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(side_effect=Exception("Redis down"))

        o = _make_ocr_instance(mock_engine, cache=mock_cache)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = o.recognize(image)
        assert result is not None

    def test_cache_set_called_after_recognition(self, mock_engine):
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        o = _make_ocr_instance(mock_engine, cache=mock_cache)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        o.recognize(image)
        mock_cache.set.assert_awaited_once()


# ---------------------------------------------------------------------------
# recognize() — lock manager path
# ---------------------------------------------------------------------------


class TestRecognizeLockPath:
    def test_lock_manager_path_runs_without_running_event_loop(self, mock_engine):
        mock_lock_manager = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=None)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_lock_manager.with_file_lock.return_value = mock_ctx

        o = _make_ocr_instance(mock_engine, lock_manager=mock_lock_manager)
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        result = o.recognize(image)
        assert result is not None


# ---------------------------------------------------------------------------
# recognize_pdf_page()
# ---------------------------------------------------------------------------


class TestRecognizePdfPage:
    def test_import_error_when_fitz_missing(self, mock_engine, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        o = _make_ocr_instance(mock_engine)

        saved = sys.modules.get("fitz")
        sys.modules["fitz"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="PyMuPDF"):
                o.recognize_pdf_page(str(pdf))
        finally:
            if saved is None:
                sys.modules.pop("fitz", None)
            else:
                sys.modules["fitz"] = saved

    def test_file_not_found_error(self, mock_engine):
        o = _make_ocr_instance(mock_engine)
        with pytest.raises(FileNotFoundError):
            o.recognize_pdf_page("/no/such/file.pdf")

    def test_invalid_page_number_raises_index_error(self, mock_engine, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3
        o = _make_ocr_instance(mock_engine)

        with patch("fitz.open", return_value=mock_doc):
            with pytest.raises(IndexError, match="Invalid page number"):
                o.recognize_pdf_page(str(pdf), page_num=99)

    def test_rgb_page_processed_correctly(self, mock_engine, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_pix = MagicMock()
        mock_pix.height = 10
        mock_pix.width = 10
        mock_pix.n = 3  # RGB
        mock_pix.samples = np.zeros(10 * 10 * 3, dtype=np.uint8).tobytes()

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.return_value = mock_page

        o = _make_ocr_instance(mock_engine)
        with patch("fitz.open", return_value=mock_doc):
            result = o.recognize_pdf_page(str(pdf), page_num=0)
            assert result is not None

    def test_rgba_page_strips_alpha_channel(self, mock_engine, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_pix = MagicMock()
        mock_pix.height = 10
        mock_pix.width = 10
        mock_pix.n = 4  # RGBA
        mock_pix.samples = np.zeros(10 * 10 * 4, dtype=np.uint8).tobytes()

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page

        o = _make_ocr_instance(mock_engine)
        with patch("fitz.open", return_value=mock_doc):
            result = o.recognize_pdf_page(str(pdf), page_num=0)
            assert result is not None


# ---------------------------------------------------------------------------
# batch_recognize()
# ---------------------------------------------------------------------------


class TestBatchRecognize:
    def test_empty_list(self, ocr):
        assert ocr.batch_recognize([]) == []

    def test_multiple_images(self, ocr):
        images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)]
        results = ocr.batch_recognize(images)
        assert len(results) == 4

    def test_results_are_ocr_result_objects(self, ocr):
        from beanllm.domain.ocr.models import OCRResult

        images = [np.zeros((10, 10, 3), dtype=np.uint8)]
        results = ocr.batch_recognize(images)
        assert isinstance(results[0], OCRResult)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_engine_language_gpu(self, mock_engine):
        o = _make_ocr_instance(mock_engine, language="ko", use_gpu=True)
        r = repr(o)
        assert "paddleocr" in r
        assert "ko" in r
        assert "True" in r
