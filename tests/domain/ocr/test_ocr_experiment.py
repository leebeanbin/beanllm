"""Tests for domain/ocr/experiment.py (OCRExperiment)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock heavy OCR deps so the module-level imports succeed
_SAVED: dict = {}


@pytest.fixture(autouse=True)
def _mock_ocr_deps():
    for mod in ("paddleocr", "cv2", "easyocr", "tesserocr", "rapidocr_onnxruntime"):
        _SAVED[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock()
    yield
    for mod, val in _SAVED.items():
        if val is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = val


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ocr_result(confidence: float = 0.90, text: str = "hello world"):
    """Build a real OCRResult-like MagicMock."""
    r = MagicMock()
    r.confidence = confidence
    r.text = text
    r.lines = ["line1", "line2"]
    return r


def _make_mock_ocr(confidence: float = 0.90, text: str = "hello world"):
    """Build a mock beanOCR that returns a predictable OCRResult."""
    from beanllm.domain.ocr.models import OCRConfig

    m = MagicMock()
    m.config = OCRConfig(engine="paddleocr", language="auto")
    m.recognize.return_value = _make_ocr_result(confidence=confidence, text=text)
    return m


# ---------------------------------------------------------------------------
# OCRExperiment construction
# ---------------------------------------------------------------------------


class TestOCRExperimentInit:
    def test_init_with_provided_ocr(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        mock_ocr = _make_mock_ocr()
        exp = OCRExperiment(ocr=mock_ocr)
        assert exp.ocr is mock_ocr

    def test_init_creates_default_ocr_if_none(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        with patch("beanllm.domain.ocr.experiment.beanOCR") as MockOCR:
            mock_instance = _make_mock_ocr()
            MockOCR.return_value = mock_instance
            exp = OCRExperiment(ocr=None)
            MockOCR.assert_called_once()
            assert exp.ocr is mock_instance

    def test_repr(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        mock_ocr = _make_mock_ocr()
        exp = OCRExperiment(ocr=mock_ocr)
        r = repr(exp)
        assert "OCRExperiment" in r


# ---------------------------------------------------------------------------
# run_experiments
# ---------------------------------------------------------------------------


class TestRunExperiments:
    def _make_exp(self, confidence: float = 0.9) -> "object":
        from beanllm.domain.ocr.experiment import OCRExperiment
        from beanllm.domain.ocr.models import OCRConfig

        mock_ocr = _make_mock_ocr(confidence=confidence)
        return OCRExperiment(ocr=mock_ocr)

    def _make_configs(self, n: int = 2):
        from beanllm.domain.ocr.models import OCRConfig

        return [OCRConfig(engine="paddleocr") for _ in range(n)]

    def test_returns_list_with_correct_length(self):
        exp = self._make_exp()
        results = exp.run_experiments(np.zeros((10, 10, 3), dtype=np.uint8), self._make_configs(3))
        assert len(results) == 3

    def test_result_dict_contains_required_keys(self):
        exp = self._make_exp()
        results = exp.run_experiments(np.zeros((10, 10, 3), dtype=np.uint8), self._make_configs(1))
        r = results[0]
        for key in (
            "label",
            "config",
            "result",
            "processing_time",
            "text_length",
            "line_count",
            "avg_confidence",
        ):
            assert key in r

    def test_auto_labels_when_not_provided(self):
        exp = self._make_exp()
        results = exp.run_experiments(np.zeros((10, 10), dtype=np.uint8), self._make_configs(3))
        assert results[0]["label"] == "Config 1"
        assert results[2]["label"] == "Config 3"

    def test_custom_labels(self):
        exp = self._make_exp()
        results = exp.run_experiments(
            np.zeros((10, 10), dtype=np.uint8),
            self._make_configs(2),
            labels=["Fast", "Accurate"],
        )
        assert results[0]["label"] == "Fast"
        assert results[1]["label"] == "Accurate"

    def test_raises_value_error_on_label_mismatch(self):
        exp = self._make_exp()
        from beanllm.domain.ocr.experiment import OCRExperiment

        with pytest.raises(ValueError, match="Number of labels"):
            exp.run_experiments(
                np.zeros((10, 10), dtype=np.uint8),
                self._make_configs(2),
                labels=["Only One"],
            )

    def test_config_restored_after_each_run(self):
        from beanllm.domain.ocr.experiment import OCRExperiment
        from beanllm.domain.ocr.models import OCRConfig

        original_config = OCRConfig(engine="paddleocr", language="auto")
        mock_ocr = MagicMock()
        mock_ocr.config = original_config
        mock_ocr.recognize.return_value = _make_ocr_result()

        exp = OCRExperiment(ocr=mock_ocr)
        exp.run_experiments(np.zeros((10, 10), dtype=np.uint8), self._make_configs(3))
        assert mock_ocr.config is original_config

    def test_processing_time_is_positive(self):
        exp = self._make_exp()
        results = exp.run_experiments(np.zeros((10, 10), dtype=np.uint8), self._make_configs(1))
        assert results[0]["processing_time"] >= 0

    def test_text_length_and_line_count_extracted(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        mock_ocr = _make_mock_ocr(text="abc")
        mock_result = _make_ocr_result(text="abc")
        mock_result.lines = ["line1", "line2", "line3"]
        mock_ocr.recognize.return_value = mock_result

        exp = OCRExperiment(ocr=mock_ocr)
        results = exp.run_experiments(np.zeros((10, 10), dtype=np.uint8), self._make_configs(1))
        assert results[0]["text_length"] == 3
        assert results[0]["line_count"] == 3


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class TestCompareResults:
    def _make_results(self, n: int = 3) -> list:
        from beanllm.domain.ocr.models import OCRConfig

        return [
            {
                "label": f"Config {i + 1}",
                "config": OCRConfig(),
                "result": _make_ocr_result(confidence=0.9 - i * 0.1),
                "processing_time": 0.5 + i * 0.2,
                "text_length": 100,
                "line_count": 5,
                "avg_confidence": 0.9 - i * 0.1,
            }
            for i in range(n)
        ]

    def test_empty_results_prints_message(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results([])
        out = capsys.readouterr().out
        assert "No results" in out

    def test_prints_table_with_headers(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results(self._make_results(3))
        out = capsys.readouterr().out
        assert "Label" in out
        assert "Confidence" in out

    def test_marks_best_confidence_with_star(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results(self._make_results(3))
        out = capsys.readouterr().out
        assert "⭐" in out

    def test_marks_best_speed_with_lightning(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results(self._make_results(3))
        out = capsys.readouterr().out
        assert "⚡" in out

    def test_show_text_preview(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        results = self._make_results(1)
        results[0]["result"].text = "This is a preview text"
        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results(results, show_text=True)
        out = capsys.readouterr().out
        assert "TEXT PREVIEW" in out

    def test_text_preview_truncated_when_long(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        results = self._make_results(1)
        results[0]["result"].text = "A" * 200
        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results(results, show_text=True, max_text_preview=50)
        out = capsys.readouterr().out
        assert "..." in out

    def test_single_result_no_crash(self, capsys):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        exp.compare_results(self._make_results(1))
        out = capsys.readouterr().out
        assert "Config 1" in out


# ---------------------------------------------------------------------------
# get_best_config
# ---------------------------------------------------------------------------


class TestGetBestConfig:
    def _make_results(self) -> list:
        from beanllm.domain.ocr.models import OCRConfig

        return [
            {
                "label": "A",
                "config": OCRConfig(engine="paddleocr"),
                "result": _make_ocr_result(0.7),
                "processing_time": 2.0,
                "text_length": 80,
                "avg_confidence": 0.7,
            },
            {
                "label": "B",
                "config": OCRConfig(engine="easyocr"),
                "result": _make_ocr_result(0.9),
                "processing_time": 0.5,
                "text_length": 120,
                "avg_confidence": 0.9,
            },
        ]

    def test_raises_value_error_on_empty_results(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        with pytest.raises(ValueError, match="No results"):
            exp.get_best_config([])

    def test_best_by_confidence(self):
        from beanllm.domain.ocr.experiment import OCRExperiment
        from beanllm.domain.ocr.models import OCRConfig

        exp = OCRExperiment(ocr=_make_mock_ocr())
        best = exp.get_best_config(self._make_results(), metric="confidence")
        assert isinstance(best, OCRConfig)
        assert best.engine == "easyocr"

    def test_best_by_speed(self):
        from beanllm.domain.ocr.experiment import OCRExperiment
        from beanllm.domain.ocr.models import OCRConfig

        exp = OCRExperiment(ocr=_make_mock_ocr())
        best = exp.get_best_config(self._make_results(), metric="speed")
        assert isinstance(best, OCRConfig)
        assert best.engine == "easyocr"  # processing_time 0.5 is faster

    def test_best_by_text_length(self):
        from beanllm.domain.ocr.experiment import OCRExperiment
        from beanllm.domain.ocr.models import OCRConfig

        exp = OCRExperiment(ocr=_make_mock_ocr())
        best = exp.get_best_config(self._make_results(), metric="text_length")
        assert isinstance(best, OCRConfig)
        assert best.engine == "easyocr"  # text_length 120

    def test_invalid_metric_raises_value_error(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        with pytest.raises(ValueError, match="Invalid metric"):
            exp.get_best_config(self._make_results(), metric="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_detailed_comparison
# ---------------------------------------------------------------------------


class TestGetDetailedComparison:
    def _make_results(self) -> list:
        from beanllm.domain.ocr.models import OCRConfig

        return [
            {
                "label": "A",
                "config": OCRConfig(),
                "result": _make_ocr_result(0.7),
                "processing_time": 1.0,
                "text_length": 50,
                "avg_confidence": 0.7,
            },
            {
                "label": "B",
                "config": OCRConfig(),
                "result": _make_ocr_result(0.9),
                "processing_time": 0.5,
                "text_length": 100,
                "avg_confidence": 0.9,
            },
        ]

    def test_empty_returns_empty_dict(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        assert exp.get_detailed_comparison([]) == {}

    def test_returns_dict_with_expected_keys(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        stats = exp.get_detailed_comparison(self._make_results())
        for key in (
            "best_confidence",
            "best_speed",
            "best_text_length",
            "avg_confidence",
            "avg_speed",
            "avg_text_length",
            "total_experiments",
        ):
            assert key in stats

    def test_avg_confidence_computed_correctly(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        stats = exp.get_detailed_comparison(self._make_results())
        assert abs(stats["avg_confidence"] - 0.8) < 1e-9

    def test_total_experiments_count(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        stats = exp.get_detailed_comparison(self._make_results())
        assert stats["total_experiments"] == 2

    def test_best_confidence_is_highest(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        results = self._make_results()
        stats = exp.get_detailed_comparison(results)
        assert stats["best_confidence"]["avg_confidence"] == 0.9

    def test_best_speed_has_lowest_processing_time(self):
        from beanllm.domain.ocr.experiment import OCRExperiment

        exp = OCRExperiment(ocr=_make_mock_ocr())
        stats = exp.get_detailed_comparison(self._make_results())
        assert stats["best_speed"]["processing_time"] == 0.5
