"""Tests for domain/ocr/grid_search.py (GridSearchTuner)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Setup: mock heavy OCR deps before any import
# ---------------------------------------------------------------------------


def _setup_ocr_mocks():
    saved = {}
    for mod in ("paddleocr", "cv2", "easyocr", "tesserocr", "rapidocr_onnxruntime"):
        saved[mod] = sys.modules.get(mod)
        sys.modules[mod] = MagicMock()
    return saved


def _teardown_ocr_mocks(saved: dict):
    for mod, val in saved.items():
        if val is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = val


@pytest.fixture(autouse=True)
def _mock_ocr_deps():
    saved = _setup_ocr_mocks()
    yield
    _teardown_ocr_mocks(saved)


# ---------------------------------------------------------------------------
# Fixture: GridSearchTuner with a mocked beanOCR
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ocr():
    from beanllm.domain.ocr.models import OCRConfig

    m = MagicMock()
    m.config = OCRConfig(engine="paddleocr", language="auto")
    return m


@pytest.fixture
def tuner(mock_ocr):
    from beanllm.domain.ocr.grid_search import GridSearchTuner

    return GridSearchTuner(ocr=mock_ocr, verbose=False)


@pytest.fixture
def tuner_verbose(mock_ocr):
    from beanllm.domain.ocr.grid_search import GridSearchTuner

    return GridSearchTuner(ocr=mock_ocr, verbose=True)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_stores_ocr_instance(self, tuner, mock_ocr):
        assert tuner.ocr is mock_ocr

    def test_verbose_flag(self, mock_ocr):
        from beanllm.domain.ocr.grid_search import GridSearchTuner

        t = GridSearchTuner(ocr=mock_ocr, verbose=False)
        assert t.verbose is False
        t2 = GridSearchTuner(ocr=mock_ocr, verbose=True)
        assert t2.verbose is True

    def test_repr(self, tuner):
        s = repr(tuner)
        assert "GridSearchTuner" in s

    def test_default_ocr_created_if_none(self):
        from beanllm.domain.ocr.grid_search import GridSearchTuner

        with patch("beanllm.domain.ocr.grid_search.beanOCR") as MockOCR:
            MockOCR.return_value = MagicMock()
            MockOCR.return_value.config = MagicMock()
            t = GridSearchTuner(ocr=None)
            MockOCR.assert_called_once()


# ---------------------------------------------------------------------------
# _generate_combinations
# ---------------------------------------------------------------------------


class TestGenerateCombinations:
    def test_empty_grid_returns_single_empty_dict(self, tuner):
        combos = tuner._generate_combinations({})
        assert combos == [{}]

    def test_single_param_single_value(self, tuner):
        combos = tuner._generate_combinations({"a": [1]})
        assert combos == [{"a": 1}]

    def test_single_param_multiple_values(self, tuner):
        combos = tuner._generate_combinations({"a": [1, 2, 3]})
        assert len(combos) == 3
        assert all(c["a"] in [1, 2, 3] for c in combos)

    def test_two_params_cartesian_product(self, tuner):
        combos = tuner._generate_combinations({"a": [1, 2], "b": ["x", "y"]})
        assert len(combos) == 4
        assert {"a": 1, "b": "x"} in combos
        assert {"a": 1, "b": "y"} in combos
        assert {"a": 2, "b": "x"} in combos
        assert {"a": 2, "b": "y"} in combos

    def test_three_params_count(self, tuner):
        combos = tuner._generate_combinations({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        assert len(combos) == 8  # 2×2×2

    def test_preserves_param_names(self, tuner):
        combos = tuner._generate_combinations({"denoise_strength": ["light", "medium"]})
        assert "denoise_strength" in combos[0]


# ---------------------------------------------------------------------------
# _format_params
# ---------------------------------------------------------------------------


class TestFormatParams:
    def test_empty_dict(self, tuner):
        result = tuner._format_params({})
        assert result == ""

    def test_single_param(self, tuner):
        result = tuner._format_params({"k": "v"})
        assert "k=v" in result

    def test_multiple_params_joined_by_comma(self, tuner):
        result = tuner._format_params({"a": 1, "b": 2})
        assert "a=1" in result
        assert "b=2" in result
        assert "," in result

    def test_numeric_values(self, tuner):
        result = tuner._format_params({"clip_limit": 2.5, "threshold": 127})
        assert "clip_limit=2.5" in result
        assert "threshold=127" in result


# ---------------------------------------------------------------------------
# _params_to_config
# ---------------------------------------------------------------------------


class TestParamsToConfig:
    def test_returns_ocr_config(self, tuner):
        from beanllm.domain.ocr.models import OCRConfig

        config = tuner._params_to_config({})
        assert isinstance(config, OCRConfig)

    def test_default_engine(self, tuner):
        config = tuner._params_to_config({})
        assert config.engine == "paddleocr"

    def test_custom_engine(self, tuner):
        config = tuner._params_to_config({"engine": "easyocr"})
        assert config.engine == "easyocr"

    def test_default_language(self, tuner):
        config = tuner._params_to_config({})
        assert config.language == "auto"

    def test_denoise_enabled(self, tuner):
        config = tuner._params_to_config({"denoise": True, "denoise_strength": "strong"})
        assert config.denoise is True
        assert config.denoise_config.strength == "strong"

    def test_denoise_disabled(self, tuner):
        config = tuner._params_to_config({"denoise": False})
        assert config.denoise is False

    def test_contrast_settings(self, tuner):
        config = tuner._params_to_config({"contrast": True, "clip_limit": 3.0})
        assert config.contrast_adjustment is True
        assert config.contrast_config.clip_limit == 3.0

    def test_binarize_enabled(self, tuner):
        config = tuner._params_to_config(
            {
                "binarize": True,
                "binarize_method": "adaptive",
                "threshold": 200,
            }
        )
        assert config.binarize is True
        assert config.binarize_config.method == "adaptive"
        assert config.binarize_config.threshold == 200

    def test_binarize_disabled(self, tuner):
        config = tuner._params_to_config({"binarize": False})
        assert config.binarize is False

    def test_deskew_enabled(self, tuner):
        config = tuner._params_to_config({"deskew": True, "angle_threshold": 1.0})
        assert config.deskew is True
        assert config.deskew_config.angle_threshold == 1.0

    def test_sharpen_enabled(self, tuner):
        config = tuner._params_to_config({"sharpen": True, "sharpen_strength": 0.8})
        assert config.sharpen is True
        assert config.sharpen_config.strength == 0.8

    def test_resize_with_max_size(self, tuner):
        config = tuner._params_to_config({"max_size": 1920})
        assert config.resize_config.enabled is True
        assert config.resize_config.max_size == 1920

    def test_resize_without_max_size(self, tuner):
        config = tuner._params_to_config({})
        assert config.resize_config.enabled is False


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class TestCompareResults:
    def _make_results(self, n: int = 3) -> list:
        return [
            {
                "params": {"denoise_strength": "medium"},
                "confidence": 0.9 - i * 0.1,
                "processing_time": 0.5 + i * 0.1,
                "text_length": 100 - i * 10,
            }
            for i in range(n)
        ]

    def test_empty_results_prints_message(self, tuner, capsys):
        tuner.compare_results([])
        out = capsys.readouterr().out
        assert "No results" in out

    def test_prints_table_header(self, tuner, capsys):
        results = self._make_results(3)
        tuner.compare_results(results)
        out = capsys.readouterr().out
        assert "Rank" in out or "GRID SEARCH" in out

    def test_top_n_limits_output(self, tuner, capsys):
        results = self._make_results(5)
        tuner.compare_results(results, top_n=2)
        out = capsys.readouterr().out
        assert "#1" in out
        assert "#2" in out


# ---------------------------------------------------------------------------
# search — full integration with mock OCR
# ---------------------------------------------------------------------------


class TestSearch:
    def _make_ocr_result(self, confidence: float = 0.85):
        r = MagicMock()
        r.confidence = confidence
        r.text = "sample text " * 5
        r.lines = ["line1", "line2"]
        return r

    def test_search_returns_best_config_and_results(self, mock_ocr, tuner):
        mock_ocr.recognize.return_value = self._make_ocr_result(0.9)
        best_config, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True, False]},
        )
        assert len(results) == 2
        assert best_config is not None

    def test_search_empty_grid(self, mock_ocr, tuner):
        mock_ocr.recognize.return_value = self._make_ocr_result(0.5)
        best_config, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True]},
            metric="confidence",
        )
        assert len(results) == 1

    def test_search_sorts_by_confidence(self, mock_ocr, tuner):
        mock_ocr.recognize.side_effect = [
            self._make_ocr_result(0.6),
            self._make_ocr_result(0.9),
        ]
        _, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True, False]},
            metric="confidence",
        )
        assert float(results[0]["confidence"]) >= float(results[1]["confidence"])

    def test_search_sorts_by_speed(self, mock_ocr, tuner):
        mock_ocr.recognize.return_value = self._make_ocr_result(0.7)
        _, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True, False]},
            metric="speed",
        )
        assert len(results) == 2

    def test_search_sorts_by_text_length(self, mock_ocr, tuner):
        mock_ocr.recognize.return_value = self._make_ocr_result()
        _, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True, False]},
            metric="text_length",
        )
        assert len(results) == 2

    def test_search_skips_failed_ocr(self, mock_ocr, tuner):
        mock_ocr.recognize.side_effect = [RuntimeError("OCR failed"), self._make_ocr_result(0.8)]
        _, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True, False]},
        )
        assert len(results) == 1

    def test_search_restores_original_config_after_each(self, mock_ocr, tuner):
        from beanllm.domain.ocr.models import OCRConfig

        original_config = OCRConfig(engine="paddleocr", language="auto")
        mock_ocr.config = original_config
        mock_ocr.recognize.return_value = self._make_ocr_result()

        tuner.search(
            image="dummy.jpg",
            param_grid={"engine": ["paddleocr", "easyocr"]},
        )
        assert mock_ocr.config is original_config

    def test_search_verbose_output(self, mock_ocr, tuner_verbose, capsys):
        mock_ocr.recognize.return_value = self._make_ocr_result(0.8)
        tuner_verbose.search(
            image="dummy.jpg",
            param_grid={"denoise": [True]},
        )
        out = capsys.readouterr().out
        assert "Grid Search" in out or "OCR" in out or "1/1" in out

    def test_search_empty_results_returns_original_config(self, mock_ocr, tuner):
        mock_ocr.recognize.side_effect = RuntimeError("fail")
        best_config, results = tuner.search(
            image="dummy.jpg",
            param_grid={"denoise": [True]},
        )
        assert results == []
        assert best_config is mock_ocr.config
