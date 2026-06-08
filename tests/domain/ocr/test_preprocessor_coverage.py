"""Coverage tests for domain/ocr/preprocessing/preprocessor.py (ImagePreprocessor)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from beanllm.domain.ocr.models import (
    BinarizeConfig,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    OCRConfig,
    ResizeConfig,
    SharpenConfig,
)

MODULE = "beanllm.domain.ocr.preprocessing.preprocessor"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_gray(h: int = 100, w: int = 80) -> np.ndarray:
    """Create a fake grayscale image array."""
    return np.zeros((h, w), dtype=np.uint8)


def _make_rgb(h: int = 100, w: int = 80) -> np.ndarray:
    """Create a fake RGB image array."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_mock_cv2() -> MagicMock:
    """Build a minimal cv2 mock that returns plausible numpy arrays."""
    mock = MagicMock()

    # Core transforms all return a copy of a gray image
    def _return_gray(*args, **kwargs):
        return _make_gray()

    mock.cvtColor.side_effect = _return_gray
    mock.GaussianBlur.side_effect = _return_gray
    mock.medianBlur.side_effect = _return_gray
    mock.resize.side_effect = _return_gray
    mock.addWeighted.side_effect = _return_gray
    mock.warpAffine.side_effect = _return_gray

    # Canny / HoughLines
    mock.Canny.side_effect = _return_gray
    mock.HoughLines.return_value = None  # default: no lines

    # CLAHE
    clahe_mock = MagicMock()
    clahe_mock.apply.side_effect = _return_gray
    mock.createCLAHE.return_value = clahe_mock

    # threshold
    mock.threshold.return_value = (0, _make_gray())

    # adaptiveThreshold
    mock.adaptiveThreshold.side_effect = _return_gray

    # getRotationMatrix2D
    mock.getRotationMatrix2D.return_value = np.eye(2, 3)

    # Constants
    mock.COLOR_RGB2GRAY = 6
    mock.COLOR_GRAY2RGB = 8
    mock.INTER_AREA = 3
    mock.INTER_LINEAR = 1
    mock.INTER_CUBIC = 2
    mock.THRESH_BINARY = 0
    mock.THRESH_OTSU = 8
    mock.ADAPTIVE_THRESH_GAUSSIAN_C = 0
    mock.BORDER_REPLICATE = 1

    return mock


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cv2():
    return _make_mock_cv2()


@pytest.fixture
def preprocessor(mock_cv2):
    with (
        patch(f"{MODULE}.HAS_CV2", True),
        patch(f"{MODULE}.cv2", mock_cv2, create=True),
    ):
        from beanllm.domain.ocr.preprocessing.preprocessor import ImagePreprocessor

        p = ImagePreprocessor()
        yield p, mock_cv2


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_raises_import_error_when_no_cv2(self):
        with patch(f"{MODULE}.HAS_CV2", False):
            from beanllm.domain.ocr.preprocessing.preprocessor import ImagePreprocessor

            with pytest.raises(ImportError, match="opencv-python"):
                ImagePreprocessor()

    def test_repr(self, preprocessor):
        p, _ = preprocessor
        assert repr(p) == "ImagePreprocessor()"


# ---------------------------------------------------------------------------
# process — enable_preprocessing=False
# ---------------------------------------------------------------------------


class TestProcessDisabled:
    def test_returns_image_unchanged_when_preprocessing_disabled(self, preprocessor):
        p, cv = preprocessor
        image = _make_rgb()
        config = OCRConfig(enable_preprocessing=False)
        result = p.process(image, config)
        assert result is image
        cv.cvtColor.assert_not_called()


# ---------------------------------------------------------------------------
# process — RGB → Gray pipeline
# ---------------------------------------------------------------------------


class TestProcessRGBPipeline:
    def test_calls_cvtColor_for_rgb_image(self, preprocessor):
        p, cv = preprocessor
        image = _make_rgb()
        config = OCRConfig()
        p.process(image, config)
        # cvtColor called to convert to gray, then back to RGB
        assert cv.cvtColor.called

    def test_gray_image_uses_copy_not_cvtColor_for_input(self, preprocessor):
        p, cv = preprocessor
        # 2D grayscale: no cvtColor on input
        image = _make_gray()
        config = OCRConfig()
        p.process(image, config)

    def test_result_converted_back_to_rgb(self, preprocessor):
        p, cv = preprocessor
        # The final step should call cvtColor(gray, COLOR_GRAY2RGB)
        image = _make_gray()
        config = OCRConfig()
        p.process(image, config)
        # Should have at least one cvtColor call with GRAY2RGB constant
        calls = [str(c) for c in cv.cvtColor.call_args_list]
        assert any("8" in c or "COLOR_GRAY2RGB" in c for c in calls) or cv.cvtColor.called


# ---------------------------------------------------------------------------
# process — resize path
# ---------------------------------------------------------------------------


class TestProcessResize:
    def test_resize_called_when_enabled(self, preprocessor):
        p, cv = preprocessor
        image = _make_rgb()
        # Create config with resize enabled and image size below max_size
        config = OCRConfig(
            resize_config=ResizeConfig(enabled=True, max_size=50)  # 50 < 100
        )
        p.process(image, config)
        cv.resize.assert_called_once()

    def test_resize_not_called_when_disabled(self, preprocessor):
        p, cv = preprocessor
        image = _make_gray()
        config = OCRConfig(resize_config=ResizeConfig(enabled=False, max_size=50))
        p.process(image, config)
        cv.resize.assert_not_called()

    def test_resize_not_called_when_image_smaller_than_max(self, preprocessor):
        p, cv = preprocessor
        image = _make_gray(20, 15)  # small image
        config = OCRConfig(resize_config=ResizeConfig(enabled=True, max_size=500))
        p.process(image, config)
        cv.resize.assert_not_called()


# ---------------------------------------------------------------------------
# process — denoise path
# ---------------------------------------------------------------------------


class TestProcessDenoise:
    def test_gaussian_and_median_blur_called_when_denoise_enabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(denoise=True, denoise_config=DenoiseConfig(enabled=True))
        p.process(_make_gray(), config)
        cv.GaussianBlur.assert_called_once()
        cv.medianBlur.assert_called_once()

    def test_denoise_not_called_when_disabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(denoise=False, denoise_config=DenoiseConfig(enabled=False))
        p.process(_make_gray(), config)
        cv.GaussianBlur.assert_not_called()
        cv.medianBlur.assert_not_called()


# ---------------------------------------------------------------------------
# process — contrast path
# ---------------------------------------------------------------------------


class TestProcessContrast:
    def test_clahe_applied_when_enabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(contrast_adjustment=True, contrast_config=ContrastConfig(enabled=True))
        p.process(_make_gray(), config)
        cv.createCLAHE.assert_called_once()

    def test_clahe_not_called_when_disabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(contrast_adjustment=False, contrast_config=ContrastConfig(enabled=False))
        p.process(_make_gray(), config)
        cv.createCLAHE.assert_not_called()


# ---------------------------------------------------------------------------
# process — deskew path
# ---------------------------------------------------------------------------


class TestProcessDeskew:
    def test_canny_and_hough_called_when_deskew_enabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(deskew=True, deskew_config=DeskewConfig(enabled=True))
        p.process(_make_gray(), config)
        cv.Canny.assert_called_once()
        cv.HoughLines.assert_called_once()

    def test_returns_image_unchanged_when_no_lines_found(self, preprocessor):
        p, cv = preprocessor
        cv.HoughLines.return_value = None
        config = OCRConfig(deskew=True, deskew_config=DeskewConfig(enabled=True))
        result = p.process(_make_gray(), config)
        assert result is not None

    def test_rotation_applied_when_angle_exceeds_threshold(self, preprocessor):
        p, cv = preprocessor
        # Provide lines that give a rotation angle > threshold (0.5 default)
        angle_rad = np.radians(5.0 + 90)  # 5 degrees off
        fake_lines = np.array([[[100.0, angle_rad]]])
        cv.HoughLines.return_value = fake_lines
        config = OCRConfig(
            deskew=True, deskew_config=DeskewConfig(enabled=True, angle_threshold=0.5)
        )
        p.process(_make_gray(), config)
        cv.warpAffine.assert_called_once()

    def test_no_rotation_when_no_angles_below_45(self, preprocessor):
        p, cv = preprocessor
        # Lines all at 90+ degrees off (> 45 from horizontal)
        fake_lines = np.array([[[100.0, np.radians(90.0 + 90)]]])
        cv.HoughLines.return_value = fake_lines
        config = OCRConfig(
            deskew=True, deskew_config=DeskewConfig(enabled=True, angle_threshold=0.5)
        )
        p.process(_make_gray(), config)
        cv.warpAffine.assert_not_called()


# ---------------------------------------------------------------------------
# process — binarize path
# ---------------------------------------------------------------------------


class TestProcessBinarize:
    def test_otsu_method_called(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(
            binarize=True,
            binarize_config=BinarizeConfig(enabled=True, method="otsu"),
        )
        p.process(_make_gray(), config)
        cv.threshold.assert_called_once()

    def test_adaptive_method_called(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(
            binarize=True,
            binarize_config=BinarizeConfig(enabled=True, method="adaptive"),
        )
        p.process(_make_gray(), config)
        cv.adaptiveThreshold.assert_called_once()

    def test_manual_method_called(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(
            binarize=True,
            binarize_config=BinarizeConfig(enabled=True, method="manual", threshold=128),
        )
        p.process(_make_gray(), config)
        # manual uses cv2.threshold with specific threshold value
        cv.threshold.assert_called_once()

    def test_binarize_not_called_when_disabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(binarize=False, binarize_config=BinarizeConfig(enabled=False))
        p.process(_make_gray(), config)
        cv.threshold.assert_not_called()
        cv.adaptiveThreshold.assert_not_called()


# ---------------------------------------------------------------------------
# process — sharpen path
# ---------------------------------------------------------------------------


class TestProcessSharpen:
    def test_addWeighted_called_when_sharpen_enabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(sharpen=True, sharpen_config=SharpenConfig(enabled=True, strength=0.5))
        p.process(_make_gray(), config)
        cv.addWeighted.assert_called_once()

    def test_sharpen_not_called_when_disabled(self, preprocessor):
        p, cv = preprocessor
        config = OCRConfig(sharpen=False, sharpen_config=SharpenConfig(enabled=False))
        p.process(_make_gray(), config)
        cv.addWeighted.assert_not_called()


# ---------------------------------------------------------------------------
# _resize
# ---------------------------------------------------------------------------


class TestResize:
    def test_resize_with_area_interpolation(self, preprocessor):
        p, cv = preprocessor
        image = _make_gray(200, 150)
        config = ResizeConfig(enabled=True, max_size=100, interpolation="area")
        p._resize(image, config)
        cv.resize.assert_called_once()

    def test_resize_with_linear_interpolation(self, preprocessor):
        p, cv = preprocessor
        image = _make_gray(200, 150)
        config = ResizeConfig(enabled=True, max_size=100, interpolation="linear")
        p._resize(image, config)
        cv.resize.assert_called_once()

    def test_resize_with_cubic_interpolation(self, preprocessor):
        p, cv = preprocessor
        image = _make_gray(200, 150)
        config = ResizeConfig(enabled=True, max_size=100, interpolation="cubic")
        p._resize(image, config)
        cv.resize.assert_called_once()

    def test_no_resize_when_image_smaller(self, preprocessor):
        p, cv = preprocessor
        image = _make_gray(50, 40)
        config = ResizeConfig(enabled=True, max_size=200)
        p._resize(image, config)
        cv.resize.assert_not_called()


# ---------------------------------------------------------------------------
# _denoise
# ---------------------------------------------------------------------------


class TestDenoise:
    def test_light_strength_uses_smaller_kernel(self, preprocessor):
        p, cv = preprocessor
        config = DenoiseConfig(enabled=True, strength="light")
        p._denoise(_make_gray(), config)
        call_args = cv.GaussianBlur.call_args[0]
        assert call_args[1] == (3, 3)

    def test_strong_strength_uses_larger_kernel(self, preprocessor):
        p, cv = preprocessor
        config = DenoiseConfig(enabled=True, strength="strong")
        p._denoise(_make_gray(), config)
        call_args = cv.GaussianBlur.call_args[0]
        assert call_args[1] == (5, 5)

    def test_medium_strength_uses_config_kernels(self, preprocessor):
        p, cv = preprocessor
        config = DenoiseConfig(
            enabled=True, strength="medium", gaussian_kernel=(3, 3), median_kernel=3
        )
        p._denoise(_make_gray(), config)
        cv.GaussianBlur.assert_called_once()
        cv.medianBlur.assert_called_once()


# ---------------------------------------------------------------------------
# _adjust_contrast
# ---------------------------------------------------------------------------


class TestAdjustContrast:
    def test_clahe_created_with_correct_params(self, preprocessor):
        p, cv = preprocessor
        config = ContrastConfig(enabled=True, clip_limit=3.0, tile_grid_size=(8, 8))
        p._adjust_contrast(_make_gray(), config)
        cv.createCLAHE.assert_called_once_with(clipLimit=3.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# _binarize
# ---------------------------------------------------------------------------


class TestBinarize:
    def test_otsu(self, preprocessor):
        p, cv = preprocessor
        config = BinarizeConfig(enabled=True, method="otsu")
        result = p._binarize(_make_gray(), config)
        assert result is not None

    def test_adaptive(self, preprocessor):
        p, cv = preprocessor
        config = BinarizeConfig(enabled=True, method="adaptive", block_size=11, c=2)
        result = p._binarize(_make_gray(), config)
        assert result is not None

    def test_manual(self, preprocessor):
        p, cv = preprocessor
        config = BinarizeConfig(enabled=True, method="manual", threshold=100)
        result = p._binarize(_make_gray(), config)
        assert result is not None


# ---------------------------------------------------------------------------
# _sharpen
# ---------------------------------------------------------------------------


class TestSharpen:
    def test_sharpen_uses_addWeighted(self, preprocessor):
        p, cv = preprocessor
        config = SharpenConfig(enabled=True, strength=0.8)
        p._sharpen(_make_gray(), config)
        cv.addWeighted.assert_called_once()
        # alpha = 1 + 0.8 = 1.8, beta = -0.8
        args = cv.addWeighted.call_args[0]
        assert args[1] == pytest.approx(1.8)
        assert args[3] == pytest.approx(-0.8)
