"""
Image Preprocessor 테스트

Note: opencv-python이 설치되지 않은 경우 대부분의 테스트는 skip됩니다.
      설치: pip install opencv-python
"""

import numpy as np
import pytest

from beanllm.domain.ocr.models import OCRConfig

# opencv-python 설치 여부 체크
try:
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ImagePreprocessor import (HAS_CV2가 True일 때만)
if HAS_CV2:
    from beanllm.domain.ocr.preprocessing import ImagePreprocessor
else:
    ImagePreprocessor = None  # type: ignore

skip_without_cv2 = pytest.mark.skipif(not HAS_CV2, reason="opencv-python not installed")


class TestImagePreprocessorImport:
    """ImagePreprocessor import 테스트"""

    def test_preprocessor_import_without_cv2(self):
        """opencv-python 없이 import 시도 (의존성 체크 테스트)"""
        if not HAS_CV2:
            from beanllm.domain.ocr.preprocessing import ImagePreprocessor

            with pytest.raises(ImportError, match="opencv-python is required"):
                ImagePreprocessor()
        else:
            from beanllm.domain.ocr.preprocessing import ImagePreprocessor

            preprocessor = ImagePreprocessor()
            assert preprocessor is not None


@skip_without_cv2
class TestImagePreprocessor:
    """ImagePreprocessor 테스트"""

    def test_preprocessor_initialization(self):
        """전처리기 초기화 테스트"""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None

    def test_preprocessor_repr(self):
        """__repr__ 테스트"""
        preprocessor = ImagePreprocessor()
        repr_str = repr(preprocessor)
        assert "ImagePreprocessor" in repr_str

    def test_process_without_preprocessing(self):
        """전처리 비활성화 시 원본 반환"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(enable_preprocessing=False)

        # 테스트 이미지
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = preprocessor.process(image, config)

        # 원본과 동일해야 함
        assert np.array_equal(result, image)

    def test_process_with_denoise(self):
        """노이즈 제거 테스트"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            enable_preprocessing=True,
            denoise=True,
            contrast_adjustment=False,
            binarize=False,
            deskew=False,
            sharpen=False,
        )

        # 노이즈가 있는 이미지
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = preprocessor.process(image, config)

        # 결과가 RGB 형식이어야 함
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_process_with_contrast(self):
        """대비 조정 테스트"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            enable_preprocessing=True,
            denoise=False,
            contrast_adjustment=True,
            binarize=False,
            deskew=False,
            sharpen=False,
        )

        # 낮은 대비 이미지
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = preprocessor.process(image, config)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_process_with_binarize(self):
        """이진화 테스트"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            enable_preprocessing=True,
            denoise=False,
            contrast_adjustment=False,
            binarize=True,
            deskew=False,
            sharpen=False,
        )

        # 그레이스케일 이미지
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = preprocessor.process(image, config)

        # 이진화된 이미지 (RGB로 변환됨)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_process_with_resize(self):
        """크기 조정 테스트"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            enable_preprocessing=True,
            max_image_size=50,  # 50px로 축소
            denoise=False,
            contrast_adjustment=False,
            binarize=False,
            deskew=False,
            sharpen=False,
        )

        # 큰 이미지
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        result = preprocessor.process(image, config)

        # 크기가 줄어들어야 함 (max_dim = 50)
        max_dim = max(result.shape[:2])
        assert max_dim <= 50

    def test_process_with_all_options(self):
        """모든 전처리 옵션 활성화 테스트"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            enable_preprocessing=True,
            denoise=True,
            contrast_adjustment=True,
            binarize=True,
            deskew=True,
            sharpen=True,
        )

        # 테스트 이미지
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = preprocessor.process(image, config)

        # 결과가 RGB 형식이어야 함
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_denoise_method(self):
        """노이즈 제거 메서드 테스트"""
        preprocessor = ImagePreprocessor()

        # Grayscale 이미지
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = preprocessor._denoise(image)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_adjust_contrast_method(self):
        """대비 조정 메서드 테스트"""
        preprocessor = ImagePreprocessor()

        # Grayscale 이미지
        image = np.ones((100, 100), dtype=np.uint8) * 128

        result = preprocessor._adjust_contrast(image)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_binarize_method(self):
        """이진화 메서드 테스트"""
        preprocessor = ImagePreprocessor()

        # Grayscale 이미지
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = preprocessor._binarize(image)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8
        # 이진화된 이미지는 0 또는 255만 가져야 함
        assert set(np.unique(result)).issubset({0, 255})

    def test_sharpen_method(self):
        """선명화 메서드 테스트"""
        preprocessor = ImagePreprocessor()

        # Grayscale 이미지
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = preprocessor._sharpen(image)

        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_resize_method(self):
        """크기 조정 메서드 테스트"""
        preprocessor = ImagePreprocessor()

        # 큰 이미지
        image = np.random.randint(0, 255, (200, 300), dtype=np.uint8)

        result = preprocessor._resize(image, max_size=100)

        # 최대 차원이 100 이하여야 함
        max_dim = max(result.shape[:2])
        assert max_dim <= 100

        # 비율이 유지되어야 함
        original_ratio = 200 / 300
        result_ratio = result.shape[0] / result.shape[1]
        assert abs(original_ratio - result_ratio) < 0.01

    def test_resize_small_image(self):
        """작은 이미지는 크기 조정 안 함"""
        preprocessor = ImagePreprocessor()

        # 작은 이미지
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        result = preprocessor._resize(image, max_size=100)

        # 크기가 그대로여야 함
        assert result.shape == (50, 50)

    def test_deskew_method(self):
        """기울기 보정 메서드 테스트"""
        preprocessor = ImagePreprocessor()

        # 텍스트가 있는 이미지 (간단한 직사각형)
        image = np.zeros((100, 200), dtype=np.uint8)
        image[40:60, 50:150] = 255  # 흰색 직사각형

        result = preprocessor._deskew(image)

        # 결과가 같은 크기여야 함
        assert result.shape == (100, 200)
        assert result.dtype == np.uint8

    def test_process_grayscale_input(self):
        """Grayscale 입력 이미지 처리"""
        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            enable_preprocessing=True,
            denoise=True,
            contrast_adjustment=True,
        )

        # Grayscale 이미지
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = preprocessor.process(image, config)

        # RGB로 변환되어야 함
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
