"""
Image Preprocessor

OCR 정확도를 높이기 위한 이미지 전처리 파이프라인.

Features:
- 노이즈 제거 (Gaussian blur, Median filter)
- 대비 조정 (Histogram equalization, CLAHE)
- 이진화 (Otsu, Adaptive)
- 기울기 보정 (Deskew)
- 크기 조정 (Resize)
- 선명화 (Sharpen)
"""

import logging
from typing import Optional

import numpy as np

from ..models import OCRConfig

logger = logging.getLogger(__name__)

# opencv-python 설치 여부 체크
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImagePreprocessor:
    """
    이미지 전처리 파이프라인

    OCR 정확도를 높이기 위한 다양한 전처리 기법 제공.

    Features:
    - 노이즈 제거
    - 대비 조정
    - 이진화
    - 기울기 보정
    - 크기 조정
    - 선명화

    Example:
        ```python
        from beanllm.domain.ocr.preprocessing import ImagePreprocessor
        from beanllm.domain.ocr.models import OCRConfig
        import numpy as np

        preprocessor = ImagePreprocessor()
        config = OCRConfig(
            denoise=True,
            contrast_adjustment=True,
            binarize=True
        )

        # 이미지 전처리
        processed_image = preprocessor.process(image, config)
        ```
    """

    def __init__(self):
        """
        이미지 전처리기 초기화

        Raises:
            ImportError: opencv-python이 설치되지 않은 경우
        """
        if not HAS_CV2:
            raise ImportError(
                "opencv-python is required for ImagePreprocessor. "
                "Install it with: pip install opencv-python"
            )

    def process(self, image: np.ndarray, config: OCRConfig) -> np.ndarray:
        """
        이미지 전처리 파이프라인 실행

        Args:
            image: 입력 이미지 (numpy array, RGB)
            config: OCR 설정 (전처리 옵션 포함)

        Returns:
            np.ndarray: 전처리된 이미지

        Example:
            ```python
            processed = preprocessor.process(image, config)
            ```
        """
        if not config.enable_preprocessing:
            return image

        # RGB → Grayscale (전처리는 grayscale에서 진행)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 1. 크기 조정 (먼저 수행)
        if config.max_image_size:
            gray = self._resize(gray, config.max_image_size)

        # 2. 노이즈 제거
        if config.denoise:
            gray = self._denoise(gray)

        # 3. 대비 조정
        if config.contrast_adjustment:
            gray = self._adjust_contrast(gray)

        # 4. 기울기 보정
        if config.deskew:
            gray = self._deskew(gray)

        # 5. 이진화
        if config.binarize:
            gray = self._binarize(gray)

        # 6. 선명화
        if config.sharpen:
            gray = self._sharpen(gray)

        # Grayscale → RGB (OCR 엔진은 RGB를 받음)
        if len(gray.shape) == 2:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            result = gray

        return result

    def _resize(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """
        이미지 크기 조정

        Args:
            image: 입력 이미지
            max_size: 최대 크기 (픽셀)

        Returns:
            np.ndarray: 크기 조정된 이미지
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)

        if max_dim > max_size:
            scale = max_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        return image

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        노이즈 제거

        Gaussian blur와 Median filter를 조합하여 노이즈 제거.

        Args:
            image: 입력 이미지 (grayscale)

        Returns:
            np.ndarray: 노이즈 제거된 이미지
        """
        # Gaussian blur (가벼운 블러)
        denoised = cv2.GaussianBlur(image, (3, 3), 0)

        # Median filter (salt-and-pepper 노이즈 제거)
        denoised = cv2.medianBlur(denoised, 3)

        return denoised

    def _adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        대비 조정

        CLAHE (Contrast Limited Adaptive Histogram Equalization) 사용.

        Args:
            image: 입력 이미지 (grayscale)

        Returns:
            np.ndarray: 대비 조정된 이미지
        """
        # CLAHE (Adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        return enhanced

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        이진화

        Otsu's method를 사용한 자동 임계값 이진화.

        Args:
            image: 입력 이미지 (grayscale)

        Returns:
            np.ndarray: 이진화된 이미지
        """
        # Otsu's binarization
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """
        기울기 보정

        Hough 변환을 사용하여 텍스트 라인의 각도를 감지하고 보정.

        Args:
            image: 입력 이미지 (grayscale)

        Returns:
            np.ndarray: 기울기 보정된 이미지
        """
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is None:
            return image

        # 각도 계산
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            # 수평선에 가까운 각도만 사용
            if abs(angle) < 45:
                angles.append(angle)

        if not angles:
            return image

        # 중간값 각도 사용 (outlier 제거)
        median_angle = np.median(angles)

        # 회전 변환 (작은 각도만 보정)
        if abs(median_angle) > 0.5:  # 0.5도 이상만 보정
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            logger.debug(f"Deskewed image by {median_angle:.2f} degrees")
            return rotated

        return image

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 선명화

        Unsharp masking을 사용한 선명화.

        Args:
            image: 입력 이미지 (grayscale)

        Returns:
            np.ndarray: 선명화된 이미지
        """
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)

        # Unsharp masking
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        return sharpened

    def __repr__(self) -> str:
        return "ImagePreprocessor()"
