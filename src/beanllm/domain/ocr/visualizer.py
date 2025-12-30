"""
OCR 시각화 도구

OCR 전처리 과정과 결과를 시각화하여 파라미터 튜닝을 돕는 유틸리티.

Features:
- 전처리 단계별 시각화 (Before → After 비교)
- OCR 결과 BoundingBox 오버레이
- 신뢰도 기반 색상 매핑
- 저장/표시 옵션
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from .models import OCRConfig, OCRResult

logger = logging.getLogger(__name__)

# matplotlib 설치 여부 체크
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# opencv-python 설치 여부 체크
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class OCRVisualizer:
    """
    OCR 시각화 도구

    전처리 과정과 OCR 결과를 시각적으로 확인할 수 있는 유틸리티.

    Features:
    - 전처리 단계별 시각화
    - OCR 결과 + BoundingBox 오버레이
    - 신뢰도 기반 색상 매핑 (빨강:낮음 → 초록:높음)

    Example:
        ```python
        from beanllm.domain.ocr import OCRVisualizer, beanOCR, OCRConfig

        viz = OCRVisualizer()

        # 전처리 단계별 시각화
        config = OCRConfig(denoise=True, binarize=True)
        viz.show_preprocessing_steps(image, config)  # 화면에 표시

        # OCR 결과 시각화
        ocr = beanOCR()
        result = ocr.recognize(image)
        viz.show_result(image, result, show_confidence=True)  # 신뢰도 포함

        # 저장
        viz.show_result(image, result, save_path="result.png")
        ```
    """

    def __init__(self):
        """
        시각화 도구 초기화

        Raises:
            ImportError: matplotlib이 설치되지 않은 경우
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for OCRVisualizer. "
                "Install it with: pip install matplotlib"
            )

    def show_preprocessing_steps(
        self,
        image: Union[np.ndarray, str, Path],
        config: OCRConfig,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        전처리 단계별 과정을 시각화

        원본 → Resize → Denoise → Contrast → Deskew → Binarize → Sharpen

        Args:
            image: 입력 이미지 (numpy array 또는 경로)
            config: OCR 설정 (전처리 옵션)
            save_path: 저장 경로 (None이면 화면에 표시)

        Example:
            ```python
            config = OCRConfig(
                denoise=True,
                contrast_adjustment=True,
                binarize=True
            )
            viz.show_preprocessing_steps(image, config)
            ```
        """
        if not HAS_CV2:
            raise ImportError(
                "opencv-python is required for preprocessing visualization. "
                "Install it with: pip install opencv-python"
            )

        from .preprocessing import ImagePreprocessor

        # 이미지 로드
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            image = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 전처리 단계별 이미지 수집
        preprocessor = ImagePreprocessor()
        steps = []
        titles = []

        # 0. 원본
        steps.append(image)
        titles.append("Original")

        # 처리할 이미지 준비
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        current = gray.copy()

        # 1. Resize
        if config.resize_config.enabled and config.resize_config.max_size:
            current = preprocessor._resize(current, config.resize_config)
            steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB) if len(current.shape) == 2 else current)
            titles.append(f"Resized ({config.resize_config.max_size}px)")

        # 2. Denoise
        if config.denoise_config.enabled:
            current = preprocessor._denoise(current, config.denoise_config)
            steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
            titles.append(f"Denoised ({config.denoise_config.strength})")

        # 3. Contrast
        if config.contrast_config.enabled:
            current = preprocessor._adjust_contrast(current, config.contrast_config)
            steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
            titles.append(f"Contrast (CLAHE {config.contrast_config.clip_limit})")

        # 4. Deskew
        if config.deskew_config.enabled:
            current = preprocessor._deskew(current, config.deskew_config)
            steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
            titles.append("Deskewed")

        # 5. Binarize
        if config.binarize_config.enabled:
            current = preprocessor._binarize(current, config.binarize_config)
            steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
            titles.append(f"Binarized ({config.binarize_config.method})")

        # 6. Sharpen
        if config.sharpen_config.enabled:
            current = preprocessor._sharpen(current, config.sharpen_config)
            steps.append(cv2.cvtColor(current, cv2.COLOR_GRAY2RGB))
            titles.append(f"Sharpened ({config.sharpen_config.strength:.1f})")

        # 시각화
        n_steps = len(steps)
        fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(18, 8))
        axes = axes.flatten()

        for idx, (step_img, title) in enumerate(zip(steps, titles)):
            axes[idx].imshow(step_img)
            axes[idx].set_title(title, fontsize=12, weight="bold")
            axes[idx].axis("off")

        # 빈 subplot 숨기기
        for idx in range(n_steps, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("OCR Preprocessing Pipeline", fontsize=16, weight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Preprocessing visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def show_result(
        self,
        image: Union[np.ndarray, str, Path],
        result: OCRResult,
        show_bbox: bool = True,
        show_confidence: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        OCR 결과를 이미지에 오버레이하여 시각화

        Args:
            image: 원본 이미지
            result: OCR 결과
            show_bbox: BoundingBox 표시 여부
            show_confidence: 신뢰도 표시 여부 (색상 매핑)
            save_path: 저장 경로 (None이면 화면에 표시)

        Example:
            ```python
            ocr = beanOCR()
            result = ocr.recognize("document.jpg")

            viz = OCRVisualizer()

            # BoundingBox + 신뢰도 색상
            viz.show_result("document.jpg", result, show_confidence=True)

            # BoundingBox만
            viz.show_result("document.jpg", result, show_confidence=False)

            # 저장
            viz.show_result("document.jpg", result, save_path="result.png")
            ```
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            image = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 시각화
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        if show_bbox and result.lines:
            for line in result.lines:
                bbox = line.bbox
                confidence = line.confidence

                # 신뢰도 기반 색상 매핑 (빨강:낮음 → 초록:높음)
                if show_confidence:
                    color = self._confidence_to_color(confidence)
                    linewidth = 2
                else:
                    color = "green"
                    linewidth = 2

                # BoundingBox 그리기
                rect = Rectangle(
                    (bbox.x0, bbox.y0),
                    bbox.width,
                    bbox.height,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor="none",
                    alpha=0.8,
                )
                ax.add_patch(rect)

                # 신뢰도 텍스트 표시
                if show_confidence:
                    ax.text(
                        bbox.x0,
                        bbox.y0 - 5,
                        f"{confidence:.2f}",
                        color=color,
                        fontsize=8,
                        weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                    )

        ax.axis("off")

        # 제목 및 통계
        title = f"OCR Result - {result.engine}"
        if result.lines:
            title += f" | {len(result.lines)} lines | Avg Confidence: {result.confidence:.2%}"
        plt.title(title, fontsize=14, weight="bold", pad=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"OCR result visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def _confidence_to_color(self, confidence: float) -> str:
        """
        신뢰도를 색상으로 매핑

        0.0-0.5: 빨강 (낮음)
        0.5-0.8: 주황/노랑 (중간)
        0.8-1.0: 초록 (높음)

        Args:
            confidence: 신뢰도 (0.0-1.0)

        Returns:
            str: 색상 (hex 코드)
        """
        if confidence < 0.5:
            return "#FF3333"  # 빨강
        elif confidence < 0.7:
            return "#FF9933"  # 주황
        elif confidence < 0.85:
            return "#FFCC33"  # 노랑
        else:
            return "#33CC33"  # 초록

    def __repr__(self) -> str:
        return "OCRVisualizer()"
