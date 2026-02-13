"""
OCR 시각화 도구

OCR 전처리 과정과 결과를 시각화하여 파라미터 튜닝을 돕는 유틸리티.

Features:
- 전처리 단계별 시각화 (Before → After 비교)
- OCR 결과 BoundingBox 오버레이
- 신뢰도 기반 색상 매핑
- 저장/표시 옵션
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from beanllm.utils.logging import get_logger

from .models import (
    BinarizeConfig,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    OCRConfig,
    OCRResult,
    ResizeConfig,
    SharpenConfig,
)

# Type alias for image input
ImageInput = Union[np.ndarray, str, Path]

logger = get_logger(__name__)

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
                "matplotlib is required for OCRVisualizer. Install it with: pip install matplotlib"
            )

    # === Image Loading (타입 안전 헬퍼) ===

    def _load_image_as_array(self, image: ImageInput) -> np.ndarray:
        """이미지를 numpy array로 로드 (타입 안전)"""
        if isinstance(image, (str, Path)):
            opened_image = Image.open(image)
            rgb_image = opened_image.convert("RGB") if opened_image.mode != "RGB" else opened_image
            result: np.ndarray = np.array(rgb_image)
            return result
        elif isinstance(image, np.ndarray):
            copied: np.ndarray = image.copy()
            return copied
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    # === Preprocessing Step Collectors (메서드 체인으로 흐름 명확화) ===

    def _collect_preprocessing_steps(
        self,
        image: np.ndarray,
        config: OCRConfig,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """전처리 단계별 이미지 수집 (각 단계를 메서드로 분리)"""
        from .preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        steps: List[np.ndarray] = [image]
        titles: List[str] = ["Original"]

        # Grayscale 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        current = gray.copy()

        # 각 전처리 단계 (Optional config 안전 접근)
        current, step_result = self._apply_resize_step(current, config.resize_config, preprocessor)
        if step_result:
            steps.append(step_result[0])
            titles.append(step_result[1])

        current, step_result = self._apply_denoise_step(
            current, config.denoise_config, preprocessor
        )
        if step_result:
            steps.append(step_result[0])
            titles.append(step_result[1])

        current, step_result = self._apply_contrast_step(
            current, config.contrast_config, preprocessor
        )
        if step_result:
            steps.append(step_result[0])
            titles.append(step_result[1])

        current, step_result = self._apply_deskew_step(current, config.deskew_config, preprocessor)
        if step_result:
            steps.append(step_result[0])
            titles.append(step_result[1])

        current, step_result = self._apply_binarize_step(
            current, config.binarize_config, preprocessor
        )
        if step_result:
            steps.append(step_result[0])
            titles.append(step_result[1])

        current, step_result = self._apply_sharpen_step(
            current, config.sharpen_config, preprocessor
        )
        if step_result:
            steps.append(step_result[0])
            titles.append(step_result[1])

        return steps, titles

    def _apply_resize_step(
        self, current: np.ndarray, config: Optional[ResizeConfig], preprocessor: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, str]]]:
        """Resize 단계 적용"""
        if config is None or not config.enabled or not config.max_size:
            return current, None
        current = preprocessor._resize(current, config)
        rgb_image = (
            cv2.cvtColor(current, cv2.COLOR_GRAY2RGB) if len(current.shape) == 2 else current
        )
        return current, (rgb_image, f"Resized ({config.max_size}px)")

    def _apply_denoise_step(
        self, current: np.ndarray, config: Optional[DenoiseConfig], preprocessor: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, str]]]:
        """Denoise 단계 적용"""
        if config is None or not config.enabled:
            return current, None
        current = preprocessor._denoise(current, config)
        rgb_image = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
        return current, (rgb_image, f"Denoised ({config.strength})")

    def _apply_contrast_step(
        self, current: np.ndarray, config: Optional[ContrastConfig], preprocessor: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, str]]]:
        """Contrast 단계 적용"""
        if config is None or not config.enabled:
            return current, None
        current = preprocessor._adjust_contrast(current, config)
        rgb_image = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
        return current, (rgb_image, f"Contrast (CLAHE {config.clip_limit})")

    def _apply_deskew_step(
        self, current: np.ndarray, config: Optional[DeskewConfig], preprocessor: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, str]]]:
        """Deskew 단계 적용"""
        if config is None or not config.enabled:
            return current, None
        current = preprocessor._deskew(current, config)
        rgb_image = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
        return current, (rgb_image, "Deskewed")

    def _apply_binarize_step(
        self, current: np.ndarray, config: Optional[BinarizeConfig], preprocessor: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, str]]]:
        """Binarize 단계 적용"""
        if config is None or not config.enabled:
            return current, None
        current = preprocessor._binarize(current, config)
        rgb_image = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
        return current, (rgb_image, f"Binarized ({config.method})")

    def _apply_sharpen_step(
        self, current: np.ndarray, config: Optional[SharpenConfig], preprocessor: Any
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, str]]]:
        """Sharpen 단계 적용"""
        if config is None or not config.enabled:
            return current, None
        current = preprocessor._sharpen(current, config)
        rgb_image = cv2.cvtColor(current, cv2.COLOR_GRAY2RGB)
        return current, (rgb_image, f"Sharpened ({config.strength:.1f})")

    def show_preprocessing_steps(
        self,
        image: ImageInput,
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
        self._require_cv2()

        # 이미지 로드 (헬퍼 메서드 사용)
        image_array = self._load_image_as_array(image)

        # 전처리 단계 수집 (헬퍼 메서드 사용)
        steps, titles = self._collect_preprocessing_steps(image_array, config)

        # 시각화 (헬퍼 메서드 사용)
        self._render_preprocessing_figure(steps, titles, save_path)

    def _require_cv2(self) -> None:
        """OpenCV 필수 확인"""
        if not HAS_CV2:
            raise ImportError(
                "opencv-python is required for preprocessing visualization. "
                "Install it with: pip install opencv-python"
            )

    def _render_preprocessing_figure(
        self,
        steps: List[np.ndarray],
        titles: List[str],
        save_path: Optional[Union[str, Path]],
    ) -> None:
        """전처리 단계 시각화 렌더링"""
        n_steps = len(steps)
        fig, axes = plt.subplots(2, (n_steps + 1) // 2, figsize=(18, 8))
        axes_flat = axes.flatten()

        for idx, (step_img, title) in enumerate(zip(steps, titles)):
            axes_flat[idx].imshow(step_img)
            axes_flat[idx].set_title(title, fontsize=12, weight="bold")
            axes_flat[idx].axis("off")

        # 빈 subplot 숨기기
        for idx in range(n_steps, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.suptitle("OCR Preprocessing Pipeline", fontsize=16, weight="bold", y=0.98)
        plt.tight_layout()

        self._save_or_show(save_path, "Preprocessing visualization")

    def _save_or_show(self, save_path: Optional[Union[str, Path]], description: str) -> None:
        """저장 또는 표시"""
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"{description} saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def show_result(
        self,
        image: ImageInput,
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
        # 이미지 로드 (헬퍼 메서드 사용)
        image_array = self._load_image_as_array(image)

        # 시각화 렌더링
        self._render_result_figure(image_array, result, show_bbox, show_confidence, save_path)

    def _render_result_figure(
        self,
        image: np.ndarray,
        result: OCRResult,
        show_bbox: bool,
        show_confidence: bool,
        save_path: Optional[Union[str, Path]],
    ) -> None:
        """OCR 결과 시각화 렌더링"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        if show_bbox and result.lines:
            self._draw_bounding_boxes(ax, result, show_confidence)

        ax.axis("off")

        # 제목 및 통계
        title = self._build_result_title(result)
        plt.title(title, fontsize=14, weight="bold", pad=10)

        plt.tight_layout()

        self._save_or_show(save_path, "OCR result visualization")

    def _draw_bounding_boxes(self, ax: Any, result: OCRResult, show_confidence: bool) -> None:
        """BoundingBox 그리기"""
        for line in result.lines:
            bbox = line.bbox
            confidence = line.confidence

            # 신뢰도 기반 색상 매핑 (빨강:낮음 → 초록:높음)
            color = self._confidence_to_color(confidence) if show_confidence else "green"

            # BoundingBox 그리기
            rect = Rectangle(
                (bbox.x0, bbox.y0),
                bbox.width,
                bbox.height,
                linewidth=2,
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

    def _build_result_title(self, result: OCRResult) -> str:
        """결과 제목 생성"""
        title = f"OCR Result - {result.engine}"
        if result.lines:
            title += f" | {len(result.lines)} lines | Avg Confidence: {result.confidence:.2%}"
        return title

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
