"""
OCR Interactive Widget for Jupyter

Jupyter Notebook/Lab/Colab에서 실시간으로 OCR 파라미터를 조정하고 결과를 확인할 수 있는 위젯.

Usage:
    ```python
    from beanllm.domain.ocr import OCRInteractiveWidget

    widget = OCRInteractiveWidget()
    widget.show("document.jpg")  # 위젯 표시
    ```

Features:
- 실시간 파라미터 조정 (슬라이더)
- 전처리 전/후 비교
- OCR 결과 즉시 확인
- 설정 export/import
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import ipywidgets as widgets
    from IPython.display import display

    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False


class OCRInteractiveWidget:
    """
    Jupyter용 OCR Interactive Widget

    실시간으로 파라미터를 조정하고 OCR 결과를 확인할 수 있는 위젯.

    Features:
    - 슬라이더로 파라미터 조정
    - 실시간 결과 업데이트
    - 전처리 전/후 비교
    - 설정 export

    Example:
        ```python
        from beanllm.domain.ocr import OCRInteractiveWidget

        # Jupyter Notebook에서
        widget = OCRInteractiveWidget()
        widget.show("document.jpg")

        # 조정 후 최종 설정 가져오기
        final_config = widget.get_config()
        ```
    """

    def __init__(self):
        """
        Interactive Widget 초기화

        Raises:
            ImportError: ipywidgets가 설치되지 않은 경우
        """
        if not HAS_IPYWIDGETS:
            raise ImportError(
                "ipywidgets is required for OCRInteractiveWidget. "
                "Install it with: pip install ipywidgets"
            )

        self.image = None
        self.image_path = None
        self._create_widgets()

    def _create_widgets(self):
        """위젯 생성"""
        from .models import OCRConfig

        # === Denoise ===
        self.denoise_enabled = widgets.Checkbox(
            value=True, description="Denoise", style={"description_width": "initial"}
        )
        self.denoise_strength = widgets.SelectionSlider(
            options=["light", "medium", "strong"],
            value="medium",
            description="Strength:",
            disabled=False,
        )

        # === Contrast ===
        self.contrast_enabled = widgets.Checkbox(
            value=True,
            description="Contrast (CLAHE)",
            style={"description_width": "initial"},
        )
        self.clip_limit = widgets.FloatSlider(
            value=2.0,
            min=0.5,
            max=5.0,
            step=0.5,
            description="Clip Limit:",
            continuous_update=False,
        )

        # === Binarize ===
        self.binarize_enabled = widgets.Checkbox(
            value=False, description="Binarize", style={"description_width": "initial"}
        )
        self.binarize_method = widgets.SelectionSlider(
            options=["otsu", "adaptive", "manual"],
            value="otsu",
            description="Method:",
            disabled=True,
        )
        self.threshold = widgets.IntSlider(
            value=127,
            min=0,
            max=255,
            step=1,
            description="Threshold:",
            disabled=True,
        )

        # === Deskew ===
        self.deskew_enabled = widgets.Checkbox(
            value=True, description="Deskew", style={"description_width": "initial"}
        )
        self.angle_threshold = widgets.FloatSlider(
            value=0.5,
            min=0.1,
            max=2.0,
            step=0.1,
            description="Angle Threshold:",
            continuous_update=False,
        )

        # === Sharpen ===
        self.sharpen_enabled = widgets.Checkbox(
            value=False, description="Sharpen", style={"description_width": "initial"}
        )
        self.sharpen_strength = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            description="Strength:",
            continuous_update=False,
        )

        # === Buttons ===
        self.run_button = widgets.Button(
            description="🚀 Run OCR",
            button_style="success",
            tooltip="Run OCR with current settings",
        )
        self.export_button = widgets.Button(
            description="💾 Export Config",
            button_style="info",
            tooltip="Export current configuration",
        )

        # === Output ===
        self.output = widgets.Output()
        self.result_output = widgets.Output()

        # === Event Handlers ===
        self.denoise_enabled.observe(self._on_denoise_toggle, names="value")
        self.binarize_enabled.observe(self._on_binarize_toggle, names="value")
        self.run_button.on_click(self._on_run_click)
        self.export_button.on_click(self._on_export_click)

    def _on_denoise_toggle(self, change):
        """Denoise 토글"""
        self.denoise_strength.disabled = not change["new"]

    def _on_binarize_toggle(self, change):
        """Binarize 토글"""
        enabled = change["new"]
        self.binarize_method.disabled = not enabled
        self.threshold.disabled = not enabled

    def _on_run_click(self, button):
        """OCR 실행 버튼 클릭"""
        if self.image is None:
            with self.result_output:
                print("⚠️ Please load an image first!")
            return

        with self.result_output:
            self.result_output.clear_output(wait=True)
            print("🔄 Running OCR...")

            try:
                from .bean_ocr import beanOCR

                config = self.get_config()
                ocr = beanOCR(config=config)
                result = ocr.recognize(self.image)

                # 결과 출력
                self.result_output.clear_output(wait=True)
                print("=" * 60)
                print("📝 OCR Result")
                print("=" * 60)
                print(f"Engine: {result.engine}")
                print(f"Lines: {result.line_count}")
                print(f"Confidence: {result.confidence:.2%}")
                print(f"Processing Time: {result.processing_time:.2f}s")
                print("-" * 60)
                print("Text:")
                print("-" * 60)
                print(result.text[:500])
                if len(result.text) > 500:
                    print(f"\n... ({len(result.text) - 500} more characters)")
                print("=" * 60)

                # 시각화 (옵션)
                try:
                    import matplotlib.pyplot as plt

                    from .visualizer import OCRVisualizer

                    viz = OCRVisualizer()

                    # 결과 시각화
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    ax.imshow(self.image)

                    if result.lines:
                        for line in result.lines:
                            bbox = line.bbox
                            confidence = line.confidence

                            # 신뢰도 기반 색상
                            color = viz._confidence_to_color(confidence)

                            # BoundingBox
                            from matplotlib.patches import Rectangle

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

                    ax.axis("off")
                    ax.set_title("OCR Result with BoundingBox", fontsize=14, weight="bold")
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                except ImportError:
                    pass

            except Exception as e:
                self.result_output.clear_output(wait=True)
                print(f"❌ Error: {e}")

    def _on_export_click(self, button):
        """설정 Export 버튼 클릭"""
        with self.result_output:
            self.result_output.clear_output(wait=True)
            config = self.get_config()
            print("=" * 60)
            print("📋 Current Configuration")
            print("=" * 60)
            print(f"Denoise: {config.denoise_config.enabled} (strength={config.denoise_config.strength})")
            print(f"Contrast: {config.contrast_config.enabled} (clip_limit={config.contrast_config.clip_limit})")
            print(f"Binarize: {config.binarize_config.enabled} (method={config.binarize_config.method})")
            print(f"Deskew: {config.deskew_config.enabled} (angle_threshold={config.deskew_config.angle_threshold})")
            print(f"Sharpen: {config.sharpen_config.enabled} (strength={config.sharpen_config.strength})")
            print("=" * 60)
            print("\n💡 Use `widget.get_config()` to get OCRConfig object")

    def show(self, image: Union[str, Path, np.ndarray, Image.Image]):
        """
        위젯 표시

        Args:
            image: 입력 이미지 (경로 또는 numpy array)

        Example:
            ```python
            widget = OCRInteractiveWidget()
            widget.show("document.jpg")
            ```
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            self.image_path = str(image)
            pil_image = Image.open(image)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            self.image = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            self.image = image.copy()
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.image = np.array(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # 이미지 미리보기
        with self.output:
            self.output.clear_output(wait=True)
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(self.image)
            ax.axis("off")
            ax.set_title("Input Image", fontsize=14, weight="bold")
            plt.tight_layout()
            plt.show()
            plt.close()

        # 레이아웃
        denoise_box = widgets.VBox([self.denoise_enabled, self.denoise_strength])
        contrast_box = widgets.VBox([self.contrast_enabled, self.clip_limit])
        binarize_box = widgets.VBox(
            [self.binarize_enabled, self.binarize_method, self.threshold]
        )
        deskew_box = widgets.VBox([self.deskew_enabled, self.angle_threshold])
        sharpen_box = widgets.VBox([self.sharpen_enabled, self.sharpen_strength])

        params_box = widgets.VBox(
            [
                widgets.HTML("<h3>⚙️ Parameters</h3>"),
                denoise_box,
                contrast_box,
                binarize_box,
                deskew_box,
                sharpen_box,
            ]
        )

        buttons_box = widgets.HBox([self.run_button, self.export_button])

        main_layout = widgets.VBox(
            [
                widgets.HTML("<h2>🔧 OCR Interactive Tuner</h2>"),
                self.output,
                params_box,
                buttons_box,
                self.result_output,
            ]
        )

        display(main_layout)

    def get_config(self):
        """
        현재 위젯 설정으로 OCRConfig 생성

        Returns:
            OCRConfig: 현재 설정

        Example:
            ```python
            config = widget.get_config()
            ocr = beanOCR(config=config)
            ```
        """
        from .models import (
            BinarizeConfig,
            ContrastConfig,
            DenoiseConfig,
            DeskewConfig,
            OCRConfig,
            SharpenConfig,
        )

        return OCRConfig(
            engine="paddleocr",
            language="auto",
            denoise=self.denoise_enabled.value,
            denoise_config=DenoiseConfig(
                enabled=self.denoise_enabled.value,
                strength=self.denoise_strength.value,
            ),
            contrast_adjustment=self.contrast_enabled.value,
            contrast_config=ContrastConfig(
                enabled=self.contrast_enabled.value,
                clip_limit=self.clip_limit.value,
            ),
            binarize=self.binarize_enabled.value,
            binarize_config=BinarizeConfig(
                enabled=self.binarize_enabled.value,
                method=self.binarize_method.value,
                threshold=self.threshold.value,
            ),
            deskew=self.deskew_enabled.value,
            deskew_config=DeskewConfig(
                enabled=self.deskew_enabled.value,
                angle_threshold=self.angle_threshold.value,
            ),
            sharpen=self.sharpen_enabled.value,
            sharpen_config=SharpenConfig(
                enabled=self.sharpen_enabled.value,
                strength=self.sharpen_strength.value,
            ),
        )

    def __repr__(self) -> str:
        return "OCRInteractiveWidget()"
