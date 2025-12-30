"""
beanOCR - Advanced OCR Module

고급 OCR 기능을 제공하는 모듈:
- 7개 OCR 엔진 지원 (PaddleOCR, EasyOCR, TrOCR, Nougat, Surya, Tesseract, Cloud API)
- 이미지 전처리 파이프라인 (노이즈 제거, 대비 조정, 회전 보정)
- LLM 후처리로 98%+ 정확도
- Hybrid 전략으로 95% 비용 절감
- 다국어 지원 (80+ languages, 한글 최적화)

Example:
    ```python
    from beanllm.domain.ocr import beanOCR, OCRConfig

    # 기본 사용
    ocr = beanOCR(engine="paddleocr", language="ko")
    result = ocr.recognize("scanned_image.jpg")
    print(result.text)
    print(f"Confidence: {result.confidence:.2%}")

    # LLM 후처리 활성화
    ocr = beanOCR(
        engine="paddleocr",
        enable_llm_postprocessing=True,
        llm_model="gpt-4o-mini"
    )
    result = ocr.recognize("noisy_image.jpg")

    # PDF 페이지 OCR
    result = ocr.recognize_pdf_page("document.pdf", page_num=0)
    ```
"""

from .bean_ocr import beanOCR
from .experiment import OCRExperiment
from .grid_search import GridSearchTuner
from .interactive_widget import OCRInteractiveWidget
from .models import (
    BinarizeConfig,
    BoundingBox,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    OCRConfig,
    OCRResult,
    OCRTextLine,
    ResizeConfig,
    SharpenConfig,
)
from .postprocessing import LLMPostprocessor
from .presets import ConfigPresets
from .visualizer import OCRVisualizer

__all__ = [
    "beanOCR",
    "BoundingBox",
    "OCRTextLine",
    "OCRResult",
    "OCRConfig",
    "DenoiseConfig",
    "ContrastConfig",
    "BinarizeConfig",
    "DeskewConfig",
    "SharpenConfig",
    "ResizeConfig",
    "OCRVisualizer",
    "ConfigPresets",
    "OCRExperiment",
    "GridSearchTuner",
    "OCRInteractiveWidget",
    "LLMPostprocessor",
]
