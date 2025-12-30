"""
OCR 엔진 모듈

7개 OCR 엔진 구현:
- PaddleOCR: 메인 엔진 (90-96% 정확도)
- EasyOCR: 대체 엔진
- TrOCR: 손글씨 전문
- Nougat: 학술 논문 (수식, 표)
- Surya: 복잡한 레이아웃
- Tesseract: Fallback
- Cloud API: Google Vision, AWS Textract 등
"""

from .base import BaseOCREngine

# PaddleOCR 엔진 (optional dependency)
try:
    from .paddleocr_engine import PaddleOCREngine

    __all__ = ["BaseOCREngine", "PaddleOCREngine"]
except ImportError:
    __all__ = ["BaseOCREngine"]
