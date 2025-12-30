"""
OCR 엔진 모듈

10개 OCR 엔진 구현:
- PaddleOCR: 메인 엔진 (90-96% 정확도)
- EasyOCR: 대체 엔진
- TrOCR: 손글씨 전문
- Nougat: 학술 논문 (수식, 표)
- Surya: 복잡한 레이아웃
- Tesseract: Fallback
- Cloud API: Google Vision, AWS Textract
- Qwen2.5-VL: 오픈소스 최고 성능 (2024-2025)
- MiniCPM-o 2.6: OCRBench 1위 (2024-2025)
- DeepSeek-OCR: 토큰 압축, 효율적 (2024-2025)
"""

from .base import BaseOCREngine

__all__ = ["BaseOCREngine"]

# PaddleOCR 엔진 (optional dependency)
try:
    from .paddleocr_engine import PaddleOCREngine

    __all__.append("PaddleOCREngine")
except ImportError:
    pass

# EasyOCR 엔진 (optional dependency)
try:
    from .easyocr_engine import EasyOCREngine

    __all__.append("EasyOCREngine")
except ImportError:
    pass

# Tesseract 엔진 (optional dependency)
try:
    from .tesseract_engine import TesseractEngine

    __all__.append("TesseractEngine")
except ImportError:
    pass

# TrOCR 엔진 (optional dependency)
try:
    from .trocr_engine import TrOCREngine

    __all__.append("TrOCREngine")
except ImportError:
    pass

# Nougat 엔진 (optional dependency)
try:
    from .nougat_engine import NougatEngine

    __all__.append("NougatEngine")
except ImportError:
    pass

# Surya 엔진 (optional dependency)
try:
    from .surya_engine import SuryaEngine

    __all__.append("SuryaEngine")
except ImportError:
    pass

# Cloud OCR 엔진 (optional dependency)
try:
    from .cloud_engine import CloudOCREngine

    __all__.append("CloudOCREngine")
except ImportError:
    pass

# Qwen2.5-VL 엔진 (optional dependency)
try:
    from .qwen2vl_engine import Qwen2VLEngine

    __all__.append("Qwen2VLEngine")
except ImportError:
    pass

# MiniCPM-o 엔진 (optional dependency)
try:
    from .minicpm_engine import MiniCPMEngine

    __all__.append("MiniCPMEngine")
except ImportError:
    pass

# DeepSeek-OCR 엔진 (optional dependency)
try:
    from .deepseek_ocr_engine import DeepSeekOCREngine

    __all__.append("DeepSeekOCREngine")
except ImportError:
    pass
