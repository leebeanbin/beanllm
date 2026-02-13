"""
OCR Engine Factory - Engine creation logic for beanOCR.
"""
from __future__ import annotations

from beanllm.domain.ocr.engines.base import BaseOCREngine


def create_ocr_engine(engine_name: str, use_gpu: bool = False) -> BaseOCREngine:
    """
    OCR 엔진 생성

    Args:
        engine_name: 엔진 이름
        use_gpu: GPU 사용 여부 (일부 엔진만 적용)

    Returns:
        BaseOCREngine: OCR 엔진 인스턴스

    Raises:
        ImportError: 엔진 의존성이 설치되지 않은 경우
        NotImplementedError: 지원하지 않는 엔진
    """
    if engine_name == "paddleocr":
        try:
            from beanllm.domain.ocr.engines.paddleocr_engine import PaddleOCREngine

            return PaddleOCREngine()
        except ImportError as e:
            raise ImportError(
                f"PaddleOCR is required for engine '{engine_name}'. "
                f"Install it with: pip install paddleocr"
            ) from e

    elif engine_name == "easyocr":
        try:
            from beanllm.domain.ocr.engines.easyocr_engine import EasyOCREngine

            return EasyOCREngine()
        except ImportError as e:
            raise ImportError(
                f"EasyOCR is required for engine '{engine_name}'. "
                f"Install it with: pip install easyocr"
            ) from e

    elif engine_name == "tesseract":
        try:
            from beanllm.domain.ocr.engines.tesseract_engine import TesseractEngine

            return TesseractEngine()
        except ImportError as e:
            raise ImportError(
                f"pytesseract is required for engine '{engine_name}'. "
                f"Install it with: pip install pytesseract\n"
                f"Also install Tesseract OCR: brew install tesseract (macOS)"
            ) from e

    elif engine_name == "trocr":
        try:
            from beanllm.domain.ocr.engines.trocr_engine import TrOCREngine

            return TrOCREngine()
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch"
            ) from e

    elif engine_name == "nougat":
        try:
            from beanllm.domain.ocr.engines.nougat_engine import NougatEngine

            return NougatEngine()
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch"
            ) from e

    elif engine_name == "surya":
        try:
            from beanllm.domain.ocr.engines.surya_engine import SuryaEngine

            return SuryaEngine()
        except ImportError as e:
            raise ImportError(
                f"surya-ocr and torch are required for engine '{engine_name}'. "
                f"Install them with: pip install surya-ocr torch"
            ) from e

    elif engine_name == "cloud-google":
        try:
            from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

            return CloudOCREngine(provider="google")
        except ImportError as e:
            raise ImportError(
                f"google-cloud-vision is required for engine '{engine_name}'. "
                f"Install it with: pip install google-cloud-vision"
            ) from e

    elif engine_name == "cloud-aws":
        try:
            from beanllm.domain.ocr.engines.cloud_engine import CloudOCREngine

            return CloudOCREngine(provider="aws")
        except ImportError as e:
            raise ImportError(
                f"boto3 is required for engine '{engine_name}'. "
                f"Install it with: pip install boto3"
            ) from e

    elif engine_name in ["qwen2vl", "qwen2vl-2b"]:
        try:
            from beanllm.domain.ocr.engines.qwen2vl_engine import Qwen2VLEngine

            return Qwen2VLEngine(model_size="2b", use_gpu=use_gpu)
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch pillow qwen-vl-utils"
            ) from e

    elif engine_name == "qwen2vl-7b":
        try:
            from beanllm.domain.ocr.engines.qwen2vl_engine import Qwen2VLEngine

            return Qwen2VLEngine(model_size="7b", use_gpu=use_gpu)
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch pillow qwen-vl-utils"
            ) from e

    elif engine_name == "qwen2vl-72b":
        try:
            from beanllm.domain.ocr.engines.qwen2vl_engine import Qwen2VLEngine

            return Qwen2VLEngine(model_size="72b", use_gpu=use_gpu)
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch pillow qwen-vl-utils"
            ) from e

    elif engine_name == "minicpm":
        try:
            from beanllm.domain.ocr.engines.minicpm_engine import MiniCPMEngine

            return MiniCPMEngine(use_gpu=use_gpu)
        except ImportError as e:
            raise ImportError(
                f"transformers, torch, and pillow are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch pillow timm"
            ) from e

    elif engine_name == "deepseek-ocr":
        try:
            from beanllm.domain.ocr.engines.deepseek_ocr_engine import DeepSeekOCREngine

            return DeepSeekOCREngine(use_gpu=use_gpu)
        except ImportError as e:
            raise ImportError(
                f"transformers, torch, and pillow are required for engine '{engine_name}'. "
                f"Install them with: pip install transformers torch pillow"
            ) from e

    raise NotImplementedError(
        f"Engine '{engine_name}' is not yet implemented. "
        f"Currently supported: paddleocr, easyocr, tesseract, trocr, nougat, surya, "
        f"cloud-google, cloud-aws, qwen2vl-2b, qwen2vl-7b, qwen2vl-72b, minicpm, deepseek-ocr"
    )
