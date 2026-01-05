"""
TrOCR Engine

TrOCR (Transformer-based OCR) 엔진 - 손글씨 전문.

Features:
- 손글씨 인식에 특화
- Transformer 기반 (BERT + Vision Transformer)
- 높은 정확도 (손글씨: 90-95%)
- HuggingFace Transformers 사용
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


class TrOCREngine(BaseOCREngine):
    """
    TrOCR 엔진 (손글씨 전문 OCR)

    Features:
    - 손글씨 인식 특화 (90-95% 정확도)
    - Transformer 기반 모델
    - 작은 이미지/패치에 최적화
    - GPU 가속 지원
    - HuggingFace 모델 사용

    Example:
        ```python
        from beanllm.domain.ocr.engines import TrOCREngine
        from beanllm.domain.ocr.models import OCRConfig

        engine = TrOCREngine()
        config = OCRConfig(language="en", use_gpu=True)
        result = engine.recognize(image, config)

        print(result["text"])
        ```

    Note:
        TrOCR은 텍스트 검출 기능이 없으므로,
        이미지 전체 또는 크롭된 텍스트 영역에만 사용해야 합니다.
    """

    def __init__(self):
        """
        TrOCR 엔진 초기화

        Raises:
            ImportError: transformers 또는 torch가 설치되지 않은 경우
        """
        super().__init__(name="TrOCR")
        self._check_dependencies()
        self._processor = None
        self._model = None

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: transformers 또는 torch가 설치되지 않은 경우
        """
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformers and torch are required for TrOCREngine. "
                "Install them with: pip install transformers torch"
            )

    def _init_model(self, use_gpu: bool) -> None:
        """
        TrOCR 모델 초기화 (lazy loading)

        Args:
            use_gpu: GPU 사용 여부
        """
        if self._model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        logger.info("Initializing TrOCR model (handwritten)")

        # 손글씨 인식용 모델
        model_name = "microsoft/trocr-base-handwritten"

        self._processor = TrOCRProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # GPU 설정
        if use_gpu and torch.cuda.is_available():
            self._model = self._model.to("cuda")
            logger.info("TrOCR model loaded on GPU")
        else:
            logger.info("TrOCR model loaded on CPU")

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        TrOCR로 텍스트 인식

        Args:
            image: 입력 이미지 (numpy array, RGB)
            config: OCR 설정

        Returns:
            dict: OCR 결과
                - text (str): 전체 텍스트
                - lines (List[OCRTextLine]): 라인별 결과 (단일 라인)
                - confidence (float): 신뢰도 (N/A, 1.0 반환)
                - language (str): 인식된 언어
                - metadata (dict): 추가 메타데이터

        Example:
            ```python
            import numpy as np
            engine = TrOCREngine()
            config = OCRConfig(language="en", use_gpu=True)
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            ```

        Note:
            TrOCR은 이미지 전체를 하나의 텍스트로 인식합니다.
            여러 라인 인식은 이미지를 라인별로 분할한 후 개별 호출이 필요합니다.
        """
        import torch
        from PIL import Image

        # 모델 초기화 (lazy loading)
        self._init_model(config.use_gpu)

        # numpy array를 PIL Image로 변환
        pil_image = Image.fromarray(image)

        # 이미지 전처리
        pixel_values = self._processor(pil_image, return_tensors="pt").pixel_values

        # GPU로 이동
        if config.use_gpu and torch.cuda.is_available():
            pixel_values = pixel_values.to("cuda")

        # 추론
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)

        # 디코딩
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 결과가 비어있는 경우
        if not generated_text.strip():
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "language": config.language,
                "metadata": {"engine": self.name, "empty_result": True},
            }

        # BoundingBox 생성 (전체 이미지)
        h, w = image.shape[:2]
        bbox = BoundingBox(x0=0, y0=0, x1=w, y1=h, confidence=1.0)

        # OCRTextLine 생성
        line = OCRTextLine(
            text=generated_text.strip(), bbox=bbox, confidence=1.0, language=config.language
        )

        return {
            "text": generated_text.strip(),
            "lines": [line],
            "confidence": 1.0,  # TrOCR은 신뢰도를 제공하지 않음
            "language": config.language,
            "metadata": {
                "engine": self.name,
                "model": "microsoft/trocr-base-handwritten",
                "note": "TrOCR does not provide confidence scores",
            },
        }

    def __repr__(self) -> str:
        model_loaded = "loaded" if self._model is not None else "not loaded"
        return f"TrOCREngine(model={model_loaded})"
