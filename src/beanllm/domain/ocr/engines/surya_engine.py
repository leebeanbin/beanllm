"""
Surya Engine

Surya OCR 엔진 - 복잡한 레이아웃 전문.

Features:
- 복잡한 레이아웃 처리
- 다단 컬럼, 표, 이미지 혼합
- 90+ 언어 지원
- Layout analysis + OCR 통합
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


class SuryaEngine(BaseOCREngine):
    """
    Surya 엔진 (복잡한 레이아웃 전문 OCR)

    Features:
    - 복잡한 레이아웃 처리 (다단, 표, 혼합)
    - 90+ 언어 지원
    - Layout detection + OCR 통합
    - GPU 가속 지원
    - surya-ocr 라이브러리 사용

    Example:
        ```python
        from beanllm.domain.ocr.engines import SuryaEngine
        from beanllm.domain.ocr.models import OCRConfig

        engine = SuryaEngine()
        config = OCRConfig(language="ko", use_gpu=True)
        result = engine.recognize(image, config)

        print(result["text"])
        ```

    Note:
        Surya는 복잡한 문서 레이아웃에 특화되어 있습니다.
        신문, 잡지, 카탈로그 등에 적합합니다.
    """

    def __init__(self):
        """
        Surya 엔진 초기화

        Raises:
            ImportError: surya-ocr가 설치되지 않은 경우
        """
        super().__init__(name="Surya")
        self._check_dependencies()
        self._model = None
        self._processor = None

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: 필요한 라이브러리가 설치되지 않은 경우
        """
        try:
            # Surya는 별도 패키지
            import surya  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "surya-ocr and torch are required for SuryaEngine. "
                "Install them with: pip install surya-ocr torch"
            )

    def _init_model(self, use_gpu: bool) -> None:
        """
        Surya 모델 초기화 (lazy loading)

        Args:
            use_gpu: GPU 사용 여부
        """
        if self._model is not None:
            return

        import torch
        from surya.model.detection import load_model as load_det_model
        from surya.model.detection import load_processor as load_det_processor
        from surya.model.recognition import load_model as load_rec_model
        from surya.model.recognition import load_processor as load_rec_processor

        logger.info("Initializing Surya models (detection + recognition)")

        # Detection 모델 (레이아웃 분석)
        self._det_model = load_det_model()
        self._det_processor = load_det_processor()

        # Recognition 모델 (텍스트 인식)
        self._model = load_rec_model()
        self._processor = load_rec_processor()

        # GPU 설정
        if use_gpu and torch.cuda.is_available():
            self._det_model = self._det_model.to("cuda")
            self._model = self._model.to("cuda")
            logger.info("Surya models loaded on GPU")
        else:
            logger.info("Surya models loaded on CPU")

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        Surya로 텍스트 인식 (복잡한 레이아웃)

        Args:
            image: 입력 이미지 (numpy array, RGB)
            config: OCR 설정

        Returns:
            dict: OCR 결과
                - text (str): 전체 텍스트
                - lines (List[OCRTextLine]): 라인별 결과
                - confidence (float): 평균 신뢰도
                - language (str): 인식된 언어
                - metadata (dict): 추가 메타데이터

        Example:
            ```python
            import numpy as np
            engine = SuryaEngine()
            config = OCRConfig(language="ko", use_gpu=True)
            # 복잡한 레이아웃 이미지
            image = np.zeros((1000, 800, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            ```
        """
        from PIL import Image
        from surya.detection import batch_detection
        from surya.recognition import batch_recognition

        # 모델 초기화 (lazy loading)
        self._init_model(config.use_gpu)

        # numpy array를 PIL Image로 변환
        pil_image = Image.fromarray(image)

        # 1. Layout Detection (텍스트 영역 검출)
        det_predictions = batch_detection([pil_image], self._det_model, self._det_processor)

        # 2. Text Recognition (검출된 영역에서 텍스트 인식)
        rec_predictions = batch_recognition(
            [pil_image], det_predictions[0].bboxes, self._model, self._processor
        )

        # 결과 변환
        return self._convert_result(rec_predictions[0], config)

    def _convert_result(self, prediction: Any, config: OCRConfig) -> Dict:
        """
        Surya 결과를 표준 형식으로 변환

        Args:
            prediction: Surya 예측 결과
            config: OCR 설정

        Returns:
            dict: 표준 형식 OCR 결과
        """
        lines = []
        text_parts = []
        total_confidence = 0.0
        valid_lines = 0

        for text_line in prediction.text_lines:
            text = text_line.text
            confidence = getattr(text_line, "confidence", 1.0)

            # 신뢰도 임계값 체크
            if confidence < config.confidence_threshold:
                continue

            # BoundingBox 생성
            bbox_coords = text_line.bbox  # [x0, y0, x1, y1]
            bbox = BoundingBox(
                x0=bbox_coords[0],
                y0=bbox_coords[1],
                x1=bbox_coords[2],
                y1=bbox_coords[3],
                confidence=confidence,
            )

            # OCRTextLine 생성
            line = OCRTextLine(text=text, bbox=bbox, confidence=confidence, language=config.language)

            lines.append(line)
            text_parts.append(text)
            total_confidence += confidence
            valid_lines += 1

        # 전체 텍스트 및 평균 신뢰도 계산
        full_text = "\n".join(text_parts)
        avg_confidence = total_confidence / valid_lines if valid_lines > 0 else 0.0

        return {
            "text": full_text,
            "lines": lines,
            "confidence": avg_confidence,
            "language": config.language,
            "metadata": {
                "engine": self.name,
                "total_lines": len(prediction.text_lines),
                "valid_lines": valid_lines,
                "filtered_lines": len(prediction.text_lines) - valid_lines,
            },
        }

    def __repr__(self) -> str:
        model_loaded = "loaded" if self._model is not None else "not loaded"
        return f"SuryaEngine(model={model_loaded})"
