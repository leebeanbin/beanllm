"""
Nougat Engine

Nougat (Neural Optical Understanding for Academic Documents) 엔진.

Features:
- 학술 논문 OCR 전문
- 수식, 표, 그래프 인식
- LaTeX 수식 출력
- PDF → Markdown 변환
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


class NougatEngine(BaseOCREngine):
    """
    Nougat 엔진 (학술 논문 전문 OCR)

    Features:
    - 학술 논문 특화 (수식, 표, 그래프)
    - LaTeX 수식 출력
    - Markdown 변환
    - Meta의 Nougat 모델 사용
    - GPU 가속 지원

    Example:
        ```python
        from beanllm.domain.ocr.engines import NougatEngine
        from beanllm.domain.ocr.models import OCRConfig

        engine = NougatEngine()
        config = OCRConfig(language="en", use_gpu=True)
        result = engine.recognize(image, config)

        print(result["text"])  # Markdown with LaTeX
        ```

    Note:
        Nougat은 학술 논문 페이지 전체를 처리하도록 설계되었습니다.
        일반 문서보다 논문 PDF에 최적화되어 있습니다.
    """

    def __init__(self):
        """
        Nougat 엔진 초기화

        Raises:
            ImportError: nougat 라이브러리가 설치되지 않은 경우
        """
        super().__init__(name="Nougat")
        self._check_dependencies()
        self._model = None

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: 필요한 라이브러리가 설치되지 않은 경우
        """
        try:
            # Nougat은 별도 패키지로 제공
            import torch  # noqa: F401
            from transformers import NougatProcessor, VisionEncoderDecoderModel  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch and transformers are required for NougatEngine. "
                "Install them with: pip install torch transformers"
            )

    def _init_model(self, use_gpu: bool) -> None:
        """
        Nougat 모델 초기화 (lazy loading)

        Args:
            use_gpu: GPU 사용 여부
        """
        if self._model is not None:
            return

        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        logger.info("Initializing Nougat model (academic documents)")

        # Nougat 모델
        model_name = "facebook/nougat-base"

        self._processor = NougatProcessor.from_pretrained(model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # GPU 설정
        if use_gpu and torch.cuda.is_available():
            self._model = self._model.to("cuda")
            logger.info("Nougat model loaded on GPU")
        else:
            logger.info("Nougat model loaded on CPU")

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        Nougat으로 텍스트 인식 (학술 논문)

        Args:
            image: 입력 이미지 (numpy array, RGB) - 논문 페이지
            config: OCR 설정

        Returns:
            dict: OCR 결과
                - text (str): Markdown 형식 텍스트 (LaTeX 수식 포함)
                - lines (List[OCRTextLine]): 라인별 결과
                - confidence (float): 신뢰도 (N/A, 1.0 반환)
                - language (str): 인식된 언어
                - metadata (dict): 추가 메타데이터

        Example:
            ```python
            import numpy as np
            engine = NougatEngine()
            config = OCRConfig(language="en", use_gpu=True)
            # 논문 페이지 이미지
            image = np.zeros((1000, 800, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            print(result["text"])  # Markdown with LaTeX
            ```
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
            outputs = self._model.generate(
                pixel_values,
                max_length=self._model.decoder.config.max_length,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
            )

        # 디코딩 (Markdown + LaTeX)
        generated_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # 결과가 비어있는 경우
        if not generated_text.strip():
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "language": config.language,
                "metadata": {"engine": self.name, "empty_result": True},
            }

        # Markdown을 라인별로 분할
        lines = []
        text_lines = generated_text.strip().split("\n")

        h, w = image.shape[:2]

        for i, line_text in enumerate(text_lines):
            if not line_text.strip():
                continue

            # 대략적인 BoundingBox (실제 위치 정보 없음)
            # 페이지를 라인 개수로 분할
            line_height = h // max(len(text_lines), 1)
            y0 = i * line_height
            y1 = (i + 1) * line_height

            bbox = BoundingBox(x0=0, y0=y0, x1=w, y1=y1, confidence=1.0)

            line = OCRTextLine(
                text=line_text.strip(), bbox=bbox, confidence=1.0, language=config.language
            )
            lines.append(line)

        return {
            "text": generated_text.strip(),
            "lines": lines,
            "confidence": 1.0,  # Nougat은 신뢰도를 제공하지 않음
            "language": config.language,
            "metadata": {
                "engine": self.name,
                "model": "facebook/nougat-base",
                "output_format": "markdown+latex",
                "note": "Optimized for academic documents",
            },
        }

    def __repr__(self) -> str:
        model_loaded = "loaded" if self._model is not None else "not loaded"
        return f"NougatEngine(model={model_loaded})"
