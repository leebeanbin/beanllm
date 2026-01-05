"""
Qwen2.5-VL OCR Engine

Alibaba의 Qwen2.5-VL 비전-언어 모델을 사용한 OCR 엔진.
오픈소스 최고 성능, DocVQA 우수.

Qwen2.5-VL 모델 특징:
- 오픈소스 VLM 중 최고 성능
- 2B/7B/72B 파라미터 옵션
- 90+ 언어 지원
- transformers 공식 지원
- DocVQA, MathVista 등 벤치마크 우수

Requirements:
    pip install transformers torch pillow qwen-vl-utils
"""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)

# transformers 설치 여부 체크
try:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False


class Qwen2VLEngine(BaseOCREngine):
    """
    Qwen2.5-VL OCR 엔진

    Alibaba의 Qwen2.5-VL 모델을 사용한 고성능 OCR 엔진.

    Features:
    - 오픈소스 최고 성능
    - 90+ 언어 지원
    - 문맥 이해 능력
    - Lazy loading (첫 호출 시 모델 로드)

    Example:
        ```python
        from beanllm.domain.ocr import beanOCR

        # Qwen2.5-VL 엔진 사용 (2B - 경량)
        ocr = beanOCR(engine="qwen2vl-2b", language="ko")
        result = ocr.recognize("document.jpg")

        # 7B 모델 (고성능)
        ocr = beanOCR(engine="qwen2vl-7b", language="ko")
        result = ocr.recognize("document.jpg")
        ```
    """

    def __init__(self, model_size: str = "2b", use_gpu: bool = True):
        """
        Qwen2.5-VL 엔진 초기화

        Args:
            model_size: 모델 크기 (2b/7b/72b)
            use_gpu: GPU 사용 여부
        """
        super().__init__()

        if not HAS_QWEN2VL:
            raise ImportError(
                "transformers and torch are required for Qwen2.5-VL engine. "
                "Install them with: pip install transformers torch pillow qwen-vl-utils"
            )

        self.model_size = model_size
        self.use_gpu = use_gpu
        self._model = None
        self._processor = None

    def _get_model_name(self) -> str:
        """모델 이름 가져오기"""
        model_map = {
            "2b": "Qwen/Qwen2.5-VL-2B-Instruct",
            "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "72b": "Qwen/Qwen2.5-VL-72B-Instruct",
        }
        return model_map.get(self.model_size, model_map["2b"])

    def _init_model(self):
        """모델 초기화 (lazy loading)"""
        if self._model is not None:
            return

        model_name = self._get_model_name()
        logger.info(f"Loading Qwen2.5-VL model: {model_name}")

        # Processor 로드
        self._processor = AutoProcessor.from_pretrained(model_name)

        # 모델 로드
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.use_gpu else torch.float32,
            device_map="auto" if self.use_gpu else "cpu",
        )

        logger.info(f"Qwen2.5-VL {self.model_size} model loaded successfully")

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        Qwen2.5-VL로 텍스트 인식

        Args:
            image: 입력 이미지 (numpy array, RGB)
            config: OCR 설정

        Returns:
            Dict: OCR 결과
                {
                    "text": str,
                    "lines": List[OCRTextLine],
                    "confidence": float,
                    "language": str,
                    "metadata": dict
                }
        """
        from PIL import Image

        # 모델 초기화
        self._init_model()

        # numpy array → PIL Image
        pil_image = Image.fromarray(image)

        # OCR 프롬프트 (언어별)
        language_prompts = {
            "ko": "이 이미지의 모든 텍스트를 정확히 추출해주세요. 원본 형식과 구조를 유지하세요.",
            "en": "Extract all text from this image accurately. Preserve the original format and structure.",
            "ja": "この画像からすべてのテキストを正確に抽出してください。元の形式と構造を保持してください。",
            "zh": "准确提取此图像中的所有文本。保持原始格式和结构。",
            "auto": "Extract all text from this image accurately. Preserve the original format and structure.",
        }
        prompt = language_prompts.get(config.language, language_prompts["auto"])

        # 대화 형식으로 입력 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 입력 준비
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )

        if self.use_gpu and torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # 추론
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        # 디코딩
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # 결과 변환
        return self._convert_result(output_text, image, config)

    def _convert_result(
        self, text: str, image: np.ndarray, config: OCRConfig
    ) -> Dict:
        """
        Qwen2.5-VL 결과를 표준 형식으로 변환

        Qwen2.5-VL은 BoundingBox를 제공하지 않으므로
        전체 텍스트만 반환하고, 각 줄을 추정하여 OCRTextLine 생성
        """
        h, w = image.shape[:2]

        # 텍스트를 줄 단위로 분리
        lines_text = text.strip().split("\n")

        # 각 줄에 대해 OCRTextLine 생성 (BoundingBox는 추정)
        lines = []
        line_height = h / max(len(lines_text), 1)

        for idx, line_text in enumerate(lines_text):
            if not line_text.strip():
                continue

            # BoundingBox 추정 (전체 너비, 균등 분할 높이)
            bbox = BoundingBox(
                x0=0,
                y0=idx * line_height,
                x1=w,
                y1=(idx + 1) * line_height,
                confidence=0.95,  # Qwen2.5-VL은 고품질이므로 높은 신뢰도
            )

            line = OCRTextLine(
                text=line_text,
                bbox=bbox,
                confidence=0.95,
                language=config.language,
            )
            lines.append(line)

        # 평균 신뢰도
        avg_confidence = 0.95 if lines else 0.0

        return {
            "text": text,
            "lines": lines,
            "confidence": avg_confidence,
            "language": config.language,
            "metadata": {
                "model": f"Qwen2.5-VL-{self.model_size.upper()}",
                "line_count": len(lines),
            },
        }

    def __repr__(self) -> str:
        return f"Qwen2VLEngine(model_size={self.model_size}, use_gpu={self.use_gpu})"
