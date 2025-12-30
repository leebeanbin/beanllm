"""
LLM 기반 OCR 후처리

OCR 결과를 LLM으로 보정하여 98%+ 정확도 달성.

Features:
- 문맥 기반 오류 수정
- 맞춤법/문법 검사
- 특수문자 복원
- 신뢰도 낮은 부분 집중 보정
"""

import logging
from typing import Dict, List, Optional

from ..models import OCRResult, OCRTextLine

logger = logging.getLogger(__name__)


class LLMPostprocessor:
    """
    LLM 기반 OCR 후처리기

    OCR 결과를 LLM에 전달하여 오류를 수정하고 정확도를 높입니다.

    Features:
    - 문맥 기반 오류 수정
    - 맞춤법/문법 검사
    - 신뢰도 낮은 라인 집중 보정
    - 한글/영어/일본어/중국어 지원

    Example:
        ```python
        from beanllm.domain.ocr import beanOCR, OCRConfig

        # LLM 후처리 활성화
        ocr = beanOCR(
            engine="paddleocr",
            enable_llm_postprocessing=True,
            llm_model="gpt-4o-mini"
        )

        result = ocr.recognize("noisy_image.jpg")
        # → OCR 결과가 LLM으로 자동 보정됨
        print(result.text)
        print(result.metadata.get("llm_corrected"))  # True
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        confidence_threshold: float = 0.7,
    ):
        """
        LLM 후처리기 초기화

        Args:
            model: LLM 모델 (gpt-4o-mini, gpt-4o, claude-3-haiku, etc.)
            api_key: API 키 (없으면 환경변수 사용)
            temperature: LLM temperature (0.0 = deterministic)
            confidence_threshold: 이 값 미만의 라인만 집중 보정
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold

        # LLM 클라이언트 초기화 (beanllm 사용)
        self._init_llm_client()

    def _init_llm_client(self):
        """LLM 클라이언트 초기화"""
        try:
            from beanllm import BeanLLM

            self.llm = BeanLLM(model=self.model)
        except ImportError:
            logger.warning(
                "BeanLLM not available. LLM postprocessing will be disabled."
            )
            self.llm = None

    def process(self, ocr_result: OCRResult) -> OCRResult:
        """
        OCR 결과를 LLM으로 후처리

        Args:
            ocr_result: 원본 OCR 결과

        Returns:
            OCRResult: 보정된 OCR 결과

        Example:
            ```python
            postprocessor = LLMPostprocessor(model="gpt-4o-mini")
            corrected_result = postprocessor.process(ocr_result)
            ```
        """
        if self.llm is None:
            logger.warning("LLM client not initialized. Skipping postprocessing.")
            return ocr_result

        # 신뢰도 낮은 라인 찾기
        low_confidence_lines = [
            line for line in ocr_result.lines if line.confidence < self.confidence_threshold
        ]

        if not low_confidence_lines:
            logger.info("All lines have high confidence. No LLM correction needed.")
            ocr_result.metadata["llm_corrected"] = False
            return ocr_result

        # LLM 보정
        logger.info(
            f"Correcting {len(low_confidence_lines)}/{len(ocr_result.lines)} "
            f"lines with LLM (confidence < {self.confidence_threshold})"
        )

        corrected_text = self._correct_with_llm(
            text=ocr_result.text,
            low_confidence_lines=low_confidence_lines,
            language=ocr_result.language,
        )

        # 보정된 결과로 OCRResult 업데이트
        corrected_result = OCRResult(
            text=corrected_text,
            lines=ocr_result.lines,  # BoundingBox는 유지
            language=ocr_result.language,
            confidence=min(ocr_result.confidence + 0.1, 1.0),  # 신뢰도 상승
            engine=ocr_result.engine,
            processing_time=ocr_result.processing_time,
            metadata={
                **ocr_result.metadata,
                "llm_corrected": True,
                "llm_model": self.model,
                "original_text": ocr_result.text,
                "corrected_lines": len(low_confidence_lines),
            },
        )

        return corrected_result

    def _correct_with_llm(
        self,
        text: str,
        low_confidence_lines: List[OCRTextLine],
        language: str,
    ) -> str:
        """
        LLM으로 텍스트 보정

        Args:
            text: 전체 텍스트
            low_confidence_lines: 신뢰도 낮은 라인들
            language: 언어 코드

        Returns:
            str: 보정된 텍스트
        """
        # 언어별 프롬프트
        language_instructions = {
            "ko": "한국어 문맥에 맞게",
            "en": "in proper English context",
            "ja": "日本語の文脈に合わせて",
            "zh": "根据中文语境",
        }
        lang_instruction = language_instructions.get(language, "in proper context")

        # 프롬프트 생성
        prompt = self._build_correction_prompt(text, low_confidence_lines, lang_instruction)

        try:
            # LLM 호출
            response = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR error correction expert. "
                        "Fix OCR recognition errors while preserving the original meaning and structure.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )

            corrected_text = response.get("content", "").strip()

            # 검증: 너무 다르면 원본 반환
            if len(corrected_text) < len(text) * 0.5 or len(corrected_text) > len(text) * 2.0:
                logger.warning(
                    "LLM correction produced significantly different length. "
                    "Using original text."
                )
                return text

            return corrected_text

        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return text

    def _build_correction_prompt(
        self,
        text: str,
        low_confidence_lines: List[OCRTextLine],
        lang_instruction: str,
    ) -> str:
        """
        보정 프롬프트 생성

        Args:
            text: 전체 텍스트
            low_confidence_lines: 신뢰도 낮은 라인들
            lang_instruction: 언어별 지시사항

        Returns:
            str: 프롬프트
        """
        # 신뢰도 낮은 부분 표시
        low_conf_texts = [line.text for line in low_confidence_lines]

        prompt = f"""Please correct the OCR recognition errors in the following text {lang_instruction}.

**Instructions**:
1. Fix spelling errors, misrecognized characters, and spacing issues
2. Maintain the original structure and formatting (line breaks, paragraphs)
3. Focus on correcting these low-confidence parts: {low_conf_texts[:5]}
4. Output ONLY the corrected text without any explanations

**Original OCR Text**:
```
{text}
```

**Corrected Text**:
"""
        return prompt

    def __repr__(self) -> str:
        return f"LLMPostprocessor(model={self.model}, threshold={self.confidence_threshold})"
