"""
EasyOCR Engine

EasyOCR 기반 OCR 엔진 (대체 엔진).

Features:
- 85-92% 정확도
- 사용하기 쉬움
- 다국어 지원 (80+ languages)
- GPU 가속
- 언어 조합별 모델 lazy loading
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


# 언어별 EasyOCR 코드 매핑
LANGUAGE_CODES = {
    "ko": "ko",  # 한글
    "zh": "ch_sim",  # 중국어 (간체)
    "ja": "ja",  # 일본어
    "en": "en",  # 영어
    "auto": "en",  # 자동 감지 시 영어 사용
}


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR 엔진 (대체 OCR 엔진)

    Features:
    - 85-92% 정확도
    - 사용하기 쉬운 API
    - 다국어 지원 (한글, 중국어, 일본어, 영어 등)
    - GPU 가속 지원
    - 여러 언어 동시 인식 가능

    Example:
        ```python
        from beanllm.domain.ocr.engines import EasyOCREngine
        from beanllm.domain.ocr.models import OCRConfig

        engine = EasyOCREngine()
        config = OCRConfig(language="ko", use_gpu=True)
        result = engine.recognize(image, config)

        print(result["text"])
        print(f"Confidence: {result['confidence']:.2%}")
        ```
    """

    def __init__(self):
        """
        EasyOCR 엔진 초기화

        Raises:
            ImportError: EasyOCR이 설치되지 않은 경우
        """
        super().__init__(name="EasyOCR")
        self._check_dependencies()
        self._readers: Dict[str, Any] = {}  # 언어 조합별 Reader 캐시

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: EasyOCR이 설치되지 않은 경우
        """
        try:
            import easyocr  # noqa: F401
        except ImportError:
            raise ImportError(
                "EasyOCR is required for EasyOCREngine. "
                "Install it with: pip install easyocr"
            )

    def _get_language_code(self, language: str) -> str:
        """
        언어 코드를 EasyOCR 형식으로 변환

        Args:
            language: 언어 코드 (ko, en, zh, ja, auto)

        Returns:
            str: EasyOCR 언어 코드

        Example:
            >>> engine._get_language_code("ko")
            'ko'
            >>> engine._get_language_code("zh")
            'ch_sim'
        """
        return LANGUAGE_CODES.get(language, "en")  # 기본값: 영어

    def _get_or_create_reader(self, language: str, use_gpu: bool) -> Any:
        """
        언어별 EasyOCR Reader 가져오기 (lazy loading)

        Args:
            language: 언어 코드
            use_gpu: GPU 사용 여부

        Returns:
            easyocr.Reader: 초기화된 Reader 인스턴스

        Note:
            Reader는 언어 조합별로 캐싱되어 재사용됩니다.
        """
        import easyocr

        lang_code = self._get_language_code(language)

        # 영어와 다른 언어를 함께 사용 (다국어 문서 대응)
        lang_list = [lang_code]
        if lang_code != "en":
            lang_list.append("en")

        cache_key = f"{'-'.join(sorted(lang_list))}_{use_gpu}"

        # 캐시에 없으면 새로 생성
        if cache_key not in self._readers:
            logger.info(f"Initializing EasyOCR Reader: {lang_list} (GPU: {use_gpu})")
            self._readers[cache_key] = easyocr.Reader(
                lang_list,
                gpu=use_gpu,
                verbose=False,  # 로그 출력 비활성화
            )

        return self._readers[cache_key]

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        EasyOCR로 텍스트 인식

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
            engine = EasyOCREngine()
            config = OCRConfig(language="ko")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            ```
        """
        # Reader 가져오기 (lazy loading)
        reader = self._get_or_create_reader(config.language, config.use_gpu)

        # OCR 실행
        # readtext returns: [([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], 'text', confidence), ...]
        raw_result = reader.readtext(image)

        # 결과가 비어있는 경우 처리
        if not raw_result:
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "language": config.language,
                "metadata": {"engine": self.name, "empty_result": True},
            }

        # 결과 변환
        return self._convert_result(raw_result, config)

    def _convert_result(self, raw_result: list, config: OCRConfig) -> Dict:
        """
        EasyOCR 결과를 표준 형식으로 변환

        Args:
            raw_result: EasyOCR 원본 결과
            config: OCR 설정

        Returns:
            dict: 표준 형식 OCR 결과

        Note:
            EasyOCR 결과 형식:
            [
                ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], "텍스트", 신뢰도),
                ...
            ]
        """
        lines = []
        text_parts = []
        total_confidence = 0.0
        valid_lines = 0

        for bbox_coords, text, confidence in raw_result:
            # 신뢰도 임계값 체크
            if confidence < config.confidence_threshold:
                continue

            # BoundingBox 생성 (4개 좌표 → x0, y0, x1, y1)
            # bbox_coords = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            x_coords = [coord[0] for coord in bbox_coords]
            y_coords = [coord[1] for coord in bbox_coords]

            bbox = BoundingBox(
                x0=min(x_coords),
                y0=min(y_coords),
                x1=max(x_coords),
                y1=max(y_coords),
                confidence=confidence,
            )

            # OCRTextLine 생성
            line = OCRTextLine(
                text=text, bbox=bbox, confidence=confidence, language=config.language
            )

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
                "total_lines": len(raw_result),
                "valid_lines": valid_lines,
                "filtered_lines": len(raw_result) - valid_lines,
            },
        }

    def __repr__(self) -> str:
        return f"EasyOCREngine(readers_loaded={len(self._readers)})"
