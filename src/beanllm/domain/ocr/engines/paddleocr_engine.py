"""
PaddleOCR Engine

PaddleOCR 기반 OCR 엔진 (메인 엔진).

Features:
- 90-96% 정확도
- 빠른 처리 속도
- 다국어 지원 (80+ languages)
- GPU 가속
- 언어별 모델 lazy loading
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


# 언어별 PaddleOCR 모델 매핑
LANGUAGE_MODELS = {
    "ko": "korean",  # 한글
    "zh": "ch",  # 중국어 (간체)
    "ja": "japan",  # 일본어
    "en": "en",  # 영어
    "auto": "ch",  # 자동 감지 시 중국어 모델 사용 (다국어 지원)
}


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR 엔진 (메인 OCR 엔진)

    Features:
    - 90-96% 정확도
    - 빠른 처리 속도 (~1초/페이지)
    - 다국어 지원 (한글, 중국어, 일본어, 영어 등)
    - GPU 가속 지원
    - 텍스트 방향 감지 (use_angle_cls)

    Example:
        ```python
        from beanllm.domain.ocr.engines import PaddleOCREngine
        from beanllm.domain.ocr.models import OCRConfig

        engine = PaddleOCREngine()
        config = OCRConfig(language="ko", use_gpu=True)
        result = engine.recognize(image, config)

        print(result["text"])
        print(f"Confidence: {result['confidence']:.2%}")
        ```
    """

    def __init__(self):
        """
        PaddleOCR 엔진 초기화

        Raises:
            ImportError: PaddleOCR가 설치되지 않은 경우
        """
        super().__init__(name="PaddleOCR")
        self._check_dependencies()
        self._models: Dict[str, Any] = {}  # 언어별 모델 캐시

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: PaddleOCR가 설치되지 않은 경우
        """
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            raise ImportError(
                "PaddleOCR is required for PaddleOCREngine. "
                "Install it with: pip install paddleocr"
            )

    def _get_language_code(self, language: str) -> str:
        """
        언어 코드를 PaddleOCR 형식으로 변환

        Args:
            language: 언어 코드 (ko, en, zh, ja, auto)

        Returns:
            str: PaddleOCR 언어 코드

        Example:
            >>> engine._get_language_code("ko")
            'korean'
            >>> engine._get_language_code("en")
            'en'
        """
        return LANGUAGE_MODELS.get(language, "ch")  # 기본값: 중국어 (다국어 지원)

    def _get_or_create_model(self, language: str, use_gpu: bool) -> Any:
        """
        언어별 PaddleOCR 모델 가져오기 (lazy loading)

        Args:
            language: 언어 코드
            use_gpu: GPU 사용 여부

        Returns:
            PaddleOCR: 초기화된 PaddleOCR 인스턴스

        Note:
            모델은 언어별로 캐싱되어 재사용됩니다.
        """
        from paddleocr import PaddleOCR

        lang_code = self._get_language_code(language)
        cache_key = f"{lang_code}_{use_gpu}"

        # 캐시에 없으면 새로 생성
        if cache_key not in self._models:
            logger.info(f"Initializing PaddleOCR model: {lang_code} (GPU: {use_gpu})")
            self._models[cache_key] = PaddleOCR(
                use_angle_cls=True,  # 텍스트 방향 감지 활성화
                lang=lang_code,
                use_gpu=use_gpu,
                show_log=False,  # 로그 출력 비활성화
            )

        return self._models[cache_key]

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        PaddleOCR로 텍스트 인식

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
            engine = PaddleOCREngine()
            config = OCRConfig(language="ko")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            ```
        """
        # 모델 가져오기 (lazy loading)
        model = self._get_or_create_model(config.language, config.use_gpu)

        # OCR 실행
        raw_result = model.ocr(image, cls=True)

        # 결과가 None이거나 비어있는 경우 처리
        if not raw_result or not raw_result[0]:
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "language": config.language,
                "metadata": {"engine": self.name, "empty_result": True},
            }

        # 결과 변환
        return self._convert_result(raw_result[0], config)

    def _convert_result(self, raw_result: list, config: OCRConfig) -> Dict:
        """
        PaddleOCR 결과를 표준 형식으로 변환

        Args:
            raw_result: PaddleOCR 원본 결과
            config: OCR 설정

        Returns:
            dict: 표준 형식 OCR 결과

        Note:
            PaddleOCR 결과 형식:
            [
                [[[x0, y0], [x1, y1], [x2, y2], [x3, y3]], ("텍스트", 신뢰도)],
                ...
            ]
        """
        lines = []
        text_parts = []
        total_confidence = 0.0
        valid_lines = 0

        for line_data in raw_result:
            # 좌표와 (텍스트, 신뢰도) 추출
            bbox_coords, (text, confidence) = line_data

            # 신뢰도 임계값 체크
            if confidence < config.confidence_threshold:
                continue

            # BoundingBox 생성 (4개 좌표 → x0, y0, x1, y1)
            # bbox_coords = [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
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
        return f"PaddleOCREngine(models_loaded={len(self._models)})"
