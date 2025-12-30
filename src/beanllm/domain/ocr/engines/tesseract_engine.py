"""
Tesseract Engine

Tesseract OCR 기반 엔진 (Fallback 엔진).

Features:
- 70-85% 정확도
- 오픈소스 OCR
- 다국어 지원 (100+ languages)
- 가볍고 빠름
- Fallback 용도로 적합
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


# 언어별 Tesseract 코드 매핑
LANGUAGE_CODES = {
    "ko": "kor",  # 한글
    "zh": "chi_sim",  # 중국어 (간체)
    "ja": "jpn",  # 일본어
    "en": "eng",  # 영어
    "auto": "eng",  # 자동 감지 시 영어 사용
}


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR 엔진 (Fallback 엔진)

    Features:
    - 70-85% 정확도
    - 오픈소스 OCR (무료)
    - 다국어 지원 (한글, 중국어, 일본어, 영어 등)
    - 가볍고 빠른 처리
    - 다른 엔진 실패 시 Fallback으로 사용

    Example:
        ```python
        from beanllm.domain.ocr.engines import TesseractEngine
        from beanllm.domain.ocr.models import OCRConfig

        engine = TesseractEngine()
        config = OCRConfig(language="ko")
        result = engine.recognize(image, config)

        print(result["text"])
        print(f"Confidence: {result['confidence']:.2%}")
        ```
    """

    def __init__(self):
        """
        Tesseract 엔진 초기화

        Raises:
            ImportError: pytesseract가 설치되지 않은 경우
        """
        super().__init__(name="Tesseract")
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: pytesseract가 설치되지 않은 경우
        """
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            raise ImportError(
                "pytesseract is required for TesseractEngine. "
                "Install it with: pip install pytesseract\n"
                "Also install Tesseract OCR: brew install tesseract (macOS) or "
                "apt-get install tesseract-ocr (Linux)"
            )

    def _get_language_code(self, language: str) -> str:
        """
        언어 코드를 Tesseract 형식으로 변환

        Args:
            language: 언어 코드 (ko, en, zh, ja, auto)

        Returns:
            str: Tesseract 언어 코드

        Example:
            >>> engine._get_language_code("ko")
            'kor'
            >>> engine._get_language_code("zh")
            'chi_sim'
        """
        return LANGUAGE_CODES.get(language, "eng")  # 기본값: 영어

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        Tesseract로 텍스트 인식

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
            engine = TesseractEngine()
            config = OCRConfig(language="ko")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            ```
        """
        import pytesseract
        from PIL import Image

        # numpy array를 PIL Image로 변환
        pil_image = Image.fromarray(image)

        # 언어 코드 가져오기
        lang_code = self._get_language_code(config.language)

        # Tesseract 설정
        custom_config = r"--oem 3 --psm 3"  # LSTM OCR, 자동 페이지 세그멘테이션

        # OCR 실행 (상세 데이터 포함)
        try:
            # image_to_data: 워드 단위 상세 정보 반환
            data = pytesseract.image_to_data(
                pil_image, lang=lang_code, config=custom_config, output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "language": config.language,
                "metadata": {"engine": self.name, "error": str(e)},
            }

        # 결과 변환
        return self._convert_result(data, config)

    def _convert_result(self, data: Dict, config: OCRConfig) -> Dict:
        """
        Tesseract 결과를 표준 형식으로 변환

        Args:
            data: Tesseract 원본 결과 (image_to_data)
            config: OCR 설정

        Returns:
            dict: 표준 형식 OCR 결과

        Note:
            Tesseract image_to_data 형식:
            {
                'level': [...],
                'page_num': [...],
                'block_num': [...],
                'par_num': [...],
                'line_num': [...],
                'word_num': [...],
                'left': [...],
                'top': [...],
                'width': [...],
                'height': [...],
                'conf': [...],
                'text': [...]
            }
        """
        # 라인별로 텍스트 그룹화
        lines_dict = {}  # {line_num: [(text, bbox, conf), ...]}

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            # 빈 텍스트나 낮은 신뢰도 제외
            if not text or conf < 0:
                continue

            # 신뢰도를 0-1 범위로 정규화 (Tesseract는 0-100)
            conf_normalized = conf / 100.0

            # 신뢰도 임계값 체크
            if conf_normalized < config.confidence_threshold:
                continue

            line_num = data["line_num"][i]
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            bbox = BoundingBox(x0=x, y0=y, x1=x + w, y1=y + h, confidence=conf_normalized)

            if line_num not in lines_dict:
                lines_dict[line_num] = []

            lines_dict[line_num].append((text, bbox, conf_normalized))

        # 라인별 OCRTextLine 생성
        lines = []
        text_parts = []
        total_confidence = 0.0
        valid_lines = 0

        for line_num in sorted(lines_dict.keys()):
            words = lines_dict[line_num]

            # 라인의 모든 단어 결합
            line_text = " ".join([word[0] for word in words])

            # 라인 전체 BoundingBox 계산
            all_bboxes = [word[1] for word in words]
            x0 = min(bbox.x0 for bbox in all_bboxes)
            y0 = min(bbox.y0 for bbox in all_bboxes)
            x1 = max(bbox.x1 for bbox in all_bboxes)
            y1 = max(bbox.y1 for bbox in all_bboxes)

            # 라인 평균 신뢰도
            line_conf = sum(word[2] for word in words) / len(words)

            line_bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, confidence=line_conf)

            # OCRTextLine 생성
            line = OCRTextLine(
                text=line_text, bbox=line_bbox, confidence=line_conf, language=config.language
            )

            lines.append(line)
            text_parts.append(line_text)
            total_confidence += line_conf
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
                "total_words": n_boxes,
                "valid_lines": valid_lines,
            },
        }

    def __repr__(self) -> str:
        return "TesseractEngine()"
