"""
Cloud OCR Engine

클라우드 OCR API 엔진 (Google Vision, AWS Textract 등).

Features:
- 클라우드 OCR API 통합
- Google Vision API
- AWS Textract
- 높은 정확도 (95%+)
- 다국어 지원
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..models import BoundingBox, OCRConfig, OCRTextLine
from .base import BaseOCREngine

logger = logging.getLogger(__name__)


class CloudOCREngine(BaseOCREngine):
    """
    Cloud OCR 엔진 (Google Vision, AWS Textract)

    Features:
    - 클라우드 OCR API 사용
    - 95%+ 정확도
    - 다국어 지원
    - Google Vision API 또는 AWS Textract
    - API 키 필요

    Example:
        ```python
        from beanllm.domain.ocr.engines import CloudOCREngine
        from beanllm.domain.ocr.models import OCRConfig

        # Google Vision
        engine = CloudOCREngine(provider="google", api_key="YOUR_API_KEY")
        config = OCRConfig(language="ko")
        result = engine.recognize(image, config)

        # AWS Textract
        engine = CloudOCREngine(
            provider="aws",
            aws_access_key="YOUR_ACCESS_KEY",
            aws_secret_key="YOUR_SECRET_KEY"
        )
        result = engine.recognize(image, config)
        ```

    Note:
        클라우드 API 사용 시 비용이 발생합니다.
        API 키 설정이 필요합니다.
    """

    def __init__(
        self,
        provider: str = "google",
        api_key: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_region: str = "us-east-1",
    ):
        """
        Cloud OCR 엔진 초기화

        Args:
            provider: OCR 제공자 ("google" 또는 "aws")
            api_key: Google Vision API 키 (provider="google"일 때)
            aws_access_key: AWS Access Key (provider="aws"일 때)
            aws_secret_key: AWS Secret Key (provider="aws"일 때)
            aws_region: AWS 리전 (기본: us-east-1)

        Raises:
            ImportError: 필요한 라이브러리가 설치되지 않은 경우
            ValueError: 잘못된 provider 또는 API 키 없음
        """
        super().__init__(name=f"CloudOCR-{provider.upper()}")
        self.provider = provider
        self.api_key = api_key
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_region = aws_region

        self._check_dependencies()
        self._validate_credentials()

    def _check_dependencies(self) -> None:
        """
        의존성 체크

        Raises:
            ImportError: 필요한 라이브러리가 설치되지 않은 경우
        """
        if self.provider == "google":
            try:
                from google.cloud import vision  # noqa: F401
            except ImportError:
                raise ImportError(
                    "google-cloud-vision is required for Google Vision API. "
                    "Install it with: pip install google-cloud-vision"
                )
        elif self.provider == "aws":
            try:
                import boto3  # noqa: F401
            except ImportError:
                raise ImportError(
                    "boto3 is required for AWS Textract. " "Install it with: pip install boto3"
                )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'google' or 'aws'")

    def _validate_credentials(self) -> None:
        """
        API 키 검증

        Raises:
            ValueError: API 키가 제공되지 않음
        """
        if self.provider == "google" and not self.api_key:
            logger.warning(
                "Google Vision API key not provided. "
                "Set GOOGLE_APPLICATION_CREDENTIALS environment variable."
            )
        elif self.provider == "aws" and (not self.aws_access_key or not self.aws_secret_key):
            logger.warning(
                "AWS credentials not provided. "
                "Configure AWS credentials via environment or ~/.aws/credentials"
            )

    def recognize(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """
        Cloud OCR로 텍스트 인식

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
            engine = CloudOCREngine(provider="google", api_key="YOUR_KEY")
            config = OCRConfig(language="ko")
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = engine.recognize(image, config)
            ```
        """
        if self.provider == "google":
            return self._recognize_google(image, config)
        elif self.provider == "aws":
            return self._recognize_aws(image, config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _recognize_google(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """Google Vision API로 OCR"""
        import io

        from google.cloud import vision
        from PIL import Image

        # numpy array를 PIL Image로 변환
        pil_image = Image.fromarray(image)

        # PIL Image를 bytes로 변환
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Google Vision 클라이언트
        client = vision.ImageAnnotatorClient()

        # 이미지 생성
        vision_image = vision.Image(content=img_byte_arr)

        # OCR 실행
        response = client.text_detection(image=vision_image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        if not texts:
            return {
                "text": "",
                "lines": [],
                "confidence": 0.0,
                "language": config.language,
                "metadata": {"engine": self.name, "empty_result": True},
            }

        # 첫 번째 요소는 전체 텍스트
        full_text = texts[0].description

        # 나머지는 개별 단어/라인
        lines = []
        for text in texts[1:]:
            vertices = text.bounding_poly.vertices
            x_coords = [v.x for v in vertices]
            y_coords = [v.y for v in vertices]

            bbox = BoundingBox(
                x0=min(x_coords), y0=min(y_coords), x1=max(x_coords), y1=max(y_coords), confidence=1.0
            )

            line = OCRTextLine(text=text.description, bbox=bbox, confidence=1.0, language=config.language)
            lines.append(line)

        return {
            "text": full_text,
            "lines": lines,
            "confidence": 1.0,
            "language": config.language,
            "metadata": {"engine": self.name, "provider": "google"},
        }

    def _recognize_aws(self, image: np.ndarray, config: OCRConfig) -> Dict:
        """AWS Textract로 OCR"""
        import io

        import boto3
        from PIL import Image

        # numpy array를 PIL Image로 변환
        pil_image = Image.fromarray(image)

        # PIL Image를 bytes로 변환
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # AWS Textract 클라이언트
        textract = boto3.client(
            "textract",
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
        )

        # OCR 실행
        response = textract.detect_document_text(Document={"Bytes": img_byte_arr})

        # 결과 변환
        lines = []
        text_parts = []
        total_confidence = 0.0
        valid_lines = 0

        for block in response["Blocks"]:
            if block["BlockType"] == "LINE":
                text = block["Text"]
                confidence = block["Confidence"] / 100.0  # 0-100 → 0-1

                if confidence < config.confidence_threshold:
                    continue

                bbox_data = block["Geometry"]["BoundingBox"]
                h, w = image.shape[:2]

                # 상대 좌표 → 절대 좌표
                bbox = BoundingBox(
                    x0=int(bbox_data["Left"] * w),
                    y0=int(bbox_data["Top"] * h),
                    x1=int((bbox_data["Left"] + bbox_data["Width"]) * w),
                    y1=int((bbox_data["Top"] + bbox_data["Height"]) * h),
                    confidence=confidence,
                )

                line = OCRTextLine(text=text, bbox=bbox, confidence=confidence, language=config.language)

                lines.append(line)
                text_parts.append(text)
                total_confidence += confidence
                valid_lines += 1

        full_text = "\n".join(text_parts)
        avg_confidence = total_confidence / valid_lines if valid_lines > 0 else 0.0

        return {
            "text": full_text,
            "lines": lines,
            "confidence": avg_confidence,
            "language": config.language,
            "metadata": {"engine": self.name, "provider": "aws"},
        }

    def __repr__(self) -> str:
        return f"CloudOCREngine(provider={self.provider})"
