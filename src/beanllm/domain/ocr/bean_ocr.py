"""
beanOCR - Main OCR Facade

고급 OCR 기능을 제공하는 메인 클래스.
"""

import time
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

from .engines.base import BaseOCREngine
from .models import BoundingBox, OCRConfig, OCRResult, OCRTextLine


class beanOCR:
    """
    통합 OCR 인터페이스

    7개 OCR 엔진을 통합하여 사용하기 쉬운 인터페이스 제공.

    Features:
    - 7개 OCR 엔진 지원 (PaddleOCR, EasyOCR, TrOCR, Nougat, Surya, Tesseract, Cloud)
    - 이미지 전처리 파이프라인
    - LLM 후처리로 98%+ 정확도
    - PDF 페이지 OCR
    - 배치 처리

    Example:
        ```python
        from beanllm.domain.ocr import beanOCR

        # 기본 사용
        ocr = beanOCR(engine="paddleocr", language="ko")
        result = ocr.recognize("scanned_image.jpg")
        print(result.text)
        print(f"Confidence: {result.confidence:.2%}")

        # LLM 후처리 활성화
        ocr = beanOCR(
            engine="paddleocr",
            enable_llm_postprocessing=True,
            llm_model="gpt-4o-mini"
        )
        result = ocr.recognize("noisy_image.jpg")

        # PDF 페이지 OCR
        result = ocr.recognize_pdf_page("document.pdf", page_num=0)

        # 배치 처리
        results = ocr.batch_recognize(["img1.jpg", "img2.jpg"])
        ```
    """

    def __init__(self, config: Optional[OCRConfig] = None, **kwargs):
        """
        Args:
            config: OCR 설정 객체 (선택)
            **kwargs: OCRConfig 파라미터 (config 대신 사용 가능)

        Example:
            ```python
            # config 객체 사용
            config = OCRConfig(engine="paddleocr", language="ko")
            ocr = beanOCR(config=config)

            # kwargs 사용
            ocr = beanOCR(engine="paddleocr", language="ko", use_gpu=True)
            ```
        """
        self.config = config or OCRConfig(**kwargs)
        self._engine: Optional[BaseOCREngine] = None
        self._preprocessor = None  # TODO: ImagePreprocessor 구현 후 초기화
        self._postprocessor = None  # TODO: LLMPostprocessor 구현 후 초기화
        self._init_components()

    def _init_components(self) -> None:
        """컴포넌트 초기화"""
        # 엔진 초기화
        self._engine = self._create_engine(self.config.engine)

        # 전처리기
        if self.config.enable_preprocessing:
            from .preprocessing import ImagePreprocessor

            self._preprocessor = ImagePreprocessor()

        # 후처리기
        if self.config.enable_llm_postprocessing:
            from .postprocessing import LLMPostprocessor

            self._postprocessor = LLMPostprocessor(
                model=self.config.llm_model
            )

    def _create_engine(self, engine_name: str) -> BaseOCREngine:
        """
        OCR 엔진 생성

        Args:
            engine_name: 엔진 이름

        Returns:
            BaseOCREngine: OCR 엔진 인스턴스

        Raises:
            ImportError: 엔진 의존성이 설치되지 않은 경우
            ValueError: 지원하지 않는 엔진
        """
        if engine_name == "paddleocr":
            try:
                from .engines.paddleocr_engine import PaddleOCREngine
                return PaddleOCREngine()
            except ImportError as e:
                raise ImportError(
                    f"PaddleOCR is required for engine '{engine_name}'. "
                    f"Install it with: pip install paddleocr"
                ) from e

        elif engine_name == "easyocr":
            try:
                from .engines.easyocr_engine import EasyOCREngine
                return EasyOCREngine()
            except ImportError as e:
                raise ImportError(
                    f"EasyOCR is required for engine '{engine_name}'. "
                    f"Install it with: pip install easyocr"
                ) from e

        elif engine_name == "tesseract":
            try:
                from .engines.tesseract_engine import TesseractEngine
                return TesseractEngine()
            except ImportError as e:
                raise ImportError(
                    f"pytesseract is required for engine '{engine_name}'. "
                    f"Install it with: pip install pytesseract\n"
                    f"Also install Tesseract OCR: brew install tesseract (macOS)"
                ) from e

        elif engine_name == "trocr":
            try:
                from .engines.trocr_engine import TrOCREngine
                return TrOCREngine()
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch"
                ) from e

        elif engine_name == "nougat":
            try:
                from .engines.nougat_engine import NougatEngine
                return NougatEngine()
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch"
                ) from e

        elif engine_name == "surya":
            try:
                from .engines.surya_engine import SuryaEngine
                return SuryaEngine()
            except ImportError as e:
                raise ImportError(
                    f"surya-ocr and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install surya-ocr torch"
                ) from e

        elif engine_name == "cloud-google":
            try:
                from .engines.cloud_engine import CloudOCREngine
                return CloudOCREngine(provider="google")
            except ImportError as e:
                raise ImportError(
                    f"google-cloud-vision is required for engine '{engine_name}'. "
                    f"Install it with: pip install google-cloud-vision"
                ) from e

        elif engine_name == "cloud-aws":
            try:
                from .engines.cloud_engine import CloudOCREngine
                return CloudOCREngine(provider="aws")
            except ImportError as e:
                raise ImportError(
                    f"boto3 is required for engine '{engine_name}'. "
                    f"Install it with: pip install boto3"
                ) from e

        elif engine_name in ["qwen2vl", "qwen2vl-2b"]:
            try:
                from .engines.qwen2vl_engine import Qwen2VLEngine
                return Qwen2VLEngine(model_size="2b", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch pillow qwen-vl-utils"
                ) from e

        elif engine_name == "qwen2vl-7b":
            try:
                from .engines.qwen2vl_engine import Qwen2VLEngine
                return Qwen2VLEngine(model_size="7b", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch pillow qwen-vl-utils"
                ) from e

        elif engine_name == "qwen2vl-72b":
            try:
                from .engines.qwen2vl_engine import Qwen2VLEngine
                return Qwen2VLEngine(model_size="72b", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch pillow qwen-vl-utils"
                ) from e

        elif engine_name == "minicpm":
            try:
                from .engines.minicpm_engine import MiniCPMEngine
                return MiniCPMEngine(use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers, torch, and pillow are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch pillow timm"
                ) from e

        elif engine_name == "deepseek-ocr":
            try:
                from .engines.deepseek_ocr_engine import DeepSeekOCREngine
                return DeepSeekOCREngine(use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers, torch, and pillow are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch pillow"
                ) from e

        # 지원하지 않는 엔진
        raise NotImplementedError(
            f"Engine '{engine_name}' is not yet implemented. "
            f"Currently supported: paddleocr, easyocr, tesseract, trocr, nougat, surya, "
            f"cloud-google, cloud-aws, qwen2vl-2b, qwen2vl-7b, qwen2vl-72b, minicpm, deepseek-ocr"
        )

    def _load_image(self, image_or_path: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        이미지 로드 및 numpy array로 변환

        Args:
            image_or_path: 이미지 경로, numpy array, 또는 PIL Image

        Returns:
            np.ndarray: 이미지 (numpy array)

        Raises:
            ValueError: 지원하지 않는 이미지 형식
            FileNotFoundError: 이미지 파일을 찾을 수 없음
        """
        # 이미 numpy array인 경우
        if isinstance(image_or_path, np.ndarray):
            return image_or_path

        # PIL Image인 경우
        if isinstance(image_or_path, Image.Image):
            # RGB로 변환 (RGBA, 그레이스케일 등 처리)
            if image_or_path.mode != "RGB":
                image_or_path = image_or_path.convert("RGB")
            return np.array(image_or_path)

        # 경로인 경우
        path = Path(image_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        # PIL로 이미지 로드
        img = Image.open(path)

        # RGB로 변환 (RGBA, 그레이스케일 등 처리)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return np.array(img)

    def recognize(self, image_or_path: Union[str, Path, np.ndarray, Image.Image], **kwargs) -> OCRResult:
        """
        이미지 OCR 인식

        Args:
            image_or_path: 이미지 경로, numpy array, 또는 PIL Image
            **kwargs: 추가 옵션 (config 오버라이드)

        Returns:
            OCRResult: OCR 결과

        Raises:
            FileNotFoundError: 이미지 파일을 찾을 수 없음
            ValueError: 잘못된 이미지 형식
            ImportError: OCR 엔진 의존성 미설치

        Example:
            ```python
            # 이미지 파일 경로
            result = ocr.recognize("scanned_image.jpg")

            # numpy array
            import cv2
            image = cv2.imread("image.jpg")
            result = ocr.recognize(image)

            # PIL Image
            from PIL import Image
            img = Image.open("image.jpg")
            result = ocr.recognize(img)
            ```
        """
        start_time = time.time()

        # 1. 이미지 로드
        image = self._load_image(image_or_path)

        # 2. 전처리
        if self._preprocessor:
            image = self._preprocessor.process(image, self.config)

        # 3. OCR 실행
        if self._engine is None:
            raise RuntimeError("OCR engine not initialized")

        raw_result = self._engine.recognize(image, self.config)

        # 4. OCRResult 생성
        result = OCRResult(
            text=raw_result["text"],
            lines=raw_result["lines"],
            language=raw_result.get("language", self.config.language),
            confidence=raw_result["confidence"],
            engine=self.config.engine,
            processing_time=time.time() - start_time,
            metadata=raw_result.get("metadata", {}),
        )

        # 5. 후처리 (LLM 보정)
        if self._postprocessor:
            result = self._postprocessor.process(result)

        return result

    def recognize_pdf_page(
        self, pdf_path: Union[str, Path], page_num: int = 0, dpi: int = 300
    ) -> OCRResult:
        """
        PDF 페이지 OCR

        Args:
            pdf_path: PDF 파일 경로
            page_num: 페이지 번호 (0부터 시작)
            dpi: 렌더링 해상도 (기본: 300, 높을수록 정확하지만 느림)

        Returns:
            OCRResult: OCR 결과

        Raises:
            FileNotFoundError: PDF 파일을 찾을 수 없음
            ImportError: PyMuPDF (fitz) 미설치
            IndexError: 잘못된 페이지 번호

        Example:
            ```python
            # 첫 페이지 OCR
            result = ocr.recognize_pdf_page("document.pdf", page_num=0)

            # 고해상도 OCR
            result = ocr.recognize_pdf_page("document.pdf", page_num=0, dpi=600)
            ```
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install it with: pip install pymupdf"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # PDF 열기
        doc = fitz.open(pdf_path)

        # 페이지 번호 검증
        if page_num < 0 or page_num >= len(doc):
            doc.close()
            raise IndexError(
                f"Invalid page number: {page_num}. "
                f"PDF has {len(doc)} pages (0-{len(doc)-1})"
            )

        # 페이지를 이미지로 변환
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)

        # numpy array로 변환
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # RGB로 변환 (PyMuPDF는 RGB 또는 RGBA 반환)
        if pix.n == 4:  # RGBA
            image = image[:, :, :3]  # Alpha 채널 제거

        doc.close()

        # OCR 실행
        return self.recognize(image)

    def batch_recognize(
        self, images: List[Union[str, Path, np.ndarray, Image.Image]], **kwargs
    ) -> List[OCRResult]:
        """
        배치 OCR 처리

        여러 이미지를 순차적으로 처리합니다.

        Args:
            images: 이미지 리스트 (경로, numpy array, PIL Image 혼합 가능)
            **kwargs: 추가 옵션

        Returns:
            List[OCRResult]: OCR 결과 리스트

        Example:
            ```python
            # 이미지 파일 배치 처리
            results = ocr.batch_recognize([
                "page1.jpg",
                "page2.jpg",
                "page3.jpg"
            ])

            for i, result in enumerate(results):
                print(f"Page {i+1}: {result.text[:50]}...")
            ```
        """
        results = []
        for img in images:
            result = self.recognize(img, **kwargs)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return (
            f"beanOCR(engine={self.config.engine}, "
            f"language={self.config.language}, "
            f"gpu={self.config.use_gpu})"
        )
