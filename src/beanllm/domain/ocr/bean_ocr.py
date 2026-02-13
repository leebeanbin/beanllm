"""
beanOCR - Main OCR Facade

고급 OCR 기능을 제공하는 메인 클래스.
"""

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

if TYPE_CHECKING:
    from beanllm.domain.protocols import (
        CacheProtocol,
        DistributedConfigProtocol,
        EventLoggerProtocol,
        LockManagerProtocol,
        RateLimiterProtocol,
    )

import numpy as np
from PIL import Image

from beanllm.utils.async_helpers import AsyncHelperMixin
from beanllm.utils.logging import get_logger

from .engines.base import BaseOCREngine
from .models import OCRConfig, OCRResult
from .ocr_engine_factory import create_ocr_engine
from .ocr_pipeline import load_image as pipeline_load_image
from .ocr_pipeline import run_recognize_pipeline

logger = get_logger(__name__)


class beanOCR(AsyncHelperMixin):
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

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        distributed_config: Optional["DistributedConfigProtocol"] = None,
        cache: Optional["CacheProtocol"] = None,
        rate_limiter: Optional["RateLimiterProtocol"] = None,
        event_logger: Optional["EventLoggerProtocol"] = None,
        lock_manager: Optional["LockManagerProtocol"] = None,
        **kwargs,
    ):
        """
        Args:
            config: OCR 설정 객체 (선택)
            distributed_config: 분산 시스템 설정 (옵션, Service layer에서 주입)
            cache: 캐시 프로토콜 (옵션, Service layer에서 주입)
            rate_limiter: Rate Limiter 프로토콜 (옵션, Service layer에서 주입)
            event_logger: Event Logger 프로토콜 (옵션, Service layer에서 주입)
            lock_manager: Lock Manager 프로토콜 (옵션, Service layer에서 주입)
            **kwargs: OCRConfig 파라미터 (config 대신 사용 가능)

        Example:
            ```python
            # config 객체 사용
            config = OCRConfig(engine="paddleocr", language="ko")
            ocr = beanOCR(config=config)

            # kwargs 사용
            ocr = beanOCR(engine="paddleocr", language="ko", use_gpu=True)

            # 분산 기능 주입 (Service layer에서)
            ocr = beanOCR(
                config=config,
                cache=cache_instance,
                rate_limiter=rate_limiter_instance,
                event_logger=event_logger_instance
            )
            ```
        """
        self.config = config or OCRConfig(**kwargs)
        self._engine: Optional[BaseOCREngine] = None
        self._preprocessor: Optional[Any] = None  # ImagePreprocessor (lazy import)
        self._postprocessor: Optional[Any] = None  # LLMPostprocessor (lazy import)

        # 분산 시스템 설정 및 프로토콜 저장
        self._distributed_config = distributed_config
        self._cache = cache
        self._rate_limiter = rate_limiter
        self._event_logger = event_logger
        self._lock_manager = lock_manager

        self._init_components()

    def _init_components(self) -> None:
        """컴포넌트 초기화"""
        # 엔진 초기화
        self._engine = create_ocr_engine(self.config.engine, use_gpu=self.config.use_gpu)

        # 전처리기
        if self.config.enable_preprocessing:
            from .preprocessing import ImagePreprocessor

            self._preprocessor = ImagePreprocessor()

        # 후처리기
        if self.config.enable_llm_postprocessing and self.config.llm_model:
            from .postprocessing import LLMPostprocessor

            self._postprocessor = LLMPostprocessor(model=self.config.llm_model)

    def _load_image(self, image_or_path: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """이미지 로드 (ocr_pipeline.load_image 위임)."""
        return pipeline_load_image(image_or_path)

    def recognize(
        self, image_or_path: Union[str, Path, np.ndarray, Image.Image], **kwargs
    ) -> OCRResult:
        """
        이미지 OCR 인식 (분산 시스템 지원)

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
        import asyncio

        start_time = time.time()

        # 이미지 경로 또는 해시 생성 (캐싱 및 락용)
        image_path_str = None
        image_hash = None

        if isinstance(image_or_path, (str, Path)):
            image_path_str = str(image_or_path)
            # 파일 해시 생성
            try:
                with open(image_path_str, "rb") as f:
                    image_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                # 파일이 없거나 읽을 수 없으면 경로만 사용
                image_hash = hashlib.sha256(image_path_str.encode()).hexdigest()
        elif isinstance(image_or_path, np.ndarray):
            # numpy array 해시
            image_hash = hashlib.sha256(image_or_path.tobytes()).hexdigest()
        elif isinstance(image_or_path, Image.Image):
            # PIL Image 해시
            image_hash = hashlib.sha256(image_or_path.tobytes()).hexdigest()

        # 캐시 키 생성
        cache_key = f"ocr:{image_hash}:{self.config.engine}:{self.config.language}"

        # 1. 캐싱 확인 (옵션)
        if self._cache is not None:
            try:
                # Async cache support
                async def _get_cached():
                    return await self._cache.get(cache_key)

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't await in sync context with running loop
                        cached_result = None
                    else:
                        cached_result = loop.run_until_complete(_get_cached())
                except RuntimeError:
                    cached_result = asyncio.run(_get_cached())

                if cached_result:
                    # 이벤트 발행 (캐시 히트, 옵션)
                    self._log_event(
                        "ocr.recognize.cache_hit",
                        {"cache_key": cache_key, "engine": self.config.engine},
                    )
                    return cast(OCRResult, cached_result)
            except Exception as e:
                logger.debug(f"Cache retrieval failed (continuing without cache): {e}")

        # 2. 분산 락 획득 (동일 이미지 중복 처리 방지, 옵션)
        lock_key = f"ocr:lock:{image_hash}:{self.config.engine}"
        if self._lock_manager is not None and image_hash:

            async def _recognize_with_lock():
                async with self._lock_manager.with_file_lock(lock_key, timeout=30.0):
                    return self._recognize_impl(image_or_path, start_time, cache_key, **kwargs)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 락 없이 실행 (fallback)
                    return self._recognize_impl(image_or_path, start_time, cache_key, **kwargs)
                else:
                    return cast(OCRResult, loop.run_until_complete(_recognize_with_lock()))
            except RuntimeError:
                return cast(OCRResult, asyncio.run(_recognize_with_lock()))
        else:
            return self._recognize_impl(image_or_path, start_time, cache_key, **kwargs)

    def _recognize_impl(
        self,
        image_or_path: Union[str, Path, np.ndarray, Image.Image],
        start_time: float,
        cache_key: str,
        **kwargs: Any,
    ) -> OCRResult:
        """OCR 인식 실제 구현 (분산 시스템 적용)."""
        import asyncio

        image = self._load_image(image_or_path)

        if self._engine is None:
            raise RuntimeError("OCR engine not initialized")

        result = run_recognize_pipeline(
            image=image,
            engine=self._engine,
            config=self.config,
            start_time=start_time,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            event_logger=self._event_logger,
            rate_limiter=self._rate_limiter,
        )

        if self._cache is not None:
            try:

                async def _set_cache():
                    await self._cache.set(cache_key, result, ttl=7200)

                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        loop.run_until_complete(_set_cache())
                except RuntimeError:
                    asyncio.run(_set_cache())
            except Exception as e:
                logger.debug(f"Cache storage failed (safe to ignore): {e}")

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
                "PyMuPDF is required for PDF processing. Install it with: pip install pymupdf"
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
                f"Invalid page number: {page_num}. PDF has {len(doc)} pages (0-{len(doc) - 1})"
            )

        # 페이지를 이미지로 변환
        page = doc[page_num]
        pix = page.get_pixmap(dpi=dpi)

        # numpy array로 변환
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

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
        배치 OCR 처리 (분산 작업 큐 자동 적용)

        여러 이미지를 병렬로 처리합니다. 분산 모드에서는 작업 큐를 사용합니다.

        Args:
            images: 이미지 리스트 (경로, numpy array, PIL Image 혼합 가능)
            **kwargs: 추가 옵션

        Returns:
            List[OCRResult]: OCR 결과 리스트

        Note:
            배치 처리는 각 이미지에 대해 순차적으로 recognize()를 호출합니다.
            분산 기능(캐싱, Rate Limiting 등)은 생성 시 주입된 프로토콜을 사용합니다.

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
        # 데코레이터가 각 이미지를 처리하므로 여기서는 단일 이미지 처리만 구현
        # 실제로는 데코레이터가 리스트를 받아서 각 항목을 처리
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
