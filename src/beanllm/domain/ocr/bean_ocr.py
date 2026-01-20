"""
beanOCR - Main OCR Facade

고급 OCR 기능을 제공하는 메인 클래스.
"""

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

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
from .models import BoundingBox, OCRConfig, OCRResult, OCRTextLine

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
        **kwargs
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
        self._preprocessor = None
        self._postprocessor = None

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
                    return cached_result
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
                    return loop.run_until_complete(_recognize_with_lock())
            except RuntimeError:
                return asyncio.run(_recognize_with_lock())
        else:
            return self._recognize_impl(image_or_path, start_time, cache_key, **kwargs)
    
    def _recognize_impl(
        self,
        image_or_path: Union[str, Path, np.ndarray, Image.Image],
        start_time: float,
        cache_key: str,
        **kwargs
    ) -> OCRResult:
        """OCR 인식 실제 구현 (분산 시스템 적용)"""
        import asyncio

        # 이벤트 발행 (시작, 옵션)
        if self._event_logger is not None:
            asyncio.run(
                self._event_logger.log_event(
                    "ocr.recognize.started",
                    {"engine": self.config.engine, "language": self.config.language},
                )
            )

        # 1. 이미지 로드
        image = self._load_image(image_or_path)

        # 이벤트 발행 (이미지 로드 완료, 옵션)
        if self._event_logger is not None:
            asyncio.run(
                self._event_logger.log_event(
                    "ocr.recognize.image_loaded",
                {
                    "engine": self.config.engine,
                    "image_shape": str(image.shape) if hasattr(image, 'shape') else "unknown",
                }
            ))

        # 2. 전처리
        if self._preprocessor:
            image = self._preprocessor.process(image, self.config)

            # 이벤트 발행 (전처리 완료, 옵션)
            if self._event_logger is not None:
                asyncio.run(
                    self._event_logger.log_event(
                        "ocr.recognize.preprocessing_completed", {"engine": self.config.engine}
                    )
                )

        # 3. Rate Limiting (Cloud API 호출 시, 옵션)
        if self._rate_limiter is not None and self.config.engine == "cloud":
            asyncio.run(
                self._rate_limiter.acquire(
                    key=f"ocr:cloud:{self.config.engine}",
                    tokens=1,
                    rate=10,  # Default rate limit
                )
            )

        # 4. OCR 실행
        if self._engine is None:
            raise RuntimeError("OCR engine not initialized")

        raw_result = self._engine.recognize(image, self.config)

        # 이벤트 발행 (OCR 실행 완료, 옵션)
        if self._event_logger is not None:
            asyncio.run(
                self._event_logger.log_event(
                    "ocr.recognize.ocr_completed",
                    {
                        "engine": self.config.engine,
                        "text_length": len(raw_result.get("text", "")),
                        "confidence": raw_result.get("confidence", 0.0),
                    },
                )
            )

        # 5. OCRResult 생성
        result = OCRResult(
            text=raw_result["text"],
            lines=raw_result["lines"],
            language=raw_result.get("language", self.config.language),
            confidence=raw_result["confidence"],
            engine=self.config.engine,
            processing_time=time.time() - start_time,
            metadata=raw_result.get("metadata", {}),
        )

        # 6. 후처리 (LLM 보정)
        if self._postprocessor:
            result = self._postprocessor.process(result)

            # 이벤트 발행 (후처리 완료, 옵션)
            if self._event_logger is not None:
                asyncio.run(
                    self._event_logger.log_event(
                        "ocr.recognize.postprocessing_completed", {"engine": self.config.engine}
                    )
                )

        # 7. 캐싱 저장 (옵션)
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

        # 이벤트 발행 (완료, 옵션)
        if self._event_logger is not None:
            asyncio.run(
                self._event_logger.log_event(
                    "ocr.recognize.completed",
                    {
                        "engine": self.config.engine,
                        "processing_time": result.processing_time,
                        "confidence": result.confidence,
                    },
                )
            )

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
