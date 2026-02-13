"""
Marker Engine - ML Layer

marker-pdf 라이브러리를 사용한 ML 기반 PDF 파싱 엔진

Features:
- 구조 보존 Markdown 변환
- 98% 정확도
- ~10초/100 pages (GPU)
- 복잡한 레이아웃 처리
- GPU 메모리 관리 & 캐싱
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from beanllm.domain.loaders.pdf.engines.base import BasePDFEngine
from beanllm.domain.loaders.pdf.engines.marker_cache import MarkerCacheMixin
from beanllm.domain.loaders.pdf.engines.marker_postprocessing import (
    MarkerPostprocessingMixin,
)

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class MarkerEngine(MarkerCacheMixin, MarkerPostprocessingMixin, BasePDFEngine):
    """
    marker-pdf 기반 ML Layer PDF 파싱 엔진

    marker-pdf를 사용하여 구조를 보존한 Markdown 변환을 수행합니다.
    복잡한 레이아웃과 표, 이미지가 많은 문서에 적합합니다.

    Example:
        ```python
        from beanllm.domain.loaders.pdf.engines import MarkerEngine

        engine = MarkerEngine(use_gpu=True)
        result = engine.extract("document.pdf", {
            "to_markdown": True,
            "extract_tables": True,
        })
        ```

    Note:
        marker-pdf 라이브러리가 설치되어 있어야 합니다:
        ```bash
        pip install marker-pdf
        ```
    """

    def __init__(
        self,
        use_gpu: bool = False,
        batch_size: int = 1,
        max_pages: Optional[int] = None,
        enable_cache: bool = True,
        cache_size: int = 10,
        name: Optional[str] = None,
    ):
        """
        Args:
            use_gpu: GPU 사용 여부 (기본: False, CPU 사용)
            batch_size: 배치 처리 크기 (기본: 1)
            max_pages: 최대 처리 페이지 수
            enable_cache: 캐싱 활성화 여부 (기본: True)
            cache_size: 캐시 최대 크기 (기본: 10개 문서)
            name: 엔진 이름 (기본: "Marker")
        """
        super().__init__(name=name or "Marker")
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_pages = max_pages
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._marker_available: Optional[bool] = None
        self._model_cache: Optional[Any] = None  # marker-pdf 모델 캐시
        self._result_cache: Dict[str, Dict] = {}  # 결과 캐시

    def _check_dependencies(self) -> None:
        """marker-pdf 라이브러리 확인"""
        try:
            import marker
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models

            self._marker_available = True
            logger.debug("marker-pdf library is available")
        except ImportError:
            self._marker_available = False
            raise ImportError(
                "marker-pdf is required for MarkerEngine. Install it with: pip install marker-pdf"
            )

    def extract(
        self,
        pdf_path: Union[str, Path],
        config: Dict,
    ) -> Dict:
        """
        marker-pdf를 사용한 PDF 추출

        Args:
            pdf_path: PDF 파일 경로
            config: 추출 설정 딕셔너리
                - to_markdown: bool - Markdown 변환 여부 (기본: True)
                - extract_tables: bool - 테이블 추출 여부 (기본: True)
                - extract_images: bool - 이미지 추출 여부 (기본: True)
                - max_pages: Optional[int] - 최대 페이지 수

        Returns:
            Dict: pages, tables, images, markdown, metadata

        Raises:
            FileNotFoundError: PDF 파일이 없을 때
            ImportError: marker-pdf가 설치되지 않았을 때
            Exception: PDF 파싱 실패 시
        """
        start_time = time.time()
        pdf_path = self._validate_pdf_path(pdf_path)

        if self._marker_available is None:
            self._check_dependencies()

        if not self._marker_available:
            raise ImportError(
                "marker-pdf is not available. Install it with: pip install marker-pdf"
            )

        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
        except ImportError as e:
            raise ImportError(
                f"Failed to import marker-pdf: {e}. Install it with: pip install marker-pdf"
            )

        max_pages = config.get("max_pages", self.max_pages)

        try:
            cache_key = None
            if self.enable_cache:
                cache_key = self._get_cache_key(pdf_path, config)
                if cache_key in self._result_cache:
                    logger.debug(f"Cache hit for {pdf_path}")
                    cached_result = self._result_cache[cache_key].copy()
                    cached_result["metadata"]["from_cache"] = True
                    return cached_result

            logger.debug(f"Loading marker-pdf models (GPU: {self.use_gpu})...")
            model_list = self._load_models_cached()

            logger.debug(f"Converting PDF with marker-pdf: {pdf_path}")
            full_text, images, metadata = convert_single_pdf(
                str(pdf_path),
                model_list,
                max_pages=max_pages,
                langs=None,
            )

            result = self._convert_marker_result(
                full_text=full_text,
                images=images,
                marker_metadata=metadata,
                config=config,
            )

            processing_time = time.time() - start_time
            result["metadata"]["processing_time"] = processing_time
            result["metadata"]["engine"] = self.name
            result["metadata"]["use_gpu"] = self.use_gpu
            result["metadata"]["from_cache"] = False

            logger.info(
                f"MarkerEngine extracted {len(result['pages'])} pages in {processing_time:.2f}s"
            )

            if self.enable_cache and cache_key:
                self._cache_result(cache_key, result)

            if self.use_gpu:
                self._cleanup_gpu_memory()

            return result

        except Exception as e:
            logger.error(f"MarkerEngine extraction failed: {e}")
            if self.use_gpu:
                self._cleanup_gpu_memory()
            raise
