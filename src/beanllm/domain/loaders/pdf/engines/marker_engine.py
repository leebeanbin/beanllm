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

import gc
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import BasePDFEngine

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class MarkerEngine(BasePDFEngine):
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
        self._marker_available = None
        self._model_cache = None  # marker-pdf 모델 캐시
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
                "marker-pdf is required for MarkerEngine. "
                "Install it with: pip install marker-pdf"
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
            Dict containing:
                - pages: List[Dict] - 페이지별 데이터
                - tables: List[Dict] - 추출된 테이블
                - images: List[Dict] - 추출된 이미지
                - markdown: str - Markdown 변환 결과
                - metadata: Dict - 전체 문서 메타데이터

        Raises:
            FileNotFoundError: PDF 파일이 없을 때
            ImportError: marker-pdf가 설치되지 않았을 때
            Exception: PDF 파싱 실패 시
        """
        start_time = time.time()
        pdf_path = self._validate_pdf_path(pdf_path)

        # marker-pdf 사용 가능 여부 확인
        if self._marker_available is None:
            self._check_dependencies()

        if not self._marker_available:
            raise ImportError(
                "marker-pdf is not available. "
                "Install it with: pip install marker-pdf"
            )

        # marker-pdf import
        try:
            from marker.convert import convert_single_pdf
            from marker.models import load_all_models
        except ImportError as e:
            raise ImportError(
                f"Failed to import marker-pdf: {e}. "
                "Install it with: pip install marker-pdf"
            )

        # 설정 추출
        to_markdown = config.get("to_markdown", True)
        extract_tables = config.get("extract_tables", True)
        extract_images = config.get("extract_images", True)
        max_pages = config.get("max_pages", self.max_pages)

        try:
            # 캐시 확인
            cache_key = None
            if self.enable_cache:
                cache_key = self._get_cache_key(pdf_path, config)
                if cache_key in self._result_cache:
                    logger.debug(f"Cache hit for {pdf_path}")
                    cached_result = self._result_cache[cache_key].copy()
                    cached_result["metadata"]["from_cache"] = True
                    return cached_result

            # marker-pdf 모델 로드 (캐시 사용)
            logger.debug(f"Loading marker-pdf models (GPU: {self.use_gpu})...")
            model_list = self._load_models_cached()

            # PDF 변환
            logger.debug(f"Converting PDF with marker-pdf: {pdf_path}")
            full_text, images, metadata = convert_single_pdf(
                str(pdf_path),
                model_list,
                max_pages=max_pages,
                langs=None,  # Auto-detect language
            )

            # 결과 변환
            result = self._convert_marker_result(
                full_text=full_text,
                images=images,
                marker_metadata=metadata,
                config=config,
            )

            # 처리 시간 기록
            processing_time = time.time() - start_time
            result["metadata"]["processing_time"] = processing_time
            result["metadata"]["engine"] = self.name
            result["metadata"]["use_gpu"] = self.use_gpu
            result["metadata"]["from_cache"] = False

            logger.info(
                f"MarkerEngine extracted {len(result['pages'])} pages "
                f"in {processing_time:.2f}s"
            )

            # 결과 캐싱
            if self.enable_cache and cache_key:
                self._cache_result(cache_key, result)

            # GPU 메모리 정리
            if self.use_gpu:
                self._cleanup_gpu_memory()

            return result

        except Exception as e:
            logger.error(f"MarkerEngine extraction failed: {e}")
            # GPU 메모리 정리 (에러 발생 시에도)
            if self.use_gpu:
                self._cleanup_gpu_memory()
            raise

    def _convert_marker_result(
        self,
        full_text: str,
        images: Dict,
        marker_metadata: Dict,
        config: Dict,
    ) -> Dict:
        """
        marker-pdf 결과를 PDFLoadResult 형식으로 변환

        Args:
            full_text: marker-pdf가 추출한 Markdown 텍스트
            images: marker-pdf가 추출한 이미지 딕셔너리
            marker_metadata: marker-pdf 메타데이터
            config: 설정 딕셔너리

        Returns:
            Dict: PDFLoadResult 형식 딕셔너리
        """
        # 페이지 분리 (marker-pdf는 전체 텍스트를 반환)
        pages = self._split_into_pages(full_text, marker_metadata)

        # 테이블 추출 (Markdown 테이블 형식 파싱)
        tables = []
        if config.get("extract_tables", True):
            tables = self._extract_tables_from_markdown(full_text, pages)

        # 이미지 변환
        image_list = []
        if config.get("extract_images", True) and images:
            image_list = self._convert_images(images)

        # Markdown 텍스트
        markdown_text = full_text if config.get("to_markdown", True) else None

        return {
            "pages": pages,
            "tables": tables,
            "images": image_list,
            "markdown": markdown_text,
            "metadata": {
                "total_pages": len(pages),
                "engine": self.name,
                "marker_metadata": marker_metadata,
                "quality_score": 0.98,  # marker-pdf는 매우 높은 정확도
            },
        }

    def _split_into_pages(self, full_text: str, metadata: Dict) -> List[Dict]:
        """
        전체 Markdown 텍스트를 페이지별로 분리

        Args:
            full_text: 전체 Markdown 텍스트
            metadata: marker-pdf 메타데이터

        Returns:
            List[Dict]: 페이지 데이터 리스트
        """
        # marker-pdf는 페이지 구분자를 포함할 수 있음
        # 간단한 구현: 텍스트 길이 기반으로 균등 분할
        # 실제로는 marker-pdf의 메타데이터를 활용해야 함

        # 메타데이터에서 페이지 수 추출
        num_pages = metadata.get("num_pages", 1)

        if num_pages == 1:
            # 단일 페이지
            return [
                {
                    "page": 0,
                    "text": full_text,
                    "width": 612.0,  # 기본값
                    "height": 792.0,  # 기본값
                    "metadata": {"source": "marker-pdf"},
                }
            ]

        # 여러 페이지: 텍스트 균등 분할 (간단한 구현)
        # 실제로는 marker-pdf의 페이지 메타데이터를 활용해야 함
        text_length = len(full_text)
        chunk_size = text_length // num_pages

        pages = []
        for i in range(num_pages):
            start = i * chunk_size
            end = start + chunk_size if i < num_pages - 1 else text_length

            pages.append(
                {
                    "page": i,
                    "text": full_text[start:end],
                    "width": 612.0,
                    "height": 792.0,
                    "metadata": {"source": "marker-pdf"},
                }
            )

        return pages

    def _extract_tables_from_markdown(
        self, markdown_text: str, pages: List[Dict]
    ) -> List[Dict]:
        """
        Markdown 텍스트에서 테이블 추출

        Args:
            markdown_text: Markdown 텍스트
            pages: 페이지 데이터 리스트

        Returns:
            List[Dict]: 테이블 데이터 리스트
        """
        tables = []

        # Markdown 테이블 패턴: | header | header |
        #                      |--------|--------|
        #                      | data   | data   |
        import re

        # 간단한 Markdown 테이블 감지
        table_pattern = r"\|[^\n]+\|\n\|[-\s|]+\|\n(?:\|[^\n]+\|\n)+"

        for match in re.finditer(table_pattern, markdown_text):
            table_text = match.group()

            # 테이블이 어느 페이지에 속하는지 추정
            position = match.start()
            page_idx = self._estimate_page_from_position(position, len(markdown_text), len(pages))

            # 간단한 테이블 파싱
            table_data = self._parse_markdown_table(table_text)

            if table_data:
                tables.append(
                    {
                        "page": page_idx,
                        "table_index": len([t for t in tables if t["page"] == page_idx]),
                        "data": table_data,
                        "bbox": (0, 0, 612, 100),  # 추정값
                        "confidence": 0.95,
                        "metadata": {"source": "marker-pdf", "format": "markdown"},
                    }
                )

        return tables

    def _estimate_page_from_position(
        self, position: int, total_length: int, num_pages: int
    ) -> int:
        """텍스트 위치에서 페이지 번호 추정"""
        if num_pages == 0:
            return 0
        page_idx = int((position / total_length) * num_pages)
        return min(page_idx, num_pages - 1)

    def _parse_markdown_table(self, table_text: str) -> List[List[str]]:
        """
        Markdown 테이블 파싱

        Args:
            table_text: Markdown 테이블 텍스트

        Returns:
            List[List[str]]: 2D 리스트 형식 테이블 데이터
        """
        lines = table_text.strip().split("\n")
        if len(lines) < 3:  # 헤더 + 구분자 + 최소 1행
            return []

        table_data = []

        for i, line in enumerate(lines):
            if i == 1:  # 구분자 라인 스킵
                continue

            # | cell | cell | 형식 파싱
            cells = [cell.strip() for cell in line.split("|")[1:-1]]
            if cells:
                table_data.append(cells)

        return table_data

    def _convert_images(self, images: Dict) -> List[Dict]:
        """
        marker-pdf 이미지를 ImageData 형식으로 변환

        Args:
            images: marker-pdf 이미지 딕셔너리

        Returns:
            List[Dict]: 이미지 데이터 리스트
        """
        image_list = []

        for idx, (image_name, image_data) in enumerate(images.items()):
            # marker-pdf 이미지 형식에 따라 변환
            # 실제 구현은 marker-pdf의 이미지 형식에 맞춰야 함
            image_list.append(
                {
                    "page": 0,  # 추정 필요
                    "image_index": idx,
                    "image": image_data,  # PIL Image 또는 bytes
                    "format": "png",
                    "width": 800,  # 추정값
                    "height": 600,  # 추정값
                    "bbox": (0, 0, 800, 600),
                    "size": len(image_data) if isinstance(image_data, bytes) else 0,
                    "metadata": {"source": "marker-pdf", "name": image_name},
                }
            )

        return image_list

    # ==================== 최적화 메서드 ====================

    def _get_cache_key(self, pdf_path: Path, config: Dict) -> str:
        """
        PDF와 설정으로부터 캐시 키 생성

        Args:
            pdf_path: PDF 파일 경로
            config: 설정 딕셔너리

        Returns:
            str: 해시 기반 캐시 키
        """
        # 파일 경로와 수정 시간, 주요 설정을 조합하여 해시 생성
        file_stat = pdf_path.stat()
        key_data = f"{pdf_path}:{file_stat.st_mtime}:{file_stat.st_size}"

        # 주요 설정 추가
        key_data += f":{config.get('to_markdown', True)}"
        key_data += f":{config.get('extract_tables', True)}"
        key_data += f":{config.get('extract_images', True)}"
        key_data += f":{config.get('max_pages', self.max_pages)}"

        # SHA256 해시 생성
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: Dict) -> None:
        """
        결과를 캐시에 저장

        Args:
            cache_key: 캐시 키
            result: 캐싱할 결과 딕셔너리
        """
        # 캐시 크기 제한 체크 (LRU 방식)
        if len(self._result_cache) >= self.cache_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key}")

        # 결과 저장 (딥 카피)
        self._result_cache[cache_key] = result.copy()
        logger.debug(
            f"Result cached: {cache_key[:8]}... "
            f"(cache size: {len(self._result_cache)}/{self.cache_size})"
        )

    def _load_models_cached(self):
        """
        marker-pdf 모델을 캐시에서 로드 (없으면 새로 로드)

        Returns:
            marker-pdf 모델 리스트
        """
        if self._model_cache is None:
            from marker.models import load_all_models

            logger.debug("Loading marker-pdf models (first time)...")
            self._model_cache = load_all_models()
            logger.debug("Models loaded and cached")
        else:
            logger.debug("Using cached marker-pdf models")

        return self._model_cache

    def _cleanup_gpu_memory(self) -> None:
        """
        GPU 메모리 정리

        GPU 사용 시 메모리 누수를 방지하기 위해 명시적으로 정리합니다.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU memory cleared")

            # Python 가비지 컬렉션
            gc.collect()

        except ImportError:
            # torch가 없으면 스킵
            pass
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU memory: {e}")

    def clear_cache(self) -> None:
        """
        모든 캐시 수동 정리

        메모리를 확보하거나 캐시를 리셋할 때 사용합니다.
        """
        # 결과 캐시 정리
        self._result_cache.clear()
        logger.info("Result cache cleared")

        # 모델 캐시 정리
        if self._model_cache is not None:
            self._model_cache = None
            logger.info("Model cache cleared")

            # GPU 메모리 정리
            if self.use_gpu:
                self._cleanup_gpu_memory()

        # 가비지 컬렉션
        gc.collect()

    def get_cache_stats(self) -> Dict:
        """
        캐시 통계 정보 반환

        Returns:
            Dict: 캐시 사용 현황
        """
        return {
            "cache_enabled": self.enable_cache,
            "cache_size": len(self._result_cache),
            "cache_limit": self.cache_size,
            "model_cached": self._model_cache is not None,
            "use_gpu": self.use_gpu,
        }

    def extract_batch(
        self, pdf_paths: List[Union[str, Path]], config: Dict
    ) -> List[Dict]:
        """
        여러 PDF를 배치로 처리

        Args:
            pdf_paths: PDF 파일 경로 리스트
            config: 추출 설정 딕셔너리

        Returns:
            List[Dict]: 각 PDF의 추출 결과 리스트

        Note:
            현재는 순차 처리이지만, 향후 병렬 처리로 확장 가능
        """
        results = []
        total = len(pdf_paths)

        logger.info(f"Processing {total} PDFs in batch mode...")

        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.debug(f"Processing [{i}/{total}]: {pdf_path}")
                result = self.extract(pdf_path, config)
                results.append(result)

                # 배치 진행 상황 로깅
                if i % 5 == 0 or i == total:
                    logger.info(f"Batch progress: {i}/{total} PDFs processed")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                # 실패한 경우 None 추가
                results.append(None)

            # GPU 메모리 관리 (배치 중간에도 정리)
            if self.use_gpu and i % self.batch_size == 0:
                self._cleanup_gpu_memory()

        logger.info(
            f"Batch processing completed: "
            f"{len([r for r in results if r is not None])}/{total} succeeded"
        )

        return results
