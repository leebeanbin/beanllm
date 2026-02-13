"""
Marker Engine - Cache & Optimization Mixin

캐싱, 모델 로딩, GPU 메모리 관리를 담당합니다.
"""

import gc
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    pass

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


class MarkerCacheMixin:
    """
    marker-pdf 결과 캐싱 및 GPU 메모리 관리를 담당하는 Mixin.

    - 결과 캐시 (LRU)
    - 모델 캐시
    - GPU 메모리 정리
    """

    # Mixin이 참조하는 속성들 (MarkerEngine에서 초기화됨)
    max_pages: Optional[int]
    cache_size: int
    enable_cache: bool
    use_gpu: bool
    batch_size: int
    _result_cache: Dict[str, Dict[str, Any]]
    _model_cache: Optional[Any]

    def _get_cache_key(self, pdf_path: Path, config: Dict) -> str:
        """
        PDF와 설정으로부터 캐시 키 생성

        Args:
            pdf_path: PDF 파일 경로
            config: 설정 딕셔너리

        Returns:
            str: 해시 기반 캐시 키
        """
        file_stat = pdf_path.stat()
        key_data = f"{pdf_path}:{file_stat.st_mtime}:{file_stat.st_size}"
        key_data += f":{config.get('to_markdown', True)}"
        key_data += f":{config.get('extract_tables', True)}"
        key_data += f":{config.get('extract_images', True)}"
        key_data += f":{config.get('max_pages', self.max_pages)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: Dict) -> None:
        """
        결과를 캐시에 저장

        Args:
            cache_key: 캐시 키
            result: 캐싱할 결과 딕셔너리
        """
        if len(self._result_cache) >= self.cache_size:
            oldest_key = next(iter(self._result_cache))
            del self._result_cache[oldest_key]
            logger.debug(f"Cache evicted: {oldest_key}")

        self._result_cache[cache_key] = result.copy()
        logger.debug(
            f"Result cached: {cache_key[:8]}... "
            f"(cache size: {len(self._result_cache)}/{self.cache_size})"
        )

    def _load_models_cached(self) -> Any:
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

            gc.collect()

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU memory: {e}")

    def clear_cache(self) -> None:
        """
        모든 캐시 수동 정리

        메모리를 확보하거나 캐시를 리셋할 때 사용합니다.
        """
        self._result_cache.clear()
        logger.info("Result cache cleared")

        if self._model_cache is not None:
            self._model_cache = None
            logger.info("Model cache cleared")

            if self.use_gpu:
                self._cleanup_gpu_memory()

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
    ) -> List[Optional[Dict]]:
        """
        여러 PDF를 배치로 처리

        Args:
            pdf_paths: PDF 파일 경로 리스트
            config: 추출 설정 딕셔너리

        Returns:
            List[Dict]: 각 PDF의 추출 결과 리스트 (실패 시 해당 위치는 None)
        """
        results: List[Optional[Dict]] = []
        total = len(pdf_paths)

        logger.info(f"Processing {total} PDFs in batch mode...")

        for i, pdf_path in enumerate(pdf_paths, 1):
            try:
                logger.debug(f"Processing [{i}/{total}]: {pdf_path}")
                result = self.extract(pdf_path, config)  # type: ignore[attr-defined]
                results.append(result)

                if i % 5 == 0 or i == total:
                    logger.info(f"Batch progress: {i}/{total} PDFs processed")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                results.append(None)

            if self.use_gpu and i % self.batch_size == 0:
                self._cleanup_gpu_memory()

        logger.info(
            f"Batch processing completed: "
            f"{len([r for r in results if r is not None])}/{total} succeeded"
        )

        return results
