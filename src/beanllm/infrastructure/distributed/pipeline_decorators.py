"""
파이프라인 분산 시스템 데코레이터 (core).

Helpers in pipeline_helpers.py, batch in pipeline_batch.py.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Optional, TypeVar, Union, cast

from beanllm.infrastructure.distributed.cache_wrapper import SyncCacheWrapper
from beanllm.infrastructure.distributed.config import get_distributed_config
from beanllm.infrastructure.distributed.factory import get_cache
from beanllm.infrastructure.distributed.lock_integration import get_lock_manager
from beanllm.infrastructure.distributed.pipeline_helpers import (
    execute_with_features_async as _execute_with_features_async,
    execute_with_features as _execute_with_features,
    generate_cache_key as _generate_cache_key,
    generate_lock_key as _generate_lock_key,
    publish_event as _publish_event,
    publish_event_async as _publish_event_async,
)
from beanllm.infrastructure.distributed.pipeline_batch import with_batch_processing

try:
    from beanllm.utils.logging import get_logger as _get_logger

    def get_logger(name: str) -> logging.Logger:
        return cast(logging.Logger, _get_logger(name))
except ImportError:

    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_logger(__name__)
T = TypeVar("T")

__all__ = ["with_distributed_features", "with_batch_processing"]


def with_distributed_features(
    pipeline_type: str = "default",
    enable_rate_limiting: Optional[bool] = None,
    enable_cache: Optional[bool] = None,
    enable_event_streaming: Optional[bool] = None,
    enable_distributed_lock: Optional[bool] = None,
    cache_key_prefix: Optional[str] = None,
    rate_limit_key: Optional[Union[str, Callable]] = None,
    lock_key: Optional[Union[str, Callable]] = None,
    event_type: Optional[str] = None,
    rate_limit_condition: Optional[Callable] = None,
):
    """
    파이프라인 함수에 분산 시스템 기능을 자동 적용하는 데코레이터

    Args:
        pipeline_type: 파이프라인 타입 ("ocr", "vision_rag", "multi_agent", "chain", "graph")
        enable_rate_limiting: Rate Limiting 활성화 (None이면 설정에서 가져옴)
        enable_cache: 캐싱 활성화 (None이면 설정에서 가져옴)
        enable_event_streaming: 이벤트 스트리밍 활성화 (None이면 설정에서 가져옴)
        enable_distributed_lock: 분산 락 활성화 (None이면 설정에서 가져옴)
        cache_key_prefix: 캐시 키 접두사 (None이면 pipeline_type 사용)
        rate_limit_key: Rate Limiting 키 (None이면 pipeline_type 사용)
        lock_key: 분산 락 키 (None이면 pipeline_type 사용)
        event_type: 이벤트 타입 (None이면 pipeline_type 사용)

    Example:
        ```python
        @with_distributed_features(
            pipeline_type="ocr",
            cache_key_prefix="ocr:recognize",
            event_type="ocr.recognize"
        )
        def recognize(self, image_path: str) -> OCRResult:
            # OCR 로직
            ...
        ```
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # 설정 가져오기
        config = get_distributed_config()
        pipeline_config = getattr(config, pipeline_type, None)

        if pipeline_config is None:
            # 기본 설정 사용
            pipeline_config = type(
                "DefaultConfig",
                (),
                {
                    "enable_rate_limiting": enable_rate_limiting or False,
                    "enable_cache": enable_cache or False,
                    "enable_event_streaming": enable_event_streaming or False,
                    "enable_distributed_lock": enable_distributed_lock or False,
                    "rate_limit_per_second": 10,
                    "cache_ttl": 3600,
                    "lock_timeout": 60.0,
                },
            )()

        # 설정 오버라이드
        if enable_rate_limiting is not None:
            pipeline_config.enable_rate_limiting = enable_rate_limiting
        if enable_cache is not None:
            pipeline_config.enable_cache = enable_cache
        if enable_event_streaming is not None:
            pipeline_config.enable_event_streaming = enable_event_streaming
        if enable_distributed_lock is not None:
            pipeline_config.enable_distributed_lock = enable_distributed_lock

        cache_prefix = cache_key_prefix or pipeline_type
        rate_key = rate_limit_key or pipeline_type
        lock_key_prefix = lock_key or pipeline_type
        event_prefix = event_type or pipeline_type

        # rate_limit_condition과 pipeline_type을 클로저로 저장
        stored_rate_limit_condition = rate_limit_condition
        stored_pipeline_type = pipeline_type

        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs) -> T:
            """동기 함수 래퍼"""
            # 캐시 키 생성
            cache_key = None
            if pipeline_config.enable_cache:
                cache_key = _generate_cache_key(cache_prefix, args, kwargs)

                # 캐시 확인
                sync_cache: SyncCacheWrapper = SyncCacheWrapper(
                    max_size=1000, ttl=pipeline_config.cache_ttl
                )
                try:
                    cached_result = sync_cache.get(cache_key)
                    if cached_result:
                        # 이벤트 발행 (캐시 히트)
                        if pipeline_config.enable_event_streaming:
                            _publish_event_async(
                                f"{event_prefix}.cache_hit",
                                {
                                    "cache_key": cache_key,
                                },
                            )
                        return cast(T, cached_result)
                except Exception as e:
                    logger.debug(f"Cache get failed: {e}")

            # 분산 락 키 생성
            lock_key_value: Optional[str] = None
            if pipeline_config.enable_distributed_lock:
                lock_prefix_str = (
                    lock_key_prefix if isinstance(lock_key_prefix, str) else pipeline_type
                )
                lock_key_value = _generate_lock_key(lock_prefix_str, args, kwargs)

            # rate_key가 callable이면 동적으로 생성
            actual_rate_key: Union[str, Callable[..., Any]] = rate_key
            if callable(rate_key):
                try:
                    actual_rate_key = rate_key(self, args, kwargs)
                    if actual_rate_key is None:
                        actual_rate_key = pipeline_type
                except Exception:
                    actual_rate_key = pipeline_type

            # 분산 락 획득 및 실행
            if lock_key_value:
                lock_manager = get_lock_manager()
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 이미 실행 중인 루프가 있으면 락 없이 실행 (fallback)
                        result = _execute_with_features(
                            func,
                            self,
                            args,
                            kwargs,
                            pipeline_config,
                            cache_key,
                            actual_rate_key,
                            event_prefix,
                        )
                    else:

                        async def _execute_with_lock():
                            async with lock_manager.with_file_lock(
                                lock_key_value, timeout=pipeline_config.lock_timeout
                            ):
                                return _execute_with_features(
                                    func,
                                    self,
                                    args,
                                    kwargs,
                                    pipeline_config,
                                    cache_key,
                                    actual_rate_key,
                                    event_prefix,
                                )

                        result = loop.run_until_complete(_execute_with_lock())
                except RuntimeError:
                    result = asyncio.run(_execute_with_lock())
            else:
                result = _execute_with_features(
                    func,
                    self,
                    args,
                    kwargs,
                    pipeline_config,
                    cache_key,
                    actual_rate_key,
                    event_prefix,
                )

            # 캐시 저장
            if pipeline_config.enable_cache and cache_key:
                sync_cache_store: SyncCacheWrapper = SyncCacheWrapper(
                    max_size=1000, ttl=pipeline_config.cache_ttl
                )
                try:
                    sync_cache_store.set(cache_key, result, ttl=pipeline_config.cache_ttl)
                except Exception as e:
                    logger.debug(f"Cache set failed: {e}")

            return cast(T, result)

        @functools.wraps(func)
        async def async_wrapper(self, *args: Any, **kwargs: Any) -> T:
            """비동기 함수 래퍼"""
            # 캐시 키 생성
            cache_key = None
            if pipeline_config.enable_cache:
                cache_key = _generate_cache_key(cache_prefix, args, kwargs)

                # 캐시 확인
                async_cache = get_cache()
                try:
                    cached_result = await async_cache.get(cache_key)
                    if cached_result:
                        # 이벤트 발행 (캐시 히트)
                        if pipeline_config.enable_event_streaming:
                            await _publish_event(
                                f"{event_prefix}.cache_hit",
                                {
                                    "cache_key": cache_key,
                                },
                            )
                        return cast(T, cached_result)
                except Exception as e:
                    logger.debug(f"Cache get failed: {e}")

            # 분산 락 키 생성
            lock_key_value = None
            if pipeline_config.enable_distributed_lock:
                lock_prefix_str = (
                    lock_key_prefix if isinstance(lock_key_prefix, str) else pipeline_type
                )
                lock_key_value = _generate_lock_key(lock_prefix_str, args, kwargs)

            # rate_key가 callable이면 동적으로 생성
            actual_rate_key_async: Union[str, Callable[..., Any]] = rate_key
            if callable(rate_key):
                try:
                    actual_rate_key_async = rate_key(self, args, kwargs)
                    if actual_rate_key_async is None:
                        actual_rate_key_async = pipeline_type
                except Exception:
                    actual_rate_key_async = pipeline_type

            # 분산 락 획득 및 실행
            if lock_key_value:
                lock_manager = get_lock_manager()
                async with lock_manager.with_file_lock(
                    lock_key_value, timeout=pipeline_config.lock_timeout
                ):
                    result = await _execute_with_features_async(
                        func,
                        self,
                        args,
                        kwargs,
                        pipeline_config,
                        cache_key,
                        actual_rate_key_async,
                        event_prefix,
                        stored_rate_limit_condition,
                        stored_pipeline_type,
                    )
            else:
                result = await _execute_with_features_async(
                    func,
                    self,
                    args,
                    kwargs,
                    pipeline_config,
                    cache_key,
                    actual_rate_key_async,
                    event_prefix,
                    stored_rate_limit_condition,
                    stored_pipeline_type,
                )

            # 캐시 저장
            if pipeline_config.enable_cache and cache_key:
                async_cache_store = get_cache()
                try:
                    await async_cache_store.set(cache_key, result, ttl=pipeline_config.cache_ttl)
                except Exception as e:
                    logger.debug(f"Cache set failed: {e}")

            return cast(T, result)

        # 동기/비동기 자동 감지
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[..., T], async_wrapper)
        return cast(Callable[..., T], sync_wrapper)

    return decorator
