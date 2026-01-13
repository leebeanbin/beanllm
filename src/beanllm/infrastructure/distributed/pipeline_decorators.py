"""
파이프라인 분산 시스템 데코레이터

각 파이프라인에 분산 시스템 기능을 쉽게 적용할 수 있는 데코레이터 제공
- Rate Limiting
- Caching
- Event Streaming
- Distributed Lock
- Task Queue (배치 처리)

중복 코드를 최소화하고 일관된 패턴 제공
"""

import asyncio
import functools
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List

from .config import DistributedConfig, get_distributed_config
from .factory import get_rate_limiter, get_cache
from .event_integration import get_event_logger
from .lock_integration import get_lock_manager
from .cache_wrapper import SyncCacheWrapper
from .task_processor import BatchProcessor

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

T = TypeVar("T")


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
                cache = SyncCacheWrapper(max_size=1000, ttl=pipeline_config.cache_ttl)
                try:
                    cached_result = cache.get(cache_key)
                    if cached_result:
                        # 이벤트 발행 (캐시 히트)
                        if pipeline_config.enable_event_streaming:
                            _publish_event_async(
                                f"{event_prefix}.cache_hit",
                                {
                                    "cache_key": cache_key,
                                },
                            )
                        return cached_result
                except Exception as e:
                    logger.debug(f"Cache get failed: {e}")

            # 분산 락 키 생성
            lock_key_value = None
            if pipeline_config.enable_distributed_lock:
                lock_key_value = _generate_lock_key(lock_key_prefix, args, kwargs)

            # rate_key가 callable이면 동적으로 생성
            actual_rate_key = rate_key
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
                cache = SyncCacheWrapper(max_size=1000, ttl=pipeline_config.cache_ttl)
                try:
                    cache.set(cache_key, result, ttl=pipeline_config.cache_ttl)
                except Exception as e:
                    logger.debug(f"Cache set failed: {e}")

            return result

        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs) -> T:
            """비동기 함수 래퍼"""
            # 캐시 키 생성
            cache_key = None
            if pipeline_config.enable_cache:
                cache_key = _generate_cache_key(cache_prefix, args, kwargs)

                # 캐시 확인
                cache = get_cache()
                try:
                    cached_result = await cache.get(cache_key)
                    if cached_result:
                        # 이벤트 발행 (캐시 히트)
                        if pipeline_config.enable_event_streaming:
                            await _publish_event(
                                f"{event_prefix}.cache_hit",
                                {
                                    "cache_key": cache_key,
                                },
                            )
                        return cached_result
                except Exception as e:
                    logger.debug(f"Cache get failed: {e}")

            # 분산 락 키 생성
            lock_key_value = None
            if pipeline_config.enable_distributed_lock:
                lock_key_value = _generate_lock_key(lock_key_prefix, args, kwargs)

            # rate_key가 callable이면 동적으로 생성 (비동기 래퍼에서 처리하지만 여기서도 미리 생성)
            actual_rate_key = rate_key
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
                        actual_rate_key,
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
                    actual_rate_key,
                    event_prefix,
                    stored_rate_limit_condition,
                    stored_pipeline_type,
                )

            # 캐시 저장
            if pipeline_config.enable_cache and cache_key:
                cache = get_cache()
                try:
                    await cache.set(cache_key, result, ttl=pipeline_config.cache_ttl)
                except Exception as e:
                    logger.debug(f"Cache set failed: {e}")

            return result

        # 동기/비동기 자동 감지
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _generate_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """캐시 키 생성"""
    import json

    # 첫 번째 인자가 이미지 경로나 파일일 수 있음
    key_data = {"args": [], "kwargs": kwargs}

    for arg in args:
        if isinstance(arg, (str, Path)):
            # 파일 경로는 해시 사용
            try:
                with open(str(arg), "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                key_data["args"].append(f"file:{file_hash}")
            except Exception:
                key_data["args"].append(str(arg))
        elif hasattr(arg, "tobytes"):
            # numpy array나 PIL Image
            key_data["args"].append(f"array:{hashlib.sha256(arg.tobytes()).hexdigest()}")
        else:
            key_data["args"].append(str(arg))

    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    return f"{prefix}:{key_hash}"


def _generate_lock_key(prefix: str, args: tuple, kwargs: dict) -> Optional[str]:
    """분산 락 키 생성"""
    # 첫 번째 인자가 파일 경로일 때만 락 사용
    if args and isinstance(args[0], (str, Path)):
        file_path = str(args[0])
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        return f"{prefix}:lock:{file_hash}"
    return None


def _execute_with_features(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    config: Any,
    cache_key: Optional[str],
    rate_key: Union[str, Callable],
    event_prefix: str,
) -> Any:
    """동기 함수 실행 (분산 기능 포함)"""
    # 이벤트 발행 (시작)
    if config.enable_event_streaming:
        _publish_event_async(f"{event_prefix}.started", {})

    # Rate Limiting
    if config.enable_rate_limiting:
        # rate_key가 callable이면 동적으로 생성
        actual_rate_key = rate_key
        if callable(rate_key):
            try:
                actual_rate_key = rate_key(self, args, kwargs)
                if actual_rate_key is None:
                    # None이면 Rate Limiting 건너뛰기
                    actual_rate_key = None
            except Exception:
                actual_rate_key = "default"

        if actual_rate_key:
            rate_limiter = get_rate_limiter()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 Rate Limiting 건너뛰기 (fallback)
                    pass
                else:
                    loop.run_until_complete(
                        rate_limiter.acquire(
                            key=actual_rate_key,
                            tokens=1,
                            rate=config.rate_limit_per_second,
                        )
                    )
            except RuntimeError:
                asyncio.run(
                    rate_limiter.acquire(
                        key=actual_rate_key,
                        tokens=1,
                        rate=config.rate_limit_per_second,
                    )
                )
            except Exception as e:
                logger.debug(f"Rate limiting failed: {e}")

    # 함수 실행
    try:
        result = func(self, *args, **kwargs)

        # 이벤트 발행 (완료)
        if config.enable_event_streaming:
            _publish_event_async(
                f"{event_prefix}.completed",
                {
                    "success": True,
                },
            )

        return result
    except Exception as e:
        # 이벤트 발행 (실패)
        if config.enable_event_streaming:
            _publish_event_async(
                f"{event_prefix}.failed",
                {
                    "error": str(e),
                },
            )
        raise


async def _execute_with_features_async(
    func: Callable,
    self: Any,
    args: tuple,
    kwargs: dict,
    config: Any,
    cache_key: Optional[str],
    rate_key: Union[str, Callable],
    event_prefix: str,
    rate_limit_condition: Optional[Callable] = None,
    pipeline_type: str = "default",
) -> Any:
    """비동기 함수 실행 (분산 기능 포함)"""
    # 이벤트 발행 (시작)
    if config.enable_event_streaming:
        await _publish_event(f"{event_prefix}.started", {})

    # Rate Limiting (조건부 적용)
    if config.enable_rate_limiting:
        # rate_limit_condition이 있으면 확인
        should_rate_limit = True
        if rate_limit_condition:
            try:
                should_rate_limit = rate_limit_condition(self, args, kwargs)
            except Exception:
                should_rate_limit = True

        if should_rate_limit:
            # rate_limit_key가 함수면 동적으로 생성
            actual_rate_key = rate_key
            if callable(rate_key):
                try:
                    actual_rate_key = rate_key(self, args, kwargs)
                    if actual_rate_key is None:
                        # None이면 Rate Limiting 건너뛰기
                        should_rate_limit = False
                except Exception:
                    actual_rate_key = pipeline_type

            if should_rate_limit and actual_rate_key:
                rate_limiter = get_rate_limiter()
                try:
                    await rate_limiter.acquire(
                        key=actual_rate_key,
                        tokens=1,
                        rate=config.rate_limit_per_second,
                    )
                except Exception as e:
                    logger.debug(f"Rate limiting failed: {e}")

    # 함수 실행
    try:
        result = await func(self, *args, **kwargs)

        # 이벤트 발행 (완료)
        if config.enable_event_streaming:
            await _publish_event(
                f"{event_prefix}.completed",
                {
                    "success": True,
                },
            )

        return result
    except Exception as e:
        # 이벤트 발행 (실패)
        if config.enable_event_streaming:
            await _publish_event(
                f"{event_prefix}.failed",
                {
                    "error": str(e),
                },
            )
        raise


def _publish_event_async(event_type: str, event_data: dict):
    """비동기 이벤트 발행 (동기 컨텍스트에서)"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 실행 중인 루프가 있으면 비동기로 실행 (fire-and-forget)
            asyncio.create_task(_publish_event(event_type, event_data))
        else:
            loop.run_until_complete(_publish_event(event_type, event_data))
    except RuntimeError:
        asyncio.run(_publish_event(event_type, event_data))
    except Exception as e:
        logger.debug(f"Event publishing failed: {e}")


async def _publish_event(event_type: str, event_data: dict):
    """이벤트 발행"""
    try:
        event_logger = get_event_logger()
        await event_logger.log_event(event_type, event_data)
    except Exception as e:
        logger.debug(f"Event publishing failed: {e}")


def with_batch_processing(
    pipeline_type: str = "default",
    max_concurrent: Optional[int] = None,
    use_distributed_queue: Optional[bool] = None,
):
    """
    배치 처리 데코레이터

    리스트를 받아서 배치로 처리하는 함수에 적용

    Args:
        pipeline_type: 파이프라인 타입
        max_concurrent: 최대 동시 처리 수
        use_distributed_queue: 분산 큐 사용 여부

    Example:
        ```python
        @with_batch_processing(pipeline_type="ocr", max_concurrent=4)
        def batch_recognize(self, images: List[str]) -> List[OCRResult]:
            # 배치 처리 로직
            ...
        ```
    """

    def decorator(func: Callable) -> Callable:
        config = get_distributed_config()
        pipeline_config = getattr(config, pipeline_type, None)

        if pipeline_config is None:
            use_queue = use_distributed_queue or False
            max_workers = max_concurrent or 4
        else:
            use_queue = (
                use_distributed_queue
                if use_distributed_queue is not None
                else pipeline_config.use_distributed_queue
            )
            max_workers = (
                max_concurrent
                if max_concurrent is not None
                else getattr(pipeline_config, "max_concurrent", 4)
            )

        @functools.wraps(func)
        def sync_wrapper(self, items: list, *args, **kwargs):
            """동기 배치 처리"""
            if use_queue and len(items) > 1:
                # BatchProcessor 사용
                processor = BatchProcessor(
                    task_type=f"{pipeline_type}.batch", max_concurrent=max_workers
                )

                async def _batch_async():
                    results = await processor.process_batch(
                        items=items,
                        handler=lambda item: func(self, [item], *args, **kwargs)[0],
                    )
                    return [r for r in results if r is not None]

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 이미 실행 중인 루프가 있으면 순차 처리 (fallback)
                        return [func(self, [item], *args, **kwargs)[0] for item in items]
                    else:
                        return loop.run_until_complete(_batch_async())
                except RuntimeError:
                    return asyncio.run(_batch_async())
            else:
                # 순차 처리
                results = []
                for item in items:
                    result = func(self, [item], *args, **kwargs)
                    results.extend(result if isinstance(result, list) else [result])
                return results

        @functools.wraps(func)
        async def async_wrapper(self, items: List, *args, **kwargs):
            """비동기 배치 처리"""
            if use_queue and len(items) > 1:
                processor = BatchProcessor(
                    task_type=f"{pipeline_type}.batch", max_concurrent=max_workers
                )

                # handler는 단일 아이템을 처리하는 함수
                async def process_item(item):
                    result = await func(self, [item], *args, **kwargs)
                    # 리스트면 첫 번째 요소, 아니면 그대로 반환
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]
                    return result

                results = await processor.process_batch(
                    items=items,
                    handler=process_item,
                )
                return [r for r in results if r is not None]
            else:
                # 순차 처리
                results = []
                for item in items:
                    result = await func(self, [item], *args, **kwargs)
                    results.extend(result if isinstance(result, list) else [result])
                return results

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
