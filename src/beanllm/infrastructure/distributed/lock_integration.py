"""
분산 락 통합 (Distributed Lock Integration)

기존 코드에 분산 락 기능 추가
기존 최적화 패턴 참고: Helper 메서드, 에러 처리
"""

from contextlib import asynccontextmanager
from typing import Callable

from .factory import get_distributed_lock
from .utils import sanitize_error_message

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


def with_distributed_lock(lock_key: str, timeout: float = 30.0):
    """
    분산 락 데코레이터

    함수 실행 시 분산 락 획득

    Args:
        lock_key: 락 키 (예: "vector_store:update:123")
        timeout: 락 타임아웃 (초)

    Example:
        ```python
        @with_distributed_lock("vector_store:update:123")
        async def update_vector_store(store_id: str):
            # 벡터 스토어 업데이트
            pass
        ```
    """

    def decorator(func: Callable) -> Callable:
        import asyncio
        import functools

        lock = get_distributed_lock()

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                async with lock.acquire(lock_key, timeout=timeout):
                    return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Failed to acquire lock {lock_key}: {sanitize_error_message(str(e))}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 동기 함수는 비동기로 래핑
            async def _async_wrapper():
                return await async_wrapper(*args, **kwargs)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 락 없이 실행 (fallback)
                    logger.warning(f"Lock {lock_key} skipped (event loop running)")
                    return func(*args, **kwargs)
                else:
                    return loop.run_until_complete(_async_wrapper())
            except RuntimeError:
                return asyncio.run(_async_wrapper())

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class LockManager:
    """
    락 관리자

    리소스별 락 관리
    """

    def __init__(self):
        self.lock = get_distributed_lock()

    @asynccontextmanager
    async def acquire_resource_lock(
        self,
        resource_type: str,
        resource_id: str,
        timeout: float = 30.0,
    ):
        """
        리소스 락 획득 (context manager)

        Args:
            resource_type: 리소스 타입 (예: "vector_store", "model", "file")
            resource_id: 리소스 ID
            timeout: 락 타임아웃 (초)

        Example:
            ```python
            lock_manager = LockManager()
            async with lock_manager.acquire_resource_lock("vector_store", "123"):
                # 벡터 스토어 업데이트
                await vector_store.add_documents(docs)
            ```
        """
        lock_key = f"{resource_type}:{resource_id}"
        try:
            async with self.lock.acquire(lock_key, timeout=timeout):
                yield
        except Exception as e:
            logger.error(
                f"Failed to acquire resource lock {lock_key}: {sanitize_error_message(str(e))}"
            )
            # 락 획득 실패 시에도 진행 (fallback)
            yield

    async def with_vector_store_lock(self, store_id: str, timeout: float = 30.0):
        """벡터 스토어 락 획득"""
        return self.acquire_resource_lock("vector_store", store_id, timeout=timeout)

    async def with_model_lock(self, model_name: str, timeout: float = 30.0):
        """모델 로딩 락 획득"""
        return self.acquire_resource_lock("model", model_name, timeout=timeout)

    async def with_file_lock(self, file_path: str, timeout: float = 30.0):
        """파일 처리 락 획득"""
        import hashlib

        # 파일 경로를 해시하여 키 생성
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        return self.acquire_resource_lock("file", file_hash, timeout=timeout)


# 전역 락 관리자
_global_lock_manager = LockManager()


def get_lock_manager() -> LockManager:
    """전역 락 관리자 반환"""
    return _global_lock_manager
