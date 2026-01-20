"""
인메모리 분산 락 (asyncio.Lock 래핑)
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncContextManager, Dict

from beanllm.infrastructure.distributed.interfaces import DistributedLockInterface


class InMemoryLock(DistributedLockInterface):
    """
    인메모리 분산 락

    asyncio.Lock을 사용하여 인메모리 락 구현
    키별로 별도의 Lock 인스턴스 관리
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()  # _locks 딕셔너리 보호용

    @asynccontextmanager
    async def acquire(self, key: str, timeout: float = 30.0) -> AsyncContextManager:
        """락 획득 (context manager)"""
        # 키에 해당하는 Lock 가져오기 또는 생성
        async with self._lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]
        
        # 락 획득 (타임아웃 지원)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Failed to acquire lock for key: {key}")
        
        try:
            yield
        finally:
            lock.release()

