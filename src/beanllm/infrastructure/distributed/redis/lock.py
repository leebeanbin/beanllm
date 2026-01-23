"""
Redis 기반 분산 락

Redis SET NX EX를 사용하여 분산 락 구현
기존 최적화 패턴 참고: 에러 처리, 로깅, fallback
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncContextManager

from beanllm.infrastructure.distributed.interfaces import DistributedLockInterface
from beanllm.infrastructure.distributed.utils import check_redis_health, LockAcquisitionError
from beanllm.utils import sanitize_error_message
from .client import get_redis_client

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


# Redis Lua Script for safe lock release
RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


class RedisLock(DistributedLockInterface):
    """
    Redis 기반 분산 락

    SET NX EX를 사용하여 원자적 락 획득
    여러 서버 간 동시성 제어
    """

    def __init__(self, redis_client=None):
        """
        Args:
            redis_client: Redis 클라이언트 (None이면 자동 생성)
        """
        self.redis = redis_client or get_redis_client()
        self._script_sha = None

    async def _load_script(self):
        """Lua Script 로드"""
        if self._script_sha is None:
            self._script_sha = await self.redis.script_load(RELEASE_LOCK_SCRIPT)
        return self._script_sha

    @asynccontextmanager
    async def acquire(self, key: str, timeout: float = 30.0) -> AsyncContextManager:
        """락 획득 (context manager)"""
        lock_key = f"lock:{key}"
        worker_id = str(uuid.uuid4())
        lock_timeout = int(timeout)

        try:
            # Redis 연결 확인
            if not await check_redis_health(self.redis):
                logger.warning(f"Redis not connected, lock acquisition skipped for key: {key}")
                # 연결 실패 시 락 없이 진행 (fallback)
                yield
                return

            # SET NX EX: 키가 없으면 설정하고 TTL 설정
            acquired = await asyncio.wait_for(
                self.redis.set(
                    lock_key,
                    worker_id.encode() if isinstance(worker_id, str) else worker_id,
                    nx=True,  # 키가 없을 때만 설정
                    ex=lock_timeout,  # TTL
                ),
                timeout=2.0,
            )

            if not acquired:
                raise LockAcquisitionError(f"Failed to acquire lock for key: {key} (lock already held)")

            try:
                yield
            finally:
                # 안전하게 락 해제 (자신이 획득한 락만 해제)
                try:
                    script_sha = await self._load_script()
                    await asyncio.wait_for(
                        self.redis.evalsha(
                            script_sha,
                            1,
                            lock_key,
                            worker_id.encode() if isinstance(worker_id, str) else worker_id,
                        ),
                        timeout=2.0,
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to release lock for key: {key}: {sanitize_error_message(str(e))}"
                    )
        except asyncio.TimeoutError:
            logger.error(f"Redis lock acquisition timeout for key: {key}")
            # 타임아웃 시 락 없이 진행 (fallback)
            yield
        except LockAcquisitionError:
            raise
        except Exception as e:
            logger.error(
                f"Redis lock error for key: {key}: {sanitize_error_message(str(e))}"
            )
            # 오류 시 락 없이 진행 (fallback)
            yield

