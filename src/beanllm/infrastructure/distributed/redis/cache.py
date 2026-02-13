"""
Redis 기반 분산 Cache

기존 최적화 패턴 참고: 에러 처리, 로깅, fallback
"""

import asyncio
import json
from typing import Generic, Optional, TypeVar, cast

from beanllm.infrastructure.distributed.interfaces import CacheInterface
from beanllm.infrastructure.distributed.utils import check_redis_health
from beanllm.utils import sanitize_error_message
from beanllm.utils.constants import REDIS_SCAN_TIMEOUT, REDIS_TIMEOUT

from .client import get_redis_client

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class RedisCache(CacheInterface[K, V], Generic[K, V]):
    """
    Redis 기반 분산 Cache

    여러 서버 간 캐시 공유
    TTL 지원
    """

    def __init__(self, redis_client=None, key_prefix: str = "cache", ttl: Optional[int] = None):
        """
        Args:
            redis_client: Redis 클라이언트 (None이면 자동 생성)
            key_prefix: 캐시 키 접두사
            ttl: 기본 TTL (초), None이면 만료 없음
        """
        self.redis = redis_client or get_redis_client()
        self.key_prefix = key_prefix
        self.default_ttl = ttl

    def _make_key(self, key: K) -> str:
        """캐시 키 생성"""
        if isinstance(key, str):
            return f"{self.key_prefix}:{key}"
        # 다른 타입은 JSON 직렬화
        key_str = json.dumps(key, sort_keys=True)
        return f"{self.key_prefix}:{key_str}"

    async def get(self, key: K) -> Optional[V]:
        """값 조회"""
        try:
            # Redis 연결 확인
            if not await check_redis_health(self.redis):
                logger.warning(f"Redis not connected, cache get skipped for key: {key}")
                return None  # 연결 실패 시 캐시 미스로 처리

            cache_key = self._make_key(key)
            value = await asyncio.wait_for(self.redis.get(cache_key), timeout=REDIS_TIMEOUT)

            if value is None:
                return None

            try:
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                return cast(V, json.loads(value))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Cache value decode error for key: {key}: {e}")
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Redis cache get timeout for key: {key}")
            return None  # 타임아웃 시 캐시 미스로 처리
        except Exception as e:
            logger.error(f"Redis cache get error for key: {key}: {sanitize_error_message(str(e))}")
            return None  # 오류 시 캐시 미스로 처리

    async def set(self, key: K, value: V, ttl: Optional[int] = None):
        """값 저장"""
        try:
            # Redis 연결 확인
            if not await check_redis_health(self.redis):
                logger.warning(f"Redis not connected, cache set skipped for key: {key}")
                return  # 연결 실패 시 저장 건너뛰기

            cache_key = self._make_key(key)
            value_json = json.dumps(value)
            ttl = ttl or self.default_ttl

            if ttl:
                await asyncio.wait_for(
                    self.redis.setex(cache_key, ttl, value_json.encode("utf-8")),
                    timeout=REDIS_TIMEOUT,
                )
            else:
                await asyncio.wait_for(
                    self.redis.set(cache_key, value_json.encode("utf-8")), timeout=REDIS_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.warning(f"Redis cache set timeout for key: {key}")
        except Exception as e:
            logger.error(f"Redis cache set error for key: {key}: {sanitize_error_message(str(e))}")

    async def delete(self, key: K):
        """값 삭제"""
        try:
            if not await check_redis_health(self.redis):
                logger.warning(f"Redis not connected, cache delete skipped for key: {key}")
                return

            cache_key = self._make_key(key)
            await asyncio.wait_for(self.redis.delete(cache_key), timeout=REDIS_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning(f"Redis cache delete timeout for key: {key}")
        except Exception as e:
            logger.error(
                f"Redis cache delete error for key: {key}: {sanitize_error_message(str(e))}"
            )

    async def clear(self):
        """모든 캐시 삭제 (접두사로 시작하는 키만)"""
        try:
            if not await check_redis_health(self.redis):
                logger.warning("Redis not connected, cache clear skipped")
                return

            # Redis SCAN으로 접두사로 시작하는 키 찾기
            pattern = f"{self.key_prefix}:*"
            cursor = 0
            while True:
                cursor, keys = await asyncio.wait_for(
                    self.redis.scan(cursor, match=pattern, count=100), timeout=REDIS_SCAN_TIMEOUT
                )
                if keys:
                    await asyncio.wait_for(self.redis.delete(*keys), timeout=REDIS_SCAN_TIMEOUT)
                if cursor == 0:
                    break
        except asyncio.TimeoutError:
            logger.warning("Redis cache clear timeout")
        except Exception as e:
            logger.error(f"Redis cache clear error: {sanitize_error_message(str(e))}")
