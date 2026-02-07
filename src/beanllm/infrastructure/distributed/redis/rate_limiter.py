"""
Redis 기반 분산 Rate Limiter

Redis Lua Script를 사용하여 원자적 Rate Limiting 구현
기존 최적화 패턴 참고: 에러 처리, 로깅, fallback
"""

import asyncio
import time
from typing import Any, Dict

from beanllm.infrastructure.distributed.interfaces import RateLimiterInterface
from beanllm.infrastructure.distributed.utils import check_redis_health
from beanllm.utils import sanitize_error_message

from .client import get_redis_client

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)


# Redis Lua Script for Token Bucket Rate Limiting
RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local capacity = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- 현재 토큰 수 조회
local current = redis.call('GET', key)
if current == false then
    current = capacity
else
    current = tonumber(current)
end

-- 마지막 업데이트 시간 조회
local last_update_key = key .. ':last_update'
local last_update = redis.call('GET', last_update_key)
if last_update ~= false then
    local elapsed = now - tonumber(last_update)
    -- 토큰 충전
    current = math.min(capacity, current + rate * elapsed)
end

-- 토큰 소비 가능 여부 확인
if current >= cost then
    redis.call('SET', key, current - cost)
    redis.call('SET', last_update_key, now)
    redis.call('EXPIRE', key, 60)
    redis.call('EXPIRE', last_update_key, 60)
    return {1, current - cost}  -- {성공, 남은 토큰}
else
    -- 필요한 토큰 계산
    local needed = cost - current
    local wait_time = needed / rate
    return {0, wait_time}  -- {실패, 대기 시간}
end
"""


class RedisRateLimiter(RateLimiterInterface):
    """
    Redis 기반 분산 Rate Limiter

    Token Bucket 알고리즘을 Redis Lua Script로 구현
    여러 서버 간 Rate Limit 공유
    """

    def __init__(
        self, redis_client=None, default_rate: float = 1.0, default_capacity: float = 20.0
    ):
        """
        Args:
            redis_client: Redis 클라이언트 (None이면 자동 생성)
            default_rate: 기본 속도 (토큰/초)
            default_capacity: 기본 용량 (최대 토큰 수)
        """
        self.redis = redis_client or get_redis_client()
        self.default_rate = default_rate
        self.default_capacity = default_capacity
        self._script_sha = None

    async def _load_script(self):
        """Lua Script 로드"""
        if self._script_sha is None:
            self._script_sha = await self.redis.script_load(RATE_LIMIT_SCRIPT)
        return self._script_sha

    async def acquire(self, key: str, cost: float = 1.0) -> bool:
        """토큰 획득 시도 (대기하지 않음)"""
        try:
            # Redis 연결 확인
            if not await check_redis_health(self.redis):
                logger.warning(f"Redis not connected, rate limit check skipped for key: {key}")
                # 연결 실패 시 허용 (fallback)
                return True

            script_sha = await self._load_script()
            rate_limit_key = f"rate_limit:{key}"

            result = await asyncio.wait_for(
                self.redis.evalsha(
                    script_sha,
                    1,  # KEYS 개수
                    rate_limit_key,
                    cost,
                    self.default_rate,
                    self.default_capacity,
                    time.time(),
                ),
                timeout=2.0,
            )

            success, _ = result
            return bool(success)
        except asyncio.TimeoutError:
            logger.warning(f"Redis rate limit check timeout for key: {key}")
            return True  # 타임아웃 시 허용 (fallback)
        except Exception as e:
            logger.error(
                f"Redis rate limit check error for key: {key}: {sanitize_error_message(str(e))}"
            )
            return True  # 오류 시 허용 (fallback)

    async def wait(self, key: str, cost: float = 1.0):
        """토큰이 충분할 때까지 대기"""
        try:
            # Redis 연결 확인
            if not await check_redis_health(self.redis):
                logger.warning(f"Redis not connected, rate limit wait skipped for key: {key}")
                # 연결 실패 시 즉시 반환 (fallback)
                return

            script_sha = await self._load_script()
            rate_limit_key = f"rate_limit:{key}"

            max_wait_time = 60.0  # 최대 대기 시간 (초)
            start_time = time.time()

            while True:
                # 최대 대기 시간 체크
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Rate limit wait timeout for key: {key}")
                    return  # 타임아웃 시 즉시 반환

                result = await asyncio.wait_for(
                    self.redis.evalsha(
                        script_sha,
                        1,
                        rate_limit_key,
                        cost,
                        self.default_rate,
                        self.default_capacity,
                        time.time(),
                    ),
                    timeout=2.0,
                )

                success, wait_time = result
                if success:
                    break

                # 대기 (최대 1초씩)
                await asyncio.sleep(min(wait_time, 1.0))
        except asyncio.TimeoutError:
            logger.warning(f"Redis rate limit wait timeout for key: {key}")
            return  # 타임아웃 시 즉시 반환 (fallback)
        except Exception as e:
            logger.error(
                f"Redis rate limit wait error for key: {key}: {sanitize_error_message(str(e))}"
            )
            return  # 오류 시 즉시 반환 (fallback)

    def get_status(self, key: str) -> Dict[str, Any]:
        """현재 상태 조회"""
        rate_limit_key = f"rate_limit:{key}"

        # 동기적으로 조회 (비동기 클라이언트이므로 await 필요)
        # 여기서는 간단히 키 존재 여부만 확인
        return {
            "key": key,
            "rate": self.default_rate,
            "capacity": self.default_capacity,
        }
