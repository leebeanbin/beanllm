"""
Redis 클라이언트 관리

환경변수에서 Redis 설정을 읽어 클라이언트를 생성합니다.
기존 최적화 패턴 참고: 에러 처리, 로깅, 연결 풀링
"""

import os
from typing import Any, Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)


logger = get_logger(__name__)

_redis_client: Optional[Any] = None


def get_redis_client():
    """
    Redis 클라이언트 반환 (싱글톤)

    환경변수:
        REDIS_HOST: Redis 호스트 (기본: localhost)
        REDIS_PORT: Redis 포트 (기본: 6379)
        REDIS_DB: Redis DB 번호 (기본: 0)
        REDIS_PASSWORD: Redis 비밀번호 (선택)
        REDIS_SOCKET_TIMEOUT: 소켓 타임아웃 (기본: 5.0)
        REDIS_SOCKET_CONNECT_TIMEOUT: 연결 타임아웃 (기본: 5.0)

    Returns:
        Redis 클라이언트 인스턴스

    Raises:
        ImportError: redis 패키지가 설치되지 않음
    """
    global _redis_client

    if redis is None:
        raise ImportError(
            "redis package is required for distributed mode. Install it with: pip install redis"
        )

    if _redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD")
        socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
        socket_connect_timeout = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))

        try:
            _redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # 바이너리 모드 (JSON 직렬화를 위해)
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                retry_on_timeout=True,  # 타임아웃 시 재시도
                health_check_interval=30,  # 30초마다 헬스 체크
            )
            logger.info(f"Redis client initialized: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            # 연결 실패해도 클라이언트는 생성 (fallback에서 처리)
            _redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
            )

    return _redis_client


def close_redis_client():
    """Redis 클라이언트 종료"""
    global _redis_client
    if _redis_client:
        try:
            _redis_client.close()
            logger.info("Redis client closed")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
        finally:
            _redis_client = None
