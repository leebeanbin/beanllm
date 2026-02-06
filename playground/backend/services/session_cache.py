"""
Session Cache Service

Redis 기반 세션 캐싱으로 MongoDB 조회 최적화.
캐시 무효화·만료 정책: docs/CACHE_AND_METRICS_POLICY.md §1 — 쓰기 시 즉시 무효화/갱신, TTL 만료 시 Redis 자동 삭제.

Usage:
    from services.session_cache import session_cache

    # 세션 캐싱
    await session_cache.set_session(session_id, session_data)
    session = await session_cache.get_session(session_id)

    # 요약 캐싱
    await session_cache.set_summary(session_id, summary)
    summary = await session_cache.get_summary(session_id)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _decode_redis_value(value: Union[bytes, str, None]) -> Optional[str]:
    """
    Redis 응답값을 문자열로 디코딩.

    Args:
        value: Redis에서 반환된 값 (bytes, str, 또는 None)

    Returns:
        디코딩된 문자열 또는 None
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _encode_for_redis(data: Union[str, Dict, Any]) -> bytes:
    """
    데이터를 Redis 저장용 bytes로 인코딩.

    Args:
        data: 저장할 데이터 (문자열 또는 dict)

    Returns:
        인코딩된 bytes
    """
    if isinstance(data, str):
        return data.encode("utf-8")
    return json.dumps(data, default=str).encode("utf-8")


# Redis 클라이언트 (선택적)
try:
    from beanllm.infrastructure.distributed.redis.client import get_redis_client

    redis_client = get_redis_client()
except ImportError:
    redis_client = None
    logger.warning("Redis not available, session caching disabled")


class SessionCacheService:
    """세션 캐싱 서비스"""

    # 캐시 TTL (초)
    SESSION_TTL = 60  # 1분 (개별 세션)
    SESSION_LIST_TTL = 300  # 5분 (세션 목록)
    SUMMARY_TTL = 3600  # 1시간 (요약)

    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """
        캐시에서 세션 가져오기.

        Args:
            session_id: 세션 ID

        Returns:
            세션 데이터 또는 None (캐시 미스 또는 에러 시)
        """
        if not redis_client:
            return None

        try:
            cache_key = f"session:{session_id}"
            cached = await redis_client.get(cache_key)
            decoded = _decode_redis_value(cached)

            if decoded:
                logger.debug(f"✅ Cache hit: session:{session_id}")
                return json.loads(decoded)

            logger.debug(f"❌ Cache miss: session:{session_id}")
            return None

        except Exception as e:
            logger.warning(f"Failed to get session from cache: {e}")
            return None

    @staticmethod
    async def set_session(
        session_id: str,
        session_data: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        세션을 캐시에 저장.

        Args:
            session_id: 세션 ID
            session_data: 세션 데이터
            ttl: TTL (초), None이면 기본값 SESSION_TTL 사용

        Returns:
            저장 성공 여부
        """
        if not redis_client:
            return False

        try:
            cache_key = f"session:{session_id}"
            ttl = ttl or SessionCacheService.SESSION_TTL

            # messages 배열이 크면 최근 50개만 저장 (메모리 절약)
            cache_data = session_data.copy()
            if "messages" in cache_data and len(cache_data["messages"]) > 100:
                cache_data["messages"] = cache_data["messages"][-50:]
                cache_data["_cached_messages_truncated"] = True

            cache_value = _encode_for_redis(cache_data)
            await redis_client.setex(cache_key, ttl, cache_value)

            logger.debug(f"✅ Cached session: {session_id} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache session: {e}")
            return False

    @staticmethod
    async def get_session_list(
        feature_mode: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Optional[Dict[str, Any]]:
        """
        캐시에서 세션 목록 가져오기.

        Args:
            feature_mode: 필터 모드 (None이면 "all")
            skip: 건너뛸 개수
            limit: 가져올 개수

        Returns:
            {"sessions": [...], "total": ...} 또는 None (캐시 미스 시)
        """
        if not redis_client:
            return None

        try:
            cache_key = f"sessions:list:{feature_mode or 'all'}:{skip}:{limit}"
            cached = await redis_client.get(cache_key)
            decoded = _decode_redis_value(cached)

            if decoded:
                logger.debug(f"✅ Cache hit: {cache_key}")
                return json.loads(decoded)

            logger.debug(f"❌ Cache miss: {cache_key}")
            return None

        except Exception as e:
            logger.warning(f"Failed to get session list from cache: {e}")
            return None

    @staticmethod
    async def set_session_list(
        sessions: List[Dict[str, Any]],
        total: int,
        feature_mode: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        세션 목록을 캐시에 저장.

        Args:
            sessions: 세션 목록
            total: 전체 개수
            feature_mode: 필터 모드 (None이면 "all")
            skip: 건너뛴 개수
            limit: 가져온 개수
            ttl: TTL (초), None이면 기본값 SESSION_LIST_TTL 사용

        Returns:
            저장 성공 여부
        """
        if not redis_client:
            return False

        try:
            cache_key = f"sessions:list:{feature_mode or 'all'}:{skip}:{limit}"
            ttl = ttl or SessionCacheService.SESSION_LIST_TTL

            cache_data = {"sessions": sessions, "total": total}
            cache_value = _encode_for_redis(cache_data)
            await redis_client.setex(cache_key, ttl, cache_value)

            logger.debug(f"✅ Cached session list: {cache_key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache session list: {e}")
            return False

    @staticmethod
    async def invalidate_session(session_id: str) -> bool:
        """
        세션 캐시 무효화.

        Args:
            session_id: 세션 ID

        Returns:
            무효화 성공 여부
        """
        if not redis_client:
            return False

        try:
            cache_key = f"session:{session_id}"
            await redis_client.delete(cache_key)
            logger.debug(f"✅ Invalidated cache: {cache_key}")
            return True

        except Exception as e:
            logger.warning(f"Failed to invalidate session cache: {e}")
            return False

    @staticmethod
    async def invalidate_session_lists() -> int:
        """
        모든 세션 목록 캐시 무효화.

        Redis async 클라이언트는 keys() 대신 scan_iter() 사용 권장.

        Returns:
            삭제된 캐시 키 개수
        """
        if not redis_client:
            return 0

        try:
            keys_to_delete = []
            async for key in redis_client.scan_iter(match="sessions:list:*"):
                decoded_key = _decode_redis_value(key)
                if decoded_key:
                    keys_to_delete.append(decoded_key)

            if keys_to_delete:
                await redis_client.delete(*keys_to_delete)
                logger.debug(f"✅ Invalidated {len(keys_to_delete)} session list caches")

            return len(keys_to_delete)

        except Exception as e:
            logger.warning(f"Failed to invalidate session list caches: {e}")
            return 0

    @staticmethod
    async def invalidate_all(session_id: Optional[str] = None):
        """
        세션 관련 모든 캐시 무효화

        Args:
            session_id: 특정 세션 ID (None이면 모든 세션 목록만 무효화)
        """
        if session_id:
            await SessionCacheService.invalidate_session(session_id)
            await SessionCacheService.invalidate_summary(session_id)
        await SessionCacheService.invalidate_session_lists()

    # ===========================================
    # 요약 캐싱 (Summary Caching)
    # ===========================================

    @staticmethod
    async def get_summary(session_id: str) -> Optional[str]:
        """
        캐시에서 세션 요약 가져오기.

        Args:
            session_id: 세션 ID

        Returns:
            요약 문자열 또는 None (캐시 미스 또는 에러 시)
        """
        if not redis_client:
            return None

        try:
            cache_key = f"summary:{session_id}"
            cached = await redis_client.get(cache_key)
            decoded = _decode_redis_value(cached)

            if decoded:
                logger.debug(f"✅ Cache hit: summary:{session_id}")
                return decoded

            logger.debug(f"❌ Cache miss: summary:{session_id}")
            return None

        except Exception as e:
            logger.warning(f"Failed to get summary from cache: {e}")
            return None

    @staticmethod
    async def set_summary(
        session_id: str,
        summary: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        세션 요약을 캐시에 저장.

        Args:
            session_id: 세션 ID
            summary: 요약 문자열
            ttl: TTL (초), None이면 기본값 SUMMARY_TTL 사용

        Returns:
            저장 성공 여부
        """
        if not redis_client:
            return False

        try:
            cache_key = f"summary:{session_id}"
            ttl = ttl or SessionCacheService.SUMMARY_TTL

            cache_value = _encode_for_redis(summary)
            await redis_client.setex(cache_key, ttl, cache_value)

            logger.debug(f"✅ Cached summary: {session_id} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache summary: {e}")
            return False

    @staticmethod
    async def invalidate_summary(session_id: str) -> bool:
        """
        세션 요약 캐시 무효화.

        Args:
            session_id: 세션 ID

        Returns:
            무효화 성공 여부
        """
        if not redis_client:
            return False

        try:
            cache_key = f"summary:{session_id}"
            await redis_client.delete(cache_key)
            logger.debug(f"✅ Invalidated cache: {cache_key}")
            return True

        except Exception as e:
            logger.warning(f"Failed to invalidate summary cache: {e}")
            return False

    @staticmethod
    async def get_summary_metadata(session_id: str) -> Optional[Dict[str, Any]]:
        """
        캐시에서 요약 메타데이터 가져오기.

        Args:
            session_id: 세션 ID

        Returns:
            메타데이터 dict 또는 None (캐시 미스 또는 에러 시)
        """
        if not redis_client:
            return None

        try:
            cache_key = f"summary_meta:{session_id}"
            cached = await redis_client.get(cache_key)
            decoded = _decode_redis_value(cached)

            if decoded:
                return json.loads(decoded)

            return None

        except Exception as e:
            logger.warning(f"Failed to get summary metadata from cache: {e}")
            return None

    @staticmethod
    async def set_summary_with_metadata(
        session_id: str,
        summary: str,
        message_count: int,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        요약과 메타데이터를 함께 캐싱.

        Args:
            session_id: 세션 ID
            summary: 요약 문자열
            message_count: 요약된 메시지 수
            ttl: TTL (초), None이면 기본값 SUMMARY_TTL 사용

        Returns:
            저장 성공 여부
        """
        if not redis_client:
            return False

        try:
            ttl = ttl or SessionCacheService.SUMMARY_TTL

            # 요약 저장
            summary_success = await SessionCacheService.set_summary(session_id, summary, ttl)

            # 메타데이터 저장
            meta_key = f"summary_meta:{session_id}"
            metadata = {
                "message_count": message_count,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_value = _encode_for_redis(metadata)
            await redis_client.setex(meta_key, ttl, meta_value)

            logger.debug(f"✅ Cached summary with metadata: {session_id}")
            return summary_success

        except Exception as e:
            logger.warning(f"Failed to cache summary with metadata: {e}")
            return False


# 전역 인스턴스
session_cache = SessionCacheService()
