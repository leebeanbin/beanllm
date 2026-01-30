"""
Session Cache Service

Redis 기반 세션 캐싱으로 MongoDB 조회 최적화.
캐시 무효화·만료 정책: docs/CACHE_AND_METRICS_POLICY.md §1 — 쓰기 시 즉시 무효화/갱신, TTL 만료 시 Redis 자동 삭제.
"""
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import timedelta

logger = logging.getLogger(__name__)

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
    
    @staticmethod
    async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
        """
        캐시에서 세션 가져오기
        
        Args:
            session_id: 세션 ID
            
        Returns:
            세션 데이터 또는 None
        """
        if not redis_client:
            return None
        
        try:
            cache_key = f"session:{session_id}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                logger.debug(f"✅ Cache hit: session:{session_id}")
                # Redis는 bytes를 반환할 수 있음
                if isinstance(cached, bytes):
                    cached = cached.decode('utf-8')
                return json.loads(cached)
            
            logger.debug(f"❌ Cache miss: session:{session_id}")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get session from cache: {e}")
            return None
    
    @staticmethod
    async def set_session(session_id: str, session_data: Dict[str, Any], ttl: int = None):
        """
        세션을 캐시에 저장
        
        Args:
            session_id: 세션 ID
            session_data: 세션 데이터
            ttl: TTL (초), None이면 기본값 사용
        """
        if not redis_client:
            return
        
        try:
            cache_key = f"session:{session_id}"
            ttl = ttl or SessionCacheService.SESSION_TTL
            
            # messages 배열이 크면 제외 (메모리 절약)
            cache_data = session_data.copy()
            if "messages" in cache_data and len(cache_data["messages"]) > 100:
                cache_data["messages"] = cache_data["messages"][-50:]  # 최근 50개만
                cache_data["_cached_messages_truncated"] = True
            
            # Redis는 bytes를 요구하므로 encode
            cache_value = json.dumps(cache_data, default=str).encode('utf-8')
            await redis_client.setex(
                cache_key,
                ttl,
                cache_value
            )
            
            logger.debug(f"✅ Cached session: {session_id} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.warning(f"Failed to cache session: {e}")
    
    @staticmethod
    async def get_session_list(
        feature_mode: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        캐시에서 세션 목록 가져오기
        
        Args:
            feature_mode: 필터 모드
            skip: 건너뛸 개수
            limit: 가져올 개수
            
        Returns:
            {"sessions": [...], "total": ...} 또는 None
        """
        if not redis_client:
            return None
        
        try:
            cache_key = f"sessions:list:{feature_mode or 'all'}:{skip}:{limit}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                logger.debug(f"✅ Cache hit: {cache_key}")
                # Redis는 bytes를 반환할 수 있음
                if isinstance(cached, bytes):
                    cached = cached.decode('utf-8')
                return json.loads(cached)
            
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
        ttl: int = None
    ):
        """
        세션 목록을 캐시에 저장
        
        Args:
            sessions: 세션 목록
            total: 전체 개수
            feature_mode: 필터 모드
            skip: 건너뛴 개수
            limit: 가져온 개수
            ttl: TTL (초)
        """
        if not redis_client:
            return
        
        try:
            cache_key = f"sessions:list:{feature_mode or 'all'}:{skip}:{limit}"
            ttl = ttl or SessionCacheService.SESSION_LIST_TTL
            
            cache_data = {
                "sessions": sessions,
                "total": total
            }
            
            # Redis는 bytes를 요구하므로 encode
            cache_value = json.dumps(cache_data, default=str).encode('utf-8')
            await redis_client.setex(
                cache_key,
                ttl,
                cache_value
            )
            
            logger.debug(f"✅ Cached session list: {cache_key} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.warning(f"Failed to cache session list: {e}")
    
    @staticmethod
    async def invalidate_session(session_id: str):
        """
        세션 캐시 무효화
        
        Args:
            session_id: 세션 ID
        """
        if not redis_client:
            return
        
        try:
            cache_key = f"session:{session_id}"
            await redis_client.delete(cache_key)
            logger.debug(f"✅ Invalidated cache: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate session cache: {e}")
    
    @staticmethod
    async def invalidate_session_lists():
        """
        모든 세션 목록 캐시 무효화
        
        Redis async 클라이언트는 keys() 대신 scan_iter() 사용 권장
        """
        if not redis_client:
            return
        
        try:
            # 패턴으로 모든 목록 캐시 삭제
            # async Redis 클라이언트는 scan_iter() 사용
            keys_to_delete = []
            async for key in redis_client.scan_iter(match="sessions:list:*"):
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                keys_to_delete.append(key)
            
            if keys_to_delete:
                await redis_client.delete(*keys_to_delete)
                logger.debug(f"✅ Invalidated {len(keys_to_delete)} session list caches")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate session list caches: {e}")
    
    @staticmethod
    async def invalidate_all(session_id: Optional[str] = None):
        """
        세션 관련 모든 캐시 무효화
        
        Args:
            session_id: 특정 세션 ID (None이면 모든 세션 목록만 무효화)
        """
        if session_id:
            await SessionCacheService.invalidate_session(session_id)
        await SessionCacheService.invalidate_session_lists()


# 전역 인스턴스
session_cache = SessionCacheService()
