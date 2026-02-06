"""
Session Manager for MCP Server

세션별 인스턴스 관리 및 MongoDB 연동.
§0 메시징 적극 활용: Redis session:{id} 1차 조회, miss 시 MongoDB (BACKEND_MCP_DISTRIBUTED_REVIEW §6.3).
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Redis 클라이언트 (백엔드와 동일 session:{id} 키 사용, 선택적)
_redis_client = None


def _get_redis_client():
    """백엔드와 동일 Redis 사용. beanllm 경유 또는 REDIS_HOST/PORT 직접."""
    global _redis_client
    if _redis_client is False:  # 이전 시도에서 실패함
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        from beanllm.infrastructure.distributed.redis.client import get_redis_client

        _redis_client = get_redis_client()
        if _redis_client:
            logger.info("MCP session cache: using beanllm Redis client")
        return _redis_client
    except ImportError:
        pass
    try:
        import redis.asyncio as redis

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        pw = os.getenv("REDIS_PASSWORD")
        _redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=pw or None,
            decode_responses=False,
        )
        logger.info(f"MCP session cache: Redis {host}:{port}")
        return _redis_client
    except Exception as e:
        logger.debug(f"MCP session cache: Redis not available ({e})")
        _redis_client = False  # 한 번 실패 후 재시도 방지
        return None


# MongoDB 클라이언트 싱글톤 (매 호출 새 클라이언트 생성 지양, BACKEND_MCP_DISTRIBUTED_REVIEW 권장)
_mongo_client = None
_mongo_db = None


def _get_mongo_db():
    global _mongo_client, _mongo_db
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None
    if _mongo_db is not None:
        return _mongo_db
    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        _mongo_client = AsyncIOMotorClient(uri)
        _mongo_db = _mongo_client[os.getenv("MONGODB_DATABASE", "beanllm")]
        logger.info("MCP MongoDB client initialized (singleton)")
        return _mongo_db
    except Exception as e:
        logger.warning(f"MCP MongoDB not available: {e}")
        return None


class SessionManager:
    """
    세션별 인스턴스 관리

    MCP Server에서 세션별로 RAG, Multi-Agent, Knowledge Graph 인스턴스를 관리합니다.
    """

    def __init__(self):
        # 세션별 인스턴스 저장
        # 구조: {session_id: {collection_name: instance}}
        self._rag_instances: Dict[str, Dict[str, Any]] = {}
        self._multiagent_systems: Dict[str, Dict[str, Any]] = {}
        self._kg_instances: Dict[str, Dict[str, Any]] = {}

        # 세션 메타데이터
        self._session_metadata: Dict[str, Dict[str, Any]] = {}

    def get_rag_instance(self, session_id: str, collection_name: str = "default") -> Optional[Any]:
        """
        세션별 RAG 인스턴스 가져오기

        Args:
            session_id: 세션 ID
            collection_name: 컬렉션 이름

        Returns:
            RAGChain 인스턴스 또는 None
        """
        return self._rag_instances.get(session_id, {}).get(collection_name)

    def set_rag_instance(self, session_id: str, collection_name: str, rag_instance: Any):
        """
        세션별 RAG 인스턴스 저장

        Args:
            session_id: 세션 ID
            collection_name: 컬렉션 이름
            rag_instance: RAGChain 인스턴스
        """
        if session_id not in self._rag_instances:
            self._rag_instances[session_id] = {}
        self._rag_instances[session_id][collection_name] = rag_instance

        # 세션 메타데이터 업데이트
        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = {
                "created_at": datetime.utcnow(),
                "last_accessed": datetime.utcnow(),
            }
        self._session_metadata[session_id]["last_accessed"] = datetime.utcnow()

        logger.debug(
            f"✅ Saved RAG instance for session {session_id}, collection {collection_name}"
        )

    def get_multiagent_system(self, session_id: str, system_name: str = "default") -> Optional[Any]:
        """세션별 Multi-Agent 시스템 가져오기"""
        return self._multiagent_systems.get(session_id, {}).get(system_name)

    def set_multiagent_system(self, session_id: str, system_name: str, system: Any):
        """세션별 Multi-Agent 시스템 저장"""
        if session_id not in self._multiagent_systems:
            self._multiagent_systems[session_id] = {}
        self._multiagent_systems[session_id][system_name] = system

        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = {
                "created_at": datetime.utcnow(),
                "last_accessed": datetime.utcnow(),
            }
        self._session_metadata[session_id]["last_accessed"] = datetime.utcnow()

    def get_kg_instance(self, session_id: str, graph_name: str = "default") -> Optional[Any]:
        """세션별 Knowledge Graph 인스턴스 가져오기"""
        return self._kg_instances.get(session_id, {}).get(graph_name)

    def set_kg_instance(self, session_id: str, graph_name: str, kg_instance: Any):
        """세션별 Knowledge Graph 인스턴스 저장"""
        if session_id not in self._kg_instances:
            self._kg_instances[session_id] = {}
        self._kg_instances[session_id][graph_name] = kg_instance

        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = {
                "created_at": datetime.utcnow(),
                "last_accessed": datetime.utcnow(),
            }
        self._session_metadata[session_id]["last_accessed"] = datetime.utcnow()

    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        세션 메시지 가져오기. §0 메시징 적극 활용: Redis session:{id} 1차 조회, miss 시 MongoDB.

        Args:
            session_id: 세션 ID

        Returns:
            메시지 리스트
        """
        try:
            # 1) Redis session:{id} 1차 조회 (백엔드 session_cache와 동일 키)
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    cache_key = f"session:{session_id}"
                    cached = await redis_client.get(cache_key)
                    if cached:
                        raw = cached.decode("utf-8") if isinstance(cached, bytes) else cached
                        session_data = json.loads(raw)
                        messages = session_data.get("messages", [])
                        for msg in messages:
                            ts = msg.get("timestamp")
                            if ts is not None and hasattr(ts, "isoformat"):
                                try:
                                    msg["timestamp"] = ts.isoformat()
                                except Exception:
                                    pass
                        logger.debug(f"MCP session cache hit: {session_id}")
                        return messages
                except Exception as e:
                    logger.debug(f"MCP session cache get failed: {e}")

            # 2) miss 시 MongoDB (싱글톤 클라이언트 사용)
            db = _get_mongo_db()
            if db is None:
                return []

            session = await db.chat_sessions.find_one({"session_id": session_id})
            if session is None:
                return []

            messages = session.get("messages", [])
            for msg in messages:
                if "timestamp" in msg and isinstance(msg["timestamp"], datetime):
                    msg["timestamp"] = msg["timestamp"].isoformat()

            return messages

        except Exception as e:
            logger.error(f"Failed to get session messages: {e}")
            return []

    async def cleanup_expired_sessions(self, ttl_hours: int = 24):
        """
        만료된 세션 정리

        Args:
            ttl_hours: 세션 TTL (시간)
        """
        now = datetime.utcnow()
        expired_sessions = []

        for session_id, metadata in self._session_metadata.items():
            last_accessed = metadata.get("last_accessed", metadata.get("created_at"))
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))

            if (now - last_accessed).total_seconds() > ttl_hours * 3600:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self._rag_instances.pop(session_id, None)
            self._multiagent_systems.pop(session_id, None)
            self._kg_instances.pop(session_id, None)
            self._session_metadata.pop(session_id, None)
            logger.info(f"✅ Cleaned up expired session: {session_id}")

    def list_sessions(self) -> List[str]:
        """활성 세션 목록 반환"""
        return list(self._session_metadata.keys())


# 전역 SessionManager 인스턴스
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """전역 SessionManager 인스턴스 가져오기"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
