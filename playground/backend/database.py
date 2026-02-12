"""
MongoDB Database Connection

Async MongoDB client using Motor for chat history storage.
"""

import logging
import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# MongoDB client singleton
_mongodb_client: Optional[AsyncIOMotorClient] = None
_mongodb_database: Optional[AsyncIOMotorDatabase] = None


def get_mongodb_client() -> Optional[AsyncIOMotorClient]:
    """Get MongoDB client singleton"""
    global _mongodb_client

    if _mongodb_client is None:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.warning("⚠️  MONGODB_URI not set, chat history will not be saved")
            return None

        try:
            _mongodb_client = AsyncIOMotorClient(
                mongodb_uri,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=45000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
            )
            logger.info("✅ MongoDB client initialized (pool: 5-50)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB client: {e}")
            return None

    return _mongodb_client


def get_mongodb_database() -> Optional[AsyncIOMotorDatabase]:
    """Get MongoDB database singleton"""
    global _mongodb_database

    if _mongodb_database is None:
        client = get_mongodb_client()
        if client is None:
            return None

        # Database name from URI or default to "beanllm"
        db_name = os.getenv("MONGODB_DATABASE", "beanllm")
        _mongodb_database = client[db_name]
        logger.info(f"✅ MongoDB database '{db_name}' initialized")

    return _mongodb_database


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global _mongodb_client, _mongodb_database

    if _mongodb_client is not None:
        _mongodb_client.close()
        _mongodb_client = None
        _mongodb_database = None
        logger.info("✅ MongoDB connection closed")


async def ping_mongodb() -> bool:
    """Check if MongoDB is connected"""
    try:
        client = get_mongodb_client()
        if client is None:
            return False

        await client.admin.command("ping")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB ping failed: {e}")
        return False


async def create_session_indexes():
    """
    세션 컬렉션 인덱스 생성

    성능 최적화를 위한 인덱스:
    - session_id: unique (빠른 조회)
    - updated_at: 정렬 최적화 (목록 조회)
    - feature_mode: 필터링 최적화
    - 복합 인덱스: feature_mode + updated_at
    """
    db = get_mongodb_database()
    if db is None:
        logger.debug("MongoDB not available, skipping index creation")
        return

    try:
        # session_id는 unique해야 함 (빠른 조회)
        await db.chat_sessions.create_index("session_id", unique=True, background=True)
        logger.info("✅ Created index: session_id (unique)")

        # updated_at으로 정렬 (목록 조회 최적화)
        await db.chat_sessions.create_index("updated_at", background=True)
        logger.info("✅ Created index: updated_at")

        # feature_mode 필터링
        await db.chat_sessions.create_index("feature_mode", background=True)
        logger.info("✅ Created index: feature_mode")

        # 복합 인덱스 (feature_mode + updated_at) - 가장 자주 사용되는 쿼리
        await db.chat_sessions.create_index(
            [("feature_mode", 1), ("updated_at", -1)], background=True
        )
        logger.info("✅ Created compound index: feature_mode + updated_at")

        # ✅ 고급 필터링을 위한 인덱스
        await db.chat_sessions.create_index("total_tokens", background=True)
        logger.info("✅ Created index: total_tokens")

        await db.chat_sessions.create_index("message_count", background=True)
        logger.info("✅ Created index: message_count")

        await db.chat_sessions.create_index("created_at", background=True)
        logger.info("✅ Created index: created_at")

        # ✅ 텍스트 검색 인덱스 (제목 검색 최적화)
        await db.chat_sessions.create_index("title", background=True)
        logger.info("✅ Created index: title")

        # ✅ 복합 인덱스 (날짜 범위 + 필터)
        await db.chat_sessions.create_index(
            [("updated_at", -1), ("total_tokens", 1)], background=True
        )
        logger.info("✅ Created compound index: updated_at + total_tokens")

        # ✅ 요약 관련 인덱스
        await db.chat_sessions.create_index("summary_created_at", background=True)
        logger.info("✅ Created index: summary_created_at")

        # ✅ 요약 존재 여부 확인용 sparse 인덱스
        await db.chat_sessions.create_index(
            "summary",
            background=True,
            sparse=True,  # summary 필드가 있는 문서만 인덱싱
        )
        logger.info("✅ Created sparse index: summary")

        logger.info("✅ All session indexes created successfully")

    except Exception as e:
        logger.error(f"❌ Failed to create session indexes: {e}")
        # 인덱스가 이미 존재하면 무시
