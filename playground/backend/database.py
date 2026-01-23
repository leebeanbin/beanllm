"""
MongoDB Database Connection

Async MongoDB client using Motor for chat history storage.
"""

import os
import logging
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
            _mongodb_client = AsyncIOMotorClient(mongodb_uri)
            logger.info("✅ MongoDB client initialized")
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
