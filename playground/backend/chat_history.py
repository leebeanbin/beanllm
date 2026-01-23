"""
Chat History API Endpoints

REST API for managing chat sessions and messages in MongoDB.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase

from database import get_mongodb_database
from models import (
    ChatSession,
    ChatMessage,
    CreateSessionRequest,
    AddMessageRequest,
    SessionListResponse,
    SessionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat/sessions", tags=["Chat History"])


def _get_db() -> AsyncIOMotorDatabase:
    """Get MongoDB database or raise error"""
    db = get_mongodb_database()
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB not configured. Set MONGODB_URI environment variable."
        )
    return db


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(request: CreateSessionRequest):
    """
    Create a new chat session

    Creates a new chat session with the specified model and feature mode.
    """
    db = _get_db()

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow()

    session_data = {
        "session_id": session_id,
        "title": request.title,
        "feature_mode": request.feature_mode,
        "model": request.model,
        "messages": [],
        "feature_options": request.feature_options,
        "created_at": now,
        "updated_at": now,
        "total_tokens": 0,
        "message_count": 0,
    }

    try:
        result = await db.chat_sessions.insert_one(session_data)
        session_data["_id"] = str(result.inserted_id)
        logger.info(f"✅ Created chat session: {session_id}")
        return SessionResponse(session=session_data)
    except Exception as e:
        logger.error(f"❌ Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    skip: int = 0,
    feature_mode: Optional[str] = None
):
    """
    List chat sessions

    Returns a list of chat sessions, optionally filtered by feature mode.
    Results are sorted by updated_at (most recent first).
    """
    db = _get_db()

    try:
        # Build query
        query = {}
        if feature_mode:
            query["feature_mode"] = feature_mode

        # Get total count
        total = await db.chat_sessions.count_documents(query)

        # Get sessions
        cursor = db.chat_sessions.find(query).sort("updated_at", -1).skip(skip).limit(limit)
        sessions = []

        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            # Don't include full messages in list view (just summary)
            doc["message_count"] = len(doc.get("messages", []))
            doc.pop("messages", None)
            sessions.append(doc)

        logger.info(f"✅ Listed {len(sessions)} sessions (total: {total})")
        return SessionListResponse(sessions=sessions, total=total)
    except Exception as e:
        logger.error(f"❌ Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get a specific session with all messages

    Returns the full session data including all messages.
    """
    db = _get_db()

    try:
        session = await db.chat_sessions.find_one({"session_id": session_id})
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session["_id"] = str(session["_id"])
        logger.info(f"✅ Retrieved session: {session_id}")
        return SessionResponse(session=session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.post("/{session_id}/messages", response_model=SessionResponse)
async def add_message(session_id: str, request: AddMessageRequest):
    """
    Add a message to a session

    Appends a new message to the session and updates statistics.
    """
    db = _get_db()

    try:
        # Check if session exists
        session = await db.chat_sessions.find_one({"session_id": session_id})
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Create message
        message = {
            "role": request.role,
            "content": request.content,
            "timestamp": datetime.utcnow(),
            "model": request.model,
            "usage": request.usage,
            "metadata": request.metadata,
        }

        # Update session
        update_data = {
            "$push": {"messages": message},
            "$set": {"updated_at": datetime.utcnow()},
            "$inc": {"message_count": 1}
        }

        # Update total tokens if usage provided
        if request.usage and "total_tokens" in request.usage:
            update_data["$inc"]["total_tokens"] = request.usage["total_tokens"]

        result = await db.chat_sessions.find_one_and_update(
            {"session_id": session_id},
            update_data,
            return_document=True  # Return updated document
        )

        result["_id"] = str(result["_id"])
        logger.info(f"✅ Added message to session: {session_id} (role: {request.role})")
        return SessionResponse(session=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to add message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")


@router.patch("/{session_id}/title")
async def update_session_title(session_id: str, title: str):
    """
    Update session title

    Updates the title of a session (useful for auto-generating titles from first message).
    """
    db = _get_db()

    try:
        result = await db.chat_sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "title": title,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        logger.info(f"✅ Updated session title: {session_id} → {title}")
        return {"success": True, "session_id": session_id, "title": title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update title: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}")


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session

    Permanently deletes a session and all its messages.
    """
    db = _get_db()

    try:
        result = await db.chat_sessions.delete_one({"session_id": session_id})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        logger.info(f"✅ Deleted session: {session_id}")
        return {"success": True, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.get("/{session_id}/messages")
async def get_messages(session_id: str, limit: Optional[int] = None):
    """
    Get messages from a session

    Returns only the messages from a session (without full session data).
    Useful for pagination.
    """
    db = _get_db()

    try:
        session = await db.chat_sessions.find_one(
            {"session_id": session_id},
            {"messages": 1}
        )

        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        messages = session.get("messages", [])

        # Apply limit if specified
        if limit is not None and limit > 0:
            messages = messages[-limit:]  # Get last N messages

        logger.info(f"✅ Retrieved {len(messages)} messages from session: {session_id}")
        return {"messages": messages, "count": len(messages)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")
