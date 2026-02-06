"""
Chat History API Endpoints

REST API for managing chat sessions and messages in MongoDB.
Redis 캐싱을 통한 성능 최적화 포함.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from database import get_mongodb_database
from fastapi import APIRouter, HTTPException
from motor.motor_asyncio import AsyncIOMotorDatabase
from schemas.database import (
    AddMessageRequest,
    CreateSessionRequest,
    SessionListResponse,
    SessionResponse,
)
from services.session_cache import session_cache
from services.title_generator import generate_chat_title

logger = logging.getLogger(__name__)

# Display title when stored title is empty
FALLBACK_TITLE = "New chat"

router = APIRouter(prefix="/api/chat/sessions", tags=["Chat History"])


def _get_db() -> AsyncIOMotorDatabase:
    """Get MongoDB database or raise error"""
    db = get_mongodb_database()
    if db is None:
        raise HTTPException(
            status_code=503, detail="MongoDB not configured. Set MONGODB_URI environment variable."
        )
    return db


def _normalize_title(title: Optional[str]) -> str:
    """Return display title; empty or None becomes FALLBACK_TITLE."""
    if not title or not str(title).strip():
        return FALLBACK_TITLE
    return str(title).strip()


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(request: CreateSessionRequest):
    """
    Create a new chat session.

    If request.title is provided (and not empty / "New chat"), an open-source
    model is used to generate a short title from it. Otherwise title is "New chat".
    """
    db = _get_db()

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    raw_title = (request.title or "").strip()
    if raw_title and raw_title.lower() != "new chat":
        generated = await generate_chat_title(raw_title)
        title = _normalize_title(generated) if generated else FALLBACK_TITLE
    else:
        title = FALLBACK_TITLE

    session_data = {
        "session_id": session_id,
        "title": title,
        "feature_mode": request.feature_mode,
        "model": request.model,
        "messages": [],
        "feature_options": request.feature_options or {},
        "created_at": now,
        "updated_at": now,
        "total_tokens": 0,
        "message_count": 0,
    }

    try:
        result = await db.chat_sessions.insert_one(session_data)
        session_data["_id"] = str(result.inserted_id)

        # ✅ Redis에 캐시 저장
        await session_cache.set_session(session_id, session_data)

        # ✅ 세션 목록 캐시 무효화 (새 세션이 추가되었으므로)
        await session_cache.invalidate_session_lists()

        # ✅ Vector DB에 인덱싱 (검색 최적화)
        try:
            from services.session_search_service import session_search

            await session_search.index_session(session_id, session_data)
        except Exception as e:
            logger.warning(f"Failed to index session in vector DB: {e}")

        logger.info(f"✅ Created chat session: {session_id}")
        return SessionResponse(session=session_data)
    except Exception as e:
        logger.error(f"❌ Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    skip: int = 0,
    feature_mode: Optional[str] = None,
    # ✅ 고급 필터링
    query: Optional[str] = None,  # 검색 쿼리 (제목/메시지 내용)
    date_from: Optional[str] = None,  # ISO format: "2026-01-01T00:00:00Z"
    date_to: Optional[str] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    min_messages: Optional[int] = None,
    max_messages: Optional[int] = None,
    sort_by: str = "updated_at",  # "updated_at", "created_at", "total_tokens", "message_count", "relevance"
    use_vector_search: bool = True,  # Vector DB 사용 여부
):
    """
    List chat sessions with advanced filtering and search (with Redis caching)

    고급 필터링 및 검색 기능:
    - 텍스트 검색: 제목/메시지 내용
    - Vector DB 검색: 의미 기반 검색 (beanllm 활용)
    - 날짜 범위 필터링
    - 토큰/메시지 수 필터링
    - 다양한 정렬 옵션

    캐싱: Redis에 5분간 캐시 (세션 생성/수정/삭제 시 자동 무효화)
    """
    db = _get_db()

    try:
        # ✅ 검색 쿼리가 있으면 Vector DB 검색 사용
        if query:
            from datetime import datetime

            from services.session_search_service import session_search

            # 날짜 파싱
            date_from_dt = (
                datetime.fromisoformat(date_from.replace("Z", "+00:00")) if date_from else None
            )
            date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00")) if date_to else None

            result = await session_search.search_sessions(
                query=query,
                db=db,
                limit=limit,
                skip=skip,
                feature_mode=feature_mode,
                date_from=date_from_dt,
                date_to=date_to_dt,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                min_messages=min_messages,
                max_messages=max_messages,
                sort_by=sort_by,
                use_vector_search=use_vector_search,
            )

            sessions_out = result["sessions"]
            for s in sessions_out:
                s["title"] = _normalize_title(s.get("title"))
            logger.info(
                f"✅ Searched sessions: {result['search_type']} search, {len(sessions_out)} results"
            )
            return SessionListResponse(sessions=sessions_out, total=result["total"])

        # ✅ 일반 목록 조회 (캐싱)
        # 캐시 키에 모든 필터 포함
        cache_key_params = f"{feature_mode}:{date_from}:{date_to}:{min_tokens}:{max_tokens}:{min_messages}:{max_messages}:{sort_by}"
        cached = await session_cache.get_session_list(cache_key_params, skip, limit)
        if cached:
            logger.debug("✅ Session list cache hit")
            for s in cached.get("sessions", []):
                s["title"] = _normalize_title(s.get("title"))
            return SessionListResponse(**cached)

        # Build query
        mongo_query = {}
        if feature_mode:
            mongo_query["feature_mode"] = feature_mode

        # 날짜 필터
        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                date_filter["$gte"] = date_from_dt
            if date_to:
                date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                date_filter["$lte"] = date_to_dt
            mongo_query["updated_at"] = date_filter

        # 토큰 필터
        if min_tokens is not None or max_tokens is not None:
            token_filter = {}
            if min_tokens is not None:
                token_filter["$gte"] = min_tokens
            if max_tokens is not None:
                token_filter["$lte"] = max_tokens
            mongo_query["total_tokens"] = token_filter

        # 메시지 수 필터
        if min_messages is not None or max_messages is not None:
            message_filter = {}
            if min_messages is not None:
                message_filter["$gte"] = min_messages
            if max_messages is not None:
                message_filter["$lte"] = max_messages
            mongo_query["message_count"] = message_filter

        # Get total count
        total = await db.chat_sessions.count_documents(mongo_query)

        # 정렬
        sort_field = "updated_at"
        sort_direction = -1

        if sort_by == "created_at":
            sort_field = "created_at"
        elif sort_by == "total_tokens":
            sort_field = "total_tokens"
            sort_direction = -1
        elif sort_by == "message_count":
            sort_field = "message_count"
            sort_direction = -1

        # Get sessions
        cursor = (
            db.chat_sessions.find(mongo_query)
            .sort(sort_field, sort_direction)
            .skip(skip)
            .limit(limit)
        )
        sessions = []

        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if "message_count" not in doc:
                doc["message_count"] = len(doc.get("messages", []))
            doc.pop("messages", None)
            doc["title"] = _normalize_title(doc.get("title"))
            sessions.append(doc)

        # ✅ Redis에 캐시 저장
        await session_cache.set_session_list(sessions, total, cache_key_params, skip, limit)

        logger.info(f"✅ Listed {len(sessions)} sessions (total: {total})")
        return SessionListResponse(sessions=sessions, total=total)
    except Exception as e:
        logger.error(f"❌ Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get a specific session with all messages (with Redis caching)

    MongoDB에서 메타데이터 조회 + Vector DB에서 실제 메시지 내용 조회

    캐싱: Redis에 1분간 캐시 (세션 수정 시 자동 무효화)
    """
    db = _get_db()

    try:
        # ✅ Redis 캐시 확인
        cached = await session_cache.get_session(session_id)
        if cached:
            logger.debug(f"✅ Session cache hit: {session_id}")
            cached["title"] = _normalize_title(cached.get("title"))
            from services.message_vector_store import message_vector_store

            messages = await message_vector_store.get_session_messages(session_id)
            if messages:
                cached["messages"] = messages
            return SessionResponse(session=cached)

        # MongoDB에서 메타데이터 조회
        session = await db.chat_sessions.find_one({"session_id": session_id})
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        session["_id"] = str(session["_id"])
        session["title"] = _normalize_title(session.get("title"))

        from services.message_vector_store import message_vector_store

        messages = await message_vector_store.get_session_messages(session_id)
        if messages:
            session["messages"] = messages

        await session_cache.set_session(session_id, session)

        logger.info(f"✅ Retrieved session: {session_id} ({len(messages)} messages from vector DB)")
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

    MongoDB에는 메타데이터만 저장, 실제 메시지 내용은 Vector DB에 저장
    """
    db = _get_db()

    try:
        # Check if session exists
        session = await db.chat_sessions.find_one({"session_id": session_id})
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # 메시지 ID 생성
        message_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)

        # ✅ Vector DB에 메시지 저장
        from services.message_vector_store import message_vector_store

        await message_vector_store.save_message(
            session_id=session_id,
            message_id=message_id,
            role=request.role,
            content=request.content,
            model=request.model,
            timestamp=now,
            metadata={"usage": request.usage, **(request.metadata or {})},
        )

        # MongoDB에는 메타데이터만 저장 (메시지 내용은 Vector DB에)
        # 메시지 요약 정보만 저장 (선택적)
        message_summary = {
            "message_id": message_id,
            "role": request.role,
            "content_preview": request.content[:100]
            if len(request.content) > 100
            else request.content,  # 미리보기만
            "timestamp": now,
            "model": request.model,
        }

        # Update session (메시지 배열은 유지하되, 요약만 저장)
        update_data = {
            "$push": {"messages": message_summary},  # 요약만 저장
            "$set": {"updated_at": now},
            "$inc": {"message_count": 1},
        }

        # Update total tokens if usage provided
        if request.usage and "total_tokens" in request.usage:
            update_data["$inc"]["total_tokens"] = request.usage["total_tokens"]

        result = await db.chat_sessions.find_one_and_update(
            {"session_id": session_id}, update_data, return_document=True
        )

        result["_id"] = str(result["_id"])

        # ✅ 세션 캐시 업데이트
        await session_cache.set_session(session_id, result)

        # ✅ 세션 목록 캐시 무효화
        await session_cache.invalidate_session_lists()

        # ✅ 세션 검색 인덱싱은 더 이상 필요 없음 (메시지가 개별적으로 Vector DB에 저장됨)

        logger.info(
            f"✅ Added message to session: {session_id} (role: {request.role}, message_id: {message_id})"
        )
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
            {"$set": {"title": title, "updated_at": datetime.now(timezone.utc)}},
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # ✅ 세션 캐시 무효화
        await session_cache.invalidate_session(session_id)
        await session_cache.invalidate_session_lists()

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

        # ✅ 세션 캐시 무효화
        await session_cache.invalidate_session(session_id)
        await session_cache.invalidate_session_lists()

        # ✅ Vector DB에서 세션의 모든 메시지 삭제
        try:
            from services.message_vector_store import message_vector_store

            await message_vector_store.delete_session_messages(session_id)
        except Exception as e:
            logger.warning(f"Failed to delete messages from vector DB: {e}")

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
        session = await db.chat_sessions.find_one({"session_id": session_id}, {"messages": 1})

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


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    Get session summary

    Returns the session's conversation summary.
    Uses Redis cache → MongoDB fallback.
    """
    try:
        # 1. Redis 캐시에서 요약 확인
        cached_summary = await session_cache.get_summary(session_id)
        if cached_summary:
            logger.debug(f"✅ Summary cache hit: {session_id}")
            return {
                "session_id": session_id,
                "summary": cached_summary,
                "source": "cache",
            }

        # 2. MongoDB에서 요약 조회
        db = _get_db()
        session = await db.chat_sessions.find_one(
            {"session_id": session_id},
            {"summary": 1, "summary_created_at": 1, "summary_message_count": 1},
        )

        if session and session.get("summary"):
            # Redis에 캐시
            await session_cache.set_summary(session_id, session["summary"])

            return {
                "session_id": session_id,
                "summary": session["summary"],
                "created_at": session.get("summary_created_at"),
                "message_count": session.get("summary_message_count"),
                "source": "database",
            }

        # 3. 요약이 없음
        return {
            "session_id": session_id,
            "summary": None,
            "message": "No summary available. Summary is generated when conversation reaches 10+ messages.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get session summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session summary: {str(e)}")


@router.post("/{session_id}/summary")
async def generate_session_summary(session_id: str, model: Optional[str] = "qwen2.5:0.5b"):
    """
    Generate session summary

    Generates a summary of the conversation using the ContextManager.
    """
    try:
        from services.context_manager import context_manager

        # 요약 생성
        summary = await context_manager.summarize_if_needed(session_id, model=model)

        if summary:
            return {
                "session_id": session_id,
                "summary": summary,
                "generated": True,
            }

        return {
            "session_id": session_id,
            "summary": None,
            "generated": False,
            "message": "Summary not generated. Session may not have enough messages.",
        }

    except Exception as e:
        logger.error(f"❌ Failed to generate session summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate session summary: {str(e)}")
