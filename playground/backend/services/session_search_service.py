"""
Session Search Service

세션 검색 서비스 - MongoDB + Vector DB 하이브리드 검색
beanllm의 Vector Store 기능을 활용하여 의미 기반 검색 제공

보안: 사용자 입력 query는 $regex 사용 전 re.escape()로 이스케이프하여
ReDoS 및 정규식 메타문자 인젝션 방지.
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Vector Store (선택적)
_vector_store = None
_embedding_function = None

try:
    from beanllm.domain.vector_stores.local.chroma import ChromaVectorStore

    # 임베딩 함수 - message_vector_store.py와 동일한 패턴
    # 우선순위: 1) Ollama (로컬 서버), 2) HuggingFace (로컬, 오픈소스)
    embedding_func = None

    # 1. Ollama 임베딩 시도 (로컬 서버, 빠름)
    try:
        from beanllm.domain.embeddings import OllamaEmbedding

        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        _embedding_function = OllamaEmbedding(model=embedding_model)
        embedding_func = _embedding_function.embed_sync  # 동기 버전 사용
        logger.info(f"✅ Session search using Ollama embedding: {embedding_model}")
    except Exception as e:
        logger.info(f"ℹ️  Ollama embedding not available: {e}")

        # 2. HuggingFace 임베딩 fallback
        try:
            from beanllm.domain.embeddings import HuggingFaceEmbedding

            hf_model = os.getenv(
                "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            _embedding_function = HuggingFaceEmbedding(
                model=hf_model,
                use_gpu=os.getenv("USE_GPU", "false").lower() == "true",
                normalize=True,
                batch_size=32,
            )
            embedding_func = _embedding_function.embed_sync
            logger.info(f"✅ Session search using HuggingFace embedding: {hf_model}")
        except Exception as e2:
            embedding_func = None
            logger.warning(f"⚠️  HuggingFace embedding not available: {e2}")

    # Vector Store 초기화 (세션 검색용)
    if embedding_func:
        _vector_store = ChromaVectorStore(
            collection_name="chat_sessions",
            embedding_function=embedding_func,
            persist_directory="./.chroma_sessions",  # 세션 검색 전용
        )
        logger.info("✅ Session search vector store initialized")
    else:
        _vector_store = None
        logger.warning("⚠️  Embedding function not available, vector search disabled")
except Exception as e:
    logger.warning(f"⚠️  Vector store not available for session search: {e}")
    _vector_store = None


class SessionSearchService:
    """세션 검색 서비스"""

    @staticmethod
    async def search_sessions(
        query: str,
        db,
        limit: int = 20,
        skip: int = 0,
        feature_mode: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None,
        sort_by: str = "relevance",  # "relevance", "updated_at", "created_at", "total_tokens"
        use_vector_search: bool = True,
    ) -> Dict[str, Any]:
        """
        세션 검색 (하이브리드: MongoDB + Vector DB)

        Args:
            query: 검색 쿼리
            db: MongoDB 데이터베이스
            limit: 결과 개수
            skip: 건너뛸 개수
            feature_mode: 필터 모드
            date_from: 시작 날짜
            date_to: 종료 날짜
            min_tokens: 최소 토큰 수
            max_tokens: 최대 토큰 수
            min_messages: 최소 메시지 수
            max_messages: 최대 메시지 수
            sort_by: 정렬 기준 ("relevance", "updated_at", "created_at", "total_tokens")
            use_vector_search: Vector DB 사용 여부

        Returns:
            {"sessions": [...], "total": ..., "search_type": "vector" | "keyword"}
        """
        # 1. MongoDB 기본 필터
        mongo_query = {}

        if feature_mode:
            mongo_query["feature_mode"] = feature_mode

        if date_from or date_to:
            date_filter = {}
            if date_from:
                date_filter["$gte"] = date_from
            if date_to:
                date_filter["$lte"] = date_to
            mongo_query["updated_at"] = date_filter

        if min_tokens is not None or max_tokens is not None:
            token_filter = {}
            if min_tokens is not None:
                token_filter["$gte"] = min_tokens
            if max_tokens is not None:
                token_filter["$lte"] = max_tokens
            mongo_query["total_tokens"] = token_filter

        if min_messages is not None or max_messages is not None:
            message_filter = {}
            if min_messages is not None:
                message_filter["$gte"] = min_messages
            if max_messages is not None:
                message_filter["$lte"] = max_messages
            mongo_query["message_count"] = message_filter

        # 2. Vector DB 검색 (의미 기반) - 메시지 내용 검색
        if use_vector_search and query:
            try:
                # ✅ 메시지 Vector DB에서 검색
                from services.message_vector_store import message_vector_store

                # 메시지 검색 (의미 기반)
                message_results = await message_vector_store.search_messages(
                    query=query,
                    session_id=None,  # 모든 세션에서 검색
                    role=None,
                    k=limit * 3,  # 더 많이 가져와서 세션별로 그룹화
                )

                if message_results:
                    # session_id로 그룹화하고 최고 점수 사용
                    session_scores = {}
                    for msg in message_results:
                        msg_session_id = msg.get("session_id")
                        if msg_session_id:
                            score = msg.get("relevance_score", 0)
                            # 최고 점수만 유지
                            if (
                                msg_session_id not in session_scores
                                or score > session_scores[msg_session_id]
                            ):
                                session_scores[msg_session_id] = score

                    # 점수 순으로 정렬된 session_id 리스트
                    sorted_session_ids = sorted(
                        session_scores.keys(), key=lambda sid: session_scores[sid], reverse=True
                    )

                    # MongoDB에서 해당 세션들 조회
                    if sorted_session_ids:
                        mongo_query["session_id"] = {"$in": sorted_session_ids}

                        # MongoDB에서 세션 메타데이터 가져오기
                        sessions = []
                        async for doc in db.chat_sessions.find(mongo_query):
                            doc["_id"] = str(doc["_id"])
                            # Vector DB 점수 추가
                            doc["relevance_score"] = session_scores.get(doc["session_id"], 0)
                            doc.pop("messages", None)  # 목록에서는 제외
                            sessions.append(doc)

                        # 점수 순으로 정렬
                        sessions.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

                        # 페이지네이션
                        total = len(sessions)
                        sessions = sessions[skip : skip + limit]

                        logger.info(
                            f"✅ Vector search: {len(sessions)} sessions found from message search"
                        )
                        return {
                            "sessions": sessions,
                            "total": total,
                            "search_type": "vector",
                            "query": query,
                        }

            except Exception as e:
                logger.warning(f"Vector search failed, falling back to keyword: {e}")

        # 3. MongoDB 키워드 검색 (Fallback)
        # 제목 또는 메시지 내용에서 검색.
        # 사용자 입력 query는 반드시 이스케이프하여 ReDoS·정규식 인젝션 방지.
        if query:
            safe_pattern = re.escape(query)
            title_query = {"title": {"$regex": safe_pattern, "$options": "i"}}
            message_query = {"messages.content": {"$regex": safe_pattern, "$options": "i"}}
            mongo_query["$or"] = [title_query, message_query]

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

        # MongoDB 조회
        total = await db.chat_sessions.count_documents(mongo_query)
        cursor = (
            db.chat_sessions.find(mongo_query)
            .sort(sort_field, sort_direction)
            .skip(skip)
            .limit(limit)
        )

        sessions = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc.pop("messages", None)  # 목록에서는 제외
            sessions.append(doc)

        logger.info(f"✅ Keyword search: {len(sessions)} sessions found")
        return {"sessions": sessions, "total": total, "search_type": "keyword", "query": query}

    @staticmethod
    async def index_session(session_id: str, session_data: Dict[str, Any]):
        """
        세션을 Vector DB에 인덱싱 (검색 최적화)

        세션의 제목과 메시지 내용을 임베딩하여 Vector DB에 저장
        """
        if not _vector_store:
            return

        try:
            # 세션 텍스트 생성 (제목 + 최근 메시지)
            messages = session_data.get("messages", [])
            recent_messages = messages[-10:] if len(messages) > 10 else messages

            # 검색 가능한 텍스트 생성
            searchable_text = f"{session_data.get('title', '')}\n"
            for msg in recent_messages:
                searchable_text += f"{msg.get('content', '')}\n"

            # Vector DB에 문서 추가
            # Chroma collection에 직접 추가하여 session_id를 ID로 사용
            texts = [searchable_text]
            metadatas = [
                {
                    "session_id": session_id,
                    "title": session_data.get("title", ""),
                    "feature_mode": session_data.get("feature_mode", ""),
                    "updated_at": session_data.get(
                        "updated_at", datetime.now(timezone.utc)
                    ).isoformat(),
                }
            ]

            # 임베딩 생성
            if _vector_store.embedding_function:
                embeddings = _vector_store.embedding_function(texts)
            else:
                embeddings = None

            # Chroma collection에 추가 (session_id를 ID로 사용)
            # 기존 문서가 있으면 업데이트 (upsert)
            if embeddings:
                _vector_store.collection.upsert(
                    documents=texts,
                    metadatas=metadatas,
                    ids=[session_id],  # session_id를 ID로 사용
                    embeddings=embeddings,
                )
            else:
                _vector_store.collection.upsert(
                    documents=texts, metadatas=metadatas, ids=[session_id]
                )

            logger.debug(f"✅ Indexed session: {session_id}")

        except Exception as e:
            logger.warning(f"Failed to index session: {e}")

    @staticmethod
    async def remove_session_index(session_id: str):
        """세션 인덱스 제거 (더 이상 필요 없음 - 메시지가 개별적으로 관리됨)"""
        # 하위 호환성을 위해 빈 함수로 유지
        # 실제 메시지 삭제는 message_vector_store.delete_session_messages()에서 처리
        pass


# 전역 인스턴스
session_search = SessionSearchService()
