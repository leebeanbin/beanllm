"""
SessionRAGService - 세션별 RAG 자동 관리

세션 생성/삭제 시 RAG 컬렉션 자동 관리
문서 업로드 시 자동 인덱싱
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionRAGInfo:
    """세션별 RAG 정보"""

    session_id: str
    collection_name: str
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sources: List[str] = field(default_factory=list)  # 문서 출처 목록


class SessionRAGService:
    """
    세션별 RAG 자동 관리 서비스

    기능:
    - 세션별 RAG 컬렉션 자동 생성
    - 세션별 RAG 인스턴스 관리
    - 문서 업로드 시 자동 인덱싱
    - 세션 삭제 시 RAG 컬렉션 정리
    """

    def __init__(
        self,
        default_chunk_size: int = 500,
        default_chunk_overlap: int = 50,
        auto_create: bool = True,
    ):
        """
        Args:
            default_chunk_size: 기본 청크 크기
            default_chunk_overlap: 기본 청크 오버랩
            auto_create: 세션 접근 시 자동 생성
        """
        self._sessions: Dict[str, SessionRAGInfo] = {}
        self._rag_instances: Dict[str, Any] = {}  # RAGChain 인스턴스
        self._chunk_size = default_chunk_size
        self._chunk_overlap = default_chunk_overlap
        self._auto_create = auto_create

        logger.info(
            f"SessionRAGService initialized: chunk_size={default_chunk_size}, "
            f"auto_create={auto_create}"
        )

    def _get_collection_name(self, session_id: str) -> str:
        """세션 ID로 컬렉션 이름 생성"""
        # 컬렉션 이름 규칙: session_rag_{session_id}
        safe_id = session_id.replace("-", "_")[:50]
        return f"session_rag_{safe_id}"

    async def get_or_create_rag(
        self,
        session_id: str,
        create_if_missing: bool = True,
    ) -> Optional[Any]:
        """
        세션 RAG 인스턴스 가져오기 (없으면 생성)

        Args:
            session_id: 세션 ID
            create_if_missing: 없으면 생성할지 여부

        Returns:
            RAGChain 인스턴스 (없으면 None)
        """
        # 이미 인스턴스가 있으면 반환
        if session_id in self._rag_instances:
            return self._rag_instances[session_id]

        # 자동 생성이 비활성화되어 있으면 None
        if not create_if_missing and not self._auto_create:
            return None

        # RAG 인스턴스 생성
        try:
            from pathlib import Path

            from beanllm.domain.vector_stores import ChromaVectorStore
            from beanllm.facade.core.rag_facade import RAGChain

            collection_name = self._get_collection_name(session_id)

            # 세션별 persist 디렉토리
            persist_dir = Path("./.session_rag_data") / collection_name
            persist_dir.mkdir(parents=True, exist_ok=True)

            # Vector Store 생성 (persist_directory로 충돌 방지)
            vector_store = ChromaVectorStore(
                collection_name=collection_name,
                persist_directory=str(persist_dir),
            )

            # RAG 체인 생성 (Vector Store 기반)
            rag = RAGChain(vector_store=vector_store)

            self._rag_instances[session_id] = rag
            self._sessions[session_id] = SessionRAGInfo(
                session_id=session_id,
                collection_name=collection_name,
            )

            logger.info(f"Created RAG instance for session {session_id}")
            return rag

        except Exception as e:
            logger.error(f"Failed to create RAG for session {session_id}: {e}")
            return None

    async def add_document(
        self,
        session_id: str,
        content: str,
        source: str = "upload",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        세션 RAG에 문서 추가

        Args:
            session_id: 세션 ID
            content: 문서 내용
            source: 문서 출처
            metadata: 추가 메타데이터

        Returns:
            성공 여부
        """
        try:
            rag = await self.get_or_create_rag(session_id)
            if rag is None:
                return False

            from beanllm.domain.loaders import Document

            doc_metadata = metadata or {}
            doc_metadata["source"] = source
            doc_metadata["session_id"] = session_id
            doc_metadata["added_at"] = datetime.now(timezone.utc).isoformat()

            doc = Document(content=content, metadata=doc_metadata)

            # RAG의 Vector Store에 문서 추가
            rag.vector_store.add_documents([doc])

            # 세션 정보 업데이트
            if session_id in self._sessions:
                info = self._sessions[session_id]
                info.document_count += 1
                info.updated_at = datetime.now(timezone.utc)
                if source not in info.sources:
                    info.sources.append(source)

            logger.info(f"Added document to session {session_id} RAG: source={source}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document to session {session_id}: {e}")
            return False

    async def add_documents_batch(
        self,
        session_id: str,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        세션 RAG에 여러 문서 추가

        Args:
            session_id: 세션 ID
            documents: 문서 리스트 [{"content": "...", "source": "...", "metadata": {...}}]

        Returns:
            추가된 문서 수
        """
        added = 0
        for doc in documents:
            success = await self.add_document(
                session_id=session_id,
                content=doc.get("content", ""),
                source=doc.get("source", "batch"),
                metadata=doc.get("metadata"),
            )
            if success:
                added += 1

        return added

    async def query(
        self,
        session_id: str,
        query: str,
        k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        세션 RAG 검색

        Args:
            session_id: 세션 ID
            query: 검색 쿼리
            k: 반환할 결과 수

        Returns:
            검색 결과 (없으면 None)
        """
        rag = await self.get_or_create_rag(session_id, create_if_missing=False)
        if rag is None:
            return None

        try:
            # RAG 검색 실행 (vector_store의 similarity_search 사용)
            results = rag.vector_store.similarity_search(query, k=k)

            return {
                "query": query,
                "results": [
                    {
                        "content": getattr(doc, "content", getattr(doc, "page_content", "")),
                        "metadata": getattr(doc, "metadata", {}),
                        "score": getattr(doc, "score", None),
                    }
                    for doc in results
                ],
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"RAG query failed for session {session_id}: {e}")
            return None

    async def query_with_generation(
        self,
        session_id: str,
        query: str,
        model: str = "qwen2.5:0.5b",
        k: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        세션 RAG 검색 + 응답 생성

        Args:
            session_id: 세션 ID
            query: 검색 쿼리
            model: 응답 생성 모델
            k: 반환할 결과 수

        Returns:
            검색 결과 및 생성된 응답
        """
        rag = await self.get_or_create_rag(session_id, create_if_missing=False)
        if rag is None:
            return None

        try:
            # RAG 검색 + 생성 (동기 메서드)
            result = rag.query(query, model=model, k=k)

            return {
                "query": query,
                "answer": result.answer if hasattr(result, "answer") else str(result),
                "sources": [
                    {
                        "content": getattr(src, "content", getattr(src, "page_content", ""))[:200],
                        "metadata": getattr(src, "metadata", {}),
                    }
                    for src in getattr(result, "sources", [])
                ],
            }

        except Exception as e:
            logger.error(f"RAG query with generation failed for session {session_id}: {e}")
            return None

    async def delete_session_rag(self, session_id: str) -> bool:
        """
        세션 RAG 삭제

        Args:
            session_id: 세션 ID

        Returns:
            성공 여부
        """
        try:
            # RAG 인스턴스 삭제
            if session_id in self._rag_instances:
                rag = self._rag_instances[session_id]

                # 벡터 DB 컬렉션 삭제
                if hasattr(rag, "delete_collection"):
                    await rag.delete_collection()

                del self._rag_instances[session_id]

            # 세션 정보 삭제
            if session_id in self._sessions:
                del self._sessions[session_id]

            logger.info(f"Deleted RAG for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete RAG for session {session_id}: {e}")
            return False

    def get_session_info(self, session_id: str) -> Optional[SessionRAGInfo]:
        """
        세션 RAG 정보 조회

        Args:
            session_id: 세션 ID

        Returns:
            세션 RAG 정보 (없으면 None)
        """
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """활성 세션 RAG 목록"""
        return [
            {
                "session_id": info.session_id,
                "collection_name": info.collection_name,
                "document_count": info.document_count,
                "sources": info.sources,
                "created_at": info.created_at.isoformat(),
                "updated_at": info.updated_at.isoformat(),
            }
            for info in self._sessions.values()
        ]

    async def cleanup_inactive_sessions(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """
        비활성 세션 RAG 정리

        Args:
            max_age_hours: 최대 비활성 시간 (시간)

        Returns:
            정리된 세션 수
        """
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (max_age_hours * 3600)

        sessions_to_delete = [
            session_id
            for session_id, info in self._sessions.items()
            if info.updated_at.timestamp() < cutoff
        ]

        deleted = 0
        for session_id in sessions_to_delete:
            if await self.delete_session_rag(session_id):
                deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} inactive session RAGs")

        return deleted


# Singleton instance
session_rag_service = SessionRAGService()
