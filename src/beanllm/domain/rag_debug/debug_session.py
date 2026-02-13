"""
DebugSession - RAG 디버깅 세션 관리
SOLID 원칙:
- SRP: 세션 관리와 데이터 수집만 담당
- OCP: 확장 가능한 데이터 수집 인터페이스
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.vector_stores import BaseVectorStore

logger = get_logger(__name__)


class DebugSession:
    """
    RAG 디버깅 세션

    책임:
    - VectorStore로부터 디버깅 데이터 수집
    - 세션 상태 관리
    - 분석 결과 캐싱

    SOLID:
    - SRP: 세션 관리만
    - DIP: VectorStore 인터페이스에 의존
    """

    def __init__(
        self,
        vector_store: "BaseVectorStore",
        session_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            vector_store: 디버깅할 VectorStore
            session_name: 세션 이름 (optional)
            session_id: 세션 ID (optional, 자동 생성됨)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.session_name = session_name or f"debug_{self.session_id[:8]}"
        self.vector_store = vector_store
        self.created_at = datetime.now()
        self.status = "initialized"

        # Cache for analysis results
        self._cache: Dict[str, Any] = {}
        self._documents: Optional[List[Any]] = None
        self._embeddings: Optional[List[List[float]]] = None
        self._metadata: Optional[Dict[str, Any]] = None

        logger.info(f"Debug session created: {self.session_id}")

    def get_documents(self) -> List[Any]:
        """
        VectorStore에서 모든 documents 가져오기

        Returns:
            List[Document]: 모든 documents

        Note:
            각 VectorStore 구현체마다 내부 API가 다를 수 있음
            현재는 generic한 접근법 사용
        """
        if self._documents is not None:
            return self._documents

        # TODO: VectorStore별 최적화 필요
        # 현재는 dummy implementation (실제 구현은 각 VectorStore에 맞게)
        logger.warning(
            "get_documents() uses generic approach. "
            "Optimize for specific VectorStore implementations."
        )

        # Try to access internal storage (implementation-specific)
        documents = []
        try:
            # Attempt 1: Check if VectorStore has _documents attribute (some implementations)
            if hasattr(self.vector_store, "_documents"):
                documents = self.vector_store._documents
            # Attempt 2: Check if it has docstore (Chroma, FAISS style)
            elif hasattr(self.vector_store, "docstore"):
                documents = list(self.vector_store.docstore._dict.values())
            # Attempt 3: Use collection.get() for Chroma
            elif hasattr(self.vector_store, "_collection"):
                result = self.vector_store._collection.get()
                # Convert to Document-like objects
                from beanllm.domain.loaders import Document

                documents = [
                    Document(content=text, metadata=meta or {})
                    for text, meta in zip(result.get("documents", []), result.get("metadatas", []))
                ]
            else:
                logger.error(f"Cannot extract documents from {type(self.vector_store)}")
                documents = []

        except Exception as e:
            logger.error(f"Error getting documents: {e}")
            documents = []

        self._documents = documents
        logger.info(f"Loaded {len(documents)} documents from VectorStore")
        return documents

    def get_embeddings(self) -> List[List[float]]:
        """
        VectorStore에서 모든 embeddings 가져오기

        Returns:
            List[List[float]]: 모든 embedding vectors

        Note:
            VectorStore별로 내부 구조가 다름
        """
        if self._embeddings is not None:
            return self._embeddings

        logger.warning(
            "get_embeddings() uses generic approach. "
            "Optimize for specific VectorStore implementations."
        )

        embeddings = []
        try:
            # Attempt 1: Check if VectorStore has _embeddings or similar
            if hasattr(self.vector_store, "_embeddings"):
                embeddings = self.vector_store._embeddings
            # Attempt 2: For Chroma, use collection.get(include=["embeddings"])
            elif hasattr(self.vector_store, "_collection"):
                result = self.vector_store._collection.get(include=["embeddings"])
                embeddings = result.get("embeddings", [])
            # Attempt 3: For FAISS, access index
            elif hasattr(self.vector_store, "index"):
                # FAISS index.reconstruct(i) for each vector
                n = self.vector_store.index.ntotal
                embeddings = [self.vector_store.index.reconstruct(i).tolist() for i in range(n)]
            else:
                logger.error(f"Cannot extract embeddings from {type(self.vector_store)}")
                embeddings = []

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            embeddings = []

        self._embeddings = embeddings
        logger.info(f"Loaded {len(embeddings)} embeddings from VectorStore")
        return embeddings

    def get_metadata(self) -> Dict[str, Any]:
        """
        VectorStore 메타데이터 수집

        Returns:
            Dict: VectorStore 메타데이터
                - num_documents: 문서 수
                - num_embeddings: 임베딩 수
                - embedding_dim: 임베딩 차원
                - vector_store_type: VectorStore 타입
        """
        if self._metadata is not None:
            return self._metadata

        documents = self.get_documents()
        embeddings = self.get_embeddings()

        embedding_dim = len(embeddings[0]) if embeddings else 0

        metadata = {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "num_documents": len(documents),
            "num_embeddings": len(embeddings),
            "embedding_dim": embedding_dim,
            "vector_store_type": type(self.vector_store).__name__,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }

        self._metadata = metadata
        logger.info(f"Collected metadata: {metadata}")
        return metadata

    def cache_result(self, key: str, value: Any) -> None:
        """
        분석 결과 캐싱

        Args:
            key: 캐시 키
            value: 캐시 값
        """
        self._cache[key] = value
        logger.debug(f"Cached result for key: {key}")

    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        캐시된 결과 가져오기

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None
        """
        return self._cache.get(key)

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache.clear()
        self._documents = None
        self._embeddings = None
        self._metadata = None
        logger.info("Cache cleared")

    def to_dict(self) -> Dict[str, Any]:
        """
        세션 정보를 dict로 변환

        Returns:
            Dict: 세션 정보
        """
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "metadata": self.get_metadata(),
        }
