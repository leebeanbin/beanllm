"""
Qdrant Vector Store Implementation

High-performance vector search engine
"""

import os
import uuid
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from beanllm.domain.loaders import Document
else:
    try:
        from beanllm.domain.loaders import Document
    except ImportError:
        Document = Any  # type: ignore

from .base import BaseVectorStore, VectorSearchResult
from .search import AdvancedSearchMixin


class QdrantVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """Qdrant vector store - 클라우드/로컬, 모던"""

    def __init__(
        self,
        collection_name: str = "beanllm",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_function=None,
        dimension: int = 1536,
        **kwargs,
    ):
        super().__init__(embedding_function)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams
        except ImportError:
            raise ImportError("Qdrant not installed. pip install qdrant-client")

        self.PointStruct = PointStruct

        # 클라이언트 설정
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = api_key or os.getenv("QDRANT_API_KEY")

        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)

        # Collection 생성/가져오기
        self.collection_name = collection_name

        # Collection 존재 확인
        try:
            self.client.get_collection(collection_name)
        except Exception:
            # Collection 생성
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

        self.dimension = dimension

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for Qdrant")

        embeddings = self.embedding_function(texts)

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # Qdrant에 추가
        points = []
        for i, (id_, embedding, text, metadata) in enumerate(
            zip(ids, embeddings, texts, metadatas)
        ):
            payload = {**metadata, "text": text}
            points.append(self.PointStruct(id=id_, vector=embedding, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Qdrant")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=k, **kwargs
        )

        # 결과 변환
        search_results = []
        for result in results:
            payload = result.payload
            text = payload.pop("text", "")

            # 런타임에 Document import
            from beanllm.domain.loaders import Document

            doc = Document(content=text, metadata=payload)
            search_results.append(
                VectorSearchResult(document=doc, score=result.score, metadata=payload)
            )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """Qdrant에서 모든 벡터 가져오기"""
        try:
            # Qdrant에서 모든 포인트 가져오기
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # 최대 10000개
            )

            vectors = []
            documents = []
            from beanllm.domain.loaders import Document

            for point in points[0]:  # points는 (points, next_offset) 튜플
                vectors.append(point.vector)
                payload = point.payload
                text = payload.pop("text", "")
                doc = Document(content=text, metadata=payload)
                documents.append(doc)

            return vectors, documents
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vec, limit=k, **kwargs
        )

        search_results = []
        for result in results:
            payload = result.payload
            text = payload.pop("text", "")
            from beanllm.domain.loaders import Document

            doc = Document(content=text, metadata=payload)
            search_results.append(
                VectorSearchResult(document=doc, score=result.score, metadata=payload)
            )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.client.delete(collection_name=self.collection_name, points_selector=ids)
        return True


