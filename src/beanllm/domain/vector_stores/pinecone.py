"""
Pinecone Vector Store Implementation

Managed vector database service
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


class PineconeVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """Pinecone vector store - 클라우드, 확장 가능"""

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        embedding_function=None,
        dimension: int = 1536,  # OpenAI default
        metric: str = "cosine",
        **kwargs,
    ):
        super().__init__(embedding_function)

        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. pip install pinecone-client")

        # API 키 설정
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

        if not api_key:
            raise ValueError("Pinecone API key not found")

        # Pinecone 초기화
        pinecone.init(api_key=api_key, environment=environment)

        # 인덱스 생성/가져오기
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension, metric=metric)

        self.index = pinecone.Index(index_name)
        self.dimension = dimension

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for Pinecone")

        embeddings = self.embedding_function(texts)

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # Pinecone에 추가
        vectors = []
        for i, (id_, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            metadata_with_text = {**metadata, "text": texts[i]}
            vectors.append((id_, embedding, metadata_with_text))

        self.index.upsert(vectors=vectors)

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Pinecone")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True, **kwargs)

        # 결과 변환
        search_results = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")

            # 런타임에 Document import
            from beanllm.domain.loaders import Document

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=match["score"], metadata=metadata)
            )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """Pinecone에서 모든 벡터 가져오기 (제한적)"""
        try:
            # Pinecone은 모든 벡터를 가져오는 API가 제한적
            # fetch()를 사용하거나 query()로 일부만 가져올 수 있음
            # 여기서는 빈 리스트 반환 (배치 검색은 Pinecone API를 직접 사용 권장)
            return [], []
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        results = self.index.query(vector=query_vec, top_k=k, include_metadata=True, **kwargs)

        search_results = []
        for match in results.matches:
            text = match.metadata.get("text", "")
            metadata = {k: v for k, v in match.metadata.items() if k != "text"}

            from beanllm.domain.loaders import Document

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=float(match.score), metadata=metadata)
            )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.index.delete(ids=ids)
        return True


