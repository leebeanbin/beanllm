"""
Chroma Vector Store Implementation

Open-source embedding database
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


class ChromaVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """Chroma vector store - 로컬, 사용하기 쉬움"""

    def __init__(
        self,
        collection_name: str = "beanllm",
        persist_directory: Optional[str] = None,
        embedding_function=None,
        **kwargs,
    ):
        super().__init__(embedding_function)

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Chroma not installed. pip install chromadb")

        # Chroma 클라이언트 설정
        if persist_directory:
            self.client = chromadb.Client(
                Settings(persist_directory=persist_directory, anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client()

        # Collection 생성/가져오기
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if self.embedding_function:
            embeddings = self.embedding_function(texts)
        else:
            embeddings = None

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # Chroma에 추가
        if embeddings:
            self.collection.add(
                documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings
            )
        else:
            self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        # 쿼리 임베딩
        if self.embedding_function:
            query_embedding = self.embedding_function([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=k, **kwargs
            )
        else:
            results = self.collection.query(query_texts=[query], n_results=k, **kwargs)

        # 결과 변환
        search_results = []
        for i in range(len(results["ids"][0])):
            # 런타임에 Document import
            from beanllm.domain.loaders import Document

            doc = Document(content=results["documents"][0][i], metadata=results["metadatas"][0][i])
            score = 1 - results["distances"][0][i]  # Cosine distance -> similarity
            search_results.append(
                VectorSearchResult(document=doc, score=score, metadata=results["metadatas"][0][i])
            )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """Chroma에서 모든 벡터 가져오기"""
        try:
            all_data = self.collection.get()

            vectors = all_data.get("embeddings", [])
            if not vectors:
                return [], []

            documents = []
            texts = all_data.get("documents", [])
            metadatas = all_data.get("metadatas", [{}] * len(texts))

            from beanllm.domain.loaders import Document

            for i, text in enumerate(texts):
                doc = Document(content=text, metadata=metadatas[i] if i < len(metadatas) else {})
                documents.append(doc)

            return vectors, documents
        except Exception:
            # 에러 발생 시 빈 리스트 반환
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        results = self.collection.query(query_embeddings=[query_vec], n_results=k, **kwargs)

        search_results = []
        for i in range(len(results["ids"][0])):
            from beanllm.domain.loaders import Document

            doc = Document(content=results["documents"][0][i], metadata=results["metadatas"][0][i])
            score = 1 - results["distances"][0][i]  # Cosine distance -> similarity
            search_results.append(
                VectorSearchResult(document=doc, score=score, metadata=results["metadatas"][0][i])
            )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.collection.delete(ids=ids)
        return True


