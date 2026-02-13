"""
Chroma Vector Store Implementation

Open-source embedding database
"""

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, cast

if TYPE_CHECKING:
    from beanllm.domain.loaders import Document
else:
    try:
        from beanllm.domain.loaders import Document
    except ImportError:
        Document = Any  # type: ignore

from beanllm.domain.vector_stores.base import BaseVectorStore, VectorSearchResult
from beanllm.domain.vector_stores.search import AdvancedSearchMixin

# Chroma의 QueryResult 타입 (Optional 필드들)
ChromaQueryResult = Dict[str, Any]


class ChromaVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """Chroma vector store - 로컬, 사용하기 쉬움"""

    def __init__(
        self,
        collection_name: str = "beanllm",
        persist_directory: Optional[str] = None,
        embedding_function=None,
        **kwargs,
    ):
        super().__init__(embedding_function=embedding_function, **kwargs)

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Chroma not installed. pip install chromadb")

        # Chroma 클라이언트 설정 (chromadb >= 0.4.0 API)
        if persist_directory:
            # PersistentClient for persistent storage
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # EphemeralClient for in-memory (each call creates new client)
            self.client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

        # Collection 생성/가져오기
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가 (분산 락 적용)"""
        # 분산 락이 제공되지 않은 경우 락 없이 실행
        if self._lock_manager is None:
            return self._add_documents_without_lock(documents, **kwargs)

        # 분산 락 사용
        import asyncio

        store_id = f"{self.collection_name}:{id(self)}"

        async def _add_documents_async():
            async with await self._lock_manager.with_vector_store_lock(store_id, timeout=60.0):
                # 이벤트 발행 (시작)
                self._publish_add_documents_event(len(documents), "add_documents.started")

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

                # 이벤트 발행 (완료)
                self._publish_add_documents_event(len(documents), "add_documents.completed")

                return ids

        # 동기 함수이므로 비동기 래퍼 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 락 없이 실행 (fallback)
                return self._add_documents_without_lock(documents, **kwargs)
            else:
                return loop.run_until_complete(_add_documents_async())
        except RuntimeError:
            return asyncio.run(_add_documents_async())

    def _add_documents_without_lock(self, documents: List[Any], **kwargs) -> List[str]:
        """락 없이 문서 추가 (fallback)"""
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
        results = self._execute_query(query, k, **kwargs)
        return self._parse_query_results(results)

    def _execute_query(self, query: str, k: int, **kwargs: Any) -> ChromaQueryResult:
        """쿼리 실행 (임베딩 함수 유무에 따라 분기)"""
        if self.embedding_function:
            query_embedding = self.embedding_function([query])[0]
            return cast(
                ChromaQueryResult,
                self.collection.query(query_embeddings=[query_embedding], n_results=k, **kwargs),
            )
        return cast(
            ChromaQueryResult,
            self.collection.query(query_texts=[query], n_results=k, **kwargs),
        )

    def _parse_query_results(self, results: ChromaQueryResult) -> List[VectorSearchResult]:
        """Chroma 쿼리 결과를 VectorSearchResult 리스트로 변환 (타입 안전)"""
        from beanllm.domain.loaders import Document

        search_results: List[VectorSearchResult] = []

        # Optional 필드들 안전하게 추출
        ids_list = results.get("ids")
        documents_list = results.get("documents")
        metadatas_list = results.get("metadatas")
        distances_list = results.get("distances")

        # 결과가 비어있으면 빈 리스트 반환
        if not ids_list or not ids_list[0]:
            return search_results

        ids = ids_list[0]
        documents = documents_list[0] if documents_list else [""] * len(ids)
        metadatas = metadatas_list[0] if metadatas_list else [{}] * len(ids)
        distances = distances_list[0] if distances_list else [0.0] * len(ids)

        for i in range(len(ids)):
            metadata_dict = self._mapping_to_dict(metadatas[i]) if i < len(metadatas) else {}
            content = documents[i] if i < len(documents) else ""
            distance = distances[i] if i < len(distances) else 0.0

            doc = Document(content=content, metadata=metadata_dict)
            score = 1 - float(distance)  # Cosine distance -> similarity

            search_results.append(
                VectorSearchResult(document=doc, score=score, metadata=metadata_dict)
            )

        return search_results

    def _mapping_to_dict(self, mapping: Any) -> Dict[str, Any]:
        """Mapping 타입을 dict로 변환"""
        if mapping is None:
            return {}
        if isinstance(mapping, dict):
            return mapping
        # Mapping[str, Any] -> dict[str, Any]
        return dict(mapping)

    def _get_all_vectors_and_docs(self) -> Tuple[List[List[float]], List[Any]]:
        """Chroma에서 모든 벡터 가져오기"""
        try:
            all_data = self.collection.get()

            embeddings = all_data.get("embeddings")
            if not embeddings:
                return [], []

            # 벡터를 List[List[float]]로 변환
            import numpy as np

            vectors: List[List[float]] = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    vectors.append(emb.tolist())
                else:
                    vectors.append([float(x) for x in emb])

            documents: List[Any] = []
            texts = all_data.get("documents") or []
            metadatas = all_data.get("metadatas") or []

            from beanllm.domain.loaders import Document

            for i, text in enumerate(texts):
                metadata = self._mapping_to_dict(metadatas[i]) if i < len(metadatas) else {}
                doc = Document(content=text, metadata=metadata)
                documents.append(doc)

            return vectors, documents
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs: Any
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        # Sequence[float]로 변환 (Chroma API 호환)
        query_sequence: Sequence[float] = query_vec
        results = cast(
            ChromaQueryResult,
            self.collection.query(query_embeddings=[query_sequence], n_results=k, **kwargs),
        )
        return self._parse_query_results(results)

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.collection.delete(ids=ids)
        return True
