"""
Vector Stores - Unified interface for vector databases
llmkit 방식: Client와 같은 패턴, Fluent API
"""
import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

from .document_loaders import Document


@dataclass
class VectorSearchResult:
    """벡터 검색 결과"""
    document: Document
    score: float
    metadata: Dict[str, Any]


class BaseVectorStore(ABC):
    """Base class for all vector stores"""

    def __init__(self, embedding_function=None, **kwargs):
        """
        Args:
            embedding_function: 임베딩 함수 (texts -> vectors)
        """
        self.embedding_function = embedding_function

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """문서 추가"""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """유사도 검색"""
        pass

    @abstractmethod
    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        pass

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        **kwargs
    ) -> List[str]:
        """텍스트 직접 추가"""
        documents = [
            Document(
                content=text,
                metadata=metadatas[i] if metadatas else {}
            )
            for i, text in enumerate(texts)
        ]
        return self.add_documents(documents, **kwargs)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """비동기 유사도 검색"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.similarity_search(query, k, **kwargs)
        )

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        Hybrid Search (벡터 + 키워드 검색)

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            alpha: 벡터 검색 가중치 (0.0 ~ 1.0)
                   0.0 = 키워드만, 1.0 = 벡터만, 0.5 = 균형
            **kwargs: 추가 파라미터

        Returns:
            검색 결과 리스트

        Example:
            # 벡터와 키워드를 균형있게
            results = store.hybrid_search("machine learning", k=5, alpha=0.5)

            # 벡터 중심
            results = store.hybrid_search("query", k=5, alpha=0.8)
        """
        # 1. 벡터 검색
        vector_results = self.similarity_search(query, k=k * 2, **kwargs)

        # 2. 키워드 검색 (간단한 BM25 스타일)
        keyword_results = self._keyword_search(query, k=k * 2)

        # 3. 점수 결합 (RRF - Reciprocal Rank Fusion)
        combined = self._combine_results(
            vector_results,
            keyword_results,
            alpha=alpha
        )

        return combined[:k]

    def _keyword_search(
        self,
        query: str,
        k: int = 10
    ) -> List[VectorSearchResult]:
        """
        키워드 기반 검색 (BM25 스타일)

        Note: 기본 구현은 단순 포함 여부.
        provider별로 override하여 더 나은 구현 가능.
        """
        # 모든 문서에서 키워드 검색
        # 기본 구현: 단순히 쿼리 단어가 포함된 문서 찾기
        query_terms = query.lower().split()

        # 문서를 가져올 방법이 없으므로 빈 리스트 반환
        # 각 provider에서 override하여 구현해야 함
        return []

    def _combine_results(
        self,
        vector_results: List[VectorSearchResult],
        keyword_results: List[VectorSearchResult],
        alpha: float = 0.5
    ) -> List[VectorSearchResult]:
        """
        벡터와 키워드 결과 결합 (RRF)

        Args:
            vector_results: 벡터 검색 결과
            keyword_results: 키워드 검색 결과
            alpha: 벡터 검색 가중치

        Returns:
            결합된 결과
        """
        # 문서 ID -> (결과, 벡터 순위, 키워드 순위)
        results_map: Dict[str, Tuple[VectorSearchResult, Optional[int], Optional[int]]] = {}

        # 벡터 검색 결과
        for rank, result in enumerate(vector_results, 1):
            doc_id = id(result.document)  # 문서 ID
            results_map[doc_id] = (result, rank, None)

        # 키워드 검색 결과
        for rank, result in enumerate(keyword_results, 1):
            doc_id = id(result.document)
            if doc_id in results_map:
                prev_result, vec_rank, _ = results_map[doc_id]
                results_map[doc_id] = (prev_result, vec_rank, rank)
            else:
                results_map[doc_id] = (result, None, rank)

        # RRF 점수 계산
        k_constant = 60  # RRF constant
        scored_results = []

        for doc_id, (result, vec_rank, key_rank) in results_map.items():
            vec_score = alpha / (k_constant + vec_rank) if vec_rank else 0
            key_score = (1 - alpha) / (k_constant + key_rank) if key_rank else 0
            total_score = vec_score + key_score

            # 새로운 점수로 결과 생성
            scored_results.append(VectorSearchResult(
                document=result.document,
                score=total_score,
                metadata=result.metadata
            ))

        # 점수로 정렬
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results

    def rerank(
        self,
        query: str,
        results: List[VectorSearchResult],
        model: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[VectorSearchResult]:
        """
        Re-ranking with Cross-encoder

        Args:
            query: 쿼리
            results: 초기 검색 결과
            model: Cross-encoder 모델 (기본: "cross-encoder/ms-marco-MiniLM-L-6-v2")
            top_k: 재순위화 후 반환할 개수

        Returns:
            재순위화된 결과

        Example:
            # 초기 검색
            results = store.similarity_search("query", k=20)

            # 재순위화 (상위 5개만)
            reranked = store.rerank("query", results, top_k=5)
        """
        if not results:
            return []

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers 필요:\n"
                "pip install sentence-transformers"
            )

        # 모델 로드
        model_name = model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cross_encoder = CrossEncoder(model_name)

        # (query, document) 쌍 생성
        pairs = [[query, result.document.content] for result in results]

        # Cross-encoder로 점수 계산
        scores = cross_encoder.predict(pairs)

        # 점수로 재정렬
        reranked_results = []
        for result, score in zip(results, scores):
            reranked_results.append(VectorSearchResult(
                document=result.document,
                score=float(score),
                metadata=result.metadata
            ))

        reranked_results.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            return reranked_results[:top_k]
        return reranked_results

    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_param: float = 0.5,
        **kwargs
    ) -> List[VectorSearchResult]:
        """
        MMR (Maximal Marginal Relevance) 검색 - 다양성 고려

        Args:
            query: 검색 쿼리
            k: 최종 반환 개수
            fetch_k: 초기 가져올 개수 (k보다 커야 함)
            lambda_param: 관련성 vs 다양성 (0.0 ~ 1.0)
                         1.0 = 관련성만, 0.0 = 다양성만
            **kwargs: 추가 파라미터

        Returns:
            다양성을 고려한 검색 결과

        Example:
            # 관련성과 다양성 균형
            results = store.mmr_search("AI", k=5, lambda_param=0.5)

            # 다양성 중심
            results = store.mmr_search("AI", k=5, lambda_param=0.3)
        """
        # 초기 검색
        candidates = self.similarity_search(query, k=fetch_k, **kwargs)

        if not candidates or len(candidates) <= k:
            return candidates

        # 쿼리 임베딩
        if not self.embedding_function:
            # 임베딩 함수 없으면 일반 검색 반환
            return candidates[:k]

        query_vec = self.embedding_function([query])[0]

        # 후보 벡터들
        candidate_vecs = [
            self.embedding_function([c.document.content])[0]
            for c in candidates
        ]

        # MMR 알고리즘
        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_score = float('-inf')
            best_idx = None

            for idx in remaining_indices:
                # 관련성 점수 (쿼리와의 유사도)
                relevance = self._cosine_similarity(query_vec, candidate_vecs[idx])

                # 다양성 점수 (이미 선택된 문서들과의 최대 유사도)
                if selected_indices:
                    diversity = max(
                        self._cosine_similarity(
                            candidate_vecs[idx],
                            candidate_vecs[selected_idx]
                        )
                        for selected_idx in selected_indices
                    )
                else:
                    diversity = 0

                # MMR 점수
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [candidates[idx] for idx in selected_indices]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        try:
            import numpy as np
            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except ImportError:
            # numpy 없으면 수동 계산
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store - 로컬, 사용하기 쉬움"""

    def __init__(
        self,
        collection_name: str = "llmkit",
        persist_directory: Optional[str] = None,
        embedding_function=None,
        **kwargs
    ):
        super().__init__(embedding_function)

        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "Chroma not installed. "
                "pip install chromadb"
            )

        # Chroma 클라이언트 설정
        if persist_directory:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
        else:
            self.client = chromadb.Client()

        # Collection 생성/가져오기
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if self.embedding_function:
            embeddings = self.embedding_function(texts)
        else:
            embeddings = None

        # ID 생성
        import uuid
        ids = [str(uuid.uuid4()) for _ in texts]

        # Chroma에 추가
        if embeddings:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
        else:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """유사도 검색"""
        # 쿼리 임베딩
        if self.embedding_function:
            query_embedding = self.embedding_function([query])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                **kwargs
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                **kwargs
            )

        # 결과 변환
        search_results = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            score = 1 - results['distances'][0][i]  # Cosine distance -> similarity
            search_results.append(VectorSearchResult(
                document=doc,
                score=score,
                metadata=results['metadatas'][0][i]
            ))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.collection.delete(ids=ids)
        return True


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store - 클라우드, 확장 가능"""

    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        embedding_function=None,
        dimension: int = 1536,  # OpenAI default
        metric: str = "cosine",
        **kwargs
    ):
        super().__init__(embedding_function)

        try:
            import pinecone
        except ImportError:
            raise ImportError(
                "Pinecone not installed. "
                "pip install pinecone-client"
            )

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
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )

        self.index = pinecone.Index(index_name)
        self.dimension = dimension

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for Pinecone")

        embeddings = self.embedding_function(texts)

        # ID 생성
        import uuid
        ids = [str(uuid.uuid4()) for _ in texts]

        # Pinecone에 추가
        vectors = []
        for i, (id_, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            metadata_with_text = {**metadata, "text": texts[i]}
            vectors.append((id_, embedding, metadata_with_text))

        self.index.upsert(vectors=vectors)

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Pinecone")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            **kwargs
        )

        # 결과 변환
        search_results = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            text = metadata.pop('text', '')

            doc = Document(content=text, metadata=metadata)
            search_results.append(VectorSearchResult(
                document=doc,
                score=match['score'],
                metadata=metadata
            ))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.index.delete(ids=ids)
        return True


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store - 로컬, 매우 빠름"""

    def __init__(
        self,
        embedding_function=None,
        dimension: int = 1536,
        index_type: str = "IndexFlatL2",
        **kwargs
    ):
        super().__init__(embedding_function)

        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "FAISS not installed. "
                "pip install faiss-cpu  # or faiss-gpu"
            )

        self.faiss = faiss
        self.np = np

        # FAISS 인덱스 생성
        if index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.dimension = dimension
        self.documents = []  # 문서 저장
        self.ids_to_index = {}  # ID -> index 매핑

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for FAISS")

        embeddings = self.embedding_function(texts)

        # numpy array로 변환
        embeddings_array = self.np.array(embeddings).astype('float32')

        # ID 생성
        import uuid
        ids = [str(uuid.uuid4()) for _ in texts]

        # 인덱스에 추가
        start_idx = len(self.documents)
        self.index.add(embeddings_array)

        # 문서 및 매핑 저장
        for i, (doc, id_) in enumerate(zip(documents, ids)):
            self.documents.append(doc)
            self.ids_to_index[id_] = start_idx + i

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for FAISS")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]
        query_array = self.np.array([query_embedding]).astype('float32')

        # 검색
        distances, indices = self.index.search(query_array, k)

        # 결과 변환
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # L2 distance -> similarity score
                score = 1 / (1 + distances[0][i])
                search_results.append(VectorSearchResult(
                    document=doc,
                    score=score,
                    metadata=doc.metadata
                ))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제 (FAISS는 삭제 미지원, 재구축 필요)"""
        # FAISS는 직접 삭제를 지원하지 않음
        # 실제로는 삭제할 문서를 제외하고 인덱스 재구축
        raise NotImplementedError(
            "FAISS does not support direct deletion. "
            "Rebuild index without deleted documents instead."
        )

    def save(self, path: str):
        """인덱스 저장"""
        import pickle

        # FAISS 인덱스 저장
        self.faiss.write_index(self.index, f"{path}.index")

        # 문서 및 매핑 저장
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'ids_to_index': self.ids_to_index
            }, f)

    def load(self, path: str):
        """인덱스 로드"""
        import pickle

        # FAISS 인덱스 로드
        self.index = self.faiss.read_index(f"{path}.index")

        # 문서 및 매핑 로드
        with open(f"{path}.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.ids_to_index = data['ids_to_index']


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store - 클라우드/로컬, 모던"""

    def __init__(
        self,
        collection_name: str = "llmkit",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_function=None,
        dimension: int = 1536,
        **kwargs
    ):
        super().__init__(embedding_function)

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise ImportError(
                "Qdrant not installed. "
                "pip install qdrant-client"
            )

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
        except:
            # Collection 생성
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

        self.dimension = dimension

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for Qdrant")

        embeddings = self.embedding_function(texts)

        # ID 생성
        import uuid
        ids = [str(uuid.uuid4()) for _ in texts]

        # Qdrant에 추가
        points = []
        for i, (id_, embedding, text, metadata) in enumerate(zip(ids, embeddings, texts, metadatas)):
            payload = {**metadata, "text": text}
            points.append(self.PointStruct(
                id=id_,
                vector=embedding,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Qdrant")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            **kwargs
        )

        # 결과 변환
        search_results = []
        for result in results:
            payload = result.payload
            text = payload.pop('text', '')

            doc = Document(content=text, metadata=payload)
            search_results.append(VectorSearchResult(
                document=doc,
                score=result.score,
                metadata=payload
            ))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
        return True


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store - 엔터프라이즈급"""

    def __init__(
        self,
        class_name: str = "LlmkitDocument",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_function=None,
        **kwargs
    ):
        super().__init__(embedding_function)

        try:
            import weaviate
        except ImportError:
            raise ImportError(
                "Weaviate not installed. "
                "pip install weaviate-client"
            )

        # 클라이언트 설정
        url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        api_key = api_key or os.getenv("WEAVIATE_API_KEY")

        if api_key:
            self.client = weaviate.Client(
                url=url,
                auth_client_secret=weaviate.AuthApiKey(api_key=api_key)
            )
        else:
            self.client = weaviate.Client(url=url)

        self.class_name = class_name

        # 스키마 생성
        schema = {
            "class": class_name,
            "vectorizer": "none",  # 우리가 직접 벡터 제공
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"]
                },
                {
                    "name": "metadata",
                    "dataType": ["object"]
                }
            ]
        }

        # 클래스 존재 확인 및 생성
        if not self.client.schema.exists(class_name):
            self.client.schema.create_class(schema)

    def add_documents(self, documents: List[Document], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for Weaviate")

        embeddings = self.embedding_function(texts)

        # Weaviate에 추가
        ids = []
        with self.client.batch as batch:
            for text, metadata, embedding in zip(texts, metadatas, embeddings):
                properties = {
                    "text": text,
                    "metadata": metadata
                }

                uuid = batch.add_data_object(
                    data_object=properties,
                    class_name=self.class_name,
                    vector=embedding
                )
                ids.append(str(uuid))

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Weaviate")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = (
            self.client.query
            .get(self.class_name, ["text", "metadata"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(k)
            .with_additional(["distance"])
            .do()
        )

        # 결과 변환
        search_results = []
        if results.get('data', {}).get('Get', {}).get(self.class_name):
            for result in results['data']['Get'][self.class_name]:
                text = result.get('text', '')
                metadata = result.get('metadata', {})
                distance = result.get('_additional', {}).get('distance', 1.0)

                # Distance -> similarity score
                score = 1 / (1 + distance)

                doc = Document(content=text, metadata=metadata)
                search_results.append(VectorSearchResult(
                    document=doc,
                    score=score,
                    metadata=metadata
                ))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        for id_ in ids:
            self.client.data_object.delete(
                uuid=id_,
                class_name=self.class_name
            )
        return True


class VectorStore:
    """
    Unified vector store interface with auto-detection
    Client 패턴과 동일한 방식
    """

    PROVIDERS = {
        "chroma": ChromaVectorStore,
        "pinecone": PineconeVectorStore,
        "faiss": FAISSVectorStore,
        "qdrant": QdrantVectorStore,
        "weaviate": WeaviateVectorStore,
    }

    PROVIDER_ENV_VARS = {
        "chroma": None,  # 로컬, API 키 불필요
        "pinecone": "PINECONE_API_KEY",
        "faiss": None,  # 로컬, API 키 불필요
        "qdrant": None,  # 로컬/클라우드, 선택적
        "weaviate": None,  # 로컬/클라우드, 선택적
    }

    def __new__(cls, provider: Optional[str] = None, **kwargs):
        """
        Factory method to create vector store instance

        Args:
            provider: Provider 이름 (선택적). None이면 자동으로 가장 좋은 provider 선택.

        Examples:
            # 방법 1: 자동 선택 (추천)
            store = VectorStore(embedding_function=embed_func)

            # 방법 2: 명시적 선택
            store = VectorStore(provider="chroma", embedding_function=embed_func)

            # 방법 3: 팩토리 메서드
            store = VectorStore.chroma(embedding_function=embed_func)
        """
        # provider 자동 선택
        if provider is None:
            provider = cls.get_default_provider()

        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(cls.PROVIDERS.keys())}"
            )

        vector_store_class = cls.PROVIDERS[provider]
        return vector_store_class(**kwargs)

    @classmethod
    def chroma(cls, **kwargs) -> ChromaVectorStore:
        """Create Chroma vector store"""
        return ChromaVectorStore(**kwargs)

    @classmethod
    def pinecone(cls, **kwargs) -> PineconeVectorStore:
        """Create Pinecone vector store"""
        return PineconeVectorStore(**kwargs)

    @classmethod
    def faiss(cls, **kwargs) -> FAISSVectorStore:
        """Create FAISS vector store"""
        return FAISSVectorStore(**kwargs)

    @classmethod
    def qdrant(cls, **kwargs) -> QdrantVectorStore:
        """Create Qdrant vector store"""
        return QdrantVectorStore(**kwargs)

    @classmethod
    def weaviate(cls, **kwargs) -> WeaviateVectorStore:
        """Create Weaviate vector store"""
        return WeaviateVectorStore(**kwargs)

    @classmethod
    def list_available_providers(cls) -> List[str]:
        """사용 가능한 provider 목록 반환"""
        available = []

        for provider, env_var in cls.PROVIDER_ENV_VARS.items():
            if env_var is None:
                # 로컬 provider (항상 사용 가능)
                available.append(provider)
            else:
                # API 키 확인
                if os.getenv(env_var):
                    available.append(provider)

        return available

    @classmethod
    def get_default_provider(cls) -> str:
        """기본 provider 반환 (우선순위 기반)"""
        # 우선순위: chroma > faiss > qdrant > pinecone > weaviate
        priority = ["chroma", "faiss", "qdrant", "pinecone", "weaviate"]
        available = cls.list_available_providers()

        for provider in priority:
            if provider in available:
                return provider

        return "chroma"  # 기본값


# Fluent API helper
class VectorStoreBuilder:
    """
    Fluent API for easy vector store creation and usage

    Example:
        store = (VectorStoreBuilder()
            .use_chroma()
            .with_embedding(embed_func)
            .build())
    """

    def __init__(self):
        self.provider = "chroma"
        self.embedding_function = None
        self.kwargs = {}

    def use_chroma(self, **kwargs) -> 'VectorStoreBuilder':
        """Use Chroma"""
        self.provider = "chroma"
        self.kwargs.update(kwargs)
        return self

    def use_pinecone(self, **kwargs) -> 'VectorStoreBuilder':
        """Use Pinecone"""
        self.provider = "pinecone"
        self.kwargs.update(kwargs)
        return self

    def use_faiss(self, **kwargs) -> 'VectorStoreBuilder':
        """Use FAISS"""
        self.provider = "faiss"
        self.kwargs.update(kwargs)
        return self

    def use_qdrant(self, **kwargs) -> 'VectorStoreBuilder':
        """Use Qdrant"""
        self.provider = "qdrant"
        self.kwargs.update(kwargs)
        return self

    def use_weaviate(self, **kwargs) -> 'VectorStoreBuilder':
        """Use Weaviate"""
        self.provider = "weaviate"
        self.kwargs.update(kwargs)
        return self

    def with_embedding(self, embedding_function) -> 'VectorStoreBuilder':
        """Set embedding function"""
        self.embedding_function = embedding_function
        return self

    def with_collection(self, name: str) -> 'VectorStoreBuilder':
        """Set collection/index name"""
        self.kwargs['collection_name'] = name
        return self

    def build(self) -> BaseVectorStore:
        """Build vector store"""
        return VectorStore(
            provider=self.provider,
            embedding_function=self.embedding_function,
            **self.kwargs
        )


# Convenience functions
def create_vector_store(
    provider: Optional[str] = None,
    embedding_function=None,
    **kwargs
) -> BaseVectorStore:
    """
    편리한 vector store 생성 함수

    Args:
        provider: Provider 이름 (선택적). None이면 자동 선택.
        embedding_function: 임베딩 함수
        **kwargs: 추가 파라미터

    Examples:
        # 자동 선택
        store = create_vector_store(embedding_function=embed_func)

        # 명시적 선택
        store = create_vector_store("chroma", embedding_function=embed_func)
    """
    return VectorStore(
        provider=provider,
        embedding_function=embedding_function,
        **kwargs
    )


def from_documents(
    documents: List[Document],
    embedding_function,
    provider: Optional[str] = None,
    **kwargs
) -> BaseVectorStore:
    """
    문서에서 직접 vector store 생성

    Args:
        documents: 문서 리스트
        embedding_function: 임베딩 함수
        provider: Provider 이름 (선택적). None이면 자동 선택.
        **kwargs: 추가 파라미터

    Examples:
        # 자동 선택 (가장 간단!)
        store = from_documents(docs, embed_func)

        # 명시적 선택
        store = from_documents(docs, embed_func, provider="chroma")
    """
    store = create_vector_store(
        provider=provider,
        embedding_function=embedding_function,
        **kwargs
    )
    store.add_documents(documents)
    return store
