"""
Vector Store Implementations - 벡터 스토어 구현체들
"""

import os
import uuid
from typing import TYPE_CHECKING, Any, List, Optional

# 순환 참조 방지를 위해 TYPE_CHECKING 사용
if TYPE_CHECKING:
    from ...domain.loaders import Document
else:
    # 런타임에만 import
    try:
        from ...domain.loaders import Document
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
            from ...domain.loaders import Document

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

            from ...domain.loaders import Document

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
            from ...domain.loaders import Document

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
            from ...domain.loaders import Document

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

            from ...domain.loaders import Document

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=float(match.score), metadata=metadata)
            )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.index.delete(ids=ids)
        return True


class FAISSVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """FAISS vector store - 로컬, 매우 빠름"""

    def __init__(
        self,
        embedding_function=None,
        dimension: int = 1536,
        index_type: str = "IndexFlatL2",
        **kwargs,
    ):
        super().__init__(embedding_function)

        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("FAISS not installed. pip install faiss-cpu  # or faiss-gpu")

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

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        texts = [doc.content for doc in documents]

        # 임베딩 생성
        if not self.embedding_function:
            raise ValueError("Embedding function required for FAISS")

        embeddings = self.embedding_function(texts)

        # numpy array로 변환
        embeddings_array = self.np.array(embeddings).astype("float32")

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # 인덱스에 추가
        start_idx = len(self.documents)
        self.index.add(embeddings_array)

        # 문서 및 매핑 저장
        for i, (doc, id_) in enumerate(zip(documents, ids)):
            self.documents.append(doc)
            self.ids_to_index[id_] = start_idx + i

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for FAISS")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]
        query_array = self.np.array([query_embedding]).astype("float32")

        # 검색
        distances, indices = self.index.search(query_array, k)

        # 결과 변환
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                # L2 distance -> similarity score
                score = 1 / (1 + distances[0][i])
                search_results.append(
                    VectorSearchResult(document=doc, score=score, metadata=doc.metadata)
                )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """FAISS에서 모든 벡터 가져오기"""
        if not self.documents:
            return [], []

        # FAISS 인덱스에서 모든 벡터 가져오기
        try:
            # FAISS는 직접 벡터를 가져올 수 없으므로 문서에서 재임베딩
            # 또는 인덱스를 재구축해야 함
            # 여기서는 간단히 빈 리스트 반환 (배치 검색은 비효율적)
            # 실제로는 인덱스에 벡터를 저장해야 함
            return [], []
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        query_array = self.np.array([query_vec]).astype("float32")
        distances, indices = self.index.search(query_array, k)

        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                score = 1 / (1 + distances[0][i])
                search_results.append(
                    VectorSearchResult(document=doc, score=score, metadata=doc.metadata)
                )
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
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "ids_to_index": self.ids_to_index}, f)

    def load(self, path: str):
        """인덱스 로드"""
        import pickle

        # FAISS 인덱스 로드
        self.index = self.faiss.read_index(f"{path}.index")

        # 문서 및 매핑 로드
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.ids_to_index = data["ids_to_index"]


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
            from ...domain.loaders import Document

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
            from ...domain.loaders import Document

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
            from ...domain.loaders import Document

            doc = Document(content=text, metadata=payload)
            search_results.append(
                VectorSearchResult(document=doc, score=result.score, metadata=payload)
            )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        self.client.delete(collection_name=self.collection_name, points_selector=ids)
        return True


class WeaviateVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """Weaviate vector store - 엔터프라이즈급"""

    def __init__(
        self,
        class_name: str = "LlmkitDocument",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_function=None,
        **kwargs,
    ):
        super().__init__(embedding_function)

        try:
            import weaviate
        except ImportError:
            raise ImportError("Weaviate not installed. pip install weaviate-client")

        # 클라이언트 설정
        url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        api_key = api_key or os.getenv("WEAVIATE_API_KEY")

        if api_key:
            self.client = weaviate.Client(
                url=url, auth_client_secret=weaviate.AuthApiKey(api_key=api_key)
            )
        else:
            self.client = weaviate.Client(url=url)

        self.class_name = class_name

        # 스키마 생성
        schema = {
            "class": class_name,
            "vectorizer": "none",  # 우리가 직접 벡터 제공
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["object"]},
            ],
        }

        # 클래스 존재 확인 및 생성
        if not self.client.schema.exists(class_name):
            self.client.schema.create_class(schema)

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
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
                properties = {"text": text, "metadata": metadata}

                uuid = batch.add_data_object(
                    data_object=properties, class_name=self.class_name, vector=embedding
                )
                ids.append(str(uuid))

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Weaviate")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = (
            self.client.query.get(self.class_name, ["text", "metadata"])
            .with_near_vector({"vector": query_embedding})
            .with_limit(k)
            .with_additional(["distance"])
            .do()
        )

        # 결과 변환
        search_results = []
        if results.get("data", {}).get("Get", {}).get(self.class_name):
            for result in results["data"]["Get"][self.class_name]:
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                distance = result.get("_additional", {}).get("distance", 1.0)

                # Distance -> similarity score
                score = 1 / (1 + distance)

                # 런타임에 Document import
                from ...domain.loaders import Document

                doc = Document(content=text, metadata=metadata)
                search_results.append(
                    VectorSearchResult(document=doc, score=score, metadata=metadata)
                )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """Weaviate에서 모든 벡터 가져오기"""
        try:
            # Weaviate에서 모든 객체 가져오기
            results = (
                self.client.query.get(self.class_name, ["text", "metadata"])
                .with_additional(["vector"])
                .with_limit(10000)  # 최대 10000개
                .do()
            )

            vectors = []
            documents = []
            from ...domain.loaders import Document

            for obj in results.get("data", {}).get("Get", {}).get(self.class_name, []):
                vector = obj.get("_additional", {}).get("vector", [])
                if vector:
                    vectors.append(vector)
                    text = obj.get("text", "")
                    metadata = obj.get("metadata", {})
                    doc = Document(content=text, metadata=metadata)
                    documents.append(doc)

            return vectors, documents
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        results = (
            self.client.query.get(self.class_name, ["text", "metadata"])
            .with_near_vector({"vector": query_vec})
            .with_limit(k)
            .with_additional(["certainty", "distance"])
            .do()
        )

        search_results = []
        for obj in results.get("data", {}).get("Get", {}).get(self.class_name, []):
            text = obj.get("text", "")
            metadata = obj.get("metadata", {})
            certainty = obj.get("_additional", {}).get("certainty", 0.0)

            from ...domain.loaders import Document

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=float(certainty), metadata=metadata)
            )
        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        for id_ in ids:
            self.client.data_object.delete(uuid=id_, class_name=self.class_name)
        return True


class MilvusVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """
    Milvus vector store - 오픈소스, 확장 가능, 엔터프라이즈급 (2024-2025)

    Milvus 특징:
    - 오픈소스 벡터 DB (LF AI & Data 재단)
    - GPU 가속 지원
    - 수십억 벡터 규모 지원
    - Zilliz Cloud (관리형 서비스)
    - Hybrid Search (Dense + Sparse)

    Example:
        ```python
        from beanllm.domain.vector_stores import MilvusVectorStore
        from beanllm.domain.embeddings import OpenAIEmbedding

        # 임베딩 모델
        embedding = OpenAIEmbedding(model="text-embedding-3-small")

        # Milvus 벡터 스토어
        vector_store = MilvusVectorStore(
            collection_name="my_docs",
            uri="http://localhost:19530",
            embedding_function=embedding.embed,
            dimension=1536
        )

        # 문서 추가
        from beanllm.domain.loaders import Document
        docs = [Document(content="Hello world", metadata={"source": "test"})]
        vector_store.add_documents(docs)

        # 검색
        results = vector_store.similarity_search("Hello", k=5)
        ```

    References:
        - https://milvus.io/
        - https://github.com/milvus-io/milvus
    """

    def __init__(
        self,
        collection_name: str = "beanllm",
        uri: Optional[str] = None,
        token: Optional[str] = None,
        embedding_function=None,
        dimension: int = 1536,
        metric_type: str = "COSINE",
        **kwargs,
    ):
        """
        Args:
            collection_name: 컬렉션 이름
            uri: Milvus URI (기본: http://localhost:19530)
            token: 인증 토큰 (Zilliz Cloud용)
            embedding_function: 임베딩 함수
            dimension: 벡터 차원
            metric_type: 거리 메트릭 (COSINE, L2, IP)
            **kwargs: 추가 파라미터
        """
        super().__init__(embedding_function)

        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
        except ImportError:
            raise ImportError(
                "pymilvus is required for MilvusVectorStore. "
                "Install it with: pip install pymilvus"
            )

        # Milvus 연결
        uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        token = token or os.getenv("MILVUS_TOKEN")

        # 연결 설정
        if token:
            connections.connect(alias="default", uri=uri, token=token)
        else:
            connections.connect(alias="default", uri=uri)

        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type

        # 스키마 정의
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields, description="beanLLM documents")

        # Collection 생성/가져오기
        try:
            from pymilvus import utility

            if utility.has_collection(collection_name):
                self.collection = Collection(name=collection_name)
            else:
                self.collection = Collection(name=collection_name, schema=schema)

                # 인덱스 생성
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": metric_type,
                    "params": {"nlist": 128},
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)

            # Collection 로드
            self.collection.load()

        except Exception as e:
            raise RuntimeError(f"Failed to create/load Milvus collection: {e}")

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Milvus")

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        embeddings = self.embedding_function(texts)

        # ID 생성
        ids = [str(uuid.uuid4())[:36] for _ in texts]  # Milvus VARCHAR 최대 길이 제한

        # 데이터 준비
        entities = [
            ids,  # id
            texts,  # text
            embeddings,  # embedding
            metadatas,  # metadata (JSON)
        ]

        # Milvus에 추가
        self.collection.insert(entities)
        self.collection.flush()

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for Milvus")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색 파라미터
        search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}

        # 검색
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text", "metadata"],
            **kwargs,
        )

        # 결과 변환
        search_results = []
        for hits in results:
            for hit in hits:
                from ...domain.loaders import Document

                text = hit.entity.get("text")
                metadata = hit.entity.get("metadata", {})
                score = hit.distance

                # COSINE 거리를 유사도로 변환
                if self.metric_type == "COSINE":
                    score = 1 - score

                doc = Document(content=text, metadata=metadata)
                search_results.append(VectorSearchResult(document=doc, score=score, metadata=metadata))

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """Milvus에서 모든 벡터 가져오기"""
        try:
            # 모든 데이터 쿼리
            results = self.collection.query(
                expr="id != ''",  # 모든 문서
                output_fields=["text", "embedding", "metadata"],
                limit=10000,
            )

            vectors = []
            documents = []
            from ...domain.loaders import Document

            for result in results:
                vectors.append(result["embedding"])
                doc = Document(content=result["text"], metadata=result.get("metadata", {}))
                documents.append(doc)

            return vectors, documents
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text", "metadata"],
            **kwargs,
        )

        search_results = []
        for hits in results:
            for hit in hits:
                from ...domain.loaders import Document

                text = hit.entity.get("text")
                metadata = hit.entity.get("metadata", {})
                score = hit.distance

                if self.metric_type == "COSINE":
                    score = 1 - score

                doc = Document(content=text, metadata=metadata)
                search_results.append(VectorSearchResult(document=doc, score=score, metadata=metadata))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        # ID 조건 생성
        id_expr = f"id in {ids}"

        # 삭제
        self.collection.delete(expr=id_expr)
        self.collection.flush()

        return True


class LanceDBVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """
    LanceDB vector store - 오픈소스, 임베디드, 매우 빠름 (2024-2025)

    LanceDB 특징:
    - 오픈소스 임베디드 벡터 DB
    - Serverless (별도 서버 불필요)
    - Lance 컬럼 형식 (빠른 검색, 적은 메모리)
    - Python/JavaScript/Rust 네이티브
    - 디스크 기반 (메모리 효율적)

    Example:
        ```python
        from beanllm.domain.vector_stores import LanceDBVectorStore
        from beanllm.domain.embeddings import OpenAIEmbedding

        # 임베딩 모델
        embedding = OpenAIEmbedding(model="text-embedding-3-small")

        # LanceDB 벡터 스토어
        vector_store = LanceDBVectorStore(
            table_name="my_docs",
            uri="./lancedb",  # 로컬 디렉토리
            embedding_function=embedding.embed
        )

        # 문서 추가
        from beanllm.domain.loaders import Document
        docs = [Document(content="Hello world", metadata={"source": "test"})]
        vector_store.add_documents(docs)

        # 검색
        results = vector_store.similarity_search("Hello", k=5)
        ```

    References:
        - https://lancedb.com/
        - https://github.com/lancedb/lancedb
    """

    def __init__(
        self,
        table_name: str = "beanllm",
        uri: str = "./lancedb",
        embedding_function=None,
        **kwargs,
    ):
        """
        Args:
            table_name: 테이블 이름
            uri: LanceDB URI (로컬 경로 또는 클라우드 URI)
            embedding_function: 임베딩 함수
            **kwargs: 추가 파라미터
        """
        super().__init__(embedding_function)

        try:
            import lancedb
        except ImportError:
            raise ImportError(
                "lancedb is required for LanceDBVectorStore. "
                "Install it with: pip install lancedb"
            )

        # LanceDB 연결
        self.db = lancedb.connect(uri)
        self.table_name = table_name

        # 테이블 생성/가져오기 (첫 문서 추가 시 생성됨)
        try:
            self.table = self.db.open_table(table_name)
        except Exception:
            # 테이블이 없으면 None으로 설정 (첫 add_documents에서 생성)
            self.table = None

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for LanceDB")

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        embeddings = self.embedding_function(texts)

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # 데이터 준비
        data = []
        for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            data.append(
                {
                    "id": id_,
                    "text": text,
                    "vector": embedding,
                    "metadata": metadata,
                }
            )

        # LanceDB에 추가
        if self.table is None:
            # 테이블 생성
            self.table = self.db.create_table(self.table_name, data=data)
        else:
            # 기존 테이블에 추가
            self.table.add(data)

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for LanceDB")

        if self.table is None:
            return []

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색
        results = self.table.search(query_embedding).limit(k).to_list()

        # 결과 변환
        search_results = []
        for result in results:
            from ...domain.loaders import Document

            text = result.get("text", "")
            metadata = result.get("metadata", {})
            score = 1 - result.get("_distance", 0)  # Distance -> similarity

            doc = Document(content=text, metadata=metadata)
            search_results.append(VectorSearchResult(document=doc, score=score, metadata=metadata))

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """LanceDB에서 모든 벡터 가져오기"""
        if self.table is None:
            return [], []

        try:
            # 모든 데이터 가져오기
            all_data = self.table.to_pandas()

            vectors = all_data["vector"].tolist()
            documents = []
            from ...domain.loaders import Document

            for _, row in all_data.iterrows():
                doc = Document(content=row["text"], metadata=row.get("metadata", {}))
                documents.append(doc)

            return vectors, documents
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        if self.table is None:
            return []

        results = self.table.search(query_vec).limit(k).to_list()

        search_results = []
        for result in results:
            from ...domain.loaders import Document

            text = result.get("text", "")
            metadata = result.get("metadata", {})
            score = 1 - result.get("_distance", 0)

            doc = Document(content=text, metadata=metadata)
            search_results.append(VectorSearchResult(document=doc, score=score, metadata=metadata))

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        if self.table is None:
            return False

        # LanceDB delete (id로 필터링)
        for id_ in ids:
            self.table.delete(f"id = '{id_}'")

        return True


class PgvectorVectorStore(BaseVectorStore, AdvancedSearchMixin):
    """
    pgvector vector store - PostgreSQL 확장, 신뢰성 높음 (2024-2025)

    pgvector 특징:
    - PostgreSQL 벡터 확장
    - ACID 트랜잭션 지원
    - SQL 쿼리와 벡터 검색 결합 가능
    - 엔터프라이즈급 안정성
    - Supabase, Neon 등에서 지원

    Example:
        ```python
        from beanllm.domain.vector_stores import PgvectorVectorStore
        from beanllm.domain.embeddings import OpenAIEmbedding

        # 임베딩 모델
        embedding = OpenAIEmbedding(model="text-embedding-3-small")

        # pgvector 벡터 스토어
        vector_store = PgvectorVectorStore(
            connection_string="postgresql://user:pass@localhost:5432/mydb",
            table_name="documents",
            embedding_function=embedding.embed,
            dimension=1536
        )

        # 문서 추가
        from beanllm.domain.loaders import Document
        docs = [Document(content="Hello world", metadata={"source": "test"})]
        vector_store.add_documents(docs)

        # 검색
        results = vector_store.similarity_search("Hello", k=5)
        ```

    References:
        - https://github.com/pgvector/pgvector
        - https://supabase.com/docs/guides/ai/vector-columns
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "beanllm_documents",
        embedding_function=None,
        dimension: int = 1536,
        **kwargs,
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            table_name: 테이블 이름
            embedding_function: 임베딩 함수
            dimension: 벡터 차원
            **kwargs: 추가 파라미터
        """
        super().__init__(embedding_function)

        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
        except ImportError:
            raise ImportError(
                "psycopg2 and pgvector are required for PgvectorVectorStore. "
                "Install with: pip install psycopg2-binary pgvector"
            )

        # 연결 문자열
        connection_string = connection_string or os.getenv(
            "PGVECTOR_CONNECTION_STRING",
            "postgresql://postgres:postgres@localhost:5432/postgres",
        )

        # PostgreSQL 연결
        self.conn = psycopg2.connect(connection_string)
        self.table_name = table_name
        self.dimension = dimension

        # pgvector 등록
        register_vector(self.conn)

        # 테이블 생성
        with self.conn.cursor() as cur:
            # pgvector 확장 활성화
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # 테이블 생성
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(100) PRIMARY KEY,
                    text TEXT,
                    embedding vector({dimension}),
                    metadata JSONB
                )
            """
            )

            # 인덱스 생성 (IVFFlat)
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """
            )

            self.conn.commit()

    def add_documents(self, documents: List[Any], **kwargs) -> List[str]:
        """문서 추가"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for pgvector")

        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 임베딩 생성
        embeddings = self.embedding_function(texts)

        # ID 생성
        ids = [str(uuid.uuid4()) for _ in texts]

        # 데이터 삽입
        import json

        with self.conn.cursor() as cur:
            for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                """,
                    (id_, text, embedding, json.dumps(metadata)),
                )

            self.conn.commit()

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for pgvector")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색 (코사인 유사도)
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, text, embedding, metadata,
                       1 - (embedding <=> %s) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s
            """,
                (query_embedding, query_embedding, k),
            )

            results = cur.fetchall()

        # 결과 변환
        search_results = []
        for row in results:
            from ...domain.loaders import Document

            id_, text, embedding, metadata, similarity = row

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=similarity, metadata=metadata)
            )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """pgvector에서 모든 벡터 가져오기"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT text, embedding, metadata FROM {self.table_name}")
                results = cur.fetchall()

            vectors = []
            documents = []
            from ...domain.loaders import Document

            for row in results:
                text, embedding, metadata = row
                vectors.append(embedding)
                doc = Document(content=text, metadata=metadata)
                documents.append(doc)

            return vectors, documents
        except Exception:
            return [], []

    async def asimilarity_search_by_vector(
        self, query_vec: List[float], k: int = 4, **kwargs
    ) -> List[VectorSearchResult]:
        """벡터로 직접 검색"""
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, text, embedding, metadata,
                       1 - (embedding <=> %s) as similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s
                LIMIT %s
            """,
                (query_vec, query_vec, k),
            )

            results = cur.fetchall()

        search_results = []
        for row in results:
            from ...domain.loaders import Document

            id_, text, embedding, metadata, similarity = row

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=similarity, metadata=metadata)
            )

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제"""
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (ids,))
            self.conn.commit()

        return True

    def __del__(self):
        """연결 종료"""
        if hasattr(self, "conn"):
            self.conn.close()
