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
