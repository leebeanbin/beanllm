"""
Milvus Vector Store Implementation

Open-source vector database
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
                from beanllm.domain.loaders import Document

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
            from beanllm.domain.loaders import Document

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
                from beanllm.domain.loaders import Document

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


