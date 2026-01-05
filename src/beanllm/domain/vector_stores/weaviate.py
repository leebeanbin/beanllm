"""
Weaviate Vector Store Implementation

Cloud-native vector search engine
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
                from beanllm.domain.loaders import Document

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
            from beanllm.domain.loaders import Document

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

            from beanllm.domain.loaders import Document

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


