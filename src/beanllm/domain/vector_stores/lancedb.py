"""
LanceDB Vector Store Implementation

Fast, embedded vector database
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
            from beanllm.domain.loaders import Document

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
            from beanllm.domain.loaders import Document

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
            from beanllm.domain.loaders import Document

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


