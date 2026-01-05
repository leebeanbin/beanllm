"""
Pgvector Vector Store Implementation

PostgreSQL vector extension
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
        use_pool: bool = True,
        pool_minconn: int = 1,
        pool_maxconn: int = 10,
        **kwargs,
    ):
        """
        Args:
            connection_string: PostgreSQL 연결 문자열
            table_name: 테이블 이름
            embedding_function: 임베딩 함수
            dimension: 벡터 차원
            use_pool: Connection Pool 사용 여부 (기본: True, 성능 향상)
            pool_minconn: Pool 최소 연결 수 (기본: 1)
            pool_maxconn: Pool 최대 연결 수 (기본: 10)
            **kwargs: 추가 파라미터
        """
        super().__init__(embedding_function)

        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector
            from psycopg2 import pool, sql
        except ImportError:
            raise ImportError(
                "psycopg2 and pgvector are required for PgvectorVectorStore. "
                "Install with: pip install psycopg2-binary pgvector"
            )

        self.psycopg2 = psycopg2
        self.register_vector = register_vector

        # 테이블 이름 검증 (SQL Injection 방지)
        self._validate_table_name(table_name)

        # 연결 문자열
        connection_string = connection_string or os.getenv(
            "PGVECTOR_CONNECTION_STRING",
            "postgresql://postgres:postgres@localhost:5432/postgres",
        )

        self.table_name = table_name
        self.dimension = dimension
        self.sql = sql  # SQL builder 모듈 저장
        self.use_pool = use_pool

        # Connection Pool 또는 단일 연결
        if use_pool:
            # Connection Pool 생성 (성능 향상)
            self.pool = pool.ThreadedConnectionPool(
                pool_minconn, pool_maxconn, connection_string
            )
            self.conn = None  # Pool을 사용할 때는 conn을 None으로
        else:
            # 단일 연결
            self.pool = None
            self.conn = psycopg2.connect(connection_string)
            # pgvector 등록
            register_vector(self.conn)

        # 테이블 생성 (Pool 사용 시 임시 연결 획득)
        self._create_table()

    def _validate_table_name(self, table_name: str):
        """
        테이블 이름 검증 (SQL Injection 방지)

        Args:
            table_name: 검증할 테이블 이름

        Raises:
            ValueError: 허용되지 않은 테이블 이름
        """
        import re

        # 테이블 이름은 영문자, 숫자, 언더스코어만 허용
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. "
                f"Table name must contain only alphanumeric characters and underscores, "
                f"and must start with a letter or underscore (SQL Injection protection)."
            )

        # 최대 길이 제한 (PostgreSQL 제한: 63자)
        if len(table_name) > 63:
            raise ValueError(f"Table name too long: {table_name} (max 63 characters)")

    def _get_connection(self):
        """
        Pool에서 연결 가져오기 또는 기존 연결 반환

        Returns:
            psycopg2.connection: 데이터베이스 연결

        Note:
            - Pool 사용 시: Pool에서 새 연결 획득
            - 단일 연결 사용 시: 기존 연결 반환
        """
        if self.use_pool and self.pool:
            conn = self.pool.getconn()
            # pgvector 등록 (연결마다 필요)
            self.register_vector(conn)
            return conn
        else:
            return self.conn

    def _return_connection(self, conn):
        """
        Pool에 연결 반환

        Args:
            conn: 반환할 연결

        Note:
            - Pool 사용 시: Pool에 연결 반환
            - 단일 연결 사용 시: 아무것도 하지 않음 (연결 유지)
        """
        if self.use_pool and self.pool and conn:
            self.pool.putconn(conn)

    def _create_table(self):
        """테이블 및 인덱스 생성 (SQL Injection 방지)"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # pgvector 확장 활성화
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # 테이블 생성 (parameterized with sql.Identifier)
                create_table_query = self.sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id VARCHAR(100) PRIMARY KEY,
                        text TEXT,
                        embedding vector({}),
                        metadata JSONB
                    )
                """).format(
                    self.sql.Identifier(self.table_name),
                    self.sql.Literal(self.dimension),
                )
                cur.execute(create_table_query)

                # 인덱스 생성 (IVFFlat)
                index_name = f"{self.table_name}_embedding_idx"
                create_index_query = self.sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {}
                    ON {} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """).format(
                    self.sql.Identifier(index_name), self.sql.Identifier(self.table_name)
                )
                cur.execute(create_index_query)

                conn.commit()
        finally:
            self._return_connection(conn)

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

        # 데이터 삽입 (SQL Injection 방지, Batch Insert)
        import json

        insert_query = self.sql.SQL("""
            INSERT INTO {} (id, text, embedding, metadata)
            VALUES (%s, %s, %s, %s)
        """).format(self.sql.Identifier(self.table_name))

        # executemany로 배치 삽입 (성능 향상)
        data_batch = [
            (id_, text, embedding, json.dumps(metadata))
            for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas)
        ]

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.executemany(insert_query, data_batch)
                conn.commit()
        finally:
            self._return_connection(conn)

        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[VectorSearchResult]:
        """유사도 검색"""
        if not self.embedding_function:
            raise ValueError("Embedding function required for pgvector")

        # 쿼리 임베딩
        query_embedding = self.embedding_function([query])[0]

        # 검색 (코사인 유사도, SQL Injection 방지)
        search_query = self.sql.SQL("""
            SELECT id, text, embedding, metadata,
                   1 - (embedding <=> %s) as similarity
            FROM {}
            ORDER BY embedding <=> %s
            LIMIT %s
        """).format(self.sql.Identifier(self.table_name))

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(search_query, (query_embedding, query_embedding, k))
                results = cur.fetchall()
        finally:
            self._return_connection(conn)

        # 결과 변환
        search_results = []
        for row in results:
            from beanllm.domain.loaders import Document

            id_, text, embedding, metadata, similarity = row

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=similarity, metadata=metadata)
            )

        return search_results

    def _get_all_vectors_and_docs(self) -> tuple[List[List[float]], List[Any]]:
        """pgvector에서 모든 벡터 가져오기 (SQL Injection 방지)"""
        try:
            select_query = self.sql.SQL("SELECT text, embedding, metadata FROM {}").format(
                self.sql.Identifier(self.table_name)
            )

            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(select_query)
                    results = cur.fetchall()
            finally:
                self._return_connection(conn)

            vectors = []
            documents = []
            from beanllm.domain.loaders import Document

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
        """벡터로 직접 검색 (SQL Injection 방지)"""
        search_query = self.sql.SQL("""
            SELECT id, text, embedding, metadata,
                   1 - (embedding <=> %s) as similarity
            FROM {}
            ORDER BY embedding <=> %s
            LIMIT %s
        """).format(self.sql.Identifier(self.table_name))

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(search_query, (query_vec, query_vec, k))
                results = cur.fetchall()
        finally:
            self._return_connection(conn)

        search_results = []
        for row in results:
            from beanllm.domain.loaders import Document

            id_, text, embedding, metadata, similarity = row

            doc = Document(content=text, metadata=metadata)
            search_results.append(
                VectorSearchResult(document=doc, score=similarity, metadata=metadata)
            )

        return search_results

    def delete(self, ids: List[str], **kwargs) -> bool:
        """문서 삭제 (SQL Injection 방지)"""
        delete_query = self.sql.SQL("DELETE FROM {} WHERE id = ANY(%s)").format(
            self.sql.Identifier(self.table_name)
        )

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(delete_query, (ids,))
                conn.commit()
        finally:
            self._return_connection(conn)

        return True

    def close(self):
        """
        연결 및 Pool 정리 (리소스 해제)

        명시적으로 리소스를 정리합니다.
        Connection Pool 사용 시 모든 연결을 닫습니다.
        """
        if self.use_pool and hasattr(self, "pool") and self.pool:
            # Pool의 모든 연결 닫기
            self.pool.closeall()
            self.pool = None
        elif hasattr(self, "conn") and self.conn:
            # 단일 연결 닫기
            self.conn.close()
            self.conn = None

    def __del__(self):
        """
        소멸자 - 리소스 자동 정리

        객체가 삭제될 때 자동으로 연결 및 Pool을 정리합니다.
        """
        try:
            self.close()
        except Exception:
            pass  # 소멸자에서는 예외를 무시
