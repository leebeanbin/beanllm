"""
PostgreSQL Document Loader

PostgreSQL 데이터베이스에서 문서를 로드합니다.
asyncpg (비동기) 또는 psycopg2 (동기)를 지원합니다.

Requirements:
    pip install psycopg2-binary  # 또는 asyncpg

Example:
    ```python
    from beanllm.domain.loaders.database import PostgreSQLLoader

    # 기본 사용
    loader = PostgreSQLLoader(
        connection_string="postgresql://user:pass@localhost:5432/mydb",
        query="SELECT id, title, content FROM articles"
    )
    docs = loader.load()

    # 특정 컬럼을 content로 사용
    loader = PostgreSQLLoader(
        connection_string="postgresql://user:pass@localhost:5432/mydb",
        query="SELECT * FROM documents WHERE status = 'published'",
        content_columns=["title", "body"],  # 이 컬럼들을 합쳐서 content로
        metadata_columns=["author", "created_at", "tags"]  # metadata에 포함
    )
    ```
"""

import logging
from typing import Any, Dict, Generator, List, Optional

from beanllm.domain.loaders.base import BaseDocumentLoader
from beanllm.domain.loaders.types import Document

logger = logging.getLogger(__name__)


class PostgreSQLLoader(BaseDocumentLoader):
    """
    PostgreSQL 데이터베이스에서 문서를 로드하는 로더

    Attributes:
        connection_string: PostgreSQL 연결 문자열
        query: 실행할 SQL 쿼리
        content_columns: content로 사용할 컬럼 목록 (기본: 모든 텍스트 컬럼)
        metadata_columns: metadata에 포함할 컬럼 목록 (기본: 모든 컬럼)
        content_separator: content 컬럼 구분자 (기본: "\\n\\n")

    Example:
        ```python
        loader = PostgreSQLLoader(
            connection_string="postgresql://user:pass@localhost:5432/db",
            query="SELECT title, content, author FROM articles",
            content_columns=["title", "content"],
            metadata_columns=["author"]
        )
        docs = loader.load()

        for doc in docs:
            print(doc.content)  # "title\\n\\ncontent"
            print(doc.metadata)  # {"author": "...", "source": "postgresql"}
        ```
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        content_separator: str = "\n\n",
        source_name: str = "postgresql",
        **kwargs: Any,
    ):
        """
        PostgreSQL 로더 초기화

        Args:
            connection_string: PostgreSQL 연결 문자열
                예: "postgresql://user:pass@localhost:5432/mydb"
            query: 실행할 SQL 쿼리
            content_columns: content로 사용할 컬럼 목록
                지정하지 않으면 모든 텍스트 컬럼 사용
            metadata_columns: metadata에 포함할 컬럼 목록
                지정하지 않으면 content_columns 제외 모든 컬럼
            content_separator: 여러 컬럼을 합칠 때 구분자
            source_name: metadata에 포함될 source 이름
            **kwargs: psycopg2 연결 옵션
        """
        self.connection_string = connection_string
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.content_separator = content_separator
        self.source_name = source_name
        self.kwargs = kwargs

        self._conn = None
        self._columns: List[str] = []

    def _get_connection(self):
        """데이터베이스 연결 생성"""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 is required for PostgreSQLLoader. "
                "Install it with: pip install psycopg2-binary"
            )

        if self._conn is None:
            self._conn = psycopg2.connect(self.connection_string, **self.kwargs)

        return self._conn

    def _close_connection(self):
        """데이터베이스 연결 종료"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _row_to_document(self, row: tuple, columns: List[str]) -> Document:
        """
        데이터베이스 행을 Document로 변환

        Args:
            row: 데이터베이스 행 (tuple)
            columns: 컬럼 이름 목록

        Returns:
            Document 객체
        """
        row_dict = dict(zip(columns, row))

        # content 컬럼 결정
        if self.content_columns:
            content_cols = self.content_columns
        else:
            # 모든 텍스트 컬럼 사용
            content_cols = [col for col, val in row_dict.items() if isinstance(val, str)]

        # content 생성
        content_parts = []
        for col in content_cols:
            if col in row_dict and row_dict[col] is not None:
                content_parts.append(str(row_dict[col]))

        content = self.content_separator.join(content_parts)

        # metadata 컬럼 결정
        if self.metadata_columns:
            metadata_cols = self.metadata_columns
        else:
            # content_cols 제외 모든 컬럼
            metadata_cols = [col for col in columns if col not in content_cols]

        # metadata 생성
        metadata: Dict[str, Any] = {
            "source": self.source_name,
        }
        for col in metadata_cols:
            if col in row_dict:
                value = row_dict[col]
                # 직렬화 가능한 타입으로 변환
                if hasattr(value, "isoformat"):  # datetime
                    value = value.isoformat()
                metadata[col] = value

        return Document(content=content, metadata=metadata)

    def load(self) -> List[Document]:
        """
        모든 문서를 메모리에 로드

        Returns:
            Document 리스트

        Example:
            ```python
            loader = PostgreSQLLoader(
                connection_string="postgresql://user:pass@localhost:5432/db",
                query="SELECT * FROM articles LIMIT 100"
            )
            docs = loader.load()
            print(f"Loaded {len(docs)} documents")
            ```
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Generator[Document, None, None]:
        """
        제너레이터로 문서를 하나씩 로드 (메모리 효율적)

        Yields:
            Document 객체

        Example:
            ```python
            loader = PostgreSQLLoader(
                connection_string="postgresql://user:pass@localhost:5432/db",
                query="SELECT * FROM large_table"
            )
            for doc in loader.lazy_load():
                process(doc)  # 하나씩 처리
            ```
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            logger.info(f"Executing query: {self.query[:100]}...")
            cursor.execute(self.query)

            # 컬럼 이름 추출
            columns = [desc[0] for desc in cursor.description]
            self._columns = columns

            # 행 단위로 처리
            row_count = 0
            for row in cursor:
                yield self._row_to_document(row, columns)
                row_count += 1

            logger.info(f"Loaded {row_count} documents from PostgreSQL")

            cursor.close()

        except Exception as e:
            logger.error(f"PostgreSQL loader error: {e}")
            raise

        finally:
            self._close_connection()

    async def aload(self) -> List[Document]:
        """
        비동기로 모든 문서 로드 (asyncpg 사용)

        Returns:
            Document 리스트

        Example:
            ```python
            loader = PostgreSQLLoader(
                connection_string="postgresql://user:pass@localhost:5432/db",
                query="SELECT * FROM articles"
            )
            docs = await loader.aload()
            ```
        """
        try:
            import asyncpg
        except ImportError:
            raise ImportError(
                "asyncpg is required for async PostgreSQL loading. "
                "Install it with: pip install asyncpg"
            )

        # connection_string 파싱 (asyncpg는 다른 형식 사용)
        conn = await asyncpg.connect(self.connection_string)

        try:
            logger.info(f"Executing async query: {self.query[:100]}...")
            rows = await conn.fetch(self.query)

            if not rows:
                return []

            # 컬럼 이름 추출
            columns = list(rows[0].keys())
            self._columns = columns

            documents = []
            for row in rows:
                row_tuple = tuple(row.values())
                documents.append(self._row_to_document(row_tuple, columns))

            logger.info(f"Loaded {len(documents)} documents from PostgreSQL (async)")
            return documents

        finally:
            await conn.close()

    def __repr__(self) -> str:
        return (
            f"PostgreSQLLoader(query={self.query[:50]}..., content_columns={self.content_columns})"
        )
