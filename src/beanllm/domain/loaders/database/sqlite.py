"""
SQLite Document Loader

SQLite 데이터베이스에서 문서를 로드합니다.
Python 기본 라이브러리 sqlite3를 사용하므로 추가 설치가 필요 없습니다.

Example:
    ```python
    from beanllm.domain.loaders.database import SQLiteLoader

    # 기본 사용
    loader = SQLiteLoader(
        db_path="data.db",
        query="SELECT id, title, content FROM articles"
    )
    docs = loader.load()

    # 특정 컬럼을 content로 사용
    loader = SQLiteLoader(
        db_path="data.db",
        table="documents",  # query 대신 table만 지정 가능
        content_columns=["title", "body"],
        metadata_columns=["author", "created_at"]
    )
    ```
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from beanllm.domain.loaders.base import BaseDocumentLoader
from beanllm.domain.loaders.types import Document
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteLoader(BaseDocumentLoader):
    """
    SQLite 데이터베이스에서 문서를 로드하는 로더

    Python 기본 라이브러리를 사용하므로 추가 의존성이 없습니다.
    로컬 개발이나 소규모 애플리케이션에 적합합니다.

    Attributes:
        db_path: SQLite 데이터베이스 파일 경로
        query: 실행할 SQL 쿼리 (table과 배타적)
        table: 쿼리할 테이블 이름 (query와 배타적)
        content_columns: content로 사용할 컬럼 목록
        metadata_columns: metadata에 포함할 컬럼 목록

    Example:
        ```python
        # 쿼리 사용
        loader = SQLiteLoader(
            db_path="data.db",
            query="SELECT title, content FROM articles WHERE published = 1",
            content_columns=["title", "content"]
        )

        # 테이블 전체 로드
        loader = SQLiteLoader(
            db_path="data.db",
            table="documents",
            where="status = 'active'"  # 선택적 WHERE 절
        )
        ```
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        query: Optional[str] = None,
        table: Optional[str] = None,
        where: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        content_separator: str = "\n\n",
        source_name: str = "sqlite",
        **kwargs: Any,
    ):
        """
        SQLite 로더 초기화

        Args:
            db_path: SQLite 데이터베이스 파일 경로
            query: 실행할 SQL 쿼리 (table과 배타적)
            table: 쿼리할 테이블 이름 (query와 배타적)
            where: table 사용 시 WHERE 절 (예: "status = 'active'")
            content_columns: content로 사용할 컬럼 목록
            metadata_columns: metadata에 포함할 컬럼 목록
            content_separator: 여러 컬럼을 합칠 때 구분자
            source_name: metadata에 포함될 source 이름
            **kwargs: sqlite3.connect 옵션
        """
        self.db_path = Path(db_path)
        self.table = table
        self.where = where
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.content_separator = content_separator
        self.source_name = source_name
        self.kwargs = kwargs

        # query 또는 table 중 하나만 지정
        if query and table:
            raise ValueError("query와 table 중 하나만 지정해야 합니다.")

        if query:
            self.query = query
        elif table:
            self.query = f"SELECT * FROM {table}"
            if where:
                self.query += f" WHERE {where}"
        else:
            raise ValueError("query 또는 table을 지정해야 합니다.")

        self._conn: Optional[sqlite3.Connection] = None
        self._columns: List[str] = []

    def _get_connection(self) -> sqlite3.Connection:
        """데이터베이스 연결 생성"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), **self.kwargs)
            # Row factory for dict-like access
            self._conn.row_factory = sqlite3.Row

        return self._conn

    def _close_connection(self):
        """데이터베이스 연결 종료"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """
        SQLite 행을 Document로 변환

        Args:
            row: SQLite Row 객체

        Returns:
            Document 객체
        """
        row_dict = dict(row)
        columns = list(row_dict.keys())

        # content 컬럼 결정
        if self.content_columns:
            content_cols = self.content_columns
        else:
            # 모든 텍스트 컬럼 사용 (TEXT, VARCHAR 등)
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
            "db_path": str(self.db_path),
        }

        if self.table:
            metadata["table"] = self.table

        for col in metadata_cols:
            if col in row_dict:
                value = row_dict[col]
                metadata[col] = value

        return Document(content=content, metadata=metadata)

    def load(self) -> List[Document]:
        """
        모든 문서를 메모리에 로드

        Returns:
            Document 리스트
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Generator[Document, None, None]:
        """
        제너레이터로 문서를 하나씩 로드 (메모리 효율적)

        Yields:
            Document 객체
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            logger.info(f"Executing SQLite query: {self.query[:100]}...")
            cursor.execute(self.query)

            # 컬럼 이름 추출
            if cursor.description:
                self._columns = [desc[0] for desc in cursor.description]

            # 행 단위로 처리
            row_count = 0
            for row in cursor:
                yield self._row_to_document(row)
                row_count += 1

            logger.info(f"Loaded {row_count} documents from SQLite")

            cursor.close()

        except Exception as e:
            logger.error(f"SQLite loader error: {e}")
            raise

        finally:
            self._close_connection()

    def get_tables(self) -> List[str]:
        """데이터베이스의 모든 테이블 목록 반환"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [row[0] for row in cursor.fetchall()]

            cursor.close()
            return tables

        finally:
            self._close_connection()

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """테이블 스키마 정보 반환"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default": row[4],
                        "pk": bool(row[5]),
                    }
                )

            cursor.close()
            return columns

        finally:
            self._close_connection()

    def count(self) -> int:
        """쿼리 결과의 문서 수 반환"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # COUNT 쿼리로 변환
            if self.table:
                count_query = f"SELECT COUNT(*) FROM {self.table}"
                if self.where:
                    count_query += f" WHERE {self.where}"
            else:
                # 서브쿼리 사용
                count_query = f"SELECT COUNT(*) FROM ({self.query})"

            cursor.execute(count_query)
            result = cursor.fetchone()
            cursor.close()

            return result[0] if result else 0

        finally:
            self._close_connection()

    def __repr__(self) -> str:
        return f"SQLiteLoader(db_path={self.db_path}, query={self.query[:50]}...)"
