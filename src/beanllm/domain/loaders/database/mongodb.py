"""
MongoDB Document Loader

MongoDB 데이터베이스에서 문서를 로드합니다.
pymongo (동기) 또는 motor (비동기)를 지원합니다.

Requirements:
    pip install pymongo  # 또는 motor

Example:
    ```python
    from beanllm.domain.loaders.database import MongoDBLoader

    # 기본 사용
    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="mydb",
        collection="documents"
    )
    docs = loader.load()

    # 필터와 함께 사용
    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="mydb",
        collection="articles",
        query={"status": "published", "category": "tech"},
        content_fields=["title", "body"],
        metadata_fields=["author", "tags", "created_at"]
    )
    ```
"""

import logging
from typing import Any, Dict, Generator, List, Optional

from beanllm.domain.loaders.base import BaseDocumentLoader
from beanllm.domain.loaders.types import Document

logger = logging.getLogger(__name__)


class MongoDBLoader(BaseDocumentLoader):
    """
    MongoDB 데이터베이스에서 문서를 로드하는 로더

    MongoDB의 유연한 스키마를 활용하여 다양한 형태의 문서를 로드합니다.
    nested field도 dot notation으로 접근 가능합니다.

    Attributes:
        connection_string: MongoDB 연결 문자열
        database: 데이터베이스 이름
        collection: 컬렉션 이름
        query: MongoDB 쿼리 (filter)
        content_fields: content로 사용할 필드 목록
        metadata_fields: metadata에 포함할 필드 목록

    Example:
        ```python
        # 기본 사용
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="mydb",
            collection="articles"
        )
        docs = loader.load()

        # 필터와 프로젝션
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="mydb",
            collection="articles",
            query={"status": "published"},
            projection={"_id": 1, "title": 1, "content": 1, "author": 1},
            content_fields=["title", "content"],
            metadata_fields=["_id", "author"]
        )
        ```
    """

    def __init__(
        self,
        connection_string: str,
        database: str,
        collection: str,
        query: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, int]] = None,
        content_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        content_separator: str = "\n\n",
        limit: int = 0,
        sort: Optional[List[tuple]] = None,
        source_name: str = "mongodb",
        **kwargs: Any,
    ):
        """
        MongoDB 로더 초기화

        Args:
            connection_string: MongoDB 연결 문자열
                예: "mongodb://user:pass@localhost:27017"
                예: "mongodb+srv://user:pass@cluster.mongodb.net"
            database: 데이터베이스 이름
            collection: 컬렉션 이름
            query: MongoDB 쿼리 필터 (기본: {})
            projection: 가져올 필드 지정 (기본: 모든 필드)
            content_fields: content로 사용할 필드 목록
                지정하지 않으면 모든 문자열 필드 사용
            metadata_fields: metadata에 포함할 필드 목록
                지정하지 않으면 content_fields 제외 모든 필드
            content_separator: 여러 필드를 합칠 때 구분자
            limit: 최대 문서 수 (0 = 무제한)
            sort: 정렬 조건 [("field", 1/-1), ...]
            source_name: metadata에 포함될 source 이름
            **kwargs: pymongo 연결 옵션
        """
        self.connection_string = connection_string
        self.database = database
        self.collection_name = collection
        self.query = query or {}
        self.projection = projection
        self.content_fields = content_fields
        self.metadata_fields = metadata_fields
        self.content_separator = content_separator
        self.limit = limit
        self.sort = sort
        self.source_name = source_name
        self.kwargs = kwargs

        self._client = None
        self._db = None
        self._collection = None

    def _get_connection(self):
        """MongoDB 연결 생성"""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "pymongo is required for MongoDBLoader. Install it with: pip install pymongo"
            )

        if self._client is None:
            self._client = MongoClient(self.connection_string, **self.kwargs)
            self._db = self._client[self.database]
            self._collection = self._db[self.collection_name]

        return self._collection

    def _close_connection(self):
        """MongoDB 연결 종료"""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None

    def _get_nested_value(self, doc: Dict, field: str) -> Any:
        """
        Nested field 값을 가져옴 (dot notation 지원)

        Args:
            doc: MongoDB 문서
            field: 필드 이름 (예: "author.name", "tags.0")

        Returns:
            필드 값 (없으면 None)
        """
        keys = field.split(".")
        value = doc

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list):
                try:
                    value = value[int(key)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

            if value is None:
                return None

        return value

    def _doc_to_document(self, mongo_doc: Dict[str, Any]) -> Document:
        """
        MongoDB 문서를 Document로 변환

        Args:
            mongo_doc: MongoDB 문서 (dict)

        Returns:
            Document 객체
        """
        # content 필드 결정
        if self.content_fields:
            content_fields = self.content_fields
        else:
            # 모든 문자열 필드 자동 감지
            content_fields = [
                key for key, value in mongo_doc.items() if isinstance(value, str) and key != "_id"
            ]

        # content 생성
        content_parts = []
        for field in content_fields:
            value = self._get_nested_value(mongo_doc, field)
            if value is not None:
                if isinstance(value, str):
                    content_parts.append(value)
                elif isinstance(value, list):
                    content_parts.append(", ".join(str(v) for v in value))
                else:
                    content_parts.append(str(value))

        content = self.content_separator.join(content_parts)

        # metadata 필드 결정
        if self.metadata_fields:
            metadata_fields = self.metadata_fields
        else:
            # content_fields 제외 모든 필드
            all_keys = list(mongo_doc.keys())
            metadata_fields = [k for k in all_keys if k not in content_fields]

        # metadata 생성
        metadata: Dict[str, Any] = {
            "source": self.source_name,
            "database": self.database,
            "collection": self.collection_name,
        }

        for field in metadata_fields:
            value = self._get_nested_value(mongo_doc, field)
            if value is not None:
                # ObjectId를 문자열로 변환
                if hasattr(value, "__str__") and type(value).__name__ == "ObjectId":
                    value = str(value)
                # datetime 변환
                elif hasattr(value, "isoformat"):
                    value = value.isoformat()
                # 중첩 dict/list는 그대로 유지 (JSON 직렬화 가능)
                metadata[field] = value

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
            collection = self._get_connection()

            logger.info(
                f"Querying MongoDB: {self.database}.{self.collection_name} with filter={self.query}"
            )

            # 쿼리 실행
            cursor = collection.find(self.query, self.projection)

            if self.sort:
                cursor = cursor.sort(self.sort)

            if self.limit > 0:
                cursor = cursor.limit(self.limit)

            # 문서 변환
            doc_count = 0
            for mongo_doc in cursor:
                yield self._doc_to_document(mongo_doc)
                doc_count += 1

            logger.info(f"Loaded {doc_count} documents from MongoDB")

        except Exception as e:
            logger.error(f"MongoDB loader error: {e}")
            raise

        finally:
            self._close_connection()

    async def aload(self) -> List[Document]:
        """
        비동기로 모든 문서 로드 (motor 사용)

        Returns:
            Document 리스트
        """
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError(
                "motor is required for async MongoDB loading. Install it with: pip install motor"
            )

        client = AsyncIOMotorClient(self.connection_string, **self.kwargs)

        try:
            db = client[self.database]
            collection = db[self.collection_name]

            logger.info(f"Async querying MongoDB: {self.database}.{self.collection_name}")

            # 쿼리 실행
            cursor = collection.find(self.query, self.projection)

            if self.sort:
                cursor = cursor.sort(self.sort)

            if self.limit > 0:
                cursor = cursor.limit(self.limit)

            # 문서 변환
            documents = []
            async for mongo_doc in cursor:
                documents.append(self._doc_to_document(mongo_doc))

            logger.info(f"Loaded {len(documents)} documents from MongoDB (async)")
            return documents

        finally:
            client.close()

    def count(self) -> int:
        """쿼리에 해당하는 문서 수 반환"""
        try:
            collection = self._get_connection()
            return collection.count_documents(self.query)
        finally:
            self._close_connection()

    def __repr__(self) -> str:
        return (
            f"MongoDBLoader("
            f"database={self.database}, "
            f"collection={self.collection_name}, "
            f"query={self.query})"
        )
