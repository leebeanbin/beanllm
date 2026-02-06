"""
Database Loaders - 데이터베이스 문서 로더

PostgreSQL, MongoDB, SQLite 등 다양한 데이터베이스에서
문서를 로드하여 RAG 파이프라인에 사용할 수 있습니다.

Example:
    ```python
    from beanllm.domain.loaders.database import PostgreSQLLoader, MongoDBLoader

    # PostgreSQL에서 문서 로드
    loader = PostgreSQLLoader(
        connection_string="postgresql://user:pass@localhost:5432/db",
        query="SELECT title, content FROM articles WHERE published = true",
        content_columns=["title", "content"],
        metadata_columns=["author", "created_at"]
    )
    docs = loader.load()

    # MongoDB에서 문서 로드
    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="mydb",
        collection="documents",
        query={"status": "active"}
    )
    docs = loader.load()
    ```
"""

from beanllm.domain.loaders.database.mongodb import MongoDBLoader
from beanllm.domain.loaders.database.postgresql import PostgreSQLLoader
from beanllm.domain.loaders.database.sqlite import SQLiteLoader

__all__ = [
    "PostgreSQLLoader",
    "MongoDBLoader",
    "SQLiteLoader",
]
