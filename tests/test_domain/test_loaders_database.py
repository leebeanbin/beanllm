"""
Database Loaders 테스트

MongoDBLoader, SQLiteLoader, PostgreSQLLoader에 대한 단위 테스트.
모든 외부 의존성(pymongo, motor, psycopg2, asyncpg)은 Mock으로 대체합니다.
sqlite3는 표준 라이브러리이므로 unittest.mock.patch를 사용합니다.
"""

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from beanllm.domain.loaders.database.mongodb import MongoDBLoader
from beanllm.domain.loaders.database.postgresql import PostgreSQLLoader
from beanllm.domain.loaders.database.sqlite import SQLiteLoader
from beanllm.domain.loaders.types import Document

# ---------------------------------------------------------------------------
# MongoDBLoader Tests
# ---------------------------------------------------------------------------


class TestMongoDBLoaderInit:
    """MongoDBLoader 초기화 테스트"""

    def test_basic_init(self):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="testdb",
            collection="articles",
        )
        assert loader.connection_string == "mongodb://localhost:27017"
        assert loader.database == "testdb"
        assert loader.collection_name == "articles"
        assert loader.query == {}
        assert loader.limit == 0
        assert loader.content_fields is None
        assert loader.metadata_fields is None

    def test_init_with_all_options(self):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="testdb",
            collection="articles",
            query={"status": "published"},
            projection={"_id": 1, "title": 1},
            content_fields=["title", "body"],
            metadata_fields=["author"],
            content_separator=" | ",
            limit=100,
            sort=[("created_at", -1)],
            source_name="my_mongo",
        )
        assert loader.query == {"status": "published"}
        assert loader.content_fields == ["title", "body"]
        assert loader.metadata_fields == ["author"]
        assert loader.content_separator == " | "
        assert loader.limit == 100
        assert loader.source_name == "my_mongo"

    def test_repr(self):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="testdb",
            collection="articles",
        )
        r = repr(loader)
        assert "MongoDBLoader" in r
        assert "testdb" in r
        assert "articles" in r


class TestMongoDBLoaderGetNestedValue:
    """_get_nested_value 메서드 테스트"""

    def setup_method(self):
        self.loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )

    def test_simple_field(self):
        doc = {"name": "Alice", "age": 30}
        assert self.loader._get_nested_value(doc, "name") == "Alice"

    def test_nested_field_dot_notation(self):
        doc = {"author": {"name": "Bob", "email": "bob@example.com"}}
        assert self.loader._get_nested_value(doc, "author.name") == "Bob"

    def test_missing_field_returns_none(self):
        doc = {"name": "Alice"}
        assert self.loader._get_nested_value(doc, "missing") is None

    def test_list_index_access(self):
        doc = {"tags": ["python", "ai", "llm"]}
        assert self.loader._get_nested_value(doc, "tags.1") == "ai"

    def test_invalid_list_index_returns_none(self):
        doc = {"tags": ["python"]}
        assert self.loader._get_nested_value(doc, "tags.99") is None


class TestMongoDBLoaderDocToDocument:
    """_doc_to_document 변환 테스트"""

    def setup_method(self):
        self.loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="mydb",
            collection="articles",
            content_fields=["title", "body"],
            metadata_fields=["author"],
        )

    def test_basic_conversion(self):
        mongo_doc = {
            "_id": "abc123",
            "title": "Hello World",
            "body": "Some content here",
            "author": "Alice",
        }
        doc = self.loader._doc_to_document(mongo_doc)
        assert isinstance(doc, Document)
        assert "Hello World" in doc.content
        assert "Some content here" in doc.content
        assert doc.metadata["source"] == "mongodb"
        assert doc.metadata["database"] == "mydb"
        assert doc.metadata["collection"] == "articles"
        assert doc.metadata["author"] == "Alice"

    def test_auto_detect_string_fields(self):
        """content_fields 미지정 시 문자열 필드 자동 감지"""
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        mongo_doc = {
            "_id": "id1",
            "title": "Auto Title",
            "count": 42,
        }
        doc = loader._doc_to_document(mongo_doc)
        assert "Auto Title" in doc.content

    def test_list_field_joined(self):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
            content_fields=["tags"],
        )
        mongo_doc = {"tags": ["python", "ai"]}
        doc = loader._doc_to_document(mongo_doc)
        assert "python" in doc.content
        assert "ai" in doc.content

    def test_datetime_converted_to_isoformat(self):
        from datetime import datetime

        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
            content_fields=["title"],
            metadata_fields=["created_at"],
        )
        dt = datetime(2024, 1, 15, 10, 30, 0)
        mongo_doc = {"title": "Test", "created_at": dt}
        doc = loader._doc_to_document(mongo_doc)
        assert "2024-01-15" in doc.metadata["created_at"]


class TestMongoDBLoaderLoad:
    """MongoDBLoader.load() / lazy_load() 테스트"""

    @pytest.fixture(autouse=True)
    def _patch_pymongo(self):
        mock_pymongo = MagicMock()
        with patch.dict("sys.modules", {"pymongo": mock_pymongo, "pymongo.errors": MagicMock()}):
            self._pymongo_mock = mock_pymongo
            yield mock_pymongo

    def _make_mock_collection(self, docs):
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = Mock(return_value=iter(docs))
        # sort/limit return self
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        mock_collection.find.return_value = mock_cursor
        return mock_collection

    def test_load_returns_documents(self):
        docs_data = [
            {"_id": "1", "title": "Doc One", "body": "Content one"},
            {"_id": "2", "title": "Doc Two", "body": "Content two"},
        ]
        mock_collection = self._make_mock_collection(docs_data)

        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
            content_fields=["title", "body"],
        )
        # Bypass _get_connection by directly pre-seeding the internal state
        loader._client = MagicMock()
        loader._db = MagicMock()
        loader._collection = mock_collection

        docs = loader.load()
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert "Doc One" in docs[0].content

    def test_load_empty_collection(self):
        mock_collection = self._make_mock_collection([])
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        loader._client = MagicMock()
        loader._db = MagicMock()
        loader._collection = mock_collection

        docs = loader.load()
        assert docs == []

    def test_load_with_sort_and_limit(self):
        docs_data = [{"_id": "1", "name": "Alice"}]
        mock_collection = self._make_mock_collection(docs_data)
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
            sort=[("name", 1)],
            limit=10,
            content_fields=["name"],
        )
        loader._client = MagicMock()
        loader._db = MagicMock()
        loader._collection = mock_collection

        docs = loader.load()
        assert len(docs) == 1
        mock_collection.find.return_value.sort.assert_called_once_with([("name", 1)])

    def test_get_connection_raises_import_error_without_pymongo(self):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        with patch.dict("sys.modules", {"pymongo": None}):
            with pytest.raises(ImportError, match="pymongo"):
                loader._get_connection()

    def test_close_connection(self):
        mock_client = MagicMock()
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        loader._client = mock_client
        loader._close_connection()
        mock_client.close.assert_called_once()
        assert loader._client is None

    def test_close_connection_when_not_connected(self):
        """연결이 없을 때 close_connection은 안전하게 처리"""
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        loader._close_connection()  # Should not raise

    def test_count_documents(self):
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 42
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        loader._client = MagicMock()
        loader._db = MagicMock()
        loader._collection = mock_collection

        count = loader.count()
        assert count == 42
        mock_collection.count_documents.assert_called_once_with({})

    def test_lazy_load_propagates_exception(self):
        mock_collection = MagicMock()
        mock_collection.find.side_effect = RuntimeError("Connection refused")
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        loader._client = MagicMock()
        loader._db = MagicMock()
        loader._collection = mock_collection

        with pytest.raises(RuntimeError, match="Connection refused"):
            list(loader.lazy_load())


class TestMongoDBLoaderAload:
    """MongoDBLoader.aload() 비동기 테스트"""

    @pytest.fixture(autouse=True)
    def _patch_pymongo(self):
        mock_pymongo = MagicMock()
        with patch.dict("sys.modules", {"pymongo": mock_pymongo, "pymongo.errors": MagicMock()}):
            self._pymongo_mock = mock_pymongo
            yield mock_pymongo

    async def test_aload_raises_import_error_without_motor(self):
        loader = MongoDBLoader(
            connection_string="mongodb://localhost:27017",
            database="db",
            collection="col",
        )
        with patch.dict("sys.modules", {"motor": None, "motor.motor_asyncio": None}):
            with pytest.raises(ImportError, match="motor"):
                await loader.aload()

    async def test_aload_returns_documents(self):
        docs_data = [
            {"_id": "1", "title": "Async Doc", "body": "Content"},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort = Mock(return_value=mock_cursor)
        mock_cursor.limit = Mock(return_value=mock_cursor)
        mock_cursor.__aiter__ = Mock(return_value=aiter_from_list(docs_data))

        mock_collection = MagicMock()
        mock_collection.find.return_value = mock_cursor

        mock_db = MagicMock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)

        mock_motor_client = MagicMock()
        mock_motor_client.__getitem__ = Mock(return_value=mock_db)
        mock_motor_client.close = Mock()

        mock_motor_asyncio = MagicMock()
        mock_motor_asyncio.AsyncIOMotorClient = MagicMock(return_value=mock_motor_client)
        mock_motor_mod = MagicMock()
        mock_motor_mod.motor_asyncio = mock_motor_asyncio

        with patch.dict(
            "sys.modules", {"motor": mock_motor_mod, "motor.motor_asyncio": mock_motor_asyncio}
        ):
            loader = MongoDBLoader(
                connection_string="mongodb://localhost:27017",
                database="db",
                collection="col",
                content_fields=["title", "body"],
            )
            docs = await loader.aload()
            assert len(docs) == 1
            assert "Async Doc" in docs[0].content


# Helper for async iteration
class AsyncIteratorWrapper:
    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


def aiter_from_list(items):
    return AsyncIteratorWrapper(items)


# ---------------------------------------------------------------------------
# SQLiteLoader Tests
# ---------------------------------------------------------------------------


class TestSQLiteLoaderInit:
    """SQLiteLoader 초기화 테스트"""

    def test_init_with_query(self):
        loader = SQLiteLoader(
            db_path="/tmp/test.db",
            query="SELECT * FROM articles",
        )
        assert loader.query == "SELECT * FROM articles"
        assert loader.table is None

    def test_init_with_table(self):
        loader = SQLiteLoader(
            db_path="/tmp/test.db",
            table="articles",
        )
        assert "SELECT * FROM articles" in loader.query
        assert loader.table == "articles"

    def test_init_with_table_and_where(self):
        loader = SQLiteLoader(
            db_path="/tmp/test.db",
            table="articles",
            where="status = 'active'",
        )
        assert "WHERE" in loader.query
        assert "status = 'active'" in loader.query

    def test_init_raises_if_both_query_and_table(self):
        with pytest.raises(ValueError, match="query와 table"):
            SQLiteLoader(
                db_path="/tmp/test.db",
                query="SELECT 1",
                table="articles",
            )

    def test_init_raises_if_neither_query_nor_table(self):
        with pytest.raises(ValueError, match="query 또는 table"):
            SQLiteLoader(db_path="/tmp/test.db")

    def test_repr(self):
        loader = SQLiteLoader(
            db_path="/tmp/test.db",
            query="SELECT * FROM articles",
        )
        r = repr(loader)
        assert "SQLiteLoader" in r


class TestSQLiteLoaderGetConnection:
    """SQLiteLoader 연결 관련 테스트"""

    def test_get_connection_raises_if_file_not_found(self):
        loader = SQLiteLoader(
            db_path="/nonexistent/path/test.db",
            query="SELECT 1",
        )
        with pytest.raises(FileNotFoundError):
            loader._get_connection()

    def test_get_connection_creates_connection(self, tmp_path):
        db_path = tmp_path / "test.db"
        db_path.touch()

        loader = SQLiteLoader(db_path=db_path, query="SELECT 1")
        conn = loader._get_connection()

        assert conn is not None
        loader._close_connection()

    def test_get_connection_reuses_existing(self, tmp_path):
        db_path = tmp_path / "test.db"
        db_path.touch()

        loader = SQLiteLoader(db_path=db_path, query="SELECT 1")
        conn1 = loader._get_connection()
        conn2 = loader._get_connection()

        # Should be same connection object
        assert conn1 is conn2
        loader._close_connection()

    def test_close_connection(self, tmp_path):
        db_path = tmp_path / "test.db"
        db_path.touch()

        loader = SQLiteLoader(db_path=db_path, query="SELECT 1")
        mock_conn = MagicMock()
        loader._conn = mock_conn

        loader._close_connection()
        mock_conn.close.assert_called_once()
        assert loader._conn is None

    def test_close_connection_when_not_connected(self):
        loader = SQLiteLoader(db_path="/tmp/test.db", query="SELECT 1")
        loader._close_connection()  # Should not raise


class TestSQLiteLoaderRowToDocument:
    """SQLiteLoader._row_to_document 테스트"""

    def test_basic_row_conversion(self):
        loader = SQLiteLoader(
            db_path="/tmp/test.db",
            query="SELECT title, body FROM articles",
            content_columns=["title", "body"],
        )
        # Create a dict that mimics sqlite3.Row
        row = {"title": "Hello", "body": "World content", "id": 1}

        class FakeRow:
            def __iter__(self):
                return iter(row.items())

            def keys(self):
                return row.keys()

        # Use dict directly since _row_to_document calls dict(row)
        with patch.object(loader, "_row_to_document", wraps=loader._row_to_document):
            # Simulate sqlite3.Row using a MagicMock that behaves like dict
            mock_row = MagicMock(spec=sqlite3.Row)
            mock_row.__iter__ = Mock(return_value=iter(row.items()))
            mock_row.keys = Mock(return_value=list(row.keys()))
            # patch dict() to return our row dict
            with patch("builtins.dict", side_effect=lambda x: row if x is mock_row else dict(x)):
                doc = loader._row_to_document(mock_row)
                assert isinstance(doc, Document)


class TestSQLiteLoaderLoad:
    """SQLiteLoader.load() テスト"""

    def test_load_with_real_sqlite(self, tmp_path):
        """실제 SQLite 사용하여 load() 테스트"""
        db_path = tmp_path / "test.db"

        # Create actual database first (no mock)
        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE articles (id INTEGER, title TEXT, body TEXT)")
        real_conn.execute("INSERT INTO articles VALUES (1, 'Hello', 'World')")
        real_conn.execute("INSERT INTO articles VALUES (2, 'Foo', 'Bar')")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(
            db_path=db_path,
            query="SELECT id, title, body FROM articles",
            content_columns=["title", "body"],
        )
        docs = loader.load()
        assert len(docs) == 2
        assert any("Hello" in d.content for d in docs)
        assert all(isinstance(d, Document) for d in docs)

    def test_load_empty_result(self, tmp_path):
        db_path = tmp_path / "test.db"

        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE articles (id INTEGER, title TEXT)")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(
            db_path=db_path,
            query="SELECT * FROM articles",
        )
        docs = loader.load()
        assert docs == []

    def test_lazy_load_yields_documents(self, tmp_path):
        db_path = tmp_path / "test.db"

        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE docs (id INTEGER, content TEXT)")
        for i in range(5):
            real_conn.execute(f"INSERT INTO docs VALUES ({i}, 'Content {i}')")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(
            db_path=db_path,
            query="SELECT id, content FROM docs",
            content_columns=["content"],
        )
        docs = list(loader.lazy_load())
        assert len(docs) == 5

    def test_load_with_table_and_metadata(self, tmp_path):
        db_path = tmp_path / "test.db"

        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE articles (id INTEGER, title TEXT, author TEXT)")
        real_conn.execute("INSERT INTO articles VALUES (1, 'Test Title', 'Author A')")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(
            db_path=db_path,
            table="articles",
            content_columns=["title"],
            metadata_columns=["author"],
        )
        docs = loader.load()
        assert len(docs) == 1
        assert "Test Title" in docs[0].content
        assert docs[0].metadata["table"] == "articles"
        assert docs[0].metadata["author"] == "Author A"

    def test_get_tables(self, tmp_path):
        db_path = tmp_path / "test.db"

        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE table_a (id INTEGER)")
        real_conn.execute("CREATE TABLE table_b (id INTEGER)")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(db_path=db_path, table="table_a")
        tables = loader.get_tables()
        assert "table_a" in tables
        assert "table_b" in tables

    def test_count(self, tmp_path):
        db_path = tmp_path / "test.db"

        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        for i in range(7):
            real_conn.execute(f"INSERT INTO items VALUES ({i}, 'item{i}')")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(db_path=db_path, table="items")
        assert loader.count() == 7

    def test_get_table_schema(self, tmp_path):
        db_path = tmp_path / "test.db"

        real_conn = sqlite3.connect(str(db_path))
        real_conn.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        real_conn.commit()
        real_conn.close()

        loader = SQLiteLoader(db_path=db_path, table="t1")
        schema = loader.get_table_schema("t1")
        assert len(schema) == 2
        col_names = [col["name"] for col in schema]
        assert "id" in col_names
        assert "name" in col_names


# ---------------------------------------------------------------------------
# PostgreSQLLoader Tests
# ---------------------------------------------------------------------------


class TestPostgreSQLLoaderInit:
    """PostgreSQLLoader 초기화 테스트"""

    def test_basic_init(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://user:pass@localhost:5432/testdb",
            query="SELECT * FROM articles",
        )
        assert loader.connection_string == "postgresql://user:pass@localhost:5432/testdb"
        assert loader.query == "SELECT * FROM articles"
        assert loader.content_columns is None
        assert loader.metadata_columns is None
        assert loader.source_name == "postgresql"

    def test_init_with_options(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, body FROM docs",
            content_columns=["title", "body"],
            metadata_columns=["author"],
            content_separator=" | ",
            source_name="my_postgres",
        )
        assert loader.content_columns == ["title", "body"]
        assert loader.metadata_columns == ["author"]
        assert loader.content_separator == " | "
        assert loader.source_name == "my_postgres"

    def test_repr(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT id, title FROM articles",
        )
        r = repr(loader)
        assert "PostgreSQLLoader" in r


class TestPostgreSQLLoaderRowToDocument:
    """PostgreSQLLoader._row_to_document 테스트"""

    def setup_method(self):
        self.loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, body, author FROM articles",
            content_columns=["title", "body"],
            metadata_columns=["author"],
        )

    def test_basic_row_conversion(self):
        row = ("My Title", "Some body text", "Alice")
        columns = ["title", "body", "author"]
        doc = self.loader._row_to_document(row, columns)
        assert isinstance(doc, Document)
        assert "My Title" in doc.content
        assert "Some body text" in doc.content
        assert doc.metadata["author"] == "Alice"
        assert doc.metadata["source"] == "postgresql"

    def test_auto_detect_text_columns(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, count FROM articles",
        )
        row = ("Auto Title", 42)
        columns = ["title", "count"]
        doc = loader._row_to_document(row, columns)
        assert "Auto Title" in doc.content
        assert "count" in doc.metadata

    def test_datetime_converted_to_isoformat(self):
        from datetime import datetime

        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, created_at FROM articles",
            content_columns=["title"],
            metadata_columns=["created_at"],
        )
        dt = datetime(2024, 6, 1, 12, 0, 0)
        row = ("Test", dt)
        columns = ["title", "created_at"]
        doc = loader._row_to_document(row, columns)
        assert "2024-06-01" in doc.metadata["created_at"]

    def test_content_separator_used(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, body FROM articles",
            content_columns=["title", "body"],
            content_separator=" --- ",
        )
        row = ("Title", "Body")
        columns = ["title", "body"]
        doc = loader._row_to_document(row, columns)
        assert " --- " in doc.content

    def test_none_values_skipped(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, body FROM articles",
            content_columns=["title", "body"],
        )
        row = ("Title", None)
        columns = ["title", "body"]
        doc = loader._row_to_document(row, columns)
        assert doc.content == "Title"


class TestPostgreSQLLoaderGetConnection:
    """PostgreSQLLoader 연결 테스트"""

    def test_get_connection_raises_import_error_without_psycopg2(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT 1",
        )
        with patch.dict("sys.modules", {"psycopg2": None}):
            with pytest.raises(ImportError, match="psycopg2"):
                loader._get_connection()

    def test_close_connection(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT 1",
        )
        mock_conn = MagicMock()
        loader._conn = mock_conn

        loader._close_connection()
        mock_conn.close.assert_called_once()
        assert loader._conn is None

    def test_close_connection_when_not_connected(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT 1",
        )
        loader._close_connection()  # Should not raise


class TestPostgreSQLLoaderLoad:
    """PostgreSQLLoader.load() 테스트"""

    @pytest.fixture(autouse=True)
    def _patch_psycopg2(self):
        mock_pg = MagicMock()
        with patch.dict("sys.modules", {"psycopg2": mock_pg, "psycopg2.extras": MagicMock()}):
            self._psycopg2_mock = mock_pg
            yield mock_pg

    def _make_loader_with_mock_conn(self, column_names, rows):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, body FROM articles",
            content_columns=["title"],
            metadata_columns=["body"],
        )
        mock_cursor = MagicMock()
        mock_cursor.description = [(col,) for col in column_names]
        mock_cursor.__iter__ = Mock(return_value=iter(rows))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        loader._conn = mock_conn
        return loader, mock_conn, mock_cursor

    @patch("psycopg2.connect")
    def test_load_returns_documents(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.description = [("title",), ("body",)]
        rows = [("Title One", "Body One"), ("Title Two", "Body Two")]
        mock_cursor.__iter__ = Mock(return_value=iter(rows))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title, body FROM articles",
            content_columns=["title"],
            metadata_columns=["body"],
        )
        docs = loader.load()
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].content == "Title One"

    @patch("psycopg2.connect")
    def test_load_empty_result(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.description = [("title",)]
        mock_cursor.__iter__ = Mock(return_value=iter([]))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT title FROM articles",
        )
        docs = loader.load()
        assert docs == []

    @patch("psycopg2.connect")
    def test_lazy_load_yields_documents(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.description = [("content",)]
        rows = [("Content 1",), ("Content 2",), ("Content 3",)]
        mock_cursor.__iter__ = Mock(return_value=iter(rows))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT content FROM docs",
            content_columns=["content"],
        )
        docs = list(loader.lazy_load())
        assert len(docs) == 3

    @patch("psycopg2.connect")
    def test_lazy_load_propagates_exception(self, mock_connect):
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = RuntimeError("Query failed")

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT bad FROM nonexistent",
        )
        with pytest.raises(RuntimeError, match="Query failed"):
            list(loader.lazy_load())

    @patch("psycopg2.connect")
    def test_connection_reused(self, mock_connect):
        """같은 로더 인스턴스에서 연결 재사용"""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT 1",
        )
        conn1 = loader._get_connection()
        conn2 = loader._get_connection()
        mock_connect.assert_called_once()
        assert conn1 is conn2


class TestPostgreSQLLoaderAload:
    """PostgreSQLLoader.aload() 비동기 테스트"""

    @pytest.fixture(autouse=True)
    def _patch_psycopg2(self):
        mock_pg = MagicMock()
        with patch.dict("sys.modules", {"psycopg2": mock_pg, "psycopg2.extras": MagicMock()}):
            yield mock_pg

    async def test_aload_raises_import_error_without_asyncpg(self):
        loader = PostgreSQLLoader(
            connection_string="postgresql://localhost/db",
            query="SELECT 1",
        )
        with patch.dict("sys.modules", {"asyncpg": None}):
            with pytest.raises(ImportError, match="asyncpg"):
                await loader.aload()

    async def test_aload_returns_empty_for_no_rows(self):
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        with patch("asyncpg.connect", return_value=mock_conn):
            loader = PostgreSQLLoader(
                connection_string="postgresql://localhost/db",
                query="SELECT * FROM empty_table",
            )
            docs = await loader.aload()
            assert docs == []

    async def test_aload_returns_documents(self):
        mock_row = MagicMock()
        mock_row.keys.return_value = ["title", "body"]
        mock_row.values.return_value = ["Async Title", "Async Body"]

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [mock_row]

        with patch("asyncpg.connect", return_value=mock_conn):
            loader = PostgreSQLLoader(
                connection_string="postgresql://localhost/db",
                query="SELECT title, body FROM articles",
                content_columns=["title"],
                metadata_columns=["body"],
            )
            docs = await loader.aload()
            assert len(docs) == 1
            assert docs[0].content == "Async Title"
