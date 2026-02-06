"""
Loaders Domain - 문서 로더 도메인

기본 로더:
- TextLoader: 텍스트 파일 (mmap 최적화)
- PDFLoader: PDF 파일 (pypdf 기반)
- CSVLoader: CSV 파일
- HTMLLoader: HTML 파일/URL
- DirectoryLoader: 디렉토리 (병렬 처리)
- JupyterLoader: Jupyter Notebook

고급 로더:
- DoclingLoader: IBM 문서 파서 (PDF, DOCX, XLSX 등)
- beanPDFLoader: 고급 PDF (5개 엔진, 테이블/이미지 추출)

데이터베이스 로더:
- PostgreSQLLoader: PostgreSQL 데이터베이스
- MongoDBLoader: MongoDB 데이터베이스
- SQLiteLoader: SQLite 데이터베이스 (의존성 없음)

Example:
    ```python
    from beanllm.domain.loaders import PostgreSQLLoader, MongoDBLoader, SQLiteLoader

    # PostgreSQL에서 문서 로드
    loader = PostgreSQLLoader(
        connection_string="postgresql://user:pass@localhost:5432/db",
        query="SELECT title, content FROM articles",
        content_columns=["title", "content"]
    )
    docs = loader.load()

    # MongoDB에서 문서 로드
    loader = MongoDBLoader(
        connection_string="mongodb://localhost:27017",
        database="mydb",
        collection="documents",
        query={"status": "active"}
    )

    # SQLite에서 문서 로드 (의존성 없음)
    loader = SQLiteLoader(db_path="data.db", table="articles")
    ```
"""

from beanllm.domain.loaders.advanced import DoclingLoader
from beanllm.domain.loaders.base import BaseDocumentLoader
from beanllm.domain.loaders.core import (
    CSVLoader,
    DirectoryLoader,
    HTMLLoader,
    JupyterLoader,
    PDFLoader,
    TextLoader,
)
from beanllm.domain.loaders.factory import DocumentLoader, load_documents
from beanllm.domain.loaders.types import Document

# beanPDFLoader (고급 PDF 로더)
try:
    from beanllm.domain.loaders.pdf import PDFLoadConfig, beanPDFLoader
except ImportError:
    # 의존성이 없을 수 있음
    beanPDFLoader = None  # type: ignore
    PDFLoadConfig = None  # type: ignore

# Database Loaders (의존성이 없을 수 있음)
try:
    from beanllm.domain.loaders.database import (
        MongoDBLoader,
        PostgreSQLLoader,
        SQLiteLoader,
    )
except ImportError:
    PostgreSQLLoader = None  # type: ignore
    MongoDBLoader = None  # type: ignore
    SQLiteLoader = None  # type: ignore

__all__ = [
    # Types
    "Document",
    # Base
    "BaseDocumentLoader",
    # Core Loaders
    "TextLoader",
    "PDFLoader",
    "CSVLoader",
    "DirectoryLoader",
    "HTMLLoader",
    "JupyterLoader",
    # Advanced Loaders
    "DoclingLoader",
    # Factory
    "DocumentLoader",
    "load_documents",
]

# beanPDFLoader 추가 (있는 경우)
if beanPDFLoader is not None:
    __all__.extend(["beanPDFLoader", "PDFLoadConfig"])

# Database Loaders 추가 (있는 경우)
if PostgreSQLLoader is not None:
    __all__.extend(["PostgreSQLLoader", "MongoDBLoader", "SQLiteLoader"])


# 편의 함수: beanPDFLoader 직접 사용
def load_pdf(
    file_path,
    extract_tables: bool = True,
    extract_images: bool = False,
    strategy: str = "auto",
    **kwargs,
):
    """
    PDF 로딩 편의 함수 (beanPDFLoader 자동 사용)

    beanPDFLoader를 간단하게 사용할 수 있는 편의 함수입니다.
    beanPDFLoader가 없으면 기본 PDFLoader를 사용합니다.

    Args:
        file_path: PDF 파일 경로
        extract_tables: 테이블 추출 여부 (기본: True)
        extract_images: 이미지 추출 여부 (기본: False)
        strategy: 파싱 전략 ("auto", "fast", "accurate")
        **kwargs: 기타 beanPDFLoader 옵션

    Returns:
        Document 리스트

    Example:
        ```python
        from beanllm.domain.loaders import load_pdf

        # 간단한 사용
        docs = load_pdf("document.pdf")

        # 테이블 추출
        docs = load_pdf("report.pdf", extract_tables=True)

        # 이미지 추출
        docs = load_pdf("images.pdf", extract_images=True)
        ```
    """
    if beanPDFLoader is not None:
        loader = beanPDFLoader(
            file_path,
            extract_tables=extract_tables,
            extract_images=extract_images,
            strategy=strategy,
            **kwargs,
        )
        return loader.load()
    else:
        # Fallback to PDFLoader
        loader = PDFLoader(file_path, **kwargs)
        return loader.load()


# 편의 함수 추가
if beanPDFLoader is not None:
    __all__.append("load_pdf")
