"""
Loaders Implementations - 문서 로더 구현체들
"""

import csv
from pathlib import Path
from typing import List, Optional, Union

from .base import BaseDocumentLoader
from .types import Document

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class TextLoader(BaseDocumentLoader):
    """
    텍스트 파일 로더

    Example:
        ```python
        from beanllm.domain.loaders import TextLoader

        loader = TextLoader("file.txt", encoding="utf-8")
        docs = loader.load()
        ```
    """

    def __init__(
        self, file_path: Union[str, Path], encoding: str = "utf-8", autodetect_encoding: bool = True
    ):
        """
        Args:
            file_path: 파일 경로
            encoding: 인코딩
            autodetect_encoding: 인코딩 자동 감지
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def load(self) -> List[Document]:
        """파일 로딩"""
        try:
            content = self._read_file()
            return [
                Document(
                    content=content,
                    metadata={"source": str(self.file_path), "encoding": self.encoding},
                )
            ]
        except Exception as e:
            logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()

    def _read_file(self) -> str:
        """파일 읽기"""
        # 인코딩 자동 감지
        if self.autodetect_encoding:
            try:
                with open(self.file_path, "r", encoding=self.encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                # UTF-8 실패 시 다른 인코딩 시도
                for encoding in ["cp949", "euc-kr", "latin-1"]:
                    try:
                        with open(self.file_path, "r", encoding=encoding) as f:
                            content = f.read()
                            self.encoding = encoding
                            logger.info(f"Auto-detected encoding: {encoding}")
                            return content
                    except UnicodeDecodeError:
                        continue
                raise
        else:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                return f.read()


class PDFLoader(BaseDocumentLoader):
    """
    PDF 로더

    Example:
        ```python
        from beanllm.domain.loaders import PDFLoader

        loader = PDFLoader("document.pdf")
        docs = loader.load()  # 페이지별로 분리

        # 특정 페이지만
        loader = PDFLoader("document.pdf", pages=[1, 2, 3])
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        password: Optional[str] = None,
    ):
        """
        Args:
            file_path: PDF 경로
            pages: 로딩할 페이지 번호 (None이면 전체)
            password: PDF 비밀번호
        """
        self.file_path = Path(file_path)
        self.pages = pages
        self.password = password

        # pypdf 확인
        try:
            import pypdf

            self.pypdf = pypdf
        except ImportError:
            raise ImportError("pypdf is required for PDFLoader. Install it with: pip install pypdf")

    def load(self) -> List[Document]:
        """PDF 로딩 (페이지별 문서)"""
        documents = []

        try:
            with open(self.file_path, "rb") as f:
                pdf_reader = self.pypdf.PdfReader(f, password=self.password)

                # 페이지 선택
                pages_to_load = self.pages or range(len(pdf_reader.pages))

                for page_num in pages_to_load:
                    if page_num >= len(pdf_reader.pages):
                        logger.warning(f"Page {page_num} out of range")
                        continue

                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()

                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": str(self.file_path),
                                "page": page_num,
                                "total_pages": len(pdf_reader.pages),
                            },
                        )
                    )

            logger.info(f"Loaded {len(documents)} pages from {self.file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        yield from self.load()


class CSVLoader(BaseDocumentLoader):
    """
    CSV 로더

    Example:
        ```python
        from beanllm.domain.loaders import CSVLoader

        # 행별로 문서 생성
        loader = CSVLoader("data.csv")
        docs = loader.load()

        # 특정 컬럼만 content로
        loader = CSVLoader("data.csv", content_columns=["text", "description"])
        ```
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ):
        """
        Args:
            file_path: CSV 경로
            content_columns: content로 사용할 컬럼들 (None이면 전체)
            metadata_columns: metadata로 저장할 컬럼들
            encoding: 인코딩
        """
        self.file_path = Path(file_path)
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        self.encoding = encoding

    def load(self) -> List[Document]:
        """CSV 로딩 (행별 문서)"""
        documents = []

        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    # Content 생성
                    if self.content_columns:
                        content_parts = [
                            f"{col}: {row.get(col, '')}"
                            for col in self.content_columns
                            if col in row
                        ]
                        content = "\n".join(content_parts)
                    else:
                        # 모든 컬럼 사용
                        content = "\n".join([f"{k}: {v}" for k, v in row.items()])

                    # Metadata
                    metadata = {"source": str(self.file_path), "row": i}

                    if self.metadata_columns:
                        for col in self.metadata_columns:
                            if col in row:
                                metadata[col] = row[col]

                    documents.append(Document(content=content, metadata=metadata))

            logger.info(f"Loaded {len(documents)} rows from {self.file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load CSV {self.file_path}: {e}")
            raise

    def lazy_load(self):
        """지연 로딩"""
        with open(self.file_path, "r", encoding=self.encoding) as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                # Content
                if self.content_columns:
                    content_parts = [
                        f"{col}: {row.get(col, '')}" for col in self.content_columns if col in row
                    ]
                    content = "\n".join(content_parts)
                else:
                    content = "\n".join([f"{k}: {v}" for k, v in row.items()])

                # Metadata
                metadata = {"source": str(self.file_path), "row": i}

                if self.metadata_columns:
                    for col in self.metadata_columns:
                        if col in row:
                            metadata[col] = row[col]

                yield Document(content=content, metadata=metadata)


class DirectoryLoader(BaseDocumentLoader):
    """
    디렉토리 로더 (재귀)

    Example:
        ```python
        from beanllm.domain.loaders import DirectoryLoader

        # 모든 .txt 파일
        loader = DirectoryLoader("./docs", glob="**/*.txt")
        docs = loader.load()

        # 모든 파일 (자동 감지)
        loader = DirectoryLoader("./docs")
        ```
    """

    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/*",
        exclude: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        """
        Args:
            path: 디렉토리 경로
            glob: 파일 패턴
            exclude: 제외할 패턴
            recursive: 재귀 검색
        """
        self.path = Path(path)
        self.glob = glob
        self.exclude = exclude or []
        self.recursive = recursive

    def load(self) -> List[Document]:
        """디렉토리 로딩"""
        from .factory import DocumentLoader

        documents = []

        # 파일 검색
        if self.recursive:
            files = self.path.glob(self.glob)
        else:
            files = self.path.glob(self.glob.replace("**/", ""))

        for file_path in files:
            # 제외 패턴 확인
            if any(file_path.match(pattern) for pattern in self.exclude):
                continue

            # 파일만
            if not file_path.is_file():
                continue

            # 자동 감지해서 로딩
            loader = DocumentLoader.get_loader(file_path)
            if loader:
                try:
                    file_docs = loader.load()
                    documents.extend(file_docs)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {self.path}")
        return documents

    def lazy_load(self):
        """지연 로딩"""
        from .factory import DocumentLoader

        if self.recursive:
            files = self.path.glob(self.glob)
        else:
            files = self.path.glob(self.glob.replace("**/", ""))

        for file_path in files:
            if any(file_path.match(pattern) for pattern in self.exclude):
                continue

            if not file_path.is_file():
                continue

            loader = DocumentLoader.get_loader(file_path)
            if loader:
                try:
                    yield from loader.lazy_load()
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
