"""
PDF Loader

PDF 파일 로더
"""

import logging
import mmap
import re
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .base import BaseDocumentLoader
from .security import validate_file_path
from .types import Document

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

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
        validate_path: bool = True,
    ):
        """
        Args:
            file_path: PDF 경로
            pages: 로딩할 페이지 번호 (None이면 전체)
            password: PDF 비밀번호
            validate_path: 경로 검증 여부 (기본: True, Path Traversal 방지)
        """
        # 경로 검증 (Path Traversal 방지)
        if validate_path:
            self.file_path = validate_file_path(file_path)
        else:
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


