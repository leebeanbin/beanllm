"""
Loaders Factory - 문서 로더 팩토리
"""

from pathlib import Path
from typing import List, Optional, Union

from .base import BaseDocumentLoader
from .loaders import CSVLoader, DirectoryLoader, PDFLoader, TextLoader
from .types import Document

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class DocumentLoader:
    """
    Document Loader 팩토리

    **beanllm 방식: 자동 감지!**

    Example:
        ```python
        from beanllm.domain.loaders import DocumentLoader

        # 자동 감지 (기본)
        docs = DocumentLoader.load("file.pdf")  # PDFLoader
        docs = DocumentLoader.load("file.csv")  # CSVLoader
        docs = DocumentLoader.load("file.txt")  # TextLoader
        docs = DocumentLoader.load("./folder") # DirectoryLoader

        # beanPDFLoader 자동 사용 (고급 옵션 감지)
        docs = DocumentLoader.load("file.pdf", extract_tables=True)  # beanPDFLoader 자동 사용
        docs = DocumentLoader.load("file.pdf", extract_images=True)  # beanPDFLoader 자동 사용
        docs = DocumentLoader.load("file.pdf", strategy="fast")  # beanPDFLoader 자동 사용
        ```
    """

    # 확장자별 로더 매핑
    LOADERS = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".pdf": PDFLoader,  # 기본 PDF 로더
        ".csv": CSVLoader,
        # 추가 가능
    }

    # 타입 이름별 로더 매핑 (명시적 선택용)
    LOADER_TYPES = {
        "text": TextLoader,
        "txt": TextLoader,
        "markdown": TextLoader,
        "md": TextLoader,
        "pdf": PDFLoader,  # 기본 PDF 로더
        "csv": CSVLoader,
        "directory": DirectoryLoader,
        "dir": DirectoryLoader,
    }

    # beanPDFLoader (고급 PDF 로더, 선택적)
    @classmethod
    def _get_bean_pdf_loader(cls):
        """beanPDFLoader 가져오기 (선택적)"""
        try:
            from .pdf import beanPDFLoader
            return beanPDFLoader
        except ImportError:
            return None

    @classmethod
    def load(
        cls, source: Union[str, Path], loader_type: Optional[str] = None, **kwargs
    ) -> List[Document]:
        """
        문서 로딩 (자동 감지 또는 명시적 지정)

        Args:
            source: 파일/디렉토리 경로
            loader_type: 로더 타입 명시 (None이면 자동 감지)
                       'text', 'pdf', 'csv', 'directory' 등
            **kwargs: 로더별 파라미터

        Returns:
            문서 리스트

        Example:
            ```python
            # 자동 감지 (기본)
            docs = DocumentLoader.load("file.pdf")  # PDFLoader 사용

            # beanPDFLoader 자동 사용 (고급 옵션 감지)
            docs = DocumentLoader.load("file.pdf", extract_tables=True)  # beanPDFLoader 자동
            docs = DocumentLoader.load("file.pdf", extract_images=True)  # beanPDFLoader 자동

            # 명시적 지정
            docs = DocumentLoader.load("file.txt", loader_type="pdf")
            docs = DocumentLoader.load("data.csv", loader_type="csv", content_columns=["text"])
            docs = DocumentLoader.load("file.pdf", loader_type="beanpdf")  # 명시적 beanPDFLoader
            ```
        """
        loader = cls.get_loader(source, loader_type=loader_type, **kwargs)

        if loader is None:
            raise ValueError(f"No suitable loader found for: {source}")

        return loader.load()

    @classmethod
    def get_loader(
        cls, source: Union[str, Path], loader_type: Optional[str] = None, **kwargs
    ) -> Optional[BaseDocumentLoader]:
        """
        적절한 로더 선택 (자동 감지 또는 명시적 지정)

        Args:
            source: 파일/디렉토리 경로
            loader_type: 로더 타입 명시 (None이면 자동 감지)
            **kwargs: 로더별 파라미터

        Returns:
            Loader 인스턴스
        """
        path = Path(source)

        # 명시적 타입 지정이 있으면 우선 사용
        if loader_type:
            loader_type_lower = loader_type.lower()

            # beanPDFLoader 체크 (고급 PDF 로더)
            if loader_type_lower in ["beanpdf", "bean-pdf", "advanced-pdf"]:
                bean_loader = cls._get_bean_pdf_loader()
                if bean_loader:
                    return bean_loader(path, **kwargs)
                else:
                    logger.warning(
                        "beanPDFLoader not available, falling back to PDFLoader. "
                        "Install: pip install PyMuPDF pdfplumber"
                    )
                    # Fallback to PDFLoader
                    return PDFLoader(path, **kwargs)

            if loader_type_lower in cls.LOADER_TYPES:
                loader_class = cls.LOADER_TYPES[loader_type_lower]
                return loader_class(path, **kwargs)
            else:
                logger.warning(
                    f"Unknown loader type: {loader_type}, falling back to auto-detection"
                )

        # 자동 감지
        # 디렉토리
        if path.is_dir():
            return DirectoryLoader(path, **kwargs)

        # 파일
        elif path.is_file():
            suffix = path.suffix.lower()

            # PDF 파일인 경우: beanPDFLoader 전용 옵션이 있으면 자동 사용
            if suffix == ".pdf":
                # beanPDFLoader 전용 옵션 체크
                beanpdf_options = {
                    "extract_tables",
                    "extract_images",
                    "to_markdown",
                    "enable_ocr",
                    "layout_analysis",
                    "strategy",
                }
                if any(key in kwargs for key in beanpdf_options):
                    # beanPDFLoader 사용
                    bean_loader = cls._get_bean_pdf_loader()
                    if bean_loader:
                        logger.debug("Auto-detected beanPDFLoader (advanced options detected)")
                        return bean_loader(path, **kwargs)
                    else:
                        logger.warning(
                            "beanPDFLoader options detected but not available. "
                            "Falling back to PDFLoader. Install: pip install PyMuPDF pdfplumber"
                        )

                # 기본: PDFLoader (기존 동작 유지)
                return PDFLoader(path, **kwargs)

            if suffix in cls.LOADERS:
                loader_class = cls.LOADERS[suffix]
                return loader_class(path, **kwargs)
            else:
                # 기본: TextLoader
                logger.warning(f"Unknown file type: {suffix}, using TextLoader")
                return TextLoader(path, **kwargs)

        else:
            logger.error(f"Path not found: {path}")
            return None


# 편의 함수
def load_documents(
    source: Union[str, Path], loader_type: Optional[str] = None, **kwargs
) -> List[Document]:
    """
    문서 로딩 편의 함수

    Args:
        source: 파일/디렉토리 경로
        loader_type: 로더 타입 명시 (None이면 자동 감지)
        **kwargs: 로더별 파라미터

    Example:
        ```python
        from beanllm.domain.loaders import load_documents

        # 자동 감지
        docs = load_documents("file.pdf")
        docs = load_documents("./folder", glob="**/*.txt")

        # 명시적 지정
        docs = load_documents("file.txt", loader_type="pdf")
        docs = load_documents("data.csv", loader_type="csv", content_columns=["name"])
        ```
    """
    return DocumentLoader.load(source, loader_type=loader_type, **kwargs)
