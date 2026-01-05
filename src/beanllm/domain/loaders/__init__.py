"""
Loaders Domain - 문서 로더 도메인
"""

from .base import BaseDocumentLoader
from .factory import DocumentLoader, load_documents
from .loaders import (
    CSVLoader,
    DirectoryLoader,
    DoclingLoader,
    HTMLLoader,
    JupyterLoader,
    PDFLoader,
    TextLoader,
)
from .types import Document

# beanPDFLoader (고급 PDF 로더)
try:
    from .pdf import PDFLoadConfig, beanPDFLoader
except ImportError:
    # 의존성이 없을 수 있음
    beanPDFLoader = None  # type: ignore
    PDFLoadConfig = None  # type: ignore

__all__ = [
    "Document",
    "BaseDocumentLoader",
    "TextLoader",
    "PDFLoader",
    "CSVLoader",
    "DirectoryLoader",
    "HTMLLoader",
    "JupyterLoader",
    "DoclingLoader",
    "DocumentLoader",
    "load_documents",
]

# beanPDFLoader 추가 (있는 경우)
if beanPDFLoader is not None:
    __all__.extend(["beanPDFLoader", "PDFLoadConfig"])


# 편의 함수: beanPDFLoader 직접 사용
def load_pdf(
    file_path,
    extract_tables: bool = True,
    extract_images: bool = False,
    strategy: str = "auto",
    **kwargs
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
            **kwargs
        )
        return loader.load()
    else:
        # Fallback to PDFLoader
        loader = PDFLoader(file_path, **kwargs)
        return loader.load()


# 편의 함수 추가
if beanPDFLoader is not None:
    __all__.append("load_pdf")
