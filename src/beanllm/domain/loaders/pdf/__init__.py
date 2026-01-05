"""
beanPDFLoader - 고급 PDF 로더 모듈

3-Layer 아키텍처를 통한 최적화된 PDF 처리:
- Fast Layer: PyMuPDF (빠른 처리)
- Accurate Layer: pdfplumber (정확한 테이블 추출)
- ML Layer: marker-pdf (구조 보존 Markdown 변환, 향후 구현)
"""

from .bean_pdf_loader import beanPDFLoader
from .extractors import ImageExtractor, TableExtractor
from .models import ImageData, PageData, PDFLoadConfig, PDFLoadResult, TableData

__all__ = [
    "beanPDFLoader",
    "PDFLoadConfig",
    "PageData",
    "TableData",
    "ImageData",
    "PDFLoadResult",
    "TableExtractor",
    "ImageExtractor",
]

