"""
PDF 엔진 모듈

다양한 PDF 파싱 엔진 구현:
- BasePDFEngine: 추상 기본 클래스
- PyMuPDFEngine: 빠른 처리 (Fast Layer)
- PDFPlumberEngine: 정확한 테이블 추출 (Accurate Layer)
- MarkerEngine: ML 기반 Markdown 변환 (ML Layer)
- PDFExtractKitEngine: DocLayout-YOLO + StructTable (2024-2025)
- DoclingEngine: DocLayNet + TableFormer (IBM, 2024-2025)
"""

from .base import BasePDFEngine
from .pdfplumber_engine import PDFPlumberEngine
from .pymupdf_engine import PyMuPDFEngine

__all__ = [
    "BasePDFEngine",
    "PyMuPDFEngine",
    "PDFPlumberEngine",
]

# MarkerEngine (optional dependency)
try:
    from .marker_engine import MarkerEngine

    __all__.append("MarkerEngine")
except ImportError:
    pass

# PDF-Extract-Kit Engine (optional dependency)
try:
    from .pdf_extract_kit_engine import PDFExtractKitEngine

    __all__.append("PDFExtractKitEngine")
except ImportError:
    pass

# Docling Engine (optional dependency)
try:
    from .docling_engine import DoclingEngine

    __all__.append("DoclingEngine")
except ImportError:
    pass

