"""
PDF 엔진 모듈

다양한 PDF 파싱 엔진 구현:
- BasePDFEngine: 추상 기본 클래스
- PyMuPDFEngine: 빠른 처리 (Fast Layer)
- PDFPlumberEngine: 정확한 테이블 추출 (Accurate Layer)
- MarkerEngine: ML 기반 Markdown 변환 (ML Layer)
"""

from .base import BasePDFEngine
from .pymupdf_engine import PyMuPDFEngine
from .pdfplumber_engine import PDFPlumberEngine

try:
    from .marker_engine import MarkerEngine

    __all__ = [
        "BasePDFEngine",
        "PyMuPDFEngine",
        "PDFPlumberEngine",
        "MarkerEngine",
    ]
except ImportError:
    # marker-pdf가 설치되지 않은 경우
    __all__ = [
        "BasePDFEngine",
        "PyMuPDFEngine",
        "PDFPlumberEngine",
    ]

