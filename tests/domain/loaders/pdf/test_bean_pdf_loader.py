"""
beanPDFLoader 통합 테스트
"""

from pathlib import Path

import pytest

from src.beanllm.domain.loaders.pdf import beanPDFLoader
from src.beanllm.domain.loaders.types import Document

# 테스트 픽스처 경로
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
TABLES_PDF = FIXTURES_DIR / "tables.pdf"
IMAGES_PDF = FIXTURES_DIR / "images.pdf"


class TestBeanPDFLoader:
    """beanPDFLoader 기본 테스트"""

    def test_loader_initialization(self):
        """로더 초기화 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF)
        assert loader.file_path == SIMPLE_PDF
        assert loader.config.strategy == "auto"

    def test_loader_with_strategy(self):
        """전략 지정 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        # Fast 전략
        loader_fast = beanPDFLoader(SIMPLE_PDF, strategy="fast")
        assert loader_fast.config.strategy == "fast"

        # Accurate 전략
        loader_accurate = beanPDFLoader(SIMPLE_PDF, strategy="accurate")
        assert loader_accurate.config.strategy == "accurate"

    def test_load_simple_pdf(self):
        """간단한 PDF 로딩 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF)
        documents = loader.load()

        assert isinstance(documents, list)
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)

    def test_document_content(self):
        """문서 내용 검증"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF)
        documents = loader.load()

        first_doc = documents[0]
        assert hasattr(first_doc, "content")
        assert hasattr(first_doc, "metadata")
        assert len(first_doc.content) > 0
        assert "beanPDFLoader" in first_doc.content

    def test_document_metadata(self):
        """문서 메타데이터 검증"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF)
        documents = loader.load()

        first_doc = documents[0]
        metadata = first_doc.metadata

        assert "source" in metadata
        assert "page" in metadata
        assert "total_pages" in metadata
        assert "engine" in metadata
        assert metadata["page"] >= 0
        assert metadata["total_pages"] > 0

    def test_load_with_tables(self):
        """테이블 추출 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True)
        documents = loader.load()

        assert len(documents) > 0

        # 테이블 메타데이터가 있는지 확인
        has_tables = any("tables" in doc.metadata for doc in documents)
        # tables.pdf이므로 테이블이 있을 가능성이 높음 (하지만 필수는 아님)

    def test_load_with_images(self):
        """이미지 추출 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="fast")
        documents = loader.load()

        assert len(documents) > 0

    def test_load_with_page_range(self):
        """페이지 범위 지정 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF, page_range=(0, 1))
        documents = loader.load()

        assert len(documents) == 1
        assert documents[0].metadata["page"] == 0

    def test_load_with_max_pages(self):
        """최대 페이지 수 제한 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF, max_pages=1)
        documents = loader.load()

        assert len(documents) <= 1

    def test_lazy_load(self):
        """지연 로딩 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF)
        documents = list(loader.lazy_load())

        assert isinstance(documents, list)
        assert len(documents) > 0

    def test_auto_strategy_selection_for_tables(self):
        """테이블 추출 시 자동 전략 선택 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        loader = beanPDFLoader(TABLES_PDF, extract_tables=True, strategy="auto")
        # auto 전략에서 extract_tables=True이면 accurate 전략 선택 예상

        documents = loader.load()
        assert len(documents) > 0

    def test_auto_strategy_selection_for_images(self):
        """이미지 추출 시 자동 전략 선택 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        loader = beanPDFLoader(IMAGES_PDF, extract_images=True, strategy="auto")
        # auto 전략에서 extract_images=True이면 fast 전략 선택 예상

        documents = loader.load()
        assert len(documents) > 0

    def test_invalid_pdf_path(self):
        """존재하지 않는 PDF 파일 테스트"""
        with pytest.raises(FileNotFoundError):
            loader = beanPDFLoader("/nonexistent/file.pdf")
            loader.load()

    def test_multiple_pages(self):
        """여러 페이지 문서 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF)
        documents = loader.load()

        # simple.pdf는 2페이지
        if len(documents) >= 2:
            # 페이지 순서 확인
            assert documents[0].metadata["page"] == 0
            assert documents[1].metadata["page"] == 1
            # 각 페이지 내용이 다름
            assert documents[0].content != documents[1].content

    def test_fast_engine_directly(self):
        """Fast 엔진 직접 사용 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF, strategy="fast")
        documents = loader.load()

        assert len(documents) > 0
        # PyMuPDF 엔진 사용 확인
        assert (
            "PyMuPDF" in documents[0].metadata["engine"]
            or "fast" in documents[0].metadata["strategy"]
        )

    def test_accurate_engine_directly(self):
        """Accurate 엔진 직접 사용 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(SIMPLE_PDF, strategy="accurate")
        documents = loader.load()

        assert len(documents) > 0
        # PDFPlumber 엔진 사용 확인
        assert (
            "PDFPlumber" in documents[0].metadata["engine"]
            or "accurate" in documents[0].metadata["strategy"]
        )

    def test_config_options(self):
        """고급 설정 옵션 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        loader = beanPDFLoader(
            SIMPLE_PDF,
            pymupdf_text_mode="dict",
            pymupdf_extract_fonts=True,
            pymupdf_extract_links=True,
            strategy="fast",
        )

        documents = loader.load()
        assert len(documents) > 0
