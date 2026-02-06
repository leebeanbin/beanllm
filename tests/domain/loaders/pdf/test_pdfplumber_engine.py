"""
PDFPlumberEngine 테스트
"""

from pathlib import Path

import pytest

from src.beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

# 테스트 픽스처 경로
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
TABLES_PDF = FIXTURES_DIR / "tables.pdf"


class TestPDFPlumberEngine:
    """PDFPlumberEngine 기본 테스트"""

    def test_engine_initialization(self):
        """엔진 초기화 테스트"""
        engine = PDFPlumberEngine()
        assert engine.name == "PDFPlumber"

    def test_engine_info(self):
        """엔진 정보 반환 테스트"""
        engine = PDFPlumberEngine()
        info = engine.get_engine_info()

        assert "name" in info
        assert "class" in info
        assert info["name"] == "PDFPlumber"
        assert info["class"] == "PDFPlumberEngine"

    def test_extract_simple_pdf(self):
        """간단한 PDF 추출 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)

        assert "pages" in result
        assert "metadata" in result
        assert len(result["pages"]) > 0
        assert result["metadata"]["total_pages"] >= 1
        assert result["metadata"]["engine"] == "PDFPlumber"

    def test_extract_with_text_content(self):
        """텍스트 내용 추출 확인"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)

        first_page = result["pages"][0]
        assert "text" in first_page
        assert len(first_page["text"]) > 0
        assert "beanPDFLoader" in first_page["text"]

    def test_extract_tables(self):
        """테이블 추출 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        engine = PDFPlumberEngine()
        config = {"extract_tables": True}

        result = engine.extract(TABLES_PDF, config)

        assert "pages" in result
        # tables.pdf에는 테이블이 있어야 함
        if "tables" in result:
            assert len(result["tables"]) > 0

            # 첫 번째 테이블 검증
            table = result["tables"][0]
            assert "page" in table
            assert "table_index" in table
            assert "data" in table
            assert "confidence" in table
            assert 0.0 <= table["confidence"] <= 1.0

    def test_extract_with_page_range(self):
        """페이지 범위 지정 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"page_range": (0, 1), "extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)

        assert len(result["pages"]) == 1
        assert result["pages"][0]["page"] == 0

    def test_extract_with_max_pages(self):
        """최대 페이지 수 제한 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"max_pages": 1, "extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)

        assert len(result["pages"]) <= 1

    def test_extract_metadata(self):
        """PDF 메타데이터 추출 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)
        metadata = result["metadata"]

        assert "total_pages" in metadata
        assert "engine" in metadata
        assert "processing_time" in metadata
        assert "file_path" in metadata
        assert "file_size" in metadata
        assert metadata["processing_time"] >= 0

    def test_extract_with_layout_preserve(self):
        """레이아웃 보존 텍스트 추출 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"pdfplumber_layout": True, "extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)

        assert "pages" in result
        assert len(result["pages"]) > 0

    def test_table_confidence_calculation(self):
        """테이블 신뢰도 계산 테스트"""
        if not TABLES_PDF.exists():
            pytest.skip(f"Test fixture not found: {TABLES_PDF}")

        engine = PDFPlumberEngine()
        config = {"extract_tables": True}

        result = engine.extract(TABLES_PDF, config)

        if "tables" in result and len(result["tables"]) > 0:
            for table in result["tables"]:
                assert "confidence" in table
                assert 0.0 <= table["confidence"] <= 1.0

    def test_invalid_pdf_path(self):
        """존재하지 않는 PDF 파일 테스트"""
        engine = PDFPlumberEngine()
        config = {}

        with pytest.raises(FileNotFoundError):
            engine.extract("/nonexistent/file.pdf", config)

    def test_page_dimensions(self):
        """페이지 크기 정보 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PDFPlumberEngine()
        config = {"extract_tables": False}

        result = engine.extract(SIMPLE_PDF, config)
        page = result["pages"][0]

        assert "width" in page
        assert "height" in page
        assert page["width"] >= 0
        assert page["height"] >= 0
