"""
PyMuPDFEngine 테스트
"""

import pytest
from pathlib import Path
from src.beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine


# 테스트 픽스처 경로
FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures" / "pdf"
SIMPLE_PDF = FIXTURES_DIR / "simple.pdf"
IMAGES_PDF = FIXTURES_DIR / "images.pdf"


class TestPyMuPDFEngine:
    """PyMuPDFEngine 기본 테스트"""

    def test_engine_initialization(self):
        """엔진 초기화 테스트"""
        engine = PyMuPDFEngine()
        assert engine.name == "PyMuPDF"

    def test_engine_info(self):
        """엔진 정보 반환 테스트"""
        engine = PyMuPDFEngine()
        info = engine.get_engine_info()

        assert "name" in info
        assert "class" in info
        assert info["name"] == "PyMuPDF"
        assert info["class"] == "PyMuPDFEngine"

    def test_extract_simple_pdf(self):
        """간단한 PDF 추출 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {"extract_tables": False, "extract_images": False}

        result = engine.extract(SIMPLE_PDF, config)

        assert "pages" in result
        assert "metadata" in result
        assert len(result["pages"]) > 0
        assert result["metadata"]["total_pages"] >= 1
        assert result["metadata"]["engine"] == "PyMuPDF"

    def test_extract_with_text_content(self):
        """텍스트 내용 추출 확인"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {}

        result = engine.extract(SIMPLE_PDF, config)

        first_page = result["pages"][0]
        assert "text" in first_page
        assert len(first_page["text"]) > 0
        assert "beanPDFLoader" in first_page["text"]

    def test_extract_with_page_range(self):
        """페이지 범위 지정 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {"page_range": (0, 1)}  # 첫 번째 페이지만

        result = engine.extract(SIMPLE_PDF, config)

        assert len(result["pages"]) == 1
        assert result["pages"][0]["page"] == 0

    def test_extract_with_max_pages(self):
        """최대 페이지 수 제한 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {"max_pages": 1}

        result = engine.extract(SIMPLE_PDF, config)

        assert len(result["pages"]) <= 1

    def test_extract_images(self):
        """이미지 추출 테스트"""
        if not IMAGES_PDF.exists():
            pytest.skip(f"Test fixture not found: {IMAGES_PDF}")

        engine = PyMuPDFEngine()
        config = {"extract_images": True}

        result = engine.extract(IMAGES_PDF, config)

        # images.pdf에는 그래픽 요소가 있을 수 있음
        assert "pages" in result
        # 이미지가 추출되었을 수도 있음 (그래픽 요소에 따라)
        if "images" in result:
            assert isinstance(result["images"], list)

    def test_extract_metadata(self):
        """PDF 메타데이터 추출 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {}

        result = engine.extract(SIMPLE_PDF, config)
        metadata = result["metadata"]

        assert "total_pages" in metadata
        assert "engine" in metadata
        assert "processing_time" in metadata
        assert "file_path" in metadata
        assert "file_size" in metadata
        assert metadata["processing_time"] >= 0

    def test_extract_with_layout_analysis(self):
        """레이아웃 분석 옵션 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {"layout_analysis": True}

        result = engine.extract(SIMPLE_PDF, config)

        assert "pages" in result
        assert len(result["pages"]) > 0

    def test_invalid_pdf_path(self):
        """존재하지 않는 PDF 파일 테스트"""
        engine = PyMuPDFEngine()
        config = {}

        with pytest.raises(FileNotFoundError):
            engine.extract("/nonexistent/file.pdf", config)

    def test_page_dimensions(self):
        """페이지 크기 정보 테스트"""
        if not SIMPLE_PDF.exists():
            pytest.skip(f"Test fixture not found: {SIMPLE_PDF}")

        engine = PyMuPDFEngine()
        config = {}

        result = engine.extract(SIMPLE_PDF, config)
        page = result["pages"][0]

        assert "width" in page
        assert "height" in page
        assert page["width"] > 0
        assert page["height"] > 0
