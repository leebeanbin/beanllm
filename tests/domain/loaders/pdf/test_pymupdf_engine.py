"""
PyMuPDFEngine 테스트
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

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


# ---------------------------------------------------------------------------
# Helper: create real PDFs via fpdf2
# ---------------------------------------------------------------------------


def _make_pdf(path: Path, pages: int = 1, text_prefix: str = "Page") -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(pages):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text=f"{text_prefix} {i + 1}")
    pdf.output(str(path))
    return path


# ---------------------------------------------------------------------------
# Tests using real fpdf2-created PDFs
# ---------------------------------------------------------------------------


class TestExtractWithRealPDFs:
    @pytest.fixture
    def engine(self):
        return PyMuPDFEngine()

    def test_single_page_returns_one_page(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "s.pdf", pages=1)
        result = engine.extract(pdf, {})
        assert len(result["pages"]) == 1

    def test_multipage_returns_all(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "m.pdf", pages=4)
        result = engine.extract(pdf, {})
        assert len(result["pages"]) == 4

    def test_page_numbering_is_zero_based(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "z.pdf", pages=2)
        result = engine.extract(pdf, {})
        assert result["pages"][0]["page"] == 0
        assert result["pages"][1]["page"] == 1

    def test_metadata_keys(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "meta.pdf", pages=1)
        result = engine.extract(pdf, {})
        for key in ("total_pages", "engine", "processing_time", "file_path", "file_size"):
            assert key in result["metadata"], f"missing key: {key}"

    def test_page_range_limits(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "pr.pdf", pages=5)
        result = engine.extract(pdf, {"page_range": (1, 3)})
        assert len(result["pages"]) == 2
        assert result["pages"][0]["page"] == 1

    def test_page_range_clamped_to_total(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "clamp.pdf", pages=3)
        result = engine.extract(pdf, {"page_range": (0, 100)})
        assert len(result["pages"]) == 3

    def test_max_pages_limits(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "mp.pdf", pages=5)
        result = engine.extract(pdf, {"max_pages": 2})
        assert len(result["pages"]) == 2

    def test_max_pages_larger_than_pdf(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "mpbig.pdf", pages=2)
        result = engine.extract(pdf, {"max_pages": 100})
        assert len(result["pages"]) == 2

    def test_dict_text_mode_returns_structured_text(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "dict.pdf", pages=1)
        result = engine.extract(pdf, {"pymupdf_text_mode": "dict"})
        assert "structured_text" in result["pages"][0]

    def test_layout_analysis_switches_to_dict_mode(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "la.pdf", pages=1)
        result = engine.extract(pdf, {"layout_analysis": True})
        assert "structured_text" in result["pages"][0]

    def test_html_text_mode(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "html.pdf", pages=1)
        result = engine.extract(pdf, {"pymupdf_text_mode": "html"})
        assert isinstance(result["pages"][0]["text"], str)
        assert "structured_text" not in result["pages"][0]

    def test_xml_text_mode(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "xml.pdf", pages=1)
        result = engine.extract(pdf, {"pymupdf_text_mode": "xml"})
        assert isinstance(result["pages"][0]["text"], str)

    def test_extract_fonts_flag(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "fonts.pdf", pages=1)
        # Should not raise; font info may or may not be present
        result = engine.extract(pdf, {"pymupdf_extract_fonts": True})
        assert len(result["pages"]) == 1

    def test_extract_links_flag(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "links.pdf", pages=1)
        result = engine.extract(pdf, {"pymupdf_extract_links": True})
        assert len(result["pages"]) == 1

    def test_extract_images_false_no_images_key(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "noimg.pdf", pages=1)
        result = engine.extract(pdf, {"extract_images": False})
        assert "images" not in result

    def test_extract_images_true_with_mocked_extractor(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "img.pdf", pages=1)
        fake = [{"page": 0, "format": "png", "width": 10, "height": 10}]
        with patch.object(engine, "_extract_images_from_page", return_value=fake):
            result = engine.extract(pdf, {"extract_images": True})
        assert "images" in result
        assert result["images"] == fake

    def test_extract_images_empty_no_images_key(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "emptyimg.pdf", pages=1)
        with patch.object(engine, "_extract_images_from_page", return_value=[]):
            result = engine.extract(pdf, {"extract_images": True})
        assert "images" not in result

    def test_accepts_str_path(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "str.pdf", pages=1)
        result = engine.extract(str(pdf), {})
        assert len(result["pages"]) == 1

    def test_file_not_found_raises(self, engine, tmp_path):
        with pytest.raises(FileNotFoundError):
            engine.extract(tmp_path / "nonexistent.pdf", {})


class TestStreamingWithRealPDFs:
    @pytest.fixture
    def engine(self):
        return PyMuPDFEngine()

    def test_streaming_yields_all_pages(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "s.pdf", pages=3)
        pages = list(engine.extract_streaming(pdf, {}))
        assert len(pages) == 3

    def test_streaming_page_structure(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "str.pdf", pages=1)
        pages = list(engine.extract_streaming(pdf, {}))
        for key in ("page", "text", "width", "height", "metadata"):
            assert key in pages[0], f"missing key: {key}"

    def test_streaming_page_range(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "pr.pdf", pages=5)
        pages = list(engine.extract_streaming(pdf, {"page_range": (1, 3)}))
        assert len(pages) == 2
        assert pages[0]["page"] == 1

    def test_streaming_max_pages(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "mp.pdf", pages=5)
        pages = list(engine.extract_streaming(pdf, {"max_pages": 2}))
        assert len(pages) == 2

    def test_streaming_dict_mode(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "dict.pdf", pages=1)
        pages = list(engine.extract_streaming(pdf, {"pymupdf_text_mode": "dict"}))
        assert "structured_text" in pages[0]

    def test_streaming_layout_analysis(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "la.pdf", pages=1)
        pages = list(engine.extract_streaming(pdf, {"layout_analysis": True}))
        assert "structured_text" in pages[0]

    def test_streaming_html_mode(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "html.pdf", pages=1)
        pages = list(engine.extract_streaming(pdf, {"pymupdf_text_mode": "html"}))
        assert isinstance(pages[0]["text"], str)

    def test_streaming_fonts(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "fonts.pdf", pages=1)
        pages = list(engine.extract_streaming(pdf, {"pymupdf_extract_fonts": True}))
        assert len(pages) == 1

    def test_streaming_links(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "links.pdf", pages=1)
        pages = list(engine.extract_streaming(pdf, {"pymupdf_extract_links": True}))
        assert len(pages) == 1

    def test_streaming_extract_images_with_mock(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "img.pdf", pages=1)
        fake = [{"page": 0, "format": "png"}]
        with patch.object(engine, "_extract_images_streaming", return_value=fake):
            pages = list(engine.extract_streaming(pdf, {"extract_images": True}))
        assert "images" in pages[0]

    def test_streaming_no_images_key_when_empty(self, engine, tmp_path):
        pdf = _make_pdf(tmp_path / "noimg.pdf", pages=1)
        with patch.object(engine, "_extract_images_streaming", return_value=[]):
            pages = list(engine.extract_streaming(pdf, {"extract_images": True}))
        assert "images" not in pages[0]

    def test_streaming_file_not_found(self, engine, tmp_path):
        with pytest.raises(FileNotFoundError):
            list(engine.extract_streaming(tmp_path / "ghost.pdf", {}))


class TestImageExtraction:
    @pytest.fixture
    def engine(self):
        return PyMuPDFEngine()

    def test_extract_images_from_page_happy_path(self, engine):
        mock_image = {
            "ext": "png",
            "width": 200,
            "height": 150,
            "image": b"x" * 500,
            "colorspace": "DeviceRGB",
            "bpc": 8,
        }
        mock_bbox = MagicMock()
        mock_bbox.x0, mock_bbox.y0, mock_bbox.x1, mock_bbox.y1 = 10.0, 20.0, 210.0, 170.0
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(1,)]
        mock_page.get_image_bbox.return_value = mock_bbox
        mock_page.parent.extract_image.return_value = mock_image

        images = engine._extract_images_from_page(mock_page, page_num=0)

        assert len(images) == 1
        assert images[0]["format"] == "png"
        assert images[0]["bbox"] == (10.0, 20.0, 210.0, 170.0)
        assert images[0]["metadata"]["bpc"] == 8

    def test_extract_images_from_page_bbox_fallback(self, engine):
        mock_image = {
            "ext": "jpeg",
            "width": 100,
            "height": 80,
            "image": b"y" * 200,
            "colorspace": "",
            "bpc": 8,
        }
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(2,)]
        mock_page.get_image_bbox.side_effect = RuntimeError("no bbox")
        mock_page.parent.extract_image.return_value = mock_image

        images = engine._extract_images_from_page(mock_page, page_num=1)
        assert images[0]["bbox"] == (0.0, 0.0, 100.0, 80.0)

    def test_extract_images_from_page_get_images_fails(self, engine):
        mock_page = MagicMock()
        mock_page.get_images.side_effect = RuntimeError("crash")
        images = engine._extract_images_from_page(mock_page, page_num=0)
        assert images == []

    def test_extract_images_streaming_happy_path(self, engine):
        mock_image = {
            "ext": "png",
            "width": 50,
            "height": 40,
            "image": b"z" * 100,
        }
        mock_bbox = MagicMock()
        mock_bbox.x0, mock_bbox.y0, mock_bbox.x1, mock_bbox.y1 = 0.0, 0.0, 50.0, 40.0
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(5,)]
        mock_page.get_image_bbox.return_value = mock_bbox
        mock_page.parent.extract_image.return_value = mock_image

        images = engine._extract_images_streaming(mock_page, page_num=2)
        assert len(images) == 1
        assert images[0]["page"] == 2

    def test_extract_images_streaming_bbox_fallback(self, engine):
        mock_image = {
            "ext": "jpeg",
            "width": 80,
            "height": 60,
            "image": b"w" * 80,
        }
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(3,)]
        mock_page.get_image_bbox.side_effect = RuntimeError("no bbox")
        mock_page.parent.extract_image.return_value = mock_image

        images = engine._extract_images_streaming(mock_page, page_num=0)
        assert images[0]["bbox"] == (0.0, 0.0, 80.0, 60.0)

    def test_extract_images_streaming_extract_image_fails_skips(self, engine):
        mock_page = MagicMock()
        mock_page.get_images.return_value = [(7,)]
        mock_page.parent.extract_image.side_effect = RuntimeError("extract fail")

        images = engine._extract_images_streaming(mock_page, page_num=0)
        assert images == []

    def test_extract_images_streaming_get_images_fails(self, engine):
        mock_page = MagicMock()
        mock_page.get_images.side_effect = RuntimeError("crash")
        images = engine._extract_images_streaming(mock_page, page_num=0)
        assert images == []


class TestExtractTextFromDict:
    @pytest.fixture
    def engine(self):
        return PyMuPDFEngine()

    def test_extracts_from_valid_structure(self, engine):
        d = {"blocks": [{"lines": [{"spans": [{"text": "Hello"}, {"text": " World"}]}]}]}
        result = engine._extract_text_from_dict(d)
        assert "Hello" in result
        assert "World" in result

    def test_empty_dict_returns_empty_string(self, engine):
        assert engine._extract_text_from_dict({}) == ""

    def test_block_without_lines_skipped(self, engine):
        d = {"blocks": [{"type": 1}]}
        assert engine._extract_text_from_dict(d) == ""

    def test_line_without_spans_skipped(self, engine):
        d = {"blocks": [{"lines": [{"type": 0}]}]}
        assert engine._extract_text_from_dict(d) == ""

    def test_span_without_text_key_skipped(self, engine):
        d = {"blocks": [{"lines": [{"spans": [{"font": "Arial"}]}]}]}
        assert engine._extract_text_from_dict(d) == ""

    def test_multiblock_concatenation(self, engine):
        d = {
            "blocks": [
                {"lines": [{"spans": [{"text": "First"}]}]},
                {"lines": [{"spans": [{"text": "Second"}]}]},
            ]
        }
        result = engine._extract_text_from_dict(d)
        assert "First" in result
        assert "Second" in result
