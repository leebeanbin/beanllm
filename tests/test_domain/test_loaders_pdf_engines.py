"""
PDF Engine 및 관련 Loader 테스트

PyMuPDFEngine, TextLoader (core), DoclingLoader에 대한 단위 테스트.
모든 외부 의존성(fitz/PyMuPDF, docling)은 Mock으로 대체합니다.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF

    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    import docling

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


# ---------------------------------------------------------------------------
# PyMuPDFEngine Tests
# ---------------------------------------------------------------------------


class TestPyMuPDFEngineInit:
    """PyMuPDFEngine 초기화 테스트"""

    def test_init_with_mocked_fitz(self):
        """fitz 없이 엔진 생성 가능한지 확인 (mock)"""
        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            assert engine.name == "PyMuPDF"

    def test_init_with_custom_name(self):
        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine(name="CustomPDF")
            assert engine.name == "CustomPDF"

    def test_get_engine_info(self):
        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            info = engine.get_engine_info()
            assert info["name"] == "PyMuPDF"
            assert info["class"] == "PyMuPDFEngine"

    def test_check_dependencies_raises_without_fitz(self):
        """fitz가 없으면 ImportError 발생"""
        # Remove fitz from modules
        orig = sys.modules.pop("fitz", None)
        try:
            # Also remove the engine module so it reimports
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            with patch.dict("sys.modules", {"fitz": None}):
                with pytest.raises((ImportError, SystemError)):
                    from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

                    PyMuPDFEngine()
        except Exception:
            pass  # Expected
        finally:
            if orig is not None:
                sys.modules["fitz"] = orig


class TestPyMuPDFEngineValidatePath:
    """_validate_pdf_path 테스트"""

    def setup_method(self):
        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            # reimport fresh
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            self.engine = PyMuPDFEngine.__new__(PyMuPDFEngine)
            self.engine.name = "PyMuPDF"

        # Import base class method directly
        from beanllm.domain.loaders.pdf.engines.base import BasePDFEngine

        self.engine._validate_pdf_path = BasePDFEngine._validate_pdf_path.__get__(
            self.engine, type(self.engine)
        )

    def test_validate_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            self.engine._validate_pdf_path("/nonexistent/file.pdf")

    def test_validate_returns_path_object(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        result = self.engine._validate_pdf_path(pdf_file)
        assert isinstance(result, Path)
        assert result == pdf_file


class TestPyMuPDFEngineExtract:
    """PyMuPDFEngine.extract() 테스트"""

    def _make_engine(self, mock_fitz_module):
        """Helper to create PyMuPDFEngine with mocked fitz"""
        with patch.dict("sys.modules", {"fitz": mock_fitz_module}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
        return engine

    def _make_mock_page(self, text="Sample text", page_num=0):
        mock_page = MagicMock()
        mock_page.get_text.return_value = text
        mock_page.rotation = 0
        mock_rect = MagicMock()
        mock_rect.width = 595.0
        mock_rect.height = 842.0
        mock_page.rect = mock_rect
        mock_page.get_fonts.return_value = []
        mock_page.get_links.return_value = []
        mock_page.get_images.return_value = []
        return mock_page

    def _make_mock_doc(self, pages, metadata=None):
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=len(pages))
        mock_doc.__getitem__ = Mock(side_effect=lambda i: pages[i])
        mock_doc.metadata = metadata or {
            "title": "Test",
            "author": "",
            "subject": "",
            "creator": "",
        }
        return mock_doc

    def test_extract_single_page(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_fitz = MagicMock()
        mock_page = self._make_mock_page("Hello World")
        mock_doc = self._make_mock_doc([mock_page])
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {})

        assert "pages" in result
        assert "metadata" in result
        assert len(result["pages"]) == 1
        assert result["pages"][0]["text"] == "Hello World"
        assert result["metadata"]["total_pages"] == 1
        assert result["metadata"]["engine"] == "PyMuPDF"

    def test_extract_multiple_pages(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        pages = [self._make_mock_page(f"Page {i}") for i in range(3)]
        mock_doc = self._make_mock_doc(pages)
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {})

        assert len(result["pages"]) == 3
        assert result["pages"][2]["text"] == "Page 2"

    def test_extract_with_max_pages(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        pages = [self._make_mock_page(f"Page {i}") for i in range(10)]
        mock_doc = self._make_mock_doc(pages)
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {"max_pages": 3})

        assert len(result["pages"]) == 3

    def test_extract_with_page_range(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        pages = [self._make_mock_page(f"Page {i}") for i in range(10)]
        mock_doc = self._make_mock_doc(pages)
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {"page_range": (2, 5)})

        assert len(result["pages"]) == 3  # pages 2, 3, 4

    def test_extract_file_not_found(self, tmp_path):
        mock_fitz = MagicMock()

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            with pytest.raises(FileNotFoundError):
                engine.extract("/nonexistent/file.pdf", {})

    def test_extract_with_dict_text_mode(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        mock_page = self._make_mock_page()
        text_dict = {
            "blocks": [
                {
                    "lines": [
                        {"spans": [{"text": "Hello"}, {"text": " World"}]},
                    ]
                }
            ]
        }
        mock_page.get_text.side_effect = (
            lambda mode="text": text_dict if mode == "dict" else "Hello World"
        )
        mock_doc = self._make_mock_doc([mock_page])
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {"pymupdf_text_mode": "dict"})

        assert len(result["pages"]) == 1
        assert "structured_text" in result["pages"][0]

    def test_extract_with_layout_analysis(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        mock_page = self._make_mock_page()
        text_dict = {"blocks": []}
        mock_page.get_text.side_effect = lambda mode="text": text_dict if mode == "dict" else ""
        mock_page.get_fonts.return_value = [(1, "Type1", "Helvetica", "Helvetica-Bold", 0, 0, 0)]
        mock_page.get_links.return_value = [{"uri": "https://example.com", "page": -1, "kind": 2}]
        mock_doc = self._make_mock_doc([mock_page])
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {"layout_analysis": True})

        assert len(result["pages"]) == 1
        page_meta = result["pages"][0]["metadata"]
        assert "fonts" in page_meta
        assert "links" in page_meta

    def test_extract_with_images(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        mock_page = self._make_mock_page()

        # Mock image extraction
        mock_img = (1, "ext", "type", "name")
        mock_page.get_images.return_value = [mock_img]

        mock_bbox = MagicMock()
        mock_bbox.x0, mock_bbox.y0, mock_bbox.x1, mock_bbox.y1 = 0, 0, 100, 100
        mock_page.get_image_bbox.return_value = mock_bbox

        base_image = {
            "ext": "png",
            "width": 100,
            "height": 100,
            "image": b"fake",
            "colorspace": "RGB",
            "bpc": 8,
        }
        mock_page.parent = MagicMock()
        mock_page.parent.extract_image.return_value = base_image

        mock_doc = self._make_mock_doc([mock_page])
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {"extract_images": True})

        assert "images" in result
        assert len(result["images"]) == 1
        assert result["images"][0]["format"] == "png"

    def test_extract_metadata_keys(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        mock_page = self._make_mock_page("Text")
        doc_metadata = {
            "title": "My PDF",
            "author": "Author Name",
            "subject": "Test",
            "creator": "Adobe",
        }
        mock_doc = self._make_mock_doc([mock_page], metadata=doc_metadata)
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            result = engine.extract(pdf_file, {})

        meta = result["metadata"]
        assert meta["title"] == "My PDF"
        assert meta["author"] == "Author Name"
        assert "processing_time" in meta
        assert "file_size" in meta
        assert "file_path" in meta

    def test_extract_raises_on_fitz_error(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        mock_fitz.open.side_effect = RuntimeError("Corrupted PDF")

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            with pytest.raises(RuntimeError, match="Corrupted PDF"):
                engine.extract(pdf_file, {})


class TestPyMuPDFEngineExtractStreaming:
    """PyMuPDFEngine.extract_streaming() 테스트"""

    def _make_mock_page(self, text="Sample text"):
        mock_page = MagicMock()
        mock_page.get_text.return_value = text
        mock_page.rotation = 0
        mock_rect = MagicMock()
        mock_rect.width = 595.0
        mock_rect.height = 842.0
        mock_page.rect = mock_rect
        mock_page.get_fonts.return_value = []
        mock_page.get_links.return_value = []
        mock_page.get_images.return_value = []
        return mock_page

    def test_extract_streaming_yields_pages(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        pages = [self._make_mock_page(f"Page {i}") for i in range(3)]
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_doc.__getitem__ = Mock(side_effect=lambda i: pages[i])
        mock_doc.metadata = {}
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            results = list(engine.extract_streaming(pdf_file, {}))

        assert len(results) == 3
        assert results[0]["page"] == 0
        assert results[0]["text"] == "Page 0"

    def test_extract_streaming_with_max_pages(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        mock_fitz = MagicMock()
        pages = [self._make_mock_page(f"Page {i}") for i in range(10)]
        mock_doc = MagicMock()
        mock_doc.__len__ = Mock(return_value=10)
        mock_doc.__getitem__ = Mock(side_effect=lambda i: pages[i])
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            engine = PyMuPDFEngine()
            results = list(engine.extract_streaming(pdf_file, {"max_pages": 4}))

        assert len(results) == 4


class TestPyMuPDFEngineExtractTextFromDict:
    """_extract_text_from_dict 테스트"""

    def setup_method(self):
        mock_fitz = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            sys.modules.pop("beanllm.domain.loaders.pdf.engines.pymupdf_engine", None)
            from beanllm.domain.loaders.pdf.engines.pymupdf_engine import PyMuPDFEngine

            self.engine = PyMuPDFEngine()

    def test_extract_from_dict(self):
        text_dict = {
            "blocks": [
                {
                    "lines": [
                        {"spans": [{"text": "Hello"}, {"text": " World"}]},
                        {"spans": [{"text": "Line two"}]},
                    ]
                }
            ]
        }
        result = self.engine._extract_text_from_dict(text_dict)
        assert "Hello" in result
        assert "World" in result
        assert "Line two" in result

    def test_extract_from_empty_dict(self):
        result = self.engine._extract_text_from_dict({})
        assert result == ""

    def test_extract_from_dict_no_blocks(self):
        result = self.engine._extract_text_from_dict({"blocks": []})
        assert result == ""


# ---------------------------------------------------------------------------
# TextLoader Tests (core/text.py)
# ---------------------------------------------------------------------------


class TestTextLoaderInit:
    """TextLoader 초기화 테스트"""

    def test_basic_init(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file)
        assert loader.encoding == "utf-8"
        assert loader.autodetect_encoding is True

    def test_init_with_custom_encoding(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, encoding="latin-1", autodetect_encoding=False)
        assert loader.encoding == "latin-1"

    def test_init_without_validation(self, tmp_path):
        """validate_path=False이면 파일이 없어도 초기화 가능"""
        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader("/nonexistent/path.txt", validate_path=False)
        assert loader.file_path == Path("/nonexistent/path.txt")


class TestTextLoaderLoad:
    """TextLoader.load() 테스트"""

    def test_load_basic_file(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello World\nSecond line", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader
        from beanllm.domain.loaders.types import Document

        loader = TextLoader(txt_file)
        docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "Hello World" in docs[0].content
        assert "Second line" in docs[0].content

    def test_load_metadata(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file)
        docs = loader.load()

        assert "source" in docs[0].metadata
        assert "encoding" in docs[0].metadata

    def test_lazy_load_yields_document(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file)
        docs = list(loader.lazy_load())
        assert len(docs) == 1
        assert "Test content" in docs[0].content

    def test_load_without_mmap(self, tmp_path):
        """use_mmap=False 강제"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("No mmap content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, use_mmap=False)
        docs = loader.load()
        assert "No mmap content" in docs[0].content

    def test_should_use_mmap_false_for_small_file(self, tmp_path):
        txt_file = tmp_path / "small.txt"
        txt_file.write_text("small content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file)
        # Small file should not use mmap
        assert loader._should_use_mmap() is False

    def test_should_use_mmap_explicit_true(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, use_mmap=True)
        assert loader._should_use_mmap() is True

    def test_should_use_mmap_explicit_false(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, use_mmap=False)
        assert loader._should_use_mmap() is False

    def test_load_streaming_normal(self, tmp_path):
        """load_streaming 스트리밍 테스트 (일반 읽기)"""
        txt_file = tmp_path / "stream.txt"
        txt_file.write_text("A" * 100, encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, use_mmap=False, chunk_size=50)
        chunks = list(loader.load_streaming())
        assert len(chunks) == 2
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_decode_with_encoding_detection_utf8(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("UTF-8 Content", encoding="utf-8")

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, autodetect_encoding=False)
        data = b"Hello"
        result = loader._decode_with_encoding_detection(data)
        assert result == "Hello"

    def test_load_autodetect_encoding_fallback(self, tmp_path):
        """인코딩 자동 감지 fallback 테스트"""
        txt_file = tmp_path / "encoded.txt"
        # Write CP949 encoded content
        content = "한글 테스트"
        txt_file.write_bytes(content.encode("latin-1", errors="replace"))

        from beanllm.domain.loaders.core.text import TextLoader

        loader = TextLoader(txt_file, encoding="latin-1", autodetect_encoding=False)
        docs = loader.load()
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# DoclingLoader Tests
# ---------------------------------------------------------------------------


def _make_docling_loader(file_path, **kwargs):
    """DoclingLoader는 abstract이므로 concrete 서브클래스를 사용한다."""
    from beanllm.domain.loaders.advanced.docling_loader import DoclingLoader

    class _ConcreteDoclingLoader(DoclingLoader):
        def lazy_load(self, *args, **kwargs2):
            return iter([])

    return _ConcreteDoclingLoader(file_path=file_path, **kwargs)


class TestDoclingLoaderInit:
    """DoclingLoader 초기화 테스트"""

    def test_basic_init(self):
        loader = _make_docling_loader("test.pdf")
        assert loader.file_path == "test.pdf"
        assert loader.file_path == "test.pdf"
        assert loader.extract_tables is True
        assert loader.extract_images is False
        assert loader.ocr_enabled is False
        assert loader.output_format == "markdown"
        assert loader.include_metadata is True

    def test_init_with_all_options(self):
        loader = _make_docling_loader(
            "document.docx",
            extract_tables=False,
            extract_images=True,
            ocr_enabled=True,
            output_format="text",
            include_metadata=False,
        )
        assert loader.output_format == "text"
        assert loader.extract_images is True
        assert loader.ocr_enabled is True

    def test_init_raises_invalid_output_format(self):
        with pytest.raises(ValueError, match="Invalid output_format"):
            _make_docling_loader("test.pdf", output_format="invalid")

    def test_output_format_case_insensitive(self):
        loader = _make_docling_loader("test.pdf", output_format="MARKDOWN")
        assert loader.output_format == "markdown"


class TestDoclingLoaderLoad:
    """DoclingLoader.load() 테스트"""

    def test_load_raises_import_error_without_docling(self):
        loader = _make_docling_loader("test.pdf")
        with patch.dict(
            "sys.modules",
            {
                "docling": None,
                "docling.datamodel.base_models": None,
                "docling.document_converter": None,
            },
        ):
            with pytest.raises((ImportError, Exception)):
                loader.load()

    def test_load_raises_file_not_found(self):
        loader = _make_docling_loader("/nonexistent/file.pdf")

        mock_docling = MagicMock()
        mock_input_format = MagicMock()
        mock_converter_class = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "docling": mock_docling,
                "docling.datamodel": MagicMock(),
                "docling.datamodel.base_models": MagicMock(InputFormat=mock_input_format),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_class),
            },
        ):
            with pytest.raises(FileNotFoundError):
                loader.load()

    def test_load_returns_document_markdown(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        from beanllm.domain.loaders.types import Document

        loader = _make_docling_loader(str(pdf_file), output_format="markdown")

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Hello\nWorld"
        mock_result.document.export_to_text.return_value = "Hello World"
        mock_result.document.title = "Test Title"
        mock_result.document.author = None
        mock_result.document.num_pages = 5
        mock_result.document.creation_date = None
        mock_result.document.modification_date = None
        mock_result.document.tables = []
        mock_result.document.pictures = []

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class = MagicMock(return_value=mock_converter)

        with patch.dict(
            "sys.modules",
            {
                "docling": MagicMock(),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.base_models": MagicMock(),
                "docling.document_converter": MagicMock(DocumentConverter=mock_converter_class),
            },
        ):
            docs = loader.load()

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert "# Hello" in docs[0].content
        assert docs[0].metadata["source"] == str(pdf_file)

    def test_load_returns_document_text_format(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        loader = _make_docling_loader(str(pdf_file), output_format="text")

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# MD"
        mock_result.document.export_to_text.return_value = "Plain text output"
        mock_result.document.title = None
        mock_result.document.num_pages = 2

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.dict(
            "sys.modules",
            {
                "docling": MagicMock(),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.base_models": MagicMock(),
                "docling.document_converter": MagicMock(
                    DocumentConverter=MagicMock(return_value=mock_converter)
                ),
            },
        ):
            docs = loader.load()

        assert "Plain text output" in docs[0].content

    def test_load_without_metadata(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        loader = _make_docling_loader(str(pdf_file), include_metadata=False)

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Content"

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result

        with patch.dict(
            "sys.modules",
            {
                "docling": MagicMock(),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.base_models": MagicMock(),
                "docling.document_converter": MagicMock(
                    DocumentConverter=MagicMock(return_value=mock_converter)
                ),
            },
        ):
            docs = loader.load()

        # Even without metadata, source is always included
        assert docs[0].metadata["source"] == str(pdf_file)

    def test_load_propagates_converter_exception(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        loader = _make_docling_loader(str(pdf_file))

        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("Conversion failed")

        with patch.dict(
            "sys.modules",
            {
                "docling": MagicMock(),
                "docling.datamodel": MagicMock(),
                "docling.datamodel.base_models": MagicMock(),
                "docling.document_converter": MagicMock(
                    DocumentConverter=MagicMock(return_value=mock_converter)
                ),
            },
        ):
            with pytest.raises(RuntimeError, match="Conversion failed"):
                loader.load()


class TestDoclingLoaderExtractMetadata:
    """DoclingLoader._extract_metadata 테스트"""

    def test_extract_metadata_basic(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        loader = _make_docling_loader(str(pdf_file), extract_images=True)

        mock_result = MagicMock()
        mock_result.document.title = "My Document"
        mock_result.document.author = "Jane"
        mock_result.document.num_pages = 10
        mock_result.document.creation_date = None
        mock_result.document.modification_date = None
        mock_result.document.tables = [1, 2]
        mock_result.document.pictures = [1]

        metadata = loader._extract_metadata(mock_result)

        assert metadata["source"] == str(pdf_file)
        assert metadata["file_name"] == "test.pdf"
        assert metadata["file_type"] == ".pdf"
        assert metadata["loader"] == "DoclingLoader"
        assert metadata["output_format"] == "markdown"
        assert metadata["title"] == "My Document"
        assert metadata["num_pages"] == 10
        assert metadata["num_tables"] == 2
        assert metadata["num_images"] == 1

    def test_extract_metadata_handles_exceptions_gracefully(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        loader = _make_docling_loader(str(pdf_file))

        mock_result = MagicMock()
        # Make doc access raise
        mock_result.document = None

        # Should not raise, just return basic metadata
        metadata = loader._extract_metadata(mock_result)
        assert "source" in metadata
        assert metadata["loader"] == "DoclingLoader"


class TestDoclingLoaderLoadAndSplit:
    """DoclingLoader.load_and_split() 테스트"""

    def test_load_and_split_returns_chunks(self, tmp_path):
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        loader = _make_docling_loader(str(pdf_file))

        long_content = "Word " * 500
        from beanllm.domain.loaders.types import Document

        mock_doc = Document(content=long_content, metadata={"source": str(pdf_file)})

        # Mock splitter module so load_and_split actually splits
        mock_splitter_cls = MagicMock()
        mock_splitter = MagicMock()
        mock_splitter.split_text.return_value = ["chunk1 " * 30, "chunk2 " * 30, "chunk3 " * 30]
        mock_splitter_cls.return_value = mock_splitter
        mock_splitters_mod = MagicMock()
        mock_splitters_mod.RecursiveCharacterTextSplitter = mock_splitter_cls

        with patch.object(loader, "load", return_value=[mock_doc]):
            with patch.dict(
                "sys.modules", {"beanllm.domain.loaders.splitters": mock_splitters_mod}
            ):
                chunks = loader.load_and_split(chunk_size=200, chunk_overlap=20)
                assert len(chunks) >= 1
                assert all(isinstance(c, Document) for c in chunks)
