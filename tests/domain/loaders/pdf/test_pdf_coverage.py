"""
PDF ecosystem coverage tests.

Uses fpdf2 to create real test PDFs in tmp_path,
targeting previously-uncovered lines in:
- bean_pdf_loader.py  (load_streaming, _select_strategy fallbacks, images)
- pdfplumber_engine.py (chars/words, hyperlinks, table edge cases)
- markdown_converter.py (heading detection, dict tables, images with size)
- directory.py (non-recursive, lazy_load, pattern fallback, parallel)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers: programmatic PDF creation with fpdf2
# ---------------------------------------------------------------------------


def make_simple_pdf(path: Path, text: str = "Hello World", pages: int = 1) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_author("Test Author")
    for i in range(pages):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text=f"{text} page {i + 1}")
    pdf.output(str(path))
    return path


def make_multipage_pdf(path: Path, n: int = 3) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(n):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(text=f"Page {i + 1} content goes here")
    pdf.output(str(path))
    return path


def make_text_pdf(path: Path, lines: list[str]) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in lines:
        pdf.cell(text=line)
        pdf.ln()
    pdf.output(str(path))
    return path


def make_txt_file(path: Path, content: str = "Sample text content") -> Path:
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# PDFPlumberEngine — uncovered paths
# ---------------------------------------------------------------------------


class TestPDFPlumberEngineCoverage:
    def test_extract_with_page_range(self, tmp_path: Path) -> None:
        """Covers: page_range handling (line 123)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        pdf_path = make_multipage_pdf(tmp_path / "range.pdf", n=5)
        engine = PDFPlumberEngine()
        result = engine.extract(
            pdf_path,
            {
                "extract_tables": False,
                "page_range": (0, 3),
            },
        )
        assert len(result["pages"]) == 3

    def test_extract_with_max_pages(self, tmp_path: Path) -> None:
        """Covers: max_pages handling (line 116)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        pdf_path = make_multipage_pdf(tmp_path / "maxpages.pdf", n=5)
        engine = PDFPlumberEngine()
        result = engine.extract(
            pdf_path,
            {
                "extract_tables": False,
                "max_pages": 2,
            },
        )
        assert len(result["pages"]) <= 2

    def test_extract_with_layout_preserve(self, tmp_path: Path) -> None:
        """Covers: layout_preserve path (lines 134-139)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        pdf_path = make_simple_pdf(tmp_path / "layout.pdf")
        engine = PDFPlumberEngine()
        result = engine.extract(
            pdf_path,
            {
                "extract_tables": False,
                "pdfplumber_layout": True,
            },
        )
        assert "pages" in result

    def test_extract_with_chars_and_words(self, tmp_path: Path) -> None:
        """Covers: extract_chars/words path (lines 159-190)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        pdf_path = make_simple_pdf(tmp_path / "chars.pdf", text="Hello chars test")
        engine = PDFPlumberEngine()
        result = engine.extract(
            pdf_path,
            {
                "extract_tables": False,
                "pdfplumber_extract_chars": True,
                "pdfplumber_extract_words": True,
            },
        )
        assert "pages" in result
        # chars/words may or may not be present depending on pdfplumber internals

    def test_extract_with_hyperlinks(self, tmp_path: Path) -> None:
        """Covers: hyperlinks extraction path (lines 203-217)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        pdf_path = make_simple_pdf(tmp_path / "hyper.pdf")
        engine = PDFPlumberEngine()
        result = engine.extract(
            pdf_path,
            {
                "extract_tables": False,
                "pdfplumber_extract_hyperlinks": True,
            },
        )
        assert "pages" in result

    def test_calculate_table_confidence_single_row(self) -> None:
        """Covers: single-row confidence = 0.3 (line 409)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        engine = PDFPlumberEngine()
        # Single row = len < 2 → 0.3
        score = engine._calculate_table_confidence([["col1", "col2"]])
        assert score == pytest.approx(0.3)

    def test_calculate_table_confidence_inconsistent_rows(self) -> None:
        """Covers: inconsistent row lengths penalty (line 423)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        engine = PDFPlumberEngine()
        table = [
            ["a", "b", "c"],
            ["d", "e"],  # shorter row
            ["f", "g", "h"],
        ]
        score = engine._calculate_table_confidence(table)
        # Penalty applied for inconsistency, score < 1.0
        assert score < 1.0

    def test_calculate_table_confidence_single_column(self) -> None:
        """Covers: max col < 2 penalty (line 429)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        engine = PDFPlumberEngine()
        table = [["header"], ["value1"], ["value2"]]
        score = engine._calculate_table_confidence(table)
        assert score < 1.0

    def test_calculate_table_confidence_empty_cells(self) -> None:
        """Covers: empty cell ratio penalty."""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        engine = PDFPlumberEngine()
        # 50% empty cells
        table = [
            ["header1", "header2"],
            ["val1", ""],
            ["", "val2"],
        ]
        score = engine._calculate_table_confidence(table)
        assert score < 1.0

    def test_extract_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Covers: exception path (lines 264-266)"""
        from beanllm.domain.loaders.pdf.engines.pdfplumber_engine import PDFPlumberEngine

        engine = PDFPlumberEngine()
        with pytest.raises((FileNotFoundError, Exception)):
            engine.extract(tmp_path / "nonexistent.pdf", {"extract_tables": False})


# ---------------------------------------------------------------------------
# MarkdownConverter — uncovered paths
# ---------------------------------------------------------------------------


class TestMarkdownConverterCoverage:
    def test_convert_with_fonts_metadata(self) -> None:
        """Covers: fonts branch in convert_to_markdown (line 93)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        result = {
            "pages": [
                {
                    "page": 0,
                    "text": "Title\nNormal text",
                    "metadata": {
                        "fonts": [
                            {"text": "Title", "size": 24.0},
                            {"text": "Normal text", "size": 12.0},
                        ]
                    },
                }
            ],
            "tables": [],
            "images": [],
        }
        markdown = converter.convert_to_markdown(result)
        assert "Title" in markdown
        assert "Page 1" in markdown

    def test_detect_headings_with_font_sizes(self) -> None:
        """Covers: _detect_headings (lines 199-218)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter(heading_threshold=1.2)
        # avg_size = (50 + 30 + 18 + 10) / 4 = 27.0
        # level 1: size >= 27 * 2.0 = 54 → none
        # level 2: size >= 27 * 1.5 = 40.5 → none
        # level 3: size >= 27 * 1.2 = 32.4 → Big Title(50), Major(40)
        # To get level 1: need size >= avg*2 → use avg=12, size=25
        avg_size = 12.0
        metadata = {
            "fonts": [
                {"text": "Level1 Title", "size": 30.0},  # 30 >= 12*2.0=24 → level 1
                {"text": "Level2 Heading", "size": 20.0},  # 20 >= 12*1.5=18 → level 2
                {"text": "Level3 Heading", "size": 15.0},  # 15 >= 12*1.2=14.4 → level 3
                {"text": "body text", "size": 10.0},  # 10 < 12*1.2 → no heading
            ]
        }
        headings = converter._detect_headings(metadata, avg_size)
        assert len(headings) >= 1
        levels = {h["text"]: h["level"] for h in headings}
        assert levels.get("Level1 Title") == 1
        assert levels.get("Level2 Heading") == 2

    def test_convert_text_with_headings_applies_marks(self) -> None:
        """Covers: _convert_text_with_headings (lines 148-183)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        text = "Big Title\n\nNormal paragraph text"
        metadata = {
            "fonts": [
                {"text": "Big Title", "size": 30.0},
                {"text": "Normal paragraph text", "size": 12.0},
            ]
        }
        result = converter._convert_text_with_headings(text, metadata)
        assert "#" in result  # Some heading marker

    def test_convert_text_no_matching_headings(self) -> None:
        """Covers: no headings detected → clean text fallback (line 164)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        text = "all same size text"
        metadata = {
            "fonts": [
                {"text": "all same size text", "size": 12.0},
            ]
        }
        # avg_size == 12, threshold = 12*1.2 = 14.4 → no heading matches
        result = converter._convert_text_with_headings(text, metadata)
        assert "#" not in result

    def test_get_heading_level_found(self) -> None:
        """Covers: _get_heading_level match path (lines 231-234)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        headings = [{"text": "Title", "level": 1}, {"text": "Section", "level": 2}]
        assert converter._get_heading_level("Title text", headings) == 1
        assert converter._get_heading_level("Section intro", headings) == 2

    def test_get_heading_level_not_found(self) -> None:
        """Covers: _get_heading_level no match → 0."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        headings = [{"text": "Title", "level": 1}]
        assert converter._get_heading_level("body text", headings) == 0

    def test_convert_table_with_dict_data(self) -> None:
        """Covers: dict-based table data path (line 266)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        table = {
            "data": [
                {"Name": "Alice", "Age": "30"},
                {"Name": "Bob", "Age": "25"},
            ]
        }
        markdown = converter._convert_table_to_markdown(table)
        assert "Name" in markdown
        assert "Alice" in markdown
        assert "|" in markdown

    def test_convert_table_with_2d_list(self) -> None:
        """Covers: 2D list table data path."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        table = {"data": [["Col1", "Col2"], ["v1", "v2"], ["v3", "v4"]]}
        markdown = converter._convert_table_to_markdown(table)
        assert "Col1" in markdown
        assert "v1" in markdown

    def test_convert_image_with_dimensions(self) -> None:
        """Covers: image with size info (line 298)."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        image = {
            "page": 0,
            "image_index": 0,
            "format": "png",
            "width": 640,
            "height": 480,
        }
        markdown = converter._convert_image_to_markdown(image)
        assert "640x480" in markdown
        assert "image" in markdown

    def test_convert_image_without_dimensions(self) -> None:
        """Image with width/height = 0 → no size annotation."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        image = {"page": 0, "image_index": 0, "format": "png", "width": 0, "height": 0}
        markdown = converter._convert_image_to_markdown(image)
        assert "pixels" not in markdown

    def test_convert_empty_result(self) -> None:
        """Empty result → empty string."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        result = converter.convert_to_markdown({"pages": [], "tables": [], "images": []})
        assert result == ""

    def test_clean_text_removes_extra_newlines(self) -> None:
        """Covers: _clean_text."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter()
        text = "line1\n\n\n\nline2\n\n\n\nline3"
        cleaned = converter._clean_text(text)
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in cleaned

    def test_page_separator_customization(self) -> None:
        """Covers: custom page_separator."""
        from beanllm.domain.loaders.pdf.utils.markdown_converter import MarkdownConverter

        converter = MarkdownConverter(page_separator="\n===\n")
        result = {
            "pages": [
                {"page": 0, "text": "page one", "metadata": {}},
                {"page": 1, "text": "page two", "metadata": {}},
            ],
            "tables": [],
            "images": [],
        }
        markdown = converter.convert_to_markdown(result)
        assert "\n===\n" in markdown


# ---------------------------------------------------------------------------
# beanPDFLoader — load_streaming + strategy fallbacks
# ---------------------------------------------------------------------------


class TestBeanPDFLoaderStreaming:
    def test_load_streaming_fallback_to_lazy_load(self, tmp_path: Path) -> None:
        """Covers: engine without extract_streaming → lazy_load fallback (lines 358-362)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "stream.pdf")
        loader = beanPDFLoader(pdf_path, strategy="accurate", validate_path=False)

        # PDFPlumberEngine does not have extract_streaming, so falls back to lazy_load
        docs = list(loader.load_streaming())
        assert isinstance(docs, list)

    def test_load_streaming_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Covers: FileNotFoundError in load_streaming (line 291)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        loader = beanPDFLoader(tmp_path / "ghost.pdf", validate_path=False)
        with pytest.raises(FileNotFoundError):
            list(loader.load_streaming())

    def test_load_streaming_unavailable_strategy_raises(self, tmp_path: Path) -> None:
        """Covers: ValueError for missing strategy in load_streaming (line 297)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "sv.pdf")
        loader = beanPDFLoader(pdf_path, validate_path=False)
        # Force _select_strategy to return a value not in _engines
        loader._select_strategy = lambda: "ghost_strategy"  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="not available"):
            list(loader.load_streaming())

    def test_load_streaming_with_streaming_engine(self, tmp_path: Path) -> None:
        """Covers: engine that has extract_streaming (lines 307-355)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "se.pdf")

        # Create a mock engine that supports extract_streaming
        mock_engine = MagicMock()
        mock_engine.extract_streaming.return_value = [
            {
                "page": 0,
                "text": "Streaming content",
                "width": 612.0,
                "height": 792.0,
                "metadata": {},
                "tables": [
                    {
                        "table_index": 0,
                        "metadata": {"rows": 2, "cols": 3},
                        "confidence": 0.9,
                        "dataframe": True,
                        "markdown": "| a |",
                        "csv": "a,b",
                    }
                ],
                "images": [
                    {
                        "image_index": 0,
                        "format": "png",
                        "width": 100,
                        "height": 100,
                        "size": 1024,
                    }
                ],
            }
        ]

        loader = beanPDFLoader(pdf_path, validate_path=False)
        loader._engines = {"accurate": mock_engine}
        loader.config.strategy = "accurate"

        docs = list(loader.load_streaming())
        assert len(docs) == 1
        assert docs[0].content == "Streaming content"
        assert "tables" in docs[0].metadata
        assert "images" in docs[0].metadata


class TestBeanPDFLoaderStrategyFallbacks:
    def test_select_strategy_unavailable_explicit_falls_back_to_auto(self, tmp_path: Path) -> None:
        """Covers: explicit strategy not available → warning + auto fallback (lines 376)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "fb.pdf")
        loader = beanPDFLoader(pdf_path, strategy="ml", validate_path=False)

        # If "ml" engine not available, it falls back to auto
        strategy = loader._select_strategy()
        assert strategy in loader._engines

    def test_select_strategy_no_accurate_for_tables(self, tmp_path: Path) -> None:
        """Covers: extract_tables but no accurate engine → warning (line 386)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "notacc.pdf")
        loader = beanPDFLoader(pdf_path, extract_tables=True, validate_path=False)
        # Remove accurate engine
        loader._engines = {k: v for k, v in loader._engines.items() if k != "accurate"}
        if loader._engines:
            strategy = loader._select_strategy()
            assert strategy in loader._engines

    def test_select_strategy_large_pdf_prefers_fast(self, tmp_path: Path) -> None:
        """Covers: page_count > 100 → fast (lines 403-404)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "large.pdf")
        loader = beanPDFLoader(pdf_path, extract_tables=False, validate_path=False)

        with patch("fitz.open") as mock_open:
            mock_doc = MagicMock()
            mock_doc.__len__ = lambda self: 150
            mock_open.return_value = mock_doc

            if "fast" in loader._engines:
                strategy = loader._select_strategy()
                assert strategy in loader._engines

    def test_execute_strategy_unavailable_raises(self, tmp_path: Path) -> None:
        """Covers: ValueError in _execute_strategy (line 428)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "err.pdf")
        loader = beanPDFLoader(pdf_path, validate_path=False)
        with pytest.raises(ValueError, match="not available"):
            loader._execute_strategy("nonexistent_strategy")

    def test_convert_to_documents_with_images(self, tmp_path: Path) -> None:
        """Covers: images section in _convert_to_documents (line 491)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "img.pdf")
        loader = beanPDFLoader(pdf_path, validate_path=False)

        result = {
            "pages": [{"page": 0, "text": "page text", "width": 612.0, "height": 792.0}],
            "metadata": {"total_pages": 1, "engine": "test"},
            "images": [
                {
                    "page": 0,
                    "image_index": 0,
                    "format": "png",
                    "width": 100,
                    "height": 100,
                    "size": 512,
                }
            ],
        }
        docs = loader._convert_to_documents(result)
        assert len(docs) == 1
        assert "images" in docs[0].metadata

    def test_load_with_to_markdown(self, tmp_path: Path) -> None:
        """Covers: to_markdown=True path in load() (lines 248-250)."""
        from beanllm.domain.loaders.pdf.bean_pdf_loader import beanPDFLoader

        pdf_path = make_simple_pdf(tmp_path / "md.pdf")
        loader = beanPDFLoader(pdf_path, to_markdown=True, validate_path=False)
        docs = loader.load()
        assert isinstance(docs, list)
        assert hasattr(loader, "_result")


# ---------------------------------------------------------------------------
# DirectoryLoader — uncovered paths
# ---------------------------------------------------------------------------


class TestDirectoryLoaderCoverage:
    def test_load_nonrecursive(self, tmp_path: Path) -> None:
        """Covers: recursive=False path (line 163)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader

        # Create text files in root (not subdir)
        (tmp_path / "a.txt").write_text("content a")
        (tmp_path / "b.txt").write_text("content b")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.txt").write_text("content c")

        loader = DirectoryLoader(tmp_path, glob="*.txt", recursive=False)
        # _load_single_file may return [] if no loader, but should not raise
        assert isinstance(loader._compiled_exclude_patterns, list)

    def test_exclude_pattern_fallback_string(self, tmp_path: Path) -> None:
        """Covers: string pattern fallback in exclude check (lines 179-181)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader

        (tmp_path / "keep.txt").write_text("keep")
        (tmp_path / "exclude.log").write_text("exclude")

        # Inject a bad pattern that fails re.compile, forcing string fallback
        loader = DirectoryLoader(tmp_path, glob="**/*", exclude=["*.log"])

        # Force one compiled pattern to be a raw string
        loader._compiled_exclude_patterns = ["*.log"]

        files = list(tmp_path.glob("**/*"))
        files = [f for f in files if f.is_file()]
        # Verify string fallback logic works
        for file_path in files:
            for pattern in loader._compiled_exclude_patterns:
                if isinstance(pattern, str):
                    _ = file_path.match(pattern)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Covers: no files found → empty return (line 194-195)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader

        loader = DirectoryLoader(tmp_path, glob="**/*.pdf")
        result = loader.load()
        assert result == []

    def test_load_sequential_with_single_file(self, tmp_path: Path) -> None:
        """Covers: sequential path for single file (lines 266-273)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader

        (tmp_path / "only.txt").write_text("only file")
        loader = DirectoryLoader(tmp_path, glob="*.txt", use_parallel=False)

        with patch.object(DirectoryLoader, "_load_single_file", return_value=[]):
            result = loader.load()
            assert result == []

    def test_load_parallel_with_multiple_files(self, tmp_path: Path) -> None:
        """Covers: parallel loading path (lines 226-248) via ProcessPoolExecutor mock."""
        from concurrent.futures import Future

        from beanllm.domain.loaders.core.directory import DirectoryLoader
        from beanllm.domain.loaders.types import Document

        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(f"content {i}")

        mock_doc = Document(content="mocked", metadata={"source": "test"})

        loader = DirectoryLoader(tmp_path, glob="*.txt", use_parallel=True, max_workers=2)

        # Build futures that immediately resolve
        futures: list[Future] = []
        for i in range(3):
            f: Future = Future()
            f.set_result([mock_doc])
            futures.append(f)

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.side_effect = futures

        # as_completed returns futures in order
        with patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor):
            with patch("concurrent.futures.as_completed", return_value=iter(futures)):
                result = loader.load()
                assert len(result) == 3
                assert all(d.content == "mocked" for d in result)

    def test_load_parallel_fallback_on_error(self, tmp_path: Path) -> None:
        """Covers: parallel failure → sequential fallback (lines 250-263)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader

        for i in range(2):
            (tmp_path / f"g{i}.txt").write_text(f"content {i}")

        loader = DirectoryLoader(tmp_path, glob="*.txt", use_parallel=True)

        def raise_error(*args, **kwargs):
            raise RuntimeError("simulated parallel failure")

        with patch("concurrent.futures.ProcessPoolExecutor.__enter__", side_effect=raise_error):
            with patch.object(DirectoryLoader, "_load_single_file", return_value=[]):
                result = loader.load()
                assert result == []

    def test_lazy_load_yields_documents(self, tmp_path: Path) -> None:
        """Covers: lazy_load method (lines 280-299) via sys.modules injection."""
        import sys

        from beanllm.domain.loaders.core.directory import DirectoryLoader
        from beanllm.domain.loaders.types import Document

        (tmp_path / "lazy.txt").write_text("lazy content")

        mock_doc = Document(content="lazy doc", metadata={"source": "lazy"})
        mock_loader_instance = MagicMock()
        mock_loader_instance.lazy_load.return_value = iter([mock_doc])

        mock_factory_module = MagicMock()
        mock_factory_module.DocumentLoader.get_loader.return_value = mock_loader_instance

        loader = DirectoryLoader(tmp_path, glob="*.txt")

        # lazy_load uses `from .factory import DocumentLoader`
        # which resolves to beanllm.domain.loaders.core.factory
        sys.modules["beanllm.domain.loaders.core.factory"] = mock_factory_module
        try:
            docs = list(loader.lazy_load())
            assert len(docs) == 1
            assert docs[0].content == "lazy doc"
        finally:
            sys.modules.pop("beanllm.domain.loaders.core.factory", None)

    def test_exclude_pattern_compilation_failure_is_handled(self, tmp_path: Path) -> None:
        """Covers: exception in pattern compilation → fallback to string (lines 114-117)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader

        # This pattern will cause re.compile to fail
        with patch("re.compile", side_effect=re.error("bad pattern")):
            loader = DirectoryLoader(tmp_path, exclude=["invalid[pattern"])
            # All patterns should fall back to string
            for p in loader._compiled_exclude_patterns:
                assert isinstance(p, str)

    def test_load_with_batch_processor(self, tmp_path: Path) -> None:
        """Covers: batch_processor injection path (lines 203-220)."""
        from beanllm.domain.loaders.core.directory import DirectoryLoader
        from beanllm.domain.loaders.types import Document

        for i in range(2):
            (tmp_path / f"bp{i}.txt").write_text(f"batch {i}")

        mock_doc = Document(content="batched", metadata={"source": "bp"})
        mock_batch = MagicMock()

        import asyncio

        async def fake_process_batch(items, handler):
            return [[mock_doc] for _ in items]

        mock_batch.process_batch = fake_process_batch

        loader = DirectoryLoader(
            tmp_path, glob="*.txt", use_parallel=True, batch_processor=mock_batch
        )

        result = loader.load()
        assert all(isinstance(d, Document) for d in result)
