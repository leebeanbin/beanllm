"""
Comprehensive tests for HTMLLoader.
Target: src/beanllm/domain/loaders/core/html.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Hello World</h1>
<p>This is a test paragraph with enough content to pass the minimum length check.</p>
<p>Another paragraph with more text to ensure proper extraction works fine here.</p>
</body>
</html>"""

SHORT_HTML = "<html><body><p>Hi</p></body></html>"


def _make_loader(source="test.html", **kwargs):
    from beanllm.domain.loaders.core.html import HTMLLoader

    return HTMLLoader(source=source, **kwargs)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestHTMLLoaderInit:
    def test_default_init_local_file(self):
        loader = _make_loader("page.html")
        assert loader.source == "page.html"
        assert loader.encoding == "utf-8"
        assert loader.is_url is False
        assert loader.fallback_chain == ["trafilatura", "readability", "beautifulsoup"]

    def test_url_detection_http(self):
        loader = _make_loader("http://example.com/page")
        assert loader.is_url is True

    def test_url_detection_https(self):
        loader = _make_loader("https://example.com/page")
        assert loader.is_url is True

    def test_path_object_source(self):
        loader = _make_loader(Path("some/file.html"))
        assert loader.is_url is False

    def test_custom_fallback_chain(self):
        loader = _make_loader("file.html", fallback_chain=["beautifulsoup"])
        assert loader.fallback_chain == ["beautifulsoup"]

    def test_custom_encoding(self):
        loader = _make_loader("file.html", encoding="latin-1")
        assert loader.encoding == "latin-1"

    def test_headers_stored(self):
        loader = _make_loader("https://x.com", headers={"User-Agent": "Bot"})
        assert loader.headers == {"User-Agent": "Bot"}

    def test_timeout_stored(self):
        loader = _make_loader("https://x.com", timeout=30)
        assert loader.timeout == 30


# ---------------------------------------------------------------------------
# _read_file tests
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_file_returns_content(self, tmp_path):
        html_file = tmp_path / "page.html"
        html_file.write_text(SIMPLE_HTML, encoding="utf-8")
        loader = _make_loader(str(html_file))
        content = loader._read_file()
        assert "Hello World" in content

    def test_read_file_with_encoding(self, tmp_path):
        html_file = tmp_path / "page.html"
        html_file.write_text(SIMPLE_HTML, encoding="utf-8")
        loader = _make_loader(str(html_file), encoding="utf-8")
        content = loader._read_file()
        assert len(content) > 0


# ---------------------------------------------------------------------------
# _parse_with_beautifulsoup tests
# ---------------------------------------------------------------------------


class TestParseWithBeautifulSoup:
    def test_extracts_text(self):
        mock_bs4 = MagicMock()
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Hello World\nThis is a test paragraph."
        mock_soup.return_value = mock_soup
        mock_soup.__call__ = MagicMock(return_value=[])  # for soup(["script", ...])
        mock_bs4.BeautifulSoup = MagicMock(return_value=mock_soup)

        # Patch out the tag decompose loop
        mock_soup.return_value.__call__ = MagicMock(return_value=[])

        with patch.dict(sys.modules, {"bs4": mock_bs4}):
            from importlib import reload

            import beanllm.domain.loaders.core.html as html_mod

            loader = html_mod.HTMLLoader("file.html")
            # Call with direct bs4 mock
            loader2 = html_mod.HTMLLoader("file.html")

        try:
            from bs4 import BeautifulSoup

            loader3 = _make_loader("file.html")
            result = loader3._parse_with_beautifulsoup(SIMPLE_HTML)
            assert "Hello World" in result or len(result) > 0
        except ImportError:
            pytest.skip("bs4 not installed")

    def test_removes_script_tags(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        html_with_script = """<html><body>
<script>var x = 1;</script>
<p>Visible text content here that is long enough to extract.</p>
</body></html>"""
        loader = _make_loader("file.html")
        result = loader._parse_with_beautifulsoup(html_with_script)
        assert "var x" not in result
        assert "Visible text" in result

    def test_import_error_raises(self):
        with patch.dict(sys.modules, {"bs4": None}):
            loader = _make_loader("file.html")
            with pytest.raises((ImportError, TypeError)):
                loader._parse_with_beautifulsoup(SIMPLE_HTML)


# ---------------------------------------------------------------------------
# _parse_html fallback chain tests
# ---------------------------------------------------------------------------


class TestParseHtmlFallbackChain:
    def test_beautifulsoup_only_chain(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        loader = _make_loader("file.html", fallback_chain=["beautifulsoup"])
        text, parser = loader._parse_html(SIMPLE_HTML)
        assert parser == "beautifulsoup"
        assert len(text) > 0

    def test_trafilatura_success(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(
            return_value="This is a long enough text extracted by trafilatura to pass the minimum check."
        )
        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html", fallback_chain=["trafilatura"])
            text, parser = loader._parse_html(SIMPLE_HTML)
            assert parser == "trafilatura"

    def test_trafilatura_returns_none_falls_to_next(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(return_value=None)
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html", fallback_chain=["trafilatura", "beautifulsoup"])
            text, parser = loader._parse_html(SIMPLE_HTML)
            assert parser == "beautifulsoup"

    def test_trafilatura_returns_short_text_falls_to_next(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(return_value="short")  # < 50 chars
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html", fallback_chain=["trafilatura", "beautifulsoup"])
            text, parser = loader._parse_html(SIMPLE_HTML)
            assert parser == "beautifulsoup"

    def test_parser_exception_falls_to_next(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(side_effect=RuntimeError("crash"))

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html", fallback_chain=["trafilatura", "beautifulsoup"])
            text, parser = loader._parse_html(SIMPLE_HTML)
            assert parser == "beautifulsoup"

    def test_readability_success(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        mock_readability_doc = MagicMock()
        mock_readability_doc.summary.return_value = (
            "<p>" + "Readability extracted long content here. " * 5 + "</p>"
        )
        mock_readability = MagicMock()
        mock_readability.Document = MagicMock(return_value=mock_readability_doc)

        with patch.dict(sys.modules, {"readability": mock_readability}):
            loader = _make_loader("file.html", fallback_chain=["readability"])
            text, parser = loader._parse_html(SIMPLE_HTML)
            assert parser == "readability"

    def test_readability_missing_falls_to_beautifulsoup(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        loader = _make_loader("file.html", fallback_chain=["readability", "beautifulsoup"])
        # readability not installed -> ImportError -> fall through
        with patch.dict(sys.modules, {"readability": None}):
            text, parser = loader._parse_html(SIMPLE_HTML)
            assert parser == "beautifulsoup"

    def test_all_parsers_fail_uses_beautifulsoup_anyway(self):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        loader = _make_loader("file.html", fallback_chain=["trafilatura", "readability"])

        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura, "readability": None}):
            text, parser = loader._parse_html(SIMPLE_HTML)
            # Falls to fallback beautifulsoup
            assert text is not None


# ---------------------------------------------------------------------------
# _extract_metadata_trafilatura tests
# ---------------------------------------------------------------------------


class TestExtractMetadataTrafilatura:
    def test_extracts_metadata(self):
        mock_meta = MagicMock()
        mock_meta.title = "Test Title"
        mock_meta.author = "Test Author"
        mock_meta.date = "2024-01-01"
        mock_meta.description = "A description"
        mock_meta.sitename = "TestSite"

        mock_trafilatura = MagicMock()
        mock_trafilatura.extract_metadata = MagicMock(return_value=mock_meta)

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html")
            meta = loader._extract_metadata_trafilatura(SIMPLE_HTML)
            assert meta["title"] == "Test Title"
            assert meta["author"] == "Test Author"
            assert meta["date"] == "2024-01-01"

    def test_metadata_none_values_become_empty_string(self):
        mock_meta = MagicMock()
        mock_meta.title = None
        mock_meta.author = None
        mock_meta.date = None
        mock_meta.description = None
        mock_meta.sitename = None

        mock_trafilatura = MagicMock()
        mock_trafilatura.extract_metadata = MagicMock(return_value=mock_meta)

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html")
            meta = loader._extract_metadata_trafilatura(SIMPLE_HTML)
            assert meta["title"] == ""

    def test_trafilatura_missing_returns_empty(self):
        with patch.dict(sys.modules, {"trafilatura": None}):
            loader = _make_loader("file.html")
            meta = loader._extract_metadata_trafilatura(SIMPLE_HTML)
            assert meta == {}

    def test_metadata_returns_none_returns_empty(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract_metadata = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html")
            meta = loader._extract_metadata_trafilatura(SIMPLE_HTML)
            assert meta == {}

    def test_exception_in_metadata_returns_empty(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract_metadata = MagicMock(side_effect=RuntimeError("boom"))

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html")
            meta = loader._extract_metadata_trafilatura(SIMPLE_HTML)
            assert meta == {}


# ---------------------------------------------------------------------------
# _fetch_url tests
# ---------------------------------------------------------------------------


class TestFetchUrl:
    def test_fetch_url_returns_html(self):
        mock_response = MagicMock()
        mock_response.text = SIMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_response.charset_encoding = "utf-8"

        mock_httpx = MagicMock()
        mock_httpx.get = MagicMock(return_value=mock_response)

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            loader = _make_loader("https://example.com/page")
            content = loader._fetch_url()
            assert content == SIMPLE_HTML

    def test_fetch_url_raises_on_error(self):
        mock_httpx = MagicMock()
        mock_httpx.get = MagicMock(side_effect=Exception("Connection refused"))

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            loader = _make_loader("https://example.com/page")
            with pytest.raises(Exception):
                loader._fetch_url()

    def test_fetch_url_raises_when_httpx_missing(self):
        with patch.dict(sys.modules, {"httpx": None}):
            loader = _make_loader("https://example.com/page")
            with pytest.raises((ImportError, TypeError)):
                loader._fetch_url()


# ---------------------------------------------------------------------------
# load() integration tests
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_local_file(self, tmp_path):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        html_file = tmp_path / "page.html"
        html_file.write_text(SIMPLE_HTML, encoding="utf-8")
        loader = _make_loader(str(html_file), fallback_chain=["beautifulsoup"])
        docs = loader.load()
        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["type"] == "file"
        assert doc.metadata["parser"] == "beautifulsoup"

    def test_load_returns_document_with_metadata(self, tmp_path):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        html_file = tmp_path / "page.html"
        html_file.write_text(SIMPLE_HTML, encoding="utf-8")
        loader = _make_loader(str(html_file), fallback_chain=["beautifulsoup"])
        docs = loader.load()
        assert "source" in docs[0].metadata

    def test_load_url_with_trafilatura(self):
        mock_response = MagicMock()
        mock_response.text = SIMPLE_HTML
        mock_response.raise_for_status = MagicMock()
        mock_response.charset_encoding = "utf-8"

        mock_httpx = MagicMock()
        mock_httpx.get = MagicMock(return_value=mock_response)

        mock_trafilatura = MagicMock()
        long_text = "This is a long article content extracted by trafilatura. " * 5
        mock_trafilatura.extract = MagicMock(return_value=long_text)
        mock_meta = MagicMock()
        mock_meta.title = "Article"
        mock_meta.author = "Author"
        mock_meta.date = "2024-01-01"
        mock_meta.description = "Desc"
        mock_meta.sitename = "Site"
        mock_trafilatura.extract_metadata = MagicMock(return_value=mock_meta)

        with patch.dict(sys.modules, {"httpx": mock_httpx, "trafilatura": mock_trafilatura}):
            loader = _make_loader("https://example.com/article", fallback_chain=["trafilatura"])
            docs = loader.load()
            assert len(docs) == 1
            assert docs[0].metadata["type"] == "url"
            assert docs[0].metadata["parser"] == "trafilatura"
            assert docs[0].metadata["title"] == "Article"

    def test_load_raises_on_exception(self, tmp_path):
        loader = _make_loader(str(tmp_path / "nonexistent.html"))
        with pytest.raises(Exception):
            loader.load()

    def test_lazy_load(self, tmp_path):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pytest.skip("bs4 not installed")

        html_file = tmp_path / "page.html"
        html_file.write_text(SIMPLE_HTML, encoding="utf-8")
        loader = _make_loader(str(html_file), fallback_chain=["beautifulsoup"])
        docs = list(loader.lazy_load())
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# _parse_with_trafilatura
# ---------------------------------------------------------------------------


class TestParseWithTrafilatura:
    def test_extracts_text(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(return_value="Extracted content here.")

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html")
            result = loader._parse_with_trafilatura(SIMPLE_HTML)
            assert result == "Extracted content here."

    def test_returns_empty_when_none(self):
        mock_trafilatura = MagicMock()
        mock_trafilatura.extract = MagicMock(return_value=None)

        with patch.dict(sys.modules, {"trafilatura": mock_trafilatura}):
            loader = _make_loader("file.html")
            result = loader._parse_with_trafilatura(SIMPLE_HTML)
            assert result == ""

    def test_raises_when_missing(self):
        with patch.dict(sys.modules, {"trafilatura": None}):
            loader = _make_loader("file.html")
            with pytest.raises((ImportError, TypeError)):
                loader._parse_with_trafilatura(SIMPLE_HTML)
