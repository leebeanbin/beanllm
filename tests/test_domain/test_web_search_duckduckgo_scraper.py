"""Tests for DuckDuckGoSearch and WebScraper."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.web_search.types import SearchResponse

# ---------------------------------------------------------------------------
# DuckDuckGoSearch
# ---------------------------------------------------------------------------


def _make_ddg(**kwargs):
    from beanllm.domain.web_search.engine_duckduckgo import DuckDuckGoSearch

    return DuckDuckGoSearch(**kwargs)


def _ddg_raw_results(num: int = 2):
    return [
        {
            "title": f"Result {i}",
            "href": f"https://example.com/result{i}",
            "body": f"Snippet {i}",
        }
        for i in range(num)
    ]


class TestDuckDuckGoInit:
    def test_init_sets_no_api_key(self):
        ddg = _make_ddg()
        assert ddg.api_key is None

    def test_init_accepts_max_results(self):
        ddg = _make_ddg(max_results=5)
        assert ddg.max_results == 5


class TestDuckDuckGoSearch:
    def test_search_returns_response_when_ddgs_available(self):
        ddg = _make_ddg()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs_instance.text.return_value = iter(_ddg_raw_results(2))

        with patch(
            "beanllm.domain.web_search.engine_duckduckgo.DDGS", return_value=mock_ddgs_instance
        ):
            result = ddg.search("AI news")
        assert isinstance(result, SearchResponse)
        assert result.engine == "duckduckgo"

    def test_search_parses_results(self):
        ddg = _make_ddg()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs_instance.text.return_value = iter(_ddg_raw_results(3))

        with patch(
            "beanllm.domain.web_search.engine_duckduckgo.DDGS", return_value=mock_ddgs_instance
        ):
            result = ddg.search("test query")
        assert len(result.results) == 3

    def test_search_uses_cache_on_second_call(self):
        ddg = _make_ddg()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs_instance.text.return_value = iter(_ddg_raw_results(1))

        with patch(
            "beanllm.domain.web_search.engine_duckduckgo.DDGS", return_value=mock_ddgs_instance
        ):
            ddg.search("cached query")
            mock_ddgs_instance.text.return_value = iter(_ddg_raw_results(1))
            ddg.search("cached query")
        # Only one DDGS call due to cache
        assert mock_ddgs_instance.text.call_count == 1

    def test_search_returns_empty_when_ddgs_not_installed(self):
        ddg = _make_ddg()
        with patch("beanllm.domain.web_search.engine_duckduckgo.DDGS", None):
            result = ddg.search("test")
        assert result.results == []
        assert "error" in result.metadata

    def test_search_handles_generic_exception(self):
        ddg = _make_ddg()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs_instance.text.side_effect = RuntimeError("Connection error")

        with patch(
            "beanllm.domain.web_search.engine_duckduckgo.DDGS", return_value=mock_ddgs_instance
        ):
            result = ddg.search("error query")
        assert result.results == []
        assert "error" in result.metadata

    def test_search_with_region(self):
        ddg = _make_ddg()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs_instance.text.return_value = iter(_ddg_raw_results(1))

        with patch(
            "beanllm.domain.web_search.engine_duckduckgo.DDGS", return_value=mock_ddgs_instance
        ):
            ddg.search("test", region="us-en")
        call_args = mock_ddgs_instance.text.call_args
        assert call_args.kwargs["region"] == "us-en"


class TestDuckDuckGoSearchAsync:
    async def test_search_async_returns_response(self):
        ddg = _make_ddg()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.__enter__ = MagicMock(return_value=mock_ddgs_instance)
        mock_ddgs_instance.__exit__ = MagicMock(return_value=None)
        mock_ddgs_instance.text.return_value = iter(_ddg_raw_results(2))

        with patch(
            "beanllm.domain.web_search.engine_duckduckgo.DDGS", return_value=mock_ddgs_instance
        ):
            result = await ddg.search_async("async ddg test")
        assert isinstance(result, SearchResponse)


# ---------------------------------------------------------------------------
# WebScraper
# ---------------------------------------------------------------------------


def _make_html_response(title: str = "Test Page", body_text: str = "Hello World"):
    html = f"""<html>
<head><title>{title}</title></head>
<body>
<p>{body_text}</p>
<script>alert('remove me');</script>
<a href="https://link1.com">Link 1</a>
<a href="https://link2.com">Link 2</a>
</body>
</html>"""
    return html.encode("utf-8")


class TestWebScraperScrape:
    def test_scrape_returns_dict_with_expected_keys(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response("My Title", "Some text")
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = WebScraper.scrape("https://example.com")

        assert "title" in result
        assert "text" in result
        assert "links" in result
        assert "metadata" in result

    def test_scrape_extracts_title(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response("Page Title")
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = WebScraper.scrape("https://example.com")

        assert result["title"] == "Page Title"

    def test_scrape_extracts_links(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = WebScraper.scrape("https://example.com")

        assert "https://link1.com" in result["links"]
        assert "https://link2.com" in result["links"]

    def test_scrape_without_validation(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch("httpx.get", return_value=mock_resp):
            result = WebScraper.scrape("https://example.com", validate=False)

        assert "title" in result

    def test_scrape_handles_exception(self):
        from beanllm.domain.web_search.scraper import WebScraper

        with patch("httpx.get", side_effect=Exception("Connection failed")):
            result = WebScraper.scrape("https://example.com", validate=False)

        assert result["title"] == ""
        assert result["text"] == ""
        assert result["links"] == []
        assert "error" in result["metadata"]

    def test_scrape_metadata_contains_url(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = WebScraper.scrape("https://example.com")

        assert result["metadata"]["url"] == "https://example.com"
        assert result["metadata"]["status_code"] == 200

    def test_scrape_removes_scripts_from_text(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response(body_text="Main content")
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.get", return_value=mock_resp),
        ):
            result = WebScraper.scrape("https://example.com")

        assert "alert" not in result["text"]


class TestWebScraperScrapeAsync:
    async def test_scrape_async_returns_dict(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.content = _make_html_response("Async Page")
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await WebScraper.scrape_async("https://example.com")

        assert "title" in result
        assert result["title"] == "Async Page"

    async def test_scrape_async_handles_exception(self):
        from beanllm.domain.web_search.scraper import WebScraper

        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Async error"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "beanllm.domain.web_search.scraper.validate_url", return_value="https://example.com"
            ),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await WebScraper.scrape_async("https://example.com")

        assert "error" in result["metadata"]
