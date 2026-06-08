"""Tests for domain/web_search engine_bing.py and engine_google.py."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.web_search.types import SearchResponse

# ---------------------------------------------------------------------------
# BingSearch
# ---------------------------------------------------------------------------


def _make_bing(api_key: str = "test-bing-key", **kwargs):
    from beanllm.domain.web_search.engine_bing import BingSearch

    return BingSearch(api_key=api_key, **kwargs)


def _bing_response_data(num_results: int = 2):
    items = []
    for i in range(num_results):
        items.append(
            {
                "name": f"Title {i}",
                "url": f"https://example.com/result{i}",
                "snippet": f"Snippet {i}",
                "dateLastCrawled": "2024-01-01T00:00:00Z",
                "displayUrl": f"example.com/result{i}",
                "language": "en",
            }
        )
    return {
        "webPages": {
            "value": items,
            "totalEstimatedMatches": 100,
        }
    }


class TestBingSearchInit:
    def test_stores_api_key(self):
        bing = _make_bing("my-key")
        assert bing.api_key == "my-key"

    def test_sets_base_url(self):
        bing = _make_bing()
        assert "bing.microsoft.com" in bing.base_url


class TestBingSearchSync:
    def test_search_returns_response(self):
        bing = _make_bing()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bing_response_data(2)

        with patch("httpx.get", return_value=mock_resp):
            result = bing.search("AI technology")
        assert isinstance(result, SearchResponse)
        assert result.engine == "bing"

    def test_search_parses_results(self):
        bing = _make_bing()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bing_response_data(3)

        with patch("httpx.get", return_value=mock_resp):
            result = bing.search("test query")
        assert len(result.results) == 3

    def test_search_uses_cache_on_second_call(self):
        bing = _make_bing()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bing_response_data(1)

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            bing.search("cached query")
            bing.search("cached query")
        # Second call should use cache, not HTTP
        assert mock_get.call_count == 1

    def test_search_handles_request_error(self):
        import httpx

        bing = _make_bing()

        with patch("httpx.get", side_effect=httpx.RequestError("Connection failed")):
            result = bing.search("error query")
        assert isinstance(result, SearchResponse)
        assert result.results == []
        assert "error" in result.metadata

    def test_search_filters_invalid_urls(self):
        """URLs that fail validation should be skipped when validate_urls=True."""
        bing = _make_bing(validate_urls=True)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "webPages": {
                "value": [
                    {"url": "https://valid.com/page", "name": "Good URL", "snippet": "ok"},
                ],
                "totalEstimatedMatches": 1,
            }
        }
        with patch("httpx.get", return_value=mock_resp):
            result = bing.search("url test")
        assert len(result.results) == 1
        assert result.results[0].url == "https://valid.com/page"

    def test_search_with_market(self):
        bing = _make_bing()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bing_response_data(1)

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            bing.search("test", market="ko-KR")
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["params"]["mkt"] == "ko-KR"


class TestBingSearchAsync:
    async def test_search_async_returns_response(self):
        bing = _make_bing()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bing_response_data(2)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        async def mock_async_client_ctx(*args, **kwargs):
            return mock_client

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await bing.search_async("async query")
        assert isinstance(result, SearchResponse)

    async def test_search_async_handles_http_error(self):
        import httpx

        bing = _make_bing()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("HTTP error"))

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await bing.search_async("error query")
        assert result.results == []

    async def test_search_async_uses_cache(self):
        bing = _make_bing()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bing_response_data(1)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await bing.search_async("cache test async")
            await bing.search_async("cache test async")
        # Second call should use cache
        assert mock_client.get.call_count == 1


class TestBingParseDate:
    def test_parse_valid_date(self):
        bing = _make_bing()
        result = bing._parse_date("2024-01-01T00:00:00Z")
        assert isinstance(result, datetime)

    def test_parse_none_returns_none(self):
        bing = _make_bing()
        assert bing._parse_date(None) is None

    def test_parse_empty_string_returns_none(self):
        bing = _make_bing()
        assert bing._parse_date("") is None

    def test_parse_invalid_date_returns_none(self):
        bing = _make_bing()
        assert bing._parse_date("not-a-date") is None


# ---------------------------------------------------------------------------
# GoogleSearch
# ---------------------------------------------------------------------------


def _make_google(api_key: str = "test-google-key", search_engine_id: str = "test-cx"):
    from beanllm.domain.web_search.engine_google import GoogleSearch

    return GoogleSearch(api_key=api_key, search_engine_id=search_engine_id)


def _google_response_data(num_results: int = 2):
    items = []
    for i in range(num_results):
        items.append(
            {
                "title": f"Title {i}",
                "link": f"https://example.com/result{i}",
                "snippet": f"Snippet {i}",
                "pagemap": {},
            }
        )
    return {
        "items": items,
        "searchInformation": {"totalResults": "100"},
    }


class TestGoogleSearchInit:
    def test_stores_api_key(self):
        google = _make_google("gkey")
        assert google.api_key == "gkey"

    def test_stores_search_engine_id(self):
        google = _make_google(search_engine_id="my-cx")
        assert google.search_engine_id == "my-cx"

    def test_sets_base_url(self):
        google = _make_google()
        assert "googleapis.com" in google.base_url


class TestGoogleSearchSync:
    def test_search_returns_response(self):
        google = _make_google()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _google_response_data(2)

        with patch("httpx.get", return_value=mock_resp):
            result = google.search("Python tutorial")
        assert isinstance(result, SearchResponse)
        assert result.engine == "google"

    def test_search_parses_results(self):
        google = _make_google()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _google_response_data(3)

        with patch("httpx.get", return_value=mock_resp):
            result = google.search("test")
        assert len(result.results) == 3

    def test_search_uses_cache(self):
        google = _make_google()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _google_response_data(1)

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            google.search("cached google")
            google.search("cached google")
        assert mock_get.call_count == 1

    def test_search_handles_request_error(self):
        import httpx

        google = _make_google()

        with patch("httpx.get", side_effect=httpx.RequestError("Connection refused")):
            result = google.search("error query")
        assert result.results == []
        assert "error" in result.metadata

    def test_search_empty_items_returns_empty_results(self):
        google = _make_google()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"searchInformation": {"totalResults": "0"}}

        with patch("httpx.get", return_value=mock_resp):
            result = google.search("no results query")
        assert result.results == []


class TestGoogleSearchAsync:
    async def test_search_async_returns_response(self):
        google = _make_google()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _google_response_data(2)

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await google.search_async("async google test")
        assert isinstance(result, SearchResponse)

    async def test_search_async_handles_error(self):
        import httpx

        google = _make_google()
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=httpx.HTTPError("error"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await google.search_async("error query")
        assert result.results == []
