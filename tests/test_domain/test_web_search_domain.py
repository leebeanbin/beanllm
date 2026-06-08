"""Tests for domain/web_search: security, types, engines base."""

import time
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.web_search.engines import BaseSearchEngine, SearchEngine
from beanllm.domain.web_search.security import validate_url
from beanllm.domain.web_search.types import SearchResponse, SearchResult

# ---------------------------------------------------------------------------
# SearchResult / SearchResponse types
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_basic_creation(self):
        r = SearchResult(title="Hello", url="https://example.com", snippet="A snippet")
        assert r.title == "Hello"
        assert r.url == "https://example.com"
        assert r.snippet == "A snippet"
        assert r.source == "unknown"
        assert r.score == 0.0

    def test_with_all_fields(self):
        from datetime import datetime

        dt = datetime(2024, 1, 1)
        r = SearchResult(
            title="T",
            url="https://x.com",
            snippet="S",
            source="google",
            score=0.9,
            published_date=dt,
            metadata={"k": "v"},
        )
        assert r.source == "google"
        assert r.score == 0.9
        assert r.published_date == dt
        assert r.metadata["k"] == "v"

    def test_str_representation(self):
        r = SearchResult(title="My Title", url="https://x.com", snippet="Short", source="bing")
        s = str(r)
        assert "My Title" in s
        assert "bing" in s

    def test_frozen(self):
        r = SearchResult(title="T", url="https://x.com", snippet="S")
        with pytest.raises((AttributeError, TypeError)):
            r.title = "changed"  # type: ignore


class TestSearchResponse:
    def _make_result(self, n=1) -> SearchResult:
        return SearchResult(title=f"R{n}", url=f"https://r{n}.com", snippet=f"S{n}")

    def test_basic_creation(self):
        r = self._make_result()
        resp = SearchResponse(query="test", results=[r], engine="google")
        assert resp.query == "test"
        assert resp.engine == "google"
        assert len(resp) == 1

    def test_len_and_iter(self):
        results = [self._make_result(i) for i in range(3)]
        resp = SearchResponse(query="q", results=results)
        assert len(resp) == 3
        assert list(resp) == results

    def test_empty_results(self):
        resp = SearchResponse(query="q", results=[])
        assert len(resp) == 0

    def test_metadata_field(self):
        resp = SearchResponse(query="q", results=[], metadata={"pages": 10})
        assert resp.metadata["pages"] == 10

    def test_total_results_optional(self):
        resp = SearchResponse(query="q", results=[], total_results=1000)
        assert resp.total_results == 1000

    def test_search_time(self):
        resp = SearchResponse(query="q", results=[], search_time=0.42)
        assert resp.search_time == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# SearchEngine enum
# ---------------------------------------------------------------------------


class TestSearchEngineEnum:
    def test_values(self):
        assert SearchEngine.GOOGLE.value == "google"
        assert SearchEngine.BING.value == "bing"
        assert SearchEngine.DUCKDUCKGO.value == "duckduckgo"

    def test_all_members(self):
        members = {e.value for e in SearchEngine}
        assert "google" in members
        assert "bing" in members
        assert "duckduckgo" in members


# ---------------------------------------------------------------------------
# BaseSearchEngine cache helpers
# ---------------------------------------------------------------------------


class _ConcreteEngine(BaseSearchEngine):
    """Minimal concrete subclass for testing base class behaviour."""

    def search(self, query, **kwargs):
        return SearchResponse(query=query, results=[], engine="test")

    async def search_async(self, query, **kwargs):
        return SearchResponse(query=query, results=[], engine="test")


class TestBaseSearchEngineInit:
    def test_defaults(self):
        eng = _ConcreteEngine()
        assert eng.max_results == 10
        assert eng.timeout == 10
        assert eng.cache_ttl == 3600
        assert eng.validate_urls is False
        assert eng.api_key is None

    def test_custom_values(self):
        eng = _ConcreteEngine(
            api_key="k", max_results=5, timeout=30, cache_ttl=60, validate_urls=True
        )
        assert eng.api_key == "k"
        assert eng.max_results == 5
        assert eng.cache_ttl == 60
        assert eng.validate_urls is True


class TestBaseSearchEngineCache:
    def setup_method(self):
        self.eng = _ConcreteEngine(cache_ttl=60)

    def _make_response(self, query="q") -> SearchResponse:
        return SearchResponse(query=query, results=[], engine="test")

    def test_cache_miss_returns_none(self):
        assert self.eng._get_from_cache("missing") is None

    def test_cache_save_and_hit(self):
        resp = self._make_response("hello")
        self.eng._save_to_cache("hello", resp)
        result = self.eng._get_from_cache("hello")
        assert result is resp

    def test_cache_expiry(self):
        resp = self._make_response("expire")
        self.eng._save_to_cache("expire", resp)
        # Manually backdate the timestamp
        self.eng._cache["expire"] = (resp, time.time() - 9999)
        assert self.eng._get_from_cache("expire") is None

    def test_cache_different_keys(self):
        r1 = self._make_response("q1")
        r2 = self._make_response("q2")
        self.eng._save_to_cache("q1", r1)
        self.eng._save_to_cache("q2", r2)
        assert self.eng._get_from_cache("q1") is r1
        assert self.eng._get_from_cache("q2") is r2


class TestBaseSearchEngineValidateUrl:
    def setup_method(self):
        self.eng = _ConcreteEngine(validate_urls=True)
        self.eng_no_validate = _ConcreteEngine(validate_urls=False)

    def test_validate_url_public_domain(self):
        with patch(
            "beanllm.domain.web_search.engines.validate_url", return_value="https://example.com"
        ):
            result = self.eng._validate_result_url("https://example.com")
        assert result == "https://example.com"

    def test_validate_url_returns_none_on_error(self):
        with patch(
            "beanllm.domain.web_search.engines.validate_url", side_effect=ValueError("blocked")
        ):
            result = self.eng._validate_result_url("http://localhost")
        assert result is None

    def test_no_validate_returns_url_as_is(self):
        result = self.eng_no_validate._validate_result_url("http://anything")
        assert result == "http://anything"


# ---------------------------------------------------------------------------
# Security: validate_url
# ---------------------------------------------------------------------------


class TestValidateUrlScheme:
    def test_https_allowed(self):
        with patch("socket.getaddrinfo", return_value=[]):
            url = validate_url("https://example.com")
        assert url == "https://example.com"

    def test_http_allowed(self):
        with patch("socket.getaddrinfo", return_value=[]):
            url = validate_url("http://example.com")
        assert url == "http://example.com"

    def test_file_scheme_blocked(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("file:///etc/passwd")

    def test_ftp_scheme_blocked(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("ftp://example.com")

    def test_custom_allowed_scheme(self):
        with patch("socket.getaddrinfo", return_value=[]):
            url = validate_url("ftp://example.com", allowed_schemes=["ftp"])
        assert url == "ftp://example.com"


class TestValidateUrlHostname:
    def test_localhost_blocked(self):
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://localhost/path")

    def test_zero_ip_blocked(self):
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://0.0.0.0/path")

    def test_gcp_metadata_blocked(self):
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://metadata.google.internal/")

    def test_aws_metadata_ip_blocked(self):
        with pytest.raises(ValueError, match="blocked"):
            validate_url("http://169.254.169.254/latest/meta-data/")


class TestValidateUrlPrivateIp:
    def _mock_getaddrinfo(self, ip: str):
        return [(None, None, None, None, (ip, 0))]

    def test_private_10_network_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_getaddrinfo("10.0.0.1")):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://internal.corp")

    def test_private_192_168_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_getaddrinfo("192.168.1.100")):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://myrouter.local")

    def test_loopback_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_getaddrinfo("127.0.0.1")):
            with pytest.raises(ValueError, match="private"):
                validate_url("http://loopback.example.com")

    def test_public_ip_allowed(self):
        with patch("socket.getaddrinfo", return_value=self._mock_getaddrinfo("8.8.8.8")):
            url = validate_url("http://dns.google")
        assert url == "http://dns.google"

    def test_dns_failure_allowed(self):
        import socket as _socket

        with patch("socket.getaddrinfo", side_effect=_socket.gaierror("no DNS")):
            url = validate_url("https://nonexistent-domain-xyz.example")
        assert url.startswith("https://")

    def test_block_private_ips_disabled(self):
        # When block_private_ips=False, private IPs should pass
        url = validate_url("http://192.168.1.1", block_private_ips=False)
        assert url == "http://192.168.1.1"

    def test_missing_hostname_raises(self):
        with pytest.raises(ValueError):
            validate_url("http://", block_private_ips=False)
