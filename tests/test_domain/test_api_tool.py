"""Tests for domain/tools/advanced/api.py — ExternalAPITool, APIConfig, APIProtocol."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from beanllm.domain.tools.advanced.api import (
    HTTPX_AVAILABLE,
    REQUESTS_AVAILABLE,
    APIConfig,
    APIProtocol,
    ExternalAPITool,
)

# ---------------------------------------------------------------------------
# APIProtocol & APIConfig
# ---------------------------------------------------------------------------


class TestAPIProtocol:
    def test_rest_value(self):
        assert APIProtocol.REST.value == "rest"

    def test_graphql_value(self):
        assert APIProtocol.GRAPHQL.value == "graphql"


class TestAPIConfig:
    def test_minimal_config(self):
        cfg = APIConfig(base_url="https://api.example.com")
        assert cfg.base_url == "https://api.example.com"
        assert cfg.protocol == APIProtocol.REST
        assert cfg.timeout == 30
        assert cfg.auth_type is None

    def test_full_config(self):
        cfg = APIConfig(
            base_url="https://api.example.com",
            protocol=APIProtocol.GRAPHQL,
            auth_type="bearer",
            auth_value="my-token",
            headers={"X-Custom": "val"},
            timeout=60,
            max_retries=5,
            rate_limit=100,
        )
        assert cfg.protocol == APIProtocol.GRAPHQL
        assert cfg.auth_type == "bearer"
        assert cfg.rate_limit == 100

    def test_default_headers_empty(self):
        cfg = APIConfig(base_url="https://example.com")
        assert cfg.headers == {}


# ---------------------------------------------------------------------------
# ExternalAPITool — requires requests
# ---------------------------------------------------------------------------


def _make_tool(auth_type=None, auth_value=None, rate_limit=None, max_retries=3):
    cfg = APIConfig(
        base_url="https://api.example.com",
        auth_type=auth_type,
        auth_value=auth_value,
        rate_limit=rate_limit,
        max_retries=max_retries,
    )
    with patch("beanllm.domain.tools.advanced.api._requests") as mock_requests:
        mock_requests.Session.return_value = MagicMock()
        mock_requests.Session.return_value.headers = {}
        tool = ExternalAPITool(cfg)
    tool.session = MagicMock()
    tool.session.headers = {}
    return tool


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
class TestExternalAPIToolInit:
    def test_creates_session(self):
        with patch("beanllm.domain.tools.advanced.api._requests") as mock_requests:
            mock_requests.Session.return_value = MagicMock()
            mock_requests.Session.return_value.headers = {}
            cfg = APIConfig(base_url="https://example.com")
            tool = ExternalAPITool(cfg)
        mock_requests.Session.assert_called_once()

    def test_raises_without_requests(self):
        with patch("beanllm.domain.tools.advanced.api.REQUESTS_AVAILABLE", False):
            with patch("beanllm.domain.tools.advanced.api._requests", None):
                cfg = APIConfig(base_url="https://example.com")
                with pytest.raises(ImportError):
                    ExternalAPITool(cfg)


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
class TestSetupAuth:
    def _make_session(self):
        session = MagicMock()
        session.headers = {}
        session.auth = None
        return session

    def test_bearer_auth_sets_authorization_header(self):
        with patch("beanllm.domain.tools.advanced.api._requests") as mock_requests:
            session = self._make_session()
            mock_requests.Session.return_value = session
            cfg = APIConfig(
                base_url="https://api.example.com",
                auth_type="bearer",
                auth_value="my-token",
            )
            tool = ExternalAPITool(cfg)
        assert "Bearer my-token" in session.headers.get("Authorization", "")

    def test_api_key_auth_sets_x_api_key_header(self):
        with patch("beanllm.domain.tools.advanced.api._requests") as mock_requests:
            session = self._make_session()
            mock_requests.Session.return_value = session
            cfg = APIConfig(
                base_url="https://api.example.com",
                auth_type="api_key",
                auth_value="key-123",
            )
            tool = ExternalAPITool(cfg)
        assert session.headers.get("X-API-Key") == "key-123"

    def test_basic_auth_sets_session_auth(self):
        with (
            patch("beanllm.domain.tools.advanced.api._requests") as mock_requests,
            patch("beanllm.domain.tools.advanced.api._HTTPBasicAuth") as mock_auth,
        ):
            session = self._make_session()
            mock_requests.Session.return_value = session
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance
            cfg = APIConfig(
                base_url="https://api.example.com",
                auth_type="basic",
                auth_value="user:pass",
            )
            tool = ExternalAPITool(cfg)
        mock_auth.assert_called_once_with("user", "pass")
        assert session.auth is mock_auth_instance

    def test_custom_headers_added_to_session(self):
        with patch("beanllm.domain.tools.advanced.api._requests") as mock_requests:
            session = MagicMock()
            session.headers = MagicMock()  # headers as MagicMock so .update is trackable
            session.auth = None
            mock_requests.Session.return_value = session
            cfg = APIConfig(
                base_url="https://api.example.com",
                headers={"X-Custom": "custom-val"},
            )
            tool = ExternalAPITool(cfg)
        session.headers.update.assert_called_with({"X-Custom": "custom-val"})

    def test_no_auth_no_setup(self):
        with patch("beanllm.domain.tools.advanced.api._requests") as mock_requests:
            session = self._make_session()
            mock_requests.Session.return_value = session
            cfg = APIConfig(base_url="https://api.example.com")
            tool = ExternalAPITool(cfg)
        assert session.headers.get("Authorization") is None
        assert session.headers.get("X-API-Key") is None


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
class TestRateLimitCheck:
    def test_no_rate_limit_returns_immediately(self):
        tool = _make_tool(rate_limit=None)
        start = time.monotonic()
        tool._rate_limit_check()
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_with_rate_limit_creates_token_bucket(self):
        tool = _make_tool(rate_limit=60)
        # Delete _token_bucket so the code recreates it
        if hasattr(tool, "_token_bucket"):
            del tool._token_bucket
        tool._rate_limit_check()
        assert "tokens" in tool._token_bucket

    def test_token_bucket_consumes_token(self):
        tool = _make_tool(rate_limit=60)
        # Pre-set bucket with full tokens
        tool._token_bucket = {
            "tokens": 5.0,
            "capacity": 6.0,
            "refill_rate": 1.0,
            "last_refill": time.time(),
        }
        tool._rate_limit_check()
        assert tool._token_bucket["tokens"] == pytest.approx(4.0, abs=0.1)

    def test_empty_bucket_triggers_wait(self):
        tool = _make_tool(rate_limit=600)  # fast rate
        tool._token_bucket = {
            "tokens": 0.0,
            "capacity": 1.0,
            "refill_rate": 100.0,  # very fast refill so wait is short
            "last_refill": time.time(),
        }
        with patch("time.sleep") as mock_sleep:
            tool._rate_limit_check()
        mock_sleep.assert_called_once()


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
class TestCall:
    def _make_response(self, json_data=None, status_code=200):
        resp = MagicMock()
        resp.json.return_value = json_data or {"result": "ok"}
        resp.status_code = status_code
        resp.raise_for_status = MagicMock()
        return resp

    def test_call_get_returns_json(self):
        tool = _make_tool()
        resp = self._make_response({"key": "value"})
        tool.session.request.return_value = resp
        result = tool.call("/api/data")
        assert result == {"key": "value"}

    def test_call_builds_correct_url(self):
        tool = _make_tool()
        resp = self._make_response()
        tool.session.request.return_value = resp
        tool.call("/api/data")
        call_kwargs = tool.session.request.call_args.kwargs
        assert "https://api.example.com/api/data" == call_kwargs.get("url")

    def test_call_post_with_data(self):
        tool = _make_tool()
        resp = self._make_response({"created": True})
        tool.session.request.return_value = resp
        result = tool.call("/api/create", method="POST", data={"name": "test"})
        assert result == {"created": True}
        call_kwargs = tool.session.request.call_args.kwargs
        assert call_kwargs.get("json") == {"name": "test"}

    def test_call_passes_params(self):
        tool = _make_tool()
        resp = self._make_response()
        tool.session.request.return_value = resp
        tool.call("/api/search", params={"q": "test"})
        call_kwargs = tool.session.request.call_args.kwargs
        assert call_kwargs.get("params") == {"q": "test"}

    def test_call_retries_on_request_exception(self):
        import requests

        tool = _make_tool(max_retries=2)
        resp = self._make_response()
        tool.session.request.side_effect = [
            requests.ConnectionError("timeout"),
            resp,
        ]
        with (
            patch("beanllm.domain.tools.advanced.api._requests", requests),
            patch("time.sleep"),
        ):
            result = tool.call("/api/data")
        assert result == {"result": "ok"}
        assert tool.session.request.call_count == 2

    def test_call_raises_after_max_retries(self):
        import requests

        tool = _make_tool(max_retries=2)
        tool.session.request.side_effect = requests.ConnectionError("always fails")
        with (
            patch("beanllm.domain.tools.advanced.api._requests", requests),
            patch("time.sleep"),
        ):
            with pytest.raises(requests.ConnectionError):
                tool.call("/api/data")
        assert tool.session.request.call_count == 2

    def test_call_raises_non_requests_exception_immediately(self):
        tool = _make_tool(max_retries=3)
        tool.session.request.side_effect = ValueError("not a requests error")
        with pytest.raises(ValueError):
            tool.call("/api/data")
        assert tool.session.request.call_count == 1


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not available")
class TestCallAsync:
    async def test_call_async_returns_json(self):
        tool = _make_tool()
        mock_client = MagicMock()
        mock_client.is_closed = False
        resp = MagicMock()
        resp.json.return_value = {"async": "result"}
        resp.raise_for_status = MagicMock()
        mock_client.request = AsyncMock(return_value=resp)
        tool._async_client = mock_client

        result = await tool.call_async("/api/data")
        assert result == {"async": "result"}

    async def test_call_async_raises_without_httpx(self):
        tool = _make_tool()
        with (
            patch("beanllm.domain.tools.advanced.api.HTTPX_AVAILABLE", False),
            patch("beanllm.domain.tools.advanced.api._httpx", None),
        ):
            with pytest.raises(ImportError):
                await tool.call_async("/api/data")

    async def test_call_async_retries_on_httpx_error(self):
        import httpx

        tool = _make_tool(max_retries=2)
        mock_client = MagicMock()
        mock_client.is_closed = False
        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        resp.raise_for_status = MagicMock()
        mock_client.request = AsyncMock(
            side_effect=[
                httpx.ConnectError("timeout"),
                resp,
            ]
        )
        tool._async_client = mock_client

        with (
            patch("beanllm.domain.tools.advanced.api._httpx", httpx),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await tool.call_async("/api/data")
        assert result == {"ok": True}

    async def test_close_clears_async_client(self):
        tool = _make_tool()
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        tool._async_client = mock_client

        await tool.close()
        mock_client.aclose.assert_awaited_once()
        assert tool._async_client is None

    async def test_close_when_no_client(self):
        tool = _make_tool()
        tool._async_client = None
        await tool.close()  # should not raise

    async def test_close_when_already_closed(self):
        tool = _make_tool()
        mock_client = AsyncMock()
        mock_client.is_closed = True
        tool._async_client = mock_client
        await tool.close()  # should not call aclose
        mock_client.aclose.assert_not_awaited()


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
class TestGetOrCreateAsyncClient:
    def test_creates_new_client_when_none(self):
        import httpx

        tool = _make_tool()
        tool._async_client = None
        mock_client = MagicMock()
        mock_client.is_closed = False

        with patch("beanllm.domain.tools.advanced.api._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Limits = httpx.Limits
            result = tool._get_or_create_async_client()
        assert result is mock_client

    def test_reuses_existing_open_client(self):
        tool = _make_tool()
        existing = MagicMock()
        existing.is_closed = False
        tool._async_client = existing

        result = tool._get_or_create_async_client()
        assert result is existing

    def test_creates_new_when_existing_is_closed(self):
        import httpx

        tool = _make_tool()
        closed_client = MagicMock()
        closed_client.is_closed = True
        tool._async_client = closed_client

        new_client = MagicMock()
        with patch("beanllm.domain.tools.advanced.api._httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = new_client
            mock_httpx.Limits = httpx.Limits
            result = tool._get_or_create_async_client()
        assert result is new_client


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not available")
class TestCallGraphQL:
    def test_graphql_calls_post_to_graphql_endpoint(self):
        tool = _make_tool()
        resp = MagicMock()
        resp.json.return_value = {"data": {"user": {"name": "Alice"}}}
        resp.raise_for_status = MagicMock()
        tool.session.request.return_value = resp

        result = tool.call_graphql("{ user { name } }")
        assert result == {"data": {"user": {"name": "Alice"}}}
        call_kwargs = tool.session.request.call_args.kwargs
        assert call_kwargs.get("method", "").upper() == "POST"
        assert "/graphql" in call_kwargs.get("url", "")

    def test_graphql_with_variables(self):
        tool = _make_tool()
        resp = MagicMock()
        resp.json.return_value = {"data": {}}
        resp.raise_for_status = MagicMock()
        tool.session.request.return_value = resp

        tool.call_graphql("query Q($id: ID!) { user(id: $id) { name } }", variables={"id": "1"})
        call_kwargs = tool.session.request.call_args.kwargs
        assert call_kwargs["json"]["variables"] == {"id": "1"}

    def test_graphql_without_variables(self):
        tool = _make_tool()
        resp = MagicMock()
        resp.json.return_value = {}
        resp.raise_for_status = MagicMock()
        tool.session.request.return_value = resp

        tool.call_graphql("{ users { id } }")
        call_kwargs = tool.session.request.call_args.kwargs
        assert "variables" not in call_kwargs.get("json", {})
