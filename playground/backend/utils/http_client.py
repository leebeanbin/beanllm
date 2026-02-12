"""
Shared HTTP Client

Singleton httpx.AsyncClient with connection pooling.
Avoids creating a new TCP connection pool per request.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_http_client: Optional[httpx.AsyncClient] = None


def get_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """
    Get or create a shared httpx.AsyncClient.

    The client is lazily created on first call and reused for all
    subsequent requests, benefiting from HTTP keep-alive and connection
    pooling.

    Args:
        timeout: Default timeout in seconds (only used on first creation).

    Returns:
        A shared httpx.AsyncClient instance.
    """
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
            ),
            follow_redirects=True,
        )
        logger.info("Shared httpx.AsyncClient initialized (pool: 20/100)")
    return _http_client


async def close_http_client() -> None:
    """Close the shared HTTP client. Call on application shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        logger.info("Shared httpx.AsyncClient closed")
