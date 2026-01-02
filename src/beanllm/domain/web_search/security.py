"""
Security utilities for web search and scraping

SSRF (Server-Side Request Forgery) 방지를 위한 URL 검증 유틸리티
"""

import ipaddress
import logging
import socket
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# 허용하지 않을 IP 범위 (Private/Reserved IP addresses)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("0.0.0.0/8"),  # Current network
    ipaddress.ip_network("10.0.0.0/8"),  # Private network
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("172.16.0.0/12"),  # Private network
    ipaddress.ip_network("192.168.0.0/16"),  # Private network
    ipaddress.ip_network("224.0.0.0/4"),  # Multicast
    ipaddress.ip_network("240.0.0.0/4"),  # Reserved
    # IPv6
    ipaddress.ip_network("::1/128"),  # Loopback
    ipaddress.ip_network("fe80::/10"),  # Link-local
    ipaddress.ip_network("fc00::/7"),  # Unique local
]

# 허용하지 않을 호스트명
BLOCKED_HOSTNAMES = [
    "localhost",
    "0.0.0.0",
    "metadata.google.internal",  # GCP metadata
    "169.254.169.254",  # AWS/Azure metadata
]


def validate_url(
    url: str,
    allowed_schemes: Optional[List[str]] = None,
    block_private_ips: bool = True,
) -> str:
    """
    URL 검증 (SSRF 방지)

    Args:
        url: 검증할 URL
        allowed_schemes: 허용할 스키마 리스트 (기본: ['http', 'https'])
        block_private_ips: Private IP 차단 여부 (기본: True)

    Returns:
        검증된 URL

    Raises:
        ValueError: 허용되지 않은 URL인 경우

    Security:
        - 허용된 스키마만 허용 (http, https)
        - Private/Internal IP 차단
        - Localhost 차단
        - Cloud metadata endpoints 차단

    Example:
        ```python
        from beanllm.domain.web_search.security import validate_url

        # 안전한 URL 검증
        safe_url = validate_url("https://example.com")

        # 허용되지 않은 URL은 에러 발생
        # validate_url("http://localhost:8080")  # ValueError
        # validate_url("http://192.168.1.1")  # ValueError
        # validate_url("file:///etc/passwd")  # ValueError
        ```
    """
    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]

    try:
        # URL 파싱
        parsed = urlparse(url)

        # 스키마 검증
        if parsed.scheme not in allowed_schemes:
            raise ValueError(
                f"URL scheme '{parsed.scheme}' not allowed. "
                f"Allowed schemes: {allowed_schemes}"
            )

        # 호스트명 추출
        hostname = parsed.hostname
        if not hostname:
            raise ValueError(f"Invalid URL: missing hostname in {url}")

        # 호스트명 차단 리스트 확인
        if hostname.lower() in BLOCKED_HOSTNAMES:
            raise ValueError(
                f"Access denied: hostname '{hostname}' is blocked (SSRF protection)"
            )

        # Private IP 차단
        if block_private_ips:
            try:
                # DNS 해석하여 IP 주소 확인
                ip_addresses = socket.getaddrinfo(hostname, None)

                for ip_info in ip_addresses:
                    ip_str = ip_info[4][0]
                    ip_addr = ipaddress.ip_address(ip_str)

                    # Private/Reserved IP 확인
                    for blocked_range in BLOCKED_IP_RANGES:
                        if ip_addr in blocked_range:
                            raise ValueError(
                                f"Access denied: {hostname} resolves to private/reserved IP {ip_str} "
                                f"(SSRF protection)"
                            )

            except socket.gaierror:
                # DNS 해석 실패 - 허용 (존재하지 않는 도메인은 나중에 HTTP 요청 시 실패)
                logger.debug(f"DNS resolution failed for {hostname}, proceeding anyway")
            except ValueError:
                # IP 파싱 실패 또는 차단된 IP - 재발생
                raise

        return url

    except ValueError:
        # 이미 적절한 에러 메시지가 있는 ValueError는 재발생
        raise
    except Exception as e:
        logger.error(f"URL validation failed for {url}: {e}")
        raise ValueError(f"Invalid URL: {url} - {e}")
