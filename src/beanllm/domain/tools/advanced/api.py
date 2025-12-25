"""
External API Integration - 외부 API 통합
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

try:
    import httpx
except ImportError:
    httpx = None

try:
    import requests
    from requests.auth import HTTPBasicAuth
except ImportError:
    requests = None
    HTTPBasicAuth = None


class APIProtocol(Enum):
    """API 프로토콜"""

    REST = "rest"
    GRAPHQL = "graphql"


@dataclass
class APIConfig:
    """API 설정"""

    base_url: str
    protocol: APIProtocol = APIProtocol.REST
    auth_type: Optional[str] = None  # "bearer", "api_key", "basic"
    auth_value: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    max_retries: int = 3
    rate_limit: Optional[int] = None  # requests per minute


class ExternalAPITool:
    """
    외부 API 통합 도구

    Mathematical Foundation:
        API Call as Function Composition:

        API_call(endpoint, params) = parse ∘ send ∘ validate ∘ prepare

        where:
        - prepare: params → request
        - validate: request → validated_request
        - send: validated_request → response
        - parse: response → result

        Error Handling with Retry:
        Result = try_with_exponential_backoff(API_call, max_retries)

        where wait_time(n) = min(max_wait, base × 2^n)
    """

    def __init__(self, config: APIConfig):
        """
        Args:
            config: API 설정
        """
        if requests is None:
            raise ImportError("requests library is required for ExternalAPITool")

        self.config = config
        self.session = requests.Session()
        self._setup_auth()
        self._last_request_time = 0

    def _setup_auth(self):
        """인증 설정"""
        if self.config.auth_type == "bearer":
            self.session.headers["Authorization"] = f"Bearer {self.config.auth_value}"
        elif self.config.auth_type == "api_key":
            self.session.headers["X-API-Key"] = self.config.auth_value
        elif self.config.auth_type == "basic" and HTTPBasicAuth is not None:
            username, password = self.config.auth_value.split(":", 1)
            self.session.auth = HTTPBasicAuth(username, password)

        # Add custom headers
        self.session.headers.update(self.config.headers)

    def _rate_limit_check(self):
        """Rate limiting (Token Bucket Algorithm)"""
        if self.config.rate_limit is None:
            return

        # Simple implementation: ensure minimum time between requests
        min_interval = 60.0 / self.config.rate_limit  # seconds per request
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    def call(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        API 호출 (동기)

        Args:
            endpoint: API 엔드포인트 (예: "/users/123")
            method: HTTP 메서드
            params: URL 쿼리 파라미터
            data: 요청 본문 데이터
            **kwargs: 추가 requests 옵션

        Returns:
            API 응답 (JSON)

        Raises:
            requests.RequestException: API 호출 실패
        """
        self._rate_limit_check()

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Exponential backoff retry
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=self.config.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()

            except requests.RequestException:
                if attempt == self.config.max_retries - 1:
                    raise

                # Exponential backoff: 2^attempt seconds
                wait_time = min(30, 2**attempt)
                time.sleep(wait_time)

        raise RuntimeError("Unexpected error in retry logic")

    async def call_async(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        API 호출 (비동기)

        Args:
            endpoint: API 엔드포인트
            method: HTTP 메서드
            params: URL 쿼리 파라미터
            data: 요청 본문 데이터
            **kwargs: 추가 httpx 옵션

        Returns:
            API 응답 (JSON)
        """
        if httpx is None:
            raise ImportError("httpx library is required for async API calls")

        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            # Setup auth headers
            headers = self.session.headers.copy()

            for attempt in range(self.config.max_retries):
                try:
                    response = await client.request(
                        method=method, url=url, params=params, json=data, headers=headers, **kwargs
                    )
                    response.raise_for_status()
                    return response.json()

                except httpx.HTTPError:
                    if attempt == self.config.max_retries - 1:
                        raise

                    wait_time = min(30, 2**attempt)
                    await asyncio.sleep(wait_time)

        raise RuntimeError("Unexpected error in retry logic")

    def call_graphql(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        GraphQL 쿼리 실행

        Args:
            query: GraphQL 쿼리 문자열
            variables: 쿼리 변수

        Returns:
            GraphQL 응답
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        return self.call(endpoint="/graphql", method="POST", data=payload)
