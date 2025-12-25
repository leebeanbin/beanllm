"""
Rate Limiter 테스트 - 에러 처리 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, patch
import asyncio
import time

try:
    from beanllm.utils.error_handling import RateLimiter, RateLimitConfig
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False


@pytest.mark.skipif(not RATE_LIMITER_AVAILABLE, reason="RateLimiter not available")
class TestRateLimiter:
    """RateLimiter 테스트"""

    @pytest.fixture
    def rate_limiter(self):
        """RateLimiter 인스턴스"""
        config = RateLimitConfig(max_calls=5, time_window=1.0)
        return RateLimiter(config=config)

    def test_rate_limiter_allow(self, rate_limiter):
        """허용 테스트"""
        def test_func():
            return "allowed"

        # 허용된 호출
        for _ in range(5):
            result = rate_limiter.call(test_func)
            assert result == "allowed"

    def test_rate_limiter_limit(self, rate_limiter):
        """제한 테스트"""
        def test_func():
            return "allowed"

        # 제한 내 호출
        for _ in range(5):
            result = rate_limiter.call(test_func)
            assert result == "allowed"

        # 제한 초과 시도
        from beanllm.utils.error_handling import RateLimitError
        with pytest.raises(RateLimitError):
            rate_limiter.call(test_func)

    @pytest.mark.asyncio
    async def test_rate_limiter_async(self, rate_limiter):
        """비동기 함수 테스트"""
        # RateLimiter는 동기 함수만 지원하므로 동기 함수로 테스트
        def sync_func():
            return "async allowed"

        result = rate_limiter.call(sync_func)
        assert result == "async allowed"

    def test_rate_limiter_reset(self, rate_limiter):
        """리셋 테스트"""
        def test_func():
            return "allowed"

        # 제한까지 호출
        for _ in range(5):
            rate_limiter.call(test_func)

        # 시간이 지나면 리셋되어 다시 호출 가능
        time.sleep(1.1)  # time_window보다 긴 시간 대기
        result = rate_limiter.call(test_func)
        assert result == "allowed"


