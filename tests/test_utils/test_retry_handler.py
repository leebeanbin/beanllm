"""
Retry Handler 테스트 - 에러 처리 유틸리티 테스트
"""

import pytest
from unittest.mock import Mock, patch
import asyncio

try:
    from beanllm.utils.error_handling import RetryHandler
    RETRY_HANDLER_AVAILABLE = True
except ImportError:
    RETRY_HANDLER_AVAILABLE = False


@pytest.mark.skipif(not RETRY_HANDLER_AVAILABLE, reason="RetryHandler not available")
class TestRetryHandler:
    """RetryHandler 테스트"""

    @pytest.fixture
    def retry_handler(self):
        """RetryHandler 인스턴스"""
        from beanllm.utils.error_handling import RetryConfig, RetryStrategy
        
        config = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            strategy=RetryStrategy.EXPONENTIAL,
            retry_on_exceptions=(Exception,),
        )
        return RetryHandler(config)

    def test_retry_handler_success(self, retry_handler):
        """성공 시 재시도 없음 테스트"""
        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_handler.execute(test_func)
        assert result == "success"
        assert call_count == 1

    def test_retry_handler_retry(self, retry_handler):
        """재시도 테스트"""
        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Retry needed")
            return "success"

        result = retry_handler.execute(test_func)
        assert result == "success"
        assert call_count == 3

    def test_retry_handler_max_retries(self, retry_handler):
        """최대 재시도 초과 테스트"""
        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fail")

        from beanllm.utils.error_handling import MaxRetriesExceededError
        with pytest.raises(MaxRetriesExceededError):
            retry_handler.execute(test_func)

        assert call_count == 3  # max_retries=3이므로 3번 시도

    def test_retry_handler_async(self, retry_handler):
        """비동기 함수 재시도 테스트 (동기 execute 사용)"""
        call_count = 0

        def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Retry needed")
            return "async success"

        result = retry_handler.execute(async_func)
        assert result == "async success"
        assert call_count == 2


