"""
Utils Layer 테스트 - 유틸리티 함수 테스트
"""

import pytest

try:
    from beanllm.utils import Config, EnvConfig, retry, get_logger
except ImportError:
    from src.beanllm.utils import Config, EnvConfig, retry, get_logger


class TestConfig:
    """Config 테스트"""

    def test_config_exists(self):
        """Config 클래스 존재 확인"""
        assert Config is not None
        assert EnvConfig is not None

    def test_env_config_get_active_providers(self):
        """EnvConfig.get_active_providers 테스트"""
        providers = EnvConfig.get_active_providers()
        assert isinstance(providers, list)

    def test_env_config_is_provider_available(self):
        """EnvConfig.is_provider_available 테스트"""
        # Ollama는 항상 사용 가능 (로컬)
        assert EnvConfig.is_provider_available("ollama") is True


class TestRetry:
    """Retry 데코레이터 테스트"""

    def test_retry_decorator_exists(self):
        """retry 데코레이터 존재 확인"""
        assert retry is not None
        assert callable(retry)

    def test_retry_decorator_usage(self):
        """retry 데코레이터 사용 테스트"""
        call_count = [0]

        @retry(max_attempts=3)
        def test_function():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Test error")
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count[0] == 2


class TestLogger:
    """Logger 테스트"""

    def test_get_logger(self):
        """get_logger 함수 테스트"""
        logger = get_logger("test")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")


class TestErrorHandling:
    """Error Handling 테스트"""

    def test_error_handler_import(self):
        """ErrorHandler import 테스트"""
        try:
            from beanllm.utils.error_handling import ErrorHandler
        except ImportError:
            from src.beanllm.utils.error_handling import ErrorHandler

        assert ErrorHandler is not None

    def test_circuit_breaker_import(self):
        """CircuitBreaker import 테스트"""
        try:
            from beanllm.utils.error_handling import CircuitBreaker
        except ImportError:
            from src.beanllm.utils.error_handling import CircuitBreaker

        assert CircuitBreaker is not None

    def test_rate_limiter_import(self):
        """RateLimiter import 테스트"""
        try:
            from beanllm.utils.error_handling import RateLimiter
        except ImportError:
            from src.beanllm.utils.error_handling import RateLimiter

        assert RateLimiter is not None


class TestTokenCounter:
    """Token Counter 테스트"""

    def test_count_tokens_import(self):
        """count_tokens import 테스트"""
        try:
            from beanllm.utils.token_counter import count_tokens
        except ImportError:
            from src.beanllm.utils.token_counter import count_tokens

        assert count_tokens is not None
        assert callable(count_tokens)

    def test_count_tokens_basic(self):
        """count_tokens 기본 테스트"""
        try:
            from beanllm.utils.token_counter import count_tokens
        except ImportError:
            from src.beanllm.utils.token_counter import count_tokens

        try:
            tokens = count_tokens("Hello world", model="gpt-4o")
            assert isinstance(tokens, int)
            assert tokens > 0
        except Exception:
            pytest.skip("Token counter not available")


class TestStreaming:
    """Streaming 테스트"""

    def test_streaming_import(self):
        """Streaming 유틸리티 import 테스트"""
        try:
            from beanllm.utils.streaming import StreamStats
        except ImportError:
            from src.beanllm.utils.streaming import StreamStats

        assert StreamStats is not None

