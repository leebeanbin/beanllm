"""
beanllm.utils.exceptions - Custom Exception Classes
커스텀 예외 클래스들

이 모듈은 beanllm에서 사용하는 모든 커스텀 예외를 정의합니다.
"""


# ===== Base Exceptions =====


class LLMManagerError(Exception):
    """Base exception for llm-model-manager"""

    pass


class LLMKitError(Exception):
    """beanllm 베이스 예외"""

    pass


# ===== Provider Exceptions =====


class ProviderError(LLMManagerError):
    """Provider 관련 에러"""

    def __init__(self, message: str, provider: str = None):
        self.provider = provider
        super().__init__(message)


class ModelNotFoundError(LLMManagerError):
    """모델을 찾을 수 없음"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model not found: {model_name}")


class AuthenticationError(ProviderError):
    """인증 실패"""

    pass


# ===== Error Handling Exceptions =====


class RateLimitError(ProviderError):
    """Rate limit 에러"""

    def __init__(self, message: str = None, provider: str = None, retry_after: int = None):
        self.retry_after = retry_after
        # Support both old and new usage patterns
        if message is None:
            message = "Rate limit exceeded"
        super().__init__(message, provider)


class TimeoutError(LLMKitError):
    """Timeout 에러"""

    pass


class ValidationError(LLMKitError):
    """검증 에러"""

    pass


class CircuitBreakerError(LLMKitError):
    """Circuit breaker open 에러"""

    pass


class MaxRetriesExceededError(LLMKitError):
    """최대 재시도 횟수 초과"""

    pass


class InvalidParameterError(LLMManagerError):
    """잘못된 파라미터"""

    pass
