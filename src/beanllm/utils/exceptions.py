"""
Custom Exceptions
독립적인 예외 클래스들
"""


class LLMManagerError(Exception):
    """Base exception for llm-model-manager"""

    pass


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


class RateLimitError(ProviderError):
    """Rate limit 초과"""

    def __init__(self, message: str, provider: str = None, retry_after: int = None):
        self.retry_after = retry_after
        super().__init__(message, provider)


class AuthenticationError(ProviderError):
    """인증 실패"""

    pass


class InvalidParameterError(LLMManagerError):
    """잘못된 파라미터"""

    pass
