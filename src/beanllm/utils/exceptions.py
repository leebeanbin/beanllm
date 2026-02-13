"""
beanllm.utils.exceptions - Custom Exception Classes
커스텀 예외 클래스들

이 모듈은 beanllm에서 사용하는 모든 커스텀 예외를 정의합니다.
"""

from typing import Optional

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

    def __init__(self, message: str, provider: Optional[str] = None):
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

    def __init__(
        self,
        message: Optional[str] = None,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
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


# ===== Domain Exceptions =====


class VectorStoreError(LLMKitError):
    """Vector Store 작업 관련 에러 (인덱싱, 검색, 연결 실패 등)"""

    pass


class DocumentLoadError(LLMKitError):
    """문서 로딩 에러 (파일 파싱, 포맷 미지원 등)"""

    def __init__(self, message: str, source: Optional[str] = None):
        self.source = source
        super().__init__(message)


class RAGPipelineError(LLMKitError):
    """RAG 파이프라인 에러 (임베딩, 검색, 생성 실패 등)"""

    pass


class KnowledgeGraphError(LLMKitError):
    """Knowledge Graph 작업 에러 (엔티티 추출, 그래프 구축, 쿼리 실패 등)"""

    pass


class EmbeddingError(LLMKitError):
    """임베딩 생성 에러"""

    def __init__(self, message: str, model: Optional[str] = None):
        self.model = model
        super().__init__(message)


class ChainExecutionError(LLMKitError):
    """Chain/Pipeline 실행 에러"""

    pass
