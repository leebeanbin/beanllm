"""
Utilities - 독립적인 유틸리티 모듈
"""

# Config
from .config import Config, EnvConfig

# CLI
from .cli import main

# Dependency Manager
from .dependency import (
    DependencyManager,
    check_available,
    require,
    require_any,
)

# Exceptions
from .exceptions import ModelNotFoundError, ProviderError, RateLimitError

# Lazy Loading
from .lazy_loading import (
    LazyLoader,
    LazyLoadMixin,
    lazy_property,
)

# Token Counter
from .token_counter import (
    CostEstimate,
    CostEstimator,
    ModelContextWindow,
    ModelPricing,
    TokenCounter,
    count_message_tokens,
    count_tokens,
    estimate_cost,
    get_cheapest_model,
    get_context_window,
)

# Tracer
from .tracer import (
    Trace,
    Tracer,
    TraceSpan,
    enable_tracing,
    get_tracer,
)

# Core Utilities
from .core import (
    DIContainer,
    EvaluationDashboard,
    LRUCache,
    get_container,
)

# Logging Utilities
from .logging import (
    LogLevel,
    StructuredLogger,
    get_logger,
    get_structured_logger,
)

# Streaming Utilities
from .streaming import (
    BufferedStreamWrapper,
    PausableStream,
    StreamBuffer,
    StreamResponse,
    StreamStats,
    pretty_stream,
    stream_collect,
    stream_print,
    stream_response,
)

STREAMING_WRAPPER_AVAILABLE = True

# Integration Utilities
from .integration import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    CostTrackingCallback,
    ErrorHandler,
    ErrorHandlerConfig,
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    FunctionCallback,
    LLMKitError,
    LoggingCallback,
    MaxRetriesExceededError,
    RAGPipelineVisualizer,
    RateLimitConfig,
    RateLimiter,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    StreamingCallback,
    TimeoutError,
    TimingCallback,
    ValidationError,
    circuit_breaker,
    create_callback_manager,
    fallback,
    get_error_tracker,
    rate_limit,
    sanitize_error_message,
    timeout,
    with_error_handling,
)

# Retry (from resilience module - supports both async and sync)
from .resilience.retry import retry

# Provider Retry Strategies
try:
    from .provider_retry_strategies import (
        PROVIDER_RETRY_STRATEGIES,
        get_error_type_retry_config,
        get_provider_retry_config,
    )
except ImportError:
    # Optional dependency
    get_provider_retry_config = None
    get_error_type_retry_config = None
    PROVIDER_RETRY_STRATEGIES = {}

# Cost Tracking
try:
    from .cost_tracker import (
        BudgetConfig,
        CostRecord,
        CostTracker,
        get_cost_tracker,
        set_cost_tracker,
    )
except ImportError:
    # Optional dependency
    CostTracker = None
    BudgetConfig = None
    CostRecord = None
    get_cost_tracker = None
    set_cost_tracker = None

# RAG Debug - 순환 참조 방지를 위해 지연 import
try:
    from .rag_debug import (
        EmbeddingInfo,
        RAGDebugger,
        SimilarityInfo,
        compare_texts,
        inspect_embedding,
        similarity_heatmap,
        validate_pipeline,
        visualize_embeddings,
        visualize_embeddings_2d,
    )

    RAG_DEBUG_AVAILABLE = True
except ImportError:
    RAG_DEBUG_AVAILABLE = False
    EmbeddingInfo = None
    RAGDebugger = None
    SimilarityInfo = None
    compare_texts = None
    inspect_embedding = None
    similarity_heatmap = None
    validate_pipeline = None
    visualize_embeddings = None
    visualize_embeddings_2d = None

__all__ = [
    # Config
    "Config",
    "EnvConfig",
    # Exceptions
    "ProviderError",
    "ModelNotFoundError",
    "RateLimitError",
    # Dependency Manager
    "DependencyManager",
    "require",
    "check_available",
    "require_any",
    # Lazy Loading
    "LazyLoadMixin",
    "LazyLoader",
    "lazy_property",
    # Token Counter
    "ModelPricing",
    "ModelContextWindow",
    "TokenCounter",
    "CostEstimate",
    "CostEstimator",
    "count_tokens",
    "count_message_tokens",
    "estimate_cost",
    "get_cheapest_model",
    "get_context_window",
    # Tracer
    "Trace",
    "TraceSpan",
    "Tracer",
    "get_tracer",
    "enable_tracing",
    # Core Utilities
    "LRUCache",
    "DIContainer",
    "get_container",
    "EvaluationDashboard",
    # Logging Utilities
    "get_logger",
    "StructuredLogger",
    "LogLevel",
    "get_structured_logger",
    # Streaming Utilities
    "StreamBuffer",
    "StreamResponse",
    "StreamStats",
    "stream_response",
    "stream_print",
    "stream_collect",
    "pretty_stream",
    "BufferedStreamWrapper",
    "PausableStream",
    # Integration Utilities
    "BaseCallback",
    "CallbackEvent",
    "CallbackManager",
    "CostTrackingCallback",
    "FunctionCallback",
    "LoggingCallback",
    "StreamingCallback",
    "TimingCallback",
    "create_callback_manager",
    "LLMKitError",
    "TimeoutError",
    "ValidationError",
    "CircuitBreakerError",
    "MaxRetriesExceededError",
    "RetryStrategy",
    "RetryConfig",
    "RetryHandler",
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "circuit_breaker",
    "RateLimitConfig",
    "RateLimiter",
    "rate_limit",
    "FallbackHandler",
    "fallback",
    "ErrorRecord",
    "ErrorTracker",
    "get_error_tracker",
    "ErrorHandlerConfig",
    "ErrorHandler",
    "with_error_handling",
    "timeout",
    "RAGPipelineVisualizer",
    "sanitize_error_message",
    # Retry
    "retry",
    # Provider Retry Strategies
    "get_provider_retry_config",
    "get_error_type_retry_config",
    "PROVIDER_RETRY_STRATEGIES",
    # Cost Tracking
    "CostTracker",
    "BudgetConfig",
    "CostRecord",
    "get_cost_tracker",
    "set_cost_tracker",
    # CLI
    "main",
]


# RAG Debug 지연 import (순환 참조 방지)
def _lazy_import_rag_debug():
    """RAG Debug 모듈 지연 import"""
    from .rag_debug import (
        EmbeddingInfo,
        RAGDebugger,
        SimilarityInfo,
        compare_texts,
        inspect_embedding,
        similarity_heatmap,
        validate_pipeline,
        visualize_embeddings,
        visualize_embeddings_2d,
    )
    from .integration.rag_visualization import RAGPipelineVisualizer

    return {
        "EmbeddingInfo": EmbeddingInfo,
        "RAGDebugger": RAGDebugger,
        "SimilarityInfo": SimilarityInfo,
        "compare_texts": compare_texts,
        "inspect_embedding": inspect_embedding,
        "similarity_heatmap": similarity_heatmap,
        "validate_pipeline": validate_pipeline,
        "visualize_embeddings": visualize_embeddings,
        "visualize_embeddings_2d": visualize_embeddings_2d,
        "RAGPipelineVisualizer": RAGPipelineVisualizer,
    }


# 지연 import를 위한 속성 접근
def __getattr__(name: str):
    """지연 import를 위한 속성 접근"""
    if name in {
        "EmbeddingInfo",
        "RAGDebugger",
        "SimilarityInfo",
        "compare_texts",
        "inspect_embedding",
        "similarity_heatmap",
        "validate_pipeline",
        "visualize_embeddings",
        "visualize_embeddings_2d",
        "RAGPipelineVisualizer",
    }:
        rag_debug = _lazy_import_rag_debug()
        return rag_debug[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
