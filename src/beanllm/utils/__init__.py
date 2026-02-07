"""
Utilities - 독립적인 유틸리티 모듈
"""

from typing import Any

# Config
# CLI
from .cli import main
from .config import Config, EnvConfig

# Core Utilities
from .core import (
    DIContainer,
    EvaluationDashboard,
    LRUCache,
    get_container,
)

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

STREAMING_WRAPPER_AVAILABLE = True

# Async Helpers
from .async_helpers import (
    AsyncHelperMixin,
    get_cached_sync,
    log_event_sync,
    run_async_in_sync,
    set_cache_sync,
)

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
RAG_DEBUG_AVAILABLE = False
EmbeddingInfo: Any = None
RAGDebugger: Any = None
SimilarityInfo: Any = None
compare_texts: Any = None
inspect_embedding: Any = None
similarity_heatmap: Any = None
validate_pipeline: Any = None
visualize_embeddings: Any = None
visualize_embeddings_2d: Any = None

try:
    from .rag_debug import (
        EmbeddingInfo as _EmbeddingInfo,
    )
    from .rag_debug import (
        RAGDebugger as _RAGDebugger,
    )
    from .rag_debug import (
        SimilarityInfo as _SimilarityInfo,
    )
    from .rag_debug import (
        compare_texts as _compare_texts,
    )
    from .rag_debug import (
        inspect_embedding as _inspect_embedding,
    )
    from .rag_debug import (
        similarity_heatmap as _similarity_heatmap,
    )
    from .rag_debug import (
        validate_pipeline as _validate_pipeline,
    )
    from .rag_debug import (
        visualize_embeddings as _visualize_embeddings,
    )
    from .rag_debug import (
        visualize_embeddings_2d as _visualize_embeddings_2d,
    )

    RAG_DEBUG_AVAILABLE = True
    EmbeddingInfo = _EmbeddingInfo
    RAGDebugger = _RAGDebugger
    SimilarityInfo = _SimilarityInfo
    compare_texts = _compare_texts
    inspect_embedding = _inspect_embedding
    similarity_heatmap = _similarity_heatmap
    validate_pipeline = _validate_pipeline
    visualize_embeddings = _visualize_embeddings
    visualize_embeddings_2d = _visualize_embeddings_2d
except ImportError:
    pass

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
    # Async Helpers
    "AsyncHelperMixin",
    "run_async_in_sync",
    "log_event_sync",
    "get_cached_sync",
    "set_cache_sync",
]


# RAG Debug 지연 import (순환 참조 방지)
def _lazy_import_rag_debug():
    """RAG Debug 모듈 지연 import"""
    from .integration.rag_visualization import RAGPipelineVisualizer
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
