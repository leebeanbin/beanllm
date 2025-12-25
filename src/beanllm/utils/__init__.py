"""
Utilities - 독립적인 유틸리티 모듈
"""

# Config
# Callbacks
from .callbacks import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
    CostTrackingCallback,
    FunctionCallback,
    LoggingCallback,
    StreamingCallback,
    TimingCallback,
    create_callback_manager,
)

# CLI
from .cli import main
from .config import Config, EnvConfig

# DI Container
from .di_container import get_container

# Error Handling
from .error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    ErrorHandler,
    ErrorHandlerConfig,
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    LLMKitError,
    MaxRetriesExceededError,
    RateLimitConfig,
    RateLimiter,
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    TimeoutError,
    ValidationError,
    circuit_breaker,
    fallback,
    get_error_tracker,
    rate_limit,
    timeout,
    with_error_handling,
)

# Exceptions
from .exceptions import ModelNotFoundError, ProviderError, RateLimitError

# Logger
from .logger import get_logger

# Retry
from .retry import retry

# Streaming
from .streaming import (
    StreamBuffer,
    StreamResponse,
    StreamStats,
    pretty_stream,
    stream_collect,
    stream_print,
    stream_response,
)

# Streaming Wrapper
try:
    from .streaming_wrapper import BufferedStreamWrapper, PausableStream

    STREAMING_WRAPPER_AVAILABLE = True
except ImportError:
    STREAMING_WRAPPER_AVAILABLE = False
    BufferedStreamWrapper = None
    PausableStream = None

# Evaluation Dashboard
try:
    from .evaluation_dashboard import EvaluationDashboard

    EVALUATION_DASHBOARD_AVAILABLE = True
except ImportError:
    EVALUATION_DASHBOARD_AVAILABLE = False
    EvaluationDashboard = None

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

# Tracer
from .tracer import (
    Trace,
    Tracer,
    TraceSpan,
    enable_tracing,
    get_tracer,
)

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

# RAG Visualization
try:
    from .rag_visualization import RAGPipelineVisualizer

    RAG_VISUALIZATION_AVAILABLE = True
except ImportError:
    RAG_VISUALIZATION_AVAILABLE = False
    RAGPipelineVisualizer = None

__all__ = [
    # Config
    "Config",
    "EnvConfig",
    # Exceptions
    "ProviderError",
    "ModelNotFoundError",
    "RateLimitError",
    # Logger
    "get_logger",
    # Retry
    "retry",
    # Error Handling
    "LLMKitError",
    "ProviderError",
    "RateLimitError",
    "TimeoutError",
    "ValidationError",
    "CircuitBreakerError",
    "MaxRetriesExceededError",
    "RetryStrategy",
    "RetryConfig",
    "RetryHandler",
    "retry_decorator",
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
    # Streaming
    "StreamStats",
    "StreamResponse",
    "StreamBuffer",
    "stream_response",
    "stream_print",
    "stream_collect",
    "pretty_stream",
    "BufferedStreamWrapper",
    "PausableStream",
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
    # Tracer
    "Trace",
    "TraceSpan",
    "Tracer",
    "get_tracer",
    "enable_tracing",
    # Callbacks
    "CallbackEvent",
    "BaseCallback",
    "LoggingCallback",
    "CostTrackingCallback",
    "TimingCallback",
    "StreamingCallback",
    "FunctionCallback",
    "CallbackManager",
    "create_callback_manager",
    # CLI
    "main",
    # DI Container
    "get_container",
    # Evaluation Dashboard
    "EvaluationDashboard",
    # RAG Debug - 지연 import로 제공
    # "EmbeddingInfo",
    # "SimilarityInfo",
    # "RAGDebugger",
    # "inspect_embedding",
    # "compare_texts",
    # "validate_pipeline",
    # "visualize_embeddings_2d",
    # "similarity_heatmap",
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
    from .rag_visualization import RAGPipelineVisualizer

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
