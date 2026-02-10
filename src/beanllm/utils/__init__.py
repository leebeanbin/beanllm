"""
Utilities - 독립적인 유틸리티 모듈
"""

import importlib

STREAMING_WRAPPER_AVAILABLE = True

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # CLI
    "main": ("beanllm.utils.cli", "main"),
    # Config
    "Config": ("beanllm.utils.config", "Config"),
    "EnvConfig": ("beanllm.utils.config", "EnvConfig"),
    # Core Utilities
    "DIContainer": ("beanllm.utils.core", "DIContainer"),
    "EvaluationDashboard": ("beanllm.utils.core", "EvaluationDashboard"),
    "LRUCache": ("beanllm.utils.core", "LRUCache"),
    "get_container": ("beanllm.utils.core", "get_container"),
    # Dependency Manager
    "DependencyManager": ("beanllm.utils.dependency", "DependencyManager"),
    "check_available": ("beanllm.utils.dependency", "check_available"),
    "require": ("beanllm.utils.dependency", "require"),
    "require_any": ("beanllm.utils.dependency", "require_any"),
    # Exceptions
    "ModelNotFoundError": ("beanllm.utils.exceptions", "ModelNotFoundError"),
    "ProviderError": ("beanllm.utils.exceptions", "ProviderError"),
    "RateLimitError": ("beanllm.utils.exceptions", "RateLimitError"),
    # Lazy Loading
    "LazyLoader": ("beanllm.utils.lazy_loading", "LazyLoader"),
    "LazyLoadMixin": ("beanllm.utils.lazy_loading", "LazyLoadMixin"),
    "lazy_property": ("beanllm.utils.lazy_loading", "lazy_property"),
    # Logging Utilities
    "LogLevel": ("beanllm.utils.logging", "LogLevel"),
    "StructuredLogger": ("beanllm.utils.logging", "StructuredLogger"),
    "get_logger": ("beanllm.utils.logging", "get_logger"),
    "get_structured_logger": ("beanllm.utils.logging", "get_structured_logger"),
    # Streaming Utilities
    "BufferedStreamWrapper": ("beanllm.utils.streaming", "BufferedStreamWrapper"),
    "PausableStream": ("beanllm.utils.streaming", "PausableStream"),
    "StreamBuffer": ("beanllm.utils.streaming", "StreamBuffer"),
    "StreamResponse": ("beanllm.utils.streaming", "StreamResponse"),
    "StreamStats": ("beanllm.utils.streaming", "StreamStats"),
    "pretty_stream": ("beanllm.utils.streaming", "pretty_stream"),
    "stream_collect": ("beanllm.utils.streaming", "stream_collect"),
    "stream_print": ("beanllm.utils.streaming", "stream_print"),
    "stream_response": ("beanllm.utils.streaming", "stream_response"),
    # Token Counter
    "CostEstimate": ("beanllm.utils.token_counter", "CostEstimate"),
    "CostEstimator": ("beanllm.utils.token_counter", "CostEstimator"),
    "ModelContextWindow": ("beanllm.utils.token_counter", "ModelContextWindow"),
    "ModelPricing": ("beanllm.utils.token_counter", "ModelPricing"),
    "TokenCounter": ("beanllm.utils.token_counter", "TokenCounter"),
    "count_message_tokens": ("beanllm.utils.token_counter", "count_message_tokens"),
    "count_tokens": ("beanllm.utils.token_counter", "count_tokens"),
    "estimate_cost": ("beanllm.utils.token_counter", "estimate_cost"),
    "get_cheapest_model": ("beanllm.utils.token_counter", "get_cheapest_model"),
    "get_context_window": ("beanllm.utils.token_counter", "get_context_window"),
    # Tracer
    "Trace": ("beanllm.utils.tracer", "Trace"),
    "Tracer": ("beanllm.utils.tracer", "Tracer"),
    "TraceSpan": ("beanllm.utils.tracer", "TraceSpan"),
    "enable_tracing": ("beanllm.utils.tracer", "enable_tracing"),
    "get_tracer": ("beanllm.utils.tracer", "get_tracer"),
    # Async Helpers
    "AsyncHelperMixin": ("beanllm.utils.async_helpers", "AsyncHelperMixin"),
    "get_cached_sync": ("beanllm.utils.async_helpers", "get_cached_sync"),
    "log_event_sync": ("beanllm.utils.async_helpers", "log_event_sync"),
    "run_async_in_sync": ("beanllm.utils.async_helpers", "run_async_in_sync"),
    "set_cache_sync": ("beanllm.utils.async_helpers", "set_cache_sync"),
    # Integration Utilities
    "BaseCallback": ("beanllm.utils.integration", "BaseCallback"),
    "CallbackEvent": ("beanllm.utils.integration", "CallbackEvent"),
    "CallbackManager": ("beanllm.utils.integration", "CallbackManager"),
    "CircuitBreaker": ("beanllm.utils.integration", "CircuitBreaker"),
    "CircuitBreakerConfig": ("beanllm.utils.integration", "CircuitBreakerConfig"),
    "CircuitBreakerError": ("beanllm.utils.integration", "CircuitBreakerError"),
    "CircuitState": ("beanllm.utils.integration", "CircuitState"),
    "CostTrackingCallback": ("beanllm.utils.integration", "CostTrackingCallback"),
    "ErrorHandler": ("beanllm.utils.integration", "ErrorHandler"),
    "ErrorHandlerConfig": ("beanllm.utils.integration", "ErrorHandlerConfig"),
    "ErrorRecord": ("beanllm.utils.integration", "ErrorRecord"),
    "ErrorTracker": ("beanllm.utils.integration", "ErrorTracker"),
    "FallbackHandler": ("beanllm.utils.integration", "FallbackHandler"),
    "FunctionCallback": ("beanllm.utils.integration", "FunctionCallback"),
    "LLMKitError": ("beanllm.utils.integration", "LLMKitError"),
    "LoggingCallback": ("beanllm.utils.integration", "LoggingCallback"),
    "MaxRetriesExceededError": ("beanllm.utils.integration", "MaxRetriesExceededError"),
    "RAGPipelineVisualizer": ("beanllm.utils.integration", "RAGPipelineVisualizer"),
    "RateLimitConfig": ("beanllm.utils.integration", "RateLimitConfig"),
    "RateLimiter": ("beanllm.utils.integration", "RateLimiter"),
    "RetryConfig": ("beanllm.utils.integration", "RetryConfig"),
    "RetryHandler": ("beanllm.utils.integration", "RetryHandler"),
    "RetryStrategy": ("beanllm.utils.integration", "RetryStrategy"),
    "StreamingCallback": ("beanllm.utils.integration", "StreamingCallback"),
    "TimeoutError": ("beanllm.utils.integration", "TimeoutError"),
    "TimingCallback": ("beanllm.utils.integration", "TimingCallback"),
    "ValidationError": ("beanllm.utils.integration", "ValidationError"),
    "circuit_breaker": ("beanllm.utils.integration", "circuit_breaker"),
    "create_callback_manager": ("beanllm.utils.integration", "create_callback_manager"),
    "fallback": ("beanllm.utils.integration", "fallback"),
    "get_error_tracker": ("beanllm.utils.integration", "get_error_tracker"),
    "rate_limit": ("beanllm.utils.integration", "rate_limit"),
    "sanitize_error_message": ("beanllm.utils.integration", "sanitize_error_message"),
    "timeout": ("beanllm.utils.integration", "timeout"),
    "with_error_handling": ("beanllm.utils.integration", "with_error_handling"),
    # Retry
    "retry": ("beanllm.utils.resilience.retry", "retry"),
}

_OPTIONAL_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # Provider Retry Strategies
    "PROVIDER_RETRY_STRATEGIES": (
        "beanllm.utils.provider_retry_strategies",
        "PROVIDER_RETRY_STRATEGIES",
    ),
    "get_error_type_retry_config": (
        "beanllm.utils.provider_retry_strategies",
        "get_error_type_retry_config",
    ),
    "get_provider_retry_config": (
        "beanllm.utils.provider_retry_strategies",
        "get_provider_retry_config",
    ),
    # Cost Tracking
    "BudgetConfig": ("beanllm.utils.cost_tracker", "BudgetConfig"),
    "CostRecord": ("beanllm.utils.cost_tracker", "CostRecord"),
    "CostTracker": ("beanllm.utils.cost_tracker", "CostTracker"),
    "get_cost_tracker": ("beanllm.utils.cost_tracker", "get_cost_tracker"),
    "set_cost_tracker": ("beanllm.utils.cost_tracker", "set_cost_tracker"),
    # RAG Debug
    "EmbeddingInfo": ("beanllm.utils.rag_debug", "EmbeddingInfo"),
    "RAGDebugger": ("beanllm.utils.rag_debug", "RAGDebugger"),
    "SimilarityInfo": ("beanllm.utils.rag_debug", "SimilarityInfo"),
    "compare_texts": ("beanllm.utils.rag_debug", "compare_texts"),
    "inspect_embedding": ("beanllm.utils.rag_debug", "inspect_embedding"),
    "similarity_heatmap": ("beanllm.utils.rag_debug", "similarity_heatmap"),
    "validate_pipeline": ("beanllm.utils.rag_debug", "validate_pipeline"),
    "visualize_embeddings": ("beanllm.utils.rag_debug", "visualize_embeddings"),
    "visualize_embeddings_2d": ("beanllm.utils.rag_debug", "visualize_embeddings_2d"),
}

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
    # RAG Debug
    "EmbeddingInfo",
    "SimilarityInfo",
    "RAGDebugger",
    "inspect_embedding",
    "compare_texts",
    "validate_pipeline",
    "visualize_embeddings",
    "visualize_embeddings_2d",
    "similarity_heatmap",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr = _LAZY_IMPORT_MAP[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    if name in _OPTIONAL_LAZY_IMPORT_MAP:
        mod_path, attr = _OPTIONAL_LAZY_IMPORT_MAP[name]
        try:
            mod = importlib.import_module(mod_path)
            val = getattr(mod, attr)
        except ImportError:
            val = None
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
