"""
Structured Logger - Consistent logging with context

Standardizes 510+ logger calls across the codebase.
"""

import logging
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Iterator, Optional


class LogLevel(str, Enum):
    """Log level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class StructuredLogger:
    """
    Structured logging with consistent format

    Eliminates inconsistent logging patterns:
    - Before: 510 occurrences of ad-hoc logger calls
    - After: Standardized structured logging

    Example:
        >>> logger = StructuredLogger(__name__)
        >>> logger.log_file_load("/path/to/file.pdf", 10, success=True)
        # Output: {"operation": "file_load", "status": "success",
        #          "filepath": "/path/to/file.pdf", "document_count": 10}
    """

    def __init__(self, name: str, enable_structured: bool = True):
        """
        Initialize structured logger

        Args:
            name: Logger name (usually __name__)
            enable_structured: If True, use structured format; if False, use plain text
        """
        self.logger = logging.getLogger(name)
        self.enable_structured = enable_structured

    def log_operation(
        self,
        level: str,
        operation: str,
        status: str,
        **context: Any
    ):
        """
        Log a generic operation with context

        Args:
            level: Log level (debug, info, warning, error, critical)
            operation: Operation name (e.g., "file_load", "api_call")
            status: Operation status (e.g., "success", "failed", "in_progress")
            **context: Additional context as key-value pairs

        Example:
            >>> logger.log_operation(
            ...     level="info",
            ...     operation="api_call",
            ...     status="success",
            ...     provider="openai",
            ...     model="gpt-4o",
            ...     latency_ms=250
            ... )
        """
        if self.enable_structured:
            msg = {
                "operation": operation,
                "status": status,
                **context
            }
        else:
            # Plain text format
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
            msg = f"{operation} {status}" + (f" ({ctx_str})" if ctx_str else "")

        getattr(self.logger, level)(msg)

    # ========================================================================
    # Domain-specific logging methods
    # ========================================================================

    def log_file_load(
        self,
        filepath: str,
        count: Optional[int] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Log file loading operation

        Args:
            filepath: Path to the file
            count: Number of documents loaded (if successful)
            success: Whether the operation succeeded
            error: Error message (if failed)

        Example:
            >>> logger.log_file_load("/path/to/file.pdf", count=5)
            >>> logger.log_file_load("/path/to/bad.pdf", success=False,
            ...                      error="File not found")
        """
        context = {"filepath": filepath}
        if count is not None:
            context["document_count"] = count
        if error:
            context["error"] = error

        self.log_operation(
            level="info" if success else "error",
            operation="file_load",
            status="success" if success else "failed",
            **context
        )

    def log_api_call(
        self,
        provider: str,
        model: str,
        success: bool = True,
        latency_ms: Optional[float] = None,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None
    ):
        """
        Log LLM API call

        Args:
            provider: Provider name (openai, anthropic, etc.)
            model: Model name
            success: Whether the call succeeded
            latency_ms: Response latency in milliseconds
            tokens_used: Total tokens used
            error: Error message (if failed)

        Example:
            >>> logger.log_api_call("openai", "gpt-4o", latency_ms=250,
            ...                     tokens_used=1500)
        """
        context = {
            "provider": provider,
            "model": model
        }
        if latency_ms is not None:
            context["latency_ms"] = latency_ms
        if tokens_used is not None:
            context["tokens_used"] = tokens_used
        if error:
            context["error"] = error

        self.log_operation(
            level="info" if success else "error",
            operation="api_call",
            status="success" if success else "failed",
            **context
        )

    def log_embedding_generation(
        self,
        text_count: int,
        embedding_dim: int,
        success: bool = True,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """
        Log embedding generation

        Args:
            text_count: Number of texts embedded
            embedding_dim: Embedding dimension
            success: Whether the operation succeeded
            latency_ms: Generation latency
            error: Error message (if failed)

        Example:
            >>> logger.log_embedding_generation(100, 1536, latency_ms=500)
        """
        context = {
            "text_count": text_count,
            "embedding_dim": embedding_dim
        }
        if latency_ms is not None:
            context["latency_ms"] = latency_ms
        if error:
            context["error"] = error

        self.log_operation(
            level="info" if success else "error",
            operation="embedding_generation",
            status="success" if success else "failed",
            **context
        )

    def log_vector_search(
        self,
        query: str,
        result_count: int,
        search_type: str = "similarity",
        latency_ms: Optional[float] = None
    ):
        """
        Log vector search operation

        Args:
            query: Search query
            result_count: Number of results returned
            search_type: Type of search (similarity, hybrid, mmr)
            latency_ms: Search latency

        Example:
            >>> logger.log_vector_search("What is RAG?", 5, "hybrid",
            ...                          latency_ms=50)
        """
        context = {
            "query": query[:100],  # Truncate long queries
            "result_count": result_count,
            "search_type": search_type
        }
        if latency_ms is not None:
            context["latency_ms"] = latency_ms

        self.log_operation(
            level="info",
            operation="vector_search",
            status="completed",
            **context
        )

    def log_cache_operation(
        self,
        cache_type: str,
        operation: str,
        hit: bool,
        key: Optional[str] = None
    ):
        """
        Log cache operation

        Args:
            cache_type: Type of cache (embedding, prompt, model)
            operation: Operation type (get, set, clear)
            hit: Whether it was a cache hit (for get operations)
            key: Cache key (optional, for debugging)

        Example:
            >>> logger.log_cache_operation("embedding", "get", hit=True)
            >>> logger.log_cache_operation("prompt", "set", hit=False,
            ...                            key="template_123")
        """
        context = {
            "cache_type": cache_type,
            "cache_operation": operation,
            "cache_hit": hit
        }
        if key:
            context["key"] = key

        self.log_operation(
            level="debug",
            operation="cache",
            status="hit" if hit else "miss",
            **context
        )

    @contextmanager
    def log_duration(
        self,
        operation: str,
        **context: Any
    ) -> Iterator[Dict[str, Any]]:
        """
        Context manager to log operation duration

        Args:
            operation: Operation name
            **context: Additional context

        Yields:
            Context dictionary (can be updated during operation)

        Example:
            >>> with logger.log_duration("pdf_parsing", file="doc.pdf") as ctx:
            ...     # ... parsing logic ...
            ...     ctx["page_count"] = 100
            # Automatically logs: {"operation": "pdf_parsing",
            #                      "status": "completed", "duration_ms": 1234,
            #                      "file": "doc.pdf", "page_count": 100}
        """
        start_time = time.time()
        log_context = dict(context)

        try:
            yield log_context
            duration_ms = (time.time() - start_time) * 1000
            log_context["duration_ms"] = round(duration_ms, 2)

            self.log_operation(
                level="info",
                operation=operation,
                status="completed",
                **log_context
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_context["duration_ms"] = round(duration_ms, 2)
            log_context["error"] = str(e)

            self.log_operation(
                level="error",
                operation=operation,
                status="failed",
                **log_context
            )
            raise

    # Convenience methods
    def debug(self, msg: str, **context: Any):
        """Log debug message"""
        self.log_operation("debug", "general", "info", message=msg, **context)

    def info(self, msg: str, **context: Any):
        """Log info message"""
        self.log_operation("info", "general", "info", message=msg, **context)

    def warning(self, msg: str, **context: Any):
        """Log warning message"""
        self.log_operation("warning", "general", "warning", message=msg, **context)

    def error(self, msg: str, **context: Any):
        """Log error message"""
        self.log_operation("error", "general", "error", message=msg, **context)


# Factory function
def get_structured_logger(name: str, enable_structured: bool = True) -> StructuredLogger:
    """
    Get or create a structured logger

    Args:
        name: Logger name (usually __name__)
        enable_structured: Enable structured logging format

    Returns:
        StructuredLogger instance

    Example:
        >>> logger = get_structured_logger(__name__)
        >>> logger.log_file_load("file.pdf", count=10)
    """
    return StructuredLogger(name, enable_structured)


__all__ = [
    "StructuredLogger",
    "LogLevel",
    "get_structured_logger",
]
