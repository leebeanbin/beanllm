"""
Domain layer protocols for Infrastructure dependencies.

This module defines Protocol interfaces that the Domain layer can depend on,
following the Dependency Inversion Principle. Infrastructure layer implements
these protocols, preventing direct Domain → Infrastructure dependencies.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for caching functionality."""

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        ...

    async def delete(self, key: str) -> None:
        """
        Delete value from cache.

        Args:
            key: Cache key
        """
        ...

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Protocol for rate limiting functionality."""

    async def acquire(self, key: str, max_requests: int = 60, window: int = 60) -> bool:
        """
        Try to acquire rate limit token.

        Args:
            key: Rate limit key
            max_requests: Maximum requests allowed
            window: Time window in seconds

        Returns:
            True if request is allowed

        Raises:
            RateLimitError: If rate limit exceeded
        """
        ...

    async def get_remaining(self, key: str) -> int:
        """
        Get remaining requests for key.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining requests
        """
        ...


@runtime_checkable
class LockManagerProtocol(Protocol):
    """Protocol for distributed lock functionality."""

    async def acquire(self, lock_name: str, timeout: int = 10) -> bool:
        """
        Acquire a distributed lock.

        Args:
            lock_name: Name of the lock
            timeout: Timeout in seconds

        Returns:
            True if lock acquired
        """
        ...

    async def release(self, lock_name: str) -> None:
        """
        Release a distributed lock.

        Args:
            lock_name: Name of the lock
        """
        ...

    def __aenter__(self) -> "LockManagerProtocol":
        """Context manager entry."""
        ...

    def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        ...


@runtime_checkable
class EventLoggerProtocol(Protocol):
    """Protocol for event logging functionality."""

    async def log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an event.

        Args:
            event_type: Type of event
            data: Event data
            metadata: Additional metadata (optional)
        """
        ...


@runtime_checkable
class EventBusProtocol(Protocol):
    """Protocol for event bus functionality."""

    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish message to event bus.

        Args:
            topic: Topic name
            message: Message data
            metadata: Additional metadata (optional)
        """
        ...

    async def subscribe(
        self,
        topic: str,
        callback: Any,  # Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to topic.

        Args:
            topic: Topic name
            callback: Callback function
        """
        ...


@runtime_checkable
class BatchProcessorProtocol(Protocol):
    """Protocol for batch processing functionality."""

    async def process_batch(
        self,
        items: List[Any],
        batch_size: int = 32,
        max_workers: Optional[int] = None,
    ) -> List[Any]:
        """
        Process items in batches.

        Args:
            items: Items to process
            batch_size: Size of each batch
            max_workers: Maximum concurrent workers

        Returns:
            Processed results
        """
        ...


@runtime_checkable
class ConcurrencyControllerProtocol(Protocol):
    """Protocol for concurrency control."""

    async def acquire_semaphore(self, max_concurrent: int = 10) -> None:
        """
        Acquire semaphore for concurrency control.

        Args:
            max_concurrent: Maximum concurrent operations
        """
        ...

    async def release_semaphore(self) -> None:
        """Release semaphore."""
        ...

    def __aenter__(self) -> "ConcurrencyControllerProtocol":
        """Context manager entry."""
        ...

    def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        ...


# Synchronous versions for non-async code


@runtime_checkable
class SyncCacheProtocol(Protocol):
    """Synchronous protocol for caching functionality."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ...

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


# Configuration protocols


@runtime_checkable
class DistributedConfigProtocol(Protocol):
    """Protocol for distributed configuration."""

    @property
    def use_distributed(self) -> bool:
        """Whether distributed mode is enabled."""
        ...

    @property
    def cache_enabled(self) -> bool:
        """Whether caching is enabled."""
        ...

    @property
    def rate_limiting_enabled(self) -> bool:
        """Whether rate limiting is enabled."""
        ...

    @property
    def event_streaming_enabled(self) -> bool:
        """Whether event streaming is enabled."""
        ...


# Re-export for convenience
__all__ = [
    "CacheProtocol",
    "RateLimiterProtocol",
    "LockManagerProtocol",
    "EventLoggerProtocol",
    "EventBusProtocol",
    "BatchProcessorProtocol",
    "ConcurrencyControllerProtocol",
    "SyncCacheProtocol",
    "DistributedConfigProtocol",
]
