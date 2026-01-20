"""
Async Helper Utilities

Provides helper methods for common async/sync patterns in beanllm.

This module reduces boilerplate code for:
- Running async code in sync contexts (asyncio.run patterns)
- Event logging with async protocols
- Cache operations with async protocols
"""

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, TypeVar

if TYPE_CHECKING:
    from beanllm.domain.protocols import CacheProtocol, EventLoggerProtocol

T = TypeVar("T")


def run_async_in_sync(coro: Coroutine[Any, Any, T]) -> Optional[T]:
    """
    Run async code in sync context safely.

    Handles three scenarios:
    1. Event loop is running → create task (fire-and-forget)
    2. Event loop exists but not running → run_until_complete
    3. No event loop → asyncio.run

    Args:
        coro: Coroutine to execute

    Returns:
        Result of coroutine, or None if fire-and-forget

    Example:
        >>> async def async_operation():
        ...     return "result"
        >>> result = run_async_in_sync(async_operation())
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Fire-and-forget (can't await in running loop)
            asyncio.create_task(coro)
            return None
        else:
            # Loop exists but not running
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists
        return asyncio.run(coro)


class AsyncHelperMixin:
    """
    Mixin class providing async helper methods.

    Use this in classes that need to call async protocols from sync methods.

    Example:
        >>> class MyClass(AsyncHelperMixin):
        ...     def __init__(self, event_logger=None, cache=None):
        ...         self._event_logger = event_logger
        ...         self._cache = cache
        ...
        ...     def sync_method(self):
        ...         # Log event (async protocol, sync method)
        ...         self._log_event("my_event", {"data": "value"})
        ...
        ...         # Get from cache
        ...         cached = self._get_cached("key")
    """

    _event_logger: Optional["EventLoggerProtocol"] = None
    _cache: Optional["CacheProtocol"] = None

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> Optional[T]:
        """
        Run async coroutine from sync method.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of coroutine, or None if fire-and-forget

        Example:
            >>> result = self._run_async(self._cache.get("key"))
        """
        return run_async_in_sync(coro)

    def _log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        fire_and_forget: bool = True,
    ) -> None:
        """
        Log event using event logger protocol (handles async in sync context).

        Args:
            event_type: Event type identifier
            data: Event data dictionary
            fire_and_forget: If True, don't wait for result (default: True)

        Example:
            >>> self._log_event("ocr.recognize.started", {"engine": "paddleocr"})

        Note:
            Requires self._event_logger to be set (EventLoggerProtocol).
            Silently does nothing if event_logger is None.
        """
        if self._event_logger is None:
            return

        event_coro = self._event_logger.log_event(event_type, data)

        if fire_and_forget:
            # Best effort - don't block
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(event_coro)
                else:
                    loop.run_until_complete(event_coro)
            except RuntimeError:
                asyncio.run(event_coro)
        else:
            # Wait for result
            self._run_async(event_coro)

    def _get_cached(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get value from cache protocol (handles async in sync context).

        Args:
            key: Cache key
            default: Default value if not found or cache unavailable

        Returns:
            Cached value or default

        Example:
            >>> cached = self._get_cached("embeddings:text123")
            >>> if cached:
            ...     return cached

        Note:
            Requires self._cache to be set (CacheProtocol).
            Returns default if cache is None.
        """
        if self._cache is None:
            return default

        try:
            result = self._run_async(self._cache.get(key))
            return result if result is not None else default
        except Exception:
            # Cache failure - return default
            return default

    def _set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        fire_and_forget: bool = True,
    ) -> None:
        """
        Set value in cache protocol (handles async in sync context).

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = use cache default)
            fire_and_forget: If True, don't wait for result (default: True)

        Example:
            >>> result = compute_expensive_operation()
            >>> self._set_cache("result:key123", result, ttl=3600)

        Note:
            Requires self._cache to be set (CacheProtocol).
            Silently does nothing if cache is None.
        """
        if self._cache is None:
            return

        cache_coro = self._cache.set(key, value, ttl=ttl) if ttl else self._cache.set(key, value)

        if fire_and_forget:
            # Best effort - don't block
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(cache_coro)
                else:
                    loop.run_until_complete(cache_coro)
            except (RuntimeError, Exception):
                try:
                    asyncio.run(cache_coro)
                except Exception:
                    pass  # Cache storage failed - continue
        else:
            # Wait for result
            try:
                self._run_async(cache_coro)
            except Exception:
                pass  # Cache storage failed - continue


# Standalone functions for use without mixin


def log_event_sync(
    event_logger: Optional["EventLoggerProtocol"],
    event_type: str,
    data: Dict[str, Any],
) -> None:
    """
    Standalone function to log event from sync context.

    Args:
        event_logger: EventLoggerProtocol instance (or None)
        event_type: Event type identifier
        data: Event data dictionary

    Example:
        >>> from beanllm.infrastructure.distributed import get_event_logger
        >>> logger = get_event_logger()
        >>> log_event_sync(logger, "task.started", {"task_id": "123"})
    """
    if event_logger is None:
        return

    event_coro = event_logger.log_event(event_type, data)
    run_async_in_sync(event_coro)


def get_cached_sync(
    cache: Optional["CacheProtocol"],
    key: str,
    default: Optional[Any] = None,
) -> Optional[Any]:
    """
    Standalone function to get from cache in sync context.

    Args:
        cache: CacheProtocol instance (or None)
        key: Cache key
        default: Default value if not found

    Returns:
        Cached value or default

    Example:
        >>> from beanllm.infrastructure.distributed import get_cache
        >>> cache = get_cache()
        >>> result = get_cached_sync(cache, "key123", default=[])
    """
    if cache is None:
        return default

    try:
        result = run_async_in_sync(cache.get(key))
        return result if result is not None else default
    except Exception:
        return default


def set_cache_sync(
    cache: Optional["CacheProtocol"],
    key: str,
    value: Any,
    ttl: Optional[int] = None,
) -> None:
    """
    Standalone function to set cache value in sync context.

    Args:
        cache: CacheProtocol instance (or None)
        key: Cache key
        value: Value to cache
        ttl: Time-to-live in seconds

    Example:
        >>> from beanllm.infrastructure.distributed import get_cache
        >>> cache = get_cache()
        >>> set_cache_sync(cache, "result:123", {"data": "value"}, ttl=3600)
    """
    if cache is None:
        return

    cache_coro = cache.set(key, value, ttl=ttl) if ttl else cache.set(key, value)

    try:
        run_async_in_sync(cache_coro)
    except Exception:
        pass  # Cache storage failed - continue
