"""
Generic LRU Cache with TTL support and automatic cleanup

Provides a thread-safe LRU cache with:
- Time-to-Live (TTL) expiration
- Automatic background cleanup of expired entries
- Proper LRU eviction when max size is reached
- Thread safety for concurrent access
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU Cache with TTL support and automatic cleanup

    Features:
        - LRU (Least Recently Used) eviction policy
        - TTL (Time-To-Live) expiration
        - Automatic background cleanup of expired entries
        - Thread-safe operations
        - Cache statistics (hits, misses, evictions)

    Mathematical Foundation:
        LRU Cache as Ordered Map with Timestamp:

        cache[key] = (value, timestamp, access_count)

        Eviction Policy:
        - If size >= max_size: evict least recently used (oldest in OrderedDict)
        - If timestamp + ttl < current_time: evict (expired)

        Cleanup Algorithm:
        - Background thread runs every cleanup_interval seconds
        - Removes all entries where: current_time - timestamp > ttl

    Example:
        ```python
        from beanllm.utils.cache import LRUCache

        # Create cache with 1000 items max, 1 hour TTL
        cache = LRUCache[str, list](max_size=1000, ttl=3600)

        # Set value
        cache.set("key1", [1, 2, 3])

        # Get value (returns None if expired or not found)
        value = cache.get("key1")

        # Get statistics
        stats = cache.stats()
        print(f"Hit rate: {stats['hit_rate']:.2%}")

        # Clear cache
        cache.clear()

        # Shutdown cleanup thread (important!)
        cache.shutdown()
        ```

    References:
        - LRU Cache: https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU
        - Python OrderedDict: https://docs.python.org/3/library/collections.html#collections.OrderedDict
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[int] = None,
        cleanup_interval: int = 60,
        on_evict: Optional[Callable[[K, V], None]] = None,
    ):
        """
        Args:
            max_size: Maximum number of cache entries (LRU eviction when exceeded)
            ttl: Time-to-live in seconds (None = no expiration)
            cleanup_interval: Interval in seconds for automatic cleanup (default: 60s)
            on_evict: Optional callback when item is evicted: on_evict(key, value)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cleanup_interval = cleanup_interval
        self.on_evict = on_evict

        # Cache storage: OrderedDict for LRU behavior
        # Value: (data, timestamp)
        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

        # Automatic cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Start automatic cleanup if TTL is enabled
        if self.ttl is not None and self.ttl > 0:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is not None:
            return  # Already running

        self._shutdown_event.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True, name="LRUCache-Cleanup"
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self):
        """Background worker that periodically removes expired entries"""
        while not self._shutdown_event.wait(timeout=self.cleanup_interval):
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove all expired entries from cache"""
        if self.ttl is None:
            return

        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, (value, timestamp) in self._cache.items():
                if current_time - timestamp > self.ttl:
                    expired_keys.append(key)

            # Remove expired entries
            for key in expired_keys:
                value, _ = self._cache.pop(key)
                self._expirations += 1

                # Call eviction callback
                if self.on_evict:
                    try:
                        self.on_evict(key, value)
                    except Exception:
                        pass  # Ignore callback errors

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value from cache

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default

        Note:
            - Updates LRU order (moves to end)
            - Removes expired entries
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default

            value, timestamp = self._cache[key]

            # Check TTL expiration
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                # Expired - remove and return default
                del self._cache[key]
                self._misses += 1
                self._expirations += 1

                # Call eviction callback
                if self.on_evict:
                    try:
                        self.on_evict(key, value)
                    except Exception:
                        pass

                return default

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache

        Note:
            - Evicts LRU item if max_size is exceeded
            - Updates timestamp for TTL
        """
        with self._lock:
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                evicted_key, (evicted_value, _) = self._cache.popitem(last=False)
                self._evictions += 1

                # Call eviction callback
                if self.on_evict:
                    try:
                        self.on_evict(evicted_key, evicted_value)
                    except Exception:
                        pass

            # Set value with current timestamp
            self._cache[key] = (value, time.time())

            # Move to end (most recently used)
            self._cache.move_to_end(key)

    def delete(self, key: K) -> bool:
        """
        Delete entry from cache

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                value, _ = self._cache.pop(key)

                # Call eviction callback
                if self.on_evict:
                    try:
                        self.on_evict(key, value)
                    except Exception:
                        pass

                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            # Call eviction callback for all items
            if self.on_evict:
                for key, (value, _) in self._cache.items():
                    try:
                        self.on_evict(key, value)
                    except Exception:
                        pass

            self._cache.clear()

            # Reset statistics
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics:
                - size: Current number of entries
                - max_size: Maximum number of entries
                - ttl: Time-to-live in seconds (None if disabled)
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Hit rate (hits / (hits + misses))
                - evictions: Number of LRU evictions
                - expirations: Number of TTL expirations
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
            }

    def shutdown(self):
        """
        Shutdown cleanup thread and clear cache

        Important: Call this before application exit to properly cleanup resources
        """
        # Stop cleanup thread
        if self._cleanup_thread is not None:
            self._shutdown_event.set()
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None

        # Clear cache
        self.clear()

    def __del__(self):
        """Destructor - ensure cleanup thread is stopped"""
        try:
            self.shutdown()
        except Exception:
            pass

    def __len__(self) -> int:
        """Return number of cache entries"""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: K) -> bool:
        """Check if key exists in cache (includes expiration check)"""
        with self._lock:
            if key not in self._cache:
                return False

            value, timestamp = self._cache[key]

            # Check TTL expiration
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                # Expired - remove
                del self._cache[key]
                self._expirations += 1

                # Call eviction callback
                if self.on_evict:
                    try:
                        self.on_evict(key, value)
                    except Exception:
                        pass

                return False

            return True
