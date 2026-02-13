"""Pipeline decorators - Helper functions for cache, lock, events, execution."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, cast

from beanllm.infrastructure.distributed.cache_wrapper import SyncCacheWrapper
from beanllm.infrastructure.distributed.event_integration import get_event_logger
from beanllm.infrastructure.distributed.factory import get_cache, get_rate_limiter

try:
    from beanllm.utils.logging import get_logger as _get_logger
    def get_logger(name: str) -> logging.Logger:
        return cast(logging.Logger, _get_logger(name))
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

logger = get_logger(__name__)


def generate_cache_key(prefix: str, args: tuple, kwargs: dict[str, Any]) -> str:
    """Generate cache key from prefix and arguments."""
    args_list: List[str] = []
    key_data: dict[str, Any] = {"args": args_list, "kwargs": kwargs}
    for arg in args:
        if isinstance(arg, (str, Path)):
            try:
                with open(str(arg), "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                args_list.append(f"file:{file_hash}")
            except Exception:
                args_list.append(str(arg))
        elif hasattr(arg, "tobytes"):
            args_list.append(f"array:{hashlib.sha256(arg.tobytes()).hexdigest()}")
        else:
            args_list.append(str(arg))
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    return f"{prefix}:{key_hash}"


def generate_lock_key(prefix: str, args: tuple, kwargs: dict) -> Optional[str]:
    """Generate distributed lock key."""
    if args and isinstance(args[0], (str, Path)):
        file_path = str(args[0])
        file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        return f"{prefix}:lock:{file_hash}"
    return None


async def publish_event(event_type: str, event_data: dict) -> None:
    """Publish event (async)."""
    try:
        event_logger = get_event_logger()
        await event_logger.log_event(event_type, event_data)
    except Exception as e:
        logger.debug(f"Event publishing failed: {e}")


def publish_event_async(event_type: str, event_data: dict) -> None:
    """Publish event from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(publish_event(event_type, event_data))
        else:
            loop.run_until_complete(publish_event(event_type, event_data))
    except RuntimeError:
        asyncio.run(publish_event(event_type, event_data))
    except Exception as e:
        logger.debug(f"Event publishing failed: {e}")


def execute_with_features(
    func: Callable[..., Any],
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: Any,
    cache_key: Optional[str],
    rate_key: Union[str, Callable[..., Any]],
    event_prefix: str,
) -> Any:
    """Execute sync function with rate limit and events."""
    if config.enable_event_streaming:
        publish_event_async(f"{event_prefix}.started", {})
    if config.enable_rate_limiting:
        actual_rate_key_str: Optional[str] = rate_key if isinstance(rate_key, str) else None
        if callable(rate_key):
            try:
                result_key = rate_key(self, args, kwargs)
                actual_rate_key_str = result_key if isinstance(result_key, str) else None
            except Exception:
                actual_rate_key_str = "default"
        if actual_rate_key_str:
            rate_limiter = get_rate_limiter()
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    pass
                else:
                    loop.run_until_complete(rate_limiter.acquire(key=actual_rate_key_str, cost=1.0))
            except RuntimeError:
                asyncio.run(rate_limiter.acquire(key=actual_rate_key_str, cost=1.0))
            except Exception as e:
                logger.debug(f"Rate limiting failed: {e}")
    try:
        result = func(self, *args, **kwargs)
        if config.enable_event_streaming:
            publish_event_async(f"{event_prefix}.completed", {"success": True})
        return result
    except Exception as e:
        if config.enable_event_streaming:
            publish_event_async(f"{event_prefix}.failed", {"error": str(e)})
        raise


async def execute_with_features_async(
    func: Callable[..., Any],
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: Any,
    cache_key: Optional[str],
    rate_key: Union[str, Callable[..., Any]],
    event_prefix: str,
    rate_limit_condition: Optional[Callable[..., bool]] = None,
    pipeline_type: str = "default",
) -> Any:
    """Execute async function with rate limit and events."""
    if config.enable_event_streaming:
        await publish_event(f"{event_prefix}.started", {})
    if config.enable_rate_limiting:
        should_rate_limit = True
        if rate_limit_condition:
            try:
                should_rate_limit = rate_limit_condition(self, args, kwargs)
            except Exception:
                should_rate_limit = True
        if should_rate_limit:
            actual_rate_key_str: Optional[str] = rate_key if isinstance(rate_key, str) else None
            if callable(rate_key):
                try:
                    result_key = rate_key(self, args, kwargs)
                    if result_key is None:
                        should_rate_limit = False
                    else:
                        actual_rate_key_str = result_key if isinstance(result_key, str) else pipeline_type
                except Exception:
                    actual_rate_key_str = pipeline_type
            if should_rate_limit and actual_rate_key_str:
                rate_limiter = get_rate_limiter()
                try:
                    await rate_limiter.acquire(key=actual_rate_key_str, cost=1.0)
                except Exception as e:
                    logger.debug(f"Rate limiting failed: {e}")
    try:
        result = await func(self, *args, **kwargs)
        if config.enable_event_streaming:
            await publish_event(f"{event_prefix}.completed", {"success": True})
        return result
    except Exception as e:
        if config.enable_event_streaming:
            await publish_event(f"{event_prefix}.failed", {"error": str(e)})
        raise
