"""
Resilience Utilities 테스트 - ErrorTracker, FallbackHandler, ProductionErrorSanitizer,
RetryHandler, CircuitBreaker
"""

import asyncio
import time

import pytest

from beanllm.utils.exceptions import CircuitBreakerError, MaxRetriesExceededError
from beanllm.utils.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    circuit_breaker,
)
from beanllm.utils.resilience.error_tracker import (
    ErrorRecord,
    ErrorTracker,
    FallbackHandler,
    ProductionErrorSanitizer,
    create_safe_error_response,
    get_error_tracker,
    sanitize_error_message,
)
from beanllm.utils.resilience.retry import (
    RetryConfig,
    RetryHandler,
    RetryStrategy,
    retry,
)


class TestErrorTracker:
    @pytest.fixture
    def tracker(self) -> ErrorTracker:
        return ErrorTracker(max_records=10)

    def test_record_exception(self, tracker: ErrorTracker) -> None:
        try:
            raise ValueError("test error")
        except ValueError as e:
            tracker.record(e)

        assert len(tracker.errors) == 1
        assert tracker.errors[0].error_type == "ValueError"

    def test_record_with_metadata(self, tracker: ErrorTracker) -> None:
        try:
            raise RuntimeError("runtime")
        except RuntimeError as e:
            tracker.record(e, metadata={"context": "test"})

        assert tracker.errors[0].metadata["context"] == "test"

    def test_get_recent_errors(self, tracker: ErrorTracker) -> None:
        for i in range(5):
            try:
                raise ValueError(f"error {i}")
            except ValueError as e:
                tracker.record(e)

        recent = tracker.get_recent_errors(3)
        assert len(recent) == 3

    def test_get_error_summary_empty(self, tracker: ErrorTracker) -> None:
        summary = tracker.get_error_summary()
        assert summary["total_errors"] == 0
        assert summary["error_types"] == {}
        assert summary["error_rate"] == 0.0

    def test_get_error_summary_with_errors(self, tracker: ErrorTracker) -> None:
        for _ in range(2):
            try:
                raise ValueError("val err")
            except ValueError as e:
                tracker.record(e)
        try:
            raise TypeError("type err")
        except TypeError as e:
            tracker.record(e)

        summary = tracker.get_error_summary()
        assert summary["total_errors"] == 3
        assert "ValueError" in summary["error_types"]
        assert "TypeError" in summary["error_types"]
        assert summary["most_common_error"] == "ValueError"

    def test_clear(self, tracker: ErrorTracker) -> None:
        try:
            raise RuntimeError("clear test")
        except RuntimeError as e:
            tracker.record(e)

        tracker.clear()
        assert len(tracker.errors) == 0

    def test_max_records_respected(self) -> None:
        tracker = ErrorTracker(max_records=3)
        for i in range(5):
            try:
                raise ValueError(f"err {i}")
            except ValueError as e:
                tracker.record(e)

        assert len(tracker.errors) == 3

    def test_global_error_tracker(self) -> None:
        t = get_error_tracker()
        assert isinstance(t, ErrorTracker)


class TestFallbackHandler:
    def test_success_no_fallback_needed(self) -> None:
        handler = FallbackHandler(fallback_value="default")

        def good_func():
            return "success"

        result = handler.call(good_func)
        assert result == "success"

    def test_fallback_value_on_error(self) -> None:
        handler = FallbackHandler(fallback_value="fallback")

        def bad_func():
            raise RuntimeError("fail")

        result = handler.call(bad_func)
        assert result == "fallback"

    def test_fallback_function_on_error(self) -> None:
        def fallback_fn(error, *args, **kwargs):
            return f"recovered: {error}"

        handler = FallbackHandler(fallback_func=fallback_fn)

        def bad_func():
            raise ValueError("oops")

        result = handler.call(bad_func)
        assert "recovered" in result

    def test_raise_on_fallback(self) -> None:
        handler = FallbackHandler(raise_on_fallback=True)

        def bad_func():
            raise RuntimeError("must reraise")

        with pytest.raises(RuntimeError):
            handler.call(bad_func)


class TestProductionErrorSanitizer:
    def test_sanitize_message_no_production(self) -> None:
        msg = "API key sk-1234567890 failed"
        sanitized = ProductionErrorSanitizer.sanitize_message(msg, production=False)
        assert sanitized == msg  # unchanged

    def test_sanitize_message_api_key(self) -> None:
        msg = "Error: api_key=sk-1234567890abcdef failed"
        sanitized = ProductionErrorSanitizer.sanitize_message(msg, production=True)
        assert "***MASKED***" in sanitized

    def test_sanitize_message_bearer_token(self) -> None:
        msg = "Authorization: Bearer abcdef1234567890xyz"
        sanitized = ProductionErrorSanitizer.sanitize_message(msg, production=True)
        assert "***MASKED***" in sanitized

    def test_sanitize_message_ip_address(self) -> None:
        msg = "Connection to 192.168.1.100 failed"
        sanitized = ProductionErrorSanitizer.sanitize_message(msg, production=True)
        assert "[IP]" in sanitized

    def test_sanitize_message_localhost_not_masked(self) -> None:
        msg = "Connection to 127.0.0.1 failed"
        sanitized = ProductionErrorSanitizer.sanitize_message(msg, production=True)
        # localhost IP should not be masked
        assert "127.0.0.1" in sanitized or "[IP]" not in sanitized

    def test_sanitize_traceback_no_production(self) -> None:
        tb = "File '/home/user/project/app.py', line 42, in main"
        sanitized = ProductionErrorSanitizer.sanitize_traceback(tb, production=False)
        assert sanitized == tb

    def test_sanitize_traceback_masks_path(self) -> None:
        tb = "File '/home/user/project/app.py', line 42, in main"
        sanitized = ProductionErrorSanitizer.sanitize_traceback(tb, production=True)
        assert "[PATH]" in sanitized

    def test_sanitize_traceback_truncates(self) -> None:
        # Create a long traceback
        lines = [f"  line {i}" for i in range(20)]
        tb = "\n".join(lines)
        sanitized = ProductionErrorSanitizer.sanitize_traceback(tb, production=True, max_frames=2)
        assert "truncated" in sanitized

    def test_create_safe_error_production(self) -> None:
        try:
            raise ValueError("api_key=sk-secret-key failed")
        except ValueError as e:
            result = ProductionErrorSanitizer.create_safe_error(e, production=True)

        assert "error_type" in result
        assert result["error_type"] == "ValueError"
        assert "message" in result
        assert result["production"] is True

    def test_create_safe_error_non_production(self) -> None:
        try:
            raise RuntimeError("debug info here")
        except RuntimeError as e:
            result = ProductionErrorSanitizer.create_safe_error(e, production=False)

        assert result["production"] is False
        assert result["message"] == "debug info here"

    def test_sanitize_error_message_helper(self) -> None:
        msg = "token=secret123456789 is invalid"
        sanitized = sanitize_error_message(msg)
        assert "secret123456789" not in sanitized or "***MASKED***" in sanitized

    def test_create_safe_error_response_helper(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError as e:
            result = create_safe_error_response(e)

        assert "error_type" in result
        assert result["error_type"] == "ValueError"


class TestRetryHandler:
    def test_success_no_retry(self) -> None:
        config = RetryConfig(max_retries=3, initial_delay=0.0)
        handler = RetryHandler(config)

        def func():
            return "ok"

        result = handler.execute(func)
        assert result == "ok"

    def test_retry_on_failure(self) -> None:
        config = RetryConfig(max_retries=3, initial_delay=0.0)
        handler = RetryHandler(config)
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "success"

        result = handler.execute(func)
        assert result == "success"
        assert call_count[0] == 3

    def test_max_retries_exceeded(self) -> None:
        config = RetryConfig(max_retries=2, initial_delay=0.0)
        handler = RetryHandler(config)

        def always_fail():
            raise ValueError("always")

        with pytest.raises(MaxRetriesExceededError):
            handler.execute(always_fail)

    def test_no_retry_for_unretriable_exception(self) -> None:
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.0,
            retry_on_exceptions=(ValueError,),
        )
        handler = RetryHandler(config)

        def raises_type_error():
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            handler.execute(raises_type_error)

    def test_retry_condition_false_stops_retry(self) -> None:
        def condition(e: Exception) -> bool:
            return False  # never retry

        config = RetryConfig(
            max_retries=3,
            initial_delay=0.0,
            retry_condition=condition,
        )
        handler = RetryHandler(config)

        def always_fail():
            raise ValueError("cond stop")

        with pytest.raises(ValueError):
            handler.execute(always_fail)

    def test_calculate_delay_fixed(self) -> None:
        config = RetryConfig(initial_delay=1.0, strategy=RetryStrategy.FIXED)
        handler = RetryHandler(config)
        delay = handler._calculate_delay(1)
        assert delay == 1.0
        delay2 = handler._calculate_delay(5)
        assert delay2 == 1.0

    def test_calculate_delay_linear(self) -> None:
        config = RetryConfig(initial_delay=1.0, strategy=RetryStrategy.LINEAR)
        handler = RetryHandler(config)
        delay2 = handler._calculate_delay(2)
        assert delay2 == 2.0

    def test_calculate_delay_exponential(self) -> None:
        config = RetryConfig(initial_delay=1.0, multiplier=2.0, strategy=RetryStrategy.EXPONENTIAL)
        handler = RetryHandler(config)
        delay1 = handler._calculate_delay(1)
        delay2 = handler._calculate_delay(2)
        assert delay1 == 1.0
        assert delay2 == 2.0

    def test_calculate_delay_jitter(self) -> None:
        config = RetryConfig(initial_delay=1.0, strategy=RetryStrategy.JITTER)
        handler = RetryHandler(config)
        delay = handler._calculate_delay(1)
        assert delay >= 1.0  # base + jitter >= base

    def test_calculate_delay_max_limit(self) -> None:
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=5.0,
            multiplier=100.0,
            strategy=RetryStrategy.EXPONENTIAL,
        )
        handler = RetryHandler(config)
        delay = handler._calculate_delay(10)
        assert delay <= 5.0

    async def test_execute_async_success(self) -> None:
        config = RetryConfig(max_retries=3, initial_delay=0.0)
        handler = RetryHandler(config)

        async def async_func():
            return "async ok"

        result = await handler.execute_async(async_func)
        assert result == "async ok"

    async def test_execute_async_retries(self) -> None:
        config = RetryConfig(max_retries=3, initial_delay=0.0)
        handler = RetryHandler(config)
        count = [0]

        async def async_func():
            count[0] += 1
            if count[0] < 3:
                raise ValueError("retry me")
            return "recovered"

        result = await handler.execute_async(async_func)
        assert result == "recovered"
        assert count[0] == 3

    async def test_execute_async_max_retries_exceeded(self) -> None:
        config = RetryConfig(max_retries=2, initial_delay=0.0)
        handler = RetryHandler(config)

        async def always_fail():
            raise RuntimeError("async fail")

        with pytest.raises(MaxRetriesExceededError):
            await handler.execute_async(always_fail)

    async def test_execute_async_no_retry_unretriable(self) -> None:
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.0,
            retry_on_exceptions=(ValueError,),
        )
        handler = RetryHandler(config)

        async def raises_type_error():
            raise TypeError("not retried async")

        with pytest.raises(TypeError):
            await handler.execute_async(raises_type_error)


class TestRetryDecorator:
    def test_retry_decorator_sync_success(self) -> None:
        @retry(max_retries=3, initial_delay=0.0)
        def func():
            return "decorated sync ok"

        result = func()
        assert result == "decorated sync ok"

    def test_retry_decorator_sync_fails(self) -> None:
        @retry(max_retries=2, initial_delay=0.0)
        def always_fail():
            raise ValueError("always")

        with pytest.raises(MaxRetriesExceededError):
            always_fail()

    async def test_retry_decorator_async_success(self) -> None:
        @retry(max_retries=3, initial_delay=0.0)
        async def async_func():
            return "decorated async ok"

        result = await async_func()
        assert result == "decorated async ok"

    async def test_retry_decorator_async_fails(self) -> None:
        @retry(max_retries=2, initial_delay=0.0)
        async def always_fail():
            raise RuntimeError("async always fail")

        with pytest.raises(MaxRetriesExceededError):
            await always_fail()


class TestCircuitBreakerFull:
    def test_initial_state_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_call_success(self) -> None:
        cb = CircuitBreaker()
        result = cb.call(lambda: "ok")
        assert result == "ok"

    def test_failure_increments_count(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        def fail():
            raise RuntimeError("fail")

        try:
            cb.call(fail)
        except RuntimeError:
            pass

        assert cb.failure_count == 1

    def test_open_after_threshold(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=3, timeout=60.0)
        cb = CircuitBreaker(config)

        def fail():
            raise RuntimeError("fail")

        for _ in range(3):
            try:
                cb.call(fail)
            except RuntimeError:
                pass

        assert cb.state == CircuitState.OPEN

    def test_open_state_raises_circuit_error(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=2, timeout=60.0)
        cb = CircuitBreaker(config)

        def fail():
            raise RuntimeError("fail")

        for _ in range(2):
            try:
                cb.call(fail)
            except RuntimeError:
                pass

        with pytest.raises(CircuitBreakerError):
            cb.call(lambda: "blocked")

    def test_half_open_recovery(self) -> None:
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.01,  # very short
            success_threshold=1,
        )
        cb = CircuitBreaker(config)

        def fail():
            raise RuntimeError("fail")

        for _ in range(2):
            try:
                cb.call(fail)
            except RuntimeError:
                pass

        assert cb.state == CircuitState.OPEN

        time.sleep(0.05)

        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self) -> None:
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.01,
            success_threshold=2,
        )
        cb = CircuitBreaker(config)

        def fail():
            raise RuntimeError("fail")

        for _ in range(2):
            try:
                cb.call(fail)
            except RuntimeError:
                pass

        time.sleep(0.05)

        try:
            cb.call(fail)
        except RuntimeError:
            pass

        assert cb.state == CircuitState.OPEN

    def test_get_state(self) -> None:
        cb = CircuitBreaker()
        state = cb.get_state()
        assert "state" in state
        assert "failure_count" in state
        assert "success_rate" in state

    def test_reset(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=2, timeout=60.0)
        cb = CircuitBreaker(config)

        def fail():
            raise RuntimeError("fail")

        for _ in range(2):
            try:
                cb.call(fail)
            except RuntimeError:
                pass

        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_circuit_breaker_decorator(self) -> None:
        @circuit_breaker(failure_threshold=5, timeout=60.0)
        def good_func():
            return "decorated ok"

        result = good_func()
        assert result == "decorated ok"

    def test_success_reduces_failure_count(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config)

        def fail():
            raise RuntimeError("fail")

        try:
            cb.call(fail)
        except RuntimeError:
            pass

        assert cb.failure_count == 1

        cb.call(lambda: "ok")
        # Success should reduce failure count by 1
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_within_limit(self) -> None:
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=5, time_window=60))
        for _ in range(5):
            limiter.call(lambda: None)

    def test_raises_when_limit_exceeded(self) -> None:
        from beanllm.utils.exceptions import RateLimitError
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=2, time_window=60))
        limiter.call(lambda: None)
        limiter.call(lambda: None)
        with pytest.raises(RateLimitError):
            limiter.call(lambda: None)

    def test_call_returns_function_result(self) -> None:
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=10, time_window=60))
        result = limiter.call(lambda: 42)
        assert result == 42

    def test_get_status_returns_dict(self) -> None:
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=10, time_window=60))
        status = limiter.get_status()
        assert "current_calls" in status
        assert "max_calls" in status
        assert "calls_remaining" in status

    def test_calls_remaining_decreases(self) -> None:
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=5, time_window=60))
        initial = limiter.get_status()["calls_remaining"]
        limiter.call(lambda: None)
        after = limiter.get_status()["calls_remaining"]
        assert after == initial - 1

    def test_wait_and_call_success(self) -> None:
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=5, time_window=60))
        result = limiter.wait_and_call(lambda: "done")
        assert result == "done"

    def test_call_with_args(self) -> None:
        from beanllm.utils.resilience.rate_limiter import RateLimitConfig, RateLimiter

        limiter = RateLimiter(RateLimitConfig(max_calls=5, time_window=60))
        result = limiter.call(lambda x, y: x + y, 2, 3)
        assert result == 5

    def test_rate_limit_decorator_allows(self) -> None:
        from beanllm.utils.resilience.rate_limiter import rate_limit

        @rate_limit(max_calls=5, time_window=60)
        def fn():
            return "ok"

        assert fn() == "ok"

    def test_rate_limit_decorator_blocks(self) -> None:
        from beanllm.utils.exceptions import RateLimitError
        from beanllm.utils.resilience.rate_limiter import rate_limit

        @rate_limit(max_calls=1, time_window=60)
        def fn():
            return "ok"

        fn()
        with pytest.raises(RateLimitError):
            fn()


# ---------------------------------------------------------------------------
# AsyncTokenBucket
# ---------------------------------------------------------------------------


class TestAsyncTokenBucket:
    async def test_acquire_success_when_tokens_available(self) -> None:
        from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket

        bucket = AsyncTokenBucket(rate=10.0, capacity=5.0)
        result = await bucket.acquire(cost=1.0)
        assert result is True

    async def test_acquire_fails_when_no_tokens(self) -> None:
        from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket

        bucket = AsyncTokenBucket(rate=1.0, capacity=1.0)
        await bucket.acquire(cost=1.0)  # drain bucket
        result = await bucket.acquire(cost=1.0)
        assert result is False

    async def test_tokens_refill_over_time(self) -> None:
        from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket

        bucket = AsyncTokenBucket(rate=100.0, capacity=1.0)
        await bucket.acquire(cost=1.0)  # drain
        # Simulate time passage by adjusting last_update
        bucket.last_update -= 0.1  # 0.1s elapsed → +10 tokens → capped at 1.0
        result = await bucket.acquire(cost=1.0)
        assert result is True

    async def test_get_status_returns_dict(self) -> None:
        from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket

        bucket = AsyncTokenBucket(rate=10.0, capacity=20.0)
        status = bucket.get_status()
        assert "tokens" in status
        assert "capacity" in status
        assert "rate" in status

    async def test_wait_acquires_token(self) -> None:
        from beanllm.utils.resilience.rate_limiter import AsyncTokenBucket

        bucket = AsyncTokenBucket(rate=100.0, capacity=5.0)
        await bucket.wait(cost=1.0)
        status = bucket.get_status()
        assert status["tokens"] < 5.0


# ---------------------------------------------------------------------------
# CircuitBreaker.async_call()
# ---------------------------------------------------------------------------


class TestCircuitBreakerAsyncCall:
    """async_call() 경로 — 이번 PR에서 추가된 신규 메서드."""

    async def test_async_call_success_returns_result(self) -> None:
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

        async def good() -> str:
            return "ok"

        result = await cb.async_call(good)
        assert result == "ok"

    async def test_async_call_success_recorded(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(config)

        async def good() -> str:
            return "recorded"

        await cb.async_call(good)
        state = cb.get_state()
        assert state["state"] == CircuitState.CLOSED.value
        assert state["failure_count"] == 0

    async def test_async_call_failure_increments_count(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config)

        async def fail() -> None:
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError):
            await cb.async_call(fail)

        assert cb.failure_count == 1

    async def test_async_call_opens_after_threshold(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=3, timeout=60.0)
        cb = CircuitBreaker(config)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.async_call(fail)

        assert cb.state == CircuitState.OPEN

    async def test_async_call_open_raises_circuit_breaker_error(self) -> None:
        config = CircuitBreakerConfig(failure_threshold=2, timeout=60.0)
        cb = CircuitBreaker(config)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.async_call(fail)

        with pytest.raises(CircuitBreakerError):
            await cb.async_call(fail)

    async def test_async_call_circuit_breaker_error_not_counted_as_failure(self) -> None:
        # CircuitBreakerError 자체가 failure_count를 늘리면 안 됨
        # — 이미 OPEN이므로 추가 카운팅은 불필요하고 오해를 야기함
        config = CircuitBreakerConfig(failure_threshold=2, timeout=60.0)
        cb = CircuitBreaker(config)

        async def fail() -> None:
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.async_call(fail)

        count_before = cb.failure_count
        with pytest.raises(CircuitBreakerError):
            await cb.async_call(fail)

        assert cb.failure_count == count_before  # 증가 없음

    async def test_async_call_with_sync_callable(self) -> None:
        """sync 함수도 async_call로 호출 가능해야 함."""
        cb = CircuitBreaker()

        def sync_func() -> int:
            return 42

        result = await cb.async_call(sync_func)
        assert result == 42


# ---------------------------------------------------------------------------
# @retry decorator — async generator 경로
# ---------------------------------------------------------------------------


class TestRetryDecoratorAsyncGen:
    """inspect.isasyncgenfunction() 분기 — 이번 PR에서 추가된 신규 경로."""

    async def test_asyncgen_success_yields_all_items(self) -> None:
        @retry(max_retries=3, initial_delay=0.0)
        async def gen():
            for i in range(3):
                yield i

        result = [item async for item in gen()]
        assert result == [0, 1, 2]

    async def test_asyncgen_retries_on_transient_error(self) -> None:
        attempt = [0]

        @retry(max_retries=3, initial_delay=0.0, retry_on=(ValueError,))
        async def gen():
            attempt[0] += 1
            if attempt[0] < 2:
                raise ValueError("transient")
            yield "ok"

        result = [item async for item in gen()]
        assert result == ["ok"]
        assert attempt[0] == 2  # 1번 실패 후 2번째 성공

    async def test_asyncgen_raises_after_max_retries(self) -> None:
        @retry(max_retries=2, initial_delay=0.0, retry_on=(ValueError,))
        async def gen():
            raise ValueError("always")
            yield  # unreachable — makes this an async generator

        with pytest.raises(MaxRetriesExceededError):
            async for _ in gen():
                pass

    async def test_asyncgen_does_not_retry_cancelled_error(self) -> None:
        call_count = [0]

        @retry(max_retries=5, initial_delay=0.0)
        async def gen():
            call_count[0] += 1
            raise asyncio.CancelledError()
            yield  # makes this async generator

        with pytest.raises(asyncio.CancelledError):
            async for _ in gen():
                pass

        assert call_count[0] == 1  # 재시도 없음

    async def test_asyncgen_does_not_retry_non_matching_exception(self) -> None:
        call_count = [0]

        @retry(max_retries=5, initial_delay=0.0, retry_on=(ValueError,))
        async def gen():
            call_count[0] += 1
            raise TypeError("not in retry_on")
            yield

        with pytest.raises(TypeError):
            async for _ in gen():
                pass

        assert call_count[0] == 1  # 재시도 없음

    async def test_asyncgen_partial_yield_then_retry(self) -> None:
        """재시도 시 generator가 처음부터 재시작되는지 확인."""
        attempt = [0]
        yielded: list = []

        @retry(max_retries=3, initial_delay=0.0, retry_on=(RuntimeError,))
        async def gen():
            attempt[0] += 1
            yield f"a{attempt[0]}"
            if attempt[0] < 2:
                raise RuntimeError("fail mid-stream")
            yield f"b{attempt[0]}"

        async for item in gen():
            yielded.append(item)

        # 2번째 시도에서 성공: ["a1"] 실패 후 ["a2", "b2"] 수집
        assert attempt[0] == 2
        assert yielded == ["a1", "a2", "b2"]
