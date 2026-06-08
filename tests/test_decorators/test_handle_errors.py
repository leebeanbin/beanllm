"""
Decorator 테스트 - handle_errors, log_errors, validate_input
"""

import pytest

from beanllm.decorators.error_handler import handle_errors, log_errors
from beanllm.decorators.validation import validate_input


class TestHandleErrorsAsync:
    async def test_success_reraise_true(self) -> None:
        @handle_errors(reraise=True)
        async def good_func():
            return "ok"

        result = await good_func()
        assert result == "ok"

    async def test_value_error_reraise_true(self) -> None:
        @handle_errors(reraise=True)
        async def bad_func():
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await bad_func()

    async def test_general_error_reraise_true(self) -> None:
        @handle_errors(reraise=True)
        async def bad_func():
            raise RuntimeError("runtime fail")

        with pytest.raises(RuntimeError):
            await bad_func()

    async def test_value_error_no_reraise_returns_default(self) -> None:
        @handle_errors(reraise=False, default_return="fallback")
        async def bad_func():
            raise ValueError("invalid")

        result = await bad_func()
        assert result == "fallback"

    async def test_general_error_no_reraise_returns_default(self) -> None:
        @handle_errors(reraise=False, default_return=42)
        async def bad_func():
            raise RuntimeError("fail")

        result = await bad_func()
        assert result == 42

    async def test_no_reraise_no_default_returns_none(self) -> None:
        @handle_errors(reraise=False)
        async def bad_func():
            raise Exception("err")

        result = await bad_func()
        assert result is None

    async def test_custom_error_message(self) -> None:
        @handle_errors(error_message="Custom failure msg", reraise=True)
        async def bad_func():
            raise ValueError("original")

        with pytest.raises(ValueError):
            await bad_func()

    async def test_preserves_function_name(self) -> None:
        @handle_errors()
        async def my_async_function():
            return "x"

        assert my_async_function.__name__ == "my_async_function"


class TestHandleErrorsSync:
    def test_sync_success(self) -> None:
        @handle_errors(reraise=True)
        def good_func():
            return "sync ok"

        result = good_func()
        assert result == "sync ok"

    def test_sync_value_error_reraise(self) -> None:
        @handle_errors(reraise=True)
        def bad_func():
            raise ValueError("sync bad")

        with pytest.raises(ValueError):
            bad_func()

    def test_sync_value_error_no_reraise(self) -> None:
        @handle_errors(reraise=False, default_return="default")
        def bad_func():
            raise ValueError("sync bad")

        result = bad_func()
        assert result == "default"

    def test_sync_general_error_no_reraise(self) -> None:
        @handle_errors(reraise=False, default_return=0)
        def bad_func():
            raise RuntimeError("oops")

        result = bad_func()
        assert result == 0

    def test_sync_preserves_function_name(self) -> None:
        @handle_errors()
        def my_sync_function():
            return "x"

        assert my_sync_function.__name__ == "my_sync_function"


class TestHandleErrorsAsyncGenerator:
    async def test_async_gen_success(self) -> None:
        @handle_errors(reraise=True)
        async def good_gen():
            yield 1
            yield 2
            yield 3

        results = [x async for x in good_gen()]
        assert results == [1, 2, 3]

    async def test_async_gen_value_error_reraise(self) -> None:
        @handle_errors(reraise=True)
        async def bad_gen():
            yield 1
            raise ValueError("gen error")

        with pytest.raises(ValueError):
            _ = [x async for x in bad_gen()]

    async def test_async_gen_general_error_reraise(self) -> None:
        @handle_errors(reraise=True)
        async def bad_gen():
            yield 1
            raise RuntimeError("runtime gen")

        with pytest.raises(RuntimeError):
            _ = [x async for x in bad_gen()]

    async def test_async_gen_no_reraise_no_default(self) -> None:
        @handle_errors(reraise=False)
        async def bad_gen():
            yield 1
            raise ValueError("stop")

        results = [x async for x in bad_gen()]
        assert results == [1]


class TestHandleErrorsSyncGenerator:
    def test_sync_gen_success(self) -> None:
        @handle_errors(reraise=True)
        def good_gen():
            yield "a"
            yield "b"

        results = list(good_gen())
        assert results == ["a", "b"]

    def test_sync_gen_value_error_reraise(self) -> None:
        @handle_errors(reraise=True)
        def bad_gen():
            yield 1
            raise ValueError("gen fail")

        with pytest.raises(ValueError):
            list(bad_gen())

    def test_sync_gen_general_error_reraise(self) -> None:
        @handle_errors(reraise=True)
        def bad_gen():
            yield "x"
            raise RuntimeError("runtime gen")

        with pytest.raises(RuntimeError):
            list(bad_gen())

    def test_sync_gen_no_reraise_stops(self) -> None:
        @handle_errors(reraise=False)
        def bad_gen():
            yield 1
            raise RuntimeError("stop")

        results = list(bad_gen())
        assert results == [1]


class TestLogErrors:
    async def test_log_errors_async_success(self) -> None:
        @log_errors
        async def good_func():
            return "logged ok"

        result = await good_func()
        assert result == "logged ok"

    async def test_log_errors_async_reraises(self) -> None:
        @log_errors
        async def bad_func():
            raise ValueError("logged error")

        with pytest.raises(ValueError, match="logged error"):
            await bad_func()

    async def test_log_errors_async_preserves_name(self) -> None:
        @log_errors
        async def my_named_func():
            return 1

        assert my_named_func.__name__ == "my_named_func"

    def test_log_errors_sync_success(self) -> None:
        @log_errors
        def good_func():
            return "sync logged"

        result = good_func()
        assert result == "sync logged"

    def test_log_errors_sync_reraises(self) -> None:
        @log_errors
        def bad_func():
            raise RuntimeError("sync logged error")

        with pytest.raises(RuntimeError):
            bad_func()

    def test_log_errors_sync_preserves_name(self) -> None:
        @log_errors
        def another_func():
            return 2

        assert another_func.__name__ == "another_func"


class TestValidateInput:
    async def test_required_param_present(self) -> None:
        @validate_input(required_params=["x"])
        async def func(x, y=None):
            return x

        result = await func(x="hello")
        assert result == "hello"

    async def test_required_param_missing_raises(self) -> None:
        @validate_input(required_params=["query"])
        async def func(query=None):
            return query

        with pytest.raises(ValueError, match="query"):
            await func(query=None)

    async def test_param_type_valid(self) -> None:
        @validate_input(param_types={"count": int})
        async def func(count):
            return count * 2

        result = await func(count=5)
        assert result == 10

    async def test_param_type_invalid_raises(self) -> None:
        @validate_input(param_types={"score": float})
        async def func(score):
            return score

        with pytest.raises(TypeError, match="score"):
            await func(score="not_a_float")

    async def test_param_range_valid(self) -> None:
        @validate_input(param_ranges={"temperature": (0.0, 2.0)})
        async def func(temperature):
            return temperature

        result = await func(temperature=1.5)
        assert result == 1.5

    async def test_param_range_below_min_raises(self) -> None:
        @validate_input(param_ranges={"k": (1, None)})
        async def func(k):
            return k

        with pytest.raises(ValueError, match="k"):
            await func(k=0)

    async def test_param_range_above_max_raises(self) -> None:
        @validate_input(param_ranges={"temperature": (0.0, 2.0)})
        async def func(temperature):
            return temperature

        with pytest.raises(ValueError, match="temperature"):
            await func(temperature=3.0)

    async def test_combined_validations(self) -> None:
        @validate_input(
            required_params=["text"],
            param_types={"max_tokens": int},
            param_ranges={"max_tokens": (1, 4096)},
        )
        async def func(text, max_tokens=100):
            return f"{text}:{max_tokens}"

        result = await func(text="hello", max_tokens=200)
        assert result == "hello:200"

    def test_sync_required_param_present(self) -> None:
        @validate_input(required_params=["name"])
        def func(name, value=0):
            return name

        result = func(name="test")
        assert result == "test"

    def test_sync_param_type_invalid_raises(self) -> None:
        @validate_input(param_types={"count": int})
        def func(count):
            return count

        with pytest.raises(TypeError):
            func(count="bad")

    def test_sync_preserves_name(self) -> None:
        @validate_input(required_params=["x"])
        def my_validator_func(x):
            return x

        assert my_validator_func.__name__ == "my_validator_func"
