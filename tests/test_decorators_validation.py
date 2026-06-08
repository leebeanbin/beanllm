"""Tests for decorators/validation.py - validate_input decorator."""

import pytest

from beanllm.decorators.validation import validate_input

# ---------------------------------------------------------------------------
# Helper functions decorated with validate_input
# ---------------------------------------------------------------------------


def _sync_fn_required(messages, model=None):
    return f"messages={messages}, model={model}"


def _sync_fn_type(value, factor=1.0):
    return value * factor


def _sync_fn_range(temperature=0.7, max_tokens=100):
    return temperature, max_tokens


# ---------------------------------------------------------------------------
# Sync function — required params
# ---------------------------------------------------------------------------


class TestValidateInputRequired:
    def test_passes_with_required_param_present(self):
        decorated = validate_input(required_params=["messages"])(_sync_fn_required)
        result = decorated(messages=["hello"])
        assert "messages" in result

    def test_raises_value_error_when_required_param_missing(self):
        decorated = validate_input(required_params=["messages"])(_sync_fn_required)
        with pytest.raises((ValueError, TypeError)):
            decorated()

    def test_raises_value_error_when_required_param_is_none(self):
        decorated = validate_input(required_params=["messages"])(_sync_fn_required)
        with pytest.raises(ValueError, match="messages"):
            decorated(messages=None)

    def test_multiple_required_params_all_present(self):
        decorated = validate_input(required_params=["messages", "model"])(_sync_fn_required)
        result = decorated(messages=["hi"], model="gpt-4o")
        assert "gpt-4o" in result

    def test_multiple_required_params_one_missing(self):
        decorated = validate_input(required_params=["messages", "model"])(_sync_fn_required)
        with pytest.raises(ValueError, match="model"):
            decorated(messages=["hi"], model=None)

    def test_no_required_params_always_passes(self):
        decorated = validate_input()(_sync_fn_required)
        result = decorated(messages=["hello"])
        assert result is not None


# ---------------------------------------------------------------------------
# Sync function — type checking
# ---------------------------------------------------------------------------


class TestValidateInputTypes:
    def test_passes_with_correct_type(self):
        decorated = validate_input(param_types={"factor": float})(_sync_fn_type)
        result = decorated(value=2, factor=1.5)
        assert result == 3.0

    def test_raises_type_error_for_wrong_type(self):
        decorated = validate_input(param_types={"factor": float})(_sync_fn_type)
        with pytest.raises(TypeError, match="factor"):
            decorated(value=2, factor="oops")

    def test_none_value_skips_type_check(self):
        def fn(x=None):
            return x

        decorated = validate_input(param_types={"x": int})(fn)
        result = decorated(x=None)
        assert result is None

    def test_multiple_type_checks_all_pass(self):
        def fn(a, b):
            return a + b

        decorated = validate_input(param_types={"a": int, "b": int})(fn)
        result = decorated(a=1, b=2)
        assert result == 3


# ---------------------------------------------------------------------------
# Sync function — range checking
# ---------------------------------------------------------------------------


class TestValidateInputRanges:
    def test_passes_when_in_range(self):
        decorated = validate_input(param_ranges={"temperature": (0.0, 2.0)})(_sync_fn_range)
        temp, _ = decorated(temperature=1.0)
        assert temp == 1.0

    def test_raises_when_below_min(self):
        decorated = validate_input(param_ranges={"temperature": (0.0, 2.0)})(_sync_fn_range)
        with pytest.raises(ValueError, match="temperature"):
            decorated(temperature=-0.1)

    def test_raises_when_above_max(self):
        decorated = validate_input(param_ranges={"temperature": (0.0, 2.0)})(_sync_fn_range)
        with pytest.raises(ValueError, match="temperature"):
            decorated(temperature=2.1)

    def test_passes_at_boundary_values(self):
        decorated = validate_input(param_ranges={"temperature": (0.0, 2.0)})(_sync_fn_range)
        temp, _ = decorated(temperature=0.0)
        assert temp == 0.0
        temp, _ = decorated(temperature=2.0)
        assert temp == 2.0

    def test_no_min_allows_negative(self):
        decorated = validate_input(param_ranges={"temperature": (None, 2.0)})(_sync_fn_range)
        temp, _ = decorated(temperature=-100.0)
        assert temp == -100.0

    def test_no_max_allows_large(self):
        decorated = validate_input(param_ranges={"max_tokens": (1, None)})(_sync_fn_range)
        _, tokens = decorated(max_tokens=999999)
        assert tokens == 999999

    def test_range_with_none_value_skips_check(self):
        def fn(x=None):
            return x

        decorated = validate_input(param_ranges={"x": (0, 100)})(fn)
        result = decorated(x=None)
        assert result is None


# ---------------------------------------------------------------------------
# Generator function
# ---------------------------------------------------------------------------


class TestValidateInputGenerator:
    def test_sync_generator_passes_valid_input(self):
        @validate_input(required_params=["n"])
        def gen(n):
            yield from range(n)

        items = list(gen(n=3))
        assert items == [0, 1, 2]

    def test_sync_generator_raises_on_invalid_input(self):
        @validate_input(required_params=["n"])
        def gen(n):
            yield from range(n)

        with pytest.raises(ValueError, match="n"):
            list(gen(n=None))

    def test_sync_generator_type_check(self):
        @validate_input(param_types={"step": int})
        def gen(start, step=1):
            yield start + step

        with pytest.raises(TypeError, match="step"):
            list(gen(start=0, step="bad"))


# ---------------------------------------------------------------------------
# Async generator function
# ---------------------------------------------------------------------------


class TestValidateInputAsyncGenerator:
    async def test_async_generator_passes_valid_input(self):
        @validate_input(required_params=["n"])
        async def agen(n):
            for i in range(n):
                yield i

        items = [i async for i in agen(n=3)]
        assert items == [0, 1, 2]

    async def test_async_generator_raises_on_missing_required(self):
        @validate_input(required_params=["n"])
        async def agen(n):
            yield n

        with pytest.raises(ValueError, match="n"):
            async for _ in agen(n=None):
                pass

    async def test_async_generator_type_check(self):
        @validate_input(param_types={"factor": float})
        async def agen(value, factor=1.0):
            yield value * factor

        with pytest.raises(TypeError, match="factor"):
            async for _ in agen(value=1, factor="bad"):
                pass


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------


class TestValidateInputCombined:
    def test_required_and_type_and_range_all_pass(self):
        @validate_input(
            required_params=["value"],
            param_types={"value": float},
            param_ranges={"value": (0.0, 1.0)},
        )
        def fn(value):
            return value

        result = fn(value=0.5)
        assert result == 0.5

    def test_required_checked_first_then_type(self):
        @validate_input(
            required_params=["value"],
            param_types={"value": int},
        )
        def fn(value=None):
            return value

        with pytest.raises(ValueError, match="value"):
            fn(value=None)

    def test_preserves_function_name(self):
        @validate_input(required_params=["x"])
        def my_function(x):
            return x

        assert my_function.__name__ == "my_function"
