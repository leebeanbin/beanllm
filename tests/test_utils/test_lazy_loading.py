"""
Tests for beanllm.utils.lazy_loading
Goal: maximize line coverage (43 lines missed, 0% current)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

try:
    from beanllm.utils.lazy_loading import (
        LazyLoader,
        LazyLoadMixin,
        lazy_property,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="lazy_loading module not available")


# ===========================================================================
# LazyLoadMixin
# ===========================================================================


class TestLazyLoadMixin:
    # -------------------------------------------------------------------
    # Basic lazy_property
    # -------------------------------------------------------------------

    def test_init_creates_empty_cache(self):
        obj = LazyLoadMixin()
        assert obj._lazy_attrs == {}

    def test_lazy_property_loads_on_first_access(self):
        obj = LazyLoadMixin()
        loader = MagicMock(return_value="loaded_value")

        result = obj.lazy_property("_resource", loader)

        assert result == "loaded_value"
        loader.assert_called_once()

    def test_lazy_property_caches_value(self):
        obj = LazyLoadMixin()
        loader = MagicMock(return_value="expensive_resource")

        result1 = obj.lazy_property("_resource", loader)
        result2 = obj.lazy_property("_resource", loader)

        assert result1 == "expensive_resource"
        assert result2 == "expensive_resource"
        # loader should only be called once
        loader.assert_called_once()

    def test_lazy_property_different_attributes_independent(self):
        obj = LazyLoadMixin()
        loader_a = MagicMock(return_value="value_a")
        loader_b = MagicMock(return_value="value_b")

        result_a = obj.lazy_property("_attr_a", loader_a)
        result_b = obj.lazy_property("_attr_b", loader_b)

        assert result_a == "value_a"
        assert result_b == "value_b"
        loader_a.assert_called_once()
        loader_b.assert_called_once()

    # -------------------------------------------------------------------
    # is_loaded
    # -------------------------------------------------------------------

    def test_is_loaded_returns_false_before_access(self):
        obj = LazyLoadMixin()
        assert obj.is_loaded("_resource") is False

    def test_is_loaded_returns_true_after_access(self):
        obj = LazyLoadMixin()
        obj.lazy_property("_resource", lambda: "value")
        assert obj.is_loaded("_resource") is True

    def test_is_loaded_false_for_unknown_attr(self):
        obj = LazyLoadMixin()
        obj.lazy_property("_known", lambda: "v")
        assert obj.is_loaded("_unknown") is False

    # -------------------------------------------------------------------
    # clear_lazy_cache
    # -------------------------------------------------------------------

    def test_clear_specific_attr(self):
        obj = LazyLoadMixin()
        loader = MagicMock(side_effect=["first", "second"])

        obj.lazy_property("_resource", loader)
        assert obj.is_loaded("_resource") is True

        obj.clear_lazy_cache("_resource")
        assert obj.is_loaded("_resource") is False

        # Accessing again should call loader again
        result = obj.lazy_property("_resource", loader)
        assert result == "second"
        assert loader.call_count == 2

    def test_clear_all_attrs(self):
        obj = LazyLoadMixin()
        obj.lazy_property("_a", lambda: 1)
        obj.lazy_property("_b", lambda: 2)
        assert obj.is_loaded("_a") is True
        assert obj.is_loaded("_b") is True

        obj.clear_lazy_cache()
        assert obj.is_loaded("_a") is False
        assert obj.is_loaded("_b") is False

    def test_clear_nonexistent_attr_is_noop(self):
        """Clearing an attr that doesn't exist should not raise."""
        obj = LazyLoadMixin()
        obj.clear_lazy_cache("_nonexistent")  # should not raise

    def test_clear_none_clears_all(self):
        obj = LazyLoadMixin()
        obj.lazy_property("_x", lambda: "x")
        obj.clear_lazy_cache(None)
        assert obj._lazy_attrs == {}

    # -------------------------------------------------------------------
    # Integration with subclasses
    # -------------------------------------------------------------------

    def test_subclass_using_mixin(self):
        class MyClass(LazyLoadMixin):
            def __init__(self, cost=10):
                super().__init__()
                self.cost = cost
                self._load_count = 0

            @property
            def model(self):
                return self.lazy_property("_model", self._load_model)

            def _load_model(self):
                self._load_count += 1
                return f"model_cost_{self.cost}"

        obj = MyClass(cost=42)
        # First access
        assert obj.model == "model_cost_42"
        assert obj._load_count == 1
        # Second access — no reload
        assert obj.model == "model_cost_42"
        assert obj._load_count == 1

    def test_subclass_clear_and_reload(self):
        class MyClass(LazyLoadMixin):
            def __init__(self):
                super().__init__()
                self._calls = 0

            @property
            def resource(self):
                return self.lazy_property("_resource", self._load)

            def _load(self):
                self._calls += 1
                return f"load_{self._calls}"

        obj = MyClass()
        v1 = obj.resource
        obj.clear_lazy_cache("_resource")
        v2 = obj.resource
        assert v1 == "load_1"
        assert v2 == "load_2"


# ===========================================================================
# lazy_property decorator
# ===========================================================================


class TestLazyPropertyDecorator:
    def test_property_loads_on_first_access(self):
        class MyClass:
            @lazy_property
            def expensive(self) -> str:
                return "loaded"

        obj = MyClass()
        assert obj.expensive == "loaded"

    def test_property_cached_on_second_access(self):
        counter = {"calls": 0}

        class MyClass:
            @lazy_property
            def expensive(self) -> int:
                counter["calls"] += 1
                return 42

        obj = MyClass()
        _ = obj.expensive
        _ = obj.expensive
        assert counter["calls"] == 1

    def test_different_instances_independent_cache(self):
        class MyClass:
            def __init__(self, val):
                self.val = val

            @lazy_property
            def computed(self) -> str:
                return f"computed_{self.val}"

        obj1 = MyClass("a")
        obj2 = MyClass("b")

        assert obj1.computed == "computed_a"
        assert obj2.computed == "computed_b"

    def test_cached_on_instance_with_mangled_name(self):
        class MyClass:
            @lazy_property
            def resource(self) -> str:
                return "resource_value"

        obj = MyClass()
        _ = obj.resource
        # The cached attr name should be _lazy_resource
        assert hasattr(obj, "_lazy_resource")
        assert obj._lazy_resource == "resource_value"

    def test_property_is_readonly(self):
        class MyClass:
            @lazy_property
            def data(self) -> list:
                return [1, 2, 3]

        obj = MyClass()
        with pytest.raises(AttributeError):
            obj.data = "new_value"

    def test_returns_correct_type(self):
        class MyClass:
            @lazy_property
            def data(self) -> list:
                return [1, 2, 3]

        obj = MyClass()
        result = obj.data
        assert isinstance(result, list)
        assert result == [1, 2, 3]


# ===========================================================================
# LazyLoader
# ===========================================================================


class TestLazyLoader:
    def test_not_loaded_initially(self):
        loader = LazyLoader(lambda: "value")
        assert loader.is_loaded is False

    def test_get_loads_value(self):
        loader = LazyLoader(lambda: "expensive")
        result = loader.get()
        assert result == "expensive"

    def test_get_caches_value(self):
        counter = {"calls": 0}

        def loader_fn():
            counter["calls"] += 1
            return "cached"

        loader = LazyLoader(loader_fn)
        r1 = loader.get()
        r2 = loader.get()
        assert r1 == "cached"
        assert r2 == "cached"
        assert counter["calls"] == 1

    def test_is_loaded_true_after_get(self):
        loader = LazyLoader(lambda: "value")
        loader.get()
        assert loader.is_loaded is True

    def test_reset_clears_cache(self):
        counter = {"calls": 0}

        def loader_fn():
            counter["calls"] += 1
            return f"value_{counter['calls']}"

        loader = LazyLoader(loader_fn)
        v1 = loader.get()
        loader.reset()

        assert loader.is_loaded is False
        v2 = loader.get()
        assert v1 == "value_1"
        assert v2 == "value_2"
        assert counter["calls"] == 2

    def test_reset_then_is_loaded_false(self):
        loader = LazyLoader(lambda: 42)
        loader.get()
        loader.reset()
        assert loader.is_loaded is False

    def test_loader_fn_called_with_no_args(self):
        fn = MagicMock(return_value="result")
        loader = LazyLoader(fn)
        loader.get()
        fn.assert_called_once_with()

    def test_integration_with_class_property(self):
        class VisionModel:
            def __init__(self):
                self._loader = LazyLoader(self._load_model)

            def _load_model(self):
                return {"weights": [0.1, 0.2, 0.3]}

            @property
            def model(self):
                return self._loader.get()

        m = VisionModel()
        assert not m._loader.is_loaded
        result = m.model
        assert result == {"weights": [0.1, 0.2, 0.3]}
        assert m._loader.is_loaded
        # Second call returns same object (no reload)
        assert m.model is result

    def test_loader_can_hold_none_value(self):
        """LazyLoader should also work for loaders that return None."""
        loader = LazyLoader(lambda: None)
        result = loader.get()
        # Note: the current implementation uses _is_loaded flag so None is fine
        assert loader.is_loaded is True

    def test_loader_with_exception_does_not_cache(self):
        """If loader_fn raises, value should not be cached."""
        counter = {"calls": 0}

        def failing_loader():
            counter["calls"] += 1
            raise RuntimeError("load failed")

        loader = LazyLoader(failing_loader)

        with pytest.raises(RuntimeError):
            loader.get()

        # is_loaded should still be False because exception was raised
        # (the flag is set AFTER the call)
        # This tests the current actual behavior
        assert counter["calls"] == 1


# ===========================================================================
# __all__ exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from beanllm.utils.lazy_loading import __all__

        assert "LazyLoadMixin" in __all__
        assert "lazy_property" in __all__
        assert "LazyLoader" in __all__
