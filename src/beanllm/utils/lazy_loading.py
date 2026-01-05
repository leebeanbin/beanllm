"""
Lazy Loading Utilities - Deferred initialization pattern

Replaces 23 duplicate lazy loading implementations across the codebase.
"""

from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

T = TypeVar('T')


class LazyLoadMixin:
    """
    Mixin for lazy loading attributes

    Eliminates duplicate lazy loading patterns:
    - Before: 23 occurrences of _model = None + if check
    - After: 1 centralized implementation

    Example:
        >>> class SAMWrapper(LazyLoadMixin):
        ...     def __init__(self, model_type: str = "sam3_hiera_large"):
        ...         super().__init__()
        ...         self.model_type = model_type
        ...
        ...     @property
        ...     def model(self):
        ...         return self.lazy_property("_model", self._load_model_impl)
        ...
        ...     def _load_model_impl(self):
        ...         from sam3.build_sam import build_sam3
        ...         return build_sam3(self.model_type)
    """

    def __init__(self):
        """Initialize lazy loading storage"""
        self._lazy_attrs: Dict[str, Any] = {}

    def lazy_property(self, attr_name: str, loader_func: Callable[[], T]) -> T:
        """
        Get or create a lazy-loaded attribute

        Args:
            attr_name: Name of the attribute to cache
            loader_func: Function to call if attribute not cached

        Returns:
            Cached or newly loaded attribute value

        Example:
            >>> def load_expensive_resource():
            ...     return "expensive resource"
            >>> obj = LazyLoadMixin()
            >>> result = obj.lazy_property("_resource", load_expensive_resource)
        """
        if attr_name not in self._lazy_attrs:
            self._lazy_attrs[attr_name] = loader_func()
        return self._lazy_attrs[attr_name]

    def clear_lazy_cache(self, attr_name: Optional[str] = None):
        """
        Clear lazy-loaded cache

        Args:
            attr_name: Specific attribute to clear, or None to clear all

        Example:
            >>> obj.clear_lazy_cache("_model")  # Clear specific
            >>> obj.clear_lazy_cache()          # Clear all
        """
        if attr_name is None:
            self._lazy_attrs.clear()
        elif attr_name in self._lazy_attrs:
            del self._lazy_attrs[attr_name]

    def is_loaded(self, attr_name: str) -> bool:
        """
        Check if attribute is loaded

        Args:
            attr_name: Name of the attribute to check

        Returns:
            True if attribute is cached, False otherwise

        Example:
            >>> if not obj.is_loaded("_model"):
            ...     print("Model not yet loaded")
        """
        return attr_name in self._lazy_attrs


def lazy_property(loader_func: Callable[[Any], T]) -> property:
    """
    Decorator for creating lazy properties (without mixin)

    Args:
        loader_func: Function to load the property value

    Returns:
        Property descriptor with lazy loading

    Example:
        >>> class MyClass:
        ...     @lazy_property
        ...     def expensive_resource(self) -> str:
        ...         print("Loading...")
        ...         return "expensive resource"
        >>>
        >>> obj = MyClass()
        >>> obj.expensive_resource  # Prints "Loading..."
        'expensive resource'
        >>> obj.expensive_resource  # No print, cached
        'expensive resource'
    """
    attr_name = f"_lazy_{loader_func.__name__}"

    @wraps(loader_func)
    def getter(self: Any) -> T:
        if not hasattr(self, attr_name):
            setattr(self, attr_name, loader_func(self))
        return getattr(self, attr_name)

    return property(getter)


class LazyLoader(Generic[T]):
    """
    Standalone lazy loader (no inheritance needed)

    Example:
        >>> class VisionModel:
        ...     def __init__(self):
        ...         self._model_loader = LazyLoader(self._load_model)
        ...
        ...     def _load_model(self):
        ...         from transformers import AutoModel
        ...         return AutoModel.from_pretrained("model-name")
        ...
        ...     @property
        ...     def model(self):
        ...         return self._model_loader.get()
    """

    def __init__(self, loader_func: Callable[[], T]):
        """
        Initialize lazy loader

        Args:
            loader_func: Function to call when loading is needed
        """
        self._loader_func = loader_func
        self._value: Optional[T] = None
        self._is_loaded = False

    def get(self) -> T:
        """
        Get the value, loading if necessary

        Returns:
            Loaded value
        """
        if not self._is_loaded:
            self._value = self._loader_func()
            self._is_loaded = True
        return self._value  # type: ignore

    def reset(self):
        """Reset the loader, clearing cached value"""
        self._value = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if value is loaded"""
        return self._is_loaded


# Example usage patterns for migration
"""
# Pattern 1: Using Mixin (for classes you control)
class SAMWrapper(BaseVisionTaskModel, LazyLoadMixin):
    def __init__(self, model_type: str = "sam3_hiera_large"):
        super().__init__()
        self.model_type = model_type

    @property
    def model(self):
        return self.lazy_property("_model", self._load_model_impl)

    def _load_model_impl(self):
        from sam3.build_sam import build_sam3
        return build_sam3(self.model_type)


# Pattern 2: Using decorator (simplest)
class Florence2Wrapper:
    @lazy_property
    def model(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained("florence-2")


# Pattern 3: Using LazyLoader (most flexible)
class YOLOWrapper:
    def __init__(self):
        self._model_loader = LazyLoader(self._load_model)

    def _load_model(self):
        from ultralytics import YOLO
        return YOLO("yolov12.pt")

    @property
    def model(self):
        return self._model_loader.get()
"""


__all__ = [
    "LazyLoadMixin",
    "lazy_property",
    "LazyLoader",
]
