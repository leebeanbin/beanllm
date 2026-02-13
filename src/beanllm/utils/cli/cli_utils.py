"""
CLI utilities - Console, Rich optional imports, registry and hybrid manager helpers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Type

# Rich library (optional dependency)
RICH_AVAILABLE = False
_Panel: Optional[Type[Any]] = None
_Progress: Optional[Type[Any]] = None
_SpinnerColumn: Optional[Type[Any]] = None
_TextColumn: Optional[Type[Any]] = None
_Syntax: Optional[Type[Any]] = None
_Table: Optional[Type[Any]] = None
_Tree: Optional[Type[Any]] = None

try:
    from rich.panel import Panel as _PanelClass
    from rich.progress import Progress as _ProgressClass
    from rich.progress import SpinnerColumn as _SpinnerColumnClass
    from rich.progress import TextColumn as _TextColumnClass
    from rich.syntax import Syntax as _SyntaxClass
    from rich.table import Table as _TableClass
    from rich.tree import Tree as _TreeClass

    RICH_AVAILABLE = True
    _Panel = _PanelClass
    _Progress = _ProgressClass
    _SpinnerColumn = _SpinnerColumnClass
    _TextColumn = _TextColumnClass
    _Syntax = _SyntaxClass
    _Table = _TableClass
    _Tree = _TreeClass
except ImportError:
    pass

# Export for use by cli_commands, cli_scan, cli_analyze
Panel = _Panel
Progress = _Progress
SpinnerColumn = _SpinnerColumn
TextColumn = _TextColumn
Syntax = _Syntax
Table = _Table
Tree = _Tree

if TYPE_CHECKING:
    from rich.panel import Panel as PanelType
    from rich.progress import Progress as ProgressType, SpinnerColumn as SpinnerColumnType
    from rich.progress import TextColumn as TextColumnType
    from rich.syntax import Syntax as SyntaxType
    from rich.table import Table as TableType
    from rich.tree import Tree as TreeType

# Infrastructure imports with fallback
_console_getter: Any = None
_logo_printer: Any = None
_error_pattern: Any = None
_hybrid_manager_creator: Any = None
_registry_getter: Any = None

try:
    from beanllm.infrastructure.hybrid import create_hybrid_manager as _create_hybrid
    from beanllm.infrastructure.registry import get_model_registry as _get_registry
    from beanllm.ui import ErrorPattern as _ErrorPatternClass
    from beanllm.ui import get_console as _get_console_impl
    from beanllm.ui import print_logo as _print_logo_impl

    _console_getter = _get_console_impl
    _logo_printer = _print_logo_impl
    _error_pattern = _ErrorPatternClass
    _hybrid_manager_creator = _create_hybrid
    _registry_getter = _get_registry
except ImportError:
    pass


def get_console() -> Any:
    """Get console with fallback to simple print."""
    if _console_getter is not None:
        return _console_getter()

    class _FallbackConsole:
        def print(self, *args: Any, **kwargs: Any) -> None:
            print(*args, **kwargs)

        def rule(self, *args: Any, **kwargs: Any) -> None:
            pass

    return _FallbackConsole()


def print_logo(*args: Any, **kwargs: Any) -> None:
    """Print logo with fallback to no-op."""
    if _logo_printer is not None:
        _logo_printer(*args, **kwargs)


class ErrorPattern:
    """Error pattern with fallback to simple print."""

    @staticmethod
    def render(*args: Any, **kwargs: Any) -> None:
        if _error_pattern is not None:
            _error_pattern.render(*args, **kwargs)
        else:
            print(*args, **kwargs)


def create_hybrid_manager(*args: Any, **kwargs: Any) -> Any:
    """Create hybrid manager with import check."""
    if _hybrid_manager_creator is None:
        raise ImportError("hybrid_manager not available")
    return _hybrid_manager_creator(*args, **kwargs)


def get_model_registry() -> Any:
    """Get model registry with import check."""
    if _registry_getter is None:
        raise ImportError("model_registry not available")
    return _registry_getter()


console = get_console()
