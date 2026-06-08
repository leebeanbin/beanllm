"""
Tests for beanllm.utils.dependency
Goal: maximize line coverage (47 lines missed, 0% current)
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

try:
    from beanllm.utils.dependency import (
        DependencyManager,
        check_available,
        require,
        require_any,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="dependency module not available")


# ===========================================================================
# DependencyManager.check_available
# ===========================================================================


class TestCheckAvailable:
    def test_returns_true_for_installed_package(self):
        # 'os' is always available
        assert DependencyManager.check_available("os") is True

    def test_returns_true_for_multiple_installed_packages(self):
        assert DependencyManager.check_available("os", "sys") is True

    def test_returns_false_for_nonexistent_package(self):
        assert DependencyManager.check_available("__nonexistent_pkg_xyz__") is False

    def test_returns_false_when_any_package_missing(self):
        # 'os' exists but __nonexistent_pkg_xyz__ does not
        assert DependencyManager.check_available("os", "__nonexistent_pkg_xyz__") is False

    def test_convenience_alias(self):
        """Module-level check_available should be the same as DependencyManager.check_available."""
        assert check_available("os") is True
        assert check_available("__nonexistent__") is False


# ===========================================================================
# DependencyManager.get_install_command
# ===========================================================================


class TestGetInstallCommand:
    def test_known_package_returns_custom_command(self):
        cmd = DependencyManager.get_install_command("transformers")
        assert cmd == "pip install transformers"

    def test_known_package_torch(self):
        cmd = DependencyManager.get_install_command("torch")
        assert "torch" in cmd

    def test_known_package_faiss(self):
        cmd = DependencyManager.get_install_command("faiss")
        assert "faiss" in cmd

    def test_unknown_package_returns_generic_command(self):
        cmd = DependencyManager.get_install_command("some_unknown_lib")
        assert cmd == "pip install some_unknown_lib"

    def test_all_registered_packages_have_commands(self):
        for pkg in DependencyManager._INSTALL_MSGS:
            cmd = DependencyManager.get_install_command(pkg)
            assert isinstance(cmd, str)
            assert len(cmd) > 0


# ===========================================================================
# DependencyManager.require (decorator)
# ===========================================================================


class TestRequireDecorator:
    def test_function_executes_when_dependency_available(self):
        @DependencyManager.require("os")
        def my_func():
            return "executed"

        result = my_func()
        assert result == "executed"

    def test_function_executes_with_multiple_available_deps(self):
        @DependencyManager.require("os", "sys")
        def my_func(x, y):
            return x + y

        assert my_func(1, 2) == 3

    def test_raises_import_error_when_dep_missing(self):
        @DependencyManager.require("__nonexistent_pkg_xyz__")
        def my_func():
            return "executed"

        with pytest.raises(ImportError, match="__nonexistent_pkg_xyz__"):
            my_func()

    def test_error_message_contains_install_command(self):
        @DependencyManager.require("transformers")
        def my_func():
            return "executed"

        # transformers is in _INSTALL_MSGS
        try:
            my_func()
        except ImportError as e:
            assert "pip install transformers" in str(e)
        except Exception:
            pass  # might succeed if transformers is installed

    def test_error_message_for_unknown_package(self):
        @DependencyManager.require("__some_custom_pkg__")
        def my_func():
            pass

        with pytest.raises(ImportError) as exc_info:
            my_func()

        assert "pip install __some_custom_pkg__" in str(exc_info.value)

    def test_decorator_preserves_function_name(self):
        @DependencyManager.require("os")
        def my_special_func():
            pass

        assert my_special_func.__name__ == "my_special_func"

    def test_decorator_preserves_args_and_kwargs(self):
        @DependencyManager.require("os")
        def add(a, b, *, c=0):
            return a + b + c

        assert add(1, 2, c=3) == 6

    def test_convenience_require_alias(self):
        @require("os")
        def func():
            return "ok"

        assert func() == "ok"

    def test_multiple_packages_first_missing(self):
        """If the first package is missing, ImportError should be raised."""

        @DependencyManager.require("__missing_pkg__", "os")
        def func():
            return "ok"

        with pytest.raises(ImportError):
            func()

    def test_multiple_packages_last_missing(self):
        """If the last package is missing, ImportError should be raised."""

        @DependencyManager.require("os", "sys", "__missing_pkg__")
        def func():
            return "ok"

        with pytest.raises(ImportError):
            func()


# ===========================================================================
# DependencyManager.require_any (decorator)
# ===========================================================================


class TestRequireAnyDecorator:
    def test_function_executes_when_one_of_group_available(self):
        @DependencyManager.require_any(("os", "__nonexistent__"))
        def my_func():
            return "executed"

        result = my_func()
        assert result == "executed"

    def test_function_executes_when_all_of_group_available(self):
        @DependencyManager.require_any(("os", "sys"))
        def my_func():
            return "both available"

        result = my_func()
        assert result == "both available"

    def test_raises_when_none_of_group_available(self):
        @DependencyManager.require_any(("__nonexistent_a__", "__nonexistent_b__"))
        def my_func():
            pass

        with pytest.raises(ImportError, match="At least one of"):
            my_func()

    def test_raises_when_one_group_completely_missing(self):
        @DependencyManager.require_any(
            ("os",),  # first group OK
            ("__nonexistent_a__", "__nonexistent_b__"),  # second group missing
        )
        def my_func():
            pass

        with pytest.raises(ImportError):
            my_func()

    def test_multiple_groups_all_satisfied(self):
        @DependencyManager.require_any(
            ("os",),
            ("sys",),
        )
        def my_func():
            return "all groups ok"

        assert my_func() == "all groups ok"

    def test_error_message_contains_package_names(self):
        @DependencyManager.require_any(("__pkg_alpha__", "__pkg_beta__"))
        def my_func():
            pass

        with pytest.raises(ImportError) as exc_info:
            my_func()

        msg = str(exc_info.value)
        assert "__pkg_alpha__" in msg or "__pkg_beta__" in msg

    def test_error_message_contains_install_commands(self):
        @DependencyManager.require_any(("transformers",))
        def my_func():
            pass

        # transformers might or might not be installed; if missing, check message
        try:
            my_func()
        except ImportError as e:
            assert "pip install" in str(e)

    def test_convenience_require_any_alias(self):
        @require_any(("os",))
        def func():
            return "alias ok"

        assert func() == "alias ok"

    def test_decorator_preserves_args(self):
        @DependencyManager.require_any(("os",))
        def compute(x, y):
            return x * y

        assert compute(3, 4) == 12


# ===========================================================================
# __all__ exports
# ===========================================================================


class TestModuleExports:
    def test_all_exports(self):
        from beanllm.utils.dependency import __all__

        assert "DependencyManager" in __all__
        assert "require" in __all__
        assert "check_available" in __all__
        assert "require_any" in __all__
