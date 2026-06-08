"""
Tests for beanllm.utils.cli.admin_system_commands
Goal: maximize line coverage (49 lines missed, 0% current)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

try:
    from beanllm.utils.cli.admin_system_commands import (
        RICH_AVAILABLE,
        console,
        get_console,
        launch_dashboard,
        print_error,
        print_help_admin,
        print_info,
        print_success,
    )

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="admin_system_commands not available")


# ===========================================================================
# get_console
# ===========================================================================


class TestGetConsole:
    def test_get_console_returns_console_when_rich_available(self):
        import beanllm.utils.cli.admin_system_commands as mod

        if not mod.RICH_AVAILABLE:
            pytest.skip("Rich not available")
        c = mod.get_console()
        assert c is not None

    def test_get_console_raises_when_no_console(self):
        import beanllm.utils.cli.admin_system_commands as mod

        orig = mod._console
        mod._console = None
        with pytest.raises(AssertionError):
            mod.get_console()
        mod._console = orig


# ===========================================================================
# print_error
# ===========================================================================


class TestPrintError:
    def test_print_error_with_rich(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        mock_console = MagicMock()
        with patch.object(mod, "_console", mock_console), patch.object(mod, "RICH_AVAILABLE", True):
            print_error("something went wrong")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "something went wrong" in call_args

    def test_print_error_without_rich(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        with patch.object(mod, "RICH_AVAILABLE", False), patch.object(mod, "_console", None):
            print_error("something went wrong")

        out = capsys.readouterr().out
        assert "something went wrong" in out

    def test_print_error_rich_but_no_console(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        with patch.object(mod, "RICH_AVAILABLE", True), patch.object(mod, "_console", None):
            print_error("error message")

        out = capsys.readouterr().out
        assert "error message" in out


# ===========================================================================
# print_success
# ===========================================================================


class TestPrintSuccess:
    def test_print_success_with_rich(self):
        import beanllm.utils.cli.admin_system_commands as mod

        mock_console = MagicMock()
        with patch.object(mod, "_console", mock_console), patch.object(mod, "RICH_AVAILABLE", True):
            print_success("all good")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "all good" in call_args

    def test_print_success_without_rich(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        with patch.object(mod, "RICH_AVAILABLE", False), patch.object(mod, "_console", None):
            print_success("success message")

        out = capsys.readouterr().out
        assert "success message" in out


# ===========================================================================
# print_info
# ===========================================================================


class TestPrintInfo:
    def test_print_info_with_rich(self):
        import beanllm.utils.cli.admin_system_commands as mod

        mock_console = MagicMock()
        with patch.object(mod, "_console", mock_console), patch.object(mod, "RICH_AVAILABLE", True):
            print_info("some info")

        mock_console.print.assert_called_once()
        call_args = str(mock_console.print.call_args)
        assert "some info" in call_args

    def test_print_info_without_rich(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        with patch.object(mod, "RICH_AVAILABLE", False), patch.object(mod, "_console", None):
            print_info("info message")

        out = capsys.readouterr().out
        assert "info message" in out


# ===========================================================================
# print_help_admin
# ===========================================================================


class TestPrintHelpAdmin:
    def test_print_help_with_rich(self):
        import beanllm.utils.cli.admin_system_commands as mod

        mock_console = MagicMock()
        panel_mock = MagicMock()

        with (
            patch.object(mod, "RICH_AVAILABLE", True),
            patch.object(mod, "_console", mock_console),
            patch(
                "beanllm.utils.cli.admin_system_commands.Panel", MagicMock(return_value=panel_mock)
            ),
        ):
            print_help_admin()

        mock_console.print.assert_called_once()

    def test_print_help_without_rich(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        with patch.object(mod, "RICH_AVAILABLE", False), patch.object(mod, "_console", None):
            print_help_admin()

        out = capsys.readouterr().out
        assert "beanllm Admin CLI" in out
        assert "analyze" in out
        assert "stats" in out

    def test_print_help_contains_commands(self, capsys):
        import beanllm.utils.cli.admin_system_commands as mod

        with patch.object(mod, "RICH_AVAILABLE", False), patch.object(mod, "_console", None):
            print_help_admin()

        out = capsys.readouterr().out
        for cmd in ["analyze", "stats", "optimize", "security", "dashboard"]:
            assert cmd in out


# ===========================================================================
# launch_dashboard
# ===========================================================================


class TestLaunchDashboard:
    def test_streamlit_not_installed(self, capsys):
        """When streamlit --version fails, should print install instruction."""
        import beanllm.utils.cli.admin_system_commands as mod

        completed_proc = MagicMock()
        completed_proc.returncode = 1

        mock_console = MagicMock()
        with (
            patch("subprocess.run", return_value=completed_proc),
            patch.object(mod, "_console", mock_console),
            patch.object(mod, "RICH_AVAILABLE", True),
        ):
            launch_dashboard()

        # print_error should have been called
        mock_console.print.assert_called()

    def test_streamlit_not_installed_no_rich(self, capsys):
        """Without rich, falls back to plain print."""
        import beanllm.utils.cli.admin_system_commands as mod

        completed_proc = MagicMock()
        completed_proc.returncode = 1

        with (
            patch("subprocess.run", return_value=completed_proc),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "_console", None),
        ):
            launch_dashboard()

        out = capsys.readouterr().out
        assert "Streamlit" in out or "streamlit" in out

    def test_dashboard_file_not_found(self, capsys):
        """When streamlit is installed but dashboard.py not found."""
        import beanllm.utils.cli.admin_system_commands as mod

        completed_proc = MagicMock()
        completed_proc.returncode = 0

        mock_console = MagicMock()

        with (
            patch("subprocess.run", return_value=completed_proc),
            patch("os.path.exists", return_value=False),
            patch.object(mod, "_console", mock_console),
            patch.object(mod, "RICH_AVAILABLE", True),
        ):
            launch_dashboard()

        mock_console.print.assert_called()

    def test_dashboard_launches_successfully(self):
        """When both streamlit and dashboard.py are available."""
        import beanllm.utils.cli.admin_system_commands as mod

        version_proc = MagicMock()
        version_proc.returncode = 0

        mock_console = MagicMock()

        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return version_proc

        with (
            patch("subprocess.run", side_effect=fake_run),
            patch("os.path.exists", return_value=True),
            patch.object(mod, "_console", mock_console),
            patch.object(mod, "RICH_AVAILABLE", True),
        ):
            launch_dashboard()

        # subprocess.run should be called at least once for version check
        assert len(run_calls) >= 1

    def test_launch_dashboard_exception_handling(self):
        """Exception in subprocess.run should be caught and printed."""
        import beanllm.utils.cli.admin_system_commands as mod

        mock_console = MagicMock()

        with (
            patch("subprocess.run", side_effect=Exception("subprocess error")),
            patch.object(mod, "_console", mock_console),
            patch.object(mod, "RICH_AVAILABLE", True),
        ):
            # Should not raise
            launch_dashboard()

        mock_console.print.assert_called()


# ===========================================================================
# Module-level alias: console == _console
# ===========================================================================


class TestModuleAlias:
    def test_console_alias(self):
        import beanllm.utils.cli.admin_system_commands as mod

        assert mod.console is mod._console
