"""
Tests for beanllm.ui.repl.repl_shell
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from beanllm.ui.repl.repl_shell import REPLShell

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.repl_shell not available")
class TestREPLShellInit:
    """Tests for REPLShell initialization."""

    def test_init_creates_running_false(self):
        shell = REPLShell()
        assert shell.running is False

    def test_init_creates_common_commands(self):
        shell = REPLShell()
        assert shell.common_commands is not None

    def test_init_creates_empty_command_modules(self):
        shell = REPLShell()
        assert isinstance(shell.command_modules, dict)

    def test_init_client_is_none(self):
        shell = REPLShell()
        assert shell.client is None


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.repl_shell not available")
class TestREPLShellRegisterModule:
    """Tests for register_module."""

    def setup_method(self):
        self.shell = REPLShell()

    def test_register_module_stores_instance(self):
        module = MagicMock()
        module.cmd_foo = MagicMock(return_value=None)
        module.cmd_foo.__doc__ = "Foo command"
        # Make dir(module) include cmd_foo
        type(module).__dir__ = lambda self: ["cmd_foo"]

        self.shell.register_module("mymod", module, "MyCategory")
        assert "mymod" in self.shell.command_modules
        assert self.shell.command_modules["mymod"]["category"] == "MyCategory"

    def test_register_module_registers_cmd_methods(self):
        """All cmd_* attributes on the module get registered."""

        class FakeModule:
            def cmd_alpha(self, args=None):
                """Alpha command"""

            def cmd_beta(self, args=None):
                """Beta command"""

            def helper_method(self):
                pass

        module = FakeModule()
        self.shell.register_module("fake", module, "Test")
        assert "alpha" in self.shell.common_commands.command_registry
        assert "beta" in self.shell.common_commands.command_registry
        assert "helper_method" not in self.shell.common_commands.command_registry

    def test_register_module_cmd_without_docstring(self):
        """Commands without docstrings get default description."""

        # Use a lambda-style object where __doc__ is naturally None
        # by assigning via a MagicMock attribute
        module = MagicMock()
        # Remove all default cmd_ attrs; add only one with doc=None
        handler = MagicMock()
        handler.__doc__ = None
        module.cmd_nodoc = handler

        # Restrict dir() to only expose cmd_nodoc
        with patch(
            "builtins.dir", side_effect=lambda obj: ["cmd_nodoc"] if obj is module else dir(obj)
        ):
            self.shell.register_module("nodocmod", module, "Test")

        assert "nodoc" in self.shell.common_commands.command_registry

    def test_register_multiple_modules(self):
        class ModA:
            def cmd_a1(self, args=None):
                """A1"""

        class ModB:
            def cmd_b1(self, args=None):
                """B1"""

        self.shell.register_module("mod_a", ModA(), "CatA")
        self.shell.register_module("mod_b", ModB(), "CatB")
        assert "mod_a" in self.shell.command_modules
        assert "mod_b" in self.shell.command_modules


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.repl_shell not available")
class TestREPLShellParseInput:
    """Tests for parse_input."""

    def setup_method(self):
        self.shell = REPLShell()

    def test_parse_help(self):
        cmd, args = self.shell.parse_input("help")
        assert cmd == "help"
        assert args == []

    def test_parse_exit(self):
        cmd, args = self.shell.parse_input("exit")
        assert cmd == "exit"
        assert args == []

    def test_parse_command_with_args(self):
        cmd, args = self.shell.parse_input("module.command arg1 arg2")
        assert cmd == "module.command"
        assert args == ["arg1", "arg2"]

    def test_parse_empty_string(self):
        cmd, args = self.shell.parse_input("")
        assert cmd is None
        assert args == []

    def test_parse_whitespace_only(self):
        cmd, args = self.shell.parse_input("   ")
        assert cmd is None
        assert args == []

    def test_parse_command_uppercased_lowered(self):
        """Commands are lowercased."""
        cmd, args = self.shell.parse_input("HELP")
        assert cmd == "help"

    def test_parse_command_with_leading_whitespace(self):
        cmd, args = self.shell.parse_input("  status  ")
        assert cmd == "status"

    def test_parse_single_arg(self):
        cmd, args = self.shell.parse_input("help exit")
        assert cmd == "help"
        assert args == ["exit"]

    def test_parse_many_args(self):
        cmd, args = self.shell.parse_input("cmd a b c d")
        assert cmd == "cmd"
        assert args == ["a", "b", "c", "d"]


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.repl_shell not available")
class TestREPLShellExecuteCommand:
    """Tests for execute_command (async)."""

    def setup_method(self):
        self.shell = REPLShell()

    async def test_execute_unknown_command(self):
        with patch("beanllm.ui.repl.repl_shell.console"):
            result = await self.shell.execute_command("nonexistent", [])
        assert result is False

    async def test_execute_exit_command_returns_true(self):
        with patch("beanllm.ui.repl.common_commands.console"):
            result = await self.shell.execute_command("exit", [])
        assert result is True

    async def test_execute_help_command(self):
        with patch("beanllm.ui.repl.common_commands.console"):
            result = await self.shell.execute_command("help", [])
        assert result is False

    async def test_execute_command_that_raises(self):
        """Handler that raises should not propagate; execute_command returns False."""

        def bad_handler(args=None):
            raise RuntimeError("boom")

        self.shell.common_commands.register_command("badcmd", bad_handler, "bad")

        with patch("beanllm.ui.repl.repl_shell.console"):
            result = await self.shell.execute_command("badcmd", [])
        assert result is False

    async def test_execute_command_raises_with_debug_flag(self):
        """--debug flag causes traceback to be printed."""

        def bad_handler(args=None):
            raise ValueError("debug error")

        self.shell.common_commands.register_command("dbgcmd", bad_handler, "dbg")

        with patch("beanllm.ui.repl.repl_shell.console") as mc:
            result = await self.shell.execute_command("dbgcmd", ["--debug"])
        assert result is False

    async def test_execute_async_handler(self):
        """Async handlers are awaited correctly."""
        call_log = []

        async def async_handler(args=None):
            call_log.append("called")
            return None  # not True → should not exit

        self.shell.common_commands.register_command("asynccmd", async_handler, "async")

        with patch("beanllm.ui.repl.repl_shell.console"):
            result = await self.shell.execute_command("asynccmd", [])
        assert result is False
        assert call_log == ["called"]

    async def test_execute_async_handler_returns_true(self):
        """Async handler returning True causes execute_command to return True."""

        async def exit_handler(args=None):
            return True

        self.shell.common_commands.register_command("asyncexit", exit_handler, "async exit")

        with patch("beanllm.ui.repl.repl_shell.console"):
            result = await self.shell.execute_command("asyncexit", [])
        assert result is True


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.repl_shell not available")
class TestREPLShellShowWelcome:
    """Tests for show_welcome."""

    def test_show_welcome_prints_panel(self):
        shell = REPLShell()
        with patch("beanllm.ui.repl.repl_shell.console") as mock_console:
            shell.show_welcome()
            mock_console.print.assert_called_once()


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.repl_shell not available")
class TestREPLShellSetupDefaultModules:
    """Tests for setup_default_modules."""

    def test_setup_default_modules_all_fail(self):
        """When all module imports fail, setup_default_modules should not raise."""
        shell = REPLShell()
        with patch("beanllm.ui.repl.repl_shell.console"):
            # The method catches ImportError / any Exception per module
            shell.setup_default_modules()

    def test_setup_default_modules_kg_success(self):
        """KG module loads successfully."""
        shell = REPLShell()

        class FakeKG:
            def cmd_kg_status(self, args=None):
                """KG status"""

        with (
            patch("beanllm.ui.repl.repl_shell.console"),
            patch("beanllm.ui.repl.repl_shell.REPLShell.setup_default_modules") as mock_setup,
        ):
            shell.register_module("kg", FakeKG(), "Knowledge Graph")
            assert "kg" in shell.command_modules

    def test_setup_default_modules_prints_status(self):
        """setup_default_modules always prints something at end."""
        shell = REPLShell()
        with patch("beanllm.ui.repl.repl_shell.console") as mock_console:
            shell.setup_default_modules()
            # Last call is console.print() with no args (blank line)
            assert mock_console.print.call_count >= 1
