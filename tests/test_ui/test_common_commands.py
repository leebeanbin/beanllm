"""
Tests for beanllm.ui.repl.common_commands
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from beanllm.ui.repl.common_commands import CommonCommands, create_common_commands

    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCommonCommandsRegistration:
    """Tests for register_command and the command registry."""

    def setup_method(self):
        self.cmd = CommonCommands()

    def test_register_command_basic(self):
        handler = MagicMock()
        self.cmd.register_command("test_cmd", handler, "A test command")
        assert "test_cmd" in self.cmd.command_registry

    def test_register_command_stores_all_fields(self):
        handler = MagicMock()
        self.cmd.register_command(
            name="mycmd",
            handler=handler,
            description="My description",
            category="MyCategory",
            usage="mycmd [arg]",
        )
        info = self.cmd.command_registry["mycmd"]
        assert info["handler"] is handler
        assert info["description"] == "My description"
        assert info["category"] == "MyCategory"
        assert info["usage"] == "mycmd [arg]"

    def test_register_command_default_category(self):
        handler = MagicMock()
        self.cmd.register_command("cmd2", handler, "desc")
        assert self.cmd.command_registry["cmd2"]["category"] == "General"

    def test_register_multiple_commands(self):
        for i in range(5):
            self.cmd.register_command(f"cmd_{i}", MagicMock(), f"Desc {i}")
        assert len(self.cmd.command_registry) == 5

    def test_overwrite_existing_command(self):
        h1 = MagicMock()
        h2 = MagicMock()
        self.cmd.register_command("dup", h1, "first")
        self.cmd.register_command("dup", h2, "second")
        assert self.cmd.command_registry["dup"]["handler"] is h2


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCmdHelp:
    """Tests for cmd_help."""

    def setup_method(self):
        self.cmd = CommonCommands()
        # Register a few commands
        self.cmd.register_command("go", MagicMock(), "Go forward", category="Nav", usage="go <dir>")
        self.cmd.register_command("stop", MagicMock(), "Stop", category="Nav")
        self.cmd.register_command("info", MagicMock(), "Show info", category="Info")

    def test_help_no_args_prints_all(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help()
            assert mock_console.print.call_count >= 1

    def test_help_no_args_none(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help(None)
            assert mock_console.print.call_count >= 1

    def test_help_specific_command_found(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help(["go"])
            assert mock_console.print.call_count >= 1

    def test_help_specific_command_with_usage(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help(["go"])
            # usage line should be printed
            calls = [str(c) for c in mock_console.print.call_args_list]
            assert any("go" in c for c in calls)

    def test_help_specific_command_no_usage(self):
        """Command with empty usage string → usage block not printed."""
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help(["stop"])
            assert mock_console.print.call_count >= 1

    def test_help_unknown_command(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help(["nonexistent"])
            # Should print "Unknown command: ..." message
            assert mock_console.print.call_count >= 1

    def test_help_empty_args_list(self):
        """Empty list triggers full help output."""
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_help([])
            assert mock_console.print.call_count >= 1

    def test_help_empty_registry(self):
        """No registered commands → help prints only header."""
        cmd = CommonCommands()
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            cmd.cmd_help()
            assert mock_console.print.call_count >= 1


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCmdExit:
    """Tests for cmd_exit and cmd_quit."""

    def setup_method(self):
        self.cmd = CommonCommands()

    def test_cmd_exit_returns_true(self):
        with patch("beanllm.ui.repl.common_commands.console"):
            result = self.cmd.cmd_exit()
        assert result is True

    def test_cmd_exit_with_args(self):
        with patch("beanllm.ui.repl.common_commands.console"):
            result = self.cmd.cmd_exit(["--force"])
        assert result is True

    def test_cmd_exit_prints_goodbye(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_exit()
            mock_console.print.assert_called_once()

    def test_cmd_quit_delegates_to_exit(self):
        with patch("beanllm.ui.repl.common_commands.console"):
            result = self.cmd.cmd_quit()
        assert result is True

    def test_cmd_quit_with_args(self):
        with patch("beanllm.ui.repl.common_commands.console"):
            result = self.cmd.cmd_quit(["arg"])
        assert result is True


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCmdClear:
    """Tests for cmd_clear."""

    def setup_method(self):
        self.cmd = CommonCommands()

    def test_cmd_clear_runs_without_error(self):
        with patch("beanllm.ui.repl.common_commands.console"), patch("os.system"):
            self.cmd.cmd_clear()

    def test_cmd_clear_calls_os_system(self):
        with patch("beanllm.ui.repl.common_commands.console"), patch("os.system") as mock_os:
            self.cmd.cmd_clear()
            mock_os.assert_called_once()

    def test_cmd_clear_prints_prompt(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console, patch("os.system"):
            self.cmd.cmd_clear()
            mock_console.print.assert_called_once()


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCmdVersion:
    """Tests for cmd_version."""

    def setup_method(self):
        self.cmd = CommonCommands()

    def test_cmd_version_with_beanllm(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_version()
            mock_console.print.assert_called_once()

    def test_cmd_version_beanllm_no_version_attr(self):
        """beanllm imported but no __version__ → 'unknown'."""
        import beanllm as _beanllm

        original = getattr(_beanllm, "__version__", None)
        try:
            if hasattr(_beanllm, "__version__"):
                del _beanllm.__version__
            with patch("beanllm.ui.repl.common_commands.console") as mock_console:
                self.cmd.cmd_version()
                mock_console.print.assert_called_once()
        finally:
            if original is not None:
                _beanllm.__version__ = original

    def test_cmd_version_import_error(self):
        """cmd_version falls back to 'unknown' when beanllm import fails."""
        import sys

        # Patch the import inside cmd_version via builtins.__import__
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None  # type: ignore[union-attr]

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "beanllm":
                raise ImportError("mocked ImportError")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch("beanllm.ui.repl.common_commands.console") as mock_console:
                self.cmd.cmd_version()
                mock_console.print.assert_called_once()


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCmdStatus:
    """Tests for cmd_status."""

    def setup_method(self):
        self.cmd = CommonCommands()

    def test_cmd_status_distributed_unavailable(self):
        """Simulate ImportError for distributed features."""
        with (
            patch("beanllm.ui.repl.common_commands.console") as mock_console,
            patch("beanllm.ui.repl.common_commands.logger"),
        ):
            self.cmd.cmd_status()
            mock_console.print.assert_called_once()

    def test_cmd_status_redis_connected(self):
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            with (
                patch(
                    "beanllm.infrastructure.distributed.check_redis_health",
                    return_value=True,
                    create=True,
                ),
                patch(
                    "beanllm.infrastructure.distributed.check_kafka_health",
                    return_value=True,
                    create=True,
                ),
            ):
                self.cmd.cmd_status()
            mock_console.print.assert_called_once()

    def test_cmd_status_with_commands_registered(self):
        for i in range(3):
            self.cmd.register_command(f"c{i}", MagicMock(), "desc")
        with patch("beanllm.ui.repl.common_commands.console") as mock_console:
            self.cmd.cmd_status()
            mock_console.print.assert_called_once()


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCmdConfig:
    """Tests for cmd_config."""

    def setup_method(self):
        self.cmd = CommonCommands()

    def test_cmd_config_no_env_vars(self):
        with (
            patch("beanllm.ui.repl.common_commands.console") as mock_console,
            patch.dict("os.environ", {}, clear=True),
        ):
            self.cmd.cmd_config()
            mock_console.print.assert_called_once()

    def test_cmd_config_with_env_vars_set(self):
        env = {
            "USE_DISTRIBUTED": "true",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        }
        with (
            patch("beanllm.ui.repl.common_commands.console") as mock_console,
            patch.dict("os.environ", env),
        ):
            self.cmd.cmd_config()
            mock_console.print.assert_called_once()


@pytest.mark.skipif(not AVAILABLE, reason="beanllm.ui.repl.common_commands not available")
class TestCreateCommonCommands:
    """Tests for the create_common_commands factory function."""

    def test_returns_common_commands_instance(self):
        result = create_common_commands()
        assert isinstance(result, CommonCommands)

    def test_registers_standard_commands(self):
        result = create_common_commands()
        for cmd in ("help", "exit", "quit", "clear", "version", "status", "config"):
            assert cmd in result.command_registry

    def test_handlers_are_callable(self):
        result = create_common_commands()
        for name, info in result.command_registry.items():
            assert callable(info["handler"]), f"Handler for '{name}' is not callable"
