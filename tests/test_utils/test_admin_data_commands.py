"""
Tests for beanllm.utils.cli.admin_data_commands
Goal: maximize line coverage (202 lines missed, 0% current)

Strategy:
  - Mock BEANLLM_AVAILABLE, RICH_AVAILABLE, env vars
  - Mock async dependencies (get_google_export_stats, get_security_events, Client)
  - Cover all four async command functions and both Rich/non-Rich branches

Note: admin_data_commands imports print_error/print_info/print_success/get_console/console
from admin_system_commands. We patch them via admin_system_commands to affect behavior.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import beanllm.utils.cli.admin_data_commands as _mod_check

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="admin_data_commands not available")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_stats(total: int = 10):
    return {
        "total_exports": total,
        "by_service": {"gmail": 5, "drive": 3, "calendar": 2},
        "top_users": [("user1@example.com", 4), ("user2@example.com", 3)],
    }


def _make_events(n: int = 3):
    return [
        {
            "timestamp": f"2024-01-0{i+1}T10:00:00",
            "user_id": f"user{i}@example.com",
            "reason": "rate_limit",
            "severity": "high",
        }
        for i in range(n)
    ]


def _make_response_mock(content: str = "Analysis complete."):
    resp = MagicMock()
    resp.content = content
    return resp


def _patch_print_fns(mock_fn=None):
    """Return a context manager that patches all print helpers in admin_data_commands."""
    if mock_fn is None:
        mock_fn = MagicMock()
    return [
        patch("beanllm.utils.cli.admin_data_commands.print_error", mock_fn),
        patch("beanllm.utils.cli.admin_data_commands.print_success", mock_fn),
        patch("beanllm.utils.cli.admin_data_commands.print_info", mock_fn),
    ]


# ===========================================================================
# analyze_with_gemini
# ===========================================================================


class TestAnalyzeWithGemini:
    @pytest.mark.asyncio
    async def test_beanllm_not_available(self, capsys):
        """When BEANLLM_AVAILABLE=False, should call print_error and return."""
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_error = MagicMock()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", False),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
        ):
            await mod.analyze_with_gemini(hours=24)

        mock_print_error.assert_called_once()
        assert (
            "beanllm" in str(mock_print_error.call_args).lower()
            or "not installed" in str(mock_print_error.call_args).lower()
        )

    @pytest.mark.asyncio
    async def test_no_gemini_api_key(self, monkeypatch):
        """When GEMINI_API_KEY not set, should print error and return."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        mock_print_error = MagicMock()
        mock_print_info = MagicMock()

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
            patch("beanllm.utils.cli.admin_data_commands.print_info", mock_print_info),
        ):
            await mod.analyze_with_gemini(hours=24)

        mock_print_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_with_rich(self, monkeypatch):
        """Full happy-path with RICH_AVAILABLE=True."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        stats = _make_stats()
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock())
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        mock_console = MagicMock()
        progress_ctx = MagicMock()
        progress_ctx.__enter__ = MagicMock(return_value=MagicMock())
        progress_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch.object(mod, "console", mock_console),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_console",
                MagicMock(return_value=mock_console),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
            patch(
                "beanllm.utils.cli.admin_data_commands.Progress",
                MagicMock(return_value=progress_ctx),
            ),
        ):
            await mod.analyze_with_gemini(hours=24)

        mock_client_cls.assert_called_once()
        mock_client_instance.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_analyze_without_rich(self, monkeypatch, capsys):
        """Full happy-path with RICH_AVAILABLE=False (plain text output)."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        stats = _make_stats()
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock())
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
        ):
            await mod.analyze_with_gemini(hours=24)

        out = capsys.readouterr().out
        assert "Gemini" in out or "statistics" in out.lower()

    @pytest.mark.asyncio
    async def test_analyze_exception_handling(self, monkeypatch):
        """Exception should be caught, print_error called."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_print_error = MagicMock()

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(side_effect=RuntimeError("DB error")),
            ),
        ):
            await mod.analyze_with_gemini(hours=24)

        mock_print_error.assert_called()


# ===========================================================================
# show_stats
# ===========================================================================


class TestShowStats:
    @pytest.mark.asyncio
    async def test_beanllm_not_available(self):
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_error = MagicMock()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", False),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
        ):
            await mod.show_stats(hours=24)

        mock_print_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_stats_with_rich(self):
        import beanllm.utils.cli.admin_data_commands as mod

        stats = _make_stats(total=10)
        mock_console = MagicMock()
        progress_ctx = MagicMock()
        progress_ctx.__enter__ = MagicMock(return_value=MagicMock())
        progress_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch.object(mod, "console", mock_console),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_console",
                MagicMock(return_value=mock_console),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.Progress",
                MagicMock(return_value=progress_ctx),
            ),
        ):
            await mod.show_stats(hours=24)

        # Console should have been used to print tables
        assert mock_console.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_show_stats_with_rich_empty_top_users(self):
        """top_users == [] should skip user table."""
        import beanllm.utils.cli.admin_data_commands as mod

        stats = {"total_exports": 5, "by_service": {"gmail": 5}, "top_users": []}
        mock_console = MagicMock()
        progress_ctx = MagicMock()
        progress_ctx.__enter__ = MagicMock(return_value=MagicMock())
        progress_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch.object(mod, "console", mock_console),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_console",
                MagicMock(return_value=mock_console),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.Progress",
                MagicMock(return_value=progress_ctx),
            ),
        ):
            await mod.show_stats(hours=24)

        assert mock_console.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_show_stats_without_rich(self, capsys):
        import beanllm.utils.cli.admin_data_commands as mod

        stats = _make_stats()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
        ):
            await mod.show_stats(hours=24)

        out = capsys.readouterr().out
        assert "Google" in out or "gmail" in out.lower()

    @pytest.mark.asyncio
    async def test_show_stats_without_rich_zero_total(self, capsys):
        """total_exports == 0 should show 0% percentages."""
        import beanllm.utils.cli.admin_data_commands as mod

        stats = {"total_exports": 0, "by_service": {"gmail": 0}, "top_users": []}
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
        ):
            await mod.show_stats(hours=24)

        out = capsys.readouterr().out
        assert "0%" in out

    @pytest.mark.asyncio
    async def test_show_stats_exception_handling(self):
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_error = MagicMock()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(side_effect=RuntimeError("Connection error")),
            ),
        ):
            await mod.show_stats(hours=24)

        mock_print_error.assert_called()


# ===========================================================================
# optimize_with_gemini
# ===========================================================================


class TestOptimizeWithGemini:
    @pytest.mark.asyncio
    async def test_beanllm_not_available(self):
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_error = MagicMock()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", False),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
        ):
            await mod.optimize_with_gemini()

        mock_print_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_gemini_api_key(self, monkeypatch):
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        mock_print_error = MagicMock()

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
        ):
            await mod.optimize_with_gemini()

        mock_print_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_with_rich(self, monkeypatch):
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        stats = _make_stats()
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock())
        mock_client_cls = MagicMock(return_value=mock_client_instance)
        mock_console = MagicMock()

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_console",
                MagicMock(return_value=mock_console),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
        ):
            await mod.optimize_with_gemini()

        mock_client_instance.chat.assert_awaited_once()
        mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_optimize_without_rich(self, monkeypatch, capsys):
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        stats = _make_stats()
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock("Cost advice."))
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(return_value=stats),
            ),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
        ):
            await mod.optimize_with_gemini()

        out = capsys.readouterr().out
        assert "Cost" in out or "advice" in out.lower() or "Gemini" in out

    @pytest.mark.asyncio
    async def test_optimize_exception_handling(self, monkeypatch):
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        mock_print_error = MagicMock()

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_google_export_stats",
                AsyncMock(side_effect=RuntimeError("DB error")),
            ),
        ):
            await mod.optimize_with_gemini()

        mock_print_error.assert_called()


# ===========================================================================
# check_security
# ===========================================================================


class TestCheckSecurity:
    @pytest.mark.asyncio
    async def test_beanllm_not_available(self):
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_error = MagicMock()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", False),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
        ):
            await mod.check_security(hours=24)

        mock_print_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_security_events(self):
        """Empty events list → print_success called."""
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_success = MagicMock()
        progress_ctx = MagicMock()
        progress_ctx.__enter__ = MagicMock(return_value=MagicMock())
        progress_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_success", mock_print_success),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(return_value=[]),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.Progress",
                MagicMock(return_value=progress_ctx),
            ),
        ):
            await mod.check_security(hours=24)

        mock_print_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_events_with_rich_no_gemini(self, monkeypatch):
        """Events present, no GEMINI_API_KEY → table printed, no Gemini analysis."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        events = _make_events(3)
        mock_console = MagicMock()
        progress_ctx = MagicMock()
        progress_ctx.__enter__ = MagicMock(return_value=MagicMock())
        progress_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch.object(mod, "console", mock_console),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_console",
                MagicMock(return_value=mock_console),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(return_value=events),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.Progress",
                MagicMock(return_value=progress_ctx),
            ),
        ):
            await mod.check_security(hours=24)

        assert mock_console.print.call_count >= 1

    @pytest.mark.asyncio
    async def test_security_events_with_gemini_analysis(self, monkeypatch):
        """Events present + GEMINI_API_KEY → Gemini analysis runs."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        events = _make_events(3)
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock("Security OK."))
        mock_client_cls = MagicMock(return_value=mock_client_instance)
        mock_console = MagicMock()
        progress_ctx = MagicMock()
        progress_ctx.__enter__ = MagicMock(return_value=MagicMock())
        progress_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", True),
            patch.object(mod, "console", mock_console),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_console",
                MagicMock(return_value=mock_console),
            ),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(return_value=events),
            ),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
            patch(
                "beanllm.utils.cli.admin_data_commands.Progress",
                MagicMock(return_value=progress_ctx),
            ),
        ):
            await mod.check_security(hours=24)

        mock_client_instance.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_security_events_without_rich(self, capsys, monkeypatch):
        """Without rich, falls back to plain print for events."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        events = _make_events(3)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(return_value=events),
            ),
        ):
            await mod.check_security(hours=24)

        out = capsys.readouterr().out
        assert "Security" in out or "user" in out.lower()

    @pytest.mark.asyncio
    async def test_security_events_without_rich_with_gemini(self, capsys, monkeypatch):
        """Without rich + GEMINI_API_KEY → plain text Gemini output."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        events = _make_events(2)
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock("Threat analysis."))
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(return_value=events),
            ),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
        ):
            await mod.check_security(hours=24)

        out = capsys.readouterr().out
        assert "Threat" in out or "analysis" in out.lower() or "Security" in out

    @pytest.mark.asyncio
    async def test_check_security_exception_handling(self):
        import beanllm.utils.cli.admin_data_commands as mod

        mock_print_error = MagicMock()
        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch("beanllm.utils.cli.admin_data_commands.print_error", mock_print_error),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(side_effect=RuntimeError("Connection refused")),
            ),
        ):
            await mod.check_security(hours=24)

        mock_print_error.assert_called()

    @pytest.mark.asyncio
    async def test_security_no_rich_no_events(self, capsys, monkeypatch):
        """Without rich and empty events, print_success message."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch(
                "beanllm.utils.cli.admin_data_commands.get_security_events",
                AsyncMock(return_value=[]),
            ),
        ):
            await mod.check_security(hours=24)

        out = capsys.readouterr().out
        # print_success falls through to print() when RICH_AVAILABLE is False
        assert "No" in out or "Success" in out or out == ""

    @pytest.mark.asyncio
    async def test_analyze_with_gemini_non_zero_hours(self, monkeypatch):
        """Verify hours parameter is forwarded correctly."""
        import beanllm.utils.cli.admin_data_commands as mod

        monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
        stats = _make_stats()
        mock_get_stats = AsyncMock(return_value=stats)
        mock_client_instance = AsyncMock()
        mock_client_instance.chat = AsyncMock(return_value=_make_response_mock())
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with (
            patch.object(mod, "BEANLLM_AVAILABLE", True),
            patch.object(mod, "RICH_AVAILABLE", False),
            patch.object(mod, "console", None),
            patch("beanllm.utils.cli.admin_data_commands.get_google_export_stats", mock_get_stats),
            patch("beanllm.utils.cli.admin_data_commands.Client", mock_client_cls),
        ):
            await mod.analyze_with_gemini(hours=168)

        mock_get_stats.assert_awaited_once_with(hours=168)
