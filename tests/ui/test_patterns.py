"""Tests for beanllm/ui/patterns.py."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

import pytest
from rich.console import Console


def _console() -> Console:
    """Return a Rich console that writes to a StringIO buffer."""
    return Console(file=StringIO(), highlight=False, markup=False)


# ---------------------------------------------------------------------------
# SuccessPattern
# ---------------------------------------------------------------------------


class TestSuccessPattern:
    def test_render_basic_message(self):
        from beanllm.ui.patterns import SuccessPattern

        console = _console()
        SuccessPattern.render("Done!", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Done!" in output

    def test_render_with_details(self):
        from beanllm.ui.patterns import SuccessPattern

        console = _console()
        SuccessPattern.render("Done!", details="Files saved.", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Files saved." in output

    def test_render_with_metadata(self):
        from beanllm.ui.patterns import SuccessPattern

        console = _console()
        SuccessPattern.render("Done!", metadata={"key": "value"}, console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "key" in output
        assert "value" in output

    def test_uses_get_console_when_none_provided(self):
        from unittest.mock import patch

        from beanllm.ui.patterns import SuccessPattern

        mock_console = _console()
        with patch("beanllm.ui.patterns.get_console", return_value=mock_console):
            SuccessPattern.render("OK")
        assert "OK" in mock_console.file.getvalue()  # type: ignore[union-attr]

    def test_render_with_multiple_metadata_entries(self):
        from beanllm.ui.patterns import SuccessPattern

        console = _console()
        SuccessPattern.render(
            "Done!",
            metadata={"duration": "1.2s", "items": "42"},
            console=console,
        )
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "duration" in output
        assert "items" in output


# ---------------------------------------------------------------------------
# ErrorPattern
# ---------------------------------------------------------------------------


class TestErrorPattern:
    def test_render_basic_message(self):
        from beanllm.ui.patterns import ErrorPattern

        console = _console()
        ErrorPattern.render("Something went wrong", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Something went wrong" in output

    def test_render_with_error_type(self):
        from beanllm.ui.patterns import ErrorPattern

        console = _console()
        ErrorPattern.render("Failed", error_type="ValueError", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "ValueError" in output

    def test_render_with_suggestion(self):
        from beanllm.ui.patterns import ErrorPattern

        console = _console()
        ErrorPattern.render("Failed", suggestion="Try again later", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Try again later" in output

    def test_uses_get_console_when_none_provided(self):
        from unittest.mock import patch

        from beanllm.ui.patterns import ErrorPattern

        mock_console = _console()
        with patch("beanllm.ui.patterns.get_console", return_value=mock_console):
            ErrorPattern.render("oops")
        assert "oops" in mock_console.file.getvalue()  # type: ignore[union-attr]

    def test_render_all_parameters(self):
        from beanllm.ui.patterns import ErrorPattern

        console = _console()
        ErrorPattern.render(
            "Crash",
            error_type="RuntimeError",
            suggestion="Restart the service",
            console=console,
        )
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Crash" in output
        assert "RuntimeError" in output
        assert "Restart" in output


# ---------------------------------------------------------------------------
# WarningPattern
# ---------------------------------------------------------------------------


class TestWarningPattern:
    def test_render_basic(self):
        from beanllm.ui.patterns import WarningPattern

        console = _console()
        WarningPattern.render("Low disk space", console=console)
        assert "Low disk space" in console.file.getvalue()  # type: ignore[union-attr]

    def test_render_with_details(self):
        from beanllm.ui.patterns import WarningPattern

        console = _console()
        WarningPattern.render("Warning!", details="Only 1 GB left", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "1 GB left" in output

    def test_uses_get_console_when_none_provided(self):
        from unittest.mock import patch

        from beanllm.ui.patterns import WarningPattern

        mock_console = _console()
        with patch("beanllm.ui.patterns.get_console", return_value=mock_console):
            WarningPattern.render("Watch out")
        assert "Watch out" in mock_console.file.getvalue()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# InfoPattern
# ---------------------------------------------------------------------------


class TestInfoPattern:
    def test_render_basic(self):
        from beanllm.ui.patterns import InfoPattern

        console = _console()
        InfoPattern.render("Processing...", console=console)
        assert "Processing" in console.file.getvalue()  # type: ignore[union-attr]

    def test_render_with_detail_list(self):
        from beanllm.ui.patterns import InfoPattern

        console = _console()
        InfoPattern.render("Info", details=["Step 1", "Step 2"], console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Step 1" in output
        assert "Step 2" in output

    def test_render_without_details(self):
        from beanllm.ui.patterns import InfoPattern

        console = _console()
        InfoPattern.render("Info only", console=console)
        assert "Info only" in console.file.getvalue()  # type: ignore[union-attr]

    def test_uses_get_console_when_none_provided(self):
        from unittest.mock import patch

        from beanllm.ui.patterns import InfoPattern

        mock_console = _console()
        with patch("beanllm.ui.patterns.get_console", return_value=mock_console):
            InfoPattern.render("Note")
        assert "Note" in mock_console.file.getvalue()  # type: ignore[union-attr]

    def test_multiple_details_all_rendered(self):
        from beanllm.ui.patterns import InfoPattern

        console = _console()
        details = ["Alpha", "Beta", "Gamma"]
        InfoPattern.render("Multi", details=details, console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        for d in details:
            assert d in output


# ---------------------------------------------------------------------------
# EmptyStatePattern
# ---------------------------------------------------------------------------


class TestEmptyStatePattern:
    def test_render_basic(self):
        from beanllm.ui.patterns import EmptyStatePattern

        console = _console()
        EmptyStatePattern.render("No documents found", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "No documents found" in output

    def test_render_with_suggestion(self):
        from beanllm.ui.patterns import EmptyStatePattern

        console = _console()
        EmptyStatePattern.render("Nothing here", suggestion="Add some files", console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Add some files" in output

    def test_render_without_suggestion(self):
        from beanllm.ui.patterns import EmptyStatePattern

        console = _console()
        EmptyStatePattern.render("Empty", console=console)
        assert "Empty" in console.file.getvalue()  # type: ignore[union-attr]

    def test_uses_get_console_when_none_provided(self):
        from unittest.mock import patch

        from beanllm.ui.patterns import EmptyStatePattern

        mock_console = _console()
        with patch("beanllm.ui.patterns.get_console", return_value=mock_console):
            EmptyStatePattern.render("Nothing")
        assert "Nothing" in mock_console.file.getvalue()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# OnboardingPattern
# ---------------------------------------------------------------------------


class TestOnboardingPattern:
    def _steps(self):
        return [
            {"title": "Install", "description": "Run pip install"},
            {"title": "Configure", "description": "Set your API key"},
        ]

    def test_render_title(self):
        from beanllm.ui.patterns import OnboardingPattern

        console = _console()
        OnboardingPattern.render("Welcome!", self._steps(), console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Welcome!" in output

    def test_render_step_titles(self):
        from beanllm.ui.patterns import OnboardingPattern

        console = _console()
        OnboardingPattern.render("Start", self._steps(), console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "Install" in output
        assert "Configure" in output

    def test_render_step_descriptions(self):
        from beanllm.ui.patterns import OnboardingPattern

        console = _console()
        OnboardingPattern.render("Start", self._steps(), console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "pip install" in output
        assert "API key" in output

    def test_render_step_without_description(self):
        from beanllm.ui.patterns import OnboardingPattern

        console = _console()
        steps = [{"title": "Step A"}]
        OnboardingPattern.render("Go", steps, console=console)
        assert "Step A" in console.file.getvalue()  # type: ignore[union-attr]

    def test_uses_get_console_when_none_provided(self):
        from unittest.mock import patch

        from beanllm.ui.patterns import OnboardingPattern

        mock_console = _console()
        with patch("beanllm.ui.patterns.get_console", return_value=mock_console):
            OnboardingPattern.render("Hi", [{"title": "x"}])
        assert "x" in mock_console.file.getvalue()  # type: ignore[union-attr]

    def test_steps_numbered(self):
        from beanllm.ui.patterns import OnboardingPattern

        console = _console()
        OnboardingPattern.render("Steps", self._steps(), console=console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "1." in output
        assert "2." in output

    def test_empty_steps_list(self):
        from beanllm.ui.patterns import OnboardingPattern

        console = _console()
        OnboardingPattern.render("No steps", [], console=console)
        assert "No steps" in console.file.getvalue()  # type: ignore[union-attr]
