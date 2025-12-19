"""
Interaction Patterns
터미널 UI 패턴
"""

from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .components import StatusIcon
from .console import get_console
from .design_tokens import tokens


class SuccessPattern:
    """성공 피드백 패턴"""

    @staticmethod
    def render(
        message: str,
        details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        console: Optional[Console] = None,
    ) -> None:
        """성공 메시지 렌더링"""
        if console is None:
            console = get_console()

        text = Text()
        text.append(StatusIcon.success(), style=tokens.color.success)
        text.append(" ", style="")
        text.append(message, style=f"{tokens.color.success} bold")

        if details:
            text.append(f"\n{tokens.spacing.get_indent('sm')}└── ", style="dim")
            text.append(details, style="dim")

        if metadata:
            for key, value in metadata.items():
                text.append(f"\n{tokens.spacing.get_indent('sm')}└── ", style="dim")
                text.append(f"{key}: ", style="dim")
                text.append(str(value), style="dim")

        console.print(text)
        console.print()


class ErrorPattern:
    """에러 피드백 패턴"""

    @staticmethod
    def render(
        message: str,
        error_type: Optional[str] = None,
        suggestion: Optional[str] = None,
        console: Optional[Console] = None,
    ) -> None:
        """에러 메시지 렌더링"""
        if console is None:
            console = get_console()

        text = Text()
        text.append(StatusIcon.error(), style=tokens.color.error)
        text.append(" ", style="")
        text.append(message, style=f"{tokens.color.error} bold")

        if error_type:
            text.append(f"\n{tokens.spacing.get_indent('sm')}└── ", style="dim")
            text.append(f"Type: {error_type}", style="dim")

        if suggestion:
            text.append(f"\n{tokens.spacing.get_indent('sm')}└── ", style="dim")
            text.append("Suggestion: ", style="dim")
            text.append(suggestion, style=f"{tokens.color.info}")

        console.print(text)
        console.print()


class WarningPattern:
    """경고 피드백 패턴"""

    @staticmethod
    def render(
        message: str, details: Optional[str] = None, console: Optional[Console] = None
    ) -> None:
        """경고 메시지 렌더링"""
        if console is None:
            console = get_console()

        text = Text()
        text.append(StatusIcon.warning(), style=tokens.color.warning)
        text.append(" ", style="")
        text.append(message, style=f"{tokens.color.warning} bold")

        if details:
            text.append(f"\n{tokens.spacing.get_indent('sm')}└── ", style="dim")
            text.append(details, style="dim")

        console.print(text)
        console.print()


class InfoPattern:
    """정보 피드백 패턴"""

    @staticmethod
    def render(
        message: str, details: Optional[List[str]] = None, console: Optional[Console] = None
    ) -> None:
        """정보 메시지 렌더링"""
        if console is None:
            console = get_console()

        text = Text()
        text.append(StatusIcon.info(), style=tokens.color.info)
        text.append(" ", style="")
        text.append(message, style=f"{tokens.color.info} bold")

        if details:
            for detail in details:
                text.append(f"\n{tokens.spacing.get_indent('sm')}└── ", style="dim")
                text.append(detail, style="dim")

        console.print(text)
        console.print()


class EmptyStatePattern:
    """빈 상태 패턴 (아무 결과 없을 때)"""

    @staticmethod
    def render(
        message: str, suggestion: Optional[str] = None, console: Optional[Console] = None
    ) -> None:
        """빈 상태 메시지 렌더링"""
        if console is None:
            console = get_console()

        content = Text()
        content.append("(empty)", style="dim italic")
        content.append(f"\n\n{message}", style="dim")

        if suggestion:
            content.append("\n\n💡 ", style="yellow")
            content.append(suggestion, style="yellow")

        panel = Panel(content, border_style=tokens.color.muted, box=box.ROUNDED, padding=(1, 2))
        console.print(panel)
        console.print()


class OnboardingPattern:
    """온보딩 패턴 (첫 실행 시)"""

    @staticmethod
    def render(title: str, steps: List[Dict[str, str]], console: Optional[Console] = None) -> None:
        """온보딩 메시지 렌더링"""
        if console is None:
            console = get_console()

        content = Text()
        content.append(title, style=f"{tokens.color.accent} bold")
        content.append("\n\n", style="")

        for i, step in enumerate(steps, 1):
            content.append(f"{i}. ", style=f"{tokens.color.primary} bold")
            content.append(step.get("title", ""), style="bold")
            if "description" in step:
                content.append(f"\n{tokens.spacing.get_indent('md')}", style="")
                content.append(step["description"], style="dim")
            content.append("\n", style="")

        panel = Panel(
            content,
            title=f"[bold {tokens.color.accent}]🚀 Getting Started[/bold {tokens.color.accent}]",
            border_style=tokens.color.accent,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(panel)
        console.print()
