"""
CLI/TUI Components
터미널 UI 컴포넌트
"""

from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .design_tokens import tokens


class Badge:
    """배지 컴포넌트 [OK], [ERROR] 등"""

    @staticmethod
    def success(text: str = "OK") -> str:
        """성공 배지"""
        return f"[{tokens.color.bg_success} {tokens.color.success} bold][{text}][/{tokens.color.bg_success} {tokens.color.success} bold]"

    @staticmethod
    def error(text: str = "ERROR") -> str:
        """에러 배지"""
        return f"[{tokens.color.bg_error} {tokens.color.error} bold][{text}][/{tokens.color.bg_error} {tokens.color.error} bold]"

    @staticmethod
    def warning(text: str = "WARN") -> str:
        """경고 배지"""
        return f"[{tokens.color.bg_warning} {tokens.color.warning} bold][{text}][/{tokens.color.bg_warning} {tokens.color.warning} bold]"

    @staticmethod
    def info(text: str = "INFO") -> str:
        """정보 배지"""
        return f"[{tokens.color.bg_info} {tokens.color.info} bold][{text}][/{tokens.color.bg_info} {tokens.color.info} bold]"

    @staticmethod
    def custom(text: str, color: str) -> str:
        """커스텀 배지"""
        return f"[{color} bold][{text}][/{color} bold]"


class StatusIcon:
    """상태 아이콘"""

    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    LOADING = "⟳"

    @staticmethod
    def success() -> str:
        return f"[{tokens.color.success}]{StatusIcon.SUCCESS}[/{tokens.color.success}]"

    @staticmethod
    def error() -> str:
        return f"[{tokens.color.error}]{StatusIcon.ERROR}[/{tokens.color.error}]"

    @staticmethod
    def warning() -> str:
        return f"[{tokens.color.warning}]{StatusIcon.WARNING}[/{tokens.color.warning}]"

    @staticmethod
    def info() -> str:
        return f"[{tokens.color.info}]{StatusIcon.INFO}[/{tokens.color.info}]"


class Divider:
    """구분선 컴포넌트"""

    @staticmethod
    def thin(length: int = 50) -> str:
        """얇은 구분선"""
        return "─" * length

    @staticmethod
    def thick(length: int = 50) -> str:
        """두꺼운 구분선"""
        return "═" * length

    @staticmethod
    def double(length: int = 50) -> str:
        """이중 구분선"""
        return "═" * length

    @staticmethod
    def styled(text: str = "", style: str = "dim") -> str:
        """스타일 적용 구분선"""
        return f"[{style}]{Divider.thin()}[/{style}]"


class CommandBlock:
    """명령어 블록 (실행 명령 강조)"""

    @staticmethod
    def render(command: str, console: Optional[Console] = None) -> None:
        """명령어 블록 렌더링"""
        if console is None:
            from .console import get_console

            console = get_console()

        text = Text()
        text.append("$ ", style=f"{tokens.color.accent} bold")
        text.append(command, style=f"{tokens.color.primary} bold")

        panel = Panel(text, border_style=tokens.color.accent, box=box.ROUNDED, padding=(0, 1))
        console.print(panel)


class OutputBlock:
    """출력 블록 (결과 출력 영역)"""

    @staticmethod
    def render(
        content: str, title: Optional[str] = None, console: Optional[Console] = None
    ) -> None:
        """출력 블록 렌더링"""
        if console is None:
            from .console import get_console

            console = get_console()

        panel = Panel(
            content, title=title, border_style=tokens.color.muted, box=box.ROUNDED, padding=(0, 1)
        )
        console.print(panel)


class Prompt:
    """프롬프트 컴포넌트 ($, > 같은 입력 시작점)"""

    @staticmethod
    def render(prompt_char: str = "$", style: str = None) -> str:
        """프롬프트 렌더링"""
        if style is None:
            style = f"{tokens.color.accent} bold"
        return f"[{style}]{prompt_char}[/{style}] "


class Spinner:
    """로딩 스피너"""

    @staticmethod
    def create(description: str = "Loading...") -> Progress:
        """스피너 생성"""
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[{tokens.color.info}]{description}[/{tokens.color.info}]"),
            console=get_console(),
            transient=True,
        )


class ProgressBar:
    """진행률 표시"""

    @staticmethod
    def create(total: int, description: str = "Progress") -> Progress:
        """진행률 바 생성"""
        return Progress(
            TextColumn(f"[{tokens.color.info}]{description}[/{tokens.color.info}]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=get_console(),
            transient=False,
        )


def get_console() -> Console:
    """Console 인스턴스 반환"""
    from .console import get_console as _get_console

    return _get_console()
