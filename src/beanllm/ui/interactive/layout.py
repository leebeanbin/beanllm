"""
Layout - Gemini CLI / aichat / OpenCode 스타일 프리미엄 TUI

핵심:
  - 그라디언트 로고 + 환경 정보 웰컴
  - 테마 기반 컬러 시스템
  - 메시지 구분선 (타임스탬프, 아이콘)
  - 단계별 Thinking 표시
  - 스타일된 에러/경고 패널
  - 컨텍스트 인디케이터
  - 실시간 마크다운 스트리밍
"""

from __future__ import annotations

import itertools
import platform
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rich.console import Console

    from beanllm.ui.interactive.session import ChatSession


# ---------------------------------------------------------------------------
# 그라디언트 로고
# ---------------------------------------------------------------------------

_LOGO_LINES = [
    "██████╗ ███████╗ █████╗ ███╗   ██╗██╗     ██╗     ███╗   ███╗",
    "██╔══██╗██╔════╝██╔══██╗████╗  ██║██║     ██║     ████╗ ████║",
    "██████╔╝█████╗  ███████║██╔██╗ ██║██║     ██║     ██╔████╔██║",
    "██╔══██╗██╔══╝  ██╔══██║██║╚██╗██║██║     ██║     ██║╚██╔╝██║",
    "██████╔╝███████╗██║  ██║██║ ╚████║███████╗███████╗██║ ╚═╝ ██║",
    "╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝",
]


def render_logo(console: "Console", animate: bool = True) -> None:
    """그라디언트 로고 렌더링"""
    from rich.align import Align
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    gradient = theme.palette.gradient

    for i, line in enumerate(_LOGO_LINES):
        color = gradient[i % len(gradient)]
        styled = Text(line, style=f"bold {color}")
        console.print(Align.center(styled))
        if animate:
            time.sleep(0.025)

    # 서브타이틀
    subtitle = Text()
    subtitle.append("  ✦ ", style=f"bold {theme.palette.brand_accent}")
    subtitle.append("Your AI Toolkit", style=f"italic {theme.palette.dim}")
    subtitle.append("  ·  ", style=f"{theme.palette.muted}")
    subtitle.append("v0.3.0", style=f"{theme.palette.muted}")
    console.print(Align.center(subtitle))
    console.print()


# ---------------------------------------------------------------------------
# 환영 화면
# ---------------------------------------------------------------------------


def render_welcome(console: "Console", session: "ChatSession") -> None:
    """환경 정보 포함 웰컴 박스 - Gemini CLI 스타일"""
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette

    # 시스템 정보
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_name = platform.system()
    cwd = str(session.working_dir)
    if len(cwd) > 50:
        cwd = "..." + cwd[-47:]

    role_name = getattr(session, "role_name", "default")

    # 좌측: 모델 + 역할 정보
    left = Text()
    left.append(f" {theme.icons['model']} ", style=f"bold {p.brand_primary}")
    left.append(f"{session.model}", style=f"bold {p.text}")
    left.append("  ·  ", style=p.muted)
    left.append(f"{role_name}", style=f"{p.brand_accent}")
    left.append("\n", style="")
    left.append(f" {theme.icons['chat']} ", style=f"{p.info_color}")
    left.append(f"{session.mode} mode", style=f"{p.dim}")
    left.append("  ·  ", style=p.muted)
    left.append(f"Python {py_ver}", style=f"{p.dim}")
    left.append("  ·  ", style=p.muted)
    left.append(f"{os_name}", style=f"{p.dim}")

    # 우측: 단축키
    right = Text()
    right.append("  /help", style=f"bold {p.brand_primary}")
    right.append(" commands  ", style=f"{p.dim}")
    right.append("@file", style=f"bold {p.brand_primary}")
    right.append(" attach\n", style=f"{p.dim}")
    right.append("  Tab", style=f"bold {p.brand_primary}")
    right.append(" complete  ", style=f"{p.dim}")
    right.append("Ctrl+R", style=f"bold {p.brand_primary}")
    right.append(" history", style=f"{p.dim}")

    content = Columns([left, right], equal=True, expand=True)

    console.print(
        Panel(
            content,
            border_style=p.border,
            padding=(0, 1),
            subtitle=f"[{p.muted}]{cwd}[/{p.muted}]",
            subtitle_align="left",
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# 메시지 구분선 & 헤더
# ---------------------------------------------------------------------------


def render_user_header(console: "Console", content: str = "") -> None:
    """사용자 메시지 헤더 (아이콘 + 타임스탬프)"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette
    now = datetime.now().strftime("%H:%M")

    header = Text()
    header.append(f" {theme.icons['user']} ", style=f"bold {p.user_color}")
    header.append("You", style=f"bold {p.user_color}")
    header.append(f"  {now}", style=f"{p.muted}")
    console.print(header)


def render_assistant_header(console: "Console", model: str = "") -> None:
    """AI 응답 헤더"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette
    now = datetime.now().strftime("%H:%M")

    header = Text()
    header.append(f" {theme.icons['assistant']} ", style=f"bold {p.assistant_color}")
    header.append("Assistant", style=f"bold {p.assistant_color}")
    if model:
        header.append(f"  ({model})", style=f"{p.muted}")
    header.append(f"  {now}", style=f"{p.muted}")
    console.print(header)


def render_separator(console: "Console") -> None:
    """메시지 간 구분선"""
    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    width = min(console.width, 80)
    console.print(f"[{theme.palette.separator}]{'─' * width}[/{theme.palette.separator}]")


# ---------------------------------------------------------------------------
# 단계별 Thinking 표시
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]
_spinner_cycle = itertools.cycle(_SPINNER_FRAMES)


def get_next_spinner_frame() -> str:
    """스피너 프레임 순환"""
    return next(_spinner_cycle)


_THINKING_PHASES = {
    "thinking": ("◐", "Thinking...", "brand_primary"),
    "searching": ("◉", "Searching...", "info_color"),
    "reading": ("◎", "Reading...", "success_color"),
    "writing": ("◈", "Writing...", "assistant_color"),
    "analyzing": ("◆", "Analyzing...", "warning_color"),
}


def render_thinking(console: "Console", phase: str = "thinking") -> None:
    """단계별 Thinking 표시"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    icon, label, color_attr = _THINKING_PHASES.get(phase, _THINKING_PHASES["thinking"])
    color = getattr(theme.palette, color_attr, theme.palette.brand_primary)
    console.print(Text(f"  {icon} {label}", style=f"{color}"))


def render_thinking_static(console: "Console") -> None:
    """정적 Thinking 표시 (스트리밍 전 한 줄)"""
    render_thinking(console, "thinking")


class ThinkingSpinner:
    """
    비동기 스피너 컨텍스트 매니저

    사용:
        async with ThinkingSpinner(console):
            answer = await _run_chat(...)
    """

    def __init__(self, console: "Console") -> None:
        self._console = console
        self._live: Optional[object] = None

    async def __aenter__(self) -> "ThinkingSpinner":
        from rich.live import Live
        from rich.text import Text

        from beanllm.ui.interactive.themes import get_theme

        theme = get_theme()
        live = Live(
            Text(
                f"  {theme.icons['thinking']} Thinking...", style=f"{theme.palette.brand_primary}"
            ),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        live.__enter__()
        self._live = live
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if self._live is not None:
            live = self._live
            if hasattr(live, "__exit__"):
                live.__exit__(exc_type, exc_val, exc_tb)  # type: ignore[union-attr]

    def update_phase(self, phase: str) -> None:
        """Thinking 단계 업데이트"""
        if self._live is not None:
            from rich.text import Text

            from beanllm.ui.interactive.themes import get_theme

            theme = get_theme()
            icon, label, color_attr = _THINKING_PHASES.get(phase, _THINKING_PHASES["thinking"])
            color = getattr(theme.palette, color_attr, theme.palette.brand_primary)
            if hasattr(self._live, "update"):
                self._live.update(Text(f"  {icon} {label}", style=f"{color}"))  # type: ignore[union-attr]

    def update_frame(self) -> None:
        """스피너 프레임 업데이트"""
        if self._live is not None:
            from rich.text import Text

            from beanllm.ui.interactive.themes import get_theme

            theme = get_theme()
            frame = get_next_spinner_frame()
            if hasattr(self._live, "update"):
                self._live.update(  # type: ignore[union-attr]
                    Text(f"  {frame} Thinking...", style=f"{theme.palette.brand_primary}")
                )


# ---------------------------------------------------------------------------
# 스타일된 에러/경고 패널
# ---------------------------------------------------------------------------


def render_error(console: "Console", message: str, suggestion: str = "") -> None:
    """스타일된 에러 패널"""
    from rich.panel import Panel
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette

    content = Text()
    content.append(f" {theme.icons['error']} ", style=f"bold {p.error_color}")
    content.append(message, style=f"{p.error_color}")
    if suggestion:
        content.append(f"\n {theme.icons['info']} ", style=f"{p.info_color}")
        content.append(suggestion, style=f"{p.dim}")

    console.print(
        Panel(
            content, border_style=p.error_color, padding=(0, 1), title="Error", title_align="left"
        )
    )


def _render_styled_message(
    console: "Console",
    message: str,
    icon_key: str,
    icon_color: str,
    text_color: str,
    bold_icon: bool = False,
) -> None:
    """공통 스타일 메시지 렌더링 (warning/success/info 통합)"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    icon_style = f"bold {icon_color}" if bold_icon else icon_color
    text = Text()
    text.append(f"  {theme.icons[icon_key]} ", style=icon_style)
    text.append(message, style=text_color)
    console.print(text)


def render_warning(console: "Console", message: str) -> None:
    """스타일된 경고 메시지"""
    from beanllm.ui.interactive.themes import get_theme

    p = get_theme().palette
    _render_styled_message(console, message, "warning", p.warning_color, p.warning_color)


def render_success(console: "Console", message: str) -> None:
    """성공 메시지"""
    from beanllm.ui.interactive.themes import get_theme

    p = get_theme().palette
    _render_styled_message(
        console, message, "success", p.success_color, p.success_color, bold_icon=True
    )


def render_info(console: "Console", message: str) -> None:
    """정보 메시지"""
    from beanllm.ui.interactive.themes import get_theme

    p = get_theme().palette
    _render_styled_message(console, message, "info", p.info_color, p.dim)


# ---------------------------------------------------------------------------
# 컨텍스트 인디케이터
# ---------------------------------------------------------------------------


def render_context_indicator(
    console: "Console",
    files: int = 0,
    rag_active: bool = False,
    token_count: int = 0,
) -> None:
    """첨부파일, RAG 상태, 토큰 사용량 시각화"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette

    parts = Text()
    has_content = False
    if files > 0:
        parts.append(
            f"  {theme.icons['file']} {files} file{'s' if files > 1 else ''}",
            style=f"{p.info_color}",
        )
        has_content = True
    if rag_active:
        if has_content:
            parts.append("  ·  ", style=p.muted)
        parts.append(f"{theme.icons['rag']} RAG active", style=f"{p.success_color}")
        has_content = True
    if token_count > 0:
        if has_content:
            parts.append("  ·  ", style=p.muted)
        parts.append(f"{theme.icons['token']} ~{token_count:,} chars", style=f"{p.dim}")
        has_content = True

    if has_content:
        console.print(parts)


def render_file_attached(console: "Console", path: str) -> None:
    """파일 첨부 알림"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette
    text = Text()
    text.append(f"  {theme.icons['file']} ", style=f"{p.info_color}")
    text.append(path, style=f"underline {p.dim}")
    console.print(text)


# ---------------------------------------------------------------------------
# 상태바
# ---------------------------------------------------------------------------


def render_status_bar(console: "Console", session: "ChatSession") -> None:
    """하단 상태바 - 테마 기반"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette
    role_name = getattr(session, "role_name", "default")
    total_chars = sum(len(m.content) for m in session.messages)
    msg_count = len(session.messages)

    bar = Text()
    bar.append(f" {theme.icons['model']} ", style=f"bold {p.brand_primary}")
    bar.append(f"{session.model}", style="bold")
    bar.append(" · ", style=p.muted)
    bar.append(f"{role_name}", style=f"{p.brand_accent}")
    bar.append(" · ", style=p.muted)
    bar.append(f"{session.mode}", style=f"{p.dim}")
    bar.append(" · ", style=p.muted)
    bar.append(f"{msg_count} msgs", style=f"{p.dim}")
    bar.append(" · ", style=p.muted)
    bar.append(f"~{total_chars:,} chars", style=f"{p.dim}")
    if session.verbose:
        bar.append(" · ", style=p.muted)
        bar.append("[verbose]", style=f"{p.warning_color}")

    console.print(bar)


# ---------------------------------------------------------------------------
# AI 응답 렌더링
# ---------------------------------------------------------------------------


def render_ai_markdown(console: "Console", content: str) -> None:
    """AI 응답을 마크다운으로 렌더링 (테마 기반)"""
    from rich.markdown import Markdown
    from rich.padding import Padding

    try:
        md = Markdown(
            content,
            code_theme="monokai",
            inline_code_lexer="python",
        )
        console.print(Padding(md, (0, 0, 0, 2)))
    except Exception:
        for line in content.split("\n"):
            console.print(f"  {line}")
    console.print()


def render_streaming_markdown(console: "Console", content: str) -> None:
    """스트리밍 중 마크다운 렌더링 (Live 업데이트용)"""
    from rich.markdown import Markdown
    from rich.padding import Padding

    try:
        md = Markdown(content, code_theme="monokai")
        return Padding(md, (0, 0, 0, 2))  # type: ignore[return-value]
    except Exception:
        from rich.text import Text

        return Padding(Text(content), (0, 0, 0, 2))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 히스토리 렌더링
# ---------------------------------------------------------------------------


def render_history_block(console: "Console", role: str, content: str, truncate: int = 300) -> None:
    """히스토리 복원용 간결 블록 (테마 기반)"""
    from rich.markdown import Markdown
    from rich.padding import Padding
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette

    display = content if len(content) <= truncate else content[:truncate] + "..."
    if role == "user":
        header = Text()
        header.append(f" {theme.icons['user']} ", style=f"bold {p.user_color}")
        header.append(display, style="bold")
        console.print(header)
    else:
        header = Text()
        header.append(f" {theme.icons['assistant']} ", style=f"bold {p.assistant_color}")
        console.print(header)
        try:
            console.print(Padding(Markdown(display, code_theme="monokai"), (0, 0, 0, 2)))
        except Exception:
            console.print(f"  {display}")
    console.print()


# ---------------------------------------------------------------------------
# 응답 메타 정보
# ---------------------------------------------------------------------------


def render_response_meta(
    console: "Console",
    elapsed_seconds: float = 0.0,
    char_count: int = 0,
) -> None:
    """응답 후 메타 정보 (시간, 길이)"""
    from rich.text import Text

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette

    meta = Text()
    if elapsed_seconds > 0:
        meta.append(f"  {theme.icons['time']} ", style=f"{p.muted}")
        meta.append(f"{elapsed_seconds:.1f}s", style=f"{p.dim}")
    if char_count > 0:
        if len(meta) > 0:
            meta.append("  ·  ", style=p.muted)
        meta.append(f"{char_count:,} chars", style=f"{p.dim}")
    if len(meta) > 0:
        console.print(meta)


# ---------------------------------------------------------------------------
# 프롬프트 (fallback - prompt_toolkit 미사용 시)
# ---------------------------------------------------------------------------


def render_prompt(console: "Console") -> str:
    """입력 프롬프트 (fallback)"""
    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette
    return console.input(
        f"[bold {p.brand_primary}]{theme.icons['prompt_chat']} [/bold {p.brand_primary}]"
    )


# ---------------------------------------------------------------------------
# 초기 프레임
# ---------------------------------------------------------------------------


def render_frame(
    console: "Console",
    session: "ChatSession",
    show_history: bool = True,
    show_logo: bool = True,
) -> None:
    """초기 화면 프레임"""
    if show_logo:
        render_logo(console)
    render_welcome(console, session)
    if show_history and session.messages:
        for msg in session.messages[-10:]:
            render_history_block(console, msg.role, msg.content)
