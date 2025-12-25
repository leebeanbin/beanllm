"""
Console Utilities
터미널 콘솔 유틸리티
"""

from typing import Optional

from rich.console import Console

# 전역 Console 인스턴스
_console: Optional[Console] = None


def get_console() -> Console:
    """Console 인스턴스 반환 (싱글톤)"""
    global _console
    if _console is None:
        _console = Console()
    return _console


def styled_print(text: str, style: str = ""):
    """스타일 적용 출력"""
    console = get_console()
    if style:
        console.print(f"[{style}]{text}[/{style}]")
    else:
        console.print(text)
