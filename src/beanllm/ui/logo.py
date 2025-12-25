"""
Terminal Logo
ASCII/Unicode 기반 로고
도움 패키지로서의 정체성: 커맨드 중심 설명
"""

from typing import Optional

from rich.console import Console
from rich.text import Text

from .console import get_console


class Logo:
    """터미널 로고 - 더 예쁜 ASCII 아트"""

    # ASCII Logo (beanllm 텍스트 - Big 폰트 스타일)
    ASCII = """
██╗     ██╗     ███╗   ███╗██╗  ██╗██╗████████╗
██║     ██║     ████╗ ████║██║ ██╔╝██║╚══██╔══╝
██║     ██║     ██╔████╔██║█████╔╝ ██║   ██║   
██║     ██║     ██║╚██╔╝██║██╔═██╗ ██║   ██║   
███████╗███████╗██║ ╚═╝ ██║██║  ██╗██║   ██║   
╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   
"""

    # Unicode Logo (box, line)
    UNICODE = """
┌─────┐ ┌─────┐ ┌─────┐
│ llm │ │ kit │ │  🚀 │
└─────┘ └─────┘ └─────┘
"""

    # Simple Text Logo
    SIMPLE = """
beanllm
"""

    # Minimal Logo
    MINIMAL = "beanllm"

    # 모토
    MOTTO = "Claude Code"  # 명확하고 간결한 코드처럼

    @classmethod
    def render(
        cls,
        style: str = "ascii",
        color: str = "magenta",
        show_motto: bool = True,
        show_commands: bool = False,
        console: Optional[Console] = None,
    ) -> None:
        """
        로고 렌더링

        Args:
            style: 로고 스타일 (ascii, unicode, simple, minimal)
            color: 색상
            show_motto: 모토 표시 여부
            show_commands: 도움 커맨드 표시 여부 (도움 패키지로서)
        """
        if console is None:
            console = get_console()

        # 로고 텍스트 선택
        logo_text = getattr(cls, style.upper(), cls.ASCII)

        # 로고 출력
        text = Text(logo_text.strip())
        text.stylize(f"bold {color}")
        console.print(text)

        # 모토 출력
        if show_motto:
            motto_text = Text(f"  {cls.MOTTO}", style=f"dim {color} italic")
            console.print(motto_text)

        # 도움 커맨드 출력 (도움 패키지로서)
        if show_commands:
            console.print()
            commands_text = Text()
            commands_text.append("  Try: ", style="dim")
            commands_text.append("beanllm list", style=f"{color} bold")
            commands_text.append("  |  ", style="dim")
            commands_text.append("beanllm show <model>", style=f"{color} bold")
            commands_text.append("  |  ", style="dim")
            commands_text.append("beanllm --help", style=f"{color} bold")
            console.print(commands_text)

        console.print()

    @classmethod
    def get_text(cls, style: str = "ascii") -> str:
        """로고 텍스트 반환"""
        return getattr(cls, style.upper(), cls.ASCII).strip()


def print_logo(
    style: str = "ascii",
    color: str = "magenta",
    show_motto: bool = True,
    show_commands: bool = False,
):
    """
    로고 출력 헬퍼

    Args:
        style: 로고 스타일
        color: 색상
        show_motto: 모토 표시 여부
        show_commands: 도움 커맨드 표시 여부 (도움 패키지로서)
    """
    Logo.render(style=style, color=color, show_motto=show_motto, show_commands=show_commands)
