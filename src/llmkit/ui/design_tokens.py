"""
Design Tokens for Terminal UI
터미널 친화적인 디자인 토큰 정의
"""

from dataclasses import dataclass


@dataclass
class ColorTokens:
    """ANSI Color 기반 색상 토큰"""

    # Status Colors
    success: str = "green"
    error: str = "red"
    warning: str = "yellow"
    info: str = "cyan"
    accent: str = "magenta"
    muted: str = "dim white"

    # Semantic Colors
    primary: str = "cyan"
    secondary: str = "blue"
    highlight: str = "bold magenta"

    # Background Colors
    bg_success: str = "on green"
    bg_error: str = "on red"
    bg_warning: str = "on yellow"
    bg_info: str = "on cyan"

    def get_style(self, token: str) -> str:
        """토큰 이름으로 스타일 반환"""
        return getattr(self, token, self.muted)


@dataclass
class TypographyTokens:
    """터미널 타이포그래피 토큰"""

    # Font Weights
    bold: str = "bold"
    normal: str = ""
    dim: str = "dim"

    # Font Styles
    italic: str = "italic"
    underline: str = "underline"

    # Monospace (터미널 기본)
    mono: str = ""  # Rich는 기본적으로 monospace

    def apply(self, text: str, *styles: str) -> str:
        """스타일 적용"""
        style_str = " ".join(filter(None, styles))
        if style_str:
            return f"[{style_str}]{text}[/{style_str}]"
        return text


@dataclass
class SpacingTokens:
    """간격 토큰"""

    # Line Spacing
    line: int = 1
    line_sm: int = 0
    line_md: int = 1
    line_lg: int = 2

    # Block Spacing
    block: int = 2
    block_sm: int = 1
    block_md: int = 2
    block_lg: int = 3

    # Indent
    indent: int = 2
    indent_sm: int = 1
    indent_md: int = 2
    indent_lg: int = 4

    def get_newlines(self, size: str = "md") -> str:
        """줄바꿈 반환"""
        count = getattr(self, f"line_{size}", self.line)
        return "\n" * count

    def get_indent(self, size: str = "md") -> str:
        """들여쓰기 반환"""
        count = getattr(self, f"indent_{size}", self.indent)
        return " " * count


@dataclass
class DesignTokens:
    """통합 디자인 토큰"""

    color: ColorTokens = None
    typography: TypographyTokens = None
    spacing: SpacingTokens = None

    def __post_init__(self):
        if self.color is None:
            self.color = ColorTokens()
        if self.typography is None:
            self.typography = TypographyTokens()
        if self.spacing is None:
            self.spacing = SpacingTokens()

    @classmethod
    def default(cls):
        """기본 토큰 반환"""
        return cls(color=ColorTokens(), typography=TypographyTokens(), spacing=SpacingTokens())


# 전역 인스턴스
tokens = DesignTokens.default()
