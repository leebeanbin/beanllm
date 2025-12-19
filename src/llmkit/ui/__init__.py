"""
Terminal UI Design System
터미널 기반 제품의 시각 아이덴티티 + UI 패턴
"""
from .design_tokens import (
    ColorTokens,
    TypographyTokens,
    SpacingTokens,
    DesignTokens
)
from .components import (
    Badge,
    Spinner,
    ProgressBar,
    CommandBlock,
    OutputBlock,
    Divider,
    Prompt,
    StatusIcon
)
from .logo import Logo, print_logo
from .patterns import (
    SuccessPattern,
    ErrorPattern,
    WarningPattern,
    InfoPattern,
    EmptyStatePattern,
    OnboardingPattern
)
from .console import get_console, styled_print

__all__ = [
    # Design Tokens
    "ColorTokens",
    "TypographyTokens",
    "SpacingTokens",
    "DesignTokens",
    # Components
    "Badge",
    "Spinner",
    "ProgressBar",
    "CommandBlock",
    "OutputBlock",
    "Divider",
    "Prompt",
    "StatusIcon",
    # Logo
    "Logo",
    "print_logo",
    # Patterns
    "SuccessPattern",
    "ErrorPattern",
    "WarningPattern",
    "InfoPattern",
    "EmptyStatePattern",
    "OnboardingPattern",
    # Console
    "get_console",
    "styled_print",
]

