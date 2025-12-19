"""
Terminal UI Design System
터미널 기반 제품의 시각 아이덴티티 + UI 패턴
"""

from .components import (
    Badge,
    CommandBlock,
    Divider,
    OutputBlock,
    ProgressBar,
    Prompt,
    Spinner,
    StatusIcon,
)
from .console import get_console, styled_print
from .design_tokens import ColorTokens, DesignTokens, SpacingTokens, TypographyTokens
from .logo import Logo, print_logo
from .patterns import (
    EmptyStatePattern,
    ErrorPattern,
    InfoPattern,
    OnboardingPattern,
    SuccessPattern,
    WarningPattern,
)

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
