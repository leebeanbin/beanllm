"""
Themes - 하위 호환 래퍼

beantui.themes로 위임합니다.
기존 코드가 이 모듈을 import하는 경우를 위한 re-export.
"""

from beantui.themes import (
    LightPalette,
    Theme,
    ThemePalette,
    get_theme,
    set_theme,
)

__all__ = [
    "Theme",
    "ThemePalette",
    "LightPalette",
    "get_theme",
    "set_theme",
]
