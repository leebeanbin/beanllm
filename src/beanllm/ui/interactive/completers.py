"""
Completers - 하위 호환 래퍼

beantui.completers로 위임합니다.
"""

from beantui.completers import (
    FilePathCompleter,
    SlashCommandCompleter,
    create_completer,
)

__all__ = ["SlashCommandCompleter", "FilePathCompleter", "create_completer"]
