"""
Input Parser - 하위 호환 래퍼

beantui.input_parser로 위임합니다.
"""

from beantui.input_parser import (
    _strip_ansi_escapes,
    parse_user_input,
    resolve_file_path,
    resolve_file_references,
    run_shell_command,
)

__all__ = [
    "_strip_ansi_escapes",
    "parse_user_input",
    "resolve_file_references",
    "resolve_file_path",
    "run_shell_command",
]
