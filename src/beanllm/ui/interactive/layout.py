"""
Layout - 하위 호환 래퍼

beantui.layout으로 위임합니다.
기존 코드가 이 모듈을 import하는 경우를 위한 re-export.
"""

from beantui.layout import (
    ThinkingSpinner,
    render_ai_markdown,
    render_assistant_header,
    render_context_indicator,
    render_error,
    render_file_attached,
    render_frame,
    render_history_block,
    render_info,
    render_logo,
    render_prompt,
    render_response_meta,
    render_separator,
    render_success,
    render_thinking,
    render_user_header,
    render_warning,
    render_welcome,
)

__all__ = [
    "render_logo",
    "render_welcome",
    "render_user_header",
    "render_assistant_header",
    "render_separator",
    "render_thinking",
    "ThinkingSpinner",
    "render_error",
    "render_warning",
    "render_success",
    "render_info",
    "render_context_indicator",
    "render_file_attached",
    "render_ai_markdown",
    "render_history_block",
    "render_response_meta",
    "render_prompt",
    "render_frame",
]
