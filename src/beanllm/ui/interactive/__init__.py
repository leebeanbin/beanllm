"""
Interactive TUI - Cursor/Claude Code 스타일 채팅

beantui 엔진 기반 인터랙티브 TUI.
기존 API를 완벽히 유지하면서 내부는 beantui로 위임합니다.
"""

from beanllm.ui.interactive.tui import run_interactive_tui

# 하위 호환: beantui.session의 ChatSession을 re-export
from beantui.session import ChatSession

__all__ = ["ChatSession", "run_interactive_tui"]
