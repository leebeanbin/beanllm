"""
beantui - Reusable interactive TUI engine for AI/LLM applications

Usage:
    from beantui import TUIEngine, TUIConfig, register_command
    from beantui.protocols import ChatBackend

    class MyBackend:
        async def stream_chat(self, messages, system, model, temperature):
            ...
        async def get_default_model(self) -> str:
            return "my-model"
        def sanitize_error(self, error: Exception) -> str:
            return str(error)

    config = TUIConfig.from_toml("beantui.toml")
    engine = TUIEngine(config=config, backend=MyBackend())
    engine.run()
"""

# builtin 커맨드 자동 등록 (import 시점에 @register_command 실행)
import beantui.commands.builtin as _  # noqa: F401, E402
from beantui.commands import register_command
from beantui.config import TUIConfig
from beantui.engine import TUIEngine
from beantui.protocols import ChatBackend, EchoBackend
from beantui.session import ChatMessage, ChatSession

__all__ = [
    "TUIEngine",
    "TUIConfig",
    "ChatBackend",
    "EchoBackend",
    "ChatSession",
    "ChatMessage",
    "register_command",
]

__version__ = "0.1.0"
