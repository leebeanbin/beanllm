"""
Slash Commands - 하위 호환 래퍼

기존 코드가 이 모듈을 import하는 경우를 위해 beantui로 위임합니다.
새로운 코드는 beantui.commands를 직접 사용하세요.
"""

from __future__ import annotations

from typing import Any, Tuple

# builtin 커맨드를 import 시점에 자동 등록
import beantui.commands.builtin  # noqa: F401
from beantui.commands import CommandRegistry, execute_command


# 하위 호환: SLASH_COMMANDS 딕셔너리 (기존 코드가 in 연산자 사용)
class _SlashCommandsCompat(dict):  # type: ignore[type-arg]
    """CommandRegistry를 dict처럼 사용하는 래퍼"""

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return CommandRegistry.has(key)
        return False

    def __getitem__(self, key: str) -> Any:
        entry = CommandRegistry.get(key)
        if entry is None:
            raise KeyError(key)
        return entry


SLASH_COMMANDS = _SlashCommandsCompat()


async def execute_slash(cmd: str, args: str, session: Any) -> Tuple[bool, str]:
    """슬래시 커맨드 실행 (하위 호환)"""
    return await execute_command(cmd, args, session)


def get_slash_command_summary() -> str:
    """커맨드 요약 Rich 포맷 (하위 호환)"""
    cmds = CommandRegistry.visible_commands()
    lines = ["\n  [bold cyan]Commands[/bold cyan]\n"]
    for name, entry in cmds.items():
        usage = f" [dim]{entry.usage}[/dim]" if entry.usage else ""
        lines.append(
            f"  [green]/{name}[/green]{' ' * max(1, 14 - len(name))}{entry.description}{usage}"
        )
    return "\n".join(lines)
