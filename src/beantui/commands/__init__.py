"""
Command Registry - 슬래시 커맨드 등록/실행 시스템

@register_command 데코레이터 또는 CommandRegistry.register()로 커맨드 등록:

    from beantui.commands import register_command

    @register_command("deploy", description="서버 배포")
    def cmd_deploy(session, args):
        return "Deploying..."

설정 파일의 plugins 목록에서 모듈을 자동 import:

    [commands]
    plugins = ["myapp.tui_commands"]
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class CommandEntry:
    """슬래시 커맨드 엔트리"""

    name: str
    handler: Callable[..., Any]
    is_async: bool = False
    description: str = ""
    usage: str = ""
    aliases: List[str] = field(default_factory=list)
    hidden: bool = False


class CommandRegistry:
    """글로벌 커맨드 레지스트리

    데코레이터와 직접 등록 양쪽을 지원합니다.
    """

    _commands: Dict[str, CommandEntry] = {}
    _alias_map: Dict[str, str] = {}

    @classmethod
    def register(
        cls,
        name: str,
        handler: Callable[..., Any],
        is_async: bool = False,
        description: str = "",
        usage: str = "",
        aliases: Optional[List[str]] = None,
        hidden: bool = False,
    ) -> None:
        """커맨드 등록

        Args:
            name: 커맨드 이름 (슬래시 없이, 예: "help")
            handler: 핸들러 함수 (session, args) -> str
            is_async: async 핸들러 여부
            description: 설명 (자동완성 메타에 표시)
            usage: 사용법 (예: "/rag [path]")
            aliases: 별칭 목록 (예: ["h"] for help)
            hidden: True이면 /help에 표시하지 않음
        """
        entry = CommandEntry(
            name=name,
            handler=handler,
            is_async=is_async,
            description=description,
            usage=usage,
            aliases=aliases or [],
            hidden=hidden,
        )
        cls._commands[name] = entry
        for alias in entry.aliases:
            cls._alias_map[alias] = name

    @classmethod
    def unregister(cls, name: str) -> None:
        """커맨드 제거"""
        entry = cls._commands.pop(name, None)
        if entry:
            for alias in entry.aliases:
                cls._alias_map.pop(alias, None)

    @classmethod
    def get(cls, name: str) -> Optional[CommandEntry]:
        """이름 또는 별칭으로 커맨드 조회"""
        if name in cls._commands:
            return cls._commands[name]
        canonical = cls._alias_map.get(name)
        if canonical:
            return cls._commands.get(canonical)
        return None

    @classmethod
    def has(cls, name: str) -> bool:
        """커맨드 존재 여부"""
        return name in cls._commands or name in cls._alias_map

    @classmethod
    def all_commands(cls) -> Dict[str, CommandEntry]:
        """모든 등록된 커맨드"""
        return dict(cls._commands)

    @classmethod
    def visible_commands(cls) -> Dict[str, CommandEntry]:
        """hidden=False인 커맨드만"""
        return {k: v for k, v in cls._commands.items() if not v.hidden}

    @classmethod
    def command_meta(cls) -> Dict[str, Tuple[str, bool]]:
        """자동완성용 메타 정보: {name: (description, has_args)}"""
        meta: Dict[str, Tuple[str, bool]] = {}
        for name, entry in cls._commands.items():
            if not entry.hidden:
                has_args = bool(entry.usage and ("<" in entry.usage or "[" in entry.usage))
                meta[name] = (entry.description, has_args)
        return meta

    @classmethod
    def load_plugins(cls, module_paths: List[str]) -> None:
        """설정 파일의 plugins 목록에서 모듈을 import하여 자동 등록

        각 모듈은 import 시점에 @register_command 데코레이터를 통해
        커맨드가 자동으로 레지스트리에 등록됩니다.

        Args:
            module_paths: Python 모듈 경로 목록 (예: ["myapp.tui_commands"])
        """
        for path in module_paths:
            try:
                importlib.import_module(path)
            except ImportError as e:
                import warnings

                warnings.warn(f"Failed to load command plugin '{path}': {e}", stacklevel=2)

    @classmethod
    def reset(cls) -> None:
        """레지스트리 초기화 (테스트용)"""
        cls._commands.clear()
        cls._alias_map.clear()


# ---------------------------------------------------------------------------
# 데코레이터
# ---------------------------------------------------------------------------


def register_command(
    name: str,
    description: str = "",
    usage: str = "",
    is_async: bool = False,
    aliases: Optional[List[str]] = None,
    hidden: bool = False,
) -> Callable[..., Any]:
    """커맨드 등록 데코레이터

    Example:
        >>> @register_command("greet", description="인사", usage="/greet [name]")
        ... def cmd_greet(session, args):
        ...     name = args.strip() or "World"
        ...     return f"Hello, {name}!"
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        CommandRegistry.register(
            name=name,
            handler=func,
            is_async=is_async,
            description=description,
            usage=usage,
            aliases=aliases,
            hidden=hidden,
        )
        return func

    return decorator


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------


async def execute_command(
    cmd: str,
    args: str,
    session: Any,
) -> Tuple[bool, str]:
    """슬래시 커맨드 실행

    Args:
        cmd: 커맨드 이름
        args: 인자 문자열
        session: ChatSession 인스턴스

    Returns:
        (should_exit, result_text)
    """
    entry = CommandRegistry.get(cmd)
    if entry is None:
        return False, f"[red]Unknown command: /{cmd}[/red]"

    try:
        if entry.is_async:
            result = await entry.handler(session, args)
        else:
            result = entry.handler(session, args)

        if result == "__EXIT__":
            return True, ""

        return False, str(result) if result else ""
    except Exception as e:
        return False, f"[red]Error: {e}[/red]"
