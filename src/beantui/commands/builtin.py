"""
Built-in Commands - 범용 슬래시 커맨드

beanllm에 의존하지 않는 순수 TUI 커맨드:
  /help, /model, /new, /plan, /chat, /verbose, /exit
  /role, /theme, /save, /load, /shell
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beantui.commands import register_command

if TYPE_CHECKING:
    from beantui.session import ChatSession


# ---------------------------------------------------------------------------
# 기본 커맨드
# ---------------------------------------------------------------------------


@register_command("help", description="도움말", aliases=["h"])
def cmd_help(session: "ChatSession", args: str) -> str:
    """전체 도움말"""
    from beantui.commands import CommandRegistry

    lines = [
        "  [bold cyan]beantui[/bold cyan] [dim]Interactive TUI[/dim]\n",
        "  [yellow]Input[/yellow]",
        "    메시지 입력        LLM과 대화",
        "    [green]@file[/green]            파일 참조  (Explain @src/main.py)",
        "    [green]![/green]                셸 모드 토글",
        "    [green]!cmd[/green]             셸 실행    (!ls -la)",
        "    [green]Meta+Enter[/green]       멀티라인 입력",
        "    [green]Ctrl+R[/green]           히스토리 검색",
        "    [green]Esc×2[/green]            입력 클리어",
        "    [green]Tab[/green]              자동완성\n",
        "  [yellow]Commands[/yellow]",
    ]
    for name, entry in CommandRegistry.visible_commands().items():
        usage_str = f" [dim]{entry.usage}[/dim]" if entry.usage else ""
        lines.append(f"    [green]/{name:<14}[/green]{entry.description}{usage_str}")

    return "\n".join(lines)


@register_command("model", description="모델 변경", usage="/model [name]")
def cmd_model(session: "ChatSession", args: str) -> str:
    if not args.strip():
        return f"[dim]Current model: {session.model}[/dim]"
    session.set_model(args.strip())
    return f"[green]✓ Model set to {session.model}[/green]"


@register_command("new", description="새 세션", aliases=["clear"])
def cmd_new(session: "ChatSession", args: str) -> str:
    session.clear()
    return "[green]✓ New session started[/green]"


@register_command("plan", description="플랜 모드")
def cmd_plan(session: "ChatSession", args: str) -> str:
    session.mode = "plan"
    session.system = (
        "You are a planning assistant. Help the user break down tasks, create plans, "
        "and outline steps. Be structured and actionable."
    )
    return "[green]✓ Plan mode[/green]"


@register_command("chat", description="채팅 모드")
def cmd_chat(session: "ChatSession", args: str) -> str:
    session.mode = "chat"
    session.system = "You are a helpful assistant."
    return "[green]✓ Chat mode[/green]"


@register_command("verbose", description="로그 토글", aliases=["logs"])
def cmd_verbose(session: "ChatSession", args: str) -> str:
    session.verbose = not session.verbose
    status = "on" if session.verbose else "off"
    return f"[green]✓ Verbose logs {status}[/green]"


@register_command("exit", description="종료", aliases=["q", "quit"], hidden=False)
def cmd_exit(session: "ChatSession", args: str) -> str:
    return "__EXIT__"


@register_command("shell", description="셸 실행", usage="/shell <cmd>", hidden=True)
def cmd_shell(session: "ChatSession", args: str) -> str:
    if not args.strip():
        return ""
    from beantui.input_parser import run_shell_command

    code, stdout, stderr = run_shell_command(args)
    output = stdout or stderr or "(no output)"
    session.attach_shell_output(args, output)
    return f"[dim]Shell: {args}[/dim]\n{output}"


# ---------------------------------------------------------------------------
# Role 커맨드
# ---------------------------------------------------------------------------


@register_command("role", description="역할 변경", usage="/role [name]", is_async=True)
async def cmd_role(session: "ChatSession", args: str) -> str:
    """역할 변경 — 인자 없으면 인터랙티브 선택"""
    from beantui.roles import BUILTIN_ROLES, get_role

    name = args.strip().lower()

    if not name:
        from beantui.commands.interactive import interactive_role_select

        name = await interactive_role_select()
        if not name:
            return "[dim]Cancelled[/dim]"

    role = get_role(name)
    if role is None:
        from beantui.roles import get_role_list_display

        return f"[red]Unknown role: {name}[/red]\n{get_role_list_display()}"

    session.system = role.system_prompt
    session.role_name = name
    return f"[green]✓ Role set to {role.icon} {role.name}[/green] — {role.description}"


# ---------------------------------------------------------------------------
# Theme 커맨드
# ---------------------------------------------------------------------------


@register_command("theme", description="테마 전환", usage="/theme [name]", is_async=True)
async def cmd_theme(session: "ChatSession", args: str) -> str:
    """테마 전환 — 인자 없으면 인터랙티브 선택"""
    from beantui.themes import set_theme

    name = args.strip().lower()

    if not name:
        from beantui.commands.interactive import interactive_theme_select

        name = await interactive_theme_select()
        if not name:
            return "[dim]Cancelled[/dim]"

    if name not in ("dark", "light"):
        return f"[red]Unknown theme: {name}[/red]  [dim](options: dark, light)[/dim]"

    set_theme(name)
    return f"[green]✓ Theme set to {name}[/green]"


# ---------------------------------------------------------------------------
# 세션 저장/복원
# ---------------------------------------------------------------------------


@register_command("save", description="세션 저장", usage="/save [title]")
def cmd_save(session: "ChatSession", args: str) -> str:
    from beantui.session_store import SessionStore

    store = SessionStore()
    title = args.strip() or None
    sid = store.save_session(session, title=title)
    store.close()
    if sid < 0:
        return "[yellow]Nothing to save (empty session)[/yellow]"
    return f"[green]✓ Session saved (id={sid})[/green]"


@register_command("load", description="세션 불러오기", usage="/load [id]")
def cmd_load(session: "ChatSession", args: str) -> str:
    from beantui.session_store import SessionStore

    store = SessionStore()
    sid_str = args.strip()

    if not sid_str:
        sessions = store.list_sessions(limit=10)
        store.close()
        if not sessions:
            return "[dim]No saved sessions[/dim]"
        lines = ["  [bold cyan]Saved Sessions[/bold cyan]\n"]
        for s in sessions:
            lines.append(
                f"  [green]#{s['id']}[/green]  {s['title'][:40]}"
                f"  [dim]{s['model']} · {s['updated_at'][:16]}[/dim]"
            )
        lines.append("\n  [dim]Usage: /load <id>[/dim]")
        return "\n".join(lines)

    try:
        sid = int(sid_str)
    except ValueError:
        store.close()
        return f"[red]Invalid session id: {sid_str}[/red]"

    loaded = store.load_session(sid)
    store.close()
    if loaded is None:
        return f"[red]Session #{sid} not found[/red]"

    session.model = loaded.model
    session.provider = loaded.provider
    session.system = loaded.system
    session.mode = loaded.mode
    session.messages = loaded.messages
    return f"[green]✓ Session #{sid} loaded ({len(loaded.messages)} messages)[/green]"
