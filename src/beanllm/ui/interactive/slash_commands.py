"""
Slash Commands - /help, /model, /search, /rag, /role, /save, /load 등
"""

from __future__ import annotations

from typing import Any, Dict

from beanllm.ui.interactive.input_parser import run_shell_command
from beanllm.ui.interactive.session import ChatSession


def get_slash_command_summary() -> str:
    """'/'만 입력 시 표시할 커맨드 요약"""
    return """
  [bold cyan]Commands[/bold cyan]

  [green]/help[/green]           전체 도움말
  [green]/model[/green] [dim][name][/dim]   모델 변경
  [green]/role[/green] [dim][name][/dim]    역할 변경  (coder, writer, ...)
  [green]/theme[/green] [dim][name][/dim]   테마 전환  (dark, light)
  [green]/new[/green]            새 세션
  [green]/save[/green]           세션 저장
  [green]/load[/green] [dim][id][/dim]     세션 불러오기
  [green]/plan[/green]           플랜 모드
  [green]/chat[/green]           채팅 모드
  [green]/rag[/green] [dim][path][/dim]    RAG 검색
  [green]/search[/green] [dim]<q>[/dim]    웹 검색
  [green]/agent[/green] [dim]<task>[/dim]  에이전트
  [green]/eval[/green]           평가
  [green]/verbose[/green]        로그 토글
  [green]/exit[/green]           종료
"""


def cmd_help(session: ChatSession, args: str) -> str:
    return """
  [bold cyan]beanllm[/bold cyan] [dim]by leebeanbin[/dim]

  [yellow]Input[/yellow]
    메시지 입력        LLM과 대화
    [green]@file[/green]            파일 참조  (Explain @src/main.py)
    [green]![/green]                셸 모드 토글
    [green]!cmd[/green]             셸 실행    (!ls -la)
    [green]Meta+Enter[/green]       멀티라인 입력
    [green]Ctrl+R[/green]           히스토리 검색
    [green]Esc×2[/green]            입력 클리어
    [green]Tab[/green]              자동완성

  [yellow]Commands[/yellow]
    [green]/help[/green]            도움말
    [green]/model[/green] [dim][name][/dim]    모델 변경
    [green]/role[/green] [dim][name][/dim]     역할  (coder, writer, translator, ...)
    [green]/theme[/green] [dim][name][/dim]    테마 전환  (dark, light)
    [green]/new[/green]             새 세션
    [green]/save[/green]            세션 저장
    [green]/load[/green] [dim][id][/dim]      세션 불러오기
    [green]/plan[/green]            플랜 모드
    [green]/chat[/green]            채팅 모드
    [green]/rag[/green] [dim][path][/dim]     RAG 검색
    [green]/search[/green] [dim]<q>[/dim]     웹 검색
    [green]/agent[/green] [dim]<task>[/dim]   에이전트
    [green]/eval[/green]            평가
    [green]/verbose[/green]         로그 토글
    [green]/exit[/green]            종료 (also: exit, q, quit)
"""


def cmd_model(session: ChatSession, args: str) -> str:
    if not args.strip():
        return f"[dim]Current model: {session.model}[/dim]"
    session.set_model(args.strip())
    return f"[green]✓ Model set to {session.model}[/green]"


def cmd_new(session: ChatSession, args: str) -> str:
    session.clear()
    return "[green]✓ New session started[/green]"


def cmd_plan(session: ChatSession, args: str) -> str:
    session.mode = "plan"
    session.system = (
        "You are a planning assistant. Help the user break down tasks, create plans, "
        "and outline steps. Be structured and actionable."
    )
    return "[green]✓ Plan mode[/green]"


def cmd_chat(session: ChatSession, args: str) -> str:
    session.mode = "chat"
    session.system = "You are a helpful assistant."
    return "[green]✓ Chat mode[/green]"


def cmd_verbose(session: ChatSession, args: str) -> str:
    session.verbose = not session.verbose
    from beanllm.ui.interactive.tui import _set_verbose

    _set_verbose(session.verbose)
    status = "on" if session.verbose else "off"
    return f"[green]✓ Verbose logs {status}[/green]"


def cmd_exit(session: ChatSession, args: str) -> str:
    return "__EXIT__"


def cmd_search(session: ChatSession, args: str) -> str:
    if not args.strip():
        return "[yellow]Usage: /search <query>[/yellow]"
    try:
        from beanllm import WebSearch

        ws = WebSearch()
        result = ws.search(args.strip(), max_results=5)
        lines = []
        for i, r in enumerate(result.results[:5], 1):
            lines.append(f"  {i}. [cyan]{r.title}[/cyan]")
            lines.append(f"     {r.url}")
            if r.snippet:
                lines.append(f"     [dim]{r.snippet[:100]}...[/dim]")
        return "\n".join(lines) if lines else "[dim]No results[/dim]"
    except Exception as e:
        return f"[red]Search error: {e}[/red]"


async def cmd_rag(session: ChatSession, args: str) -> str:
    path = args.strip() or "."
    try:
        from beanllm import RAGChain

        rag = RAGChain.from_documents(path)
        session.attached_contexts.clear()
        session._rag["path"] = path
        session._rag["chain"] = rag
        return f"[green]✓ RAG loaded from {path}[/green]"
    except Exception as e:
        return f"[red]RAG error: {e}[/red]"


async def cmd_agent(session: ChatSession, args: str) -> str:
    if not args.strip():
        return "[yellow]Usage: /agent <task>[/yellow]"
    try:
        from beanllm import Agent

        agent = Agent(model=session.model)
        result = await agent.run(args.strip())
        return result.answer or "[dim]No answer[/dim]"
    except Exception as e:
        return f"[red]Agent error: {e}[/red]"


def cmd_eval(session: ChatSession, args: str) -> str:
    parts = args.strip().split(maxsplit=1)
    if len(parts) < 2:
        return "[yellow]Usage: /eval <prediction> <reference>[/yellow]"
    pred, ref = parts[0], parts[1]
    try:
        from beanllm import EvaluatorFacade
        from beanllm.domain.evaluation.metrics import BLEUMetric, ROUGEMetric

        ev = EvaluatorFacade()
        ev.add_metric(BLEUMetric()).add_metric(ROUGEMetric())
        result = ev.evaluate(pred, ref)
        lines = [f"  {m}: {s:.4f}" for m, s in result.scores.items()]
        return "\n".join(lines) if lines else "[dim]No scores[/dim]"
    except Exception as e:
        return f"[red]Eval error: {e}[/red]"


def cmd_shell(session: ChatSession, args: str) -> str:
    if not args.strip():
        return ""
    code, stdout, stderr = run_shell_command(args)
    output = stdout or stderr or "(no output)"
    session.attach_shell_output(args, output)
    return f"[dim]Shell: {args}[/dim]\n{output}"


async def cmd_role(session: ChatSession, args: str) -> str:
    """역할 변경 커맨드 — 인자 없으면 인터랙티브 선택"""
    from beanllm.ui.interactive.roles import BUILTIN_ROLES, get_role

    name = args.strip().lower()

    # 인자 없으면 인터랙티브 선택 메뉴
    if not name:
        name = await _interactive_role_select()
        if not name:
            return "[dim]Cancelled[/dim]"

    role = get_role(name)
    if role is None:
        from beanllm.ui.interactive.roles import get_role_list_display

        return f"[red]Unknown role: {name}[/red]\n{get_role_list_display()}"

    session.system = role.system_prompt
    session.role_name = name  # type: ignore[attr-defined]
    return f"[green]✓ Role set to {role.icon} {role.name}[/green] — {role.description}"


async def _interactive_select(
    title: str,
    items: list[tuple[str, str, str, str]],
    default_idx: int = 0,
) -> str:
    """
    공통 인터랙티브 선택 메뉴 (화살표 키 + Enter)

    Args:
        title: 메뉴 타이틀
        items: [(value, icon, label, description), ...]
        default_idx: 초기 선택 인덱스

    Returns:
        선택된 value 문자열 (취소 시 빈 문자열)
    """
    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()
    p = theme.palette

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.styles import Style

        selected_idx = {"value": default_idx}
        cancelled = {"value": False}

        bindings = KeyBindings()

        @bindings.add(Keys.Up)
        def _up(event) -> None:  # type: ignore[no-untyped-def]
            if selected_idx["value"] > 0:
                selected_idx["value"] -= 1

        @bindings.add(Keys.Down)
        def _down(event) -> None:  # type: ignore[no-untyped-def]
            if selected_idx["value"] < len(items) - 1:
                selected_idx["value"] += 1

        @bindings.add(Keys.Enter)
        def _enter(event) -> None:  # type: ignore[no-untyped-def]
            event.app.exit(result="")

        @bindings.add(Keys.Escape)
        def _esc(event) -> None:  # type: ignore[no-untyped-def]
            cancelled["value"] = True
            event.app.exit(result="")

        pt_style = Style.from_dict(theme.prompt_toolkit_style)

        def _build_menu() -> HTML:
            lines = [
                f"<b>  {title}</b>  <style fg='{p.muted}'>(↑↓ navigate, Enter select, Esc cancel)</style>\n"
            ]
            for i, (_value, icon, label, desc) in enumerate(items):
                if i == selected_idx["value"]:
                    lines.append(
                        f"  <style fg='{p.brand_primary}'><b>▸ {icon} {label:<12}</b></style>"
                        f"  <style fg='{p.text}'>{desc}</style>"
                    )
                else:
                    lines.append(
                        f"  <style fg='{p.dim}'>  {icon} {label:<12}</style>"
                        f"  <style fg='{p.muted}'>{desc}</style>"
                    )
            return HTML("\n".join(lines))

        ps = PromptSession(key_bindings=bindings, style=pt_style)
        await ps.prompt_async(message=_build_menu, refresh_interval=0.1)

        if cancelled["value"]:
            return ""
        return items[selected_idx["value"]][0]

    except ImportError:
        return ""


async def _interactive_role_select() -> str:
    """인터랙티브 역할 선택"""
    from beanllm.ui.interactive.roles import BUILTIN_ROLES

    items = [(name, role.icon, name, role.description) for name, role in BUILTIN_ROLES.items()]
    return await _interactive_select("Select Role", items)


def cmd_save(session: ChatSession, args: str) -> str:
    """세션 저장"""
    from beanllm.ui.interactive.session_store import SessionStore

    store = SessionStore()
    title = args.strip() or None
    sid = store.save_session(session, title=title)
    store.close()
    if sid < 0:
        return "[yellow]Nothing to save (empty session)[/yellow]"
    return f"[green]✓ Session saved (id={sid})[/green]"


def cmd_load(session: ChatSession, args: str) -> str:
    """세션 불러오기"""
    from beanllm.ui.interactive.session_store import SessionStore

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

    # 세션 복원
    session.model = loaded.model
    session.provider = loaded.provider
    session.system = loaded.system
    session.mode = loaded.mode
    session.messages = loaded.messages
    return f"[green]✓ Session #{sid} loaded ({len(loaded.messages)} messages)[/green]"


async def cmd_theme(session: ChatSession, args: str) -> str:
    """테마 전환 커맨드 — 인자 없으면 인터랙티브 선택"""
    from beanllm.ui.interactive.themes import get_theme, set_theme

    name = args.strip().lower()

    if not name:
        name = await _interactive_theme_select()
        if not name:
            return "[dim]Cancelled[/dim]"

    if name not in ("dark", "light"):
        return f"[red]Unknown theme: {name}[/red]  [dim](options: dark, light)[/dim]"

    set_theme(name)
    return f"[green]✓ Theme set to {name}[/green]"


async def _interactive_theme_select() -> str:
    """인터랙티브 테마 선택"""
    from beanllm.ui.interactive.themes import get_theme

    current = get_theme().name
    items = [
        ("dark", "🌙", "Dark Mode", "어두운 배경, 밝은 텍스트"),
        ("light", "☀️", "Light Mode", "밝은 배경, 어두운 텍스트"),
    ]
    default_idx = 0 if current == "dark" else 1
    return await _interactive_select("Select Theme", items, default_idx=default_idx)


# ---------------------------------------------------------------------------
# 커맨드 레지스트리
# ---------------------------------------------------------------------------

SLASH_COMMANDS: Dict[str, Dict[str, Any]] = {
    "help": {"handler": cmd_help, "async": False},
    "h": {"handler": cmd_help, "async": False},
    "model": {"handler": cmd_model, "async": False},
    "new": {"handler": cmd_new, "async": False},
    "clear": {"handler": cmd_new, "async": False},
    "plan": {"handler": cmd_plan, "async": False},
    "chat": {"handler": cmd_chat, "async": False},
    "exit": {"handler": cmd_exit, "async": False},
    "q": {"handler": cmd_exit, "async": False},
    "quit": {"handler": cmd_exit, "async": False},
    "search": {"handler": cmd_search, "async": False},
    "rag": {"handler": cmd_rag, "async": True},
    "agent": {"handler": cmd_agent, "async": True},
    "eval": {"handler": cmd_eval, "async": False},
    "verbose": {"handler": cmd_verbose, "async": False},
    "logs": {"handler": cmd_verbose, "async": False},
    "shell": {"handler": cmd_shell, "async": False},
    "role": {"handler": cmd_role, "async": True},
    "theme": {"handler": cmd_theme, "async": True},
    "save": {"handler": cmd_save, "async": False},
    "load": {"handler": cmd_load, "async": False},
}


async def execute_slash(cmd: str, args: str, session: ChatSession) -> tuple[bool, str]:
    if cmd not in SLASH_COMMANDS:
        return False, f"[red]Unknown command: /{cmd}[/red]"

    entry = SLASH_COMMANDS[cmd]
    handler = entry["handler"]
    is_async = entry["async"]

    try:
        if is_async:
            result = await handler(session, args)
        else:
            result = handler(session, args)

        if result == "__EXIT__":
            return True, ""

        return False, str(result) if result else ""
    except Exception as e:
        return False, f"[red]Error: {e}[/red]"
