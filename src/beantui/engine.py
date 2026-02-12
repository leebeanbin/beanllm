"""
TUI Engine - 범용 인터랙티브 TUI 메인 루프

Config + ChatBackend Protocol + CommandRegistry 기반.
beanllm에 의존하지 않는 순수 엔진.

사용:
    from beantui import TUIEngine, TUIConfig
    from beantui.protocols import EchoBackend

    config = TUIConfig.auto_discover()
    engine = TUIEngine(config=config, backend=EchoBackend())
    engine.run()
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

from beantui.config import TUIConfig
from beantui.input_parser import (
    _strip_ansi_escapes,
    parse_user_input,
    resolve_file_references,
)
from beantui.protocols import ChatBackend
from beantui.session import ChatSession

# ---------------------------------------------------------------------------
# 로그 차단
# ---------------------------------------------------------------------------


class _TUILogFilter(logging.Filter):
    """TUI 모드에서 내부 로그 차단 — blocked_prefixes는 config에서 주입"""

    verbose: bool = False
    _blocked: tuple[str, ...] = ()

    def filter(self, record: logging.LogRecord) -> bool:
        if self.verbose:
            return True
        return not any(record.name.startswith(p) for p in self._blocked)


class _StdoutGuard:
    """stdout 래퍼 - 로그 패턴 차단"""

    _LOG_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+-\s+")

    def __init__(self, original: io.TextIOWrapper) -> None:
        self._original = original
        self.verbose = False

    def write(self, text: str) -> int:
        if self.verbose:
            return self._original.write(text)
        if text.strip() and self._LOG_PATTERN.match(text.lstrip()):
            return len(text)
        return self._original.write(text)

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._original, name)


# ---------------------------------------------------------------------------
# 모델 응답 정리
# ---------------------------------------------------------------------------

_ARTIFACT_PATTERNS = [
    re.compile(r"\n---\s*\n+#{1,4}\s+(Instruction|Solution|Follow-up|Response)", re.IGNORECASE),
    re.compile(r"\n#{1,4}\s+Instruction[:\s]", re.IGNORECASE),
    re.compile(r"\n##\s+Instruction\s+in\s+", re.IGNORECASE),
    re.compile(r"<\|(end|assistant|user|im_end|im_start)\|>"),
    re.compile(r"\[INST\]|\[/INST\]"),
    re.compile(r"</?s>"),
]

_TRAILING_HR_PATTERN = re.compile(r"\n\s*---\s*$")


def _clean_response(text: str) -> str:
    for pat in _ARTIFACT_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[: m.start()]
    text = _remove_duplicate_blocks(text)
    text = _TRAILING_HR_PATTERN.sub("", text)
    return text.rstrip()


def _remove_duplicate_blocks(text: str) -> str:
    parts = re.split(r"\n\s*---\s*\n", text, maxsplit=1)
    if len(parts) != 2:
        return text
    first = parts[0].strip()
    second = parts[1].strip()
    if not first or not second:
        return text
    first_words = first.split()[:5]
    second_words = second.split()[:5]
    if len(first_words) >= 3 and first_words[:3] == second_words[:3]:
        return first
    return text


# ---------------------------------------------------------------------------
# TUIEngine
# ---------------------------------------------------------------------------


class TUIEngine:
    """범용 TUI 엔진

    Config + ChatBackend + CommandRegistry를 조합하여
    어떤 프로젝트에서든 재사용 가능한 인터랙티브 TUI를 제공합니다.

    Args:
        config: TUIConfig 인스턴스
        backend: ChatBackend 프로토콜 구현체
        working_dir: 작업 디렉토리

    Example:
        >>> engine = TUIEngine(
        ...     config=TUIConfig.auto_discover(),
        ...     backend=MyLLMBackend(),
        ... )
        >>> engine.run()
    """

    def __init__(
        self,
        config: Optional[TUIConfig] = None,
        backend: Optional[ChatBackend] = None,
        working_dir: Optional[Path] = None,
    ) -> None:
        self._config = config or TUIConfig()
        self._backend = backend
        self._working_dir = working_dir or Path.cwd()

        # 내부 상태
        self._log_filter = _TUILogFilter()
        self._log_filter._blocked = tuple(self._config.logging.blocked_prefixes)
        self._stdout_guard: Optional[_StdoutGuard] = None
        self._shell_mode: dict[str, bool] = {"active": False}
        self._console: Optional[object] = None

    # --- 공개 API ---

    def run(self) -> None:
        """TUI 실행 (동기 래퍼)"""
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        """TUI 비동기 실행"""
        # 테마 설정
        from beantui.themes import set_theme

        set_theme(self._config.theme.default)

        # builtin 커맨드 로드
        if self._config.commands.builtin:
            import beantui.commands.builtin  # noqa: F401 — registers commands

        # 플러그인 커맨드 로드
        if self._config.commands.plugins:
            from beantui.commands import CommandRegistry

            CommandRegistry.load_plugins(self._config.commands.plugins)

        await self._main_loop()

    # --- 내부 ---

    def _install_log_filter(self) -> None:
        root = logging.getLogger()
        if self._log_filter not in root.filters:
            root.addFilter(self._log_filter)
        for h in root.handlers:
            if self._log_filter not in h.filters:
                h.addFilter(self._log_filter)

    def _set_verbose(self, on: bool) -> None:
        self._log_filter.verbose = on
        self._install_log_filter()
        if self._stdout_guard:
            self._stdout_guard.verbose = on

    def _install_stdout_guard(self) -> None:
        if isinstance(sys.stdout, _StdoutGuard):
            return
        self._stdout_guard = _StdoutGuard(sys.stdout)
        sys.stdout = self._stdout_guard  # type: ignore[assignment]

    def _get_console(self):  # type: ignore[no-untyped-def]
        try:
            from rich.console import Console

            return Console(stderr=True)
        except ImportError:
            return None

    async def _get_default_model(self) -> str:
        if self._backend:
            try:
                return await self._backend.get_default_model()
            except Exception:
                pass
        return self._config.app.default_model

    def _toggle_shell_mode(self) -> bool:
        self._shell_mode["active"] = not self._shell_mode["active"]
        return self._shell_mode["active"]

    def _build_prompt_message(self, session: ChatSession, is_shell: bool):  # type: ignore[no-untyped-def]
        from prompt_toolkit.formatted_text import HTML

        from beantui.themes import get_theme

        theme = get_theme()
        if is_shell:
            color = theme.palette.shell_prompt
            icon = theme.icons["prompt_shell"]
        else:
            color = theme.get_prompt_color(session.mode)
            icon = theme.get_prompt_icon(session.mode)
        return HTML(f"<style fg='{color}'><b>{icon} </b></style>")

    def _create_prompt_session(
        self,
        session_ref: Optional[ChatSession] = None,
    ):  # type: ignore[no-untyped-def]
        from prompt_toolkit import PromptSession
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from prompt_toolkit.styles import Style

        from beantui.completers import create_completer
        from beantui.session_store import SessionStore
        from beantui.themes import get_theme

        theme = get_theme()

        history = InMemoryHistory()
        try:
            store = SessionStore(db_path=self._config.resolved_db_path)
            for item in store.get_history(limit=self._config.session.history_limit):
                history.append_string(item)
            store.close()
        except Exception:
            pass

        bindings = KeyBindings()
        _last_esc_time = {"value": 0.0}

        @bindings.add(Keys.Escape)
        def _handle_esc(event) -> None:  # type: ignore[no-untyped-def]
            now = time.monotonic()
            if now - _last_esc_time["value"] < 0.5:
                event.current_buffer.reset()
                _last_esc_time["value"] = 0.0
            else:
                _last_esc_time["value"] = now

        @bindings.add(Keys.Escape, Keys.Enter)
        def _meta_enter(event) -> None:  # type: ignore[no-untyped-def]
            event.current_buffer.insert_text("\n")
            _last_esc_time["value"] = 0.0

        completer = create_completer(self._working_dir)
        pt_style = Style.from_dict(theme.prompt_toolkit_style)

        app_name = self._config.app.name
        _toolbar_cache: dict[str, object] = {"last_msg_count": 0, "total_chars": 0}

        def _bottom_toolbar() -> HTML:
            if session_ref is None:
                return HTML(f"<b>{app_name}</b>")

            t = get_theme()
            p = t.palette
            msg_count = len(session_ref.messages)
            if msg_count != _toolbar_cache["last_msg_count"]:
                _toolbar_cache["total_chars"] = sum(len(m.content) for m in session_ref.messages)
                _toolbar_cache["last_msg_count"] = msg_count
            total_chars = _toolbar_cache["total_chars"]
            role_name = getattr(session_ref, "role_name", "default")
            is_shell = self._shell_mode.get("active", False)
            shell_indicator = " <ansiyellow>$ shell</ansiyellow>" if is_shell else ""

            parts = [
                f"<b>{t.icons['model']} {session_ref.model}</b>",
                f"<style fg='{p.brand_accent}'>{role_name}</style>",
                f"{session_ref.mode}",
                f"{msg_count} msgs",
                f"~{total_chars:,} chars",
            ]
            bar = " · ".join(parts)
            return HTML(f" {bar}{shell_indicator} ")

        pt_session = PromptSession(
            completer=completer,
            history=history,
            enable_history_search=True,
            complete_while_typing=True,
            key_bindings=bindings,
            multiline=False,
            bottom_toolbar=_bottom_toolbar,
            style=pt_style,
        )
        return pt_session, history

    def _save_input_history(self, content: str) -> None:
        try:
            from beantui.session_store import SessionStore

            store = SessionStore(db_path=self._config.resolved_db_path)
            store.append_history(content)
            store.close()
        except Exception:
            pass

    def _auto_save_session(self, session: ChatSession) -> None:
        if not session.messages:
            return
        try:
            from beantui.session_store import SessionStore

            store = SessionStore(db_path=self._config.resolved_db_path)
            store.save_session(session)
            store.close()
        except Exception:
            pass

    async def _run_chat(self, session: ChatSession, console) -> tuple[Optional[str], float]:  # type: ignore[no-untyped-def]
        """LLM 스트리밍 채팅 — ChatBackend Protocol 사용"""
        if self._backend is None:
            if console:
                from beantui.layout import render_error

                render_error(console, "No ChatBackend configured")
            return None, 0.0

        messages = session.build_messages_for_llm()
        if not messages:
            return None, 0.0

        start_time = time.monotonic()
        try:
            self._install_log_filter()
            stream = self._backend.stream_chat(
                messages=messages,
                system=session.system,
                model=session.model,
                temperature=self._config.app.temperature,
            )
            self._install_log_filter()

            chunks: list[str] = []

            if console:
                from rich.live import Live
                from rich.markdown import Markdown
                from rich.padding import Padding

                accumulated = ""
                with Live(
                    Padding(Markdown(""), (0, 0, 0, 2)),
                    console=console,
                    refresh_per_second=8,
                    vertical_overflow="visible",
                ) as live:
                    async for chunk in stream:
                        chunks.append(chunk)
                        accumulated += chunk
                        try:
                            md = Markdown(accumulated, code_theme="monokai")
                            live.update(Padding(md, (0, 0, 0, 2)))
                        except Exception:
                            from rich.text import Text

                            live.update(Padding(Text(accumulated), (0, 0, 0, 2)))
            else:
                async for chunk in stream:
                    chunks.append(chunk)
                    print(chunk, end="", flush=True)
                print()

            raw = "".join(chunks)
            elapsed = time.monotonic() - start_time
            return _clean_response(raw), elapsed

        except Exception as e:
            elapsed = time.monotonic() - start_time
            msg = self._backend.sanitize_error(e) if self._backend else str(e)
            if console:
                from beantui.layout import render_error

                render_error(console, str(msg), suggestion="Check your API key or model name")
            else:
                print(f"Error: {msg}")
            return None, elapsed

    async def _run_rag_query(
        self,
        session: ChatSession,
        question: str,
        console,  # type: ignore[no-untyped-def]
    ) -> tuple[Optional[str], float]:
        start_time = time.monotonic()
        if not session._rag.get("chain"):
            if console:
                from beantui.layout import render_warning

                render_warning(console, "Load RAG first: /rag <path>")
            return None, 0.0

        try:
            rag = session._rag["chain"]
            result = await rag.aquery(question)
            raw = (result[0] if isinstance(result, tuple) else result) or ""
            elapsed = time.monotonic() - start_time
            return _clean_response(raw), elapsed
        except Exception as e:
            elapsed = time.monotonic() - start_time
            msg = self._backend.sanitize_error(e) if self._backend else str(e)
            if console:
                from beantui.layout import render_error

                render_error(console, f"RAG error: {msg}")
            else:
                print(f"RAG error: {msg}")
            return None, elapsed

    # -----------------------------------------------------------------------
    # 메인 루프
    # -----------------------------------------------------------------------

    async def _main_loop(self) -> None:
        self._install_log_filter()
        console = self._get_console()
        self._console = console

        default_model = await self._get_default_model()
        session = ChatSession(model=default_model, working_dir=self._working_dir)
        session.role_name = "default"

        if console:
            from beantui.layout import render_frame

            render_frame(console, session, config=self._config)
        else:
            print(f"\n{self._config.app.name} · Type /help | @file | !cmd | /exit\n")

        # prompt_toolkit 세션 — stdout guard 설치 전에
        use_pt = False
        prompt_session = None
        try:
            prompt_session, _pt_history = self._create_prompt_session(session_ref=session)
            use_pt = True
            if console:
                from beantui.layout import render_info

                render_info(console, "prompt_toolkit active (Tab: autocomplete, Ctrl+R: history)")
                console.print()
        except Exception as _pt_err:
            if console:
                from beantui.layout import render_warning

                render_warning(console, f"prompt_toolkit unavailable: {_pt_err}")

        self._install_stdout_guard()

        while True:
            try:
                # --- 입력 ---
                if use_pt and prompt_session:
                    prompt_msg = self._build_prompt_message(session, self._shell_mode["active"])
                    user_input = await prompt_session.prompt_async(message=prompt_msg)
                elif console:
                    from beantui.layout import render_prompt

                    user_input = render_prompt(console)
                else:
                    user_input = input("❯ ")

                if not user_input or not user_input.strip():
                    continue

                self._save_input_history(user_input.strip())
                stripped = _strip_ansi_escapes(user_input)

                # --- 셸 모드 처리 ---
                if self._shell_mode["active"] and not stripped.startswith("/"):
                    from beantui.input_parser import run_shell_command

                    code, stdout, stderr = run_shell_command(
                        stripped, timeout=self._config.session.shell_timeout
                    )
                    output = stdout or stderr or "(no output)"
                    if console:
                        if code == 0:
                            console.print(f"[dim]{output}[/dim]")
                        else:
                            from beantui.layout import render_error

                            render_error(console, output)
                    else:
                        print(output)
                    continue

                # --- 셸 모드 토글 ---
                if stripped == "!":
                    is_on = self._toggle_shell_mode()
                    if console:
                        from beantui.layout import render_info, render_success

                        if is_on:
                            render_success(console, "Shell mode ON")
                        else:
                            render_info(console, "Shell mode OFF")
                    else:
                        print(f"Shell mode {'ON' if is_on else 'OFF'}")
                    continue

                # --- 파싱 ---
                slash_cmd, slash_args, message = parse_user_input(user_input, session)

                if stripped == "/":
                    from beantui.commands import CommandRegistry

                    cmds = CommandRegistry.visible_commands()
                    lines = ["\n  [bold cyan]Commands[/bold cyan]\n"]
                    for name, entry in cmds.items():
                        usage = f" [dim]{entry.usage}[/dim]" if entry.usage else ""
                        lines.append(
                            f"  [green]/{name}[/green]{' ' * max(1, 14 - len(name))}{entry.description}{usage}"
                        )
                    summary = "\n".join(lines)
                    if console:
                        console.print(summary)
                    else:
                        print(summary)
                    continue

                if slash_cmd == "":
                    continue

                # --- 슬래시 커맨드 ---
                if slash_cmd:
                    from beantui.commands import CommandRegistry, execute_command

                    if not CommandRegistry.has(slash_cmd):
                        message = stripped.lstrip("/").strip() or stripped
                        slash_cmd = None
                    else:
                        # verbose 연동
                        should_exit, result = await execute_command(
                            slash_cmd, slash_args or "", session
                        )

                        # verbose 커맨드 후 엔진 상태 동기화
                        if slash_cmd in ("verbose", "logs"):
                            self._set_verbose(session.verbose)

                        if should_exit:
                            self._auto_save_session(session)
                            if console:
                                from beantui.layout import render_info

                                render_info(console, "Goodbye!")
                                console.print()
                            else:
                                print("\nGoodbye!\n")
                            return

                        if result and console:
                            console.print(result)
                        elif result:
                            print(result)

                        if slash_cmd in ("new", "clear") and console:
                            from beantui.layout import render_frame

                            render_frame(
                                console,
                                session,
                                config=self._config,
                                show_history=False,
                                show_logo=False,
                            )

                        if slash_cmd == "load" and console and session.messages:
                            from beantui.layout import render_history_block

                            console.print()
                            for msg in session.messages[-6:]:
                                render_history_block(console, msg.role, msg.content)

                        continue

                if not message:
                    continue

                # --- 파일 참조 ---
                cleaned_msg, file_refs = resolve_file_references(message, session.working_dir)
                for path, content in file_refs:
                    session.attach_file(path, content)
                    if console:
                        from beantui.layout import render_file_attached

                        render_file_attached(console, path)

                # --- 사용자 메시지 헤더 ---
                if console:
                    from beantui.layout import (
                        render_assistant_header,
                        render_context_indicator,
                        render_response_meta,
                        render_separator,
                        render_thinking,
                        render_user_header,
                    )

                    render_separator(console)
                    render_user_header(console)

                # --- 컨텍스트 인디케이터 ---
                if console and (session.attached_contexts or session._rag.get("chain")):
                    render_context_indicator(
                        console,
                        files=len(session.attached_contexts),
                        rag_active=bool(session._rag.get("chain")),
                        token_count=sum(len(m.content) for m in session.messages),
                    )

                # --- RAG 모드 ---
                if session._rag.get("chain"):
                    session.add_user_message(cleaned_msg)
                    if console:
                        render_thinking(console, "searching")
                    answer, elapsed = await self._run_rag_query(session, cleaned_msg, console)
                    if answer:
                        session.add_assistant_message(answer)
                        if console:
                            console.print()
                            render_assistant_header(console, session.model)
                            from beantui.layout import render_ai_markdown

                            render_ai_markdown(console, answer)
                            render_response_meta(
                                console, elapsed_seconds=elapsed, char_count=len(answer)
                            )
                        else:
                            print(answer)
                    session.attached_contexts.clear()
                    continue

                # --- 일반 채팅 ---
                session.add_user_message(cleaned_msg)
                if console:
                    console.print()
                    render_assistant_header(console, session.model)
                    render_thinking(console, "thinking")

                response, elapsed = await self._run_chat(session, console)
                if response:
                    session.add_assistant_message(response)
                    if console:
                        render_response_meta(
                            console, elapsed_seconds=elapsed, char_count=len(response)
                        )
                        console.print()

                session.attached_contexts.clear()

            except KeyboardInterrupt:
                if self._shell_mode["active"]:
                    self._shell_mode["active"] = False
                    if console:
                        from beantui.layout import render_info

                        render_info(console, "Shell mode OFF")
                    continue
                if console:
                    from beantui.layout import render_warning

                    render_warning(console, "Ctrl+C — /exit to quit")
                else:
                    print("\nCtrl+C — /exit to quit")
            except EOFError:
                self._auto_save_session(session)
                if console:
                    from beantui.layout import render_info

                    console.print()
                    render_info(console, "Goodbye!")
                    console.print()
                else:
                    print("\nGoodbye!\n")
                return
