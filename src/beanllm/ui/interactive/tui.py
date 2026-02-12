"""
Interactive TUI - Gemini CLI / aichat / OpenCode 스타일 프리미엄

핵심:
  - 테마 시스템 (dark/light 팔레트)
  - 그라디언트 로고 + 환경 정보 웰컴
  - prompt_toolkit 기반 입력 (자동완성, 멀티라인, 히스토리)
  - 모드별 프롬프트 색상 + 아이콘
  - 실시간 마크다운 스트리밍 (Rich Live)
  - 메시지 구분선 (타임스탬프, 아이콘)
  - 단계별 Thinking 표시
  - 스타일된 에러/경고 패널
  - 컨텍스트 인디케이터 (첨부파일, RAG, 토큰)
  - 응답 메타 (시간, 길이)
  - 세션 저장/복원 · Role 프리셋 · 로그 차단
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

from beanllm.ui.interactive.input_parser import (
    _strip_ansi_escapes,
    parse_user_input,
    resolve_file_references,
)
from beanllm.ui.interactive.session import ChatSession
from beanllm.ui.interactive.slash_commands import execute_slash

# ---------------------------------------------------------------------------
# 로그 완전 차단
# ---------------------------------------------------------------------------


class _TUILogFilter(logging.Filter):
    """TUI 모드에서 beanllm/httpx 등 내부 로그 차단"""

    verbose: bool = False
    _blocked = ("beanllm", "httpx", "httpcore", "urllib3", "llm_model_manager")

    def filter(self, record: logging.LogRecord) -> bool:
        if self.verbose:
            return True
        return not any(record.name.startswith(p) for p in self._blocked)


_log_filter = _TUILogFilter()


def _install_log_filter() -> None:
    """모든 로거 + 모든 핸들러에 필터 설치"""
    root = logging.getLogger()
    if _log_filter not in root.filters:
        root.addFilter(_log_filter)
    for h in root.handlers:
        if _log_filter not in h.filters:
            h.addFilter(_log_filter)
    for _, logger_obj in logging.Logger.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            for h in logger_obj.handlers:
                if _log_filter not in h.filters:
                    h.addFilter(_log_filter)


def _set_verbose(on: bool) -> None:
    _log_filter.verbose = on
    _install_log_filter()
    if _stdout_guard:
        _stdout_guard.verbose = on


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


_stdout_guard: Optional[_StdoutGuard] = None


def _install_stdout_guard() -> None:
    global _stdout_guard
    if isinstance(sys.stdout, _StdoutGuard):
        return
    _stdout_guard = _StdoutGuard(sys.stdout)
    sys.stdout = _stdout_guard  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 콘솔
# ---------------------------------------------------------------------------


def _get_console():  # type: ignore[no-untyped-def]
    try:
        from rich.console import Console

        return Console(stderr=True)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# prompt_toolkit 입력 시스템
# ---------------------------------------------------------------------------


def _create_prompt_session(
    working_dir: Path,
    session_ref: Optional[ChatSession] = None,
    shell_mode_ref: Optional[dict] = None,
):  # type: ignore[no-untyped-def]
    """prompt_toolkit PromptSession 생성 — 테마 기반"""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.styles import Style

    from beanllm.ui.interactive.completers import create_completer
    from beanllm.ui.interactive.session_store import SessionStore
    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()

    # 히스토리: SQLite에서 복원
    history = InMemoryHistory()
    try:
        store = SessionStore()
        for item in store.get_history(limit=200):
            history.append_string(item)
        store.close()
    except Exception:
        pass

    # 키바인딩
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

    completer = create_completer(working_dir)

    # prompt_toolkit 스타일 (테마에서 가져옴)
    pt_style = Style.from_dict(theme.prompt_toolkit_style)

    # 상태바 콜백
    def _bottom_toolbar() -> HTML:
        if session_ref is None:
            return HTML("<b>beanllm</b>")

        t = get_theme()
        p = t.palette
        role_name = getattr(session_ref, "role_name", "default")
        msg_count = len(session_ref.messages)
        total_chars = sum(len(m.content) for m in session_ref.messages)
        mode = session_ref.mode
        model = session_ref.model
        is_shell = shell_mode_ref.get("active", False) if shell_mode_ref else False

        shell_indicator = " <ansiyellow>$ shell</ansiyellow>" if is_shell else ""

        parts = [
            f"<b>{t.icons['model']} {model}</b>",
            f"<style fg='{p.brand_accent}'>{role_name}</style>",
            f"{mode}",
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


def _save_input_history(content: str) -> None:
    """사용자 입력을 SQLite 히스토리에 저장"""
    try:
        from beanllm.ui.interactive.session_store import SessionStore

        store = SessionStore()
        store.append_history(content)
        store.close()
    except Exception:
        pass


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


def _clean_response(text: str) -> str:
    for pat in _ARTIFACT_PATTERNS:
        m = pat.search(text)
        if m:
            text = text[: m.start()]
    text = _remove_duplicate_blocks(text)
    text = re.sub(r"\n\s*---\s*$", "", text)
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
# 기본 모델
# ---------------------------------------------------------------------------


def _get_default_model() -> str:
    try:
        from beanllm.infrastructure.registry import get_model_registry

        registry = get_model_registry()
        models = registry.get_available_models()
        active = [
            m for m in models if m.provider in [p.name for p in registry.get_active_providers()]
        ]
        if active:
            return active[0].model_name
    except Exception:
        pass
    return "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Chat / RAG 실행 (마크다운 스트리밍)
# ---------------------------------------------------------------------------


async def _run_chat(session: ChatSession, console) -> tuple[Optional[str], float]:  # type: ignore[no-untyped-def]
    """LLM 스트리밍 채팅 — 실시간 마크다운 렌더링"""
    messages = session.build_messages_for_llm()
    if not messages:
        return None, 0.0

    start_time = time.monotonic()

    try:
        from beanllm import Client

        _install_log_filter()

        client = Client(model=session.model, provider=session.provider)
        stream = client.stream_chat(
            messages=messages,
            system=session.system,
            temperature=0.7,
        )

        _install_log_filter()

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
        try:
            from beanllm.utils.integration.security import sanitize_error_message

            msg = sanitize_error_message(e)
        except ImportError:
            msg = str(e)

        if console:
            from beanllm.ui.interactive.layout import render_error

            render_error(console, str(msg), suggestion="Check your API key or model name")
        else:
            print(f"Error: {msg}")
        return None, elapsed


async def _run_rag_query(
    session: ChatSession,
    question: str,
    console,  # type: ignore[no-untyped-def]
) -> tuple[Optional[str], float]:
    start_time = time.monotonic()

    if not session._rag.get("chain"):
        if console:
            from beanllm.ui.interactive.layout import render_warning

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
        try:
            from beanllm.utils.integration.security import sanitize_error_message

            msg = sanitize_error_message(e)
        except ImportError:
            msg = str(e)
        if console:
            from beanllm.ui.interactive.layout import render_error

            render_error(console, f"RAG error: {msg}")
        else:
            print(f"RAG error: {msg}")
        return None, elapsed


# ---------------------------------------------------------------------------
# ! 셸 모드 토글
# ---------------------------------------------------------------------------

_shell_mode: dict[str, bool] = {"active": False}


def _toggle_shell_mode() -> bool:
    """셸 모드 토글"""
    _shell_mode["active"] = not _shell_mode["active"]
    return _shell_mode["active"]


# ---------------------------------------------------------------------------
# 프롬프트 메시지 생성
# ---------------------------------------------------------------------------


def _build_prompt_message(session: ChatSession, is_shell: bool):  # type: ignore[no-untyped-def]
    """모드별 프롬프트 스타일"""
    from prompt_toolkit.formatted_text import HTML

    from beanllm.ui.interactive.themes import get_theme

    theme = get_theme()

    if is_shell:
        color = theme.palette.shell_prompt
        icon = theme.icons["prompt_shell"]
    else:
        color = theme.get_prompt_color(session.mode)
        icon = theme.get_prompt_icon(session.mode)

    return HTML(f"<style fg='{color}'><b>{icon} </b></style>")


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------


async def run_interactive_tui(working_dir: Optional[Path] = None) -> None:
    """prompt_toolkit + Rich 기반 프리미엄 TUI 메인 루프"""

    # 1. 로그 필터 설치
    _install_log_filter()

    console = _get_console()
    work_dir = working_dir or Path.cwd()

    session = ChatSession(
        model=_get_default_model(),
        working_dir=work_dir,
    )
    session.role_name = "default"

    # 2. 초기 화면
    if console:
        from beanllm.ui.interactive.layout import render_frame

        render_frame(console, session)
    else:
        print("\nbeanllm · Type /help | @file | !cmd | /exit\n")

    # 3. prompt_toolkit 세션 생성 — stdout guard 설치 전에!
    try:
        prompt_session, pt_history = _create_prompt_session(
            work_dir, session_ref=session, shell_mode_ref=_shell_mode
        )
        use_pt = True
        if console:
            from beanllm.ui.interactive.layout import render_info

            render_info(console, "prompt_toolkit active (Tab: autocomplete, Ctrl+R: history)")
            console.print()
    except Exception as _pt_err:
        prompt_session = None
        pt_history = None
        use_pt = False
        if console:
            from beanllm.ui.interactive.layout import render_warning

            render_warning(console, f"prompt_toolkit unavailable: {_pt_err}")
            from beanllm.ui.interactive.layout import render_info

            render_info(console, "pip install prompt-toolkit  to enable autocomplete")
            console.print()

    # 4. prompt_toolkit이 stdout 참조를 확보한 뒤에 guard 설치
    _install_stdout_guard()

    while True:
        try:
            # --- 입력 ---
            if use_pt and prompt_session:
                prompt_msg = _build_prompt_message(session, _shell_mode["active"])
                user_input = await prompt_session.prompt_async(message=prompt_msg)
            elif console:
                from beanllm.ui.interactive.layout import render_prompt

                user_input = render_prompt(console)
            else:
                user_input = input("❯ ")

            if not user_input or not user_input.strip():
                continue

            # 히스토리 저장
            _save_input_history(user_input.strip())

            # --- 셸 모드 처리 ---
            stripped = _strip_ansi_escapes(user_input)

            if _shell_mode["active"] and not stripped.startswith("/"):
                from beanllm.ui.interactive.input_parser import run_shell_command

                code, stdout, stderr = run_shell_command(stripped)
                output = stdout or stderr or "(no output)"
                if console:
                    if code == 0:
                        console.print(f"[dim]{output}[/dim]")
                    else:
                        from beanllm.ui.interactive.layout import render_error

                        render_error(console, output)
                else:
                    print(output)
                continue

            # --- 셸 모드 토글 ---
            if stripped == "!":
                is_on = _toggle_shell_mode()
                if console:
                    from beanllm.ui.interactive.layout import render_info, render_success

                    if is_on:
                        render_success(console, "Shell mode ON")
                    else:
                        render_info(console, "Shell mode OFF")
                else:
                    print(f"Shell mode {'ON' if is_on else 'OFF'}")
                continue

            # --- 파싱 ---
            slash_cmd, slash_args, message = parse_user_input(user_input, session)

            # "/" 만 입력 → 커맨드 요약
            if stripped == "/":
                from beanllm.ui.interactive.slash_commands import get_slash_command_summary

                if console:
                    console.print(get_slash_command_summary())
                else:
                    print(get_slash_command_summary().strip())
                continue

            if slash_cmd == "":
                continue

            # --- 슬래시 커맨드 ---
            if slash_cmd:
                from beanllm.ui.interactive.slash_commands import SLASH_COMMANDS

                if slash_cmd not in SLASH_COMMANDS:
                    message = stripped.lstrip("/").strip() or stripped
                    slash_cmd = None
                else:
                    should_exit, result = await execute_slash(slash_cmd, slash_args or "", session)

                    if should_exit:
                        _auto_save_session(session)
                        if console:
                            from beanllm.ui.interactive.layout import render_info

                            render_info(console, "Goodbye! 👋")
                            console.print()
                        else:
                            print("\nGoodbye!\n")
                        return

                    if result and console:
                        console.print(result)
                    elif result:
                        print(result)

                    if slash_cmd in ("new", "clear") and console:
                        from beanllm.ui.interactive.layout import render_frame

                        render_frame(console, session, show_history=False, show_logo=False)

                    if slash_cmd == "load" and console and session.messages:
                        from beanllm.ui.interactive.layout import render_history_block

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
                    from beanllm.ui.interactive.layout import render_file_attached

                    render_file_attached(console, path)

            # --- 사용자 메시지 헤더 ---
            if console:
                from beanllm.ui.interactive.layout import (
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

                answer, elapsed = await _run_rag_query(session, cleaned_msg, console)
                if answer:
                    session.add_assistant_message(answer)
                    if console:
                        console.print()
                        render_assistant_header(console, session.model)
                        from beanllm.ui.interactive.layout import render_ai_markdown

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

            response, elapsed = await _run_chat(session, console)
            if response:
                session.add_assistant_message(response)
                if console:
                    render_response_meta(console, elapsed_seconds=elapsed, char_count=len(response))
                    console.print()

            session.attached_contexts.clear()

        except KeyboardInterrupt:
            if _shell_mode["active"]:
                _shell_mode["active"] = False
                if console:
                    from beanllm.ui.interactive.layout import render_info

                    render_info(console, "Shell mode OFF")
                continue
            if console:
                from beanllm.ui.interactive.layout import render_warning

                render_warning(console, "Ctrl+C — /exit to quit")
            else:
                print("\nCtrl+C — /exit to quit")
        except EOFError:
            _auto_save_session(session)
            if console:
                from beanllm.ui.interactive.layout import render_info

                console.print()
                render_info(console, "Goodbye! 👋")
                console.print()
            else:
                print("\nGoodbye!\n")
            return


# ---------------------------------------------------------------------------
# 자동 저장
# ---------------------------------------------------------------------------


def _auto_save_session(session: ChatSession) -> None:
    """종료 시 세션 자동 저장"""
    if not session.messages:
        return
    try:
        from beanllm.ui.interactive.session_store import SessionStore

        store = SessionStore()
        store.save_session(session)
        store.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=str, default=None, help="Working directory")
    parser.add_argument("--model", type=str, default=None, help="Default model")
    parser.add_argument("--theme", type=str, default="dark", help="Theme (dark/light)")
    args, _ = parser.parse_known_args()

    # 테마 설정
    from beanllm.ui.interactive.themes import set_theme

    set_theme(args.theme)

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    asyncio.run(run_interactive_tui(working_dir=cwd))


if __name__ == "__main__":
    main()
