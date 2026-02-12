"""
Completers - 슬래시 커맨드 + @file 경로 자동완성

Gemini CLI / aichat 스타일:
  /mod → /model 자동완성 (실시간 필터링)
  @src/ → 파일 경로 Tab 완성
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    merge_completers,
)
from prompt_toolkit.document import Document

# ---------------------------------------------------------------------------
# 슬래시 커맨드 자동완성
# ---------------------------------------------------------------------------

# 커맨드 메타: (설명, 인자 필요 여부)
COMMAND_META: Dict[str, tuple[str, bool]] = {
    "help": ("도움말", False),
    "model": ("모델 변경", True),
    "new": ("새 세션", False),
    "clear": ("세션 초기화", False),
    "plan": ("플랜 모드", False),
    "chat": ("채팅 모드", False),
    "rag": ("RAG 검색", True),
    "search": ("웹 검색", True),
    "agent": ("에이전트 실행", True),
    "eval": ("평가", True),
    "verbose": ("로그 토글", False),
    "role": ("역할 변경", True),
    "theme": ("테마 전환", True),
    "save": ("세션 저장", False),
    "load": ("세션 불러오기", True),
    "exit": ("종료", False),
}


class SlashCommandCompleter(Completer):
    """/ 입력 시 실시간 커맨드 자동완성"""

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()

        if not text.startswith("/"):
            return

        # "/" 뒤의 부분 매칭
        query = text[1:].lower()

        for cmd, (desc, _) in COMMAND_META.items():
            if cmd.startswith(query):
                yield Completion(
                    text=cmd,
                    start_position=-len(query),
                    display=f"/{cmd}",
                    display_meta=desc,
                )


# ---------------------------------------------------------------------------
# @file 경로 자동완성
# ---------------------------------------------------------------------------


class FilePathCompleter(Completer):
    """@path 입력 시 파일 경로 자동완성"""

    def __init__(self, working_dir: Optional[Path] = None) -> None:
        self._working_dir = working_dir or Path.cwd()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor

        # @ 뒤의 경로 부분 찾기
        at_idx = text.rfind("@")
        if at_idx == -1:
            return

        # @ 바로 앞이 공백이거나 시작이어야
        if at_idx > 0 and text[at_idx - 1] not in (" ", "\t", "\n"):
            return

        partial = text[at_idx + 1 :]
        if " " in partial:
            return

        base_dir = self._working_dir / Path(partial).parent if "/" in partial else self._working_dir
        prefix = Path(partial).name if "/" in partial else partial
        parent_prefix = str(Path(partial).parent) + "/" if "/" in partial else ""

        try:
            if not base_dir.exists():
                return
            for entry in sorted(base_dir.iterdir()):
                name = entry.name
                if name.startswith("."):
                    continue
                if name.lower().startswith(prefix.lower()):
                    display_name = parent_prefix + name
                    suffix = "/" if entry.is_dir() else ""
                    yield Completion(
                        text=display_name + suffix,
                        start_position=-len(partial),
                        display=name + suffix,
                        display_meta="dir" if entry.is_dir() else _file_size(entry),
                    )
        except PermissionError:
            return


def _file_size(p: Path) -> str:
    """파일 크기 표시"""
    try:
        size = p.stat().st_size
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size // 1024}KB"
        else:
            return f"{size // (1024 * 1024)}MB"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 통합 Completer
# ---------------------------------------------------------------------------


def create_completer(working_dir: Optional[Path] = None) -> Completer:
    """슬래시 커맨드 + @file 경로 통합 Completer"""
    return merge_completers(
        [
            SlashCommandCompleter(),
            FilePathCompleter(working_dir=working_dir),
        ]
    )
