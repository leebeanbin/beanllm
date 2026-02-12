"""
Completers - 슬래시 커맨드 + @file 경로 자동완성

CommandRegistry에서 자동으로 커맨드 목록을 가져와 완성합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    merge_completers,
)
from prompt_toolkit.document import Document


class SlashCommandCompleter(Completer):
    """/ 입력 시 실시간 커맨드 자동완성

    CommandRegistry에서 메타 정보를 가져옵니다.
    커스텀 메타를 override할 수도 있습니다.
    """

    def __init__(self, custom_meta: Optional[Dict[str, Tuple[str, bool]]] = None) -> None:
        self._custom_meta = custom_meta

    def _get_meta(self) -> Dict[str, Tuple[str, bool]]:
        if self._custom_meta is not None:
            return self._custom_meta
        from beantui.commands import CommandRegistry

        return CommandRegistry.command_meta()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()

        if not text.startswith("/"):
            return

        query = text[1:].lower()
        meta = self._get_meta()

        for cmd, (desc, _) in meta.items():
            if cmd.startswith(query):
                yield Completion(
                    text=cmd,
                    start_position=-len(query),
                    display=f"/{cmd}",
                    display_meta=desc,
                )


class FilePathCompleter(Completer):
    """@path 입력 시 파일 경로 자동완성"""

    def __init__(self, working_dir: Optional[Path] = None) -> None:
        self._working_dir = working_dir or Path.cwd()

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        at_idx = text.rfind("@")
        if at_idx == -1:
            return

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


def create_completer(working_dir: Optional[Path] = None) -> Completer:
    """슬래시 커맨드 + @file 경로 통합 Completer"""
    return merge_completers(
        [
            SlashCommandCompleter(),
            FilePathCompleter(working_dir=working_dir),
        ]
    )
