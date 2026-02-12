"""
Input Parser - @file, !shell, /command 파싱
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from beanllm.ui.interactive.session import ChatSession


def _strip_ansi_escapes(s: str) -> str:
    """터미널 ESC 문자(\x1b) 등 제거 - /exit 인식 실패 방지"""
    # CSI 시퀀스: ESC [ ... [letter]
    s = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)
    # 단일 ESC (Esc 키만 눌렀을 때)
    s = s.replace("\x1b", "")
    return s.strip()


def parse_user_input(raw: str, session: ChatSession) -> Tuple[Optional[str], Optional[str], str]:
    """
    사용자 입력 파싱

    Returns:
        (slash_cmd, slash_args, message)
    """
    stripped = _strip_ansi_escapes(raw)
    if not stripped:
        return None, None, ""

    # "exit", "q", "quit" (슬래시 없이) → /exit로 처리 (터미널 오입력/단축어)
    lower = stripped.lower()
    if lower in ("exit", "q", "quit"):
        return "exit", "", ""

    if stripped.startswith("/"):
        rest = stripped[1:].strip()
        if not rest:
            return "", "", ""  # "/" 만 입력 시
        parts = rest.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return cmd, args, ""

    if stripped.startswith("!"):
        return "shell", stripped[1:].strip(), ""

    return None, None, stripped


def resolve_file_references(message: str, working_dir: Path) -> Tuple[str, List[Tuple[str, str]]]:
    pattern = r"@([^\s]+)"
    matches = re.findall(pattern, message)

    if not matches:
        return message, []

    resolved: List[Tuple[str, str]] = []
    for ref in matches:
        path = resolve_file_path(ref, working_dir)
        if path and path.exists():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                resolved.append((str(path), content))
            except Exception:
                pass

    return message, resolved


def resolve_file_path(ref: str, working_dir: Path) -> Optional[Path]:
    ref = ref.strip()
    if not ref:
        return None

    p = Path(ref)
    if p.is_absolute():
        return p if p.exists() else None

    candidate = working_dir / ref
    if candidate.exists():
        return candidate

    for path in working_dir.rglob(ref.split("/")[-1]):
        if ref in str(path) or str(path).endswith(ref):
            return path

    return None


def run_shell_command(cmd: str) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path.cwd(),
        )
        return (
            result.returncode,
            result.stdout or "",
            result.stderr or "",
        )
    except subprocess.TimeoutExpired:
        return (-1, "", "Command timed out (60s)")
    except Exception as e:
        return (-1, "", str(e))
