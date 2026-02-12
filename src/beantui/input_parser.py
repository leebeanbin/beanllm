"""
Input Parser - @file, !shell, /command 파싱
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _strip_ansi_escapes(s: str) -> str:
    """터미널 ESC 문자 제거 — /exit 인식 실패 방지"""
    s = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", s)
    s = s.replace("\x1b", "")
    return s.strip()


def parse_user_input(raw: str, session: Any = None) -> Tuple[Optional[str], Optional[str], str]:
    """사용자 입력 파싱

    Returns:
        (slash_cmd, slash_args, message)
    """
    stripped = _strip_ansi_escapes(raw)
    if not stripped:
        return None, None, ""

    lower = stripped.lower()
    if lower in ("exit", "q", "quit"):
        return "exit", "", ""

    if stripped.startswith("/"):
        rest = stripped[1:].strip()
        if not rest:
            return "", "", ""
        parts = rest.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        return cmd, args, ""

    if stripped.startswith("!"):
        return "shell", stripped[1:].strip(), ""

    return None, None, stripped


def resolve_file_references(message: str, working_dir: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """@path 패턴에서 파일 내용 추출"""
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
    """파일 참조를 절대 경로로 변환"""
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


def run_shell_command(
    cmd: str, timeout: int = 60, cwd: Optional[Path] = None
) -> Tuple[int, str, str]:
    """셸 명령 실행

    Args:
        cmd: 실행할 명령어
        timeout: 타임아웃 (초)
        cwd: 작업 디렉토리

    Returns:
        (returncode, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or Path.cwd(),
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out ({timeout}s)"
    except Exception as e:
        return -1, "", str(e)
