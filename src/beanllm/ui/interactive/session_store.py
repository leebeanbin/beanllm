"""
Session Store - SQLite 기반 세션 저장/복원

OpenCode 스타일:
  /save → 현재 세션 SQLite 저장
  /load [id] → 과거 세션 복원
  자동 저장 (세션 종료 시)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from beanllm.ui.interactive.session import ChatMessage, ChatSession

# ---------------------------------------------------------------------------
# 기본 DB 경로
# ---------------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".beanllm"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "sessions.db"


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class SessionStore:
    """SQLite 기반 세션 영속 저장소"""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_table()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                model       TEXT NOT NULL,
                provider    TEXT,
                system      TEXT,
                mode        TEXT DEFAULT 'chat',
                messages    TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS input_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # 세션 CRUD
    # ------------------------------------------------------------------

    def save_session(self, session: ChatSession, title: Optional[str] = None) -> int:
        """현재 세션 저장, session_id 반환"""
        if not session.messages:
            return -1

        auto_title = title or _auto_title(session)
        messages_json = json.dumps(
            [{"role": m.role, "content": m.content} for m in session.messages],
            ensure_ascii=False,
        )
        now = datetime.now().isoformat()
        token_count = sum(len(m.content) for m in session.messages)  # 근사치

        conn = self._get_conn()
        cur = conn.execute(
            """
            INSERT INTO sessions (title, model, provider, system, mode, messages, token_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                auto_title,
                session.model,
                session.provider or "",
                session.system,
                session.mode,
                messages_json,
                token_count,
                now,
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid or -1

    def load_session(self, session_id: int) -> Optional[ChatSession]:
        """저장된 세션 복원"""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None

        session = ChatSession(
            model=row["model"],
            provider=row["provider"] or None,
            system=row["system"],
        )
        session.mode = row["mode"]

        messages = json.loads(row["messages"])
        for msg in messages:
            session.messages.append(ChatMessage(role=msg["role"], content=msg["content"]))

        return session

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """최근 세션 목록"""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT id, title, model, mode, token_count, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"],
                "model": r["model"],
                "mode": r["mode"],
                "token_count": r["token_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: int) -> bool:
        """세션 삭제"""
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # 입력 히스토리
    # ------------------------------------------------------------------

    def append_history(self, content: str) -> None:
        """사용자 입력 히스토리 저장"""
        if not content.strip():
            return
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO input_history (content, created_at) VALUES (?, ?)",
            (content, datetime.now().isoformat()),
        )
        conn.commit()

    def get_history(self, limit: int = 500) -> List[str]:
        """최근 입력 히스토리 반환 (오래된 순)"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT content FROM input_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["content"] for r in reversed(rows)]

    # ------------------------------------------------------------------
    # 정리
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------


def _auto_title(session: ChatSession) -> str:
    """첫 사용자 메시지에서 자동 제목 생성"""
    for msg in session.messages:
        if msg.role == "user":
            title = msg.content[:60].strip()
            if len(msg.content) > 60:
                title += "..."
            return title
    return f"Session ({session.model})"
