"""
Shared Memory System - Agent 간 지식 공유 (Whiteboard 패턴)
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional


class SharedWhiteboard:
    """
    Agent 간 지식 공유를 위한 공용 화이트보드

    - Thread-safe (asyncio.Lock 사용)
    - Key-Value 저장소 기능
    - 히스토리 추적 (max_history 엔트리 제한으로 OOM 방지)
    - to_summary_string()은 GIL-safe snapshot을 사용해 동기 컨텍스트에서 안전하게 호출 가능
    """

    def __init__(self, max_entries: int = 500, max_history: int = 200) -> None:
        """
        Args:
            max_entries: 저장 가능한 최대 키 수. 초과 시 가장 오래된 키부터 제거.
            max_history: 보관할 최대 변경 히스토리 수.
        """
        self._data: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._lock: Optional[asyncio.Lock] = None
        self._max_entries = max_entries
        self._max_history = max_history

    def _get_lock(self) -> asyncio.Lock:
        """이벤트 루프별로 Lock을 지연 생성해 cross-loop 오류를 방지."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def set(self, key: str, value: Any, agent_id: str) -> None:
        """데이터 저장. max_entries 초과 시 가장 오래된 키 제거."""
        async with self._get_lock():
            old_value = self._data.get(key)
            self._data[key] = value

            if len(self._data) > self._max_entries:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]

            self._history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "set",
                    "agent_id": agent_id,
                    "key": key,
                    "old_value": old_value,
                    "new_value": value,
                }
            )
            if len(self._history) > self._max_history:
                del self._history[: len(self._history) - self._max_history]

    async def get(self, key: str) -> Any:
        """데이터 조회."""
        async with self._get_lock():
            return self._data.get(key)

    async def append(self, key: str, value: Any, agent_id: str) -> None:
        """리스트 형태 데이터에 추가."""
        async with self._get_lock():
            if key not in self._data or not isinstance(self._data[key], list):
                self._data[key] = []
            self._data[key].append(value)
            self._history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "append",
                    "agent_id": agent_id,
                    "key": key,
                    "value": value,
                }
            )
            if len(self._history) > self._max_history:
                del self._history[: len(self._history) - self._max_history]

    async def get_all(self) -> Dict[str, Any]:
        """모든 데이터 조회."""
        async with self._get_lock():
            return self._data.copy()

    async def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """변경 히스토리 조회."""
        async with self._get_lock():
            return self._history[-limit:]

    def to_summary_string(self) -> str:
        """프롬프트 삽입용 요약 문자열 생성.

        asyncio.Lock을 획득할 수 없는 동기 컨텍스트에서 안전하게 호출 가능.
        CPython의 GIL 하에서 dict.copy()는 원자적이므로 경쟁 조건 없이 스냅샷 읽기 가능.
        """
        data = self._data.copy()  # GIL-safe atomic copy
        if not data:
            return "Shared Whiteboard is empty."
        lines = "\n".join(f"- {k}: {v}" for k, v in data.items())
        return f"Shared Whiteboard Knowledge:\n{lines}"
