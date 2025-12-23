"""
GraphState - 그래프 상태
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GraphState:
    """
    그래프 상태

    노드 간 데이터 전달용
    """

    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """값 가져오기"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """값 설정"""
        self.data[key] = value

    def update(self, updates: Dict[str, Any]):
        """여러 값 업데이트"""
        self.data.update(updates)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data
