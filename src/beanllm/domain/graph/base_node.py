"""
BaseNode - 노드 베이스 클래스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .graph_state import GraphState


class BaseNode(ABC):
    """
    노드 베이스 클래스
    """

    def __init__(self, name: str, cache: bool = False, description: Optional[str] = None):
        """
        Args:
            name: 노드 이름
            cache: 캐싱 사용 여부
            description: 설명
        """
        self.name = name
        self.cache_enabled = cache
        self.description = description or ""

    @abstractmethod
    async def execute(self, state: GraphState) -> Dict[str, Any]:
        """
        노드 실행

        Args:
            state: 현재 상태

        Returns:
            상태 업데이트 딕셔너리
        """
        pass
