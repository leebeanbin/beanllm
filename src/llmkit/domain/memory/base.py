"""
Memory Base Classes
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """메시지"""

    role: str  # user, assistant, system
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class BaseMemory(ABC):
    """
    메모리 베이스 클래스

    모든 메모리 구현체의 기본 인터페이스
    """

    @abstractmethod
    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        pass

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """메시지 가져오기"""
        pass

    @abstractmethod
    def clear(self):
        """메모리 초기화"""
        pass

    def get_dict_messages(self) -> List[Dict]:
        """딕셔너리 형태로 메시지 반환"""
        return [{"role": msg.role, "content": msg.content} for msg in self.get_messages()]
