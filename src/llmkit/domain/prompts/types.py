"""
Prompts Types - 프롬프트 데이터 타입
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PromptExample:
    """Few-shot 예제"""

    input: str
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """채팅 메시지"""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result
