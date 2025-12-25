"""
Memory Factory
"""

from .base import BaseMemory
from .implementations import (
    BufferMemory,
    ConversationMemory,
    SummaryMemory,
    TokenMemory,
    WindowMemory,
)


def create_memory(memory_type: str = "buffer", **kwargs) -> BaseMemory:
    """
    메모리 생성 팩토리

    Args:
        memory_type: 메모리 타입 (buffer, window, token, summary, conversation)
        **kwargs: 메모리별 파라미터

    Returns:
        BaseMemory: 메모리 인스턴스

    Example:
        ```python
        from beanllm.domain.memory import create_memory

        # 버퍼 메모리
        memory = create_memory("buffer", max_messages=100)

        # 윈도우 메모리
        memory = create_memory("window", window_size=10)

        # 토큰 메모리
        memory = create_memory("token", max_tokens=4000)
        ```
    """
    memory_map = {
        "buffer": BufferMemory,
        "window": WindowMemory,
        "token": TokenMemory,
        "summary": SummaryMemory,
        "conversation": ConversationMemory,
    }

    memory_class = memory_map.get(memory_type)
    if not memory_class:
        raise ValueError(f"Unknown memory type: {memory_type}")

    return memory_class(**kwargs)
