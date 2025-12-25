"""
Memory Implementations
"""

from typing import Any, List, Optional

from ...utils.logger import get_logger
from .base import BaseMemory, Message

logger = get_logger(__name__)


class BufferMemory(BaseMemory):
    """
    버퍼 메모리

    모든 메시지를 저장하는 기본 메모리

    Example:
        ```python
        from beanllm.domain.memory import BufferMemory

        memory = BufferMemory()
        memory.add_message("user", "안녕하세요")
        memory.add_message("assistant", "안녕하세요! 무엇을 도와드릴까요?")

        messages = memory.get_messages()
        print(f"Total messages: {len(messages)}")
        ```
    """

    def __init__(self, max_messages: Optional[int] = None):
        """
        Args:
            max_messages: 최대 메시지 수 (None이면 무제한)
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(msg)

        # 최대 메시지 수 제한
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

        logger.debug(f"Added message: {role} ({len(content)} chars)")

    def get_messages(self) -> List[Message]:
        """메시지 가져오기"""
        return self.messages.copy()

    def clear(self):
        """메모리 초기화"""
        count = len(self.messages)
        self.messages.clear()
        logger.info(f"Cleared {count} messages")

    def __len__(self):
        return len(self.messages)


class WindowMemory(BaseMemory):
    """
    윈도우 메모리

    최근 N개의 메시지만 유지

    Example:
        ```python
        from beanllm.domain.memory import WindowMemory

        # 최근 10개만 유지
        memory = WindowMemory(window_size=10)

        for i in range(20):
            memory.add_message("user", f"Message {i}")

        # 10개만 남음
        assert len(memory) == 10
        ```
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: 윈도우 크기 (메시지 개수)
        """
        self.messages: List[Message] = []
        self.window_size = window_size

    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(msg)

        # 윈도우 크기 유지
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size :]

    def get_messages(self) -> List[Message]:
        """메시지 가져오기"""
        return self.messages.copy()

    def clear(self):
        """메모리 초기화"""
        self.messages.clear()

    def __len__(self):
        return len(self.messages)


class TokenMemory(BaseMemory):
    """
    토큰 제한 메모리

    토큰 수 기준으로 메시지 유지

    Example:
        ```python
        from beanllm.domain.memory import TokenMemory

        # 최대 1000 토큰까지
        memory = TokenMemory(max_tokens=1000)

        memory.add_message("user", "긴 메시지...")
        memory.add_message("assistant", "응답...")

        # 토큰 초과 시 오래된 메시지부터 제거
        ```
    """

    def __init__(self, max_tokens: int = 4000):
        """
        Args:
            max_tokens: 최대 토큰 수
        """
        self.messages: List[Message] = []
        self.max_tokens = max_tokens

    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(msg)

        # 토큰 수 제한
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            logger.debug(f"Removed message to fit token limit: {removed.role}")

    def get_messages(self) -> List[Message]:
        """메시지 가져오기"""
        return self.messages.copy()

    def clear(self):
        """메모리 초기화"""
        self.messages.clear()

    def _estimate_tokens(self) -> int:
        """토큰 수 추정 (단어 수 기준)"""
        total = 0
        for msg in self.messages:
            # 간단한 추정: 단어 수 * 1.3
            words = len(msg.content.split())
            total += int(words * 1.3)
        return total

    def __len__(self):
        return len(self.messages)


class SummaryMemory(BaseMemory):
    """
    요약 메모리

    오래된 대화는 요약하여 저장

    Example:
        ```python
        from beanllm import Client
        from beanllm.domain.memory import SummaryMemory

        client = Client(model="gpt-4o-mini")
        memory = SummaryMemory(
            summarizer=client,
            max_messages=10
        )

        # 10개 초과 시 자동 요약
        for i in range(20):
            memory.add_message("user", f"Question {i}")
            memory.add_message("assistant", f"Answer {i}")
        ```
    """

    def __init__(
        self, summarizer: Optional[Any] = None, max_messages: int = 10, summary_trigger: int = 5
    ):
        """
        Args:
            summarizer: 요약에 사용할 Client 인스턴스
            max_messages: 최대 메시지 수
            summary_trigger: 요약 트리거 (이 개수 초과 시 요약)
        """
        self.messages: List[Message] = []
        self.summary: Optional[str] = None
        self.summarizer = summarizer
        self.max_messages = max_messages
        self.summary_trigger = summary_trigger

    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(msg)

        # 요약 트리거
        if len(self.messages) > self.summary_trigger:
            self._maybe_summarize()

    def get_messages(self) -> List[Message]:
        """메시지 가져오기"""
        messages = []

        # 요약이 있으면 system 메시지로 추가
        if self.summary:
            messages.append(
                Message(role="system", content=f"Previous conversation summary:\n{self.summary}")
            )

        # 최근 메시지 추가
        messages.extend(self.messages.copy())
        return messages

    def clear(self):
        """메모리 초기화"""
        self.messages.clear()
        self.summary = None

    def _maybe_summarize(self):
        """필요 시 요약 실행"""
        if not self.summarizer:
            # 요약기 없으면 오래된 메시지 제거
            while len(self.messages) > self.max_messages:
                self.messages.pop(0)
            return

        # 요약 실행 (비동기 처리는 추후 개선)
        # 현재는 간단하게 오래된 메시지만 제거
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def __len__(self):
        return len(self.messages)


class ConversationMemory(BaseMemory):
    """
    대화 메모리

    User-Assistant 쌍으로 관리

    Example:
        ```python
        from beanllm.domain.memory import ConversationMemory

        memory = ConversationMemory()

        memory.add_user_message("안녕하세요")
        memory.add_ai_message("안녕하세요! 무엇을 도와드릴까요?")

        memory.add_user_message("날씨 알려줘")
        memory.add_ai_message("오늘 날씨는 맑습니다")

        # 대화 쌍으로 관리
        pairs = memory.get_conversation_pairs()
        ```
    """

    def __init__(self, max_pairs: Optional[int] = None):
        """
        Args:
            max_pairs: 최대 대화 쌍 수
        """
        self.messages: List[Message] = []
        self.max_pairs = max_pairs

    def add_message(self, role: str, content: str, **kwargs):
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=kwargs)
        self.messages.append(msg)

        # 대화 쌍 제한
        if self.max_pairs:
            self._trim_to_pairs()

    def add_user_message(self, content: str, **kwargs):
        """사용자 메시지 추가"""
        self.add_message("user", content, **kwargs)

    def add_ai_message(self, content: str, **kwargs):
        """AI 메시지 추가"""
        self.add_message("assistant", content, **kwargs)

    def get_messages(self) -> List[Message]:
        """메시지 가져오기"""
        return self.messages.copy()

    def get_conversation_pairs(self) -> List[tuple]:
        """대화 쌍 가져오기"""
        pairs = []
        for i in range(0, len(self.messages) - 1, 2):
            if i + 1 < len(self.messages):
                pairs.append((self.messages[i], self.messages[i + 1]))
        return pairs

    def clear(self):
        """메모리 초기화"""
        self.messages.clear()

    def _trim_to_pairs(self):
        """대화 쌍 수 제한"""
        # User-Assistant 쌍으로 계산
        pair_count = len(self.messages) // 2
        if pair_count > self.max_pairs:
            excess = (pair_count - self.max_pairs) * 2
            self.messages = self.messages[excess:]

    def __len__(self):
        return len(self.messages)
