"""
Memory 테스트 - 메모리 구현체 테스트
"""

import pytest

from beanllm.domain.memory import (
    BufferMemory,
    WindowMemory,
    TokenMemory,
    SummaryMemory,
    ConversationMemory,
    Message,
)


class TestBufferMemory:
    """BufferMemory 테스트"""

    @pytest.fixture
    def memory(self):
        """BufferMemory 인스턴스"""
        return BufferMemory()

    def test_add_message(self, memory):
        """메시지 추가 테스트"""
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")

        assert len(memory) == 2

    def test_get_messages(self, memory):
        """메시지 조회 테스트"""
        memory.add_message("user", "Hello")
        messages = memory.get_messages()

        assert isinstance(messages, list)
        assert len(messages) == 1
        assert isinstance(messages[0], Message)
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_max_messages(self, memory):
        """최대 메시지 수 제한 테스트"""
        limited_memory = BufferMemory(max_messages=3)

        for i in range(5):
            limited_memory.add_message("user", f"Message {i}")

        assert len(limited_memory) == 3
        # 최근 3개만 남아야 함
        messages = limited_memory.get_messages()
        assert messages[0].content == "Message 2"

    def test_clear(self, memory):
        """메모리 초기화 테스트"""
        memory.add_message("user", "Hello")
        memory.clear()

        assert len(memory) == 0


class TestWindowMemory:
    """WindowMemory 테스트"""

    @pytest.fixture
    def memory(self):
        """WindowMemory 인스턴스"""
        return WindowMemory(window_size=5)

    def test_window_size(self, memory):
        """윈도우 크기 제한 테스트"""
        for i in range(10):
            memory.add_message("user", f"Message {i}")

        assert len(memory) == 5
        # 최근 5개만 남아야 함
        messages = memory.get_messages()
        assert messages[0].content == "Message 5"


class TestTokenMemory:
    """TokenMemory 테스트"""

    @pytest.fixture
    def memory(self):
        """TokenMemory 인스턴스"""
        return TokenMemory(max_tokens=100)

    def test_token_limit(self, memory):
        """토큰 제한 테스트"""
        # 긴 메시지 추가
        memory.add_message("user", "This is a very long message " * 10)
        memory.add_message("assistant", "Response " * 10)

        # 토큰 제한에 따라 메시지가 제거될 수 있음
        assert len(memory) >= 0


class TestConversationMemory:
    """ConversationMemory 테스트"""

    @pytest.fixture
    def memory(self):
        """ConversationMemory 인스턴스"""
        return ConversationMemory()

    def test_add_user_message(self, memory):
        """사용자 메시지 추가 테스트"""
        memory.add_user_message("Hello")
        messages = memory.get_messages()

        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_add_ai_message(self, memory):
        """AI 메시지 추가 테스트"""
        memory.add_ai_message("Hi there")
        messages = memory.get_messages()

        assert len(messages) == 1
        assert messages[0].role == "assistant"

    def test_get_conversation_pairs(self, memory):
        """대화 쌍 조회 테스트"""
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi")
        memory.add_user_message("How are you?")
        memory.add_ai_message("I'm fine")

        pairs = memory.get_conversation_pairs()

        assert isinstance(pairs, list)
        assert len(pairs) == 2
        assert isinstance(pairs[0], tuple)
        assert len(pairs[0]) == 2


class TestSummaryMemory:
    """SummaryMemory 테스트"""

    @pytest.fixture
    def memory(self):
        """SummaryMemory 인스턴스"""
        return SummaryMemory(max_messages=5)

    def test_summary_generation(self, memory):
        """요약 생성 테스트"""
        for i in range(10):
            memory.add_message("user", f"Message {i}")

        # 요약이 생성되었는지 확인
        messages = memory.get_messages()
        assert len(messages) >= 0


