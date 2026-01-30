"""
Context Manager Service

beanllm 메모리 시스템을 활용한 컨텍스트 관리
세션별 메모리 관리, 자동 요약, 토큰 제한 지원

Usage:
    from services.context_manager import context_manager

    # 메시지 추가
    context_manager.add_message(session_id, "user", "안녕하세요")
    context_manager.add_message(session_id, "assistant", "안녕하세요!")

    # 컨텍스트 가져오기
    messages = context_manager.get_context(session_id)

    # 세션 정리
    context_manager.clear_session(session_id)
"""

import logging
from typing import Any, Dict, List, Optional

from beanllm.domain.memory import (
    BaseMemory,
    BufferMemory,
    ConversationMemory,
    SummaryMemory,
    TokenMemory,
    WindowMemory,
    create_memory,
)

logger = logging.getLogger(__name__)


class ContextManager:
    """
    세션별 컨텍스트 관리자

    beanllm 메모리 시스템을 활용하여:
    - 세션별 메시지 저장
    - 토큰 제한 자동 관리
    - 오래된 대화 요약
    """

    def __init__(
        self,
        default_memory_type: str = "token",
        max_tokens: int = 4000,
        max_messages: int = 20,
        summary_trigger: int = 15,
    ):
        """
        Args:
            default_memory_type: 기본 메모리 타입 (buffer, window, token, summary)
            max_tokens: 최대 토큰 수 (token 메모리용)
            max_messages: 최대 메시지 수 (window, summary 메모리용)
            summary_trigger: 요약 트리거 (summary 메모리용)
        """
        self._sessions: Dict[str, BaseMemory] = {}
        self._default_memory_type = default_memory_type
        self._max_tokens = max_tokens
        self._max_messages = max_messages
        self._summary_trigger = summary_trigger
        logger.info(
            f"ContextManager initialized: type={default_memory_type}, "
            f"max_tokens={max_tokens}, max_messages={max_messages}"
        )

    def _get_or_create_memory(
        self,
        session_id: str,
        memory_type: Optional[str] = None,
    ) -> BaseMemory:
        """세션용 메모리 가져오기 또는 생성"""
        if session_id not in self._sessions:
            mem_type = memory_type or self._default_memory_type
            memory = self._create_memory(mem_type)
            self._sessions[session_id] = memory
            logger.info(f"Created new memory for session {session_id}: type={mem_type}")
        return self._sessions[session_id]

    def _create_memory(self, memory_type: str) -> BaseMemory:
        """메모리 인스턴스 생성"""
        memory_configs = {
            "buffer": {"max_messages": self._max_messages},
            "window": {"window_size": self._max_messages},
            "token": {"max_tokens": self._max_tokens},
            "summary": {
                "max_messages": self._max_messages,
                "summary_trigger": self._summary_trigger,
            },
            "conversation": {"max_pairs": self._max_messages // 2},
        }
        config = memory_configs.get(memory_type, {})
        return create_memory(memory_type, **config)

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        memory_type: Optional[str] = None,
        **metadata,
    ) -> None:
        """
        메시지 추가

        Args:
            session_id: 세션 ID
            role: 역할 (user, assistant, system)
            content: 메시지 내용
            memory_type: 메모리 타입 (선택, 첫 메시지 시 설정)
            **metadata: 추가 메타데이터
        """
        memory = self._get_or_create_memory(session_id, memory_type)
        memory.add_message(role, content, **metadata)
        logger.debug(
            f"Added message to session {session_id}: role={role}, len={len(content)}"
        )

    def get_context(
        self,
        session_id: str,
        as_dict: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        세션 컨텍스트 가져오기

        Args:
            session_id: 세션 ID
            as_dict: True면 dict 리스트 반환, False면 Message 객체 리스트

        Returns:
            메시지 리스트
        """
        if session_id not in self._sessions:
            return []

        memory = self._sessions[session_id]
        messages = memory.get_messages()

        if as_dict:
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    **(msg.metadata if hasattr(msg, "metadata") and msg.metadata else {}),
                }
                for msg in messages
            ]
        return messages

    def get_context_for_llm(
        self,
        session_id: str,
        include_system: bool = True,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        LLM 호출용 컨텍스트 가져오기

        Args:
            session_id: 세션 ID
            include_system: 시스템 프롬프트 포함 여부
            system_prompt: 추가 시스템 프롬프트

        Returns:
            LLM 호출용 메시지 리스트
        """
        messages = []

        # 시스템 프롬프트 추가
        if include_system and system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 세션 메시지 추가
        context = self.get_context(session_id, as_dict=True)
        for msg in context:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    def clear_session(self, session_id: str) -> bool:
        """
        세션 메모리 초기화

        Args:
            session_id: 세션 ID

        Returns:
            성공 여부
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info(f"Cleared session {session_id}")
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """
        세션 삭제

        Args:
            session_id: 세션 ID

        Returns:
            성공 여부
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        세션 통계 가져오기

        Args:
            session_id: 세션 ID

        Returns:
            세션 통계
        """
        if session_id not in self._sessions:
            return {"exists": False}

        memory = self._sessions[session_id]
        messages = memory.get_messages()

        total_tokens = sum(
            len(msg.content.split()) * 1.3 for msg in messages  # 간단한 토큰 추정
        )

        return {
            "exists": True,
            "message_count": len(memory),
            "estimated_tokens": int(total_tokens),
            "memory_type": type(memory).__name__,
            "has_summary": (
                hasattr(memory, "summary") and memory.summary is not None
            ),
        }

    def list_sessions(self) -> List[str]:
        """활성 세션 목록"""
        return list(self._sessions.keys())

    def cleanup_inactive_sessions(self, max_sessions: int = 100) -> int:
        """
        비활성 세션 정리 (LRU 방식)

        Args:
            max_sessions: 최대 세션 수

        Returns:
            삭제된 세션 수
        """
        if len(self._sessions) <= max_sessions:
            return 0

        # 오래된 세션부터 삭제 (FIFO)
        sessions_to_delete = list(self._sessions.keys())[
            : len(self._sessions) - max_sessions
        ]

        for session_id in sessions_to_delete:
            del self._sessions[session_id]

        logger.info(f"Cleaned up {len(sessions_to_delete)} inactive sessions")
        return len(sessions_to_delete)

    # ===========================================
    # 요약 기능 (Summarization)
    # ===========================================

    def needs_summarization(self, session_id: str) -> bool:
        """
        요약이 필요한지 확인

        Args:
            session_id: 세션 ID

        Returns:
            요약 필요 여부
        """
        if session_id not in self._sessions:
            return False

        memory = self._sessions[session_id]
        message_count = len(memory)

        # 메시지 수가 summary_trigger 초과 시 요약 필요
        if message_count > self._summary_trigger:
            # 이미 요약이 있으면 불필요
            if hasattr(memory, "summary") and memory.summary:
                return False
            return True

        return False

    async def summarize_if_needed(
        self,
        session_id: str,
        model: str = "qwen2.5:0.5b",
    ) -> Optional[str]:
        """
        필요 시 요약 실행

        Args:
            session_id: 세션 ID
            model: 요약에 사용할 모델

        Returns:
            생성된 요약 (없으면 None)
        """
        if not self.needs_summarization(session_id):
            return None

        memory = self._sessions[session_id]
        messages = memory.get_messages()

        # 요약할 메시지 선택 (최근 10개 제외)
        messages_to_summarize = messages[:-10] if len(messages) > 10 else messages[:-5]

        if len(messages_to_summarize) < 5:
            return None

        # 요약 생성
        summary = await self._generate_summary(messages_to_summarize, model)

        # 메모리에 저장
        if hasattr(memory, "summary"):
            memory.summary = summary

        # 세션 메타데이터에 저장
        if session_id not in self._session_summaries:
            self._session_summaries = getattr(self, "_session_summaries", {})
        self._session_summaries[session_id] = summary

        logger.info(f"Generated summary for session {session_id}: {len(summary)} chars")
        return summary

    async def _generate_summary(
        self,
        messages: List[Any],
        model: str = "qwen2.5:0.5b",
    ) -> str:
        """
        LLM을 사용하여 요약 생성

        Args:
            messages: 요약할 메시지들
            model: 요약에 사용할 모델

        Returns:
            생성된 요약
        """
        from beanllm.facade.core import Client

        # 대화 내용 구성
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content[:500]}"  # 메시지당 500자 제한
            for msg in messages
        ])

        # 요약 프롬프트
        SUMMARIZATION_PROMPT = """다음 대화 내용을 요약해주세요. 다음 정보는 반드시 포함해야 합니다:

1. 주요 주제 및 목적
2. 중요한 결정 사항
3. 사용자가 언급한 특별한 요구사항
4. 해결된 문제와 해결 방법
5. 미완료된 작업이나 다음 단계

대화 내용:
{conversation_history}

요약 (200-300자):"""

        prompt = SUMMARIZATION_PROMPT.format(
            conversation_history=conversation_text[:3000]  # 3000자 제한
        )

        try:
            # 요약 모델 사용 (경량 모델)
            summarizer = Client(model=model)
            response = await summarizer.chat([
                {"role": "user", "content": prompt}
            ])
            return response.content
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # 폴백: 간단한 요약
            return f"대화 {len(messages)}개 메시지 요약: {messages[-1].content[:100]}..."

    def get_summary(self, session_id: str) -> Optional[str]:
        """
        세션 요약 가져오기

        Args:
            session_id: 세션 ID

        Returns:
            요약 (없으면 None)
        """
        # 메모리에서 요약 확인
        if session_id in self._sessions:
            memory = self._sessions[session_id]
            if hasattr(memory, "summary") and memory.summary:
                return memory.summary

        # 세션 메타데이터에서 확인
        summaries = getattr(self, "_session_summaries", {})
        return summaries.get(session_id)

    def get_context_with_summary(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        요약이 포함된 LLM 컨텍스트 가져오기

        Args:
            session_id: 세션 ID
            system_prompt: 추가 시스템 프롬프트

        Returns:
            LLM 호출용 메시지 리스트 (요약 포함)
        """
        messages = []

        # 1. 요약이 있으면 system 메시지로 추가
        summary = self.get_summary(session_id)
        if summary:
            summary_content = f"이전 대화 요약:\n{summary}\n\n최근 대화:"
            if system_prompt:
                summary_content = f"{system_prompt}\n\n{summary_content}"
            messages.append({"role": "system", "content": summary_content})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. 최근 메시지 추가
        context = self.get_context(session_id, as_dict=True)
        for msg in context:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages


# Singleton instance
context_manager = ContextManager()
