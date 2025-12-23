"""
Communication System - Agent 간 통신
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ...utils.logger import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """메시지 타입"""

    REQUEST = "request"  # 작업 요청
    RESPONSE = "response"  # 작업 응답
    BROADCAST = "broadcast"  # 전체 공지
    QUERY = "query"  # 정보 요청
    INFORM = "inform"  # 정보 전달
    DELEGATE = "delegate"  # 작업 위임
    VOTE = "vote"  # 투표
    CONSENSUS = "consensus"  # 합의


@dataclass
class AgentMessage:
    """
    Agent 간 메시지

    Mathematical Foundation:
        Message Passing Model에서 메시지는 튜플로 표현됩니다:
        m = (sender, receiver, content, timestamp)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""  # 송신자 agent ID
    receiver: Optional[str] = None  # 수신자 (None이면 broadcast)
    message_type: MessageType = MessageType.INFORM
    content: Any = None  # 메시지 내용
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None  # 답장하는 메시지 ID

    def reply(
        self, content: Any, message_type: MessageType = MessageType.RESPONSE
    ) -> "AgentMessage":
        """이 메시지에 대한 답장 생성"""
        return AgentMessage(
            sender=self.receiver,
            receiver=self.sender,
            message_type=message_type,
            content=content,
            reply_to=self.id,
        )


class CommunicationBus:
    """
    Agent 간 통신 버스

    Publish-Subscribe 패턴 구현
    """

    def __init__(self, delivery_guarantee: str = "at-most-once"):
        """
        Args:
            delivery_guarantee: 전송 보장 수준
                - "at-most-once": 최대 1번 (빠름, 손실 가능)
                - "at-least-once": 최소 1번 (중복 가능)
                - "exactly-once": 정확히 1번 (느림, 보장)
        """
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[Callable]] = {}  # agent_id -> [callbacks]
        self.delivery_guarantee = delivery_guarantee
        self.delivered_messages: set = set()  # For exactly-once

    def subscribe(self, agent_id: str, callback: Callable[[AgentMessage], None]):
        """메시지 구독"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        logger.debug(f"Agent {agent_id} subscribed to bus")

    def unsubscribe(self, agent_id: str, callback: Optional[Callable] = None):
        """구독 취소"""
        if agent_id in self.subscribers:
            if callback:
                self.subscribers[agent_id].remove(callback)
            else:
                del self.subscribers[agent_id]

    async def publish(self, message: AgentMessage):
        """
        메시지 발행

        Time Complexity: O(n) where n = number of subscribers
        """
        self.messages.append(message)

        # Exactly-once: 중복 방지
        if self.delivery_guarantee == "exactly-once":
            if message.id in self.delivered_messages:
                logger.debug(f"Message {message.id} already delivered, skipping")
                return
            self.delivered_messages.add(message.id)

        # 수신자에게 전달
        if message.receiver:
            # Unicast (1:1)
            if message.receiver in self.subscribers:
                for callback in self.subscribers[message.receiver]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
        else:
            # Broadcast (1:N)
            for agent_id, callbacks in self.subscribers.items():
                # 자기 자신은 제외
                if agent_id == message.sender:
                    continue

                for callback in callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in callback for {agent_id}: {e}")

    def get_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[AgentMessage]:
        """메시지 히스토리 조회"""
        if agent_id:
            filtered = [m for m in self.messages if m.sender == agent_id or m.receiver == agent_id]
            return filtered[-limit:]
        return self.messages[-limit:]
