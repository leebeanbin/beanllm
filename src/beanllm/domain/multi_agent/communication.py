"""
Communication System - Agent 간 통신
"""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from beanllm.domain.protocols import EventBusProtocol, EventLoggerProtocol

from beanllm.utils.logging import get_logger

logger = get_logger(__name__)

# 환경변수로 분산 모드 활성화 여부 확인
USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"


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
            sender=self.receiver or "unknown",
            receiver=self.sender,
            message_type=message_type,
            content=content,
            reply_to=self.id,
        )


class CommunicationBus:
    """
    Agent 간 통신 버스

    Publish-Subscribe 패턴 구현
    분산 모드: Kafka를 통한 메시지 스트리밍 지원
    """

    def __init__(
        self,
        delivery_guarantee: str = "at-most-once",
        use_kafka: Optional[bool] = None,
        event_bus: Optional["EventBusProtocol"] = None,
        event_logger: Optional["EventLoggerProtocol"] = None,
    ):
        """
        Args:
            delivery_guarantee: 전송 보장 수준
                - "at-most-once": 최대 1번 (빠름, 손실 가능)
                - "at-least-once": 최소 1번 (중복 가능)
                - "exactly-once": 정확히 1번 (느림, 보장)
            use_kafka: Kafka 사용 여부 (None이면 USE_DISTRIBUTED 환경변수 사용)
            event_bus: 이벤트 버스 프로토콜 (옵션, Service layer에서 주입)
            event_logger: 이벤트 로거 프로토콜 (옵션, Service layer에서 주입)
        """
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[Callable]] = {}  # agent_id -> [callbacks]
        self.delivery_guarantee = delivery_guarantee
        self.delivered_messages: set = set()  # For exactly-once

        # 분산 모드: Kafka 이벤트 버스
        self.use_kafka = use_kafka if use_kafka is not None else USE_DISTRIBUTED
        self.kafka_producer = None
        self.kafka_consumer = None
        self.event_logger = event_logger

        # 외부에서 주입된 이벤트 버스 사용
        if event_bus is not None:
            self.kafka_producer = event_bus
            self.kafka_consumer = event_bus
            self.use_kafka = True
            logger.info("CommunicationBus: Event bus enabled (injected)")
        elif self.use_kafka:
            # 분산 모드가 활성화되어 있지만 주입되지 않은 경우 경고
            logger.warning(
                "USE_DISTRIBUTED=true but event_bus not injected. "
                "Falling back to in-memory mode. "
                "Please inject event_bus from Service layer for distributed features."
            )
            self.use_kafka = False

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
        분산 모드: Kafka를 통한 메시지 스트리밍
        """
        self.messages.append(message)

        # Exactly-once: 중복 방지
        if self.delivery_guarantee == "exactly-once":
            if message.id in self.delivered_messages:
                logger.debug(f"Message {message.id} already delivered, skipping")
                return
            self.delivered_messages.add(message.id)

        # 분산 모드: Kafka에 메시지 발행
        if self.use_kafka and self.kafka_producer:
            try:
                import json

                message_data = {
                    "id": message.id,
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "message_type": message.message_type.value,
                    "content": str(message.content)[:1000],  # 처음 1000자만
                    "timestamp": message.timestamp.isoformat(),
                    "reply_to": message.reply_to,
                }
                await self.kafka_producer.publish("multi_agent.messages", message_data)

                # 이벤트 로깅
                if self.event_logger:
                    await self.event_logger.log_event(
                        "multi_agent.message_published",
                        {
                            "message_id": message.id,
                            "sender": message.sender,
                            "receiver": message.receiver or "broadcast",
                            "message_type": message.message_type.value,
                        },
                        level="info",
                    )
            except Exception as e:
                logger.warning(f"Failed to publish message to Kafka: {e}")

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
