"""
인메모리 이벤트 버스 (기존 CommunicationBus 래핑)
"""

import asyncio
from typing import Any, AsyncIterator, Callable, Dict

from ..interfaces import EventProducerInterface, EventConsumerInterface


class InMemoryEventBus(EventProducerInterface, EventConsumerInterface):
    """
    인메모리 이벤트 버스

    기존 CommunicationBus를 참고하여 인메모리 Pub/Sub 구현
    """

    def __init__(self):
        self._subscribers: Dict[str, list] = {}  # topic -> [handlers]
        self._events: list = []  # 이벤트 히스토리
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, event: Dict[str, Any]):
        """이벤트 발행"""
        event_with_topic = {
            "topic": topic,
            "event": event,
            "timestamp": asyncio.get_event_loop().time(),
        }
        
        # 이벤트 히스토리에 추가
        async with self._lock:
            self._events.append(event_with_topic)
            # 최근 1000개만 유지
            if len(self._events) > 1000:
                self._events = self._events[-1000:]
        
        # 구독자에게 전달
        if topic in self._subscribers:
            for handler in self._subscribers[topic]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    # 핸들러 오류는 무시 (로깅만)
                    import logging
                    logging.error(f"Event handler error: {e}", exc_info=True)

    async def subscribe(self, topic: str, handler: Any) -> AsyncIterator[Dict[str, Any]]:
        """이벤트 구독"""
        # 구독자 등록
        async with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(handler)
        
        # 기존 이벤트 재생 (선택적)
        # 여기서는 실시간 이벤트만 전달
        queue = asyncio.Queue()
        
        async def _handler(event: Dict[str, Any]):
            await queue.put(event)
        
        # 핸들러를 큐에 연결
        async with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            # 기존 핸들러는 유지하고 큐 핸들러 추가
            self._subscribers[topic].append(_handler)
        
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            # 구독 해제
            async with self._lock:
                if topic in self._subscribers and _handler in self._subscribers[topic]:
                    self._subscribers[topic].remove(_handler)

