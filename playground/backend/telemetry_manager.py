import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket

from beanllm.domain.multi_agent.communication import TelemetryEvent

logger = logging.getLogger(__name__)


class TelemetryManager:
    """
    WebSocket을 통한 실시간 텔레메트리 스트리밍 및 영속화 관리

    IPlanRepository 프로토콜 구현 — playground에서 MultiAgentServiceImpl에 주입 가능:
        from telemetry_manager import telemetry_manager
        service = MultiAgentServiceImpl(plan_repository=telemetry_manager)
    """

    def __init__(self):
        # execution_id -> list of connected WebSockets
        self.active_sessions: Dict[str, Set[WebSocket]] = {}
        # execution_id -> list of buffered events (for late joiners)
        self.event_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self.buffer_limit = 100
        # execution_id -> asyncio.Event for resuming
        self.resume_events: Dict[str, asyncio.Event] = {}
        # execution_id -> modified plan from user
        self.modified_plans: Dict[str, List[Dict[str, Any]]] = {}

    def get_resume_event(self, execution_id: str) -> asyncio.Event:
        """재개 이벤트 가져오기 (없으면 생성)"""
        if execution_id not in self.resume_events:
            self.resume_events[execution_id] = asyncio.Event()
        return self.resume_events[execution_id]

    def set_modified_plan(self, execution_id: str, plan: List[Dict[str, Any]]):
        """수정된 계획 저장"""
        self.modified_plans[execution_id] = plan

    def get_modified_plan(self, execution_id: str) -> Optional[List[Dict[str, Any]]]:
        """수정된 계획 가져오기"""
        return self.modified_plans.get(execution_id)

    def resume_execution(self, execution_id: str):
        """실행 재개 신호"""
        if execution_id in self.resume_events:
            self.resume_events[execution_id].set()

    async def connect(self, execution_id: str, websocket: WebSocket):
        """세션에 WebSocket 연결 추가"""
        if execution_id not in self.active_sessions:
            self.active_sessions[execution_id] = set()
            self.event_buffers[execution_id] = []

        self.active_sessions[execution_id].add(websocket)

        # 기존 버퍼링된 이벤트 전송
        for event in self.event_buffers[execution_id]:
            try:
                await websocket.send_json(event)
            except Exception as e:
                logger.error(f"Error sending buffered event: {e}")

    def disconnect(self, execution_id: str, websocket: WebSocket):
        """세션에서 WebSocket 연결 제거"""
        if execution_id in self.active_sessions:
            self.active_sessions[execution_id].discard(websocket)

    async def broadcast_event(self, execution_id: str, event: TelemetryEvent):
        """세션의 모든 구독자에게 이벤트 전송 및 DB 영속화 (최적화)"""
        event_dict = {
            "type": "telemetry",
            "execution_id": execution_id,
            "event_type": event.event_type.value,
            "agent_id": event.agent_id,
            "timestamp": event.timestamp.isoformat(),
            "content": event.content,
            "metadata": event.metadata,
        }

        # 1. DB 영속화 (최적화: THINKING_CHUNK는 저장하지 않음)
        if event.event_type.value != "thinking_chunk":
            asyncio.create_task(self._persist_event(event_dict))

        # 2. 버퍼에 추가
        if execution_id not in self.event_buffers:
            self.event_buffers[execution_id] = []

        self.event_buffers[execution_id].append(event_dict)
        if len(self.event_buffers[execution_id]) > self.buffer_limit:
            self.event_buffers[execution_id].pop(0)

        # 3. 브로드캐스트
        if execution_id in self.active_sessions:
            disconnected = set()
            for ws in self.active_sessions[execution_id]:
                try:
                    await ws.send_json(event_dict)
                except Exception:
                    disconnected.add(ws)

            for ws in disconnected:
                self.active_sessions[execution_id].discard(ws)

    async def _persist_event(self, event_dict: Dict[str, Any]):
        """이벤트를 MongoDB에 저장"""
        try:
            from database import get_mongodb_database

            db = get_mongodb_database()
            if db:
                await db.telemetry_events.insert_one(event_dict)
        except Exception as e:
            logger.debug(f"Event persistence failed (non-critical): {e!r}")

    def cleanup_session(self, execution_id: str):
        """세션 데이터 정리"""
        self.active_sessions.pop(execution_id, None)
        self.event_buffers.pop(execution_id, None)
        self.resume_events.pop(execution_id, None)
        self.modified_plans.pop(execution_id, None)


# 글로벌 인스턴스
telemetry_manager = TelemetryManager()
