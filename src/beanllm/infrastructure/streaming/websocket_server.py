"""
WebSocket Server for Real-time Streaming

책임:
- WebSocket 연결 관리
- 실시간 이벤트 브로드캐스팅
- 세션 관리

SOLID:
- SRP: WebSocket 통신만 담당
- DIP: 메시지 인터페이스에 의존
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any

try:
    from beanllm.utils.logging import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class StreamingMessage:
    """스트리밍 메시지"""

    type: str  # "progress", "result", "error", "complete"
    session_id: str
    data: Dict[str, Any]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()

    def to_json(self) -> str:
        """JSON 직렬화"""
        return json.dumps(asdict(self))


class StreamingSession:
    """
    스트리밍 세션

    단일 WebSocket 연결과 관련 작업을 관리
    """

    def __init__(self, session_id: str, websocket: WebSocketServerProtocol):
        self.session_id = session_id
        self.websocket = websocket
        self.created_at = datetime.utcnow()
        self.is_active = True
        self.metadata: Dict[str, Any] = {}

    async def send_message(self, message: StreamingMessage) -> bool:
        """
        메시지 전송

        Returns:
            성공 여부
        """
        if not self.is_active:
            return False

        try:
            await self.websocket.send(message.to_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to session {self.session_id}: {e}")
            self.is_active = False
            return False

    async def send_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """진행 상황 전송"""
        progress_data = {
            "current": current,
            "total": total,
            "percentage": (current / total * 100) if total > 0 else 0,
            "message": message,
        }
        if metadata:
            progress_data.update(metadata)

        msg = StreamingMessage(
            type="progress",
            session_id=self.session_id,
            data=progress_data,
        )
        return await self.send_message(msg)

    async def send_result(self, result: Dict[str, Any]) -> bool:
        """결과 전송"""
        msg = StreamingMessage(
            type="result",
            session_id=self.session_id,
            data=result,
        )
        return await self.send_message(msg)

    async def send_error(self, error: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """에러 전송"""
        error_data = {"error": error}
        if details:
            error_data.update(details)

        msg = StreamingMessage(
            type="error",
            session_id=self.session_id,
            data=error_data,
        )
        return await self.send_message(msg)

    async def send_complete(self, final_result: Optional[Dict[str, Any]] = None) -> bool:
        """완료 메시지 전송"""
        msg = StreamingMessage(
            type="complete",
            session_id=self.session_id,
            data=final_result or {},
        )
        return await self.send_message(msg)

    async def close(self):
        """세션 종료"""
        self.is_active = False
        try:
            await self.websocket.close()
        except Exception as e:
            logger.debug(f"Error closing websocket for session {self.session_id}: {e}")


class WebSocketServer:
    """
    WebSocket 서버

    여러 클라이언트 연결을 관리하고 메시지를 브로드캐스트
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        초기화

        Args:
            host: 서버 호스트
            port: 서버 포트
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets is required for streaming. " "Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self.sessions: Dict[str, StreamingSession] = {}
        self.server = None
        self._is_running = False

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        새로운 WebSocket 연결 처리

        Args:
            websocket: WebSocket 연결
            path: 요청 경로
        """
        # 세션 생성
        session_id = str(uuid.uuid4())
        session = StreamingSession(session_id=session_id, websocket=websocket)
        self.sessions[session_id] = session

        logger.info(f"New WebSocket connection: {session_id} (path: {path})")

        # 연결 확인 메시지 전송
        await session.send_message(
            StreamingMessage(
                type="connected",
                session_id=session_id,
                data={"message": "Connected to beanllm streaming server"},
            )
        )

        try:
            # 메시지 수신 대기
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(session, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {session_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {session_id}: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {session_id}")
        finally:
            # 세션 정리
            if session_id in self.sessions:
                del self.sessions[session_id]
            await session.close()

    async def _handle_client_message(
        self,
        session: StreamingSession,
        data: Dict[str, Any],
    ):
        """
        클라이언트 메시지 처리

        Args:
            session: 클라이언트 세션
            data: 메시지 데이터
        """
        msg_type = data.get("type")

        if msg_type == "ping":
            # Ping-pong for keepalive
            await session.send_message(
                StreamingMessage(
                    type="pong",
                    session_id=session.session_id,
                    data={},
                )
            )
        elif msg_type == "subscribe":
            # 특정 이벤트 구독
            event_type = data.get("event_type")
            if event_type:
                session.metadata["subscribed_events"] = session.metadata.get(
                    "subscribed_events", []
                ) + [event_type]
                logger.info(f"Session {session.session_id} subscribed to {event_type}")
        else:
            logger.warning(f"Unknown message type from client: {msg_type}")

    async def start(self):
        """서버 시작"""
        if self._is_running:
            logger.warning("Server is already running")
            return

        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        self._is_running = True

        logger.info(f"WebSocket server started: ws://{self.host}:{self.port}")

    async def stop(self):
        """서버 중지"""
        if not self._is_running:
            return

        logger.info("Stopping WebSocket server")

        # 모든 세션 종료
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()

        # 서버 종료
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        self._is_running = False
        logger.info("WebSocket server stopped")

    def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """세션 조회"""
        return self.sessions.get(session_id)

    async def broadcast(self, message: StreamingMessage):
        """
        모든 활성 세션에 메시지 브로드캐스트

        Args:
            message: 브로드캐스트할 메시지
        """
        tasks = []
        for session in list(self.sessions.values()):
            if session.is_active:
                tasks.append(session.send_message(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_subscribed(
        self,
        event_type: str,
        message: StreamingMessage,
    ):
        """
        특정 이벤트를 구독한 세션에만 브로드캐스트

        Args:
            event_type: 이벤트 타입
            message: 브로드캐스트할 메시지
        """
        tasks = []
        for session in list(self.sessions.values()):
            if session.is_active:
                subscribed = session.metadata.get("subscribed_events", [])
                if event_type in subscribed:
                    tasks.append(session.send_message(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_active_sessions(self) -> List[str]:
        """활성 세션 ID 목록 반환"""
        return [session_id for session_id, session in self.sessions.items() if session.is_active]

    def get_stats(self) -> Dict[str, Any]:
        """서버 통계 반환"""
        return {
            "is_running": self._is_running,
            "host": self.host,
            "port": self.port,
            "total_sessions": len(self.sessions),
            "active_sessions": len([s for s in self.sessions.values() if s.is_active]),
        }


# Singleton instance
_server_instance: Optional[WebSocketServer] = None


def get_websocket_server(
    host: str = "localhost",
    port: int = 8765,
) -> WebSocketServer:
    """
    WebSocket 서버 싱글톤 인스턴스 반환

    Args:
        host: 서버 호스트
        port: 서버 포트

    Returns:
        WebSocketServer 인스턴스
    """
    global _server_instance

    if _server_instance is None:
        _server_instance = WebSocketServer(host=host, port=port)

    return _server_instance
