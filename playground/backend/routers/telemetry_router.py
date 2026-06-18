import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from telemetry_manager import telemetry_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.websocket("/ws/{execution_id}")
async def telemetry_websocket(websocket: WebSocket, execution_id: str):
    """
    실시간 텔레메트리 구독을 위한 WebSocket 엔드포인트
    """
    await telemetry_manager.connect(execution_id, websocket)
    try:
        while True:
            # 클라이언트로부터의 메시지 수신 (주로 하트비트)
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        telemetry_manager.disconnect(execution_id, websocket)
    except Exception as e:
        logger.error(f"Telemetry WS error for {execution_id}: {e}")
        telemetry_manager.disconnect(execution_id, websocket)
