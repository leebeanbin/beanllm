"""
Tests for infrastructure/streaming/websocket_server.py

Mocks the websockets library and tests StreamingMessage, StreamingSession,
WebSocketServer, and the singleton helper.
"""

import asyncio
import json
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch websockets before importing the module under test
# ---------------------------------------------------------------------------


def _make_websockets_mock():
    mock = MagicMock()
    mock.serve = AsyncMock()
    mock.exceptions = MagicMock()
    mock.exceptions.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
    mock.legacy = MagicMock()
    mock.legacy.server = MagicMock()
    mock.legacy.server.WebSocketServerProtocol = object
    return mock


_WS_MOCK = _make_websockets_mock()


@pytest.fixture(autouse=True)
def patch_websockets():
    sys.modules["websockets"] = _WS_MOCK
    sys.modules["websockets.legacy"] = _WS_MOCK.legacy
    sys.modules["websockets.legacy.server"] = _WS_MOCK.legacy.server
    # Force reimport of the module to pick up mock
    import importlib

    import beanllm.infrastructure.streaming.websocket_server as ws_mod

    ws_mod.WEBSOCKETS_AVAILABLE = True
    ws_mod.websockets = _WS_MOCK
    yield


# Import after patching
from beanllm.infrastructure.streaming.websocket_server import (
    StreamingMessage,
    StreamingSession,
    WebSocketServer,
    get_websocket_server,
)

# ---------------------------------------------------------------------------
# StreamingMessage tests
# ---------------------------------------------------------------------------


class TestStreamingMessage:
    def test_creates_with_required_fields(self):
        msg = StreamingMessage(type="progress", session_id="abc", data={"key": "val"})
        assert msg.type == "progress"
        assert msg.session_id == "abc"
        assert msg.data == {"key": "val"}

    def test_timestamp_auto_set_on_creation(self):
        msg = StreamingMessage(type="result", session_id="s1", data={})
        assert msg.timestamp is not None

    def test_to_json_returns_valid_json(self):
        msg = StreamingMessage(type="error", session_id="s2", data={"error": "oops"})
        json_str = msg.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "error"
        assert parsed["session_id"] == "s2"
        assert parsed["data"]["error"] == "oops"

    def test_to_json_includes_timestamp(self):
        msg = StreamingMessage(type="complete", session_id="s3", data={})
        parsed = json.loads(msg.to_json())
        assert "timestamp" in parsed

    def test_custom_timestamp_preserved(self):
        ts = "2025-01-01T00:00:00"
        msg = StreamingMessage(type="pong", session_id="s4", data={}, timestamp=ts)
        assert msg.timestamp == ts


# ---------------------------------------------------------------------------
# StreamingSession tests
# ---------------------------------------------------------------------------


def make_mock_websocket():
    ws = MagicMock()
    ws.send = AsyncMock(return_value=None)
    ws.close = AsyncMock(return_value=None)
    return ws


class TestStreamingSession:
    def test_init_sets_fields(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="sess-1", websocket=ws)
        assert session.session_id == "sess-1"
        assert session.is_active is True
        assert session.metadata == {}

    async def test_send_message_returns_true_on_success(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        msg = StreamingMessage(type="test", session_id="s1", data={})
        result = await session.send_message(msg)
        assert result is True
        ws.send.assert_called_once()

    async def test_send_message_returns_false_when_inactive(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        session.is_active = False
        msg = StreamingMessage(type="test", session_id="s1", data={})
        result = await session.send_message(msg)
        assert result is False
        ws.send.assert_not_called()

    async def test_send_message_marks_inactive_on_exception(self):
        ws = make_mock_websocket()
        ws.send.side_effect = Exception("Connection lost")
        session = StreamingSession(session_id="s1", websocket=ws)
        msg = StreamingMessage(type="test", session_id="s1", data={})
        result = await session.send_message(msg)
        assert result is False
        assert session.is_active is False

    async def test_send_progress(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_progress(current=5, total=10, message="half done")
        assert result is True
        # Verify JSON includes progress data
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["type"] == "progress"
        assert parsed["data"]["current"] == 5
        assert parsed["data"]["total"] == 10
        assert parsed["data"]["percentage"] == 50.0

    async def test_send_progress_with_metadata(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_progress(current=3, total=10, metadata={"extra": "info"})
        assert result is True
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["data"]["extra"] == "info"

    async def test_send_progress_zero_total(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_progress(current=0, total=0)
        assert result is True
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["data"]["percentage"] == 0

    async def test_send_result(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_result({"answer": "42"})
        assert result is True
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["type"] == "result"
        assert parsed["data"]["answer"] == "42"

    async def test_send_error(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_error("Something failed")
        assert result is True
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["type"] == "error"
        assert parsed["data"]["error"] == "Something failed"

    async def test_send_error_with_details(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_error("fail", details={"code": 503})
        assert result is True
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["data"]["code"] == 503

    async def test_send_complete(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_complete(final_result={"status": "done"})
        assert result is True
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["type"] == "complete"
        assert parsed["data"]["status"] == "done"

    async def test_send_complete_no_result(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        result = await session.send_complete()
        assert result is True

    async def test_close_sets_inactive(self):
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        await session.close()
        assert session.is_active is False
        ws.close.assert_called_once()

    async def test_close_handles_websocket_exception(self):
        ws = make_mock_websocket()
        ws.close.side_effect = Exception("already closed")
        session = StreamingSession(session_id="s1", websocket=ws)
        await session.close()  # Should not raise
        assert session.is_active is False


# ---------------------------------------------------------------------------
# WebSocketServer tests
# ---------------------------------------------------------------------------


class TestWebSocketServer:
    def test_init_sets_host_and_port(self):
        server = WebSocketServer(host="0.0.0.0", port=9000)
        assert server.host == "0.0.0.0"
        assert server.port == 9000

    def test_init_empty_sessions(self):
        server = WebSocketServer()
        assert server.sessions == {}

    def test_init_not_running(self):
        server = WebSocketServer()
        assert server._is_running is False

    def test_get_stats_initial(self):
        server = WebSocketServer()
        stats = server.get_stats()
        assert stats["is_running"] is False
        assert stats["total_sessions"] == 0
        assert stats["active_sessions"] == 0

    def test_get_active_sessions_empty(self):
        server = WebSocketServer()
        assert server.get_active_sessions() == []

    def test_get_session_returns_none_for_unknown(self):
        server = WebSocketServer()
        assert server.get_session("nonexistent") is None

    def test_get_session_returns_session(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        server.sessions["s1"] = session
        assert server.get_session("s1") is session

    async def test_broadcast_sends_to_all_active(self):
        server = WebSocketServer()
        ws1 = make_mock_websocket()
        ws2 = make_mock_websocket()
        ws3 = make_mock_websocket()
        s1 = StreamingSession(session_id="s1", websocket=ws1)
        s2 = StreamingSession(session_id="s2", websocket=ws2)
        s3 = StreamingSession(session_id="s3", websocket=ws3)
        s3.is_active = False

        server.sessions = {"s1": s1, "s2": s2, "s3": s3}
        msg = StreamingMessage(type="event", session_id="broadcast", data={"x": 1})
        await server.broadcast(msg)

        ws1.send.assert_called_once()
        ws2.send.assert_called_once()
        ws3.send.assert_not_called()

    async def test_broadcast_empty_sessions_does_not_raise(self):
        server = WebSocketServer()
        msg = StreamingMessage(type="event", session_id="none", data={})
        await server.broadcast(msg)  # Should not raise

    async def test_broadcast_to_subscribed(self):
        server = WebSocketServer()
        ws1 = make_mock_websocket()
        ws2 = make_mock_websocket()
        s1 = StreamingSession(session_id="s1", websocket=ws1)
        s2 = StreamingSession(session_id="s2", websocket=ws2)
        s1.metadata["subscribed_events"] = ["kg.build"]
        # s2 has no subscriptions

        server.sessions = {"s1": s1, "s2": s2}
        msg = StreamingMessage(type="kg.build", session_id="broadcast", data={})
        await server.broadcast_to_subscribed("kg.build", msg)

        ws1.send.assert_called_once()
        ws2.send.assert_not_called()

    async def test_broadcast_to_subscribed_no_match(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        s = StreamingSession(session_id="s1", websocket=ws)
        s.metadata["subscribed_events"] = ["other.event"]
        server.sessions = {"s1": s}

        msg = StreamingMessage(type="kg.build", session_id="b", data={})
        await server.broadcast_to_subscribed("kg.build", msg)
        ws.send.assert_not_called()

    def test_get_active_sessions_list(self):
        server = WebSocketServer()
        ws1 = make_mock_websocket()
        ws2 = make_mock_websocket()
        s1 = StreamingSession(session_id="s1", websocket=ws1)
        s2 = StreamingSession(session_id="s2", websocket=ws2)
        s2.is_active = False
        server.sessions = {"s1": s1, "s2": s2}

        active = server.get_active_sessions()
        assert "s1" in active
        assert "s2" not in active

    async def test_start_sets_is_running(self):
        server = WebSocketServer()
        mock_server = MagicMock()
        _WS_MOCK.serve.return_value = mock_server
        await server.start()
        assert server._is_running is True

    async def test_start_twice_logs_warning(self):
        server = WebSocketServer()
        mock_server = MagicMock()
        _WS_MOCK.serve.return_value = mock_server
        await server.start()
        _WS_MOCK.serve.reset_mock()
        await server.start()
        # Should not call serve again
        _WS_MOCK.serve.assert_not_called()

    async def test_stop_clears_sessions(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        s = StreamingSession(session_id="s1", websocket=ws)
        server.sessions = {"s1": s}
        server._is_running = True
        mock_server = MagicMock()
        mock_server.close = MagicMock()
        mock_server.wait_closed = AsyncMock()
        server.server = mock_server

        await server.stop()

        assert len(server.sessions) == 0
        assert server._is_running is False

    async def test_stop_when_not_running(self):
        server = WebSocketServer()
        await server.stop()  # Should not raise

    async def test_get_stats_with_sessions(self):
        server = WebSocketServer()
        ws1 = make_mock_websocket()
        ws2 = make_mock_websocket()
        s1 = StreamingSession(session_id="s1", websocket=ws1)
        s2 = StreamingSession(session_id="s2", websocket=ws2)
        s2.is_active = False
        server.sessions = {"s1": s1, "s2": s2}
        server._is_running = True

        stats = server.get_stats()
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 1
        assert stats["is_running"] is True


# ---------------------------------------------------------------------------
# _handle_client_message tests
# ---------------------------------------------------------------------------


class TestHandleClientMessage:
    async def test_ping_sends_pong(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        await server._handle_client_message(session, {"type": "ping"})
        sent_str = ws.send.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["type"] == "pong"

    async def test_subscribe_adds_event_to_metadata(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        await server._handle_client_message(
            session, {"type": "subscribe", "event_type": "kg.build"}
        )
        assert "kg.build" in session.metadata.get("subscribed_events", [])

    async def test_subscribe_appends_multiple_events(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        await server._handle_client_message(session, {"type": "subscribe", "event_type": "event.a"})
        await server._handle_client_message(session, {"type": "subscribe", "event_type": "event.b"})
        subscribed = session.metadata.get("subscribed_events", [])
        assert "event.a" in subscribed
        assert "event.b" in subscribed

    async def test_unknown_message_type_does_not_raise(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        await server._handle_client_message(session, {"type": "unknown_cmd"})

    async def test_subscribe_without_event_type_does_not_raise(self):
        server = WebSocketServer()
        ws = make_mock_websocket()
        session = StreamingSession(session_id="s1", websocket=ws)
        await server._handle_client_message(session, {"type": "subscribe"})


# ---------------------------------------------------------------------------
# get_websocket_server singleton
# ---------------------------------------------------------------------------


class TestGetWebSocketServer:
    def test_returns_websocket_server_instance(self):
        import beanllm.infrastructure.streaming.websocket_server as ws_mod

        # Reset singleton
        ws_mod._server_instance = None
        server = get_websocket_server()
        assert isinstance(server, WebSocketServer)

    def test_returns_same_instance_on_second_call(self):
        import beanllm.infrastructure.streaming.websocket_server as ws_mod

        ws_mod._server_instance = None
        server1 = get_websocket_server()
        server2 = get_websocket_server()
        assert server1 is server2

    def test_custom_host_and_port(self):
        import beanllm.infrastructure.streaming.websocket_server as ws_mod

        ws_mod._server_instance = None
        server = get_websocket_server(host="0.0.0.0", port=9999)
        assert server.host == "0.0.0.0"
        assert server.port == 9999
        ws_mod._server_instance = None  # cleanup
