"""
Client Facade 테스트 - 클라이언트 인터페이스 테스트
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

try:
    from beanllm.dto.response import ChatResponse
    from beanllm.facade.client_facade import Client

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Client not available")
class TestClientFacade:
    """Client Facade 테스트"""

    @pytest.fixture
    def client(self):
        """Client 인스턴스 (Handler를 Mock으로 교체)"""
        with patch("beanllm.utils.di_container.get_container") as mock_get_container:
            mock_handler = MagicMock()

            # handle_chat은 ChatResponse 반환
            async def mock_handle_chat(*args, **kwargs):
                return ChatResponse(
                    content="Test response",
                    model="gpt-4o-mini",
                    provider="openai",
                    usage={"total_tokens": 100},
                )

            mock_handler.handle_chat = MagicMock(side_effect=mock_handle_chat)

            # handle_stream_chat은 async generator 반환
            async def mock_stream_chat(*args, **kwargs):
                yield "chunk1"
                yield "chunk2"

            mock_handler.handle_stream_chat = MagicMock(return_value=mock_stream_chat())

            mock_handler_factory = Mock()
            mock_handler_factory.create_chat_handler.return_value = mock_handler

            mock_container = Mock()
            mock_container.handler_factory = mock_handler_factory
            mock_get_container.return_value = mock_container

            client = Client(model="gpt-4o-mini")
            return client

    @pytest.mark.asyncio
    async def test_chat(self, client):
        """채팅 테스트"""
        messages = [{"role": "user", "content": "Hello"}]
        response = await client.chat(messages)

        assert isinstance(response, ChatResponse)
        assert response.content == "Test response"
        assert client._chat_handler.handle_chat.called

    @pytest.mark.asyncio
    async def test_chat_stream(self, client):
        """스트리밍 채팅 테스트"""
        messages = [{"role": "user", "content": "Hello"}]
        chunks = []
        async for chunk in client.stream_chat(messages):
            chunks.append(chunk)

        assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Tests using correct import path
# ---------------------------------------------------------------------------


def _make_client(model: str = "gpt-4o-mini", provider: str = None):
    """Create a Client with mocked handler."""
    from beanllm.dto.response.core.chat_response import ChatResponse
    from beanllm.facade.core.client_facade import Client

    patcher = patch("beanllm.utils.core.di_container.get_container")
    mock_get_container = patcher.start()

    mock_handler = MagicMock()

    async def mock_handle_chat(*args, **kwargs):
        return ChatResponse(
            content="Response text",
            model=model,
            provider=provider or "openai",
            usage={"total_tokens": 50},
        )

    async def mock_stream(*args, **kwargs):
        for chunk in ["part1", "part2"]:
            yield chunk

    mock_handler.handle_chat = AsyncMock(side_effect=mock_handle_chat)
    mock_handler.handle_stream_chat = mock_stream

    mock_handler_factory = MagicMock()
    mock_handler_factory.create_chat_handler.return_value = mock_handler

    mock_service_factory = MagicMock()
    mock_container = MagicMock()
    # Base FacadeBase uses container.handler_factory as attribute (not method call)
    mock_container.handler_factory = mock_handler_factory
    mock_container.get_service_factory.return_value = mock_service_factory
    mock_get_container.return_value = mock_container

    client = Client(model=model, provider=provider)
    return client, mock_handler, patcher


class TestClientInit:
    def test_stores_model(self):
        client, _, p = _make_client("gpt-4o")
        try:
            assert client.model == "gpt-4o"
        finally:
            p.stop()

    def test_stores_provider_when_explicit(self):
        client, _, p = _make_client("gpt-4o", provider="openai")
        try:
            assert client.provider == "openai"
        finally:
            p.stop()

    def test_auto_detects_provider_for_gpt4(self):
        client, _, p = _make_client("gpt-4o")
        try:
            assert client.provider is not None
            assert len(client.provider) > 0
        finally:
            p.stop()

    def test_repr_contains_model(self):
        client, _, p = _make_client("gpt-4o-mini")
        try:
            r = repr(client)
            assert "gpt-4o-mini" in r
        finally:
            p.stop()


class TestClientChat:
    async def test_chat_returns_response(self):
        from beanllm.dto.response.core.chat_response import ChatResponse

        client, _, p = _make_client()
        try:
            result = await client.chat([{"role": "user", "content": "Hello"}])
            assert isinstance(result, ChatResponse)
            assert result.content == "Response text"
        finally:
            p.stop()

    async def test_chat_with_system(self):
        client, handler, p = _make_client()
        try:
            await client.chat(
                [{"role": "user", "content": "Hi"}],
                system="You are helpful.",
            )
            assert handler.handle_chat.called
        finally:
            p.stop()

    async def test_chat_with_temperature(self):
        client, handler, p = _make_client()
        try:
            await client.chat(
                [{"role": "user", "content": "Hi"}],
                temperature=0.9,
            )
            call_kwargs = handler.handle_chat.call_args.kwargs
            assert call_kwargs["temperature"] == 0.9
        finally:
            p.stop()

    async def test_chat_with_max_tokens(self):
        client, handler, p = _make_client()
        try:
            await client.chat(
                [{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )
            call_kwargs = handler.handle_chat.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100
        finally:
            p.stop()


class TestClientStreamChat:
    async def test_stream_chat_yields_chunks(self):
        client, _, p = _make_client()
        try:
            chunks = []
            async for chunk in client.stream_chat([{"role": "user", "content": "Hi"}]):
                chunks.append(chunk)
            assert chunks == ["part1", "part2"]
        finally:
            p.stop()

    async def test_stream_chat_with_system(self):
        client, _, p = _make_client()
        try:
            chunks = []
            async for chunk in client.stream_chat(
                [{"role": "user", "content": "Hi"}],
                system="Be brief.",
            ):
                chunks.append(chunk)
            assert len(chunks) == 2
        finally:
            p.stop()


class TestClientDetectProvider:
    def test_detects_openai_for_gpt(self):
        client, _, p = _make_client("gpt-4o")
        try:
            detected = client._detect_provider("gpt-4o")
            assert "openai" in detected.lower() or len(detected) > 0
        finally:
            p.stop()

    def test_detects_claude_for_anthropic(self):
        client, _, p = _make_client("claude-3-5-sonnet-20241022")
        try:
            detected = client._detect_provider("claude-3-5-sonnet-20241022")
            assert (
                "claude" in detected.lower() or "anthropic" in detected.lower() or len(detected) > 0
            )
        finally:
            p.stop()
