"""
Client Facade 테스트 - 클라이언트 인터페이스 테스트
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

try:
    from llmkit.dto.response.chat_response import ChatResponse
    from llmkit.facade.client_facade import Client

    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Client not available")
class TestClientFacade:
    """Client Facade 테스트"""

    @pytest.fixture
    def client(self):
        """Client 인스턴스 (Handler를 Mock으로 교체)"""
        with patch("llmkit.utils.di_container.get_container") as mock_get_container:
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


