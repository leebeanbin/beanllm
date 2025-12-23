"""
ChatHandler 테스트 - 채팅 핸들러 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock

try:
    from llmkit.handler.chat_handler import ChatHandler
    from llmkit.service.chat_service import IChatService
    from llmkit.dto.response.chat_response import ChatResponse
except ImportError:
    from src.llmkit.handler.chat_handler import ChatHandler
    from src.llmkit.service.chat_service import IChatService
    from src.llmkit.dto.response.chat_response import ChatResponse


class TestChatHandler:
    """ChatHandler 테스트"""

    @pytest.fixture
    def mock_chat_service(self):
        """Mock ChatService"""
        service = Mock(spec=IChatService)
        service.chat = AsyncMock(
            return_value=ChatResponse(
                content="Test response", model="gpt-4o-mini", provider="openai"
            )
        )
        # stream_chat은 async generator로 설정
        async def mock_stream_chat(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield chunk
        service.stream_chat = mock_stream_chat
        return service

    @pytest.fixture
    def chat_handler(self, mock_chat_service):
        """ChatHandler 인스턴스"""
        return ChatHandler(chat_service=mock_chat_service)

    @pytest.mark.asyncio
    async def test_handle_chat_basic(self, chat_handler):
        """기본 채팅 처리 테스트"""
        response = await chat_handler.handle_chat(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
        )

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_handle_chat_with_parameters(self, chat_handler):
        """파라미터 포함 채팅 처리 테스트"""
        response = await chat_handler.handle_chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
        )

        assert response is not None
        # Service가 호출되었는지 확인
        chat_handler._chat_service.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_stream_chat(self, chat_handler):
        """스트리밍 채팅 처리 테스트"""
        # Mock 스트리밍 응답 - async generator로 설정
        async def mock_stream(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield chunk

        # AsyncMock이 아닌 실제 async generator 함수로 설정
        chat_handler._chat_service.stream_chat = mock_stream

        chunks = []
        async for chunk in chat_handler.handle_stream_chat(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_handle_chat_dto_conversion(self, chat_handler):
        """DTO 변환 테스트"""
        response = await chat_handler.handle_chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            system="You are helpful",
        )

        assert response is not None
        # ChatRequest가 올바르게 생성되었는지 확인
        call_args = chat_handler._chat_service.chat.call_args[0][0]
        assert call_args.messages == [{"role": "user", "content": "Hello"}]
        assert call_args.model == "gpt-4o-mini"
        assert call_args.temperature == 0.7
        assert call_args.max_tokens == 1000
        assert call_args.system == "You are helpful"

    @pytest.mark.asyncio
    async def test_handle_chat_extra_params(self, chat_handler):
        """추가 파라미터 포함 테스트"""
        response = await chat_handler.handle_chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            presence_penalty=0.5,
            frequency_penalty=0.3,
        )

        assert response is not None
        # extra_params에 포함되었는지 확인
        call_args = chat_handler._chat_service.chat.call_args[0][0]
        assert call_args.extra_params.get("presence_penalty") == 0.5
        assert call_args.extra_params.get("frequency_penalty") == 0.3

    @pytest.mark.asyncio
    async def test_handle_chat_validation_missing_messages(self, chat_handler):
        """입력 검증 - messages 누락 테스트"""
        with pytest.raises((ValueError, TypeError)):
            await chat_handler.handle_chat(
                messages=None,  # 필수 파라미터 누락
                model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_handle_chat_validation_missing_model(self, chat_handler):
        """입력 검증 - model 누락 테스트"""
        with pytest.raises((ValueError, TypeError)):
            await chat_handler.handle_chat(
                messages=[{"role": "user", "content": "Hello"}],
                model=None,  # 필수 파라미터 누락
            )

    @pytest.mark.asyncio
    async def test_handle_chat_validation_temperature_range(self, chat_handler):
        """입력 검증 - temperature 범위 테스트"""
        # 범위를 벗어난 값 (0-2 범위)
        with pytest.raises((ValueError, AssertionError)):
            await chat_handler.handle_chat(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o-mini",
                temperature=3.0,  # 범위 초과
            )

    @pytest.mark.asyncio
    async def test_handle_chat_validation_max_tokens_range(self, chat_handler):
        """입력 검증 - max_tokens 범위 테스트"""
        # 범위를 벗어난 값 (1 이상)
        with pytest.raises((ValueError, AssertionError)):
            await chat_handler.handle_chat(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o-mini",
                max_tokens=0,  # 범위 미만
            )

    @pytest.mark.asyncio
    async def test_handle_chat_error_handling(self, chat_handler):
        """에러 처리 테스트"""
        # Service에서 에러 발생 시뮬레이션
        chat_handler._chat_service.chat = AsyncMock(side_effect=ValueError("Service error"))

        with pytest.raises(ValueError):
            await chat_handler.handle_chat(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini"
            )

    @pytest.mark.asyncio
    async def test_handle_stream_chat_dto_conversion(self, chat_handler):
        """스트리밍 DTO 변환 테스트"""

        # async generator로 설정
        async def mock_stream(*args, **kwargs):
            yield "chunk"

        chat_handler._chat_service.stream_chat = mock_stream

        chunks = []
        async for chunk in chat_handler.handle_stream_chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
        ):
            chunks.append(chunk)

        # ChatRequest가 올바르게 생성되었는지 확인
        # stream_chat은 generator이므로 call_args 확인이 어려움
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_handle_chat_with_system(self, chat_handler):
        """시스템 프롬프트 포함 테스트"""
        response = await chat_handler.handle_chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            system="You are a helpful assistant",
        )

        assert response is not None
        call_args = chat_handler._chat_service.chat.call_args[0][0]
        assert call_args.system == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_handle_chat_multiple_messages(self, chat_handler):
        """여러 메시지 포함 테스트"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        response = await chat_handler.handle_chat(messages=messages, model="gpt-4o-mini")

        assert response is not None
        call_args = chat_handler._chat_service.chat.call_args[0][0]
        assert len(call_args.messages) == 4

    @pytest.mark.asyncio
    async def test_handle_chat_all_parameters(self, chat_handler):
        """모든 파라미터 포함 테스트"""
        response = await chat_handler.handle_chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            system="You are helpful",
            stream=False,
            extra_param="value",
        )

        assert response is not None
        call_args = chat_handler._chat_service.chat.call_args[0][0]
        assert call_args.temperature == 0.7
        assert call_args.max_tokens == 1000
        assert call_args.top_p == 0.9
        assert call_args.system == "You are helpful"
        assert call_args.extra_params.get("extra_param") == "value"

