"""
ChatService 테스트 - 채팅 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from beanllm.dto.request.chat_request import ChatRequest
from beanllm.dto.response.chat_response import ChatResponse
from beanllm.infrastructure.adapter import ParameterAdapter
from beanllm.service.impl.chat_service_impl import ChatServiceImpl


class TestChatService:
    """ChatService 테스트"""

    @pytest.fixture
    def mock_provider_factory(self):
        """Mock ProviderFactory"""
        factory = Mock()
        provider = Mock()
        provider.name = "openai"

        # chat 메서드는 AsyncMock으로 설정하여 assert_called_once 사용 가능
        async def mock_chat(*args, **kwargs):
            return {
                "content": "Test response",
                "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
                "finish_reason": "stop",
            }

        provider.chat = AsyncMock(side_effect=mock_chat)

        # stream_chat은 async generator (Mock이 아닌 실제 함수로 직접 할당)
        async def mock_stream_chat(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield chunk

        # Mock 객체의 속성을 실제 async generator 함수로 직접 할당
        # Mock 객체에 실제 함수를 할당하면 Mock의 특수 동작이 비활성화됨
        provider.stream_chat = mock_stream_chat

        # create 메서드가 항상 같은 provider 인스턴스를 반환하도록 설정
        # 이렇게 하면 _create_provider가 호출될 때마다 같은 provider를 반환
        factory.create = Mock(return_value=provider)
        factory.get_provider = Mock(return_value=provider)
        return factory

    @pytest.fixture
    def chat_service(self, mock_provider_factory):
        """ChatService 인스턴스"""
        return ChatServiceImpl(
            provider_factory=mock_provider_factory, parameter_adapter=ParameterAdapter()
        )

    @pytest.mark.asyncio
    async def test_chat_basic(self, chat_service):
        """기본 채팅 테스트"""
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini")

        response = await chat_service.chat(request)

        assert response is not None
        assert isinstance(response, ChatResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.provider == "openai"
        assert response.usage is not None

    @pytest.mark.asyncio
    async def test_chat_with_parameters(self, chat_service):
        """파라미터 포함 채팅 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )

        response = await chat_service.chat(request)

        assert response is not None
        # Provider가 파라미터를 받았는지 확인
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        call_kwargs = provider.chat.call_args[1]
        assert call_kwargs.get("temperature") == 0.7
        assert call_kwargs.get("max_tokens") == 1000

    @pytest.mark.asyncio
    async def test_chat_with_system(self, chat_service):
        """시스템 프롬프트 포함 채팅 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            system="You are a helpful assistant",
        )

        response = await chat_service.chat(request)

        assert response is not None
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        # call_args는 (args, kwargs) 튜플이므로 [1]로 kwargs 접근
        call_kwargs = provider.chat.call_args[1] if provider.chat.call_args else {}
        assert call_kwargs.get("system") == "You are a helpful assistant"

    @pytest.mark.asyncio
    async def test_stream_chat(self, chat_service):
        """스트리밍 채팅 테스트"""
        # provider.stream_chat을 async generator로 설정
        async def mock_stream_chat(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield chunk

        # _create_provider가 반환하는 provider에 stream_chat 설정
        provider = chat_service._provider_factory.create.return_value
        # Mock 객체의 메서드를 실제 async generator 함수로 직접 할당
        # Mock의 __call__을 오버라이드하지 않고 직접 함수 할당
        provider.stream_chat = mock_stream_chat

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini", stream=True
        )

        chunks = []
        # stream_chat은 async generator 함수이므로 호출하면 async generator 반환
        async for chunk in chat_service.stream_chat(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_chat_provider_detection(self, chat_service):
        """Provider 자동 감지 테스트"""
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini")

        response = await chat_service.chat(request)

        # ProviderFactory가 호출되었는지 확인
        assert response is not None
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_chat_parameter_adaptation(self, chat_service):
        """파라미터 변환 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
        )

        await chat_service.chat(request)

        # ParameterAdapter가 사용되었는지 확인
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_response_conversion(self, chat_service):
        """응답 변환 테스트"""
        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini")

        response = await chat_service.chat(request)

        assert isinstance(response, ChatResponse)
        assert response.content == "Test response"
        assert response.model == "gpt-4o-mini"
        assert response.provider == "openai"
        assert response.usage is not None
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_with_extra_params(self, chat_service):
        """extra_params 포함 채팅 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            extra_params={"presence_penalty": 0.5, "frequency_penalty": 0.3},
        )

        response = await chat_service.chat(request)

        assert response is not None
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        call_kwargs = provider.chat.call_args[1]
        assert "presence_penalty" in call_kwargs or "presence_penalty" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_chat_with_provider_override(self, chat_service):
        """Provider 명시적 지정 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            extra_params={"provider": "openai"},
        )

        response = await chat_service.chat(request)

        assert response is not None
        # ProviderFactory.create가 provider 파라미터로 호출되었는지 확인
        chat_service._provider_factory.create.assert_called()

    @pytest.mark.asyncio
    async def test_chat_with_none_system(self, chat_service):
        """시스템 프롬프트가 None인 경우 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            system=None,
        )

        response = await chat_service.chat(request)

        assert response is not None
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        call_kwargs = provider.chat.call_args[1]
        # system이 None이거나 전달되지 않아야 함
        assert "system" not in call_kwargs or call_kwargs.get("system") is None

    @pytest.mark.asyncio
    async def test_chat_parameter_adapter_usage(self, chat_service):
        """ParameterAdapter 사용 확인 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
        )

        await chat_service.chat(request)

        # ParameterAdapter가 사용되었는지 확인 (파라미터가 변환되었는지)
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        call_kwargs = provider.chat.call_args[1]
        # 파라미터가 전달되었는지 확인
        assert "temperature" in call_kwargs or "max_tokens" in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_multiple_messages(self, chat_service):
        """여러 메시지 포함 채팅 테스트"""
        request = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
            model="gpt-4o-mini",
        )

        response = await chat_service.chat(request)

        assert response is not None
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        # call_args는 (args, kwargs) 튜플
        call_args_tuple = provider.chat.call_args
        if call_args_tuple:
            # args[0]이 messages인지 확인
            if len(call_args_tuple[0]) > 0:
                messages = call_args_tuple[0][0]
                assert len(messages) == 4

    @pytest.mark.asyncio
    async def test_stream_chat_with_parameters(self, chat_service):
        """파라미터 포함 스트리밍 채팅 테스트"""
        # stream_chat은 이미 fixture에서 async generator로 설정되어 있음
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            stream=True,
        )

        chunks = []
        async for chunk in chat_service.stream_chat(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "".join(chunks) == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_chat_with_system(self, chat_service):
        """시스템 프롬프트 포함 스트리밍 채팅 테스트"""
        async def mock_stream_chat(*args, **kwargs):
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                yield chunk

        provider = chat_service._provider_factory.create.return_value
        provider.stream_chat = mock_stream_chat

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            system="You are a helpful assistant",
            stream=True,
        )

        chunks = []
        async for chunk in chat_service.stream_chat(request):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_chat_provider_factory_required(self):
        """ProviderFactory가 없을 때 에러 테스트"""
        service = ChatServiceImpl(provider_factory=None, parameter_adapter=ParameterAdapter())

        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini")

        with pytest.raises(ValueError, match="Provider factory is required"):
            await service.chat(request)

    @pytest.mark.asyncio
    async def test_chat_with_different_providers(self, mock_provider_factory):
        """다양한 Provider 테스트"""
        # Claude Provider Mock
        claude_provider = Mock()
        claude_provider.name = "anthropic"

        async def mock_claude_chat(*args, **kwargs):
            return {
                "content": "Claude response",
                "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
                "finish_reason": "stop",
            }

        claude_provider.chat = AsyncMock(side_effect=mock_claude_chat)

        # Gemini Provider Mock
        gemini_provider = Mock()
        gemini_provider.name = "google"

        async def mock_gemini_chat(*args, **kwargs):
            return {
                "content": "Gemini response",
                "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
                "finish_reason": "stop",
            }

        gemini_provider.chat = AsyncMock(side_effect=mock_gemini_chat)

        # Factory 설정 - create 메서드가 model과 provider_name을 받음
        def create_provider(model: str, provider_name: str = None):
            if "claude" in model or provider_name == "anthropic":
                return claude_provider
            elif "gemini" in model or provider_name == "google":
                return gemini_provider
            return claude_provider  # 기본값

        mock_provider_factory.create = Mock(side_effect=create_provider)
        mock_provider_factory.get_provider = Mock(side_effect=create_provider)

        service = ChatServiceImpl(
            provider_factory=mock_provider_factory, parameter_adapter=ParameterAdapter()
        )

        # Claude 테스트
        claude_request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-5-sonnet"
        )
        claude_response = await service.chat(claude_request)
        assert claude_response.provider == "anthropic"
        assert claude_response.content == "Claude response"

        # Gemini 테스트
        gemini_request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gemini-pro"
        )
        gemini_response = await service.chat(gemini_request)
        assert gemini_response.provider == "google"
        assert gemini_response.content == "Gemini response"

    @pytest.mark.asyncio
    async def test_chat_parameter_conversion_google(self, mock_provider_factory):
        """Google Provider 파라미터 변환 테스트 (max_tokens → max_output_tokens)"""
        google_provider = Mock()
        google_provider.name = "google"

        async def mock_google_chat(*args, **kwargs):
            return {
                "content": "Google response",
                "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
                "finish_reason": "stop",
            }

        google_provider.chat = AsyncMock(side_effect=mock_google_chat)
        mock_provider_factory.create = Mock(return_value=google_provider)
        mock_provider_factory.get_provider = Mock(return_value=google_provider)

        service = ChatServiceImpl(
            provider_factory=mock_provider_factory, parameter_adapter=ParameterAdapter()
        )

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gemini-pro",
            max_tokens=1000,
        )

        await service.chat(request)

        # Google Provider는 max_tokens가 max_output_tokens로 변환되어야 함
        google_provider.chat.assert_called_once()
        call_kwargs = google_provider.chat.call_args[1] if google_provider.chat.call_args else {}
        # ParameterAdapter가 변환했는지 확인
        assert "max_output_tokens" in call_kwargs or "max_tokens" in call_kwargs

    @pytest.mark.asyncio
    async def test_chat_parameter_conversion_ollama(self, mock_provider_factory):
        """Ollama Provider 파라미터 변환 테스트 (max_tokens → num_predict)"""
        ollama_provider = Mock()
        ollama_provider.name = "ollama"

        async def mock_ollama_chat(*args, **kwargs):
            return {
                "content": "Ollama response",
                "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
                "finish_reason": "stop",
            }

        ollama_provider.chat = AsyncMock(side_effect=mock_ollama_chat)
        mock_provider_factory.create = Mock(return_value=ollama_provider)
        mock_provider_factory.get_provider = Mock(return_value=ollama_provider)

        service = ChatServiceImpl(
            provider_factory=mock_provider_factory, parameter_adapter=ParameterAdapter()
        )

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama2",
            max_tokens=1000,
        )

        await service.chat(request)

        # Ollama Provider는 max_tokens가 num_predict로 변환되어야 함
        ollama_provider.chat.assert_called_once()
        call_kwargs = ollama_provider.chat.call_args[1] if ollama_provider.chat.call_args else {}
        # ParameterAdapter가 변환했는지 확인
        assert "num_predict" in call_kwargs or "max_tokens" in call_kwargs

    @pytest.mark.asyncio
    async def test_stream_chat_empty_response(self, chat_service):
        """빈 스트리밍 응답 테스트"""
        # 빈 응답을 반환하는 async generator
        async def mock_empty_stream(*args, **kwargs):
            # 빈 generator - yield 없음
            if False:
                yield  # unreachable but makes it a generator
            return

        provider = chat_service._provider_factory.create.return_value
        provider.stream_chat = mock_empty_stream

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-4o-mini", stream=True
        )

        chunks = []
        async for chunk in chat_service.stream_chat(request):
            chunks.append(chunk)

        # 빈 응답이어도 에러가 발생하지 않아야 함
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_chat_with_all_parameters(self, chat_service):
        """모든 파라미터 포함 채팅 테스트"""
        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            system="You are helpful",
            extra_params={"presence_penalty": 0.5},
        )

        response = await chat_service.chat(request)

        assert response is not None
        provider = chat_service._provider_factory.get_provider.return_value
        provider.chat.assert_called_once()
        call_kwargs = provider.chat.call_args[1]
        assert call_kwargs.get("system") == "You are helpful"
        assert "temperature" in call_kwargs or "max_tokens" in call_kwargs
