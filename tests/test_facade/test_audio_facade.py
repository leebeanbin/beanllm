"""
Audio Facade 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from llmkit.facade.audio_facade import WhisperSTT, TextToSpeech, AudioRAG
    from llmkit.domain.audio.types import TranscriptionResult, AudioSegment
    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Audio Facade not available")
class TestWhisperSTT:
    @pytest.fixture
    def whisper_stt(self):
        with patch('llmkit.facade.audio_facade.HandlerFactory') as mock_factory:
            mock_handler = MagicMock()
            mock_result = TranscriptionResult(
                text="Test transcription",
                language="en",
                segments=[]
            )
            async def mock_handle_transcribe(*args, **kwargs):
                return mock_result
            mock_handler.handle_transcribe = MagicMock(side_effect=mock_handle_transcribe)
            
            mock_handler_factory = Mock()
            mock_handler_factory.create_audio_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory
            
            stt = WhisperSTT(model='base')
            stt._audio_handler = mock_handler
            return stt

    def test_transcribe(self, whisper_stt):
        result = whisper_stt.transcribe("test_audio.mp3")
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"
        assert whisper_stt._audio_handler.handle_transcribe.called

    @pytest.mark.asyncio
    async def test_transcribe_async(self, whisper_stt):
        result = await whisper_stt.transcribe_async("test_audio.mp3")
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"
        assert whisper_stt._audio_handler.handle_transcribe.called


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Audio Facade not available")
class TestTextToSpeech:
    @pytest.fixture
    def tts(self):
        with patch('llmkit.facade.audio_facade.HandlerFactory') as mock_factory:
            mock_handler = MagicMock()
            mock_audio = Mock(spec=AudioSegment)
            async def mock_handle_synthesize(*args, **kwargs):
                return mock_audio
            mock_handler.handle_synthesize = MagicMock(side_effect=mock_handle_synthesize)
            
            mock_handler_factory = Mock()
            mock_handler_factory.create_audio_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory
            
            tts = TextToSpeech(provider='openai', voice='alloy')
            tts._audio_handler = mock_handler
            return tts

    def test_synthesize(self, tts):
        result = tts.synthesize("Hello, world!")
        assert isinstance(result, AudioSegment)
        assert tts._audio_handler.handle_synthesize.called


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Audio Facade not available")
class TestAudioRAG:
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.similarity_search = Mock(return_value=[])
        return store

    @pytest.fixture
    def audio_rag(self, mock_vector_store):
        with patch('llmkit.facade.audio_facade.HandlerFactory') as mock_factory:
            from unittest.mock import AsyncMock
            mock_handler = MagicMock()
            mock_results = []
            # AsyncMock을 사용하여 실제 coroutine 반환
            mock_handler.handle_search_audio = AsyncMock(return_value=mock_results)
            
            mock_handler_factory = Mock()
            mock_handler_factory.create_audio_handler.return_value = mock_handler
            mock_factory.return_value = mock_handler_factory
            
            rag = AudioRAG(vector_store=mock_vector_store)
            rag._audio_handler = mock_handler
            return rag

    def test_search(self, audio_rag):
        results = audio_rag.search("What was discussed?")
        assert isinstance(results, list)
        assert audio_rag._audio_handler.handle_search_audio.called


