"""
Audio Facade 테스트
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

try:
    from beanllm.facade.audio_facade import WhisperSTT, TextToSpeech, AudioRAG
    from beanllm.domain.audio.types import TranscriptionResult, AudioSegment
    from beanllm.dto.response.audio_response import AudioResponse
    FACADE_AVAILABLE = True
except ImportError:
    FACADE_AVAILABLE = False


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Audio Facade not available")
class TestWhisperSTT:
    @pytest.fixture
    def whisper_stt(self):
        # Patch AudioServiceImpl where it's imported
        patcher = patch("beanllm.service.impl.audio_service_impl.AudioServiceImpl")
        mock_audio_service_class = patcher.start()

        from unittest.mock import AsyncMock

        # Mock service instance
        mock_service = Mock()
        mock_transcription = TranscriptionResult(
            text="Test transcription",
            language="en",
            segments=[]
        )
        mock_service.transcribe = AsyncMock(return_value=AudioResponse(
            transcription_result=mock_transcription
        ))
        mock_audio_service_class.return_value = mock_service

        stt = WhisperSTT(model='base')

        yield stt

        patcher.stop()

    def test_transcribe(self, whisper_stt):
        result = whisper_stt.transcribe("test_audio.mp3")
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"

    @pytest.mark.asyncio
    async def test_transcribe_async(self, whisper_stt):
        result = await whisper_stt.transcribe_async("test_audio.mp3")
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Audio Facade not available")
class TestTextToSpeech:
    @pytest.fixture
    def tts(self):
        # Patch AudioServiceImpl where it's imported
        patcher = patch("beanllm.service.impl.audio_service_impl.AudioServiceImpl")
        mock_audio_service_class = patcher.start()

        from unittest.mock import AsyncMock

        # Mock service instance
        mock_service = Mock()
        mock_audio = AudioSegment(audio_data=b"fake", format="mp3", sample_rate=24000)
        mock_service.synthesize = AsyncMock(return_value=AudioResponse(
            audio_segment=mock_audio
        ))
        mock_audio_service_class.return_value = mock_service

        tts = TextToSpeech(provider='openai', voice='alloy')

        yield tts

        patcher.stop()

    def test_synthesize(self, tts):
        result = tts.synthesize("Hello, world!")
        assert isinstance(result, AudioSegment)


@pytest.mark.skipif(not FACADE_AVAILABLE, reason="Audio Facade not available")
class TestAudioRAG:
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.similarity_search = Mock(return_value=[])
        # search 메서드는 리스트를 반환해야 함 (iterate 가능)
        store.search = Mock(return_value=[])
        return store

    @pytest.fixture
    def audio_rag(self, mock_vector_store):
        # Patch AudioServiceImpl where it's imported
        patcher1 = patch("beanllm.service.impl.audio_service_impl.AudioServiceImpl")
        patcher2 = patch("beanllm.facade.audio_facade.WhisperSTT")

        mock_audio_service_class = patcher1.start()
        mock_whisper_stt_class = patcher2.start()

        from unittest.mock import AsyncMock

        # Mock WhisperSTT 인스턴스
        mock_stt = Mock()
        mock_stt.model_name = "base"
        mock_stt.device = None
        mock_stt.language = None
        mock_whisper_stt_class.return_value = mock_stt

        # Mock service instance
        mock_service = Mock()
        mock_results = []
        mock_service.search_audio = AsyncMock(return_value=AudioResponse(
            search_results=mock_results
        ))
        mock_audio_service_class.return_value = mock_service

        rag = AudioRAG(vector_store=mock_vector_store)

        yield rag

        patcher1.stop()
        patcher2.stop()

    def test_search(self, audio_rag):
        results = audio_rag.search("What was discussed?")
        assert isinstance(results, list)


