"""Tests for facade/ml/audio_facade.py — WhisperSTT, TextToSpeech, AudioRAG."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.audio import AudioSegment, TranscriptionResult, TTSProvider, WhisperModel
from beanllm.facade.ml.audio_facade import (
    AudioRAG,
    TextToSpeech,
    WhisperSTT,
    text_to_speech,
    transcribe_audio,
)


def _make_whisper_stt(model="base", device=None, language=None):
    """Create WhisperSTT with fully mocked dependencies."""
    mock_handler = MagicMock()
    mock_result = MagicMock(spec=TranscriptionResult)
    mock_result.text = "Test transcription"
    mock_response = MagicMock()
    mock_response.transcription_result = mock_result

    mock_handler.handle_transcribe = AsyncMock(return_value=mock_response)

    container_patcher = patch("beanllm.utils.core.di_container.get_container")
    handler_patcher = patch("beanllm.facade.ml.audio_facade.AudioHandler")

    mock_get_container = container_patcher.start()
    MockHandler = handler_patcher.start()

    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    MockHandler.return_value = mock_handler

    stt = WhisperSTT(model=model, device=device, language=language)
    return stt, mock_handler, [container_patcher, handler_patcher]


def _make_tts(provider="openai", voice=None):
    """Create TextToSpeech with fully mocked dependencies."""
    mock_handler = MagicMock()
    mock_audio = MagicMock(spec=AudioSegment)
    mock_response = MagicMock()
    mock_response.audio_segment = mock_audio

    mock_handler.handle_synthesize = AsyncMock(return_value=mock_response)

    container_patcher = patch("beanllm.utils.core.di_container.get_container")
    handler_patcher = patch("beanllm.facade.ml.audio_facade.AudioHandler")

    mock_get_container = container_patcher.start()
    MockHandler = handler_patcher.start()

    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    MockHandler.return_value = mock_handler

    tts = TextToSpeech(provider=provider, voice=voice)
    return tts, mock_handler, [container_patcher, handler_patcher]


def _make_audio_rag():
    """Create AudioRAG with fully mocked dependencies."""
    mock_handler = MagicMock()
    mock_handler.handle_add_audio = AsyncMock()
    mock_handler.handle_search_audio = AsyncMock()
    mock_handler.handle_get_transcription = AsyncMock()
    mock_handler.handle_list_audios = AsyncMock()

    mock_stt = MagicMock(spec=WhisperSTT)
    mock_stt.model_name = "base"
    mock_stt.device = None
    mock_stt.language = None

    container_patcher = patch("beanllm.utils.core.di_container.get_container")
    handler_patcher = patch("beanllm.facade.ml.audio_facade.AudioHandler")

    mock_get_container = container_patcher.start()
    MockHandler = handler_patcher.start()

    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    MockHandler.return_value = mock_handler

    rag = AudioRAG(stt=mock_stt)
    return rag, mock_handler, mock_stt, [container_patcher, handler_patcher]


def _stop(patchers):
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------------------
# WhisperSTT
# ---------------------------------------------------------------------------


class TestWhisperSTTInit:
    def test_stores_model_name(self):
        stt, _, p = _make_whisper_stt(model="large")
        try:
            assert stt.model_name == "large"
        finally:
            _stop(p)

    def test_whisper_model_enum_converted(self):
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.audio_facade.AudioHandler"),
        ):
            mc.return_value = MagicMock()
            stt = WhisperSTT(model=WhisperModel.BASE)
        assert stt.model_name == WhisperModel.BASE.value

    def test_stores_device(self):
        stt, _, p = _make_whisper_stt(device="cpu")
        try:
            assert stt.device == "cpu"
        finally:
            _stop(p)

    def test_stores_language(self):
        stt, _, p = _make_whisper_stt(language="ko")
        try:
            assert stt.language == "ko"
        finally:
            _stop(p)


class TestWhisperSTTTranscribe:
    def test_transcribe_returns_result(self):
        stt, _, p = _make_whisper_stt()
        try:
            result = stt.transcribe("audio.mp3")
            assert result is not None
        finally:
            _stop(p)

    def test_transcribe_passes_audio(self):
        stt, handler, p = _make_whisper_stt()
        try:
            stt.transcribe("test.mp3")
            call_kwargs = handler.handle_transcribe.call_args.kwargs
            assert call_kwargs.get("audio") == "test.mp3"
        finally:
            _stop(p)

    def test_transcribe_passes_language(self):
        stt, handler, p = _make_whisper_stt()
        try:
            stt.transcribe("test.mp3", language="en")
            call_kwargs = handler.handle_transcribe.call_args.kwargs
            assert call_kwargs.get("language") == "en"
        finally:
            _stop(p)

    def test_transcribe_raises_on_none_result(self):
        stt, handler, p = _make_whisper_stt()
        try:
            mock_response = MagicMock()
            mock_response.transcription_result = None
            handler.handle_transcribe = AsyncMock(return_value=mock_response)
            with pytest.raises(ValueError, match="Transcription result is None"):
                stt.transcribe("audio.mp3")
        finally:
            _stop(p)

    async def test_transcribe_async_returns_result(self):
        stt, handler, p = _make_whisper_stt()
        try:
            result = await stt.transcribe_async("audio.mp3")
            assert result is not None
        finally:
            _stop(p)

    async def test_transcribe_async_raises_on_none(self):
        stt, handler, p = _make_whisper_stt()
        try:
            mock_response = MagicMock()
            mock_response.transcription_result = None
            handler.handle_transcribe = AsyncMock(return_value=mock_response)
            with pytest.raises(ValueError):
                await stt.transcribe_async("audio.mp3")
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# TextToSpeech
# ---------------------------------------------------------------------------


class TestTextToSpeechInit:
    def test_stores_provider_enum(self):
        tts, _, p = _make_tts(provider="openai")
        try:
            assert tts.provider == TTSProvider.OPENAI
        finally:
            _stop(p)

    def test_provider_string_converted_to_enum(self):
        tts, _, p = _make_tts(provider="openai")
        try:
            assert isinstance(tts.provider, TTSProvider)
        finally:
            _stop(p)

    def test_stores_voice(self):
        tts, _, p = _make_tts(voice="alloy")
        try:
            assert tts.voice == "alloy"
        finally:
            _stop(p)


class TestTextToSpeechSynthesize:
    def test_synthesize_returns_audio_segment(self):
        tts, _, p = _make_tts()
        try:
            result = tts.synthesize("Hello world")
            assert result is not None
        finally:
            _stop(p)

    def test_synthesize_passes_text(self):
        tts, handler, p = _make_tts()
        try:
            tts.synthesize("Hello!")
            call_kwargs = handler.handle_synthesize.call_args.kwargs
            assert call_kwargs.get("text") == "Hello!"
        finally:
            _stop(p)

    def test_synthesize_passes_voice(self):
        tts, handler, p = _make_tts(voice="alloy")
        try:
            tts.synthesize("Hi")
            call_kwargs = handler.handle_synthesize.call_args.kwargs
            assert call_kwargs.get("voice") == "alloy"
        finally:
            _stop(p)

    def test_synthesize_raises_on_none_audio(self):
        tts, handler, p = _make_tts()
        try:
            mock_response = MagicMock()
            mock_response.audio_segment = None
            handler.handle_synthesize = AsyncMock(return_value=mock_response)
            with pytest.raises(ValueError, match="Audio segment is None"):
                tts.synthesize("Hello")
        finally:
            _stop(p)

    async def test_synthesize_async_returns_audio(self):
        tts, _, p = _make_tts()
        try:
            result = await tts.synthesize_async("Hello world")
            assert result is not None
        finally:
            _stop(p)

    async def test_synthesize_async_raises_on_none(self):
        tts, handler, p = _make_tts()
        try:
            mock_response = MagicMock()
            mock_response.audio_segment = None
            handler.handle_synthesize = AsyncMock(return_value=mock_response)
            with pytest.raises(ValueError):
                await tts.synthesize_async("Hi")
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# AudioRAG
# ---------------------------------------------------------------------------


class TestAudioRAGInit:
    def test_stores_stt(self):
        rag, _, stt, p = _make_audio_rag()
        try:
            assert rag.stt is stt
        finally:
            _stop(p)


class TestAudioRAGAddAudio:
    def test_add_audio_returns_transcription(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_transcription = MagicMock(spec=TranscriptionResult)
            mock_response = MagicMock()
            mock_response.transcription = mock_transcription
            handler.handle_add_audio = AsyncMock(return_value=mock_response)
            result = rag.add_audio("meeting.wav")
            assert result is mock_transcription
        finally:
            _stop(p)

    def test_add_audio_raises_on_none(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_response = MagicMock()
            mock_response.transcription = None
            handler.handle_add_audio = AsyncMock(return_value=mock_response)
            with pytest.raises(ValueError):
                rag.add_audio("file.wav")
        finally:
            _stop(p)

    async def test_add_audio_async_returns_transcription(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_transcription = MagicMock(spec=TranscriptionResult)
            mock_response = MagicMock()
            mock_response.transcription = mock_transcription
            handler.handle_add_audio = AsyncMock(return_value=mock_response)
            result = await rag.add_audio_async("meeting.wav")
            assert result is mock_transcription
        finally:
            _stop(p)


class TestAudioRAGSearch:
    def test_search_returns_results(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_response = MagicMock()
            mock_response.search_results = [{"text": "hello"}, {"text": "world"}]
            handler.handle_search_audio = AsyncMock(return_value=mock_response)
            results = rag.search("What was discussed?")
            assert len(results) == 2
        finally:
            _stop(p)

    def test_search_returns_empty_on_none_response(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            handler.handle_search_audio = AsyncMock(return_value=None)
            results = rag.search("q")
            assert results == []
        finally:
            _stop(p)

    def test_search_passes_query(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_response = MagicMock()
            mock_response.search_results = []
            handler.handle_search_audio = AsyncMock(return_value=mock_response)
            rag.search("my query", top_k=3)
            call_kwargs = handler.handle_search_audio.call_args.kwargs
            assert call_kwargs.get("query") == "my query"
            assert call_kwargs.get("top_k") == 3
        finally:
            _stop(p)


class TestAudioRAGGetTranscription:
    def test_get_transcription_returns_result(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_transcription = MagicMock(spec=TranscriptionResult)
            mock_response = MagicMock()
            mock_response.transcription = mock_transcription
            handler.handle_get_transcription = AsyncMock(return_value=mock_response)
            result = rag.get_transcription("audio-123")
            assert result is mock_transcription
        finally:
            _stop(p)

    def test_get_transcription_returns_none_on_none_response(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            handler.handle_get_transcription = AsyncMock(return_value=None)
            result = rag.get_transcription("unknown-id")
            assert result is None
        finally:
            _stop(p)


class TestAudioRAGListAudios:
    def test_list_audios_returns_ids(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            mock_response = MagicMock()
            mock_response.audio_ids = ["id-1", "id-2"]
            handler.handle_list_audios = AsyncMock(return_value=mock_response)
            result = rag.list_audios()
            assert result == ["id-1", "id-2"]
        finally:
            _stop(p)

    def test_list_audios_returns_empty_on_none(self):
        rag, handler, _, p = _make_audio_rag()
        try:
            handler.handle_list_audios = AsyncMock(return_value=None)
            result = rag.list_audios()
            assert result == []
        finally:
            _stop(p)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestTranscribeAudio:
    def test_transcribe_audio_calls_whisper(self):
        mock_result = MagicMock(spec=TranscriptionResult)
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.audio_facade.AudioHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_response = MagicMock()
            mock_response.transcription_result = mock_result
            mock_handler.handle_transcribe = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            result = transcribe_audio("test.mp3", model="base")
        assert result is mock_result


class TestTextToSpeechConvenience:
    def test_text_to_speech_returns_audio(self):
        mock_audio = MagicMock(spec=AudioSegment)
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.audio_facade.AudioHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_response = MagicMock()
            mock_response.audio_segment = mock_audio
            mock_handler.handle_synthesize = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            result = text_to_speech("Hello", provider="openai")
        assert result is mock_audio

    def test_text_to_speech_saves_file_if_output_given(self, tmp_path):
        mock_audio = MagicMock(spec=AudioSegment)
        output = str(tmp_path / "out.mp3")
        with (
            patch("beanllm.utils.core.di_container.get_container") as mc,
            patch("beanllm.facade.ml.audio_facade.AudioHandler") as MockH,
        ):
            mc.return_value = MagicMock()
            mock_handler = MagicMock()
            mock_response = MagicMock()
            mock_response.audio_segment = mock_audio
            mock_handler.handle_synthesize = AsyncMock(return_value=mock_response)
            MockH.return_value = mock_handler
            text_to_speech("Hello", output_file=output)
        mock_audio.to_file.assert_called_once_with(output)
