"""
AudioServiceImpl - Audio 서비스 구현체 (Coordinator)

Delegates to focused modules:
- AudioTranscriptionMixin: Whisper STT
- AudioTTSMixin: TTS (OpenAI, Google, Azure, ElevenLabs)
- AudioRAGMixin: AudioRAG (add, search, get, list)

SOLID:
- SRP: Coordinator only; logic in mixins
- DIP: Interface dependency via IAudioService
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from beanllm.domain.audio import TranscriptionResult, TTSProvider, WhisperModel
from beanllm.dto.request.ml.audio_request import AudioRequest
from beanllm.dto.response.ml.audio_response import AudioResponse
from beanllm.service.audio_service import IAudioService
from beanllm.service.impl.ml.audio_rag import AudioRAGMixin
from beanllm.service.impl.ml.audio_transcription import AudioTranscriptionMixin
from beanllm.service.impl.ml.audio_tts import AudioTTSMixin
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.embeddings import BaseEmbedding
    from beanllm.service.types import VectorStoreProtocol

logger = get_logger(__name__)


class AudioServiceImpl(AudioTranscriptionMixin, AudioTTSMixin, AudioRAGMixin, IAudioService):
    """
    Audio service implementation.

    Coordinates transcription (Whisper), TTS (multi-provider), and AudioRAG.
    """

    def __init__(
        self,
        whisper_model: Optional[Union[str, WhisperModel]] = None,
        whisper_device: Optional[str] = None,
        whisper_language: Optional[str] = None,
        tts_provider: Optional[Union[str, TTSProvider]] = None,
        tts_api_key: Optional[str] = None,
        tts_model: Optional[str] = None,
        tts_voice: Optional[str] = None,
        vector_store: Optional["VectorStoreProtocol"] = None,
        embedding_model: Optional["BaseEmbedding"] = None,
    ) -> None:
        """
        Initialize with dependency injection.

        Args:
            whisper_model: Whisper model size.
            whisper_device: Whisper device.
            whisper_language: Whisper language hint.
            tts_provider: TTS provider.
            tts_api_key: TTS API key.
            tts_model: TTS model.
            tts_voice: TTS voice.
            vector_store: Vector store for AudioRAG.
            embedding_model: Embedding model for AudioRAG.
        """
        self._whisper_model_name = (
            whisper_model.value
            if isinstance(whisper_model, WhisperModel)
            else (whisper_model or "base")
        )
        self._whisper_device = whisper_device
        self._whisper_language = whisper_language
        self._whisper_model = None

        if isinstance(tts_provider, str):
            tts_provider = TTSProvider(tts_provider)
        elif tts_provider is None:
            tts_provider = TTSProvider.OPENAI

        self._tts_provider = tts_provider
        self._tts_api_key = tts_api_key
        self._tts_model = tts_model
        self._tts_voice = tts_voice

        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._transcriptions: dict[str, TranscriptionResult] = {}
