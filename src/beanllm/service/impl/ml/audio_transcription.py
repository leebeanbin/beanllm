"""
Audio Transcription - Whisper STT (Speech-to-Text) logic

Extracted from AudioServiceImpl for SRP compliance.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from beanllm.domain.audio import (
    AudioSegment,
    TranscriptionResult,
    TranscriptionSegment,
)
from beanllm.dto.request.ml.audio_request import AudioRequest
from beanllm.dto.response.ml.audio_response import AudioResponse
from beanllm.infrastructure.distributed.pipeline_decorators import (
    with_distributed_features,
)
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class AudioTranscriptionMixin:
    """
    Mixin providing Whisper-based speech-to-text transcription.

    Expects on self (set by AudioServiceImpl):
        _whisper_model: Loaded Whisper model (lazy-loaded)
        _whisper_model_name: Model name (e.g. "base")
        _whisper_device: Device for inference
        _whisper_language: Default language hint
    """

    _whisper_model: Any = None
    _whisper_model_name: str = ""
    _whisper_device: Optional[str] = None
    _whisper_language: Optional[str] = None

    def _load_whisper_model(self) -> None:
        """
        Load Whisper model (lazy loading).

        Raises:
            ImportError: If openai-whisper is not installed.
        """
        if self._whisper_model is not None:
            return

        try:
            import whisper

            self._whisper_model = whisper.load_model(
                self._whisper_model_name, device=self._whisper_device
            )
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. Install with: pip install openai-whisper"
            )

    def _resolve_audio_path(self, audio: Union[str, Path, AudioSegment, bytes]) -> tuple[str, bool]:
        """
        Resolve audio input to a file path for Whisper.

        Args:
            audio: Audio source (path, AudioSegment, or bytes).

        Returns:
            Tuple of (audio_path, should_cleanup) where should_cleanup is True
            if a temp file was created and should be deleted after use.
        """
        if isinstance(audio, (str, Path)):
            return str(audio), False
        if isinstance(audio, AudioSegment):
            with tempfile.NamedTemporaryFile(suffix=f".{audio.format}", delete=False) as f:
                f.write(audio.audio_data)
                return f.name, True
        if isinstance(audio, bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                return f.name, True
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    def _build_transcription_result(self, result: dict) -> TranscriptionResult:
        """Convert Whisper raw result to TranscriptionResult."""
        segments = []
        for seg in result.get("segments", []):
            segments.append(
                TranscriptionSegment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    confidence=seg.get("confidence", 1.0),
                    language=result.get("language"),
                )
            )
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
            duration=result.get("duration", 0.0),
            model=self._whisper_model_name,
            metadata=result,
        )

    @with_distributed_features(
        pipeline_type="audio",
        enable_rate_limiting=True,
        rate_limit_key=lambda self, args, kwargs: f"audio:transcribe:{self._whisper_model_name}",
    )
    async def transcribe(self, request: AudioRequest) -> AudioResponse:
        """
        Convert speech to text using Whisper.

        Args:
            request: Audio request DTO.

        Returns:
            AudioResponse with transcription_result field.
        """
        self._load_whisper_model()

        audio = request.audio
        if audio is None:
            raise ValueError("audio is required for transcription")
        language = request.language or self._whisper_language
        task = request.task
        kwargs = request.extra_params or {}

        audio_path, should_cleanup = self._resolve_audio_path(audio)
        try:
            options = {"language": language, "task": task, **kwargs}
            model = self._whisper_model
            if model is None:
                raise RuntimeError("Whisper model failed to load")
            result = model.transcribe(audio_path, **options)
            transcription_result = self._build_transcription_result(result)
            return AudioResponse(transcription_result=transcription_result)
        finally:
            if should_cleanup:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass
