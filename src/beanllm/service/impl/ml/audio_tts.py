"""
Audio TTS - Text-to-Speech logic with multiple providers

Extracted from AudioServiceImpl for SRP compliance.
Supports: OpenAI, Google Cloud, Azure, ElevenLabs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from beanllm.domain.audio import AudioSegment, TTSProvider
from beanllm.dto.request.ml.audio_request import AudioRequest
from beanllm.dto.response.ml.audio_response import AudioResponse
from beanllm.infrastructure.distributed.pipeline_decorators import (
    with_distributed_features,
)
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class AudioTTSMixin:
    """
    Mixin providing text-to-speech synthesis via multiple providers.

    Expects on self (set by AudioServiceImpl):
        _tts_provider: TTSProvider enum
        _tts_api_key: API key for provider
        _tts_model: Model name
        _tts_voice: Voice ID/name
    """

    _tts_provider: TTSProvider = TTSProvider.OPENAI
    _tts_api_key: Optional[str] = None
    _tts_model: Optional[str] = None
    _tts_voice: Optional[str] = None

    def _get_tts_synthesizer_registry(
        self,
    ) -> Dict[TTSProvider, Callable[..., Any]]:
        """TTS Provider → synthesizer method mapping."""
        return {
            TTSProvider.OPENAI: self._synthesize_openai,
            TTSProvider.GOOGLE: self._synthesize_google,
            TTSProvider.AZURE: self._synthesize_azure,
            TTSProvider.ELEVENLABS: self._synthesize_elevenlabs,
        }

    @with_distributed_features(
        pipeline_type="audio",
        enable_rate_limiting=True,
        rate_limit_key=lambda self,
        args,
        kwargs: f"audio:synthesize:{(self._tts_provider.value if hasattr(self._tts_provider, 'value') else str(self._tts_provider))}",
    )
    async def synthesize(self, request: AudioRequest) -> AudioResponse:
        """
        Convert text to speech using configured TTS provider.

        Args:
            request: Audio request DTO.

        Returns:
            AudioResponse with audio_segment field.
        """
        text = request.text
        voice = request.voice or self._tts_voice
        speed = request.speed
        api_key = request.api_key or self._tts_api_key
        kwargs = request.extra_params or {}

        registry = self._get_tts_synthesizer_registry()
        synthesizer = registry.get(self._tts_provider)
        if synthesizer is None:
            raise ValueError(f"Unsupported TTS provider: {self._tts_provider}")

        if self._tts_provider in (TTSProvider.OPENAI, TTSProvider.ELEVENLABS):
            audio_segment = await synthesizer(
                text, voice, speed, api_key, request.tts_model, **kwargs
            )
        else:
            audio_segment = await synthesizer(text, voice, speed, api_key, **kwargs)

        return AudioResponse(audio_segment=audio_segment)

    async def _synthesize_openai(
        self,
        text: str,
        voice: str,
        speed: float,
        api_key: Optional[str],
        model: Optional[str],
        **kwargs: Any,
    ) -> AudioSegment:
        """OpenAI TTS synthesis."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. pip install openai")

        client = OpenAI(api_key=api_key)
        response = client.audio.speech.create(
            model=model or self._tts_model or "tts-1",
            voice=voice or "alloy",
            input=text,
            speed=speed,
            **kwargs,
        )
        return AudioSegment(
            audio_data=response.content,
            sample_rate=24000,
            format="mp3",
            metadata={
                "provider": "openai",
                "voice": voice,
                "model": model or self._tts_model or "tts-1",
            },
        )

    async def _synthesize_google(
        self,
        text: str,
        voice: Optional[str],
        speed: float,
        api_key: Optional[str],
        **kwargs: Any,
    ) -> AudioSegment:
        """Google Cloud TTS synthesis."""
        try:
            from google.cloud import texttospeech
        except ImportError:
            raise ImportError(
                "google-cloud-texttospeech not installed. " "pip install google-cloud-texttospeech"
            )

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=kwargs.get("language_code", "en-US"), name=voice
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speed
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        return AudioSegment(
            audio_data=response.audio_content,
            format="mp3",
            metadata={"provider": "google", "voice": voice},
        )

    async def _synthesize_azure(
        self,
        text: str,
        voice: Optional[str],
        speed: float,
        api_key: Optional[str],
        **kwargs: Any,
    ) -> AudioSegment:
        """Azure TTS synthesis."""
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-speech not installed. "
                "pip install azure-cognitiveservices-speech"
            )

        speech_config = speechsdk.SpeechConfig(
            subscription=api_key, region=kwargs.get("region", "eastus")
        )
        if voice:
            speech_config.speech_synthesis_voice_name = voice

        speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return AudioSegment(
                audio_data=result.audio_data,
                format="wav",
                metadata={"provider": "azure", "voice": voice},
            )
        raise RuntimeError(f"Azure TTS failed: {result.reason}")

    async def _synthesize_elevenlabs(
        self,
        text: str,
        voice: Optional[str],
        speed: float,
        api_key: Optional[str],
        model: Optional[str],
        **kwargs: Any,
    ) -> AudioSegment:
        """ElevenLabs TTS synthesis."""
        import httpx

        if not voice:
            voice = "21m00Tcm4TlvDq8ikWAM"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers: Dict[str, str] = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key or "",
        }
        data = {
            "text": text,
            "model_id": model or self._tts_model or "eleven_monolingual_v1",
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.5),
            },
        }
        response = httpx.post(url, json=data, headers=headers)
        response.raise_for_status()
        return AudioSegment(
            audio_data=response.content,
            format="mp3",
            metadata={"provider": "elevenlabs", "voice": voice},
        )
