"""
AudioServiceImpl - Audio 서비스 구현체
SOLID 원칙:
- SRP: Audio 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (의존성 주입)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from ...domain.audio import (
    AudioSegment,
    TranscriptionResult,
    TranscriptionSegment,
    TTSProvider,
    WhisperModel,
)
from ...dto.request.audio_request import AudioRequest
from ...dto.response.audio_response import AudioResponse
from ...utils.logger import get_logger
from ..audio_service import IAudioService

if TYPE_CHECKING:
    from ...domain.embeddings import BaseEmbedding
    from ...service.types import VectorStoreProtocol

logger = get_logger(__name__)


class AudioServiceImpl(IAudioService):
    """
    Audio 서비스 구현체

    책임:
    - Audio 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - SRP: Audio 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
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
        의존성 주입을 통한 생성자

        Args:
            whisper_model: Whisper 모델 크기
            whisper_device: Whisper 디바이스
            whisper_language: Whisper 언어
            tts_provider: TTS 제공자
            tts_api_key: TTS API 키
            tts_model: TTS 모델
            tts_voice: TTS 음성
            vector_store: 벡터 스토어 (AudioRAG용)
            embedding_model: 임베딩 모델 (AudioRAG용)
        """
        # WhisperSTT 설정
        self._whisper_model_name = (
            whisper_model.value
            if isinstance(whisper_model, WhisperModel)
            else (whisper_model or "base")
        )
        self._whisper_device = whisper_device
        self._whisper_language = whisper_language
        self._whisper_model = None

        # TextToSpeech 설정
        if isinstance(tts_provider, str):
            tts_provider = TTSProvider(tts_provider)
        elif tts_provider is None:
            tts_provider = TTSProvider.OPENAI

        self._tts_provider = tts_provider
        self._tts_api_key = tts_api_key
        self._tts_model = tts_model
        self._tts_voice = tts_voice

        # AudioRAG 설정
        self._vector_store = vector_store
        self._embedding_model = embedding_model
        self._transcriptions: Dict[str, TranscriptionResult] = {}

    def _load_whisper_model(self):
        """Whisper 모델 로드 (lazy loading) (기존 audio_speech.py의 WhisperSTT._load_model() 정확히 마이그레이션)"""
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

    async def transcribe(self, request: AudioRequest) -> AudioResponse:
        """
        음성을 텍스트로 변환 (기존 audio_speech.py의 WhisperSTT.transcribe() 정확히 마이그레이션)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (transcription_result 필드 포함)
        """
        self._load_whisper_model()

        audio = request.audio
        language = request.language or self._whisper_language
        task = request.task
        kwargs = request.extra_params or {}

        # 오디오 준비 (기존과 동일)
        if isinstance(audio, (str, Path)):
            audio_path = str(audio)
        elif isinstance(audio, AudioSegment):
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=f".{audio.format}", delete=False) as f:
                f.write(audio.audio_data)
                audio_path = f.name
        elif isinstance(audio, bytes):
            # bytes를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                audio_path = f.name
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        # 전사 실행 (기존과 동일)
        options = {"language": language, "task": task, **kwargs}

        result = self._whisper_model.transcribe(audio_path, **options)

        # 결과 변환 (기존과 동일)
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

        # 임시 파일 정리 (기존과 동일)
        if isinstance(audio, (AudioSegment, bytes)):
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        transcription_result = TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
            duration=result.get("duration", 0.0),
            model=self._whisper_model_name,
            metadata=result,
        )

        return AudioResponse(transcription_result=transcription_result)

    async def synthesize(self, request: AudioRequest) -> AudioResponse:
        """
        텍스트를 음성으로 변환 (기존 audio_speech.py의 TextToSpeech.synthesize() 정확히 마이그레이션)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (audio_segment 필드 포함)
        """
        text = request.text
        voice = request.voice or self._tts_voice
        speed = request.speed
        api_key = request.api_key or self._tts_api_key
        kwargs = request.extra_params or {}

        # Provider별 합성 (기존과 동일)
        if self._tts_provider == TTSProvider.OPENAI:
            audio_segment = await self._synthesize_openai(
                text, voice, speed, api_key, request.tts_model, **kwargs
            )
        elif self._tts_provider == TTSProvider.GOOGLE:
            audio_segment = await self._synthesize_google(text, voice, speed, api_key, **kwargs)
        elif self._tts_provider == TTSProvider.AZURE:
            audio_segment = await self._synthesize_azure(text, voice, speed, api_key, **kwargs)
        elif self._tts_provider == TTSProvider.ELEVENLABS:
            audio_segment = await self._synthesize_elevenlabs(
                text, voice, speed, api_key, request.tts_model, **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {self._tts_provider}")

        return AudioResponse(audio_segment=audio_segment)

    async def _synthesize_openai(
        self,
        text: str,
        voice: str,
        speed: float,
        api_key: Optional[str],
        model: Optional[str],
        **kwargs,
    ) -> AudioSegment:
        """OpenAI TTS (기존 audio_speech.py의 TextToSpeech._synthesize_openai() 정확히 마이그레이션)"""
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

        # Response is audio bytes (기존과 동일)
        audio_data = response.content

        return AudioSegment(
            audio_data=audio_data,
            sample_rate=24000,  # OpenAI TTS default
            format="mp3",
            metadata={
                "provider": "openai",
                "voice": voice,
                "model": model or self._tts_model or "tts-1",
            },
        )

    async def _synthesize_google(
        self, text: str, voice: Optional[str], speed: float, api_key: Optional[str], **kwargs
    ) -> AudioSegment:
        """Google Cloud TTS (기존 audio_speech.py의 TextToSpeech._synthesize_google() 정확히 마이그레이션)"""
        try:
            from google.cloud import texttospeech
        except ImportError:
            raise ImportError(
                "google-cloud-texttospeech not installed. pip install google-cloud-texttospeech"
            )

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Voice parameters (기존과 동일)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=kwargs.get("language_code", "en-US"), name=voice
        )

        # Audio config (기존과 동일)
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
        self, text: str, voice: Optional[str], speed: float, api_key: Optional[str], **kwargs
    ) -> AudioSegment:
        """Azure TTS (기존 audio_speech.py의 TextToSpeech._synthesize_azure() 정확히 마이그레이션)"""
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

        # Synthesize to in-memory stream (기존과 동일)
        speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return AudioSegment(
                audio_data=result.audio_data,
                format="wav",
                metadata={"provider": "azure", "voice": voice},
            )
        else:
            raise RuntimeError(f"Azure TTS failed: {result.reason}")

    async def _synthesize_elevenlabs(
        self,
        text: str,
        voice: Optional[str],
        speed: float,
        api_key: Optional[str],
        model: Optional[str],
        **kwargs,
    ) -> AudioSegment:
        """ElevenLabs TTS (기존 audio_speech.py의 TextToSpeech._synthesize_elevenlabs() 정확히 마이그레이션)"""
        import requests

        if not voice:
            voice = "21m00Tcm4TlvDq8ikWAM"  # Default voice

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        }

        data = {
            "text": text,
            "model_id": model or self._tts_model or "eleven_monolingual_v1",
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.5),
            },
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        return AudioSegment(
            audio_data=response.content,
            format="mp3",
            metadata={"provider": "elevenlabs", "voice": voice},
        )

    async def add_audio(self, request: AudioRequest) -> AudioResponse:
        """
        오디오를 전사하고 RAG 시스템에 추가 (기존 audio_speech.py의 AudioRAG.add_audio() 정확히 마이그레이션)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (transcription 필드 포함)
        """
        audio = request.audio
        audio_id = request.audio_id
        metadata = request.metadata or {}

        # 전사 (기존과 동일)
        transcribe_request = AudioRequest(
            audio=audio,
            language=request.language,
            task=request.task,
            model=request.model,
            device=request.device,
            extra_params=request.extra_params,
        )
        transcription = await self.transcribe(transcribe_request)
        transcription_result = transcription.transcription_result

        if not transcription_result:
            raise ValueError("Transcription failed")

        # ID 생성 (기존과 동일)
        if audio_id is None:
            if isinstance(audio, (str, Path)):
                audio_id = str(Path(audio).stem)
            else:
                audio_id = f"audio_{len(self._transcriptions)}"

        # 저장 (기존과 동일)
        self._transcriptions[audio_id] = transcription_result

        # Vector store에 추가 (있는 경우) (기존과 동일)
        if self._vector_store is not None and self._embedding_model is not None:
            # 각 세그먼트를 별도 문서로 추가
            from ...domain.loaders import Document

            documents = []
            for i, segment in enumerate(transcription_result.segments):
                doc = Document(
                    content=segment.text,
                    metadata={
                        "audio_id": audio_id,
                        "segment_id": i,
                        "start": segment.start,
                        "end": segment.end,
                        "language": segment.language,
                        **metadata,
                    },
                )
                documents.append(doc)

            self._vector_store.add_documents(documents, self._embedding_model)

        return AudioResponse(transcription=transcription_result)

    async def search_audio(self, request: AudioRequest) -> AudioResponse:
        """
        쿼리로 관련 음성 세그먼트 검색 (기존 audio_speech.py의 AudioRAG.search() 정확히 마이그레이션)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (search_results 필드 포함)
        """
        query = request.query or ""
        top_k = request.top_k
        kwargs = request.extra_params or {}

        if self._vector_store is None:
            # Fallback: 단순 텍스트 매칭 (기존과 동일)
            results = []
            for audio_id, transcription in self._transcriptions.items():
                for i, segment in enumerate(transcription.segments):
                    if query.lower() in segment.text.lower():
                        results.append({"audio_id": audio_id, "segment": segment, "score": 1.0})

            return AudioResponse(search_results=results[:top_k])

        # Vector search (기존과 동일)
        search_results = self._vector_store.search(query, k=top_k, **kwargs)

        results = []
        for result in search_results:
            metadata = result.metadata
            audio_id = metadata.get("audio_id")
            segment_id = metadata.get("segment_id")

            if audio_id in self._transcriptions:
                transcription = self._transcriptions[audio_id]
                segment = transcription.segments[segment_id]

                results.append(
                    {
                        "audio_id": audio_id,
                        "segment": segment,
                        "score": result.score,
                        "text": result.content,
                    }
                )

        return AudioResponse(search_results=results)

    async def get_transcription(self, request: AudioRequest) -> AudioResponse:
        """
        오디오 ID로 전사 결과 조회 (기존 audio_speech.py의 AudioRAG.get_transcription() 정확히 마이그레이션)

        Args:
            request: Audio 요청 DTO (audio_id 필드 사용)

        Returns:
            AudioResponse: Audio 응답 DTO (transcription 필드 포함)
        """
        audio_id = request.audio_id
        if not audio_id:
            raise ValueError("audio_id is required")

        transcription = self._transcriptions.get(audio_id)
        return AudioResponse(transcription=transcription)

    async def list_audios(self, request: AudioRequest) -> AudioResponse:
        """
        저장된 모든 오디오 ID 목록 조회 (기존 audio_speech.py의 AudioRAG.list_audios() 정확히 마이그레이션)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (audio_ids 필드 포함)
        """
        audio_ids = list(self._transcriptions.keys())
        return AudioResponse(audio_ids=audio_ids)
