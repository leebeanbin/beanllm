"""
Audio Facade - 기존 Audio API를 위한 Facade
책임: 하위 호환성 유지, 내부적으로는 Handler/Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..domain.audio import AudioSegment, TranscriptionResult, TTSProvider, WhisperModel
from ..handler.audio_handler import AudioHandler
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..embeddings import BaseEmbedding
    from ..service.types import VectorStoreProtocol

logger = get_logger(__name__)


class WhisperSTT:
    """
    Whisper Speech-to-Text (Facade 패턴)

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        >>> stt = WhisperSTT(model='base')
        >>> result = stt.transcribe('audio.mp3', language='en')
        >>> print(result.text)
    """

    def __init__(
        self,
        model: Union[str, WhisperModel] = WhisperModel.BASE,
        device: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Args:
            model: Whisper 모델 크기
            device: 디바이스 ('cpu', 'cuda', 'mps')
            language: 언어 지정 (None이면 자동 감지)
        """
        if isinstance(model, WhisperModel):
            model = model.value

        self.model_name = model
        self.device = device
        self.language = language

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..service.impl.audio_service_impl import AudioServiceImpl
        from ..utils.di_container import get_container

        get_container()

        # AudioService 생성 (커스텀 의존성)
        audio_service = AudioServiceImpl(
            whisper_model=self.model_name,
            whisper_device=self.device,
            whisper_language=self.language,
        )

        # AudioHandler 생성 (직접 생성 - 커스텀 Service 사용)

        self._audio_handler = AudioHandler(audio_service)

    def transcribe(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> TranscriptionResult:
        """
        음성을 텍스트로 변환 (기존 audio_speech.py의 WhisperSTT.transcribe() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            audio: 오디오 파일 경로, AudioSegment, 또는 bytes
            language: 언어 코드 (예: 'en', 'ko')
            task: 'transcribe' 또는 'translate' (영어로 번역)
            **kwargs: Whisper 추가 옵션

        Returns:
            TranscriptionResult
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._audio_handler.handle_transcribe(
                audio=audio,
                language=language or self.language,
                task=task,
                model=self.model_name,
                device=self.device,
                **kwargs,
            )
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        if not response.transcription_result:
            raise ValueError("Transcription result is None")
        return response.transcription_result

    async def transcribe_async(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> TranscriptionResult:
        """
        비동기 전사 (기존 audio_speech.py의 WhisperSTT.transcribe_async() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            audio: 오디오 파일 경로, AudioSegment, 또는 bytes
            language: 언어 코드
            task: 'transcribe' 또는 'translate'
            **kwargs: Whisper 추가 옵션

        Returns:
            TranscriptionResult
        """
        response = await self._audio_handler.handle_transcribe(
            audio=audio,
            language=language or self.language,
            task=task,
            model=self.model_name,
            device=self.device,
            **kwargs,
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        if not response.transcription_result:
            raise ValueError("Transcription result is None")
        return response.transcription_result


class TextToSpeech:
    """
    Text-to-Speech 통합 (Facade 패턴)

    여러 TTS 제공자를 지원합니다.

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        >>> tts = TextToSpeech(provider='openai', voice='alloy')
        >>> audio = tts.synthesize("Hello, world!")
        >>> audio.to_file('output.mp3')
    """

    def __init__(
        self,
        provider: Union[str, TTSProvider] = TTSProvider.OPENAI,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        """
        Args:
            provider: TTS 제공자
            api_key: API 키
            model: 모델 이름
            voice: 음성 ID
        """
        if isinstance(provider, str):
            provider = TTSProvider(provider)

        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.voice = voice

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..service.impl.audio_service_impl import AudioServiceImpl

        # AudioService 생성 (커스텀 의존성)
        audio_service = AudioServiceImpl(
            tts_provider=self.provider,
            tts_api_key=self.api_key,
            tts_model=self.model,
            tts_voice=self.voice,
        )

        # AudioHandler 생성 (직접 생성 - 커스텀 Service 사용)
        self._audio_handler = AudioHandler(audio_service)

    def synthesize(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """
        텍스트를 음성으로 변환 (기존 audio_speech.py의 TextToSpeech.synthesize() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            text: 변환할 텍스트
            voice: 음성 ID (provider별로 다름)
            speed: 속도 (0.5 ~ 2.0)
            **kwargs: 제공자별 추가 옵션

        Returns:
            AudioSegment
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._audio_handler.handle_synthesize(
                text=text,
                provider=self.provider.value,
                voice=voice or self.voice,
                speed=speed,
                api_key=self.api_key,
                model=self.model,
                **kwargs,
            )
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        if not response.audio_segment:
            raise ValueError("Audio segment is None")
        return response.audio_segment

    async def synthesize_async(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """
        비동기 음성 합성 (기존 audio_speech.py의 TextToSpeech.synthesize_async() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            text: 변환할 텍스트
            voice: 음성 ID
            speed: 속도 (0.5 ~ 2.0)
            **kwargs: 제공자별 추가 옵션

        Returns:
            AudioSegment
        """
        response = await self._audio_handler.handle_synthesize(
            text=text,
            provider=self.provider.value,
            voice=voice or self.voice,
            speed=speed,
            api_key=self.api_key,
            model=self.model,
            **kwargs,
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        if not response.audio_segment:
            raise ValueError("Audio segment is None")
        return response.audio_segment


class AudioRAG:
    """
    Audio RAG (Retrieval-Augmented Generation) (Facade 패턴)

    음성 파일을 전사하여 검색 가능하게 만들고,
    쿼리에 대해 관련 음성 세그먼트를 검색합니다.

    기존 API를 유지하면서 내부적으로는 Handler/Service 사용

    Example:
        >>> rag = AudioRAG()
        >>> rag.add_audio("meeting.wav")
        >>> results = rag.search("회의에서 논의된 내용은?")
    """

    def __init__(
        self,
        stt: Optional[WhisperSTT] = None,
        vector_store: Optional["VectorStoreProtocol"] = None,
        embedding_model: Optional["BaseEmbedding"] = None,
    ):
        """
        Args:
            stt: Speech-to-Text 모델
            vector_store: 벡터 저장소
            embedding_model: 임베딩 모델
        """
        self.stt = stt or WhisperSTT(model=WhisperModel.BASE)
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        # Handler/Service 초기화 (의존성 주입)
        self._init_services()

    def _init_services(self) -> None:
        """Service 및 Handler 초기화 (의존성 주입) - DI Container 사용"""
        from ..service.impl.audio_service_impl import AudioServiceImpl

        # stt에서 설정 가져오기
        whisper_model = self.stt.model_name if hasattr(self.stt, "model_name") else "base"
        whisper_device = self.stt.device if hasattr(self.stt, "device") else None
        whisper_language = self.stt.language if hasattr(self.stt, "language") else None

        # AudioService 생성 (커스텀 의존성)
        audio_service = AudioServiceImpl(
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_language=whisper_language,
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
        )

        # AudioHandler 생성 (직접 생성 - 커스텀 Service 사용)
        self._audio_handler = AudioHandler(audio_service)

    def add_audio(
        self,
        audio: Union[str, Path, AudioSegment],
        audio_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """
        오디오를 전사하고 RAG 시스템에 추가 (기존 audio_speech.py의 AudioRAG.add_audio() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            audio: 오디오 파일 또는 AudioSegment
            audio_id: 오디오 식별자
            metadata: 추가 메타데이터

        Returns:
            TranscriptionResult
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._audio_handler.handle_add_audio(
                audio=audio,
                audio_id=audio_id,
                metadata=metadata,
                language=self.stt.language if hasattr(self.stt, "language") else None,
                task="transcribe",
                model=self.stt.model_name if hasattr(self.stt, "model_name") else None,
                device=self.stt.device if hasattr(self.stt, "device") else None,
            )
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        if not response.transcription:
            raise ValueError("Transcription result is None")
        return response.transcription

    async def add_audio_async(
        self,
        audio: Union[str, Path, AudioSegment],
        audio_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """
        비동기 오디오 추가 (기존 audio_speech.py의 AudioRAG.add_audio_async() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            audio: 오디오 파일 또는 AudioSegment
            audio_id: 오디오 식별자
            metadata: 추가 메타데이터

        Returns:
            TranscriptionResult
        """
        await self._audio_handler.handle_add_audio(
            audio=audio,
            audio_id=audio_id,
            metadata=metadata,
            language=self.stt.language if hasattr(self.stt, "language") else None,
            task="transcribe",
            model=self.stt.model_name if hasattr(self.stt, "model_name") else None,
            device=self.stt.device if hasattr(self.stt, "device") else None,
        )

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        쿼리로 관련 음성 세그먼트 검색 (기존 audio_speech.py의 AudioRAG.search() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            **kwargs: 추가 검색 옵션

        Returns:
            검색 결과 리스트 (각 결과는 세그먼트 정보 포함)
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(
            self._audio_handler.handle_search_audio(query=query, top_k=top_k, **kwargs)
        )
        # DTO에서 값 추출 (기존 API 호환성 유지)
        return response.search_results or []

    def get_transcription(self, audio_id: str) -> Optional[TranscriptionResult]:
        """
        오디오 ID로 전사 결과 조회 (기존 audio_speech.py의 AudioRAG.get_transcription() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Args:
            audio_id: 오디오 식별자

        Returns:
            TranscriptionResult: 전사 결과 (없으면 None)
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(self._audio_handler.handle_get_transcription(audio_id=audio_id))
        # DTO에서 값 추출 (기존 API 호환성 유지)
        return response.transcription

    def list_audios(self) -> List[str]:
        """
        저장된 모든 오디오 ID 목록 (기존 audio_speech.py의 AudioRAG.list_audios() 정확히 마이그레이션)

        내부적으로 Handler를 사용하여 처리

        Returns:
            오디오 ID 리스트
        """
        # 동기 메서드이지만 내부적으로는 비동기 사용
        response = asyncio.run(self._audio_handler.handle_list_audios())
        # DTO에서 값 추출 (기존 API 호환성 유지)
        return response.audio_ids or []


# 편의 함수
def transcribe_audio(
    audio: Union[str, Path, AudioSegment, bytes],
    model: str = "base",
    language: Optional[str] = None,
    **kwargs,
) -> TranscriptionResult:
    """
    간편한 음성 전사 함수 (기존 audio_speech.py의 transcribe_audio() 정확히 마이그레이션)

    Args:
        audio: 오디오 파일 경로, AudioSegment, 또는 bytes
        model: Whisper 모델 크기
        language: 언어 코드
        **kwargs: 추가 옵션

    Returns:
        TranscriptionResult

    Example:
        >>> result = transcribe_audio('audio.mp3', model='base', language='en')
        >>> print(result.text)
    """
    stt = WhisperSTT(model=model, language=language)
    return stt.transcribe(audio, **kwargs)


def text_to_speech(
    text: str,
    provider: str = "openai",
    voice: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    **kwargs,
) -> AudioSegment:
    """
    간편한 TTS 함수 (기존 audio_speech.py의 text_to_speech() 정확히 마이그레이션)

    Args:
        text: 변환할 텍스트
        provider: TTS 제공자 ('openai', 'google', 'azure', 'elevenlabs')
        voice: 음성 ID
        output_file: 저장할 파일 경로 (선택)
        **kwargs: 제공자별 옵션

    Returns:
        AudioSegment

    Example:
        >>> audio = text_to_speech("Hello", provider='openai', voice='alloy')
        >>> audio.to_file('output.mp3')
    """
    tts = TextToSpeech(provider=provider, voice=voice)
    audio = tts.synthesize(text, **kwargs)

    if output_file:
        audio.to_file(output_file)

    return audio
