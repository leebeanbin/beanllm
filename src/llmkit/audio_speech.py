"""
Audio & Speech Processing

Whisper (Speech-to-Text), Text-to-Speech, Audio RAG 등
음성 처리 기능을 제공합니다.

Mathematical Foundations:
=======================

1. Fourier Transform (푸리에 변환):
   F(ω) = ∫_{-∞}^{∞} f(t) e^{-iωt} dt

   Discrete Fourier Transform (DFT):
   X[k] = Σ_{n=0}^{N-1} x[n] e^{-i2πkn/N}

2. Short-Time Fourier Transform (STFT):
   STFT{x[n]}(m, ω) = Σ_{n=-∞}^{∞} x[n] w[n - m] e^{-iωn}

   where w[n] is window function

3. Mel-Frequency Cepstral Coefficients (MFCC):
   mel(f) = 2595 × log₁₀(1 + f/700)

   Steps:
   1. Frame signal
   2. Apply FFT
   3. Mel filterbank
   4. Log
   5. DCT → MFCC

4. Dynamic Time Warping (DTW):
   DTW(X, Y) = min Σ d(x_i, y_j)

   for optimal alignment path

5. CTC Loss (Connectionist Temporal Classification):
   L_CTC = -log Σ_{π ∈ B^{-1}(y)} P(π|x)

   where B is collapsing function (removing blanks and repeats)

References:
----------
- Rabiner, L. R. (1989). "A tutorial on hidden Markov models". IEEE
- Graves, A., et al. (2006). "Connectionist Temporal Classification". ICML
- Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)

Author: LLMKit Team
"""

import asyncio
import base64
import os
import tempfile
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================================
# Part 1: Audio Data Structures
# ============================================================================


@dataclass
class AudioSegment:
    """
    음성 세그먼트

    Attributes:
        audio_data: Raw audio bytes
        sample_rate: 샘플링 레이트 (Hz)
        duration: 길이 (초)
        format: 오디오 포맷 (wav, mp3, etc.)
        channels: 채널 수 (1=mono, 2=stereo)
        metadata: 추가 메타데이터
    """

    audio_data: bytes
    sample_rate: int = 16000
    duration: float = 0.0
    format: str = "wav"
    channels: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "AudioSegment":
        """파일에서 AudioSegment 생성"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        with open(file_path, "rb") as f:
            audio_data = f.read()

        # WAV 파일인 경우 메타데이터 추출
        if file_path.suffix.lower() == ".wav":
            with wave.open(str(file_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                frames = wav_file.getnframes()
                duration = frames / sample_rate

            return cls(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration=duration,
                format="wav",
                channels=channels,
                metadata={"file_path": str(file_path)},
            )
        else:
            # 다른 포맷은 기본값 사용
            return cls(
                audio_data=audio_data,
                format=file_path.suffix.lstrip("."),
                metadata={"file_path": str(file_path)},
            )

    def to_file(self, file_path: Union[str, Path]):
        """파일로 저장"""
        file_path = Path(file_path)
        with open(file_path, "wb") as f:
            f.write(self.audio_data)

    def to_base64(self) -> str:
        """Base64 인코딩"""
        return base64.b64encode(self.audio_data).decode("utf-8")


@dataclass
class TranscriptionSegment:
    """
    전사(Transcription) 세그먼트

    Attributes:
        text: 전사된 텍스트
        start: 시작 시간 (초)
        end: 종료 시간 (초)
        confidence: 신뢰도 (0-1)
        language: 언어 코드
        speaker: 화자 ID (선택)
    """

    text: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0
    language: Optional[str] = None
    speaker: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.start:.2f}s - {self.end:.2f}s] {self.text}"


@dataclass
class TranscriptionResult:
    """
    전사 결과

    Attributes:
        text: 전체 전사 텍스트
        segments: 세그먼트 리스트
        language: 감지된 언어
        duration: 오디오 길이
        model: 사용된 모델
        metadata: 추가 메타데이터
    """

    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None
    duration: float = 0.0
    model: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


# ============================================================================
# Part 2: Speech-to-Text (Whisper)
# ============================================================================


class WhisperModel(Enum):
    """Whisper 모델 크기"""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class WhisperSTT:
    """
    Whisper Speech-to-Text

    OpenAI의 Whisper 모델을 사용한 음성 인식

    Mathematical Foundation:
        Whisper는 Transformer 기반 encoder-decoder 모델:

        1. Audio → Mel Spectrogram
           mel(f) = 2595 × log₁₀(1 + f/700)

        2. Encoder: Multi-head self-attention
           Attention(Q, K, V) = softmax(QK^T / √d_k) V

        3. Decoder: Autoregressive text generation
           P(y|x) = Π_{t=1}^T P(y_t | y_{<t}, x)

        4. Training: Cross-entropy loss
           L = -Σ log P(y_t | y_{<t}, x)
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
        self._model = None

    def _load_model(self):
        """모델 로드 (lazy loading)"""
        if self._model is not None:
            return

        try:
            import whisper

            self._model = whisper.load_model(self.model_name, device=self.device)
        except ImportError:
            raise ImportError(
                "openai-whisper not installed. " "Install with: pip install openai-whisper"
            )

    def transcribe(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> TranscriptionResult:
        """
        음성을 텍스트로 변환

        Args:
            audio: 오디오 파일 경로, AudioSegment, 또는 bytes
            language: 언어 코드 (예: 'en', 'ko')
            task: 'transcribe' 또는 'translate' (영어로 번역)
            **kwargs: Whisper 추가 옵션

        Returns:
            TranscriptionResult

        Example:
            >>> stt = WhisperSTT(model='base')
            >>> result = stt.transcribe('audio.mp3', language='en')
            >>> print(result.text)
        """
        self._load_model()

        # 오디오 준비
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

        # 전사 실행
        language = language or self.language

        options = {"language": language, "task": task, **kwargs}

        result = self._model.transcribe(audio_path, **options)

        # 결과 변환
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

        # 임시 파일 정리
        if isinstance(audio, (AudioSegment, bytes)):
            try:
                os.unlink(audio_path)
            except:
                pass

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
            duration=result.get("duration", 0.0),
            model=self.model_name,
            metadata=result,
        )

    async def transcribe_async(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs,
    ) -> TranscriptionResult:
        """비동기 전사"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio, language, task, **kwargs)


# ============================================================================
# Part 3: Text-to-Speech
# ============================================================================


class TTSProvider(Enum):
    """TTS 제공자"""

    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"


class TextToSpeech:
    """
    Text-to-Speech 통합

    여러 TTS 제공자를 지원합니다.

    Mathematical Foundation:
        Modern TTS는 주로 neural vocoder 사용:

        1. Text → Phonemes
        2. Phonemes → Mel Spectrogram (Tacotron2, FastSpeech)
           mel_t = model(phonemes)

        3. Mel Spectrogram → Audio Waveform (WaveNet, HiFi-GAN)
           y = vocoder(mel)

        WaveNet:
        p(y_t | y_{<t}) = softmax(f(y_{<t}))
        where f is dilated causal convolutions
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
        self.api_key = api_key or os.getenv(f"{provider.value.upper()}_API_KEY")
        self.model = model
        self.voice = voice

    def synthesize(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice: 음성 ID (provider별로 다름)
            speed: 속도 (0.5 ~ 2.0)
            **kwargs: 제공자별 추가 옵션

        Returns:
            AudioSegment

        Example:
            >>> tts = TextToSpeech(provider='openai', voice='alloy')
            >>> audio = tts.synthesize("Hello, world!")
            >>> audio.to_file('output.mp3')
        """
        voice = voice or self.voice

        if self.provider == TTSProvider.OPENAI:
            return self._synthesize_openai(text, voice, speed, **kwargs)
        elif self.provider == TTSProvider.GOOGLE:
            return self._synthesize_google(text, voice, speed, **kwargs)
        elif self.provider == TTSProvider.AZURE:
            return self._synthesize_azure(text, voice, speed, **kwargs)
        elif self.provider == TTSProvider.ELEVENLABS:
            return self._synthesize_elevenlabs(text, voice, speed, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _synthesize_openai(
        self, text: str, voice: str = "alloy", speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """OpenAI TTS"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. pip install openai")

        client = OpenAI(api_key=self.api_key)

        response = client.audio.speech.create(
            model=self.model or "tts-1", voice=voice, input=text, speed=speed, **kwargs
        )

        # Response is audio bytes
        audio_data = response.content

        return AudioSegment(
            audio_data=audio_data,
            sample_rate=24000,  # OpenAI TTS default
            format="mp3",
            metadata={"provider": "openai", "voice": voice, "model": self.model or "tts-1"},
        )

    def _synthesize_google(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """Google Cloud TTS"""
        try:
            from google.cloud import texttospeech
        except ImportError:
            raise ImportError(
                "google-cloud-texttospeech not installed. " "pip install google-cloud-texttospeech"
            )

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Voice parameters
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=kwargs.get("language_code", "en-US"), name=voice
        )

        # Audio config
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

    def _synthesize_azure(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """Azure TTS"""
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            raise ImportError(
                "azure-cognitiveservices-speech not installed. "
                "pip install azure-cognitiveservices-speech"
            )

        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key, region=kwargs.get("region", "eastus")
        )

        if voice:
            speech_config.speech_synthesis_voice_name = voice

        # Synthesize to in-memory stream
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
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

    def _synthesize_elevenlabs(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """ElevenLabs TTS"""
        import requests

        if not voice:
            voice = "21m00Tcm4TlvDq8ikWAM"  # Default voice

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }

        data = {
            "text": text,
            "model_id": self.model or "eleven_monolingual_v1",
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

    async def synthesize_async(
        self, text: str, voice: Optional[str] = None, speed: float = 1.0, **kwargs
    ) -> AudioSegment:
        """비동기 음성 합성"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text, voice, speed)


# ============================================================================
# Part 4: Audio RAG
# ============================================================================


class AudioRAG:
    """
    Audio RAG (Retrieval-Augmented Generation)

    음성 파일을 전사하여 검색 가능하게 만들고,
    쿼리에 대해 관련 음성 세그먼트를 검색합니다.

    Workflow:
    1. Audio → Transcription (Whisper)
    2. Transcription → Embeddings
    3. Store in Vector DB
    4. Query → Retrieve relevant segments
    5. Generate response with LLM
    """

    def __init__(self, stt: Optional[WhisperSTT] = None, vector_store=None, embedding_model=None):
        """
        Args:
            stt: Speech-to-Text 모델
            vector_store: 벡터 저장소
            embedding_model: 임베딩 모델
        """
        self.stt = stt or WhisperSTT(model=WhisperModel.BASE)
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self._transcriptions: Dict[str, TranscriptionResult] = {}

    def add_audio(
        self,
        audio: Union[str, Path, AudioSegment],
        audio_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """
        오디오를 전사하고 RAG 시스템에 추가

        Args:
            audio: 오디오 파일 또는 AudioSegment
            audio_id: 오디오 식별자
            metadata: 추가 메타데이터

        Returns:
            TranscriptionResult
        """
        # 전사
        transcription = self.stt.transcribe(audio)

        # ID 생성
        if audio_id is None:
            if isinstance(audio, (str, Path)):
                audio_id = str(Path(audio).stem)
            else:
                audio_id = f"audio_{len(self._transcriptions)}"

        # 저장
        self._transcriptions[audio_id] = transcription

        # Vector store에 추가 (있는 경우)
        if self.vector_store is not None and self.embedding_model is not None:
            # 각 세그먼트를 별도 문서로 추가
            from llmkit import Document

            documents = []
            for i, segment in enumerate(transcription.segments):
                doc = Document(
                    content=segment.text,
                    metadata={
                        "audio_id": audio_id,
                        "segment_id": i,
                        "start": segment.start,
                        "end": segment.end,
                        "language": segment.language,
                        **(metadata or {}),
                    },
                )
                documents.append(doc)

            self.vector_store.add_documents(documents, self.embedding_model)

        return transcription

    async def add_audio_async(
        self,
        audio: Union[str, Path, AudioSegment],
        audio_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """비동기 오디오 추가"""
        # 전사
        transcription = await self.stt.transcribe_async(audio)

        # ID 생성
        if audio_id is None:
            if isinstance(audio, (str, Path)):
                audio_id = str(Path(audio).stem)
            else:
                audio_id = f"audio_{len(self._transcriptions)}"

        # 저장
        self._transcriptions[audio_id] = transcription

        # Vector store에 추가
        if self.vector_store is not None and self.embedding_model is not None:
            from llmkit import Document

            documents = []
            for i, segment in enumerate(transcription.segments):
                doc = Document(
                    content=segment.text,
                    metadata={
                        "audio_id": audio_id,
                        "segment_id": i,
                        "start": segment.start,
                        "end": segment.end,
                        "language": segment.language,
                        **(metadata or {}),
                    },
                )
                documents.append(doc)

            self.vector_store.add_documents(documents, self.embedding_model)

        return transcription

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        쿼리로 관련 음성 세그먼트 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            **kwargs: 추가 검색 옵션

        Returns:
            검색 결과 리스트 (각 결과는 세그먼트 정보 포함)
        """
        if self.vector_store is None:
            # Fallback: 단순 텍스트 매칭
            results = []
            for audio_id, transcription in self._transcriptions.items():
                for i, segment in enumerate(transcription.segments):
                    if query.lower() in segment.text.lower():
                        results.append({"audio_id": audio_id, "segment": segment, "score": 1.0})

            return results[:top_k]

        # Vector search
        search_results = self.vector_store.search(query, k=top_k, **kwargs)

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

        return results

    def get_transcription(self, audio_id: str) -> Optional[TranscriptionResult]:
        """오디오 ID로 전사 결과 조회"""
        return self._transcriptions.get(audio_id)

    def list_audios(self) -> List[str]:
        """저장된 모든 오디오 ID 목록"""
        return list(self._transcriptions.keys())


# ============================================================================
# Convenience Functions
# ============================================================================


def transcribe_audio(
    audio: Union[str, Path, AudioSegment, bytes],
    model: str = "base",
    language: Optional[str] = None,
    **kwargs,
) -> TranscriptionResult:
    """
    간편한 음성 전사 함수

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
    간편한 TTS 함수

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
