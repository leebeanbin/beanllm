"""
Audio Types - 오디오 및 전사 데이터 구조
"""

import base64
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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
