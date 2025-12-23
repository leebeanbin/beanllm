"""
Audio Enums - 오디오 관련 열거형
"""

from enum import Enum


class WhisperModel(Enum):
    """Whisper 모델 크기"""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class TTSProvider(Enum):
    """TTS 제공자"""

    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"
