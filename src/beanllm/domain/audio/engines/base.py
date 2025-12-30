"""
Base STT Engine

모든 STT 엔진이 상속받는 기본 클래스.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..models import STTConfig
from ..types import TranscriptionResult


class BaseSTTEngine(ABC):
    """
    STT 엔진 베이스 클래스

    모든 STT 엔진(Whisper, Parakeet, Canary 등)은 이 클래스를 상속받아 구현합니다.

    Example:
        ```python
        class WhisperEngine(BaseSTTEngine):
            def transcribe(self, audio_path, config):
                # Whisper 전사 로직
                return {
                    "text": "...",
                    "segments": [...],
                    "language": "ko"
                }
        ```
    """

    @abstractmethod
    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        오디오 전사 (음성 → 텍스트)

        Args:
            audio_path: 오디오 파일 경로 또는 numpy array
            config: STT 설정

        Returns:
            Dict: 전사 결과
                {
                    "text": str,  # 전체 텍스트
                    "segments": List[Dict],  # 세그먼트 리스트
                    "language": str,  # 감지된 언어
                    "duration": float,  # 오디오 길이 (초)
                    "metadata": dict  # 추가 메타데이터
                }

        Example:
            ```python
            result = engine.transcribe("audio.mp3", config)
            print(result["text"])
            print(result["language"])
            ```
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
