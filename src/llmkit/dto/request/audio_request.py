"""
AudioRequest - Audio 요청 DTO
책임: Audio 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from ...domain.audio import AudioSegment


@dataclass
class AudioRequest:
    """
    Audio 요청 DTO

    책임:
    - 데이터 구조 정의만
    - 검증 없음 (Handler에서 처리)
    - 비즈니스 로직 없음 (Service에서 처리)
    """

    # transcribe 메서드용
    audio: Optional[Union[str, Path, "AudioSegment", bytes]] = None
    language: Optional[str] = None
    task: str = "transcribe"  # 'transcribe' 또는 'translate'
    model: Optional[str] = None  # Whisper 모델 크기
    device: Optional[str] = None  # 디바이스 ('cpu', 'cuda', 'mps')

    # synthesize 메서드용
    text: Optional[str] = None
    provider: Optional[str] = None  # TTS 제공자
    voice: Optional[str] = None  # 음성 ID
    speed: float = 1.0  # 속도 (0.5 ~ 2.0)
    api_key: Optional[str] = None  # API 키
    tts_model: Optional[str] = None  # TTS 모델

    # add_audio 메서드용 (AudioRAG)
    audio_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # search 메서드용 (AudioRAG)
    query: Optional[str] = None
    top_k: int = 5

    # 추가 파라미터
    extra_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """기본값 설정"""
        if self.extra_params is None:
            self.extra_params = {}
        if self.metadata is None:
            self.metadata = {}
