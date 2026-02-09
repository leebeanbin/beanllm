"""
AudioRequest - Audio 요청 DTO
책임: Audio 요청 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from beanllm.domain.audio import AudioSegment


@dataclass(slots=True, kw_only=True)
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
    task: str = "transcribe"
    model: Optional[str] = None
    device: Optional[str] = None

    # synthesize 메서드용
    text: Optional[str] = None
    provider: Optional[str] = None
    voice: Optional[str] = None
    speed: float = 1.0
    api_key: Optional[str] = None
    tts_model: Optional[str] = None

    # add_audio 메서드용 (AudioRAG)
    audio_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # search 메서드용 (AudioRAG)
    query: Optional[str] = None
    top_k: int = 5

    # 추가 파라미터
    extra_params: Dict[str, Any] = field(default_factory=dict)
