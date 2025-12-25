"""
AudioResponse - Audio 응답 DTO
책임: Audio 응답 데이터만 전달
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ...domain.audio import AudioSegment, TranscriptionResult


@dataclass
class AudioResponse:
    """
    Audio 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    # transcribe 메서드 응답
    transcription_result: Optional["TranscriptionResult"] = None

    # synthesize 메서드 응답
    audio_segment: Optional["AudioSegment"] = None

    # search 메서드 응답 (AudioRAG)
    search_results: Optional[List[Dict[str, Any]]] = None

    # get_transcription 메서드 응답 (AudioRAG)
    transcription: Optional["TranscriptionResult"] = None

    # list_audios 메서드 응답 (AudioRAG)
    audio_ids: Optional[List[str]] = None

    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """기본값 설정"""
        if self.metadata is None:
            self.metadata = {}
