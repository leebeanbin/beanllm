"""
AudioResponse - Audio 응답 DTO
책임: Audio 응답 데이터만 전달
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import ConfigDict

from beanllm.dto.response.base_response import BaseResponse

if TYPE_CHECKING:
    from beanllm.domain.audio import AudioSegment, TranscriptionResult


class AudioResponse(BaseResponse):
    """
    Audio 응답 DTO

    책임:
    - 응답 데이터 구조 정의만
    - 변환 로직 없음 (Service에서 처리)
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, arbitrary_types_allowed=True
    )

    # transcribe 메서드 응답
    transcription_result: Optional["TranscriptionResult"] = None

    # synthesize 메서드 응답
    audio_segment: Optional["AudioSegment"] = None

    # search 메서드 응답 (AudioRAG)
    search_results: Optional[list[dict[str, object]]] = None

    # get_transcription 메서드 응답 (AudioRAG)
    transcription: Optional["TranscriptionResult"] = None

    # list_audios 메서드 응답 (AudioRAG)
    audio_ids: Optional[list[str]] = None

    # 메타데이터
    metadata: dict[str, object] = {}
