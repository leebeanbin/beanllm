"""
IAudioService - Audio 서비스 인터페이스
SOLID 원칙:
- ISP: Audio 관련 메서드만 포함
- DIP: 인터페이스에 의존
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..dto.request.audio_request import AudioRequest
from ..dto.response.audio_response import AudioResponse


class IAudioService(ABC):
    """
    Audio 서비스 인터페이스

    책임:
    - Audio 비즈니스 로직 정의만
    - 검증, 에러 처리 없음 (Handler에서 처리)

    SOLID:
    - ISP: Audio 관련 메서드만 (작은 인터페이스)
    - DIP: 구현체가 아닌 인터페이스에 의존
    """

    @abstractmethod
    async def transcribe(self, request: AudioRequest) -> AudioResponse:
        """
        음성을 텍스트로 변환 (Speech-to-Text)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (transcription_result 필드 포함)

        책임:
            - 음성 전사 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def synthesize(self, request: AudioRequest) -> AudioResponse:
        """
        텍스트를 음성으로 변환 (Text-to-Speech)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (audio_segment 필드 포함)

        책임:
            - 음성 합성 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def add_audio(self, request: AudioRequest) -> AudioResponse:
        """
        오디오를 전사하고 RAG 시스템에 추가 (AudioRAG)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (transcription 필드 포함)

        책임:
            - 오디오 추가 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def search_audio(self, request: AudioRequest) -> AudioResponse:
        """
        쿼리로 관련 음성 세그먼트 검색 (AudioRAG)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (search_results 필드 포함)

        책임:
            - 오디오 검색 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def get_transcription(self, request: AudioRequest) -> AudioResponse:
        """
        오디오 ID로 전사 결과 조회 (AudioRAG)

        Args:
            request: Audio 요청 DTO (audio_id 필드 사용)

        Returns:
            AudioResponse: Audio 응답 DTO (transcription 필드 포함)

        책임:
            - 전사 결과 조회 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass

    @abstractmethod
    async def list_audios(self, request: AudioRequest) -> AudioResponse:
        """
        저장된 모든 오디오 ID 목록 조회 (AudioRAG)

        Args:
            request: Audio 요청 DTO

        Returns:
            AudioResponse: Audio 응답 DTO (audio_ids 필드 포함)

        책임:
            - 오디오 목록 조회 비즈니스 로직만
            - 검증 없음 (Handler에서 처리)
            - 에러 처리 없음 (Handler에서 처리)
        """
        pass
