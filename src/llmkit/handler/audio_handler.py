"""
AudioHandler - Audio 요청 처리 (Controller 역할)
책임 분리:
- 모든 if-else/try-catch 처리
- 입력 검증
- DTO 변환
- 결과 출력
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..decorators.error_handler import handle_errors
from ..decorators.logger import log_handler_call
from ..decorators.validation import validate_input
from ..domain.audio import AudioSegment
from ..dto.request.audio_request import AudioRequest
from ..dto.response.audio_response import AudioResponse
from ..service.audio_service import IAudioService
from .base_handler import BaseHandler


class AudioHandler(BaseHandler):
    """
    Audio 요청 처리 Handler

    책임:
    - 입력 검증 (if-else)
    - 에러 처리 (try-catch)
    - DTO 변환
    - Service 호출
    - 비즈니스 로직 없음
    """

    def __init__(self, audio_service: IAudioService) -> None:
        """
        의존성 주입

        Args:
            audio_service: Audio 서비스 (인터페이스에 의존 - DIP)
        """
        super().__init__(audio_service)
        self._audio_service = audio_service  # BaseHandler._service와 동일하지만 명시적으로 유지

    @log_handler_call
    @handle_errors(error_message="Audio transcription failed")
    @validate_input(
        required_params=["audio"],
        param_types={"audio": (str, Path, AudioSegment, bytes), "language": str, "task": str},
    )
    async def handle_transcribe(
        self,
        audio: Union[str, Path, AudioSegment, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        model: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        """
        음성 전사 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            audio: 오디오 파일 경로, AudioSegment, 또는 bytes
            language: 언어 코드 (예: 'en', 'ko')
            task: 'transcribe' 또는 'translate' (영어로 번역)
            model: Whisper 모델 크기
            device: 디바이스 ('cpu', 'cuda', 'mps')
            **kwargs: 추가 파라미터

        Returns:
            AudioResponse: Audio 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = AudioRequest(
            audio=audio,
            language=language,
            task=task,
            model=model,
            device=device,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._call_service("transcribe", request)

    @log_handler_call
    @handle_errors(error_message="Audio synthesis failed")
    @validate_input(
        required_params=["text"],
        param_types={"text": str, "voice": str, "speed": float},
        param_ranges={"speed": (0.5, 2.0)},
    )
    async def handle_synthesize(
        self,
        text: str,
        provider: Optional[str] = None,
        voice: Optional[str] = None,
        speed: float = 1.0,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        """
        텍스트를 음성으로 변환 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            text: 변환할 텍스트
            provider: TTS 제공자 ('openai', 'google', 'azure', 'elevenlabs')
            voice: 음성 ID (provider별로 다름)
            speed: 속도 (0.5 ~ 2.0)
            api_key: API 키
            model: TTS 모델
            **kwargs: 추가 파라미터

        Returns:
            AudioResponse: Audio 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = AudioRequest(
            text=text,
            provider=provider,
            voice=voice,
            speed=speed,
            api_key=api_key,
            tts_model=model,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._audio_service.synthesize(request)

    @log_handler_call
    @handle_errors(error_message="Audio RAG add_audio failed")
    @validate_input(
        required_params=["audio"],
        param_types={"audio": (str, Path, AudioSegment), "audio_id": str},
    )
    async def handle_add_audio(
        self,
        audio: Union[str, Path, AudioSegment],
        audio_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        model: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> AudioResponse:
        """
        오디오를 전사하고 RAG 시스템에 추가 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            audio: 오디오 파일 또는 AudioSegment
            audio_id: 오디오 식별자
            metadata: 추가 메타데이터
            language: 언어 코드
            task: 'transcribe' 또는 'translate'
            model: Whisper 모델 크기
            device: 디바이스
            **kwargs: 추가 파라미터

        Returns:
            AudioResponse: Audio 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = AudioRequest(
            audio=audio,
            audio_id=audio_id,
            metadata=metadata,
            language=language,
            task=task,
            model=model,
            device=device,
            extra_params=kwargs,
        )

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._call_service("add_audio", request)

    @log_handler_call
    @handle_errors(error_message="Audio RAG search failed")
    @validate_input(
        required_params=["query"],
        param_types={"query": str, "top_k": int},
        param_ranges={"top_k": (1, None)},
    )
    async def handle_search_audio(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> AudioResponse:
        """
        쿼리로 관련 음성 세그먼트 검색 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 결과 수
            **kwargs: 추가 파라미터

        Returns:
            AudioResponse: Audio 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = AudioRequest(query=query, top_k=top_k, extra_params=kwargs)

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._call_service("search_audio", request)

    @log_handler_call
    @handle_errors(error_message="Audio RAG get_transcription failed")
    @validate_input(required_params=["audio_id"], param_types={"audio_id": str})
    async def handle_get_transcription(
        self,
        audio_id: str,
        **kwargs: Any,
    ) -> AudioResponse:
        """
        오디오 ID로 전사 결과 조회 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            audio_id: 오디오 식별자
            **kwargs: 추가 파라미터

        Returns:
            AudioResponse: Audio 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = AudioRequest(audio_id=audio_id, extra_params=kwargs)

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._audio_service.get_transcription(request)

    @log_handler_call
    @handle_errors(error_message="Audio RAG list_audios failed")
    async def handle_list_audios(
        self,
        **kwargs: Any,
    ) -> AudioResponse:
        """
        저장된 모든 오디오 ID 목록 조회 요청 처리 (모든 검증 및 에러 처리 포함)

        Args:
            **kwargs: 추가 파라미터

        Returns:
            AudioResponse: Audio 응답 DTO

        책임:
            - 입력 검증 (decorator로 처리)
            - 에러 처리 (decorator로 처리)
            - DTO 변환
            - Service 호출
        """
        # DTO 생성
        request = AudioRequest(extra_params=kwargs)

        # Service 호출 (에러 처리는 decorator가 담당)
        return await self._call_service("list_audios", request)
