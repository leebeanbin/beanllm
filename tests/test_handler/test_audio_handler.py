"""
AudioHandler 테스트 - Audio Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock
from pathlib import Path

from beanllm.dto.request.audio_request import AudioRequest
from beanllm.dto.response.audio_response import AudioResponse
from beanllm.handler.audio_handler import AudioHandler


class TestAudioHandler:
    """AudioHandler 테스트"""

    @pytest.fixture
    def mock_audio_service(self):
        """Mock AudioService"""
        from beanllm.domain.audio import TranscriptionResult, TranscriptionSegment, AudioSegment
        from beanllm.service.audio_service import IAudioService

        service = Mock(spec=IAudioService)
        service.transcribe = AsyncMock(
            return_value=AudioResponse(
                transcription_result=TranscriptionResult(
                    text="Hello world",
                    segments=[],
                    language="en",
                    duration=1.0,
                    model="base",
                )
            )
        )
        service.synthesize = AsyncMock(
            return_value=AudioResponse(
                audio_segment=AudioSegment(
                    audio_data=b"fake audio",
                    format="mp3",
                    sample_rate=24000,
                )
            )
        )
        service.add_audio = AsyncMock(
            return_value=AudioResponse(
                transcription=TranscriptionResult(
                    text="Transcribed",
                    segments=[],
                    language="en",
                    duration=1.0,
                    model="base",
                )
            )
        )
        service.search_audio = AsyncMock(
            return_value=AudioResponse(
                search_results=[]
            )
        )
        service.get_transcription = AsyncMock(
            return_value=AudioResponse(
                transcription=TranscriptionResult(
                    text="Transcribed",
                    segments=[],
                    language="en",
                    duration=1.0,
                    model="base",
                )
            )
        )
        service.list_audios = AsyncMock(
            return_value=AudioResponse(
                audio_ids=["audio_1", "audio_2"]
            )
        )
        return service

    @pytest.fixture
    def audio_handler(self, mock_audio_service):
        """AudioHandler 인스턴스"""
        return AudioHandler(audio_service=mock_audio_service)

    @pytest.mark.asyncio
    async def test_handle_transcribe(self, audio_handler, tmp_path):
        """음성 전사 테스트"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")

        # handle_transcribe는 AudioResponse를 반환
        from beanllm.domain.audio import TranscriptionResult
        from beanllm.dto.response import AudioResponse

        result = await audio_handler.handle_transcribe(
            audio=str(audio_file),
            language="en",
        )

        assert result is not None
        assert isinstance(result, AudioResponse)
        assert result.transcription_result is not None
        assert isinstance(result.transcription_result, TranscriptionResult)

    @pytest.mark.asyncio
    async def test_handle_synthesize(self, audio_handler):
        """음성 합성 테스트"""
        # handle_synthesize는 AudioResponse를 반환
        from beanllm.domain.audio import AudioSegment
        from beanllm.dto.response import AudioResponse

        result = await audio_handler.handle_synthesize(
            text="Hello world",
            provider="openai",
            voice="alloy",
        )

        assert result is not None
        assert isinstance(result, AudioResponse)
        assert result.audio_segment is not None
        assert isinstance(result.audio_segment, AudioSegment)

    @pytest.mark.asyncio
    async def test_handle_add_audio(self, audio_handler, tmp_path):
        """오디오 추가 테스트"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")

        # handle_add_audio는 AudioResponse를 반환
        from beanllm.domain.audio import TranscriptionResult
        from beanllm.dto.response import AudioResponse

        result = await audio_handler.handle_add_audio(
            audio=str(audio_file),
            audio_id="audio_1",
        )

        assert result is not None
        assert isinstance(result, AudioResponse)
        assert result.transcription is not None
        assert isinstance(result.transcription, TranscriptionResult)

    @pytest.mark.asyncio
    async def test_handle_search_audio(self, audio_handler):
        """오디오 검색 테스트"""
        # handle_search_audio는 AudioResponse를 반환
        from beanllm.dto.response import AudioResponse

        result = await audio_handler.handle_search_audio(
            query="test query",
            top_k=5,
        )

        assert result is not None
        assert isinstance(result, AudioResponse)
        assert result.search_results is not None
        assert isinstance(result.search_results, list)

    @pytest.mark.asyncio
    async def test_handle_get_transcription(self, audio_handler):
        """전사 결과 조회 테스트"""
        # handle_get_transcription은 AudioResponse를 반환
        from beanllm.domain.audio import TranscriptionResult
        from beanllm.dto.response import AudioResponse

        result = await audio_handler.handle_get_transcription(
            audio_id="audio_1",
        )

        assert result is not None
        assert isinstance(result, AudioResponse)
        assert result.transcription is not None
        assert isinstance(result.transcription, TranscriptionResult)

    @pytest.mark.asyncio
    async def test_handle_list_audios(self, audio_handler):
        """오디오 목록 조회 테스트"""
        # handle_list_audios는 AudioResponse를 반환
        from beanllm.dto.response import AudioResponse

        result = await audio_handler.handle_list_audios()

        assert result is not None
        assert isinstance(result, AudioResponse)
        assert result.audio_ids is not None
        assert isinstance(result.audio_ids, list)
        assert len(result.audio_ids) == 2


