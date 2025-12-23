"""
AudioHandler 테스트 - Audio Handler 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock
from pathlib import Path

from llmkit.dto.request.audio_request import AudioRequest
from llmkit.dto.response.audio_response import AudioResponse
from llmkit.handler.audio_handler import AudioHandler


class TestAudioHandler:
    """AudioHandler 테스트"""

    @pytest.fixture
    def mock_audio_service(self):
        """Mock AudioService"""
        from llmkit.domain.audio import TranscriptionResult, TranscriptionSegment, AudioSegment
        
        service = Mock()
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

        # handle_transcribe는 TranscriptionResult를 반환
        from llmkit.domain.audio import TranscriptionResult
        
        result = await audio_handler.handle_transcribe(
            audio=str(audio_file),
            language="en",
        )

        assert result is not None
        assert isinstance(result, TranscriptionResult)

    @pytest.mark.asyncio
    async def test_handle_synthesize(self, audio_handler):
        """음성 합성 테스트"""
        # handle_synthesize는 AudioSegment를 반환
        from llmkit.domain.audio import AudioSegment
        
        audio_segment = await audio_handler.handle_synthesize(
            text="Hello world",
            provider="openai",
            voice="alloy",
        )

        assert audio_segment is not None
        assert isinstance(audio_segment, AudioSegment)

    @pytest.mark.asyncio
    async def test_handle_add_audio(self, audio_handler, tmp_path):
        """오디오 추가 테스트"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio")

        # handle_add_audio는 TranscriptionResult를 반환
        from llmkit.domain.audio import TranscriptionResult
        
        result = await audio_handler.handle_add_audio(
            audio=str(audio_file),
            audio_id="audio_1",
        )

        assert result is not None
        assert isinstance(result, TranscriptionResult)

    @pytest.mark.asyncio
    async def test_handle_search_audio(self, audio_handler):
        """오디오 검색 테스트"""
        # handle_search_audio는 List를 반환
        results = await audio_handler.handle_search_audio(
            query="test query",
            top_k=5,
        )

        assert results is not None
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_handle_get_transcription(self, audio_handler):
        """전사 결과 조회 테스트"""
        # handle_get_transcription은 TranscriptionResult를 반환
        from llmkit.domain.audio import TranscriptionResult
        
        result = await audio_handler.handle_get_transcription(
            audio_id="audio_1",
        )

        assert result is not None
        assert isinstance(result, TranscriptionResult)

    @pytest.mark.asyncio
    async def test_handle_list_audios(self, audio_handler):
        """오디오 목록 조회 테스트"""
        # handle_list_audios는 List[str]을 반환
        audio_ids = await audio_handler.handle_list_audios()

        assert audio_ids is not None
        assert isinstance(audio_ids, list)
        assert len(audio_ids) == 2


