"""
AudioService 테스트 - Audio 서비스 구현체 테스트
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path

from llmkit.dto.request.audio_request import AudioRequest
from llmkit.dto.response.audio_response import AudioResponse
from llmkit.domain.audio import AudioSegment, TranscriptionResult, TranscriptionSegment, TTSProvider
from llmkit.service.impl.audio_service_impl import AudioServiceImpl


class TestAudioService:
    """AudioService 테스트"""

    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper 모델"""
        model = Mock()
        model.transcribe = Mock(
            return_value={
                "text": "Hello world",
                "segments": [
                    {
                        "text": "Hello world",
                        "start": 0.0,
                        "end": 1.0,
                        "confidence": 0.95,
                    }
                ],
                "language": "en",
                "duration": 1.0,
            }
        )
        return model

    @pytest.fixture
    def audio_service(self, mock_whisper_model):
        """AudioService 인스턴스"""
        service = AudioServiceImpl()
        # Whisper 모델을 직접 설정 (로드 우회)
        service._whisper_model = mock_whisper_model
        return service

    @pytest.mark.asyncio
    async def test_transcribe_from_path(self, audio_service, tmp_path):
        """경로로부터 음성 전사 테스트"""
        # 임시 오디오 파일 생성
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        request = AudioRequest(
            audio=str(audio_file),
            language="en",
            task="transcribe",
        )

        response = await audio_service.transcribe(request)

        assert response is not None
        assert isinstance(response, AudioResponse)
        assert response.transcription_result is not None
        assert response.transcription_result.text == "Hello world"
        assert response.transcription_result.language == "en"

    @pytest.mark.asyncio
    async def test_transcribe_from_bytes(self, audio_service):
        """바이트로부터 음성 전사 테스트"""
        request = AudioRequest(
            audio=b"fake audio data",
            language="en",
            task="transcribe",
        )

        response = await audio_service.transcribe(request)

        assert response is not None
        assert response.transcription_result is not None
        assert response.transcription_result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_from_audio_segment(self, audio_service):
        """AudioSegment로부터 음성 전사 테스트"""
        audio_segment = AudioSegment(
            audio_data=b"fake audio data",
            format="wav",
            sample_rate=16000,
        )

        request = AudioRequest(
            audio=audio_segment,
            language="en",
            task="transcribe",
        )

        response = await audio_service.transcribe(request)

        assert response is not None
        assert response.transcription_result is not None
        assert response.transcription_result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_transcribe_translate_task(self, audio_service, tmp_path):
        """번역 작업 테스트"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        request = AudioRequest(
            audio=str(audio_file),
            language="ko",
            task="translate",
        )

        response = await audio_service.transcribe(request)

        assert response is not None
        assert response.transcription_result is not None

    @pytest.mark.asyncio
    async def test_transcribe_extra_params(self, audio_service, tmp_path):
        """추가 파라미터 포함 전사 테스트"""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        request = AudioRequest(
            audio=str(audio_file),
            language="en",
            task="transcribe",
            extra_params={"temperature": 0.0},
        )

        response = await audio_service.transcribe(request)

        assert response is not None
        # Whisper 모델이 extra_params로 호출되었는지 확인
        call_kwargs = audio_service._whisper_model.transcribe.call_args[1]
        assert call_kwargs.get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_synthesize_openai(self, audio_service):
        """OpenAI TTS 합성 테스트"""
        # Mock _synthesize_openai 메서드
        mock_audio_segment = AudioSegment(
            audio_data=b"fake audio",
            format="mp3",
            sample_rate=24000,
        )

        # _synthesize_openai 메서드를 직접 Mock
        async def mock_synthesize_openai(*args, **kwargs):
            return mock_audio_segment

        audio_service._synthesize_openai = mock_synthesize_openai

        # TTS provider 설정
        audio_service._tts_provider = TTSProvider.OPENAI

        request = AudioRequest(
            text="Hello world",
            provider="openai",
            voice="alloy",
            speed=1.0,
        )

        response = await audio_service.synthesize(request)

        assert response is not None
        assert isinstance(response, AudioResponse)
        assert response.audio_segment is not None
        assert response.audio_segment.format == "mp3"

    @pytest.mark.asyncio
    async def test_synthesize_elevenlabs(self, audio_service):
        """ElevenLabs TTS 합성 테스트"""
        # Mock _synthesize_elevenlabs 메서드
        mock_audio_segment = AudioSegment(
            audio_data=b"fake audio",
            format="mp3",
            sample_rate=24000,
        )

        # _synthesize_elevenlabs 메서드를 직접 Mock
        async def mock_synthesize_elevenlabs(*args, **kwargs):
            return mock_audio_segment

        audio_service._synthesize_elevenlabs = mock_synthesize_elevenlabs

        # TTS provider 설정
        audio_service._tts_provider = TTSProvider.ELEVENLABS

        request = AudioRequest(
            text="Hello world",
            provider="elevenlabs",
            voice="21m00Tcm4TlvDq8ikWAM",
            api_key="test_key",
        )

        response = await audio_service.synthesize(request)

        assert response is not None
        assert response.audio_segment is not None

    @pytest.mark.asyncio
    async def test_add_audio(self, audio_service):
        """AudioRAG 오디오 추가 테스트"""
        # Mock vector_store와 embedding_model
        mock_vector_store = Mock()
        mock_vector_store.add_documents = AsyncMock(return_value=["doc_id_1"])

        mock_embedding = Mock()
        mock_embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])

        audio_service._vector_store = mock_vector_store
        audio_service._embedding_model = mock_embedding

        request = AudioRequest(
            audio="test_audio.wav",
            audio_id="audio_1",
            metadata={"title": "Test Audio"},
        )

        response = await audio_service.add_audio(request)

        assert response is not None
        # 전사 결과가 저장되었는지 확인
        assert "audio_1" in audio_service._transcriptions

    @pytest.mark.asyncio
    async def test_search(self, audio_service):
        """AudioRAG 검색 테스트"""
        # Mock vector_store와 embedding_model
        # vector_store.search는 동기 함수이고 SearchResult 객체를 반환
        try:
            from llmkit.vector_stores.search import SearchResult
        except ImportError:
            # SearchResult가 없으면 Mock 사용
            SearchResult = Mock

        mock_result1 = Mock()
        mock_result1.metadata = {"audio_id": "audio_1", "segment_id": 0}
        mock_result1.score = 0.9
        mock_result1.content = "Test content 1"

        mock_result2 = Mock()
        mock_result2.metadata = {"audio_id": "audio_2", "segment_id": 0}
        mock_result2.score = 0.8
        mock_result2.content = "Test content 2"

        mock_vector_store = Mock()
        mock_vector_store.search = Mock(return_value=[mock_result1, mock_result2])

        mock_embedding = Mock()
        mock_embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])

        # 전사 결과도 필요 (search_audio에서 사용)
        transcription1 = TranscriptionResult(
            text="Test content 1",
            segments=[TranscriptionSegment(text="Test content 1", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
            model="base",
        )
        transcription2 = TranscriptionResult(
            text="Test content 2",
            segments=[TranscriptionSegment(text="Test content 2", start=0.0, end=1.0)],
            language="en",
            duration=1.0,
            model="base",
        )
        audio_service._transcriptions["audio_1"] = transcription1
        audio_service._transcriptions["audio_2"] = transcription2

        audio_service._vector_store = mock_vector_store
        audio_service._embedding_model = mock_embedding

        request = AudioRequest(
            query="What is this about?",
            top_k=5,
        )

        # 메서드 이름이 search_audio
        response = await audio_service.search_audio(request)

        assert response is not None
        assert response.search_results is not None
        assert len(response.search_results) == 2

    @pytest.mark.asyncio
    async def test_get_transcription(self, audio_service):
        """전사 결과 조회 테스트"""
        # 전사 결과 저장
        transcription = TranscriptionResult(
            text="Hello world",
            segments=[],
            language="en",
            duration=1.0,
            model="base",
        )
        audio_service._transcriptions["audio_1"] = transcription

        request = AudioRequest(audio_id="audio_1")

        response = await audio_service.get_transcription(request)

        assert response is not None
        assert response.transcription is not None
        assert response.transcription.text == "Hello world"

    @pytest.mark.asyncio
    async def test_get_transcription_not_found(self, audio_service):
        """전사 결과 없음 테스트"""
        request = AudioRequest(audio_id="nonexistent")

        response = await audio_service.get_transcription(request)

        # transcription이 None인 경우 그냥 반환
        assert response is not None
        assert response.transcription is None

    @pytest.mark.asyncio
    async def test_list_audios(self, audio_service):
        """오디오 목록 조회 테스트"""
        # 전사 결과 저장
        transcription1 = TranscriptionResult(
            text="Audio 1",
            segments=[],
            language="en",
            duration=1.0,
            model="base",
        )
        transcription2 = TranscriptionResult(
            text="Audio 2",
            segments=[],
            language="en",
            duration=2.0,
            model="base",
        )
        audio_service._transcriptions["audio_1"] = transcription1
        audio_service._transcriptions["audio_2"] = transcription2

        request = AudioRequest()

        response = await audio_service.list_audios(request)

        assert response is not None
        assert response.audio_ids is not None
        assert len(response.audio_ids) == 2
        assert "audio_1" in response.audio_ids
        assert "audio_2" in response.audio_ids


