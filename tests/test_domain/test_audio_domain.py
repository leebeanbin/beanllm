"""Tests for domain/audio: types, models, beanSTT."""

import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from beanllm.domain.audio.bean_stt import beanSTT
from beanllm.domain.audio.models import STTConfig
from beanllm.domain.audio.types import AudioSegment, TranscriptionResult, TranscriptionSegment

# ---------------------------------------------------------------------------
# STTConfig
# ---------------------------------------------------------------------------


class TestSTTConfig:
    def test_defaults(self):
        config = STTConfig()
        assert config.engine == "whisper-v3-turbo"
        assert config.language == "auto"
        assert config.use_gpu is True
        assert config.task == "transcribe"

    def test_custom_values(self):
        config = STTConfig(engine="canary", language="ko", use_gpu=False, task="translate")
        assert config.engine == "canary"
        assert config.language == "ko"
        assert config.use_gpu is False
        assert config.task == "translate"

    def test_all_fields_settable(self):
        config = STTConfig(
            engine="moonshine",
            language="en",
            beam_size=10,
            best_of=3,
            temperature=0.1,
            vad_filter=False,
            timestamp=False,
            word_timestamps=True,
        )
        assert config.engine == "moonshine"
        assert config.beam_size == 10
        assert config.temperature == 0.1


# ---------------------------------------------------------------------------
# AudioSegment
# ---------------------------------------------------------------------------


def _make_wav_file(tmp_path: Path) -> Path:
    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 0.1 seconds of silence
        frames = struct.pack("<" + "h" * 1600, *([0] * 1600))
        wf.writeframes(frames)
    return wav_path


class TestAudioSegment:
    def test_basic_creation(self):
        seg = AudioSegment(audio_data=b"data", sample_rate=16000, duration=1.0)
        assert seg.audio_data == b"data"
        assert seg.sample_rate == 16000
        assert seg.duration == 1.0

    def test_defaults(self):
        seg = AudioSegment(audio_data=b"x")
        assert seg.sample_rate == 16000
        assert seg.channels == 1
        assert seg.format == "wav"

    def test_from_file_wav(self, tmp_path):
        wav_path = _make_wav_file(tmp_path)
        seg = AudioSegment.from_file(wav_path)
        assert seg.sample_rate == 16000
        assert seg.channels == 1
        assert seg.format == "wav"
        assert seg.duration == pytest.approx(0.1, abs=0.01)

    def test_from_file_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AudioSegment.from_file(tmp_path / "missing.wav")

    def test_from_file_non_wav(self, tmp_path):
        mp3_path = tmp_path / "test.mp3"
        mp3_path.write_bytes(b"fake mp3 data")
        seg = AudioSegment.from_file(mp3_path)
        assert seg.audio_data == b"fake mp3 data"
        assert seg.format == "mp3"

    def test_frozen(self):
        seg = AudioSegment(audio_data=b"x")
        with pytest.raises((AttributeError, TypeError)):
            seg.sample_rate = 8000  # type: ignore


# ---------------------------------------------------------------------------
# TranscriptionSegment
# ---------------------------------------------------------------------------


class TestTranscriptionSegment:
    def test_basic(self):
        seg = TranscriptionSegment(text="hello", start=0.0, end=1.5)
        assert seg.text == "hello"
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.confidence == 1.0

    def test_str(self):
        seg = TranscriptionSegment(text="world", start=1.0, end=2.5)
        s = str(seg)
        assert "world" in s
        assert "1.00" in s
        assert "2.50" in s

    def test_with_speaker(self):
        seg = TranscriptionSegment(text="hi", speaker="speaker_A")
        assert seg.speaker == "speaker_A"


# ---------------------------------------------------------------------------
# TranscriptionResult
# ---------------------------------------------------------------------------


class TestTranscriptionResult:
    def test_basic(self):
        result = TranscriptionResult(text="hello world", language="en", model="whisper")
        assert result.text == "hello world"
        assert str(result) == "hello world"

    def test_with_segments(self):
        segs = [
            TranscriptionSegment(text="hello", start=0.0, end=0.5),
            TranscriptionSegment(text="world", start=0.5, end=1.0),
        ]
        result = TranscriptionResult(text="hello world", segments=segs)
        assert len(result.segments) == 2

    def test_metadata_default_empty(self):
        result = TranscriptionResult(text="x")
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# beanSTT
# ---------------------------------------------------------------------------


def _make_stt_no_engine(**kwargs) -> beanSTT:
    """Create beanSTT without triggering engine initialisation."""
    with patch.object(beanSTT, "_init_engine", return_value=None):
        stt = beanSTT(**kwargs)
    stt._engine = None
    return stt


class TestBeanSTTInit:
    def test_default_config(self):
        stt = _make_stt_no_engine()
        assert stt.config.engine == "whisper-v3-turbo"
        assert stt._engine is None

    def test_custom_config(self):
        config = STTConfig(engine="canary", language="ko")
        stt = _make_stt_no_engine(config=config)
        assert stt.config.engine == "canary"
        assert stt.config.language == "ko"

    def test_kwargs_override(self):
        stt = _make_stt_no_engine(engine="moonshine", language="en")
        assert stt.config.engine == "moonshine"
        assert stt.config.language == "en"

    def test_repr(self):
        stt = _make_stt_no_engine(engine="whisper-v3-turbo", language="ko")
        r = repr(stt)
        assert "whisper-v3-turbo" in r
        assert "ko" in r


class TestBeanSTTCreateEngine:
    def _make_stt(self, engine="whisper-v3-turbo"):
        return _make_stt_no_engine(engine=engine)

    def test_unknown_engine_raises(self):
        stt = self._make_stt()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            stt._create_engine("unknown-engine-xyz")

    def test_whisper_import_error(self):
        stt = self._make_stt()
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(ImportError, match="transformers"):
                stt._create_engine("whisper-v3-turbo")

    def test_parakeet_import_error(self):
        stt = self._make_stt()
        with patch.dict("sys.modules", {"nemo": None, "nemo_toolkit": None}):
            with pytest.raises(ImportError, match="nemo_toolkit"):
                stt._create_engine("parakeet")

    def test_sensevoice_import_error(self):
        stt = self._make_stt()
        with patch.dict("sys.modules", {"funasr": None, "modelscope": None}):
            with pytest.raises(ImportError, match="funasr"):
                stt._create_engine("sensevoice")

    def test_granite_import_error(self):
        stt = self._make_stt()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError):
                stt._create_engine("granite")

    def test_create_engine_with_mock(self):
        stt = self._make_stt()
        mock_engine_cls = MagicMock()
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_whisper_mod = MagicMock()
        mock_whisper_mod.WhisperEngine = mock_engine_cls

        with patch.dict(
            "sys.modules", {"beanllm.domain.audio.engines.whisper_engine": mock_whisper_mod}
        ):
            engine = stt._create_engine("whisper-v3-turbo")
        assert engine is mock_engine


class TestBeanSTTTranscribe:
    def _make_stt_with_engine(self):
        stt = _make_stt_no_engine(engine="whisper-v3-turbo", language="ko")
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = {
            "text": "안녕하세요",
            "language": "ko",
            "duration": 2.5,
            "segments": [{"text": "안녕하세요", "start": 0.0, "end": 2.5, "confidence": 0.95}],
            "metadata": {},
        }
        stt._engine = mock_engine
        return stt, mock_engine

    def test_transcribe_returns_result(self):
        stt, mock_engine = self._make_stt_with_engine()
        result = stt.transcribe("audio.wav")
        assert result.text == "안녕하세요"
        assert result.language == "ko"
        assert result.duration == 2.5
        assert len(result.segments) == 1

    def test_transcribe_segment_fields(self):
        stt, _ = self._make_stt_with_engine()
        result = stt.transcribe("audio.wav")
        seg = result.segments[0]
        assert seg.text == "안녕하세요"
        assert seg.start == 0.0
        assert seg.end == 2.5
        assert seg.confidence == 0.95

    def test_transcribe_no_engine_raises(self):
        stt = _make_stt_no_engine(engine="whisper-v3-turbo")
        stt._engine = None
        with pytest.raises(RuntimeError, match="not initialized"):
            stt.transcribe("audio.wav")

    def test_transcribe_total_time_in_metadata(self):
        stt, _ = self._make_stt_with_engine()
        result = stt.transcribe("audio.wav")
        assert "total_time" in result.metadata

    def test_transcribe_no_segments(self):
        stt = _make_stt_no_engine(engine="whisper-v3-turbo")
        mock_engine = MagicMock()
        mock_engine.transcribe.return_value = {
            "text": "hello",
            "segments": [],
            "metadata": {},
        }
        stt._engine = mock_engine
        result = stt.transcribe("audio.wav")
        assert result.text == "hello"
        assert result.segments == []

    def test_batch_transcribe(self):
        stt, mock_engine = self._make_stt_with_engine()
        results = stt.batch_transcribe(["audio1.wav", "audio2.wav"])
        assert len(results) == 2
        assert mock_engine.transcribe.call_count == 2
