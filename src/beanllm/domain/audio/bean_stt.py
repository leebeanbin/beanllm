"""
beanSTT - Main STT Facade

음성 인식(Speech-to-Text) 기능을 제공하는 메인 클래스.
"""

import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .engines.base import BaseSTTEngine
from .models import STTConfig
from .types import TranscriptionResult, TranscriptionSegment


class beanSTT:
    """
    통합 STT 인터페이스

    8개 STT 엔진을 통합하여 사용하기 쉬운 인터페이스 제공.

    Features:
    - 8개 STT 엔진 지원 (Whisper V3 Turbo, Distil-Whisper, Parakeet, Canary, Moonshine, SenseVoice, Granite)
    - 99+ 언어 지원 (엔진별 차이 있음)
    - 실시간 전사
    - 번역 지원
    - 배치 처리
    - 감정 분석 (SenseVoice)
    - 엔터프라이즈급 정확도 (Granite)

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # 기본 사용
        stt = beanSTT(engine="whisper-v3-turbo", language="ko")
        result = stt.transcribe("audio.mp3")
        print(result.text)
        print(f"Language: {result.language}")

        # 실시간 최적화 (Parakeet)
        stt = beanSTT(engine="parakeet", language="en")
        result = stt.transcribe("audio.mp3")

        # 번역 (한국어 → 영어)
        stt = beanSTT(
            engine="whisper-v3-turbo",
            language="ko",
            task="translate"
        )
        result = stt.transcribe("korean_audio.mp3")

        # 배치 처리
        results = stt.batch_transcribe(["audio1.mp3", "audio2.mp3"])
        ```
    """

    def __init__(self, config: Optional[STTConfig] = None, **kwargs):
        """
        Args:
            config: STT 설정 객체 (선택)
            **kwargs: STTConfig 파라미터 (config 대신 사용 가능)

        Example:
            ```python
            # config 객체 사용
            config = STTConfig(engine="whisper-v3-turbo", language="ko")
            stt = beanSTT(config=config)

            # kwargs 사용
            stt = beanSTT(engine="whisper-v3-turbo", language="ko", use_gpu=True)
            ```
        """
        self.config = config or STTConfig(**kwargs)
        self._engine: Optional[BaseSTTEngine] = None
        self._init_engine()

    def _init_engine(self) -> None:
        """엔진 초기화"""
        self._engine = self._create_engine(self.config.engine)

    def _create_engine(self, engine_name: str) -> BaseSTTEngine:
        """
        STT 엔진 생성

        Args:
            engine_name: 엔진 이름

        Returns:
            BaseSTTEngine: STT 엔진 인스턴스

        Raises:
            ImportError: 엔진 의존성이 설치되지 않은 경우
            ValueError: 지원하지 않는 엔진
        """
        if engine_name == "whisper-v3-turbo":
            try:
                from .engines.whisper_engine import WhisperEngine
                return WhisperEngine(use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch torchaudio"
                ) from e

        elif engine_name == "distil-whisper":
            try:
                from .engines.distil_whisper_engine import DistilWhisperEngine
                return DistilWhisperEngine(use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch torchaudio"
                ) from e

        elif engine_name in ["parakeet", "parakeet-1.1b"]:
            try:
                from .engines.parakeet_engine import ParakeetEngine
                return ParakeetEngine(use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"nemo_toolkit is required for engine '{engine_name}'. "
                    f"Install it with: pip install nemo_toolkit[asr]"
                ) from e

        elif engine_name in ["canary", "canary-1b"]:
            try:
                from .engines.canary_engine import CanaryEngine
                return CanaryEngine(model_variant="1b", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"nemo_toolkit is required for engine '{engine_name}'. "
                    f"Install it with: pip install nemo_toolkit[asr]"
                ) from e

        elif engine_name == "canary-flash":
            try:
                from .engines.canary_engine import CanaryEngine
                return CanaryEngine(model_variant="flash", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"nemo_toolkit is required for engine '{engine_name}'. "
                    f"Install it with: pip install nemo_toolkit[asr]"
                ) from e

        elif engine_name in ["moonshine", "moonshine-base"]:
            try:
                from .engines.moonshine_engine import MoonshineEngine
                return MoonshineEngine(model_size="base", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch torchaudio"
                ) from e

        elif engine_name == "moonshine-tiny":
            try:
                from .engines.moonshine_engine import MoonshineEngine
                return MoonshineEngine(model_size="tiny", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch torchaudio"
                ) from e

        elif engine_name in ["sensevoice", "sensevoice-small"]:
            try:
                from .engines.sensevoice_engine import SenseVoiceEngine
                return SenseVoiceEngine(model_size="small", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"funasr is required for engine '{engine_name}'. "
                    f"Install it with: pip install funasr modelscope"
                ) from e

        elif engine_name == "sensevoice-large":
            try:
                from .engines.sensevoice_engine import SenseVoiceEngine
                return SenseVoiceEngine(model_size="large", use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"funasr is required for engine '{engine_name}'. "
                    f"Install it with: pip install funasr modelscope"
                ) from e

        elif engine_name in ["granite", "granite-8b", "granite-speech"]:
            try:
                from .engines.granite_engine import GraniteEngine
                return GraniteEngine(use_gpu=self.config.use_gpu)
            except ImportError as e:
                raise ImportError(
                    f"transformers and torch are required for engine '{engine_name}'. "
                    f"Install them with: pip install transformers torch torchaudio"
                ) from e

        # 지원하지 않는 엔진
        raise NotImplementedError(
            f"Engine '{engine_name}' is not yet implemented. "
            f"Currently supported: whisper-v3-turbo, distil-whisper, parakeet, "
            f"canary, canary-flash, moonshine-tiny, moonshine-base, sensevoice, granite"
        )

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], **kwargs
    ) -> TranscriptionResult:
        """
        오디오 전사 (음성 → 텍스트)

        Args:
            audio_path: 오디오 파일 경로 또는 numpy array
            **kwargs: 추가 옵션 (config 오버라이드)

        Returns:
            TranscriptionResult: 전사 결과

        Raises:
            FileNotFoundError: 오디오 파일을 찾을 수 없음
            ValueError: 잘못된 오디오 형식
            ImportError: STT 엔진 의존성 미설치

        Example:
            ```python
            # 오디오 파일 경로
            result = stt.transcribe("audio.mp3")

            # numpy array
            import librosa
            audio, sr = librosa.load("audio.mp3", sr=16000)
            result = stt.transcribe(audio)
            ```
        """
        start_time = time.time()

        # 엔진 실행
        if self._engine is None:
            raise RuntimeError("STT engine not initialized")

        raw_result = self._engine.transcribe(audio_path, self.config)

        # TranscriptionResult 생성
        segments = []
        for seg_dict in raw_result.get("segments", []):
            segment = TranscriptionSegment(
                text=seg_dict["text"],
                start=seg_dict.get("start", 0.0),
                end=seg_dict.get("end", 0.0),
                confidence=seg_dict.get("confidence", 1.0),
                language=raw_result.get("language"),
            )
            segments.append(segment)

        result = TranscriptionResult(
            text=raw_result["text"],
            segments=segments,
            language=raw_result.get("language", self.config.language),
            duration=raw_result.get("duration", 0.0),
            model=self.config.engine,
            metadata=raw_result.get("metadata", {}),
        )

        # 총 처리 시간 추가
        result.metadata["total_time"] = time.time() - start_time

        return result

    def batch_transcribe(
        self, audio_paths: List[Union[str, Path, np.ndarray]], **kwargs
    ) -> List[TranscriptionResult]:
        """
        배치 전사 처리

        여러 오디오를 순차적으로 처리합니다.

        Args:
            audio_paths: 오디오 리스트 (경로 또는 numpy array)
            **kwargs: 추가 옵션

        Returns:
            List[TranscriptionResult]: 전사 결과 리스트

        Example:
            ```python
            # 오디오 파일 배치 처리
            results = stt.batch_transcribe([
                "audio1.mp3",
                "audio2.mp3",
                "audio3.mp3"
            ])

            for i, result in enumerate(results):
                print(f"Audio {i+1}: {result.text[:50]}...")
            ```
        """
        results = []
        for audio_path in audio_paths:
            result = self.transcribe(audio_path, **kwargs)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return (
            f"beanSTT(engine={self.config.engine}, "
            f"language={self.config.language}, "
            f"task={self.config.task}, "
            f"gpu={self.config.use_gpu})"
        )
