"""
STT (Speech-to-Text) 모델 및 설정

음성 인식을 위한 설정과 데이터 모델.
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class STTConfig:
    """
    STT (Speech-to-Text) 설정

    음성 인식 엔진 선택, 언어 설정, 고급 옵션을 포함합니다.

    Attributes:
        engine: STT 엔진 선택
            - "whisper-v3-turbo": Whisper Large V3 Turbo (6x faster, 99+ 언어)
            - "distil-whisper": Distil-Whisper (6x faster, 압축 모델)
            - "parakeet": NVIDIA Parakeet TDT (실시간, RTFx >2000)
            - "canary": Canary-1B (다국어, 번역 지원)
            - "canary-flash": Canary-1B-Flash (초고속, RTFx >1000)
            - "moonshine": Moonshine (온디바이스, 초경량)

        language: 언어 설정
            - "auto": 자동 감지
            - "ko": 한국어
            - "en": 영어
            - "zh": 중국어
            - "ja": 일본어
            - 기타 99+ languages (Whisper 기준)

        use_gpu: GPU 사용 여부 (기본: True)
        task: 작업 유형
            - "transcribe": 음성 → 텍스트 (동일 언어)
            - "translate": 음성 → 영어 텍스트 (번역)

        timestamp: 타임스탬프 생성 여부 (기본: True)
        word_timestamps: 단어 수준 타임스탬프 (기본: False)
        vad_filter: Voice Activity Detection 필터 (기본: True)

        # 고급 옵션
        beam_size: Beam search 크기 (기본: 5, 높을수록 정확하지만 느림)
        best_of: 후보 개수 (기본: 5)
        temperature: 샘플링 온도 (0.0-1.0, 기본: 0.0=deterministic)
        compression_ratio_threshold: 압축 비율 임계값 (기본: 2.4)
        log_prob_threshold: 로그 확률 임계값 (기본: -1.0)
        no_speech_threshold: 무음 임계값 (기본: 0.6)

    Example:
        ```python
        # 기본 설정
        config = STTConfig(engine="whisper-v3-turbo", language="ko")

        # 고급 설정 (고정밀)
        config = STTConfig(
            engine="whisper-v3-turbo",
            language="ko",
            use_gpu=True,
            timestamp=True,
            word_timestamps=True,
            beam_size=10,
            temperature=0.0
        )

        # 실시간 설정 (고속)
        config = STTConfig(
            engine="parakeet",
            language="en",
            use_gpu=True,
            beam_size=1,
            vad_filter=True
        )

        # 번역 설정
        config = STTConfig(
            engine="canary",
            language="ko",
            task="translate"  # 한국어 → 영어
        )
        ```
    """

    # 엔진 설정
    engine: str = "whisper-v3-turbo"
    language: str = "auto"
    use_gpu: bool = True
    task: Literal["transcribe", "translate"] = "transcribe"

    # 타임스탬프 옵션
    timestamp: bool = True
    word_timestamps: bool = False

    # 전처리 옵션
    vad_filter: bool = True  # Voice Activity Detection

    # Beam search 설정
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0

    # 품질 임계값
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # 고급 옵션
    batch_size: int = 1
    return_language: bool = True  # 감지된 언어 반환

    def __post_init__(self):
        """설정 유효성 검증"""
        # 엔진 유효성 검사
        valid_engines = {
            "whisper-v3-turbo",
            "distil-whisper",
            "parakeet",
            "parakeet-1.1b",
            "canary",
            "canary-1b",
            "canary-flash",
            "moonshine",
            "moonshine-tiny",
            "moonshine-base",
        }
        if self.engine not in valid_engines:
            raise ValueError(
                f"Invalid engine: {self.engine}. "
                f"Must be one of {valid_engines}"
            )

        # 언어 유효성 검사 (일부만 체크)
        if self.language not in ["auto", "ko", "en", "zh", "ja", "es", "fr", "de", "ru", "ar", "hi"]:
            # 경고만 출력 (99+ languages 지원하므로)
            import warnings

            warnings.warn(
                f"Language '{self.language}' may not be supported by all engines. "
                f"Common languages: auto, ko, en, zh, ja, es, fr, de, ru, ar, hi"
            )

        # 온도 범위 검사
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(
                f"temperature must be between 0.0 and 1.0, got {self.temperature}"
            )

        # Beam size 범위 검사
        if self.beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, got {self.beam_size}")

    def __repr__(self) -> str:
        return (
            f"STTConfig(engine={self.engine}, lang={self.language}, "
            f"task={self.task}, gpu={self.use_gpu})"
        )
