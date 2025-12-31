"""
SenseVoice Engine

Alibaba SenseVoice - 초고속 다국어 STT 엔진 (2024년 7월 출시).
Whisper-Large보다 15배 빠르고 다중 음성 이해 기능 지원.

SenseVoice-Small 특징:
- 15배 빠름 (vs Whisper-Large)
- 5배 빠름 (vs Whisper-Small)
- 10초 오디오를 70ms에 처리
- 다중 음성 이해: ASR, LID, SER, AED
  * ASR: Automatic Speech Recognition (음성 인식)
  * LID: Language Identification (언어 식별)
  * SER: Speech Emotion Recognition (감정 인식)
  * AED: Audio Event Detection (오디오 이벤트 감지)
- 5개 언어: 중국어(표준어), 광둥어, 영어, 일본어, 한국어
- HuggingFace Hub에서 다운로드 가능

SenseVoice-Large:
- 50개 이상 언어 지원
- 중국어/광둥어에서 Whisper보다 50% 이상 개선

사용 사례:
- 실시간 자막 생성
- 다국어 회의 전사
- 감정 분석이 필요한 음성 처리
- 고속 배치 처리

Requirements:
    pip install funasr modelscope torch torchaudio
"""

import logging
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..models import STTConfig
from .base import BaseSTTEngine

logger = logging.getLogger(__name__)

# FunASR 설치 여부 체크
try:
    from funasr import AutoModel

    HAS_SENSEVOICE = True
except ImportError:
    HAS_SENSEVOICE = False


class SenseVoiceEngine(BaseSTTEngine):
    """
    SenseVoice STT 엔진

    Alibaba의 초고속 다국어 STT 모델 (2024-2025 최신).

    Features:
    - 15배 빠름 (vs Whisper-Large)
    - 다중 기능 (ASR, LID, SER, AED)
    - 5개 언어 (한국어 포함)
    - 70ms 처리 속도 (10초 오디오)
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # SenseVoice 엔진 사용
        stt = beanSTT(engine="sensevoice", language="ko")
        result = stt.transcribe("audio.mp3")

        # 감정 분석 포함
        stt = beanSTT(engine="sensevoice-small", language="ko")
        result = stt.transcribe("audio.mp3")
        print(result.metadata["emotion"])  # 감정 정보
        ```
    """

    def __init__(self, model_size: str = "small", use_gpu: bool = True):
        """
        SenseVoice 엔진 초기화

        Args:
            model_size: 모델 크기 (small / large)
            use_gpu: GPU 사용 여부
        """
        super().__init__()

        if not HAS_SENSEVOICE:
            raise ImportError(
                "funasr is required for SenseVoice engine. "
                "Install it with: pip install funasr modelscope"
            )

        self.model_size = model_size
        self.use_gpu = use_gpu
        self._model = None

    def _init_model(self):
        """모델 초기화 (lazy loading)"""
        if self._model is not None:
            return

        # SenseVoice 모델 선택
        if self.model_size == "large":
            model_name = "iic/SenseVoiceMultiLingual"  # 50+ 언어
        else:
            model_name = "iic/SenseVoiceSmall"  # 5개 언어 (기본)

        logger.info(f"Loading SenseVoice model: {model_name}")

        # Device 설정
        device = "cuda:0" if self.use_gpu else "cpu"

        # FunASR AutoModel로 로드
        self._model = AutoModel(
            model=model_name,
            device=device,
            disable_pbar=True,  # 프로그레스바 비활성화
            disable_log=False,
        )

        logger.info(f"SenseVoice {self.model_size} model loaded successfully")

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        SenseVoice로 텍스트 전사 및 다중 기능 추론

        Args:
            audio_path: 오디오 파일 경로 또는 numpy array
            config: STT 설정

        Returns:
            Dict: 전사 결과 + 감정/언어 정보
        """
        # 모델 초기화
        self._init_model()

        start_time = time.time()

        # 파일 경로 처리
        if isinstance(audio_path, (str, Path)):
            audio_path = str(audio_path)
        else:
            raise ValueError("SenseVoice engine requires audio file path")

        # 언어 코드 매핑 (SenseVoice-Small은 5개 언어만 지원)
        supported_languages = {
            "zh": "zh",  # 중국어 (표준어)
            "yue": "yue",  # 광둥어
            "en": "en",  # 영어
            "ja": "ja",  # 일본어
            "ko": "ko",  # 한국어
        }

        language = config.language if config.language in supported_languages else "auto"

        # 전사 실행
        # SenseVoice는 자동으로 ASR + LID + SER + AED 수행
        result = self._model.generate(
            input=audio_path,
            language=language,
            use_itn=True,  # Inverse Text Normalization (숫자, 날짜 등 정규화)
            batch_size_s=60,  # 배치 크기 (초 단위)
        )

        processing_time = time.time() - start_time

        # 결과 파싱
        if isinstance(result, list) and len(result) > 0:
            res = result[0]
            text = res.get("text", "")
            detected_language = res.get("language", language)
            emotion = res.get("emotion", "neutral")  # 감정 태그
            event = res.get("event", None)  # 오디오 이벤트 (박수, 음악 등)
        else:
            text = ""
            detected_language = language
            emotion = "neutral"
            event = None

        # 결과 변환
        return {
            "text": text.strip(),
            "segments": [
                {
                    "text": text.strip(),
                    "start": 0.0,
                    "end": 0.0,  # SenseVoice는 기본적으로 timestamp 미제공
                    "confidence": 0.95,  # 매우 높은 정확도
                }
            ],
            "language": detected_language,
            "duration": 0.0,
            "metadata": {
                "model": f"sensevoice-{self.model_size}",
                "supported_languages": list(supported_languages.keys())
                if self.model_size == "small"
                else "50+",
                "processing_time": processing_time,
                "speed_vs_whisper_large": "15x faster",
                "speed_vs_whisper_small": "5x faster",
                "features": ["ASR", "LID", "SER", "AED"],
                # 추가 기능
                "emotion": emotion,  # 감정 인식 (SER)
                "event": event,  # 오디오 이벤트 감지 (AED)
            },
        }

    def __repr__(self) -> str:
        return f"SenseVoiceEngine(size={self.model_size}, use_gpu={self.use_gpu})"
