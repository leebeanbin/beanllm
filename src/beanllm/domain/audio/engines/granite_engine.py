"""
Granite Speech Engine

IBM Granite Speech 8B - 고성능 다국어 STT 엔진 (2024-2025).
Open ASR Leaderboard 2위, Apache 2.0 라이선스.

Granite Speech 8B 특징:
- Open ASR Leaderboard 2위 (WER 5.85%)
- 8B 파라미터
- 5개 언어: 영어, 프랑스어, 독일어, 스페인어, 포르투갈어
- STT + 번역 기능 (영어↔일본어, 영어↔중국어)
- Two-pass 설계: 음성 전사와 텍스트 처리 분리
- Apache 2.0 라이선스 (상업적 사용 가능)
- IBM Granite 3.3 릴리스 (2024년 10월)

사용 사례:
- 엔터프라이즈급 음성 인식
- 다국어 회의 전사
- 고정확도가 필요한 프로덕션 환경
- 상업적 애플리케이션

Requirements:
    pip install transformers torch torchaudio
"""

import logging
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..models import STTConfig
from .base import BaseSTTEngine

logger = logging.getLogger(__name__)

# transformers 설치 여부 체크
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    HAS_GRANITE = True
except ImportError:
    HAS_GRANITE = False


class GraniteEngine(BaseSTTEngine):
    """
    Granite Speech STT 엔진

    IBM의 엔터프라이즈급 STT 모델 (2024-2025 최신).

    Features:
    - Open ASR 2위 (WER 5.85%)
    - 5개 언어 지원
    - Two-pass 아키텍처
    - 번역 기능
    - Apache 2.0 라이선스
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # Granite 엔진 사용
        stt = beanSTT(engine="granite", language="en")
        result = stt.transcribe("audio.mp3")

        # 번역 모드 (영어 → 프랑스어)
        stt = beanSTT(engine="granite-8b", language="en", task="translate")
        result = stt.transcribe("english_audio.mp3")
        ```
    """

    def __init__(self, use_gpu: bool = True):
        """
        Granite 엔진 초기화

        Args:
            use_gpu: GPU 사용 여부
        """
        super().__init__()

        if not HAS_GRANITE:
            raise ImportError(
                "transformers and torch are required for Granite engine. "
                "Install them with: pip install transformers torch torchaudio"
            )

        self.use_gpu = use_gpu
        self._pipeline = None

    def _init_model(self):
        """모델 초기화 (lazy loading)"""
        if self._pipeline is not None:
            return

        model_name = "ibm-granite/granite-speech-3.3-8b"

        logger.info(f"Loading Granite Speech model: {model_name}")

        # Device 설정
        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # 모델 로드
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

        # Processor 로드
        processor = AutoProcessor.from_pretrained(model_name)

        # Pipeline 생성
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        logger.info("Granite Speech 8B model loaded successfully")

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        Granite Speech로 텍스트 전사 및 번역

        Args:
            audio_path: 오디오 파일 경로 또는 numpy array
            config: STT 설정

        Returns:
            Dict: 전사 결과
        """
        # 모델 초기화
        self._init_model()

        start_time = time.time()

        # 파일 경로 처리
        if isinstance(audio_path, (str, Path)):
            audio_path = str(audio_path)

        # 언어 코드 매핑 (Granite는 5개 언어 지원)
        supported_languages = {
            "en": "english",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "pt": "portuguese",
        }

        language = config.language if config.language in supported_languages else "en"

        # Pipeline 옵션 설정
        generate_kwargs = {
            "task": config.task if config.task in ["transcribe", "translate"] else "transcribe",
            "language": supported_languages.get(language, "english"),
        }

        # 전사 실행 (timestamp 지원)
        result = self._pipeline(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

        processing_time = time.time() - start_time

        # 결과 변환
        text = result.get("text", "")
        chunks = result.get("chunks", [])

        # 세그먼트 생성
        segments = []
        if chunks:
            for chunk in chunks:
                segments.append(
                    {
                        "text": chunk.get("text", ""),
                        "start": chunk.get("timestamp", [0.0, 0.0])[0],
                        "end": chunk.get("timestamp", [0.0, 0.0])[1],
                        "confidence": 0.94,  # WER 5.85% → ~94% accuracy
                    }
                )
        else:
            segments.append(
                {
                    "text": text,
                    "start": 0.0,
                    "end": 0.0,
                    "confidence": 0.94,
                }
            )

        return {
            "text": text.strip(),
            "segments": segments,
            "language": language,
            "duration": 0.0,
            "metadata": {
                "model": "granite-speech-3.3-8b",
                "parameters": "8B",
                "leaderboard_rank": 2,
                "wer": 5.85,
                "supported_languages": list(supported_languages.keys()),
                "architecture": "two-pass (speech + text processing)",
                "license": "Apache 2.0",
                "processing_time": processing_time,
                "task": config.task,
            },
        }

    def __repr__(self) -> str:
        return f"GraniteEngine(use_gpu={self.use_gpu})"
