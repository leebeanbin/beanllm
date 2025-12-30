"""
NVIDIA Parakeet TDT Engine

NVIDIA의 Parakeet TDT 모델을 사용한 실시간 최적화 STT 엔진.
RTFx >2000, 초고속 추론.

Parakeet TDT 특징:
- 1.1B 파라미터
- RTFx (Real-Time Factor) >2000
- 실시간 애플리케이션에 최적화
- 영어 중심 (다국어 제한적)
- FastConformer + TDT (Token-and-Duration Transducer)

Requirements:
    pip install transformers torch torchaudio nemo_toolkit
"""

import logging
import time
from pathlib import Path
from typing import Dict, Union

import numpy as np

from ..models import STTConfig
from .base import BaseSTTEngine

logger = logging.getLogger(__name__)

# NeMo toolkit 설치 여부 체크
try:
    import nemo.collections.asr as nemo_asr

    HAS_PARAKEET = True
except ImportError:
    HAS_PARAKEET = False


class ParakeetEngine(BaseSTTEngine):
    """
    NVIDIA Parakeet TDT STT 엔진

    실시간 애플리케이션에 최적화된 초고속 STT.

    Features:
    - RTFx >2000 (극도로 빠름)
    - 실시간 전사
    - FastConformer 아키텍처
    - TDT (Token-and-Duration) 방식
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # Parakeet 엔진 사용 (실시간)
        stt = beanSTT(engine="parakeet", language="en")
        result = stt.transcribe("audio.mp3")
        ```
    """

    def __init__(self, use_gpu: bool = True):
        """
        Parakeet 엔진 초기화

        Args:
            use_gpu: GPU 사용 여부
        """
        super().__init__()

        if not HAS_PARAKEET:
            raise ImportError(
                "nemo_toolkit is required for Parakeet engine. "
                "Install it with: pip install nemo_toolkit[asr]"
            )

        self.use_gpu = use_gpu
        self._model = None

    def _init_model(self):
        """모델 초기화 (lazy loading)"""
        if self._model is not None:
            return

        model_name = "nvidia/parakeet-tdt-1.1b"
        logger.info(f"Loading Parakeet model: {model_name}")

        # NeMo 모델 로드
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name)

        if self.use_gpu:
            self._model = self._model.cuda()
        self._model.eval()

        logger.info("Parakeet TDT model loaded successfully")

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        Parakeet으로 텍스트 전사

        Args:
            audio_path: 오디오 파일 경로
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
        else:
            raise ValueError("Parakeet engine requires audio file path")

        # 전사 실행
        transcriptions = self._model.transcribe([audio_path])
        text = transcriptions[0] if transcriptions else ""

        processing_time = time.time() - start_time

        # 결과 변환 (Parakeet은 timestamp 미제공)
        return {
            "text": text.strip(),
            "segments": [
                {
                    "text": text.strip(),
                    "start": 0.0,
                    "end": 0.0,  # Parakeet은 timestamp 미제공
                    "confidence": 1.0,
                }
            ],
            "language": config.language if config.language != "auto" else "en",
            "duration": 0.0,
            "metadata": {
                "model": "nvidia-parakeet-tdt-1.1b",
                "processing_time": processing_time,
                "rtfx": ">2000",
                "optimized_for": "real-time",
            },
        }

    def __repr__(self) -> str:
        return f"ParakeetEngine(use_gpu={self.use_gpu})"
