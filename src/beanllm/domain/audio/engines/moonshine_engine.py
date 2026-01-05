"""
Moonshine Engine

Moonshine 초경량 STT 엔진 (온디바이스용).
Whisper Tiny/Small보다 작지만 동급 성능.

Moonshine 특징:
- <100M 파라미터 (초경량)
- 온디바이스 최적화
- Whisper Tiny/Small 능가
- 프라이버시 중시
- 오프라인 동작

사용 사례:
- 온디바이스 음성 비서
- 오프라인 산업 장비
- 프라이버시 민감 애플리케이션
- 대역폭 제한 환경

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

    HAS_MOONSHINE = True
except ImportError:
    HAS_MOONSHINE = False


class MoonshineEngine(BaseSTTEngine):
    """
    Moonshine STT 엔진

    초경량 온디바이스 STT 모델.

    Features:
    - <100M 파라미터 (경량)
    - Whisper Tiny/Small 능가
    - 온디바이스 최적화
    - 프라이버시 보호
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # Moonshine 엔진 사용 (온디바이스)
        stt = beanSTT(engine="moonshine", language="en")
        result = stt.transcribe("audio.mp3")
        ```
    """

    def __init__(self, model_size: str = "base", use_gpu: bool = False):
        """
        Moonshine 엔진 초기화

        Args:
            model_size: 모델 크기 (tiny / base)
            use_gpu: GPU 사용 여부 (온디바이스는 보통 CPU)
        """
        super().__init__()

        if not HAS_MOONSHINE:
            raise ImportError(
                "transformers and torch are required for Moonshine engine. "
                "Install them with: pip install transformers torch torchaudio"
            )

        self.model_size = model_size
        self.use_gpu = use_gpu
        self._pipeline = None

    def _init_model(self, config: STTConfig):
        """모델 초기화 (lazy loading)"""
        if self._pipeline is not None:
            return

        # Moonshine 모델 선택
        if self.model_size == "tiny":
            model_name = "UsefulSensors/moonshine-tiny"
        else:
            model_name = "UsefulSensors/moonshine-base"

        logger.info(f"Loading Moonshine model: {model_name}")

        # Device 설정 (온디바이스는 보통 CPU)
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

        logger.info(f"Moonshine {self.model_size} model loaded successfully")

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        Moonshine로 텍스트 전사

        Args:
            audio_path: 오디오 파일 경로 또는 numpy array
            config: STT 설정

        Returns:
            Dict: 전사 결과
        """
        # 모델 초기화
        self._init_model(config)

        start_time = time.time()

        # 파일 경로 처리
        if isinstance(audio_path, (str, Path)):
            audio_path = str(audio_path)

        # Pipeline 옵션 설정 (간단하게)
        generate_kwargs = {
            "task": "transcribe",
            "language": config.language if config.language != "auto" else "en",
        }

        # 전사 실행 (timestamp 미지원)
        result = self._pipeline(
            audio_path,
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )

        processing_time = time.time() - start_time

        # 결과 변환
        text = result.get("text", "")

        return {
            "text": text.strip(),
            "segments": [
                {
                    "text": text.strip(),
                    "start": 0.0,
                    "end": 0.0,
                    "confidence": 1.0,
                }
            ],
            "language": config.language if config.language != "auto" else "en",
            "duration": 0.0,
            "metadata": {
                "model": f"moonshine-{self.model_size}",
                "parameters": "<100M",
                "optimized_for": "on-device",
                "processing_time": processing_time,
            },
        }

    def __repr__(self) -> str:
        return f"MoonshineEngine(size={self.model_size}, use_gpu={self.use_gpu})"
