"""
Distil-Whisper Engine

Distil-Whisper 압축 모델을 사용한 STT 엔진.
6x faster, 정확도는 Large V3 대비 1% 이내.

Distil-Whisper 특징:
- 756M 파라미터 (Large V3의 1.54B에서 압축)
- Knowledge distillation으로 생성
- 6x 빠른 추론
- WER은 Large V3 대비 1% 이내
- Out-of-distribution 오디오에서도 우수

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

    HAS_DISTIL_WHISPER = True
except ImportError:
    HAS_DISTIL_WHISPER = False


class DistilWhisperEngine(BaseSTTEngine):
    """
    Distil-Whisper STT 엔진

    압축된 Whisper 모델로 빠른 전사 제공.

    Features:
    - 6x faster than Whisper Large V3
    - 99+ 언어 지원
    - 정확도 유지 (1% 차이)
    - 메모리 효율적
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # Distil-Whisper 엔진 사용
        stt = beanSTT(engine="distil-whisper", language="en")
        result = stt.transcribe("audio.mp3")
        ```
    """

    def __init__(self, use_gpu: bool = True):
        """
        Distil-Whisper 엔진 초기화

        Args:
            use_gpu: GPU 사용 여부
        """
        super().__init__()

        if not HAS_DISTIL_WHISPER:
            raise ImportError(
                "transformers and torch are required for Distil-Whisper engine. "
                "Install them with: pip install transformers torch torchaudio"
            )

        self.use_gpu = use_gpu
        self._pipeline = None

    def _init_model(self, config: STTConfig):
        """모델 초기화 (lazy loading)"""
        if self._pipeline is not None:
            return

        model_name = "distil-whisper/distil-large-v3"
        logger.info(f"Loading Distil-Whisper model: {model_name}")

        # Device 설정
        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # 모델 로드
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
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

        logger.info("Distil-Whisper model loaded successfully")

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        Distil-Whisper로 텍스트 전사

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

        # Pipeline 옵션 설정
        generate_kwargs = {
            "task": config.task,
            "language": None if config.language == "auto" else config.language,
        }

        # Beam search 설정
        if config.beam_size > 1:
            generate_kwargs["num_beams"] = config.beam_size

        # 전사 실행
        if config.timestamp:
            result = self._pipeline(
                audio_path,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
        else:
            result = self._pipeline(
                audio_path,
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )

        # 결과 변환
        return self._convert_result(result, config, time.time() - start_time)

    def _convert_result(
        self, result: Dict, config: STTConfig, processing_time: float
    ) -> Dict:
        """Distil-Whisper 결과를 표준 형식으로 변환"""
        text = result.get("text", "")

        # Segments 추출
        segments = []
        chunks = result.get("chunks", [])

        for chunk in chunks:
            segment_dict = {
                "text": chunk["text"],
                "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                "confidence": 1.0,
            }
            segments.append(segment_dict)

        # 언어 감지
        detected_language = config.language if config.language != "auto" else "en"

        # Duration 계산
        duration = chunks[-1]["timestamp"][1] if chunks and chunks[-1]["timestamp"][1] else 0.0

        return {
            "text": text.strip(),
            "segments": segments,
            "language": detected_language,
            "duration": duration,
            "metadata": {
                "model": "distil-whisper-large-v3",
                "task": config.task,
                "processing_time": processing_time,
                "segment_count": len(segments),
                "speedup": "6x faster than Whisper Large V3",
            },
        }

    def __repr__(self) -> str:
        return f"DistilWhisperEngine(use_gpu={self.use_gpu})"
