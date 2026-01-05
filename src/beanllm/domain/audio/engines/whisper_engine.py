"""
Whisper V3 Turbo Engine

OpenAI의 Whisper Large V3 Turbo 모델을 사용한 STT 엔진.
6x faster than Large V3, 99+ 언어 지원.

Whisper V3 Turbo 특징:
- 809M 파라미터 (Large V3의 1.55B에서 감소)
- Decoder 레이어 32 → 4로 감소
- 6x 빠른 추론 속도
- 정확도는 Large V3 대비 1-2% 이내
- 99+ 언어 지원

Requirements:
    pip install transformers torch torchaudio
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..models import STTConfig
from ..types import TranscriptionSegment
from .base import BaseSTTEngine

logger = logging.getLogger(__name__)

# transformers 설치 여부 체크
try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class WhisperEngine(BaseSTTEngine):
    """
    Whisper V3 Turbo STT 엔진

    OpenAI의 Whisper Large V3 Turbo 모델을 사용한 고성능 STT.

    Features:
    - 99+ 언어 지원
    - 6x faster than V3
    - Timestamp 지원
    - 번역 지원 (translate task)
    - Lazy loading

    Example:
        ```python
        from beanllm.domain.audio import beanSTT

        # Whisper V3 Turbo 엔진 사용
        stt = beanSTT(engine="whisper-v3-turbo", language="ko")
        result = stt.transcribe("audio.mp3")
        print(result.text)

        # 번역 (한국어 → 영어)
        stt = beanSTT(engine="whisper-v3-turbo", language="ko", task="translate")
        result = stt.transcribe("korean_audio.mp3")
        ```
    """

    def __init__(self, use_gpu: bool = True):
        """
        Whisper V3 Turbo 엔진 초기화

        Args:
            use_gpu: GPU 사용 여부
        """
        super().__init__()

        if not HAS_WHISPER:
            raise ImportError(
                "transformers and torch are required for Whisper engine. "
                "Install them with: pip install transformers torch torchaudio"
            )

        self.use_gpu = use_gpu
        self._pipeline = None

    def _init_model(self, config: STTConfig):
        """모델 초기화 (lazy loading)"""
        if self._pipeline is not None:
            return

        model_name = "openai/whisper-large-v3-turbo"
        logger.info(f"Loading Whisper model: {model_name}")

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

        logger.info("Whisper V3 Turbo model loaded successfully")

    def transcribe(
        self, audio_path: Union[str, Path, np.ndarray], config: STTConfig
    ) -> Dict:
        """
        Whisper로 텍스트 전사

        Args:
            audio_path: 오디오 파일 경로 또는 numpy array
            config: STT 설정

        Returns:
            Dict: 전사 결과
                {
                    "text": str,
                    "segments": List[Dict],
                    "language": str,
                    "duration": float,
                    "metadata": dict
                }
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

        # 온도 설정
        if config.temperature > 0:
            generate_kwargs["temperature"] = config.temperature
            generate_kwargs["do_sample"] = True

        # 전사 실행
        if config.timestamp:
            # Timestamp 포함
            result = self._pipeline(
                audio_path,
                generate_kwargs=generate_kwargs,
                return_timestamps=True,
            )
        else:
            # Timestamp 없음
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
        """
        Whisper 결과를 표준 형식으로 변환
        """
        # 텍스트 추출
        text = result.get("text", "")

        # Segments 추출
        segments = []
        chunks = result.get("chunks", [])

        for chunk in chunks:
            segment_dict = {
                "text": chunk["text"],
                "start": chunk["timestamp"][0] if chunk["timestamp"][0] is not None else 0.0,
                "end": chunk["timestamp"][1] if chunk["timestamp"][1] is not None else 0.0,
                "confidence": 1.0,  # Whisper는 confidence를 제공하지 않음
            }
            segments.append(segment_dict)

        # 언어 감지 (Whisper pipeline은 언어를 자동 감지)
        detected_language = config.language if config.language != "auto" else "en"

        # Duration 계산
        duration = chunks[-1]["timestamp"][1] if chunks and chunks[-1]["timestamp"][1] else 0.0

        return {
            "text": text.strip(),
            "segments": segments,
            "language": detected_language,
            "duration": duration,
            "metadata": {
                "model": "whisper-large-v3-turbo",
                "task": config.task,
                "processing_time": processing_time,
                "segment_count": len(segments),
            },
        }

    def __repr__(self) -> str:
        return f"WhisperEngine(use_gpu={self.use_gpu})"
