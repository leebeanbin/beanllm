"""
OCR Pipeline - Image loading and recognition processing logic.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image

from beanllm.domain.ocr.engines.base import BaseOCREngine
from beanllm.domain.ocr.models import OCRConfig, OCRResult


def load_image(
    image_or_path: Union[str, Path, np.ndarray, Image.Image],
) -> np.ndarray:
    """
    이미지 로드 및 numpy array로 변환

    Args:
        image_or_path: 이미지 경로, numpy array, 또는 PIL Image

    Returns:
        np.ndarray: 이미지 (numpy array)

    Raises:
        ValueError: 지원하지 않는 이미지 형식
        FileNotFoundError: 이미지 파일을 찾을 수 없음
    """
    if isinstance(image_or_path, np.ndarray):
        return image_or_path

    if isinstance(image_or_path, Image.Image):
        if image_or_path.mode != "RGB":
            image_or_path = image_or_path.convert("RGB")
        return np.array(image_or_path)

    path = Path(image_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img: Image.Image = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def run_recognize_pipeline(
    image: np.ndarray,
    engine: BaseOCREngine,
    config: OCRConfig,
    start_time: float,
    preprocessor: Optional[Any] = None,
    postprocessor: Optional[Any] = None,
    event_logger: Optional[Any] = None,
    rate_limiter: Optional[Any] = None,
) -> OCRResult:
    """
    Run the OCR pipeline: preprocess -> rate limit -> recognize -> build result -> postprocess.

    Args:
        image: Image as numpy array
        engine: OCR engine instance
        config: OCR config
        start_time: Start time for processing_time
        preprocessor: Optional image preprocessor
        postprocessor: Optional LLM postprocessor
        event_logger: Optional event logger protocol
        rate_limiter: Optional rate limiter (for cloud engine)

    Returns:
        OCRResult
    """
    import asyncio

    if event_logger is not None:
        asyncio.run(
            event_logger.log_event(
                "ocr.recognize.started",
                {"engine": config.engine, "language": config.language},
            )
        )

    if event_logger is not None:
        asyncio.run(
            event_logger.log_event(
                "ocr.recognize.image_loaded",
                {
                    "engine": config.engine,
                    "image_shape": str(image.shape) if hasattr(image, "shape") else "unknown",
                },
            )
        )

    if preprocessor is not None:
        image = preprocessor.process(image, config)
        if event_logger is not None:
            asyncio.run(
                event_logger.log_event(
                    "ocr.recognize.preprocessing_completed", {"engine": config.engine}
                )
            )

    if rate_limiter is not None and config.engine == "cloud":
        asyncio.run(
            rate_limiter.acquire(
                key=f"ocr:cloud:{config.engine}",
                max_requests=10,
                window=60,
            )
        )

    raw_result = engine.recognize(image, config)

    if event_logger is not None:
        asyncio.run(
            event_logger.log_event(
                "ocr.recognize.ocr_completed",
                {
                    "engine": config.engine,
                    "text_length": len(raw_result.get("text", "")),
                    "confidence": raw_result.get("confidence", 0.0),
                },
            )
        )

    result = OCRResult(
        text=raw_result["text"],
        lines=raw_result["lines"],
        language=raw_result.get("language", config.language),
        confidence=raw_result["confidence"],
        engine=config.engine,
        processing_time=time.time() - start_time,
        metadata=raw_result.get("metadata", {}),
    )

    if postprocessor is not None:
        result = postprocessor.process(result)
        if event_logger is not None:
            asyncio.run(
                event_logger.log_event(
                    "ocr.recognize.postprocessing_completed", {"engine": config.engine}
                )
            )

    if event_logger is not None:
        asyncio.run(
            event_logger.log_event(
                "ocr.recognize.completed",
                {
                    "engine": config.engine,
                    "processing_time": result.processing_time,
                    "confidence": result.confidence,
                },
            )
        )

    return result
