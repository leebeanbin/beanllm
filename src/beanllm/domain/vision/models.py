"""
Vision Models - 비전 태스크 모델 (2024-2025)

최신 비전 모델 래퍼 통합 모듈.

Main models (separate files):
- SAM (Segment Anything Model) - sam.py
- Florence-2 (Microsoft) - florence.py  
- YOLO (Object Detection) - yolo.py

Additional models (this file):
- Qwen3-VL (Vision-Language Model)
- EVA-CLIP (Vision Embeddings)
- DINOv2 (Self-Supervised Vision)

Requirements:
    pip install transformers torch pillow opencv-python ultralytics
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base_task_model import BaseVisionTaskModel

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)

# Re-export main models from separate files
from .florence import Florence2Wrapper
from .sam import SAMWrapper
from .yolo import YOLOWrapper

