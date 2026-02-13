"""
Vision Domain - 비전 및 멀티모달 도메인
"""

# Vision Task Models (선택적 의존성, 2024-2025)
from typing import Any, Optional, Type

from beanllm.domain.vision.base_task_model import BaseVisionTaskModel
from beanllm.domain.vision.embeddings import (
    CLIPEmbedding,
    MobileCLIPEmbedding,
    MultimodalEmbedding,
    SigLIPEmbedding,
    create_vision_embedding,
)
from beanllm.domain.vision.factory import create_vision_task_model, list_available_models
from beanllm.domain.vision.loaders import (
    ImageDocument,
    ImageLoader,
    PDFWithImagesLoader,
    load_images,
    load_pdf_with_images,
)

Florence2Wrapper: Optional[Type[Any]] = None
Qwen3VLWrapper: Optional[Type[Any]] = None
SAMWrapper: Optional[Type[Any]] = None
YOLOWrapper: Optional[Type[Any]] = None

try:
    from beanllm.domain.vision.sam import SAMWrapper as _SAM

    SAMWrapper = _SAM
except ImportError:
    pass

try:
    from beanllm.domain.vision.florence import Florence2Wrapper as _F2

    Florence2Wrapper = _F2
except ImportError:
    pass

try:
    from beanllm.domain.vision.yolo import YOLOWrapper as _Y

    YOLOWrapper = _Y
except ImportError:
    pass

__all__ = [
    # Base Classes
    "BaseVisionTaskModel",
    # Embeddings
    "CLIPEmbedding",
    "SigLIPEmbedding",
    "MobileCLIPEmbedding",
    "MultimodalEmbedding",
    "create_vision_embedding",
    # Loaders
    "ImageDocument",
    "ImageLoader",
    "PDFWithImagesLoader",
    "load_images",
    "load_pdf_with_images",
    # Task Models (2024-2025)
    "SAMWrapper",
    "Florence2Wrapper",
    "YOLOWrapper",
    "Qwen3VLWrapper",
    "create_vision_task_model",
    "list_available_models",
]
