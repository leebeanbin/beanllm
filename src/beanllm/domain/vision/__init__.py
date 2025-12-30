"""
Vision Domain - 비전 및 멀티모달 도메인
"""

from .embeddings import (
    CLIPEmbedding,
    MobileCLIPEmbedding,
    MultimodalEmbedding,
    SigLIPEmbedding,
    create_vision_embedding,
)
from .loaders import (
    ImageDocument,
    ImageLoader,
    PDFWithImagesLoader,
    load_images,
    load_pdf_with_images,
)

# Vision Task Models (선택적 의존성, 2024-2025)
try:
    from .models import Florence2Wrapper, SAMWrapper, YOLOWrapper
except ImportError:
    Florence2Wrapper = None  # type: ignore
    SAMWrapper = None  # type: ignore
    YOLOWrapper = None  # type: ignore

__all__ = [
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
]
