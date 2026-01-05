"""
Vision Task Model Factory - 비전 태스크 모델 생성 함수

비전 태스크 모델을 쉽게 생성할 수 있는 Factory 함수를 제공합니다.
"""

from typing import Optional

from .base_task_model import BaseVisionTaskModel

try:
    from beanllm.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def create_vision_task_model(
    model: str,
    **kwargs,
) -> BaseVisionTaskModel:
    """
    비전 태스크 모델 생성 (Factory 함수)

    Args:
        model: 모델 종류
            - "sam" or "sam2": Segment Anything Model (Segmentation)
            - "florence2" or "florence-2": Florence-2 (Captioning, Detection, VQA)
            - "yolo": YOLO (Object Detection, Segmentation)
            - "qwen3vl" or "qwen-vl": Qwen3-VL (Vision-Language Model, VQA, Captioning, OCR)
        **kwargs: 모델별 초기화 파라미터
            - SAM: model_type="sam2_hiera_large", device=None
            - Florence-2: model_size="large", device=None
            - YOLO: version="11", model_size="m", task="detect"
            - Qwen3-VL: model_size="8B", device=None

    Returns:
        BaseVisionTaskModel 인스턴스

    Raises:
        ValueError: 알 수 없는 모델
        ImportError: 모델이 설치되지 않음

    Example:
        ```python
        from beanllm.domain.vision import create_vision_task_model

        # SAM 2
        sam = create_vision_task_model(
            model="sam2",
            model_type="sam2_hiera_large"
        )

        masks = sam.predict(
            image="photo.jpg",
            points=[[500, 375]],
            labels=[1]
        )

        # Florence-2
        florence = create_vision_task_model(
            model="florence2",
            model_size="large"
        )

        caption = florence.predict(
            image="photo.jpg",
            task="caption"
        )

        # YOLO
        yolo = create_vision_task_model(
            model="yolo",
            version="11",
            task="detect"
        )

        detections = yolo.predict(
            image="photo.jpg",
            conf=0.5
        )
        ```
    """
    model = model.lower()

    if model in ["sam", "sam2", "segment-anything"]:
        try:
            from .models import SAMWrapper
            logger.info("Creating SAM model")
            return SAMWrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "segment-anything or sam2 required. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git "
                "or pip install git+https://github.com/facebookresearch/sam2.git"
            )

    elif model in ["florence2", "florence-2", "florence"]:
        try:
            from .models import Florence2Wrapper
            logger.info("Creating Florence-2 model")
            return Florence2Wrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "transformers required for Florence-2. "
                "Install with: pip install transformers"
            )

    elif model in ["yolo", "yolov8", "yolov11", "yolov12"]:
        try:
            from .models import YOLOWrapper
            logger.info("Creating YOLO model")
            return YOLOWrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "ultralytics required for YOLO. "
                "Install with: pip install ultralytics"
            )

    elif model in ["qwen3vl", "qwen-vl", "qwen3-vl"]:
        try:
            from .models import Qwen3VLWrapper
            logger.info("Creating Qwen3-VL model")
            return Qwen3VLWrapper(**kwargs)
        except ImportError:
            raise ImportError(
                "transformers required for Qwen3-VL. "
                "Install with: pip install transformers torch"
            )

    else:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Available: sam, florence2, yolo, qwen3vl"
        )


def list_available_models() -> dict:
    """
    사용 가능한 비전 태스크 모델 목록

    Returns:
        {"model_name": "description", ...}

    Example:
        ```python
        from beanllm.domain.vision import list_available_models

        models = list_available_models()
        print(models)
        # {
        #     "sam": "Segment Anything Model - 제로샷 segmentation",
        #     "florence2": "Florence-2 - Captioning, Detection, VQA",
        #     "yolo": "YOLO - Object Detection, Segmentation"
        # }
        ```
    """
    return {
        "sam": "Segment Anything Model (SAM 3/SAM 2) - 제로샷 segmentation",
        "florence2": "Florence-2 (Microsoft) - Captioning, Detection, VQA",
        "yolo": "YOLO (YOLOv12/v11/v8) - Object Detection, Segmentation",
        "qwen3vl": "Qwen3-VL (Alibaba) - Vision-Language Model, VQA, Captioning, OCR",
    }
