"""
YOLO (You Only Look Once) Wrapper

Ultralytics YOLOv12 객체 검출 및 세그멘테이션 모델 래퍼.

Features:
- Object Detection
- Instance Segmentation
- Pose Estimation
- Real-time Inference

Requirements:
    pip install ultralytics opencv-python
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

class YOLOWrapper(BaseVisionTaskModel):
    """
    YOLO (You Only Look Once) 래퍼 (2025년 최신)

    Ultralytics의 YOLO object detection 모델.

    YOLO 버전:
    - YOLOv12 (2025년 2월): Attention-centric architecture, 40.6% mAP
    - YOLOv11 (2024): Improved efficiency
    - YOLOv10: Dual label assignment
    - YOLOv8: Baseline

    YOLO 특징:
    - 실시간 object detection
    - Detection, Segmentation, Pose, Classification 지원
    - 다양한 모델 크기 (n/s/m/l/x)

    YOLOv12 주요 개선:
    - 2.1%/1.2% mAP 향상 (vs v10/v11)
    - Attention-centric architecture
    - 더욱 빠른 추론 속도
    - 40.6% mAP on COCO val2017

    Example:
        ```python
        from beanllm.domain.vision import YOLOWrapper

        # YOLOv12 사용 (최신, 권장)
        yolo = YOLOWrapper(version="12", model_size="m")

        # Object detection
        results = yolo.detect("image.jpg")
        for obj in results:
            print(f"{obj['class']}: {obj['confidence']:.2f}, box: {obj['box']}")

        # Segmentation
        yolo = YOLOWrapper(version="12", task="segment")
        results = yolo.segment("image.jpg")
        ```

    References:
        - YOLOv12: NeurIPS 2025
        - GitHub: https://github.com/ultralytics/ultralytics
    """

    def __init__(
        self,
        version: str = "12",
        model_size: str = "m",
        task: str = "detect",
        **kwargs,
    ):
        """
        Args:
            version: YOLO 버전
                - "12": YOLOv12 (최신, 권장, 2025년 2월)
                - "11": YOLOv11 (2024)
                - "10": YOLOv10
                - "8": YOLOv8
            model_size: 모델 크기 (n/s/m/l/x)
                - n: Nano (가장 빠름)
                - s: Small
                - m: Medium (균형, 권장)
                - l: Large
                - x: XLarge (가장 정확)
            task: 태스크 (detect/segment/pose/classify)
            **kwargs: 추가 설정
        """
        self.version = version
        self.model_size = model_size
        self.task = task
        self.kwargs = kwargs

        # Lazy loading
        self._model = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            # 모델 이름 생성
            model_name = f"yolo{self.version}{self.model_size}"
            if self.task != "detect":
                model_name += f"-{self.task}"
            model_name += ".pt"

            logger.info(f"Loading YOLO: {model_name}")

            self._model = YOLO(model_name)

            logger.info("YOLO loaded successfully")

        except ImportError:
            raise ImportError("ultralytics required. Install with: pip install ultralytics")

    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Object detection

        Args:
            image: 이미지
            conf: 신뢰도 임계값
            iou: IoU 임계값

        Returns:
            [{"class": str, "confidence": float, "box": [x1, y1, x2, y2]}, ...]
        """
        self._load_model()

        # 추론
        results = self._model(image, conf=conf, iou=iou)

        # 결과 파싱
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                })

        logger.info(f"YOLO detected {len(detections)} objects")

        return detections

    def segment(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Instance segmentation

        Args:
            image: 이미지
            conf: 신뢰도 임계값
            iou: IoU 임계값

        Returns:
            [{"class": str, "confidence": float, "box": [...], "mask": np.ndarray}, ...]
        """
        if self.task != "segment":
            logger.warning("YOLOWrapper task is not 'segment'. Switching to segment.")
            self.task = "segment"
            self._model = None  # 모델 재로드

        self._load_model()

        # 추론
        results = self._model(image, conf=conf, iou=iou)

        # 결과 파싱
        segments = []
        for result in results:
            if result.masks is None:
                continue

            for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
                segments.append({
                    "class": result.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "box": box.xyxy[0].tolist(),
                    "mask": mask.data.cpu().numpy(),
                })

        logger.info(f"YOLO segmented {len(segments)} objects")

        return segments

    # BaseVisionTaskModel 추상 메서드 구현

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.7,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        예측 실행 (BaseVisionTaskModel 인터페이스)

        태스크에 따라 detect() 또는 segment()를 호출합니다.

        Args:
            image: 이미지
            conf: 신뢰도 임계값
            iou: IoU 임계값
            **kwargs: 추가 파라미터

        Returns:
            Detection 또는 Segmentation 결과

        Example:
            ```python
            # Detection
            detections = model.predict("photo.jpg", conf=0.5)

            # Segmentation (task="segment"로 초기화된 경우)
            segments = model.predict("photo.jpg", conf=0.5)
            ```
        """
        if self.task == "segment":
            return self.segment(image=image, conf=conf, iou=iou)
        else:
            # detect가 기본
            return self.detect(image=image, conf=conf, iou=iou)

    def __repr__(self) -> str:
        return f"YOLOWrapper(version={self.version}, size={self.model_size}, task={self.task})"


