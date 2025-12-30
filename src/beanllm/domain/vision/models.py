"""
Vision Models - 비전 태스크 모델 (2024-2025)

최신 비전 모델 래퍼:
- SAM (Segment Anything Model)
- Florence-2 (Microsoft)
- YOLO (Object Detection)

Requirements:
    pip install transformers torch pillow opencv-python ultralytics
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base_task_model import BaseVisionTaskModel

try:
    from ...utils.logger import get_logger
except ImportError:
    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class SAMWrapper(BaseVisionTaskModel):
    """
    Segment Anything Model (SAM) 래퍼

    Meta AI의 SAM은 제로샷 이미지 segmentation 모델입니다.

    SAM 특징:
    - 제로샷 segmentation
    - Point, Box, Mask prompt 지원
    - 10억+ 마스크 데이터로 훈련
    - SAM 2: 비디오 segmentation 지원

    Example:
        ```python
        from beanllm.domain.vision import SAMWrapper

        # SAM 2 사용 (최신)
        sam = SAMWrapper(model_type="sam2_hiera_large")

        # 이미지에서 객체 분할
        masks = sam.segment(
            image="photo.jpg",
            points=[[500, 375]],  # 클릭 포인트
            labels=[1]  # 1=foreground, 0=background
        )

        # 모든 객체 자동 분할
        all_masks = sam.segment_everything("photo.jpg")
        ```
    """

    def __init__(
        self,
        model_type: str = "sam2_hiera_large",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model_type: SAM 모델 타입
                - "sam2_hiera_large": SAM 2 Large (최신, 권장)
                - "sam2_hiera_base_plus": SAM 2 Base+
                - "sam2_hiera_small": SAM 2 Small
                - "sam2_hiera_tiny": SAM 2 Tiny
                - "sam_vit_h": SAM ViT-H (원본)
                - "sam_vit_l": SAM ViT-L
                - "sam_vit_b": SAM ViT-B
            device: 디바이스 (cuda/cpu/mps)
            **kwargs: 추가 설정
        """
        self.model_type = model_type
        self.kwargs = kwargs

        # Device 설정
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Lazy loading
        self._model = None
        self._predictor = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            if self.model_type.startswith("sam2"):
                # SAM 2
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                checkpoint = self._get_sam2_checkpoint()
                config = self._get_sam2_config()

                self._model = build_sam2(config, checkpoint, device=self.device)
                self._predictor = SAM2ImagePredictor(self._model)
            else:
                # SAM (원본)
                from segment_anything import sam_model_registry, SamPredictor

                checkpoint = self._get_sam_checkpoint()
                self._model = sam_model_registry[self.model_type](checkpoint=checkpoint)
                self._model.to(device=self.device)
                self._predictor = SamPredictor(self._model)

            logger.info(f"SAM model loaded: {self.model_type} on {self.device}")

        except ImportError:
            raise ImportError(
                "segment-anything or sam2 required. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git "
                "or pip install git+https://github.com/facebookresearch/sam2.git"
            )

    def _get_sam2_checkpoint(self) -> str:
        """SAM 2 체크포인트 경로"""
        checkpoint_map = {
            "sam2_hiera_large": "checkpoints/sam2_hiera_large.pt",
            "sam2_hiera_base_plus": "checkpoints/sam2_hiera_base_plus.pt",
            "sam2_hiera_small": "checkpoints/sam2_hiera_small.pt",
            "sam2_hiera_tiny": "checkpoints/sam2_hiera_tiny.pt",
        }
        return checkpoint_map.get(self.model_type, checkpoint_map["sam2_hiera_large"])

    def _get_sam2_config(self) -> str:
        """SAM 2 config 경로"""
        config_map = {
            "sam2_hiera_large": "sam2_hiera_l.yaml",
            "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml",
            "sam2_hiera_tiny": "sam2_hiera_t.yaml",
        }
        return config_map.get(self.model_type, config_map["sam2_hiera_large"])

    def _get_sam_checkpoint(self) -> str:
        """SAM 체크포인트 경로"""
        checkpoint_map = {
            "sam_vit_h": "checkpoints/sam_vit_h_4b8939.pth",
            "sam_vit_l": "checkpoints/sam_vit_l_0b3195.pth",
            "sam_vit_b": "checkpoints/sam_vit_b_01ec64.pth",
        }
        return checkpoint_map.get(self.model_type, checkpoint_map["sam_vit_h"])

    def segment(
        self,
        image: Union[str, Path, np.ndarray],
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        boxes: Optional[List[List[int]]] = None,
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """
        이미지 segmentation

        Args:
            image: 이미지 (경로 또는 numpy array)
            points: 포인트 프롬프트 [[x, y], ...]
            labels: 포인트 레이블 [1=foreground, 0=background]
            boxes: 박스 프롬프트 [[x1, y1, x2, y2], ...]
            multimask_output: 여러 마스크 출력 여부

        Returns:
            {"masks": np.ndarray, "scores": List[float], "logits": np.ndarray}
        """
        self._load_model()

        # 이미지 로드
        if isinstance(image, (str, Path)):
            from PIL import Image
            image_pil = Image.open(image).convert("RGB")
            image = np.array(image_pil)

        # 이미지 설정
        self._predictor.set_image(image)

        # Prompt 설정
        point_coords = np.array(points) if points else None
        point_labels = np.array(labels) if labels else None
        box_coords = np.array(boxes) if boxes else None

        # 예측
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords[0] if box_coords is not None and len(box_coords) == 1 else None,
            multimask_output=multimask_output,
        )

        return {
            "masks": masks,
            "scores": scores.tolist(),
            "logits": logits,
        }

    def segment_everything(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        자동으로 모든 객체 분할

        Args:
            image: 이미지

        Returns:
            [{"segmentation": mask, "area": int, "bbox": [x, y, w, h], "predicted_iou": float}, ...]
        """
        self._load_model()

        from segment_anything import SamAutomaticMaskGenerator

        # 이미지 로드
        if isinstance(image, (str, Path)):
            from PIL import Image
            image_pil = Image.open(image).convert("RGB")
            image = np.array(image_pil)

        # Mask generator
        mask_generator = SamAutomaticMaskGenerator(self._model)

        # 예측
        masks = mask_generator.generate(image)

        logger.info(f"SAM generated {len(masks)} masks")

        return masks

    # BaseVisionTaskModel 추상 메서드 구현

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        points: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        boxes: Optional[List[List[int]]] = None,
        multimask_output: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        예측 실행 (BaseVisionTaskModel 인터페이스)

        기본적으로 segment() 메서드를 호출합니다.

        Args:
            image: 이미지
            points: 포인트 프롬프트 (optional)
            labels: 포인트 레이블 (optional)
            boxes: 박스 프롬프트 (optional)
            multimask_output: 여러 마스크 출력 여부
            **kwargs: 추가 파라미터

        Returns:
            {"masks": np.ndarray, "scores": List[float], "logits": np.ndarray}
        """
        return self.segment(
            image=image,
            points=points,
            labels=labels,
            boxes=boxes,
            multimask_output=multimask_output,
        )

    def __repr__(self) -> str:
        return f"SAMWrapper(model_type={self.model_type}, device={self.device})"


class Florence2Wrapper(BaseVisionTaskModel):
    """
    Florence-2 모델 래퍼 (Microsoft)

    Microsoft의 Florence-2는 통합 비전-언어 모델입니다.

    Florence-2 특징:
    - Vision-Language 통합 모델
    - Object Detection, Segmentation, Captioning, VQA 통합
    - 0.2B/0.7B 파라미터 옵션
    - 오픈소스 (MIT License)

    Example:
        ```python
        from beanllm.domain.vision import Florence2Wrapper

        # Florence-2 모델 로드
        florence = Florence2Wrapper(model_size="large")

        # Image Captioning
        caption = florence.caption("image.jpg")
        print(caption)  # "A cat sitting on a couch"

        # Object Detection
        objects = florence.detect_objects("image.jpg")
        print(objects)  # [{"label": "cat", "box": [x1, y1, x2, y2], "score": 0.95}]

        # Visual Question Answering
        answer = florence.vqa("image.jpg", "What is the cat doing?")
        print(answer)  # "sitting"
        ```
    """

    def __init__(
        self,
        model_size: str = "large",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model_size: 모델 크기 (base/large)
                - "base": Florence-2-base (0.2B)
                - "large": Florence-2-large (0.7B)
            device: 디바이스
            **kwargs: 추가 설정
        """
        self.model_size = model_size
        self.kwargs = kwargs

        # Device 설정
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Lazy loading
        self._model = None
        self._processor = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            model_map = {
                "base": "microsoft/Florence-2-base",
                "large": "microsoft/Florence-2-large",
            }
            model_name = model_map.get(self.model_size, model_map["large"])

            logger.info(f"Loading Florence-2: {model_name}")

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(self.device)

            self._processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            logger.info("Florence-2 loaded successfully")

        except ImportError:
            raise ImportError("transformers required. Install with: pip install transformers")

    def _run_task(
        self,
        task: str,
        image: Union[str, Path, np.ndarray],
        text_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Florence-2 태스크 실행

        Args:
            task: 태스크 이름 (e.g., "<CAPTION>", "<DETAILED_CAPTION>")
            image: 이미지
            text_input: 추가 텍스트 입력

        Returns:
            결과 딕셔너리
        """
        self._load_model()

        # 이미지 로드
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        # 입력 준비
        if text_input:
            prompt = f"{task} {text_input}"
        else:
            prompt = task

        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        # 추론
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

        # 디코드
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # 파싱
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        return parsed

    def caption(
        self,
        image: Union[str, Path, np.ndarray],
        detailed: bool = False,
    ) -> str:
        """
        Image captioning

        Args:
            image: 이미지
            detailed: 상세 캡션 생성 여부

        Returns:
            캡션 텍스트
        """
        task = "<MORE_DETAILED_CAPTION>" if detailed else "<CAPTION>"
        result = self._run_task(task, image)
        return result.get(task, "")

    def detect_objects(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Object detection

        Args:
            image: 이미지

        Returns:
            [{"label": str, "box": [x1, y1, x2, y2], "score": float}, ...]
        """
        result = self._run_task("<OD>", image)
        return result.get("<OD>", {}).get("bboxes", [])

    def vqa(
        self,
        image: Union[str, Path, np.ndarray],
        question: str,
    ) -> str:
        """
        Visual Question Answering

        Args:
            image: 이미지
            question: 질문

        Returns:
            답변
        """
        result = self._run_task("<VQA>", image, text_input=question)
        return result.get("<VQA>", "")

    # BaseVisionTaskModel 추상 메서드 구현

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        task: str = "caption",
        **kwargs,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        예측 실행 (BaseVisionTaskModel 인터페이스)

        Args:
            image: 이미지
            task: 태스크 종류 (caption/detect/vqa)
            **kwargs: 태스크별 추가 파라미터
                - caption: detailed=False
                - vqa: question (필수)

        Returns:
            태스크별 결과
            - caption: str
            - detect: List[Dict]
            - vqa: str

        Example:
            ```python
            # Caption
            caption = model.predict(image="photo.jpg", task="caption")

            # Object detection
            objects = model.predict(image="photo.jpg", task="detect")

            # VQA
            answer = model.predict(
                image="photo.jpg",
                task="vqa",
                question="What is this?"
            )
            ```
        """
        if task == "caption":
            return self.caption(image, **kwargs)
        elif task == "detect":
            return self.detect_objects(image)
        elif task == "vqa":
            if "question" not in kwargs:
                raise ValueError("VQA task requires 'question' parameter")
            return self.vqa(image, kwargs["question"])
        else:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Available: caption, detect, vqa"
            )

    def __repr__(self) -> str:
        return f"Florence2Wrapper(model_size={self.model_size}, device={self.device})"


class YOLOWrapper(BaseVisionTaskModel):
    """
    YOLO (You Only Look Once) 래퍼

    Ultralytics의 YOLOv8/YOLOv11 object detection 모델.

    YOLO 특징:
    - 실시간 object detection
    - Detection, Segmentation, Pose, Classification 지원
    - YOLOv11: 최신 버전 (2024)
    - 다양한 모델 크기 (n/s/m/l/x)

    Example:
        ```python
        from beanllm.domain.vision import YOLOWrapper

        # YOLOv11 사용
        yolo = YOLOWrapper(version="11", model_size="m")

        # Object detection
        results = yolo.detect("image.jpg")
        for obj in results:
            print(f"{obj['class']}: {obj['confidence']:.2f}, box: {obj['box']}")

        # Segmentation
        yolo = YOLOWrapper(version="11", task="segment")
        results = yolo.segment("image.jpg")
        ```
    """

    def __init__(
        self,
        version: str = "11",
        model_size: str = "m",
        task: str = "detect",
        **kwargs,
    ):
        """
        Args:
            version: YOLO 버전 (8/9/10/11)
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
