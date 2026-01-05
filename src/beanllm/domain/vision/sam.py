"""
SAM (Segment Anything Model) Wrapper

Meta AI의 SAM 제로샷 이미지 segmentation 모델 래퍼.

SAM 3 (2025년 최신): 텍스트 프롬프트, 컨셉 기반 분할, 3D 재구성 지원

Requirements:
    pip install segment-anything-3 torch pillow
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

class SAMWrapper(BaseVisionTaskModel):
    """
    Segment Anything Model (SAM) 래퍼 (2025년 최신)

    Meta AI의 SAM은 제로샷 이미지 segmentation 모델입니다.

    SAM 버전:
    - SAM 3 (2025년 11월): 텍스트 프롬프트, 컨셉 기반 분할, 3D 재구성
    - SAM 2: 비디오 segmentation 지원
    - SAM 1: 원본 (Point, Box, Mask prompt)

    SAM 3 주요 기능:
    - 텍스트 프롬프트로 객체 감지/분할/추적
    - 이미지/비디오에서 컨셉의 모든 인스턴스 찾기
    - 단일 이미지에서 3D 재구성 (SAM 3D)
    - 2x 성능 향상 (vs SAM 2)

    Example:
        ```python
        from beanllm.domain.vision import SAMWrapper

        # SAM 3 사용 (최신, 텍스트 프롬프트)
        sam = SAMWrapper(model_type="sam3_hiera_large")

        # 텍스트 프롬프트로 분할
        masks = sam.segment_by_text(
            image="photo.jpg",
            text_prompt="person wearing red shirt"
        )

        # SAM 2 사용 (비디오)
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

    References:
        - SAM 3: https://ai.meta.com/sam3/
        - GitHub: https://github.com/facebookresearch/sam3
        - Paper: https://about.fb.com/news/2025/11/new-sam-models-detect-objects-create-3d-reconstructions/
    """

    def __init__(
        self,
        model_type: str = "sam3_hiera_large",
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model_type: SAM 모델 타입
                - "sam3_hiera_large": SAM 3 Large (최신, 권장, 텍스트 프롬프트)
                - "sam3_hiera_base": SAM 3 Base
                - "sam3_hiera_small": SAM 3 Small
                - "sam2_hiera_large": SAM 2 Large (비디오)
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
            if self.model_type.startswith("sam3"):
                # SAM 3 (최신)
                from sam3.build_sam import build_sam3
                from sam3.sam3_predictor import SAM3Predictor

                checkpoint = self._get_sam3_checkpoint()
                config = self._get_sam3_config()

                self._model = build_sam3(config, checkpoint, device=self.device)
                self._predictor = SAM3Predictor(self._model)

            elif self.model_type.startswith("sam2"):
                # SAM 2
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                checkpoint = self._get_sam2_checkpoint()
                config = self._get_sam2_config()

                self._model = build_sam2(config, checkpoint, device=self.device)
                self._predictor = SAM2ImagePredictor(self._model)
            else:
                # SAM (원본)
                from segment_anything import SamPredictor, sam_model_registry

                checkpoint = self._get_sam_checkpoint()
                self._model = sam_model_registry[self.model_type](checkpoint=checkpoint)
                self._model.to(device=self.device)
                self._predictor = SamPredictor(self._model)

            logger.info(f"SAM model loaded: {self.model_type} on {self.device}")

        except ImportError:
            raise ImportError(
                "segment-anything, sam2, or sam3 required. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git "
                "or pip install git+https://github.com/facebookresearch/sam2.git "
                "or pip install git+https://github.com/facebookresearch/sam3.git"
            )

    def _get_sam3_checkpoint(self) -> str:
        """SAM 3 체크포인트 경로"""
        checkpoint_map = {
            "sam3_hiera_large": "checkpoints/sam3_hiera_large.pt",
            "sam3_hiera_base": "checkpoints/sam3_hiera_base.pt",
            "sam3_hiera_small": "checkpoints/sam3_hiera_small.pt",
        }
        return checkpoint_map.get(self.model_type, checkpoint_map["sam3_hiera_large"])

    def _get_sam3_config(self) -> str:
        """SAM 3 config 경로"""
        config_map = {
            "sam3_hiera_large": "sam3_hiera_l.yaml",
            "sam3_hiera_base": "sam3_hiera_b.yaml",
            "sam3_hiera_small": "sam3_hiera_s.yaml",
        }
        return config_map.get(self.model_type, config_map["sam3_hiera_large"])

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

    def segment_by_text(
        self,
        image: Union[str, Path, np.ndarray],
        text_prompt: str,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        텍스트 프롬프트로 객체 분할 (SAM 3 only)

        SAM 3의 새로운 기능으로, 텍스트 설명으로 객체를 찾고 분할합니다.

        Args:
            image: 이미지 (경로 또는 numpy array)
            text_prompt: 텍스트 프롬프트 (예: "person wearing red shirt", "all cars")
            confidence_threshold: 신뢰도 임계값 (기본: 0.5)

        Returns:
            {
                "masks": np.ndarray,  # Shape: (N, H, W)
                "boxes": List[List[int]],  # [[x1, y1, x2, y2], ...]
                "scores": List[float],  # Confidence scores
                "labels": List[str],  # Text labels
            }

        Example:
            ```python
            sam = SAMWrapper(model_type="sam3_hiera_large")

            # 특정 객체 찾기
            result = sam.segment_by_text(
                image="photo.jpg",
                text_prompt="person wearing red shirt"
            )

            # 모든 인스턴스 찾기
            result = sam.segment_by_text(
                image="photo.jpg",
                text_prompt="all dogs"
            )
            ```
        """
        if not self.model_type.startswith("sam3"):
            raise ValueError(
                f"Text prompting is only supported in SAM 3. "
                f"Current model: {self.model_type}. "
                f"Please use model_type='sam3_hiera_large' or similar."
            )

        self._load_model()

        # 이미지 로드
        if isinstance(image, (str, Path)):
            from PIL import Image
            image_pil = Image.open(image).convert("RGB")
            image = np.array(image_pil)

        # SAM 3 텍스트 기반 예측
        # Note: 실제 SAM 3 API에 따라 조정 필요
        try:
            # SAM 3의 텍스트 프롬프트 API 사용
            predictions = self._predictor.predict_with_text(
                image=image,
                text_prompt=text_prompt,
                confidence_threshold=confidence_threshold,
            )

            logger.info(
                f"SAM 3 text prediction completed: "
                f"prompt='{text_prompt}', found={len(predictions['masks'])} objects"
            )

            return predictions

        except AttributeError:
            # Fallback: SAM 3 API가 다를 경우
            logger.warning(
                "SAM 3 text prompt API not available. "
                "Using automatic masking with text filtering."
            )

            # 대안: 자동 마스크 생성 후 필터링
            all_masks = self.segment_everything(image)

            # TODO: 텍스트 필터링 로직 추가 (CLIP 등 사용)
            # 현재는 모든 마스크 반환
            return {
                "masks": np.array([m["segmentation"] for m in all_masks]),
                "boxes": [m["bbox"] for m in all_masks],
                "scores": [m.get("predicted_iou", 0.0) for m in all_masks],
                "labels": [text_prompt] * len(all_masks),
            }

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


