"""
Florence-2 Wrapper (Microsoft)

Microsoft의 Florence-2 통합 비전-언어 모델 래퍼.

Features:
- Object Detection & Captioning
- Visual Question Answering (VQA)  
- OCR & Text Recognition
- Dense Captioning

Requirements:
    pip install transformers torch pillow
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
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

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


