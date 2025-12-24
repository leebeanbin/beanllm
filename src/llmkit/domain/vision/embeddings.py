"""
Vision Embeddings - 이미지 임베딩 및 멀티모달 임베딩
"""

from pathlib import Path
from typing import List, Optional, Union

from ...domain.embeddings import BaseEmbedding


class CLIPEmbedding(BaseEmbedding):
    """
    CLIP 임베딩

    텍스트와 이미지를 동일한 벡터 공간에 임베딩

    Example:
        embed = CLIPEmbedding()

        # 텍스트 임베딩
        text_vec = embed.embed_sync(["a cat"])

        # 이미지 임베딩
        image_vec = embed.embed_images(["cat.jpg"])

        # 유사도 계산
        similarity = embed.similarity(text_vec[0], image_vec[0])
    """

    def __init__(self, model: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Args:
            model: CLIP 모델 이름
            device: 디바이스 (cuda, cpu 등)
        """
        super().__init__(model=model)
        self.device = device or "cpu"
        self._model = None
        self._processor = None

    def _load_model(self):
        """모델 로드 (lazy loading)"""
        if self._model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
            except ImportError:
                raise ImportError("transformers 및 torch 필요:\npip install transformers torch")

            self._processor = CLIPProcessor.from_pretrained(self.model)
            self._model = CLIPModel.from_pretrained(self.model)
            self._model.to(self.device)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 임베딩

        Args:
            texts: 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        self._load_model()

        import torch

        # 입력 처리
        inputs = self._processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 임베딩 생성
        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().tolist()

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """비동기 텍스트 임베딩"""
        return self.embed_sync(texts)

    def embed_images(self, images: List[Union[str, Path]], **kwargs) -> List[List[float]]:
        """
        이미지 임베딩

        Args:
            images: 이미지 파일 경로 리스트

        Returns:
            임베딩 벡터 리스트

        Example:
            vecs = embed.embed_images(["cat.jpg", "dog.jpg"])
        """
        self._load_model()

        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow 필요:\npip install pillow")

        # 이미지 로드
        pil_images = [Image.open(img) for img in images]

        # 입력 처리
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 임베딩 생성
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy().tolist()

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        코사인 유사도

        Args:
            vec1: 벡터 1
            vec2: 벡터 2

        Returns:
            유사도 (0.0 ~ 1.0)
        """
        try:
            import numpy as np
        except ImportError:
            # numpy 없으면 수동 계산
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b))  # 이미 normalized됨


class MultimodalEmbedding(BaseEmbedding):
    """
    멀티모달 임베딩

    텍스트와 이미지를 함께 처리

    Example:
        embed = MultimodalEmbedding()

        # 텍스트 + 이미지 임베딩
        vec = embed.embed_multimodal(
            text="a cat sitting on a mat",
            image="cat.jpg"
        )
    """

    def __init__(
        self,
        text_model: str = "text-embedding-3-small",
        vision_model: str = "openai/clip-vit-base-patch32",
        fusion_method: str = "concat",  # concat, average, weighted
    ):
        """
        Args:
            text_model: 텍스트 임베딩 모델
            vision_model: 비전 임베딩 모델
            fusion_method: 융합 방법 (concat, average, weighted)
        """
        super().__init__(model=text_model)
        try:
            from ...domain.embeddings import Embedding  # 이미 위에서 import됨
        except ImportError:
            from ...domain.embeddings import Embedding

        self.text_embedder = Embedding(model=text_model)
        self.vision_embedder = CLIPEmbedding(model=vision_model)
        self.fusion_method = fusion_method

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트만 임베딩"""
        return self.text_embedder.embed_sync(texts)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """비동기 텍스트 임베딩"""
        return self.embed_sync(texts)

    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path]] = None,
        text_weight: float = 0.5,
    ) -> List[float]:
        """
        멀티모달 임베딩

        Args:
            text: 텍스트 (옵션)
            image: 이미지 경로 (옵션)
            text_weight: 텍스트 가중치 (fusion_method='weighted'일 때)

        Returns:
            임베딩 벡터
        """
        if not text and not image:
            raise ValueError("At least one of text or image must be provided")

        vectors = []

        # 텍스트 임베딩
        if text:
            text_vec = self.text_embedder.embed_sync([text])[0]
            vectors.append(("text", text_vec))

        # 이미지 임베딩
        if image:
            image_vec = self.vision_embedder.embed_images([image])[0]
            vectors.append(("vision", image_vec))

        # 융합
        if len(vectors) == 1:
            return vectors[0][1]

        return self._fuse_vectors(vectors, text_weight)

    def _fuse_vectors(self, vectors: List[tuple], text_weight: float) -> List[float]:
        """
        벡터 융합

        Args:
            vectors: [(type, vector), ...] 리스트
            text_weight: 텍스트 가중치

        Returns:
            융합된 벡터
        """
        try:
            import numpy as np
        except ImportError:
            # numpy 없으면 concat만 지원
            if self.fusion_method == "concat":
                return [v for _, vec in vectors for v in vec]
            else:
                raise ImportError("numpy required for fusion methods other than 'concat'")

        if self.fusion_method == "concat":
            # 연결
            return [v for _, vec in vectors for v in vec]

        elif self.fusion_method == "average":
            # 평균
            arrays = [np.array(vec) for _, vec in vectors]
            return np.mean(arrays, axis=0).tolist()

        elif self.fusion_method == "weighted":
            # 가중 평균
            text_vecs = [vec for type, vec in vectors if type == "text"]
            vision_vecs = [vec for type, vec in vectors if type == "vision"]

            if text_vecs and vision_vecs:
                text_arr = np.array(text_vecs[0])
                vision_arr = np.array(vision_vecs[0])
                fused = text_weight * text_arr + (1 - text_weight) * vision_arr
                return fused.tolist()
            else:
                # 하나만 있으면 그대로 반환
                return vectors[0][1]

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


# 편의 함수
def create_vision_embedding(model: str = "clip", **kwargs) -> BaseEmbedding:
    """
    Vision 임베딩 생성 (간편 함수)

    Args:
        model: 모델 타입 (clip, multimodal)
        **kwargs: 추가 파라미터

    Returns:
        임베딩 인스턴스

    Example:
        # CLIP
        embed = create_vision_embedding("clip")

        # Multimodal
        embed = create_vision_embedding("multimodal", fusion_method="concat")
    """
    if model == "clip":
        return CLIPEmbedding(**kwargs)
    elif model == "multimodal":
        return MultimodalEmbedding(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")
