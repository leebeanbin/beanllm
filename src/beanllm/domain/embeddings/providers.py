"""
Embeddings Providers - 임베딩 Provider 구현체들
"""

import os
from typing import List, Optional

from .base import BaseEmbedding

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embeddings

    Example:
        ```python
        from beanllm.domain.embeddings import OpenAIEmbedding

        emb = OpenAIEmbedding(model="text-embedding-3-small")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: OpenAI embedding 모델
            api_key: OpenAI API 키 (None이면 환경변수)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        # OpenAI 클라이언트 초기화
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbedding. Install it with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        try:
            response = await self.async_client.embeddings.create(
                input=texts, model=self.model, **self.kwargs
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(
                f"Embedded {len(texts)} texts using {self.model}, "
                f"usage: {response.usage.total_tokens} tokens"
            )

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.sync_client.embeddings.create(
                input=texts, model=self.model, **self.kwargs
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(
                f"Embedded {len(texts)} texts using {self.model}, "
                f"usage: {response.usage.total_tokens} tokens"
            )

            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise


class GeminiEmbedding(BaseEmbedding):
    """
    Google Gemini Embeddings

    Example:
        ```python
        from beanllm.domain.embeddings import GeminiEmbedding

        emb = GeminiEmbedding(model="models/embedding-001")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "models/embedding-001", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: Gemini embedding 모델
            api_key: Google API 키 (None이면 환경변수)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        # Gemini 클라이언트 초기화
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai is required for GeminiEmbedding. "
                "Install it with: pip install beanllm[gemini]"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=self.api_key)
        self.genai = genai

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # Gemini SDK는 async 지원 안 함, sync 사용
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            embeddings = []
            # Gemini는 배치 임베딩을 지원하지 않으므로 하나씩 처리
            for text in texts:
                result = self.genai.embed_content(model=self.model, content=text, **self.kwargs)
                embeddings.append(result["embedding"])

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise


class OllamaEmbedding(BaseEmbedding):
    """
    Ollama Embeddings (로컬)

    Example:
        ```python
        from beanllm.domain.embeddings import OllamaEmbedding

        emb = OllamaEmbedding(model="nomic-embed-text")
        vectors = emb.embed_sync(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", **kwargs
    ):
        """
        Args:
            model: Ollama embedding 모델
            base_url: Ollama 서버 URL
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is required for OllamaEmbedding. "
                "Install it with: pip install beanllm[ollama]"
            )

        self.client = ollama.Client(host=base_url)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # Ollama는 async 지원 안 함
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings(model=self.model, prompt=text)
                embeddings.append(response["embedding"])

            logger.info(f"Embedded {len(texts)} texts using Ollama {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            raise


class VoyageEmbedding(BaseEmbedding):
    """
    Voyage AI Embeddings

    Example:
        ```python
        from beanllm.domain.embeddings import VoyageEmbedding

        emb = VoyageEmbedding(model="voyage-2")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(self, model: str = "voyage-2", api_key: Optional[str] = None, **kwargs):
        """
        Args:
            model: Voyage AI 모델
            api_key: Voyage AI API 키
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai is required for VoyageEmbedding. Install it with: pip install voyageai"
            )

        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY not found in environment variables")

        self.client = voyageai.Client(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embed(texts=texts, model=self.model, **self.kwargs)

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return response.embeddings

        except Exception as e:
            logger.error(f"Voyage AI embedding failed: {e}")
            raise


class JinaEmbedding(BaseEmbedding):
    """
    Jina AI Embeddings

    Example:
        ```python
        from beanllm.domain.embeddings import JinaEmbedding

        emb = JinaEmbedding(model="jina-embeddings-v2-base-en")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self, model: str = "jina-embeddings-v2-base-en", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: Jina AI 모델
            api_key: Jina AI API 키
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")

        self.url = "https://api.jina.ai/v1/embeddings"

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {"model": self.model, "input": texts, **self.kwargs}

            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Jina AI embedding failed: {e}")
            raise


class MistralEmbedding(BaseEmbedding):
    """
    Mistral AI Embeddings

    Example:
        ```python
        from beanllm.domain.embeddings import MistralEmbedding

        emb = MistralEmbedding(model="mistral-embed")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(self, model: str = "mistral-embed", api_key: Optional[str] = None, **kwargs):
        """
        Args:
            model: Mistral AI 모델
            api_key: Mistral AI API 키
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        try:
            from mistralai.client import MistralClient
        except ImportError:
            raise ImportError(
                "mistralai is required for MistralEmbedding. Install it with: pip install mistralai"
            )

        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")

        self.client = MistralClient(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embeddings(model=self.model, input=texts)

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return embeddings

        except Exception as e:
            logger.error(f"Mistral AI embedding failed: {e}")
            raise


class CohereEmbedding(BaseEmbedding):
    """
    Cohere Embeddings

    Example:
        ```python
        from beanllm.domain.embeddings import CohereEmbedding

        emb = CohereEmbedding(model="embed-english-v3.0")
        vectors = await emb.embed(["text1", "text2"])
        ```
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: str = "search_document",
        **kwargs,
    ):
        """
        Args:
            model: Cohere embedding 모델
            api_key: Cohere API 키 (None이면 환경변수)
            input_type: "search_document", "search_query", "classification", "clustering"
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        # Cohere 클라이언트 초기화
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere is required for CohereEmbedding. Install it with: pip install cohere"
            )

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")

        self.client = cohere.Client(api_key=self.api_key)
        self.input_type = input_type

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # Cohere SDK는 async 지원 안 함, sync 사용
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embed(
                texts=texts, model=self.model, input_type=self.input_type, **self.kwargs
            )

            logger.info(f"Embedded {len(texts)} texts using {self.model}")
            return response.embeddings

        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            raise


class HuggingFaceEmbedding(BaseEmbedding):
    """
    HuggingFace Sentence Transformers 범용 임베딩 (로컬)

    sentence-transformers 라이브러리를 사용하여 HuggingFace Hub의
    모든 임베딩 모델을 지원합니다.

    지원 모델 예시:
    - NVIDIA NV-Embed: "nvidia/NV-Embed-v2" (MTEB #1, 69.32)
    - SFR-Embedding: "Salesforce/SFR-Embedding-Mistral"
    - GTE: "Alibaba-NLP/gte-large-en-v1.5"
    - BGE: "BAAI/bge-large-en-v1.5"
    - E5: "intfloat/e5-large-v2"
    - MiniLM: "sentence-transformers/all-MiniLM-L6-v2"
    - 기타 7,000+ 모델

    Features:
    - Lazy loading (첫 사용 시 모델 로드)
    - GPU/CPU 자동 선택
    - 배치 처리
    - 임베딩 정규화 옵션
    - Mean pooling with attention mask

    Example:
        ```python
        from beanllm.domain.embeddings import HuggingFaceEmbedding

        # NVIDIA NV-Embed (MTEB #1)
        emb = HuggingFaceEmbedding(model="nvidia/NV-Embed-v2", use_gpu=True)
        vectors = emb.embed_sync(["text1", "text2"])

        # SFR-Embedding-Mistral
        emb = HuggingFaceEmbedding(model="Salesforce/SFR-Embedding-Mistral")
        vectors = emb.embed_sync(["query: what is AI?"])

        # 경량 모델 (MiniLM, 22MB)
        emb = HuggingFaceEmbedding(model="sentence-transformers/all-MiniLM-L6-v2")
        vectors = emb.embed_sync(["text"])
        ```
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Args:
            model: HuggingFace 모델 이름
            use_gpu: GPU 사용 여부 (기본: True)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 32)
            **kwargs: 추가 파라미터 (max_seq_length 등)
        """
        super().__init__(model, **kwargs)

        self.use_gpu = use_gpu
        self.normalize = normalize
        self.batch_size = batch_size

        # Lazy loading
        self._model = None
        self._device = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for HuggingFaceEmbedding. "
                "Install it with: pip install sentence-transformers"
            )

        # Device 설정
        if self.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading HuggingFace model: {self.model} on {self._device}")

        # 모델 로드
        self._model = SentenceTransformer(self.model, device=self._device)

        # max_seq_length 설정 (kwargs에서)
        if "max_seq_length" in self.kwargs:
            self._model.max_seq_length = self.kwargs["max_seq_length"]

        logger.info(
            f"HuggingFace model loaded: {self.model} "
            f"(max_seq_length: {self._model.max_seq_length})"
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        # sentence-transformers는 async 지원 안 함, sync 사용
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        # 모델 로드
        self._load_model()

        try:
            # Encode with batch processing
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            logger.info(
                f"Embedded {len(texts)} texts using {self.model} "
                f"(shape: {embeddings.shape}, device: {self._device})"
            )

            # Convert to list
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise


class NVEmbedEmbedding(BaseEmbedding):
    """
    NVIDIA NV-Embed-v2 임베딩 (MTEB 1위, 2024-2025)

    NVIDIA의 최신 임베딩 모델로 MTEB 벤치마크 1위 (69.32)를 달성했습니다.

    성능:
    - MTEB Score: 69.32 (1위)
    - Retrieval: 60.92
    - Classification: 80.19
    - Clustering: 54.23
    - Pair Classification: 89.68
    - Reranking: 62.58
    - STS: 87.86

    Features:
    - Instruction-aware embedding
    - Passage 및 Query prefix 지원
    - Latent attention layer
    - 최대 32K 토큰 지원

    Example:
        ```python
        from beanllm.domain.embeddings import NVEmbedEmbedding

        # 기본 사용 (passage)
        emb = NVEmbedEmbedding(use_gpu=True)
        vectors = emb.embed_sync(["This is a passage."])

        # Query 임베딩
        emb = NVEmbedEmbedding(prefix="query")
        vectors = emb.embed_sync(["What is AI?"])

        # Instruction 사용
        emb = NVEmbedEmbedding(
            prefix="query",
            instruction="Retrieve relevant passages for the query"
        )
        vectors = emb.embed_sync(["machine learning"])
        ```
    """

    def __init__(
        self,
        model: str = "nvidia/NV-Embed-v2",
        use_gpu: bool = True,
        prefix: str = "passage",
        instruction: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Args:
            model: NVIDIA NV-Embed 모델 이름
            use_gpu: GPU 사용 여부 (기본: True, 권장)
            prefix: "passage" 또는 "query" (기본: "passage")
            instruction: 추가 instruction (선택)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 32)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        self.use_gpu = use_gpu
        self.prefix = prefix
        self.instruction = instruction
        self.normalize = normalize
        self.batch_size = batch_size

        # Lazy loading
        self._model = None
        self._device = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for NVEmbedEmbedding. "
                "Install it with: pip install sentence-transformers"
            )

        # Device 설정
        if self.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
            logger.warning("NV-Embed works best on GPU. CPU mode may be slow.")

        logger.info(f"Loading NVIDIA NV-Embed-v2 on {self._device}")

        # 모델 로드
        self._model = SentenceTransformer(self.model, device=self._device, trust_remote_code=True)

        logger.info(
            f"NVIDIA NV-Embed-v2 loaded (max_seq_length: {self._model.max_seq_length})"
        )

    def _prepare_texts(self, texts: List[str]) -> List[str]:
        """
        NV-Embed 포맷으로 텍스트 준비

        Format:
        - Passage: "passage: {text}"
        - Query: "query: {text}"
        - Instruction: "Instruct: {instruction}\nQuery: {text}"
        """
        prepared = []

        for text in texts:
            if self.instruction:
                # Instruction mode
                prepared_text = f"Instruct: {self.instruction}\nQuery: {text}"
            else:
                # Prefix mode
                prepared_text = f"{self.prefix}: {text}"

            prepared.append(prepared_text)

        return prepared

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        # 모델 로드
        self._load_model()

        try:
            # NV-Embed 포맷으로 준비
            prepared_texts = self._prepare_texts(texts)

            # Encode
            embeddings = self._model.encode(
                prepared_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            logger.info(
                f"Embedded {len(texts)} texts using NVIDIA NV-Embed-v2 "
                f"(prefix: {self.prefix}, shape: {embeddings.shape})"
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"NVIDIA NV-Embed embedding failed: {e}")
            raise
