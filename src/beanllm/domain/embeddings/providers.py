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
    Voyage AI Embeddings (v3 시리즈, 2024-2025)

    Voyage AI v3는 특정 벤치마크에서 #1 성능을 달성한 최신 임베딩입니다.

    모델 라인업:
    - voyage-3-large: 최고 성능 (특정 태스크 1위)
    - voyage-3: 범용 고성능
    - voyage-3.5: 균형잡힌 성능
    - voyage-code-3: 코드 임베딩 특화
    - voyage-multimodal-3: 멀티모달 지원

    Example:
        ```python
        from beanllm.domain.embeddings import VoyageEmbedding

        # v3-large (최고 성능)
        emb = VoyageEmbedding(model="voyage-3-large")
        vectors = await emb.embed(["text1", "text2"])

        # 코드 임베딩
        emb = VoyageEmbedding(model="voyage-code-3")
        vectors = await emb.embed(["def hello(): print('world')"])

        # 멀티모달
        emb = VoyageEmbedding(model="voyage-multimodal-3")
        vectors = await emb.embed(["text with image context"])
        ```
    """

    def __init__(self, model: str = "voyage-3", api_key: Optional[str] = None, **kwargs):
        """
        Args:
            model: Voyage AI 모델 (v3 시리즈)
                - voyage-3-large: 최고 성능
                - voyage-3: 범용 (기본값)
                - voyage-3.5: 균형
                - voyage-code-3: 코드
                - voyage-multimodal-3: 멀티모달
            api_key: Voyage AI API 키
            **kwargs: 추가 파라미터 (input_type, truncation 등)
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
    Jina AI Embeddings (v3 시리즈, 2024-2025)

    Jina AI v3는 89개 언어 지원, LoRA 어댑터, Matryoshka 임베딩을 제공합니다.

    주요 기능:
    - 89개 언어 지원 (다국어 최강)
    - LoRA 어댑터로 도메인 특화 fine-tuning
    - Matryoshka 표현 학습 (가변 차원)
    - 8192 컨텍스트 윈도우

    모델 라인업:
    - jina-embeddings-v3: 다목적 (1024 dim, 기본값)
    - jina-clip-v2: 멀티모달 (이미지 + 텍스트)
    - jina-colbert-v2: Late interaction retrieval

    Example:
        ```python
        from beanllm.domain.embeddings import JinaEmbedding

        # v3 기본 모델 (89개 언어)
        emb = JinaEmbedding(model="jina-embeddings-v3")
        vectors = await emb.embed(["Hello", "안녕하세요", "こんにちは"])

        # Matryoshka - 가변 차원
        emb = JinaEmbedding(model="jina-embeddings-v3", dimensions=256)
        vectors = await emb.embed(["text"])  # 256차원 출력

        # 태스크별 최적화
        emb = JinaEmbedding(model="jina-embeddings-v3", task="retrieval.passage")
        vectors = await emb.embed(["This is a document passage."])
        ```
    """

    def __init__(
        self, model: str = "jina-embeddings-v3", api_key: Optional[str] = None, **kwargs
    ):
        """
        Args:
            model: Jina AI 모델 (v3 시리즈)
                - jina-embeddings-v3: 범용 다국어 (기본값)
                - jina-clip-v2: 멀티모달
                - jina-colbert-v2: Late interaction
            api_key: Jina AI API 키
            **kwargs: 추가 파라미터
                - dimensions: Matryoshka 차원 (64, 128, 256, 512, 1024)
                - task: "retrieval.query", "retrieval.passage", "text-matching", "classification" 등
                - late_chunking: 청킹 최적화 (bool)
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


class Qwen3Embedding(BaseEmbedding):
    """
    Qwen3-Embedding - Alibaba의 최신 임베딩 모델 (2025년)

    Qwen3-Embedding 특징:
    - Alibaba Cloud의 최신 임베딩 모델 (2025년 1월 출시)
    - 8B 파라미터 (대규모 성능)
    - 다국어 지원 (영어, 중국어, 일본어, 한국어 등)
    - MTEB 벤치마크 상위권
    - 긴 컨텍스트 지원 (8192 토큰)

    지원 모델:
    - Qwen/Qwen3-Embedding-8B: 메인 모델 (8B 파라미터)
    - Qwen/Qwen3-Embedding-1.5B: 경량 모델

    Example:
        ```python
        from beanllm.domain.embeddings import Qwen3Embedding

        # Qwen3-Embedding-8B 사용
        emb = Qwen3Embedding(model="Qwen/Qwen3-Embedding-8B", use_gpu=True)
        vectors = emb.embed_sync(["텍스트 1", "텍스트 2"])

        # 경량 모델 사용
        emb = Qwen3Embedding(model="Qwen/Qwen3-Embedding-1.5B")
        vectors = emb.embed_sync(["text"])
        ```

    References:
        - https://huggingface.co/Qwen/Qwen3-Embedding-8B
        - https://qwenlm.github.io/
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Embedding-8B",
        use_gpu: bool = True,
        normalize: bool = True,
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Args:
            model: Qwen3 모델 이름 (Qwen/Qwen3-Embedding-8B 또는 1.5B)
            use_gpu: GPU 사용 여부 (기본: True)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 16, 8B 모델용)
            **kwargs: 추가 파라미터
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
                "sentence-transformers is required for Qwen3Embedding. "
                "Install it with: pip install sentence-transformers"
            )

        # Device 설정
        if self.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading Qwen3 model: {self.model} on {self._device}")

        # 모델 로드
        self._model = SentenceTransformer(self.model, device=self._device)

        logger.info(
            f"Qwen3 model loaded: {self.model} "
            f"(max_seq_length: {self._model.max_seq_length})"
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기, 내부적으로 동기 사용)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        self._load_model()

        try:
            # Sentence Transformers로 임베딩
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            logger.info(
                f"Embedded {len(texts)} texts using {self.model} "
                f"(shape: {embeddings.shape})"
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Qwen3 embedding failed: {e}")
            raise


class CodeEmbedding(BaseEmbedding):
    """
    Code Embedding - 코드 전용 임베딩 모델 (2024-2025)

    코드 검색, 코드 이해, 코드 생성을 위한 전용 임베딩입니다.

    지원 모델:
    - microsoft/codebert-base: CodeBERT (기본)
    - microsoft/graphcodebert-base: GraphCodeBERT (그래프 구조 이해)
    - microsoft/unixcoder-base: UniXcoder (다국어 코드)
    - Salesforce/codet5-base: CodeT5 (코드-텍스트)

    Features:
    - 프로그래밍 언어 자동 감지
    - 코드 구조 이해 (AST, 데이터 플로우)
    - 자연어-코드 간 의미 매칭
    - 코드 검색 및 유사도 비교

    Example:
        ```python
        from beanllm.domain.embeddings import CodeEmbedding

        # CodeBERT 사용
        emb = CodeEmbedding(model="microsoft/codebert-base")

        # 코드 임베딩
        code_vectors = emb.embed_sync([
            "def hello(): print('Hello')",
            "function hello() { console.log('Hello'); }"
        ])

        # 자연어 쿼리로 코드 검색
        query_vec = emb.embed_sync(["print hello to console"])[0]
        # query_vec와 code_vectors 비교하여 관련 코드 찾기
        ```

    Use Cases:
    - 코드 검색 (Semantic Code Search)
    - 코드 복제 감지 (Clone Detection)
    - 코드 문서화 자동 생성
    - 코드 추천 시스템

    References:
        - CodeBERT: https://arxiv.org/abs/2002.08155
        - GraphCodeBERT: https://arxiv.org/abs/2009.08366
        - UniXcoder: https://arxiv.org/abs/2203.03850
    """

    def __init__(
        self,
        model: str = "microsoft/codebert-base",
        use_gpu: bool = True,
        normalize: bool = True,
        batch_size: int = 16,
        **kwargs,
    ):
        """
        Args:
            model: 코드 임베딩 모델
                - microsoft/codebert-base: CodeBERT (기본)
                - microsoft/graphcodebert-base: GraphCodeBERT
                - microsoft/unixcoder-base: UniXcoder
                - Salesforce/codet5-base: CodeT5
            use_gpu: GPU 사용 여부 (기본: True)
            normalize: 임베딩 정규화 여부 (기본: True)
            batch_size: 배치 크기 (기본: 16)
            **kwargs: 추가 파라미터
        """
        super().__init__(model, **kwargs)

        self.use_gpu = use_gpu
        self.normalize = normalize
        self.batch_size = batch_size

        # Lazy loading
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """모델 로딩 (lazy loading)"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers is required for CodeEmbedding. "
                "Install it with: pip install transformers torch"
            )

        # Device 설정
        if self.use_gpu and torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        logger.info(f"Loading Code model: {self.model} on {self._device}")

        # 모델 및 토크나이저 로드
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModel.from_pretrained(self.model)
        self._model.to(self._device)
        self._model.eval()

        logger.info(f"Code model loaded: {self.model}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        import torch

        token_embeddings = model_output[0]  # First element = token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """코드들을 임베딩 (비동기, 내부적으로 동기 사용)"""
        return self.embed_sync(texts)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """코드들을 임베딩 (동기)"""
        self._load_model()

        try:
            import torch

            all_embeddings = []

            # 배치 처리
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # 토크나이징
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                # 추론
                with torch.no_grad():
                    model_output = self._model(**encoded)

                # Mean pooling
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                # 정규화
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # CPU로 이동 및 리스트 변환
                batch_embeddings = embeddings.cpu().numpy().tolist()
                all_embeddings.extend(batch_embeddings)

            logger.info(
                f"Embedded {len(texts)} code snippets using {self.model} "
                f"(batch_size: {self.batch_size})"
            )

            return all_embeddings

        except Exception as e:
            logger.error(f"Code embedding failed: {e}")
            raise
