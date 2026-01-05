"""
API-Based Embeddings - API 기반 임베딩 Provider 구현체들

이 모듈은 외부 API를 사용하는 7개의 임베딩 Provider를 포함합니다:
- OpenAIEmbedding: OpenAI의 text-embedding 모델
- GeminiEmbedding: Google Gemini 임베딩
- OllamaEmbedding: Ollama 로컬 서버 임베딩
- VoyageEmbedding: Voyage AI v3 시리즈
- JinaEmbedding: Jina AI v3 시리즈 (89개 언어)
- MistralEmbedding: Mistral AI 임베딩
- CohereEmbedding: Cohere 임베딩

Template Method Pattern을 사용하여 중복 코드 제거
"""

import os
from typing import List, Optional

from .base import BaseAPIEmbedding

try:
    from beanllm.utils.logger import get_logger
except ImportError:
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger(__name__)


class OpenAIEmbedding(BaseAPIEmbedding):
    """
    OpenAI Embeddings (Template Method Pattern 적용)

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

        # Import 검증
        self._validate_import("openai", "openai")

        from openai import AsyncOpenAI, OpenAI

        # API 키 가져오기
        self.api_key = self._get_api_key(api_key, ["OPENAI_API_KEY"], "OpenAI")

        # 클라이언트 초기화
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (비동기, OpenAI는 진정한 async 지원)"""
        try:
            response = await self.async_client.embeddings.create(
                input=texts, model=self.model, **self.kwargs
            )

            embeddings = [item.embedding for item in response.data]
            self._log_embed_success(len(texts), f"usage: {response.usage.total_tokens} tokens")

            return embeddings

        except Exception as e:
            self._handle_embed_error("OpenAI", e)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.sync_client.embeddings.create(
                input=texts, model=self.model, **self.kwargs
            )

            embeddings = [item.embedding for item in response.data]
            self._log_embed_success(len(texts), f"usage: {response.usage.total_tokens} tokens")

            return embeddings

        except Exception as e:
            self._handle_embed_error("OpenAI", e)


class GeminiEmbedding(BaseAPIEmbedding):
    """
    Google Gemini Embeddings (Template Method Pattern 적용)

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

        # Import 검증
        self._validate_import("google.generativeai", "beanllm", "gemini")

        import google.generativeai as genai

        # API 키 가져오기 (GOOGLE_API_KEY 또는 GEMINI_API_KEY)
        self.api_key = self._get_api_key(
            api_key, ["GOOGLE_API_KEY", "GEMINI_API_KEY"], "Google Gemini"
        )

        # 클라이언트 초기화
        genai.configure(api_key=self.api_key)
        self.genai = genai

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (동기, 배치 처리)

        Performance Optimization:
            - Uses batch API when possible (multiple texts in single request)
            - Fallback to sequential processing if batch fails
            - Reduces API calls significantly (n calls → 1 call for batch)

        Mathematical Foundation:
            Batch embedding reduces API overhead:
            - Sequential: O(n) API calls, O(n × latency) time
            - Batch: O(1) API call, O(latency + n × processing) time

            Where latency >> processing, batch is much faster.
        """
        try:
            embeddings = []

            # Try batch embedding first (Gemini API supports batch embed_content)
            try:
                # Batch API: send all texts in one request
                result = self.genai.embed_content(
                    model=self.model, content=texts, **self.kwargs
                )

                # Extract embeddings from batch response
                if isinstance(result, dict) and "embedding" in result:
                    embeddings = [result["embedding"]]
                elif isinstance(result, dict) and "embeddings" in result:
                    embeddings = result["embeddings"]
                elif isinstance(result, list):
                    embeddings = result
                else:
                    raise ValueError("Unexpected batch response format")

                self._log_embed_success(len(texts), "batch mode, 1 API call")

            except (ValueError, TypeError, KeyError) as batch_error:
                # Batch failed - fallback to sequential processing
                logger.warning(f"Batch embedding failed ({batch_error}), falling back to sequential mode")

                embeddings = []
                for text in texts:
                    result = self.genai.embed_content(
                        model=self.model, content=text, **self.kwargs
                    )
                    embeddings.append(result["embedding"])

                self._log_embed_success(len(texts), f"sequential mode, {len(texts)} API calls")

            return embeddings

        except Exception as e:
            self._handle_embed_error("Gemini", e)


class OllamaEmbedding(BaseAPIEmbedding):
    """
    Ollama Embeddings (로컬, Template Method Pattern 적용)

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

        # Import 검증
        self._validate_import("ollama", "beanllm", "ollama")

        import ollama

        # 클라이언트 초기화
        self.client = ollama.Client(host=base_url)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트들을 임베딩 (동기, 배치 처리 최적화)

        Performance Optimization:
            - Uses batch processing for multiple texts
            - Reduces network overhead and server processing time
            - Ollama server processes batch more efficiently than sequential

        Mathematical Foundation:
            Batch processing efficiency:
            - Sequential: n × (network + processing) time
            - Batch: network + batch_processing time

            Where batch_processing << n × processing due to:
            1. Shared model loading (load once, use n times)
            2. Vectorized operations on GPU
            3. Reduced context switching
        """
        try:
            embeddings = []

            # Try batch embedding (Ollama supports batch since v0.1.17+)
            try:
                # Modern Ollama API: batch embed via 'embed' method
                if hasattr(self.client, "embed"):
                    response = self.client.embed(model=self.model, input=texts)

                    # Extract embeddings from response
                    if isinstance(response, dict) and "embeddings" in response:
                        embeddings = response["embeddings"]
                    elif isinstance(response, list):
                        embeddings = response
                    else:
                        raise ValueError("Unexpected batch response format")

                    self._log_embed_success(len(texts), "batch mode, 1 request")

                else:
                    raise AttributeError("Batch API not available")

            except (AttributeError, ValueError, KeyError, TypeError) as batch_error:
                # Batch failed - fallback to sequential processing
                logger.warning(f"Batch embedding failed ({batch_error}), falling back to sequential mode")

                embeddings = []
                for text in texts:
                    response = self.client.embeddings(model=self.model, prompt=text)
                    embeddings.append(response["embedding"])

                self._log_embed_success(len(texts), f"sequential mode, {len(texts)} requests")

            return embeddings

        except Exception as e:
            self._handle_embed_error("Ollama", e)


class VoyageEmbedding(BaseAPIEmbedding):
    """
    Voyage AI Embeddings (v3 시리즈, 2024-2025, Template Method Pattern 적용)

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

        # Import 검증
        self._validate_import("voyageai", "voyageai")

        import voyageai

        # API 키 가져오기
        self.api_key = self._get_api_key(api_key, ["VOYAGE_API_KEY"], "Voyage AI")

        # 클라이언트 초기화
        self.client = voyageai.Client(api_key=self.api_key)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embed(texts=texts, model=self.model, **self.kwargs)

            self._log_embed_success(len(texts))
            return response.embeddings

        except Exception as e:
            self._handle_embed_error("Voyage AI", e)


class JinaEmbedding(BaseAPIEmbedding):
    """
    Jina AI Embeddings (v3 시리즈, 2024-2025, Template Method Pattern 적용)

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

        # API 키 가져오기
        self.api_key = self._get_api_key(api_key, ["JINA_API_KEY"], "Jina AI")

        # API URL
        self.url = "https://api.jina.ai/v1/embeddings"

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {"model": self.model, "input": texts, **self.kwargs}

            response = httpx.post(self.url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            self._log_embed_success(len(texts))
            return embeddings

        except Exception as e:
            self._handle_embed_error("Jina AI", e)


class MistralEmbedding(BaseAPIEmbedding):
    """
    Mistral AI Embeddings (Template Method Pattern 적용)

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

        # Import 검증
        self._validate_import("mistralai.client", "mistralai")

        from mistralai.client import MistralClient

        # API 키 가져오기
        self.api_key = self._get_api_key(api_key, ["MISTRAL_API_KEY"], "Mistral AI")

        # 클라이언트 초기화
        self.client = MistralClient(api_key=self.api_key)

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embeddings(model=self.model, input=texts)

            embeddings = [item.embedding for item in response.data]
            self._log_embed_success(len(texts))
            return embeddings

        except Exception as e:
            self._handle_embed_error("Mistral AI", e)


class CohereEmbedding(BaseAPIEmbedding):
    """
    Cohere Embeddings (Template Method Pattern 적용)

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

        # Import 검증
        self._validate_import("cohere", "cohere")

        import cohere

        # API 키 가져오기
        self.api_key = self._get_api_key(api_key, ["COHERE_API_KEY"], "Cohere")

        # 클라이언트 초기화
        self.client = cohere.Client(api_key=self.api_key)
        self.input_type = input_type

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """텍스트들을 임베딩 (동기)"""
        try:
            response = self.client.embed(
                texts=texts, model=self.model, input_type=self.input_type, **self.kwargs
            )

            self._log_embed_success(len(texts))
            return response.embeddings

        except Exception as e:
            self._handle_embed_error("Cohere", e)
