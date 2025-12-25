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
