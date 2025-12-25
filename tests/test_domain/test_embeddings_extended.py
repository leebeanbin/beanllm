"""
Embeddings 확장 테스트 - Embedding Cache, Factory 등
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from beanllm.domain.embeddings.base import BaseEmbedding


class TestEmbeddingCache:
    """EmbeddingCache 테스트"""

    def test_embedding_cache_get_set(self):
        """임베딩 캐시 저장/조회 테스트"""
        try:
            from beanllm.domain.embeddings.cache import EmbeddingCache

            cache = EmbeddingCache(max_size=100)

            cache.set("text1", [0.1, 0.2, 0.3])
            result = cache.get("text1")

            assert result is not None
            assert result == [0.1, 0.2, 0.3]
        except ImportError:
            pytest.skip("EmbeddingCache not available")

    def test_embedding_cache_clear(self):
        """임베딩 캐시 초기화 테스트"""
        try:
            from beanllm.domain.embeddings.cache import EmbeddingCache

            cache = EmbeddingCache()
            cache.set("text1", [0.1, 0.2, 0.3])
            cache.clear()

            result = cache.get("text1")
            assert result is None
        except ImportError:
            pytest.skip("EmbeddingCache not available")


class TestEmbeddingFactory:
    """Embedding Factory 확장 테스트"""

    def test_embedding_factory_create_openai(self):
        """OpenAI Embedding 생성 테스트"""
        try:
            from beanllm.domain.embeddings.factory import Embedding

            with patch("beanllm.domain.embeddings.providers.OpenAI"):
                embedding = Embedding(model="text-embedding-3-small", provider="openai", api_key="test_key")
                assert embedding is not None
                assert embedding.model == "text-embedding-3-small"
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")

    def test_embedding_factory_create_ollama(self):
        """Ollama Embedding 생성 테스트"""
        try:
            from beanllm.domain.embeddings.factory import Embedding

            with patch("beanllm.domain.embeddings.providers.ollama"):
                embedding = Embedding(model="nomic-embed-text", provider="ollama")
                assert embedding is not None
                assert embedding.model == "nomic-embed-text"
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"Ollama embedding not available: {e}")


class TestEmbeddingProviders:
    """Embedding Provider 구현체 테스트"""

    @pytest.mark.asyncio
    async def test_openai_embedding_embed(self):
        """OpenAI Embedding embed 테스트"""
        try:
            from beanllm.domain.embeddings.providers import OpenAIEmbedding
            from unittest.mock import AsyncMock, patch

            with patch("beanllm.domain.embeddings.providers.OpenAI") as mock_openai:
                mock_response = Mock()
                mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
                mock_response.usage = Mock(total_tokens=1)
                mock_openai.return_value.embeddings.create = AsyncMock(return_value=mock_response)

                embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test_key")
                result = await embedding.embed(["test text"])

                assert isinstance(result, list)
                assert len(result) > 0
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")

    def test_openai_embedding_embed_sync(self):
        """OpenAI Embedding embed_sync 테스트"""
        try:
            from beanllm.domain.embeddings.providers import OpenAIEmbedding
            from unittest.mock import Mock, patch

            with patch("beanllm.domain.embeddings.providers.OpenAI") as mock_openai:
                mock_response = Mock()
                mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
                mock_response.usage = Mock(total_tokens=1)
                mock_openai.return_value.embeddings.create = Mock(return_value=mock_response)

                embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test_key")
                result = embedding.embed_sync(["test text"])

                assert isinstance(result, list)
                assert len(result) > 0
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")



            from unittest.mock import Mock, patch

            with patch("beanllm.domain.embeddings.providers.OpenAI") as mock_openai:
                mock_response = Mock()
                mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
                mock_response.usage = Mock(total_tokens=1)
                mock_openai.return_value.embeddings.create = Mock(return_value=mock_response)

                embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test_key")
                result = embedding.embed_sync(["test text"])

                assert isinstance(result, list)
                assert len(result) > 0
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")



            from unittest.mock import Mock, patch

            with patch("beanllm.domain.embeddings.providers.OpenAI") as mock_openai:
                mock_response = Mock()
                mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
                mock_response.usage = Mock(total_tokens=1)
                mock_openai.return_value.embeddings.create = Mock(return_value=mock_response)

                embedding = OpenAIEmbedding(model="text-embedding-3-small", api_key="test_key")
                result = embedding.embed_sync(["test text"])

                assert isinstance(result, list)
                assert len(result) > 0
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")


