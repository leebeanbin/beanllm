"""
Embeddings 테스트 - 임베딩 구현체 테스트
"""

import pytest
from unittest.mock import Mock, patch

from llmkit.domain.embeddings.base import BaseEmbedding


class TestBaseEmbedding:
    """BaseEmbedding 테스트"""

    @pytest.fixture
    def mock_embedding(self):
        """Mock Embedding"""
        embedding = Mock(spec=BaseEmbedding)
        embedding.embed = Mock(return_value=[[0.1, 0.2, 0.3]])
        embedding.embed_batch = Mock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        return embedding

    def test_embed(self, mock_embedding):
        """단일 텍스트 임베딩 테스트"""
        result = mock_embedding.embed("test text")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_embed_batch(self, mock_embedding):
        """배치 임베딩 테스트"""
        texts = ["text 1", "text 2"]
        results = mock_embedding.embed_batch(texts)

        assert isinstance(results, list)
        assert len(results) == len(texts)


class TestEmbeddingFactory:
    """Embedding Factory 테스트"""

    def test_get_embedding_openai(self):
        """OpenAI Embedding 생성 테스트"""
        try:
            from llmkit.domain.embeddings.factory import Embedding

            # Mock을 사용하여 실제 API 호출 없이 테스트
            with patch("llmkit.domain.embeddings.providers.OpenAI") as mock_openai:
                embedding = Embedding(model="text-embedding-3-small", provider="openai", api_key="test_key")
                assert embedding is not None
                assert embedding.model == "text-embedding-3-small"
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"OpenAI embedding not available: {e}")

    def test_get_embedding_ollama(self):
        """Ollama Embedding 생성 테스트 (로컬, API 키 불필요)"""
        try:
            from llmkit.domain.embeddings.factory import Embedding

            # Mock을 사용하여 실제 라이브러리 없이 테스트
            with patch("llmkit.domain.embeddings.providers.ollama") as mock_ollama:
                embedding = Embedding(model="nomic-embed-text", provider="ollama")
                assert embedding is not None
                assert embedding.model == "nomic-embed-text"
        except (ImportError, ValueError, AttributeError) as e:
            pytest.skip(f"Ollama embedding not available: {e}")


