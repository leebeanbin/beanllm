"""
Retrieval Domain 테스트 - 검색 타입, Query Expansion, Hybrid Search
"""

from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beanllm.domain.retrieval.types import RerankResult, SearchResult

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class TestRerankResult:
    def test_create_rerank_result(self) -> None:
        result = RerankResult(text="Hello world", score=0.95, index=0)
        assert result.text == "Hello world"
        assert result.score == 0.95
        assert result.index == 0

    def test_rerank_result_with_metadata(self) -> None:
        result = RerankResult(
            text="Document content",
            score=0.8,
            index=2,
            metadata={"source": "test.txt"},
        )
        assert result.metadata["source"] == "test.txt"

    def test_rerank_result_repr(self) -> None:
        result = RerankResult(text="Short text", score=0.7, index=1)
        r = repr(result)
        assert "0.7000" in r or "0.70" in r
        assert "1" in r

    def test_rerank_result_no_metadata(self) -> None:
        result = RerankResult(text="text", score=0.5, index=0)
        assert result.metadata is None


class TestSearchResult:
    def test_create_search_result(self) -> None:
        result = SearchResult(text="Found document", score=0.9)
        assert result.text == "Found document"
        assert result.score == 0.9

    def test_search_result_with_metadata(self) -> None:
        result = SearchResult(
            text="Doc text",
            score=0.85,
            metadata={"page": 1, "source": "manual.pdf"},
        )
        assert result.metadata["page"] == 1

    def test_search_result_no_metadata(self) -> None:
        result = SearchResult(text="text", score=0.7)
        assert result.metadata is None


@pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not available")
class TestHybridRetriever:
    @pytest.fixture
    def embedding_function(self):
        def embed(text: str) -> List[float]:
            return [0.1, 0.2, 0.3, 0.4, 0.5]

        return embed

    @pytest.fixture
    def documents(self):
        return [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks.",
            "Natural language processing analyzes text.",
            "Computer vision processes images.",
            "Reinforcement learning uses rewards.",
        ]

    def test_hybrid_retriever_initialization(self, documents, embedding_function) -> None:
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        retriever = HybridRetriever(
            documents=documents,
            embedding_function=embedding_function,
        )
        assert len(retriever.documents) == 5

    def test_hybrid_search_returns_results(self, documents, embedding_function) -> None:
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        retriever = HybridRetriever(
            documents=documents,
            embedding_function=embedding_function,
            fusion_method="rrf",
        )
        results = retriever.search("machine learning AI", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_hybrid_search_result_type(self, documents, embedding_function) -> None:
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        retriever = HybridRetriever(
            documents=documents,
            embedding_function=embedding_function,
        )
        results = retriever.search("neural networks", top_k=2)
        for r in results:
            assert isinstance(r, SearchResult)
            assert r.score >= 0

    def test_hybrid_search_fusion_methods(self, documents, embedding_function) -> None:
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        for method in ["rrf", "weighted_sum", "distribution_based"]:
            retriever = HybridRetriever(
                documents=documents,
                embedding_function=embedding_function,
                fusion_method=method,
            )
            results = retriever.search("AI", top_k=3)
            assert isinstance(results, list)

    def test_hybrid_retriever_invalid_fusion_raises(self, documents, embedding_function) -> None:
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        with pytest.raises(ValueError):
            HybridRetriever(
                documents=documents,
                embedding_function=embedding_function,
                fusion_method="invalid_method",
            )

    def test_hybrid_search_top_k_limit(self, documents, embedding_function) -> None:
        from beanllm.domain.retrieval.hybrid_search import HybridRetriever

        retriever = HybridRetriever(
            documents=documents,
            embedding_function=embedding_function,
        )
        results = retriever.search("AI", top_k=2)
        assert len(results) <= 2


class TestQueryExpansion:
    def test_multi_query_expander_with_mock_llm(self) -> None:
        from beanllm.domain.retrieval.query_expansion import MultiQueryExpander

        mock_llm = MagicMock()
        mock_llm.return_value = (
            "1. What are the core concepts of machine learning?\n"
            "2. How do machine learning algorithms learn from data?\n"
            "3. What is the difference between supervised and unsupervised learning?"
        )

        expander = MultiQueryExpander(llm_function=mock_llm, num_queries=3)
        results = expander.expand("What is machine learning?")

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_step_back_expander_with_mock_llm(self) -> None:
        from beanllm.domain.retrieval.query_expansion import StepBackExpander

        mock_llm = MagicMock()
        mock_llm.return_value = "What are the fundamental principles of AI?"

        expander = StepBackExpander(llm_function=mock_llm)
        result = expander.expand("What is the accuracy of GPT-4?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_hyde_expander_with_mock_client(self) -> None:
        from beanllm.domain.retrieval.query_expansion import HyDEExpander

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Machine learning is a method of data analysis that automates..."

        # HyDEExpander calls client.chat() which is async, so provide sync callable
        mock_client.chat_sync = MagicMock(return_value=mock_response)

        # Use a direct llm_function approach since client API may differ
        def mock_llm(prompt: str) -> str:
            return "Machine learning is a method of data analysis that automates..."

        expander = HyDEExpander(llm_function=mock_llm)
        result = expander.expand("What is machine learning?")

        assert isinstance(result, (str, list))


@pytest.mark.skipif(not CROSS_ENCODER_AVAILABLE, reason="sentence-transformers not available")
class TestCrossEncoderReranker:
    def test_reranker_initialization(self) -> None:
        from beanllm.domain.retrieval.reranker_cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            use_gpu=False,
        )
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_rerank_documents(self) -> None:
        from beanllm.domain.retrieval.reranker_cross_encoder import CrossEncoderReranker

        reranker = CrossEncoderReranker(use_gpu=False)
        documents = [
            "Python is a programming language.",
            "Deep learning uses neural networks.",
            "Python was created by Guido van Rossum.",
        ]
        results = reranker.rerank("Python programming language", documents, top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2
        for r in results:
            assert isinstance(r, RerankResult)


class TestPositionEngineeringReranker:
    def test_position_reranker_initialization(self) -> None:
        from beanllm.domain.retrieval.reranker_position import PositionEngineeringReranker

        reranker = PositionEngineeringReranker()
        assert reranker is not None

    def test_position_rerank_returns_results(self) -> None:
        from beanllm.domain.retrieval.reranker_position import PositionEngineeringReranker

        reranker = PositionEngineeringReranker()
        documents = ["doc 1", "doc 2", "doc 3"]
        results = reranker.rerank("query", documents, top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
