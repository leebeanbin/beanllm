"""
Retrieval Domain - 검색 및 재순위화 도메인

검색 전략:
- HybridRetriever: BM25 + Dense 하이브리드 검색 (30-50% 품질 향상)
- ColBERTRetriever: Multi-vector late interaction (10-30% 품질 향상)
- ColPaliRetriever: 비전 문서 검색 (OCR 불필요)

쿼리 확장:
- HyDEExpander: 가상 답변 생성 후 검색
- MultiQueryExpander: 다중 쿼리 변형
- StepBackExpander: 추상화된 쿼리 생성

재순위화:
- BGEReranker: BAAI의 크로스 인코더
- CohereReranker: Cohere Rerank API
- CrossEncoderReranker: 범용 크로스 인코더
- PositionEngineeringReranker: 위치 기반 재순위화

Example:
    ```python
    from beanllm.domain.retrieval import ColBERTRetriever, HybridRetriever

    # ColBERT 검색 (가장 정확)
    retriever = ColBERTRetriever(
        documents=["Doc 1", "Doc 2"],
        model="colbert-ir/colbertv2.0"
    )
    results = retriever.search("query", k=5)

    # 하이브리드 검색 (빠르고 효과적)
    hybrid = HybridRetriever(
        documents=["Doc 1", "Doc 2"],
        embedding_function=embed_fn,
        fusion_method="rrf"
    )
    ```
"""

from beanllm.domain.retrieval.base import BaseReranker
from beanllm.domain.retrieval.hybrid_search import HybridRetriever
from beanllm.domain.retrieval.query_expansion import (
    BaseQueryExpander,
    HyDEExpander,
    MultiQueryExpander,
    StepBackExpander,
)
from beanllm.domain.retrieval.rerankers import (
    BGEReranker,
    CohereReranker,
    CrossEncoderReranker,
    PositionEngineeringReranker,
)
from beanllm.domain.retrieval.types import RerankResult, SearchResult

# ColBERT Retrievers (의존성이 없을 수 있음)
try:
    from beanllm.domain.retrieval.colbert import ColBERTRetriever, ColPaliRetriever
except ImportError:
    ColBERTRetriever = None  # type: ignore
    ColPaliRetriever = None  # type: ignore

__all__ = [
    # Types
    "RerankResult",
    "SearchResult",
    # Base
    "BaseReranker",
    "BaseQueryExpander",
    # Rerankers
    "BGEReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "PositionEngineeringReranker",
    # Hybrid Search
    "HybridRetriever",
    # Query Expansion
    "HyDEExpander",
    "MultiQueryExpander",
    "StepBackExpander",
]

# ColBERT Retrievers 추가 (있는 경우)
if ColBERTRetriever is not None:
    __all__.extend(["ColBERTRetriever", "ColPaliRetriever"])
