"""
LlamaIndex Integration - LlamaIndex 통합 (2024-2025)

LlamaIndex는 LLM 애플리케이션을 위한 데이터 프레임워크입니다.

LlamaIndex 특징:
- Advanced RAG (Multi-step retrieval, Query transformation)
- 다양한 Index 타입 (VectorStoreIndex, TreeIndex, etc.)
- Query Engine (Response synthesis, Sub-question query)
- Agent & Tool 통합
- 200+ Data Connectors

beanLLM 통합:
- beanLLM Document → LlamaIndex Document 변환
- beanLLM Embeddings → LlamaIndex Embeddings 래핑
- Query Engine을 beanLLM 스타일로 제공

Requirements:
    pip install llama-index

References:
    - https://github.com/run-llama/llama_index
    - https://docs.llamaindex.ai/
"""

from .bridge import LlamaIndexBridge
from .query_engine import LlamaIndexQueryEngine, create_llamaindex_query_engine

__all__ = [
    "LlamaIndexBridge",
    "LlamaIndexQueryEngine",
    "create_llamaindex_query_engine",
]
