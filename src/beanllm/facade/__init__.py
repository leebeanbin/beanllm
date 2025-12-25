"""
Facade - 기존 API를 위한 Facade 패턴
책임: 하위 호환성 유지, 내부적으로는 Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

from .agent_facade import Agent
from .chain_facade import (
    Chain,
    ChainBuilder,
    ChainResult,
    ParallelChain,
    PromptChain,
    SequentialChain,
    create_chain,
)
from .client_facade import Client
from .rag_facade import RAG, RAGBuilder, RAGChain, create_rag

__all__ = [
    "Client",
    "RAGChain",
    "RAG",
    "RAGBuilder",
    "create_rag",
    "Agent",
    "Chain",
    "ChainBuilder",
    "ChainResult",
    "ParallelChain",
    "PromptChain",
    "SequentialChain",
    "create_chain",
]
