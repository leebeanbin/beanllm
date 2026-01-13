"""
Facade - 기존 API를 위한 Facade 패턴
책임: 하위 호환성 유지, 내부적으로는 Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

# Backward compatibility: Re-export from subdirectories
from .core.agent_facade import Agent
from .core.chain_facade import (
    Chain,
    ChainBuilder,
    ChainResult,
    ParallelChain,
    PromptChain,
    SequentialChain,
    create_chain,
)
from .core.client_facade import Client
from .core.rag_facade import RAG, RAGBuilder, RAGChain, create_rag

# Advanced facades
from .advanced.graph_facade import Graph
from .advanced.knowledge_graph_facade import KnowledgeGraph
from .advanced.multi_agent_facade import MultiAgentCoordinator as MultiAgent
from .advanced.optimizer_facade import Optimizer
from .advanced.orchestrator_facade import Orchestrator
from .advanced.rag_debug_facade import RAGDebug
from .advanced.state_graph_facade import StateGraph

# ML facades
from .ml.audio_facade import WhisperSTT, TextToSpeech as TTS, AudioRAG
from .ml.evaluation_facade import EvaluatorFacade
from .ml.finetuning_facade import FineTuningManagerFacade
from .ml.vision_rag_facade import VisionRAG, MultimodalRAG, create_vision_rag
from .ml.web_search_facade import WebSearch

__all__ = [
    # Core facades (backward compatibility)
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
    # Advanced facades
    "Graph",
    "KnowledgeGraph",
    "StateGraph",
    "MultiAgent",
    "RAGDebug",
    "Orchestrator",
    "Optimizer",
    # ML facades
    "WhisperSTT",
    "TTS",
    "AudioRAG",
    "VisionRAG",
    "MultimodalRAG",
    "create_vision_rag",
    "EvaluatorFacade",
    "FineTuningManagerFacade",
    "WebSearch",
]
