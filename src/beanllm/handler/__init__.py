"""Handlers - Controller 역할 (모든 if-else/try-catch 처리)"""

# Advanced handlers
from .advanced.graph_handler import GraphHandler
from .advanced.knowledge_graph_handler import KnowledgeGraphHandler
from .advanced.multi_agent_handler import MultiAgentHandler
from .advanced.optimizer_handler import OptimizerHandler
from .advanced.orchestrator_handler import OrchestratorHandler
from .advanced.rag_debug_handler import RAGDebugHandler
from .advanced.state_graph_handler import StateGraphHandler
from .base_handler import BaseHandler

# Backward compatibility: Re-export from subdirectories
from .core.agent_handler import AgentHandler
from .core.chain_handler import ChainHandler
from .core.chat_handler import ChatHandler
from .core.rag_handler import RAGHandler

# ML handlers
from .ml.audio_handler import AudioHandler
from .ml.evaluation_handler import EvaluationHandler
from .ml.finetuning_handler import FinetuningHandler
from .ml.vision_rag_handler import VisionRAGHandler
from .ml.web_search_handler import WebSearchHandler

__all__ = [
    "BaseHandler",
    # Core handlers (backward compatibility)
    "ChatHandler",
    "RAGHandler",
    "AgentHandler",
    "ChainHandler",
    # Advanced handlers
    "GraphHandler",
    "KnowledgeGraphHandler",
    "StateGraphHandler",
    "MultiAgentHandler",
    "RAGDebugHandler",
    "OrchestratorHandler",
    "OptimizerHandler",
    # ML handlers
    "AudioHandler",
    "VisionRAGHandler",
    "WebSearchHandler",
    "EvaluationHandler",
    "FinetuningHandler",
]
