"""Service Implementations - 서비스 구현체"""

# Backward compatibility: Re-export from subdirectories
from .core.agent_service_impl import AgentServiceImpl
from .core.chat_service_impl import ChatServiceImpl
from .core.rag_service_impl import RAGServiceImpl
from .core.chain_service_impl import ChainServiceImpl

# Advanced services (선택적 의존성)
try:
    from .advanced.graph_service_impl import GraphServiceImpl
except ImportError:
    GraphServiceImpl = None  # type: ignore

try:
    from .advanced.knowledge_graph_service_impl import KnowledgeGraphServiceImpl
except ImportError:
    KnowledgeGraphServiceImpl = None  # type: ignore

try:
    from .advanced.multi_agent_service_impl import MultiAgentServiceImpl
except ImportError:
    MultiAgentServiceImpl = None  # type: ignore

try:
    from .advanced.optimizer_service_impl import OptimizerServiceImpl
except ImportError:
    OptimizerServiceImpl = None  # type: ignore

try:
    from .advanced.orchestrator_service_impl import OrchestratorServiceImpl
except ImportError:
    OrchestratorServiceImpl = None  # type: ignore

try:
    from .advanced.rag_debug_service_impl import RAGDebugServiceImpl
except ImportError:
    RAGDebugServiceImpl = None  # type: ignore

try:
    from .advanced.state_graph_service_impl import StateGraphServiceImpl
except ImportError:
    StateGraphServiceImpl = None  # type: ignore

# ML services (선택적 의존성)
try:
    from .ml.audio_service_impl import AudioServiceImpl
except ImportError:
    AudioServiceImpl = None  # type: ignore

try:
    from .ml.evaluation_service_impl import EvaluationServiceImpl
except ImportError:
    EvaluationServiceImpl = None  # type: ignore

try:
    from .ml.finetuning_service_impl import FinetuningServiceImpl
except ImportError:
    FinetuningServiceImpl = None  # type: ignore

try:
    from .ml.vision_rag_service_impl import VisionRAGServiceImpl
except ImportError:
    VisionRAGServiceImpl = None  # type: ignore

try:
    from .ml.web_search_service_impl import WebSearchServiceImpl
except ImportError:
    WebSearchServiceImpl = None  # type: ignore

__all__ = [
    # Core services (backward compatibility)
    "ChatServiceImpl",
    "RAGServiceImpl",
    "AgentServiceImpl",
    "ChainServiceImpl",
    # Advanced services
    "GraphServiceImpl",
    "KnowledgeGraphServiceImpl",
    "StateGraphServiceImpl",
    "MultiAgentServiceImpl",
    "RAGDebugServiceImpl",
    "OrchestratorServiceImpl",
    "OptimizerServiceImpl",
    # ML services
    "AudioServiceImpl",
    "VisionRAGServiceImpl",
    "EvaluationServiceImpl",
    "FinetuningServiceImpl",
    "WebSearchServiceImpl",
]
