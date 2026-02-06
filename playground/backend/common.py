"""
Common Utilities & State Management

Shared state, helper functions, and initialization logic.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from beanllm.facade.advanced import (
    KnowledgeGraph,
    MultiAgent,
    Optimizer,
    Orchestrator,
    RAGDebug,
)
from beanllm.facade.core import Client, RAGChain
from beanllm.facade.ml import (
    AudioRAG,
    EvaluatorFacade,
    FineTuningManagerFacade,
    VisionRAG,
    WebSearch,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Global State (Lazy Initialization)
# ============================================================================

_client: Optional[Client] = None
_kg: Optional[KnowledgeGraph] = None
_rag_chains: Dict[str, RAGChain] = {}  # collection_name -> RAGChain
_web_search: Optional[WebSearch] = None
_rag_debugger: Optional[RAGDebug] = None
_optimizer: Optional[Optimizer] = None
_multi_agent: Optional[MultiAgent] = None
_orchestrator: Optional[Orchestrator] = None
_vision_rag: Optional[VisionRAG] = None
_audio_rag: Optional[AudioRAG] = None
_evaluator: Optional[EvaluatorFacade] = None
_finetuning: Optional[FineTuningManagerFacade] = None

# Downloaded models tracking
_downloaded_models: Dict[str, str] = {}


# ============================================================================
# Helper Functions
# ============================================================================


def get_client(model: str = "qwen2.5:0.5b") -> Client:
    """Get or create Client instance"""
    global _client
    if _client is None:
        _client = Client(model=model)
        logger.info(f"✅ Client initialized with model: {model}")
    return _client


def get_kg(graph_name: str = "default") -> KnowledgeGraph:
    """Get or create KnowledgeGraph instance"""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph(graph_name=graph_name)
        logger.info(f"✅ KnowledgeGraph initialized: {graph_name}")
    return _kg


def get_rag_chain(collection_name: str) -> Optional[RAGChain]:
    """Get existing RAG chain"""
    return _rag_chains.get(collection_name)


def set_rag_chain(collection_name: str, rag: RAGChain):
    """Store RAG chain"""
    _rag_chains[collection_name] = rag
    logger.info(f"✅ RAG chain stored: {collection_name}")


def get_web_search() -> WebSearch:
    """Get or create WebSearch instance"""
    global _web_search
    if _web_search is None:
        _web_search = WebSearch()
        logger.info("✅ WebSearch initialized")
    return _web_search


def get_rag_debugger() -> RAGDebug:
    """Get or create RAGDebug instance"""
    global _rag_debugger
    if _rag_debugger is None:
        _rag_debugger = RAGDebug()
        logger.info("✅ RAGDebug initialized")
    return _rag_debugger


def get_optimizer() -> Optimizer:
    """Get or create Optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = Optimizer()
        logger.info("✅ Optimizer initialized")
    return _optimizer


def get_multi_agent() -> MultiAgent:
    """Get or create MultiAgent instance"""
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = MultiAgent()
        logger.info("✅ MultiAgent initialized")
    return _multi_agent


def get_orchestrator() -> Orchestrator:
    """Get or create Orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
        logger.info("✅ Orchestrator initialized")
    return _orchestrator


def get_vision_rag() -> VisionRAG:
    """Get or create VisionRAG instance"""
    global _vision_rag
    if _vision_rag is None:
        _vision_rag = VisionRAG()
        logger.info("✅ VisionRAG initialized")
    return _vision_rag


def get_audio_rag() -> AudioRAG:
    """Get or create AudioRAG instance"""
    global _audio_rag
    if _audio_rag is None:
        _audio_rag = AudioRAG()
        logger.info("✅ AudioRAG initialized")
    return _audio_rag


def get_evaluator() -> EvaluatorFacade:
    """Get or create EvaluatorFacade instance"""
    global _evaluator
    if _evaluator is None:
        _evaluator = EvaluatorFacade()
        logger.info("✅ Evaluator initialized")
    return _evaluator


def get_finetuning() -> FineTuningManagerFacade:
    """Get or create FineTuningManagerFacade instance"""
    global _finetuning
    if _finetuning is None:
        _finetuning = FineTuningManagerFacade()
        logger.info("✅ FineTuningManager initialized")
    return _finetuning


# ============================================================================
# Model Name Mapping
# ============================================================================


def get_ollama_model_name_for_chat(model_name: str) -> str:
    """
    Map display model name to actual Ollama model name for chat.

    Examples:
        "phi3.5" -> "phi3"
        "qwen2.5:0.5b" -> "qwen2.5:0.5b" (unchanged)
    """
    # Specific mappings
    mapping = {
        "phi3.5": "phi3",
        "phi3.5:latest": "phi3:latest",
    }

    if model_name in mapping:
        return mapping[model_name]

    # If has colon (tag specified), use as-is
    if ":" in model_name:
        return model_name

    # Default: add :latest tag
    return f"{model_name}:latest"


def track_downloaded_model(display_name: str, ollama_name: str):
    """Track downloaded model mapping"""
    _downloaded_models[display_name] = ollama_name
    logger.info(f"✅ Tracked model: {display_name} -> {ollama_name}")


def get_downloaded_models() -> Dict[str, str]:
    """Get downloaded models mapping"""
    return _downloaded_models.copy()
