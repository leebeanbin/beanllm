"""
Facade - 기존 API를 위한 Facade 패턴
책임: 하위 호환성 유지, 내부적으로는 Service 사용
SOLID 원칙:
- Facade 패턴: 복잡한 내부 구조를 단순한 인터페이스로
"""

import importlib

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # Base class
    "FacadeBase": ("beanllm.facade.base", "FacadeBase"),
    # Core facades
    "Client": ("beanllm.facade.core.client_facade", "Client"),
    "RAGChain": ("beanllm.facade.core.rag_facade", "RAGChain"),
    "RAG": ("beanllm.facade.core.rag_facade", "RAG"),
    "RAGBuilder": ("beanllm.facade.core.rag_facade", "RAGBuilder"),
    "create_rag": ("beanllm.facade.core.rag_facade", "create_rag"),
    "Agent": ("beanllm.facade.core.agent_facade", "Agent"),
    "Chain": ("beanllm.facade.core.chain_facade", "Chain"),
    "ChainBuilder": ("beanllm.facade.core.chain_facade", "ChainBuilder"),
    "ChainResult": ("beanllm.facade.core.chain_facade", "ChainResult"),
    "ParallelChain": ("beanllm.facade.core.chain_facade", "ParallelChain"),
    "PromptChain": ("beanllm.facade.core.chain_facade", "PromptChain"),
    "SequentialChain": ("beanllm.facade.core.chain_facade", "SequentialChain"),
    "create_chain": ("beanllm.facade.core.chain_facade", "create_chain"),
    # Advanced facades
    "Graph": ("beanllm.facade.advanced.graph_facade", "Graph"),
    "KnowledgeGraph": ("beanllm.facade.advanced.knowledge_graph_facade", "KnowledgeGraph"),
    "StateGraph": ("beanllm.facade.advanced.state_graph_facade", "StateGraph"),
    "MultiAgent": ("beanllm.facade.advanced.multi_agent_facade", "MultiAgentCoordinator"),
    "RAGDebug": ("beanllm.facade.advanced.rag_debug_facade", "RAGDebug"),
    "Orchestrator": ("beanllm.facade.advanced.orchestrator_facade", "Orchestrator"),
    "Optimizer": ("beanllm.facade.advanced.optimizer_facade", "Optimizer"),
    # ML facades
    "WhisperSTT": ("beanllm.facade.ml.audio_facade", "WhisperSTT"),
    "TTS": ("beanllm.facade.ml.audio_facade", "TextToSpeech"),
    "AudioRAG": ("beanllm.facade.ml.audio_facade", "AudioRAG"),
    "VisionRAG": ("beanllm.facade.ml.vision_rag_facade", "VisionRAG"),
    "MultimodalRAG": ("beanllm.facade.ml.vision_rag_facade", "MultimodalRAG"),
    "create_vision_rag": ("beanllm.facade.ml.vision_rag_facade", "create_vision_rag"),
    "EvaluatorFacade": ("beanllm.facade.ml.evaluation_facade", "EvaluatorFacade"),
    "FineTuningManagerFacade": ("beanllm.facade.ml.finetuning_facade", "FineTuningManagerFacade"),
    "WebSearch": ("beanllm.facade.ml.web_search_facade", "WebSearch"),
}

__all__ = [
    # Base class
    "FacadeBase",
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


def __getattr__(name: str):
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr = _LAZY_IMPORT_MAP[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
