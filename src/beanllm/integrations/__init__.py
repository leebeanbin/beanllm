"""
Integrations - 외부 프레임워크 통합

beanLLM과 외부 LLM 프레임워크를 통합합니다.
"""

# LlamaIndex 통합
try:
    from .llamaindex import (
        LlamaIndexBridge,
        LlamaIndexQueryEngine,
        create_llamaindex_query_engine,
    )
except ImportError:
    LlamaIndexBridge = None  # type: ignore
    LlamaIndexQueryEngine = None  # type: ignore
    create_llamaindex_query_engine = None  # type: ignore

# LangGraph 통합
try:
    from .langgraph import (
        LangGraphBridge,
        LangGraphWorkflow,
        WorkflowBuilder,
        create_workflow,
    )
except ImportError:
    LangGraphBridge = None  # type: ignore
    LangGraphWorkflow = None  # type: ignore
    WorkflowBuilder = None  # type: ignore
    create_workflow = None  # type: ignore

__all__ = [
    # LlamaIndex
    "LlamaIndexBridge",
    "LlamaIndexQueryEngine",
    "create_llamaindex_query_engine",
    # LangGraph
    "LangGraphBridge",
    "LangGraphWorkflow",
    "WorkflowBuilder",
    "create_workflow",
]
