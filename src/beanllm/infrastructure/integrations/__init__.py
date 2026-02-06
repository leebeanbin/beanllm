"""
Integrations - 외부 프레임워크 통합

beanLLM과 외부 LLM 프레임워크를 통합합니다.

이 모듈은 Infrastructure 레이어의 일부로, 외부 시스템과의 통합을 담당합니다.

지원하는 통합:
- LangGraph: 복잡한 에이전트 워크플로우를 위한 그래프 기반 프레임워크
- LlamaIndex: LLM 애플리케이션을 위한 데이터 프레임워크

사용 예제:
    ```python
    # LangGraph 통합
    from beanllm.infrastructure.integrations import LangGraphBridge, create_workflow
    from beanllm.domain.state_graph import GraphState

    class MyState(GraphState):
        query: str
        answer: str

    # LangGraph 워크플로우 생성
    workflow = create_workflow(MyState)
    result = workflow.run({"query": "Hello"})

    # LlamaIndex 통합
    from beanllm.infrastructure.integrations import LlamaIndexBridge, create_llamaindex_query_engine
    from beanllm.domain.loaders import TextLoader
    from beanllm.domain.embeddings import OpenAIEmbedding

    # 문서 로드 및 Query Engine 생성
    loader = TextLoader("document.txt")
    docs = loader.load()
    embedding = OpenAIEmbedding()

    query_engine = create_llamaindex_query_engine(
        documents=docs,
        embedding_function=embedding.embed
    )
    response = query_engine.query("What is this about?")
    ```

Requirements:
    - LangGraph: pip install langgraph
    - LlamaIndex: pip install llama-index

참고:
    - 이 모듈은 선택적 의존성입니다. 필요한 경우에만 설치하면 됩니다.
    - 의존성이 없어도 beanLLM의 다른 기능은 정상적으로 작동합니다.
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
