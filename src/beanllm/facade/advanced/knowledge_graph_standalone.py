"""
Standalone Knowledge Graph functions - quick_knowledge_graph, quick_graph_rag.

Module-level one-liners that create a KnowledgeGraph instance and run a single operation.
"""

from __future__ import annotations

from typing import Any, Optional

from beanllm.dto.response.graph.kg_response import BuildGraphResponse


async def quick_knowledge_graph(
    documents: list[str],
    client: Optional[Any] = None,
) -> BuildGraphResponse:
    """
    빠른 Knowledge Graph 구축 (standalone function)

    Args:
        documents: 문서 목록
        client: LLM Client (optional)

    Returns:
        BuildGraphResponse: 그래프 정보

    Example:
        ```python
        from beanllm import quick_knowledge_graph

        response = await quick_knowledge_graph(
            documents=["Apple was founded by Steve Jobs in 1976."]
        )

        print(f"Graph ID: {response.graph_id}")
        print(f"Nodes: {response.num_nodes}, Edges: {response.num_edges}")
        ```
    """
    from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph

    kg = KnowledgeGraph(client=client)
    return await kg.quick_build(documents=documents)


async def quick_graph_rag(
    query: str,
    graph_id: str,
    client: Optional[Any] = None,
) -> str:
    """
    빠른 Graph RAG 질의 (standalone function)

    Args:
        query: 사용자 질의
        graph_id: 그래프 ID
        client: LLM Client (optional)

    Returns:
        str: 답변

    Example:
        ```python
        from beanllm import quick_graph_rag

        answer = await quick_graph_rag(
            query="Who founded Apple?",
            graph_id="tech_companies"
        )
        print(answer)
        ```
    """
    from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph

    kg = KnowledgeGraph(client=client)
    return await kg.ask(query=query, graph_id=graph_id)
