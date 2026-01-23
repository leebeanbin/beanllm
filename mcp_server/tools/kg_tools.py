"""
Knowledge Graph Tools - 기존 beanllm KG 기능을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastmcp import FastMCP

# 기존 beanllm 코드 import (wrapping 대상)
from beanllm.facade.advanced import KnowledgeGraphFacade
from beanllm.dto.request.graph import KGRequest, KGQueryRequest
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스 생성
mcp = FastMCP("Knowledge Graph Tools")

# 전역 KG 인스턴스 캐시
_kg_instances: Dict[str, KnowledgeGraphFacade] = {}


@mcp.tool()
async def build_knowledge_graph(
    documents_path: str,
    graph_name: str = "default",
    model: str = MCPServerConfig.DEFAULT_CHAT_MODEL,
    use_neo4j: bool = False,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
) -> dict:
    """
    지식 그래프 구축 (기존 코드 재사용)

    Args:
        documents_path: 문서 경로 (디렉토리 또는 파일)
        graph_name: 그래프 식별 이름
        model: 엔티티/관계 추출에 사용할 LLM 모델
        use_neo4j: Neo4j 사용 여부 (False면 메모리 그래프)
        neo4j_uri: Neo4j URI (use_neo4j=True일 때)
        neo4j_user: Neo4j 사용자명
        neo4j_password: Neo4j 비밀번호

    Returns:
        dict: 성공 여부, 엔티티 개수, 관계 개수

    Example:
        User: "이 문서들로 지식 그래프 만들어줘"
        → build_knowledge_graph(
            documents_path="/path/to/docs",
            graph_name="my_kg"
        )
    """
    try:
        # 🎯 기존 beanllm 코드 재사용!
        # 1. KGRequest 생성
        request = KGRequest(
            documents_path=documents_path,
            model=model,
            use_neo4j=use_neo4j,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )

        # 2. KnowledgeGraphFacade 생성
        facade = KnowledgeGraphFacade.from_documents(request)

        # 3. 캐시에 저장
        _kg_instances[graph_name] = facade

        # 4. 통계 수집
        stats = facade.get_stats()

        return {
            "success": True,
            "graph_name": graph_name,
            "entity_count": stats.get("entity_count", 0),
            "relation_count": stats.get("relation_count", 0),
            "entity_types": stats.get("entity_types", []),
            "relation_types": stats.get("relation_types", []),
            "model": model,
            "backend": "neo4j" if use_neo4j else "memory",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def query_knowledge_graph(
    query: str,
    graph_name: str = "default",
    model: str = MCPServerConfig.DEFAULT_CHAT_MODEL,
    max_depth: int = 2,
) -> dict:
    """
    지식 그래프에 질의 (기존 코드 재사용)

    Args:
        query: 질문
        graph_name: 사용할 그래프 이름
        model: 답변 생성에 사용할 LLM 모델
        max_depth: 탐색 깊이

    Returns:
        dict: 답변, 관련 엔티티, 관계

    Example:
        User: "beanllm과 관련된 기술은 뭐야?"
        → query_knowledge_graph(
            query="beanllm과 관련된 기술은?",
            graph_name="my_kg"
        )
    """
    try:
        # 1. 캐시에서 KG 가져오기
        if graph_name not in _kg_instances:
            return {
                "success": False,
                "error": f"Knowledge graph '{graph_name}' not found. Please build it first.",
            }

        facade = _kg_instances[graph_name]

        # 2. 🎯 기존 KnowledgeGraphFacade.query() 메서드 사용!
        request = KGQueryRequest(
            query=query,
            model=model,
            max_depth=max_depth,
        )

        result = await asyncio.to_thread(facade.query, request)

        # 3. 결과 포매팅
        entities = []
        for entity in result.entities:
            entities.append(
                {
                    "name": entity.name,
                    "type": entity.type,
                    "properties": entity.properties,
                }
            )

        relations = []
        for relation in result.relations:
            relations.append(
                {
                    "source": relation.source,
                    "target": relation.target,
                    "type": relation.type,
                    "properties": relation.properties,
                }
            )

        return {
            "success": True,
            "answer": result.answer,
            "entities": entities,
            "relations": relations,
            "model": model,
            "query": query,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_kg_stats(graph_name: str = "default") -> dict:
    """
    지식 그래프 통계 조회

    Args:
        graph_name: 그래프 이름

    Returns:
        dict: 엔티티/관계 개수, 타입 분포

    Example:
        User: "지식 그래프 통계 알려줘"
        → get_kg_stats(graph_name="my_kg")
    """
    try:
        if graph_name not in _kg_instances:
            return {
                "success": False,
                "error": f"Knowledge graph '{graph_name}' not found.",
            }

        facade = _kg_instances[graph_name]
        stats = facade.get_stats()

        return {
            "success": True,
            "graph_name": graph_name,
            **stats,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def visualize_knowledge_graph(
    graph_name: str = "default",
    output_path: Optional[str] = None,
) -> dict:
    """
    지식 그래프 시각화 (기존 코드 재사용)

    Args:
        graph_name: 그래프 이름
        output_path: 출력 파일 경로 (None이면 임시 파일)

    Returns:
        dict: 성공 여부, 파일 경로

    Example:
        User: "지식 그래프를 시각화해줘"
        → visualize_knowledge_graph(graph_name="my_kg")
    """
    try:
        if graph_name not in _kg_instances:
            return {
                "success": False,
                "error": f"Knowledge graph '{graph_name}' not found.",
            }

        facade = _kg_instances[graph_name]

        # 출력 경로 설정
        if output_path is None:
            output_path = str(MCPServerConfig.DATA_DIR / f"{graph_name}_graph.html")

        # 🎯 기존 visualize() 메서드 사용
        result_path = facade.visualize(output_path=output_path)

        return {
            "success": True,
            "graph_name": graph_name,
            "output_path": result_path,
            "message": f"Graph visualization saved to {result_path}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def list_knowledge_graphs() -> dict:
    """
    생성된 지식 그래프 목록 조회

    Returns:
        dict: 그래프 이름 목록

    Example:
        User: "어떤 지식 그래프들이 있어?"
        → list_knowledge_graphs()
    """
    return {
        "success": True,
        "graphs": list(_kg_instances.keys()),
        "count": len(_kg_instances),
    }


@mcp.tool()
async def delete_knowledge_graph(graph_name: str) -> dict:
    """
    지식 그래프 삭제

    Args:
        graph_name: 삭제할 그래프 이름

    Returns:
        dict: 성공 여부

    Example:
        User: "my_kg 그래프 삭제해줘"
        → delete_knowledge_graph(graph_name="my_kg")
    """
    try:
        if graph_name not in _kg_instances:
            return {
                "success": False,
                "error": f"Knowledge graph '{graph_name}' not found.",
            }

        del _kg_instances[graph_name]

        return {
            "success": True,
            "message": f"Knowledge graph '{graph_name}' deleted successfully.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
