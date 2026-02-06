"""
Session Resources - 세션 데이터 및 통계를 MCP Resource로 노출

MCP Resources는 Claude가 읽을 수 있는 데이터입니다.
Tools는 실행 가능한 함수, Resources는 데이터 소스입니다.
"""

import json
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

# 기존 beanllm 인프라 import
from beanllm.infrastructure.distributed.google_events import (
    get_google_export_stats,
    get_security_events,
)
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스
mcp = FastMCP("Session Resources")


@mcp.resource("session://stats/google_exports")
async def get_google_export_stats_resource(hours: int = 24) -> str:
    """
    Google Workspace 내보내기 통계 (MCP Resource)

    Claude가 이 resource를 읽으면 최근 Google 서비스 사용 통계를 볼 수 있습니다.

    Args:
        hours: 조회 기간 (시간)

    Returns:
        str: JSON 형식 통계 데이터

    Usage in Claude:
        "session://stats/google_exports?hours=24" 를 읽어줘
    """
    try:
        stats = await get_google_export_stats(hours=hours)

        # JSON 문자열로 반환 (MCP Resource는 문자열 반환)
        return json.dumps(
            {
                "period_hours": hours,
                "total_exports": stats.get("total_exports", 0),
                "by_service": stats.get("by_service", {}),
                "top_users": stats.get("top_users", []),
                "hourly_pattern": stats.get("hourly_pattern", {}),
            },
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("session://stats/security_events")
async def get_security_events_resource(hours: int = 24) -> str:
    """
    보안 이벤트 통계 (MCP Resource)

    고위험 이벤트, Rate limit 초과, 비정상 활동 등을 조회합니다.

    Args:
        hours: 조회 기간 (시간)

    Returns:
        str: JSON 형식 보안 이벤트 데이터

    Usage in Claude:
        "session://stats/security_events?hours=24" 를 읽어줘
    """
    try:
        events = await get_security_events(hours=hours)

        return json.dumps(
            {
                "period_hours": hours,
                "high_risk_events": events.get("high_risk_events", []),
                "rate_limit_exceeded": events.get("rate_limit_exceeded", []),
                "abnormal_activities": events.get("abnormal_activities", []),
            },
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("session://config/server")
async def get_server_config_resource() -> str:
    """
    MCP 서버 설정 (MCP Resource)

    현재 서버 설정 정보를 조회합니다.

    Returns:
        str: JSON 형식 서버 설정

    Usage in Claude:
        "session://config/server" 를 읽어줘
    """
    config_data = {
        "server_name": MCPServerConfig.SERVER_NAME,
        "server_version": MCPServerConfig.SERVER_VERSION,
        "host": MCPServerConfig.HOST,
        "port": MCPServerConfig.PORT,
        "default_models": {
            "chat": MCPServerConfig.DEFAULT_CHAT_MODEL,
            "embedding": MCPServerConfig.DEFAULT_EMBEDDING_MODEL,
        },
        "chunk_settings": {
            "size": MCPServerConfig.DEFAULT_CHUNK_SIZE,
            "overlap": MCPServerConfig.DEFAULT_CHUNK_OVERLAP,
        },
        "session_ttl_seconds": MCPServerConfig.SESSION_TTL_SECONDS,
        "api_keys_configured": {
            "openai": MCPServerConfig.OPENAI_API_KEY is not None,
            "anthropic": MCPServerConfig.ANTHROPIC_API_KEY is not None,
            "gemini": MCPServerConfig.GEMINI_API_KEY is not None,
            "deepseek": MCPServerConfig.DEEPSEEK_API_KEY is not None,
            "perplexity": MCPServerConfig.PERPLEXITY_API_KEY is not None,
        },
        "databases_configured": {
            "mongodb": MCPServerConfig.MONGODB_URI is not None,
            "redis": MCPServerConfig.REDIS_URL is not None,
        },
    }

    return json.dumps(config_data, indent=2)


@mcp.resource("session://info/rag_systems")
async def get_rag_systems_info_resource() -> str:
    """
    RAG 시스템 정보 (MCP Resource)

    현재 생성된 RAG 시스템 목록 및 통계

    Returns:
        str: JSON 형식 RAG 시스템 정보

    Usage in Claude:
        "session://info/rag_systems" 를 읽어줘
    """
    try:
        # rag_tools에서 _rag_instances import
        from mcp_server.tools.rag_tools import _rag_instances

        systems = []
        for name, rag in _rag_instances.items():
            collection = rag._vector_store._collection
            data = collection.get()

            systems.append(
                {
                    "name": name,
                    "total_chunks": len(data["ids"]),
                    "embedding_dimension": (
                        len(data["embeddings"][0]) if data["embeddings"] else 0
                    ),
                    "vector_store_type": type(rag._vector_store).__name__,
                }
            )

        return json.dumps(
            {"total_systems": len(systems), "systems": systems},
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("session://info/multiagent_systems")
async def get_multiagent_systems_info_resource() -> str:
    """
    Multi-Agent 시스템 정보 (MCP Resource)

    현재 생성된 Multi-Agent 시스템 목록 및 통계

    Returns:
        str: JSON 형식 Multi-Agent 시스템 정보

    Usage in Claude:
        "session://info/multiagent_systems" 를 읽어줘
    """
    try:
        from mcp_server.tools.agent_tools import _multiagent_systems

        systems = []
        for name, facade in _multiagent_systems.items():
            agents_info = []
            for agent in facade._agents:
                agents_info.append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "model": agent.model,
                    }
                )

            systems.append(
                {
                    "name": name,
                    "agent_count": len(agents_info),
                    "agents": agents_info,
                    "strategy": facade._strategy.value,
                    "max_rounds": facade._max_rounds,
                }
            )

        return json.dumps(
            {"total_systems": len(systems), "systems": systems},
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("session://info/knowledge_graphs")
async def get_knowledge_graphs_info_resource() -> str:
    """
    Knowledge Graph 정보 (MCP Resource)

    현재 생성된 지식 그래프 목록 및 통계

    Returns:
        str: JSON 형식 지식 그래프 정보

    Usage in Claude:
        "session://info/knowledge_graphs" 를 읽어줘
    """
    try:
        from mcp_server.tools.kg_tools import _kg_instances

        graphs = []
        for name, facade in _kg_instances.items():
            stats = facade.get_stats()
            graphs.append(
                {
                    "name": name,
                    "entity_count": stats.get("entity_count", 0),
                    "relation_count": stats.get("relation_count", 0),
                    "entity_types": stats.get("entity_types", []),
                    "relation_types": stats.get("relation_types", []),
                }
            )

        return json.dumps(
            {"total_graphs": len(graphs), "graphs": graphs},
            indent=2,
        )

    except Exception as e:
        return json.dumps({"error": str(e)})
