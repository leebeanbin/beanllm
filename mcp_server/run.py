#!/usr/bin/env python3
"""
beanllm MCP Server - 메인 엔트리포인트

기존 beanllm 코드를 Model Context Protocol로 wrapping하여
Claude Desktop, Cursor, ChatGPT 등에서 사용 가능하게 합니다.

Usage:
    # Development
    python mcp_server/run.py

    # Production with uvicorn
    uvicorn mcp_server.run:app --host 0.0.0.0 --port 8765

    # Claude Desktop 설정 (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "beanllm": {
          "command": "python",
          "args": ["/path/to/llmkit/mcp_server/run.py"]
        }
      }
    }
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP

from mcp_server.config import MCPServerConfig

# ============================================================================
# FastMCP 서버 생성
# ============================================================================
mcp = FastMCP(
    name=MCPServerConfig.SERVER_NAME,
    version=MCPServerConfig.SERVER_VERSION,
    description=MCPServerConfig.SERVER_DESCRIPTION,
)

# ============================================================================
# 모든 Tools, Resources, Prompts 로드
# ============================================================================

print("🚀 Loading beanllm MCP Server...")

# Tools
try:
    from mcp_server.tools import agent_tools, google_tools, kg_tools, ml_tools, rag_tools

    print("✅ Tools loaded:")
    print("  - RAG Tools (5 tools)")
    print("  - Multi-Agent Tools (6 tools)")
    print("  - Knowledge Graph Tools (7 tools)")
    print("  - ML Tools (9 tools: audio, ocr, evaluation)")
    print("  - Google Workspace Tools (6 tools)")
    print("  Total: 33 tools")
except Exception as e:
    print(f"⚠️  Warning: Failed to load some tools: {e}")

# Resources
try:
    from mcp_server.resources import session_resources

    print("✅ Resources loaded:")
    print("  - Session Resources (7 resources)")
except Exception as e:
    print(f"⚠️  Warning: Failed to load resources: {e}")

# Prompts
try:
    from mcp_server.prompts import templates

    print("✅ Prompts loaded:")
    print("  - Prompt Templates (8 templates)")
except Exception as e:
    print(f"⚠️  Warning: Failed to load prompts: {e}")

# ============================================================================
# Server Info
# ============================================================================


@mcp.tool()
def get_server_info() -> dict:
    """
    MCP 서버 정보 조회

    Returns:
        dict: 서버 버전, 사용 가능한 도구 목록

    Example:
        User: "서버 정보 알려줘"
        → get_server_info()
    """
    return {
        "name": MCPServerConfig.SERVER_NAME,
        "version": MCPServerConfig.SERVER_VERSION,
        "description": MCPServerConfig.SERVER_DESCRIPTION,
        "total_tools": 33,
        "total_resources": 7,
        "total_prompts": 8,
        "categories": {
            "rag": "RAG 시스템 구축 및 질의",
            "multi_agent": "다중 에이전트 협업",
            "knowledge_graph": "지식 그래프 구축 및 탐색",
            "audio": "음성 인식 및 전사",
            "ocr": "이미지 텍스트 인식",
            "evaluation": "모델 평가 및 벤치마킹",
            "google_workspace": "Google Docs/Drive/Gmail 연동",
        },
        "config": {
            "default_chat_model": MCPServerConfig.DEFAULT_CHAT_MODEL,
            "default_embedding_model": MCPServerConfig.DEFAULT_EMBEDDING_MODEL,
            "chunk_size": MCPServerConfig.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": MCPServerConfig.DEFAULT_CHUNK_OVERLAP,
            "top_k": MCPServerConfig.DEFAULT_TOP_K,
        },
    }


# ============================================================================
# Main
# ============================================================================


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print(f"🫘 {MCPServerConfig.SERVER_NAME} v{MCPServerConfig.SERVER_VERSION}")
    print("=" * 60)
    print(f"Host: {MCPServerConfig.HOST}")
    print(f"Port: {MCPServerConfig.PORT}")
    print(f"Default Chat Model: {MCPServerConfig.DEFAULT_CHAT_MODEL}")
    print(f"Default Embedding Model: {MCPServerConfig.DEFAULT_EMBEDDING_MODEL}")
    print("=" * 60)
    print("\n🎯 MCP Server is ready!")
    print("\n📚 Usage:")
    print("  1. Add to Claude Desktop config:")
    print("     ~/.config/claude/claude_desktop_config.json")
    print("     {")
    print('       "mcpServers": {')
    print('         "beanllm": {')
    print('           "command": "python",')
    print(f'           "args": ["{Path(__file__).absolute()}"]')
    print("         }")
    print("       }")
    print("     }")
    print("\n  2. Restart Claude Desktop")
    print("\n  3. Start chatting with beanllm tools!")
    print("     Example: '이 폴더의 PDF로 RAG 시스템 만들어줘'")
    print("\n" + "=" * 60 + "\n")

    # Run FastMCP server
    mcp.run()


if __name__ == "__main__":
    main()
