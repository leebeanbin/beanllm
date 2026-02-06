"""
MCP Client Service

MCP 서버의 tools를 직접 함수 호출로 실행하는 클라이언트
중앙 관리 포인트: MCP 서버의 tools만 사용 (Single Source of Truth)

Usage:
    from services.mcp_client_service import mcp_client

    result = await mcp_client.call_tool(
        tool_name="query_rag_system",
        arguments={"query": "...", "collection_name": "default"},
        session_id="session_123"
    )
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPClientService:
    """
    MCP 서버와 통신하는 클라이언트 (직접 함수 호출)

    HTTP 통신이 아닌 Python 함수 직접 호출 방식으로 MCP tools 실행
    """

    def __init__(self):
        self._tools_cache: Dict[str, Callable] = {}
        self._tool_modules_loaded = False

    def _load_tool_modules(self) -> None:
        """MCP tool 모듈 로드 (lazy loading)"""
        if self._tool_modules_loaded:
            return

        try:
            # MCP tools 모듈 import
            from mcp_server.tools import (
                agent_tools,
                google_tools,
                kg_tools,
                ml_tools,
                rag_tools,
            )

            # 각 모듈에서 tool 함수 수집
            self._register_module_tools("rag", rag_tools)
            self._register_module_tools("agent", agent_tools)
            self._register_module_tools("kg", kg_tools)
            self._register_module_tools("ml", ml_tools)
            self._register_module_tools("google", google_tools)

            self._tool_modules_loaded = True
            logger.info(f"MCP tools loaded: {len(self._tools_cache)} tools available")

        except ImportError as e:
            logger.error(f"Failed to load MCP tools: {e}")
            raise

    def _register_module_tools(self, module_name: str, module: Any) -> None:
        """모듈에서 tool 함수 등록"""
        # @mcp.tool() 데코레이터가 적용된 함수 찾기
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            attr = getattr(module, attr_name)
            if callable(attr) and asyncio.iscoroutinefunction(attr):
                # async 함수만 등록 (MCP tools는 모두 async)
                self._tools_cache[attr_name] = attr
                logger.debug(f"Registered tool: {attr_name} from {module_name}")

    def get_available_tools(self) -> List[str]:
        """사용 가능한 tool 목록 반환"""
        self._load_tool_modules()
        return list(self._tools_cache.keys())

    def has_tool(self, tool_name: str) -> bool:
        """특정 tool이 존재하는지 확인"""
        self._load_tool_modules()
        return tool_name in self._tools_cache

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        MCP tool 호출

        Args:
            tool_name: tool 이름 (예: "query_rag_system", "run_multiagent_task")
            arguments: tool 인자
            session_id: 세션 ID (세션별 인스턴스 관리용)

        Returns:
            tool 실행 결과 (dict)

        Raises:
            ValueError: tool을 찾을 수 없는 경우
            Exception: tool 실행 중 오류 발생

        Example:
            result = await mcp_client.call_tool(
                tool_name="query_rag_system",
                arguments={
                    "query": "beanllm이 뭐야?",
                    "collection_name": "default",
                    "top_k": 5,
                },
                session_id="session_123"
            )
        """
        self._load_tool_modules()

        # Tool 함수 찾기
        tool_func = self._tools_cache.get(tool_name)
        if tool_func is None:
            available = ", ".join(sorted(self._tools_cache.keys())[:10])
            raise ValueError(
                f"Tool '{tool_name}' not found in MCP server. " f"Available tools: {available}..."
            )

        # session_id 추가 (tool이 지원하는 경우)
        if session_id and "session_id" not in arguments:
            arguments["session_id"] = session_id

        # Tool 실행
        logger.info(f"Calling MCP tool: {tool_name} with session_id={session_id}")
        try:
            result = await tool_func(**arguments)
            logger.info(f"MCP tool {tool_name} completed: success={result.get('success', 'N/A')}")
            return result
        except Exception as e:
            logger.error(f"MCP tool {tool_name} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }

    async def call_rag_query(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int = 5,
        model: Optional[str] = None,
        temperature: float = 0.7,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RAG 질의 (편의 메서드)"""
        arguments = {
            "query": query,
            "collection_name": collection_name,
            "top_k": top_k,
            "temperature": temperature,
        }
        if model:
            arguments["model"] = model

        return await self.call_tool("query_rag_system", arguments, session_id)

    async def call_rag_build(
        self,
        documents_path: str,
        collection_name: str = "default",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """RAG 시스템 구축 (편의 메서드)"""
        return await self.call_tool(
            "build_rag_system",
            {
                "documents_path": documents_path,
                "collection_name": collection_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
            session_id,
        )

    async def call_multiagent_run(
        self,
        system_name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Multi-Agent 작업 실행 (편의 메서드)"""
        return await self.call_tool(
            "run_multiagent_task",
            {
                "system_name": system_name,
                "task": task,
                "context": context or {},
            },
            session_id,
        )

    async def call_kg_query(
        self,
        query: str,
        graph_name: str = "default",
        model: Optional[str] = None,
        max_depth: int = 2,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Knowledge Graph 질의 (편의 메서드)"""
        arguments = {
            "query": query,
            "graph_name": graph_name,
            "max_depth": max_depth,
        }
        if model:
            arguments["model"] = model

        return await self.call_tool("query_knowledge_graph", arguments, session_id)

    async def call_audio_transcribe(
        self,
        audio_path: str,
        engine: str = "whisper",
        language: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Audio 전사 (편의 메서드)"""
        arguments = {
            "audio_path": audio_path,
            "engine": engine,
        }
        if language:
            arguments["language"] = language

        return await self.call_tool("transcribe_audio", arguments, session_id)

    async def call_ocr(
        self,
        image_path: str,
        engine: str = "tesseract",
        language: str = "eng",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """OCR 텍스트 인식 (편의 메서드)"""
        return await self.call_tool(
            "recognize_text_ocr",
            {
                "image_path": image_path,
                "engine": engine,
                "language": language,
            },
            session_id,
        )

    async def call_evaluation(
        self,
        model: str,
        evaluation_type: str = "answer_relevancy",
        test_data: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """모델 평가 (편의 메서드)"""
        return await self.call_tool(
            "evaluate_model",
            {
                "model": model,
                "evaluation_type": evaluation_type,
                "test_data": test_data or [],
            },
            session_id,
        )


# Singleton instance
mcp_client = MCPClientService()
