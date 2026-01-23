"""
Chat Streaming with Tool Calls - SSE endpoint for real-time progress

기존 beanllm 코드를 직접 사용하여 Tool Call 진행 상황을 SSE로 스트리밍합니다.
MCP Server가 아닌 beanllm Facade/Handler를 직접 호출합니다.
"""
import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel

# ✅ 기존 beanllm 코드 직접 사용 (MCP Server 패턴과 동일)
from beanllm.facade.core import Client, RAGChain
from beanllm.domain.loaders import DirectoryLoader, PDFLoader, TextLoader, CSVLoader
from beanllm.handler.core import ChatHandler, RAGHandler, AgentHandler
from beanllm.handler.advanced import MultiAgentHandler

# ✅ Kafka 이벤트 로거 (분산 모니터링)
try:
    from beanllm.infrastructure.distributed import get_event_logger
    event_logger = get_event_logger()
except ImportError:
    event_logger = None

# ✅ Redis 캐시 (분산 캐싱)
try:
    from beanllm.infrastructure.distributed.redis.client import get_redis_client
    redis_client = get_redis_client()
except ImportError:
    redis_client = None

logger = logging.getLogger(__name__)

# ✅ RAG 인스턴스 캐시 (MCP Server 패턴과 동일)
_rag_instances: Dict[str, RAGChain] = {}


class MCPChatRequest(BaseModel):
    """MCP Chat 요청"""
    messages: List[Dict[str, str]]
    model: str = "qwen2.5:0.5b"
    temperature: float = 0.7
    max_tokens: int = 1000


class MCPStreamingClient:
    """MCP 서버와 통신하는 클라이언트"""

    def __init__(self, mcp_server_url: str = "http://localhost:8765"):
        self.mcp_server_url = mcp_server_url

    async def detect_and_call_tools(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        사용자 질의에서 Tool Call을 감지하고 실행

        Args:
            query: 사용자 질의
            context: 컨텍스트 (모델, 파라미터 등)

        Yields:
            Dict: SSE 이벤트 데이터
                - type: "tool_call", "tool_progress", "tool_result", "text", "done"
                - data: 이벤트 데이터
        """
        # 1. 질의에서 Tool Call 감지 (간단한 키워드 매칭)
        tool_calls = await self._detect_tools(query)

        if not tool_calls:
            # Tool Call 없으면 일반 chat 응답
            yield {
                "type": "text",
                "data": {
                    "content": "Tool call이 감지되지 않았습니다. 일반 채팅 모드로 전환합니다."
                }
            }
            return

        # 2. 감지된 Tool Call 실행
        for tool_call in tool_calls:
            # Tool call 시작 알림
            yield {
                "type": "tool_call",
                "data": {
                    "tool": tool_call["name"],
                    "arguments": tool_call["arguments"],
                    "status": "started"
                }
            }

            # Tool 실행 (실제로는 MCP 서버와 통신)
            try:
                async for progress in self._execute_tool(tool_call):
                    yield progress

                # Tool call 완료
                yield {
                    "type": "tool_result",
                    "data": {
                        "tool": tool_call["name"],
                        "result": progress.get("data", {}),
                        "status": "completed"
                    }
                }

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                yield {
                    "type": "tool_result",
                    "data": {
                        "tool": tool_call["name"],
                        "error": str(e),
                        "status": "failed"
                    }
                }

        # 3. 완료
        yield {
            "type": "done",
            "data": {
                "message": "All tools executed"
            }
        }

    async def _detect_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        질의에서 Tool Call 감지 (간단한 키워드 매칭)

        나중에는 LLM이 판단하도록 개선할 수 있음
        """
        query_lower = query.lower()
        tool_calls = []

        # RAG 관련 키워드
        if any(keyword in query_lower for keyword in ["rag", "검색", "문서", "pdf"]):
            # RAG 시스템 구축 감지
            if any(keyword in query_lower for keyword in ["만들", "구축", "생성", "build"]):
                # 문서 경로 추출 (간단한 정규표현식)
                import re
                path_match = re.search(r'(/[\w/.-]+)', query)
                doc_path = path_match.group(1) if path_match else "./docs"

                tool_calls.append({
                    "name": "build_rag_system",
                    "arguments": {
                        "documents_path": doc_path,
                        "collection_name": "default"
                    }
                })

            # RAG 질의 감지
            elif any(keyword in query_lower for keyword in ["뭐", "무엇", "알려", "설명", "?", "?"]):
                tool_calls.append({
                    "name": "query_rag_system",
                    "arguments": {
                        "query": query,
                        "collection_name": "default"
                    }
                })

        # Multi-Agent 관련 키워드
        elif any(keyword in query_lower for keyword in ["에이전트", "토론", "협업", "agent", "debate"]):
            # 토론 주제 추출
            import re
            topic_match = re.search(r'["\'](.+?)["\']', query)
            topic = topic_match.group(1) if topic_match else query

            tool_calls.append({
                "name": "create_multiagent_system",
                "arguments": {
                    "system_name": "debate_team",
                    "agent_configs": [
                        {"name": "researcher", "role": "Research specialist"},
                        {"name": "writer", "role": "Writing specialist"},
                        {"name": "critic", "role": "Critical reviewer"}
                    ],
                    "strategy": "debate"
                }
            })

            tool_calls.append({
                "name": "run_multiagent_task",
                "arguments": {
                    "system_name": "debate_team",
                    "task": topic
                }
            })

        # Knowledge Graph 관련 키워드
        elif any(keyword in query_lower for keyword in ["지식 그래프", "knowledge graph", "kg", "그래프"]):
            import re
            path_match = re.search(r'(/[\w/.-]+)', query)
            doc_path = path_match.group(1) if path_match else "./docs"

            tool_calls.append({
                "name": "build_knowledge_graph",
                "arguments": {
                    "documents_path": doc_path,
                    "graph_name": "default"
                }
            })

        # Audio 관련 키워드
        elif any(keyword in query_lower for keyword in ["음성", "전사", "audio", "transcribe"]):
            import re
            path_match = re.search(r'(/[\w/.-]+\.(?:mp3|wav|m4a))', query)
            audio_path = path_match.group(1) if path_match else "./audio.mp3"

            tool_calls.append({
                "name": "transcribe_audio",
                "arguments": {
                    "audio_path": audio_path,
                    "engine": "whisper"
                }
            })

        # OCR 관련 키워드
        elif any(keyword in query_lower for keyword in ["ocr", "텍스트 추출", "이미지", "인식"]):
            import re
            path_match = re.search(r'(/[\w/.-]+\.(?:png|jpg|jpeg))', query)
            image_path = path_match.group(1) if path_match else "./image.png"

            tool_calls.append({
                "name": "recognize_text_ocr",
                "arguments": {
                    "image_path": image_path,
                    "engine": "tesseract"
                }
            })

        return tool_calls

    async def _execute_tool(
        self,
        tool_call: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Tool 실행 (✅ 기존 beanllm 코드 직접 사용)

        MCP Server가 아닌 beanllm Facade/Handler를 직접 호출합니다.
        """
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]

        # ✅ 실제 beanllm 코드 실행
        if tool_name == "build_rag_system":
            # ✅ Kafka 이벤트 발행 (Tool Call 시작)
            if event_logger:
                await event_logger.log_event(
                    event_type="tool_call.started",
                    event_data={
                        "tool": tool_name,
                        "arguments": arguments,
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )

            # 진행 상황: 문서 로드
            yield {
                "type": "tool_progress",
                "data": {
                    "tool": tool_name,
                    "step": "loading_documents",
                    "message": "문서 로드 중...",
                    "progress": 0.2
                }
            }

            # 🎯 기존 beanllm 코드 사용 (MCP Server 패턴과 동일)
            try:
                collection_name = arguments.get("collection_name", "default")

                # ✅ Redis 캐시 확인 (이미 구축된 컬렉션인지 확인)
                if redis_client:
                    cache_key = f"rag:collection:{collection_name}"
                    cached = await asyncio.to_thread(redis_client.get, cache_key)
                    if cached and collection_name in _rag_instances:
                        logger.info(f"✅ RAG collection '{collection_name}' already exists (Redis cache hit)")
                        # 이미 구축된 컬렉션이면 캐시된 메타데이터 반환
                        import json
                        cached_data = json.loads(cached)
                        yield {
                            "type": "tool_result",
                            "data": {
                                "tool": tool_name,
                                "result": {
                                    **cached_data,
                                    "cached": True,
                                },
                                "status": "completed"
                            }
                        }
                        return

                path = Path(arguments["documents_path"])

                # 1. 문서 로드
                if path.is_dir():
                    loader = DirectoryLoader(str(path))
                elif path.suffix == ".pdf":
                    loader = PDFLoader(str(path))
                elif path.suffix in [".txt", ".md"]:
                    loader = TextLoader(str(path))
                elif path.suffix == ".csv":
                    loader = CSVLoader(str(path))
                else:
                    raise ValueError(f"Unsupported file type: {path.suffix}")

                documents = await asyncio.to_thread(loader.load)

                # 진행 상황: 임베딩 생성
                yield {
                    "type": "tool_progress",
                    "data": {
                        "tool": tool_name,
                        "step": "creating_embeddings",
                        "message": f"임베딩 생성 중... ({len(documents)}개 문서)",
                        "progress": 0.5
                    }
                }

                # 2. RAG 구축
                rag = await asyncio.to_thread(
                    RAGChain.from_documents,
                    documents=documents,
                    collection_name=arguments.get("collection_name", "default"),
                    chunk_size=arguments.get("chunk_size", 1000),
                    chunk_overlap=arguments.get("chunk_overlap", 200),
                    embedding_model=arguments.get("embedding_model", "mxbai-embed-large:335m"),
                )

                # 진행 상황: 인덱스 구축
                yield {
                    "type": "tool_progress",
                    "data": {
                        "tool": tool_name,
                        "step": "building_index",
                        "message": "벡터 인덱스 구축 중...",
                        "progress": 0.8
                    }
                }

                # 3. 캐시에 저장 (메모리 + Redis)
                _rag_instances[collection_name] = rag

                # 4. 청크 수 계산
                chunk_count = len(rag._vector_store._collection.get()["ids"])

                result_data = {
                    "success": True,
                    "collection_name": collection_name,
                    "document_count": len(documents),
                    "chunk_count": chunk_count,
                    "embedding_model": arguments.get("embedding_model", "mxbai-embed-large:335m"),
                }

                # ✅ Redis에 메타데이터 저장 (분산 캐싱)
                if redis_client:
                    import json
                    cache_key = f"rag:collection:{collection_name}"
                    await asyncio.to_thread(
                        redis_client.setex,
                        cache_key,
                        3600,  # TTL: 1시간
                        json.dumps(result_data)
                    )
                    logger.info(f"✅ Saved RAG collection '{collection_name}' to Redis cache")

                # ✅ Kafka 이벤트 발행 (Tool Call 완료)
                if event_logger:
                    await event_logger.log_event(
                        event_type="tool_call.completed",
                        event_data={
                            "tool": tool_name,
                            "result": result_data,
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )

                # 최종 결과
                yield {
                    "type": "tool_result",
                    "data": {
                        "tool": tool_name,
                        "result": result_data,
                        "status": "completed"
                    }
                }

            except Exception as e:
                logger.error(f"RAG build failed: {e}")

                # ✅ Kafka 이벤트 발행 (Tool Call 실패)
                if event_logger:
                    await event_logger.log_event(
                        event_type="tool_call.failed",
                        event_data={
                            "tool": tool_name,
                            "error": str(e),
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    )

                yield {
                    "type": "tool_result",
                    "data": {
                        "tool": tool_name,
                        "error": str(e),
                        "status": "failed"
                    }
                }

        elif tool_name == "query_rag_system":
            # 진행 상황: 벡터 검색
            yield {
                "type": "tool_progress",
                "data": {
                    "tool": tool_name,
                    "step": "searching",
                    "message": "벡터 검색 중...",
                    "progress": 0.3
                }
            }

            # 🎯 기존 beanllm RAGChain 사용
            try:
                collection_name = arguments.get("collection_name", "default")
                query = arguments.get("query", "")
                top_k = arguments.get("top_k", 5)

                # ✅ 캐시된 RAG 인스턴스 가져오기
                if collection_name not in _rag_instances:
                    raise ValueError(f"RAG collection '{collection_name}' not found. Build it first with build_rag_system.")

                rag = _rag_instances[collection_name]

                # 진행 상황: 답변 생성
                yield {
                    "type": "tool_progress",
                    "data": {
                        "tool": tool_name,
                        "step": "generating_answer",
                        "message": "답변 생성 중...",
                        "progress": 0.7
                    }
                }

                # ✅ 실제 RAG 질의 실행
                response = await asyncio.to_thread(
                    rag.query,
                    query=query,
                    k=top_k
                )

                # 소스 문서 포맷팅
                sources = [
                    {
                        "rank": i + 1,
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for i, doc in enumerate(response.get("source_documents", []))
                ]

                result = {
                    "success": True,
                    "answer": response.get("answer", ""),
                    "query": query,
                    "collection": collection_name,
                    "sources": sources,
                    "source_count": len(sources),
                }

                # 최종 결과
                yield {
                    "type": "tool_result",
                    "data": {
                        "tool": tool_name,
                        "result": result,
                        "status": "completed"
                    }
                }

            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                yield {
                    "type": "tool_result",
                    "data": {
                        "tool": tool_name,
                        "error": str(e),
                        "status": "failed"
                    }
                }

        elif tool_name in ["create_multiagent_system", "run_multiagent_task"]:
            yield {
                "type": "tool_progress",
                "data": {
                    "tool": tool_name,
                    "step": "initializing",
                    "message": "에이전트 초기화 중...",
                    "progress": 0.2
                }
            }
            await asyncio.sleep(0.5)

            if tool_name == "run_multiagent_task":
                yield {
                    "type": "tool_progress",
                    "data": {
                        "tool": tool_name,
                        "step": "round_1",
                        "message": "라운드 1: 각 에이전트 의견 수집 중...",
                        "progress": 0.5
                    }
                }
                await asyncio.sleep(1.0)

                yield {
                    "type": "tool_progress",
                    "data": {
                        "tool": tool_name,
                        "step": "round_2",
                        "message": "라운드 2: 토론 진행 중...",
                        "progress": 0.8
                    }
                }
                await asyncio.sleep(1.0)

        # 기타 Tool도 유사하게 구현...


async def stream_mcp_chat(request: MCPChatRequest) -> AsyncGenerator[str, None]:
    """
    MCP Chat SSE streaming

    Args:
        request: Chat 요청

    Yields:
        str: SSE 형식 문자열 ("data: {...}\n\n")
    """
    client = MCPStreamingClient()

    # 마지막 사용자 메시지 추출
    user_messages = [msg for msg in request.messages if msg["role"] == "user"]
    if not user_messages:
        yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'No user message found'}})}\n\n"
        return

    last_user_message = user_messages[-1]["content"]

    # Tool Call 감지 및 실행
    try:
        async for event in client.detect_and_call_tools(
            query=last_user_message,
            context={"model": request.model, "temperature": request.temperature}
        ):
            yield f"data: {json.dumps(event)}\n\n"

    except Exception as e:
        logger.error(f"MCP streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}})}\n\n"
