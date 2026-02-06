"""
Agentic Orchestrator Service

Intent 분류 결과와 선택된 도구를 받아 실행하고 결과를 스트리밍합니다.
SSE (Server-Sent Events) 형식으로 진행 상황을 실시간 전달합니다.
"""

import asyncio
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from services.intent_classifier import IntentResult, IntentType
from services.mcp_client_service import mcp_client
from services.tool_registry import Tool, ToolCheckResult, ToolRegistry, tool_registry

logger = logging.getLogger(__name__)

# 그래프/노드 제한 (제안 단계·검증용)
MAX_NODES = 6


class EventType(str, Enum):
    """SSE 이벤트 타입"""

    INTENT = "intent"  # 의도 분류 결과
    TOOL_SELECT = "tool_select"  # 도구 선택
    PROPOSAL = "proposal"  # 제안 단계: 노드/파이프라인 제안 (챗 전용)
    HUMAN_APPROVAL = "human_approval"  # Human-in-the-loop: 도구 실행 전 승인 요청
    TOOL_START = "tool_start"  # 도구 실행 시작
    TOOL_PROGRESS = "tool_progress"  # 도구 진행 상황
    TOOL_RESULT = "tool_result"  # 도구 실행 결과
    TEXT = "text"  # 텍스트 청크 (스트리밍)
    TEXT_DONE = "text_done"  # 텍스트 완료
    ERROR = "error"  # 오류
    DONE = "done"  # 전체 완료
    STREAM_PAUSED = "stream_paused"  # Human-in-the-loop: 승인 대기로 스트림 일시 중단
    # 병렬 처리 관련
    PARALLEL_START = "parallel_start"  # 병렬 작업 시작
    PARALLEL_PROGRESS = "parallel_progress"  # 병렬 작업 진행 상황 (개별)
    PARALLEL_DONE = "parallel_done"  # 병렬 작업 완료


@dataclass
class AgenticEvent:
    """SSE 이벤트"""

    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sse(self) -> str:
        """SSE 형식 문자열로 변환"""
        event_data = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OrchestratorContext:
    """Orchestrator 실행 컨텍스트"""

    query: str  # 원본 쿼리
    intent: IntentResult  # 분류된 의도
    selected_tools: List[ToolCheckResult]  # 선택된 도구들
    model: str = "qwen2.5:0.5b"  # 사용할 모델
    temperature: float = 0.7  # 온도
    max_tokens: int = 2000  # 최대 토큰
    session_id: Optional[str] = None  # 세션 ID
    user_id: Optional[str] = None  # 사용자 ID
    extra_params: Dict[str, Any] = field(default_factory=dict)
    # 제안 단계: 유저가 챗으로 보낸 값 (approved | modified | custom_spec)
    proposal_action: Optional[str] = None
    proposal_pipeline: Optional[List[str]] = None  # custom_spec 시 도구 순서
    # Human-in-the-loop: run_id(요청 단위), 승인 대기 시 True
    run_id: Optional[str] = None
    require_approval: bool = False


class AgenticOrchestrator:
    """
    Agentic Orchestrator - 도구 실행 및 결과 스트리밍

    Usage:
        orchestrator = AgenticOrchestrator()
        async for event in orchestrator.execute(context):
            yield event.to_sse()
    """

    def __init__(self, registry: ToolRegistry = None):
        self._registry = registry or tool_registry
        self._tool_handlers: Dict[str, Any] = {}
        self._mcp_client = mcp_client  # MCP Client Service (중앙 관리 포인트)
        self._init_handlers()

    # ===========================================
    # Google OAuth Helper (중복 제거)
    # ===========================================

    async def _get_google_access_token(
        self,
        context: OrchestratorContext,
        tool: Tool,
    ) -> Optional[str]:
        """
        Google OAuth 액세스 토큰 가져오기 (공통 인증 로직).

        Args:
            context: 실행 컨텍스트
            tool: 현재 도구

        Returns:
            액세스 토큰 또는 None (인증 필요 시)

        Note:
            None 반환 시 호출자는 ERROR 이벤트를 yield해야 함
        """
        from database import get_mongodb_database

        from services.google_oauth_service import google_oauth_service

        db = get_mongodb_database()
        user_id = context.user_id or "default"

        return await google_oauth_service.get_valid_access_token(user_id, db)

    def _create_google_auth_error_event(self, tool: Tool) -> AgenticEvent:
        """Google 인증 필요 에러 이벤트 생성."""
        return AgenticEvent(
            type=EventType.ERROR,
            data={
                "tool": tool.name,
                "message": "Google sign-in required. Sign in with Google in Settings.",
                "auth_required": True,
            },
        )

    def _create_progress_event(
        self,
        tool: Tool,
        step: str,
        message: str,
        progress: float,
    ) -> AgenticEvent:
        """진행 상황 이벤트 생성 (공통 패턴)."""
        return AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": step,
                "message": message,
                "progress": progress,
            },
        )

    def _init_handlers(self):
        """도구 핸들러 초기화"""
        # 각 도구에 대한 핸들러 등록
        self._tool_handlers = {
            "chat": self._handle_chat,
            "rag": self._handle_rag,
            "agent": self._handle_agent,
            "multi_agent": self._handle_multi_agent,
            "web_search": self._handle_web_search,
            "code": self._handle_code,
            # Google services
            "google_drive": self._handle_google_drive,
            "google_docs": self._handle_google_docs,
            "google_gmail": self._handle_google_gmail,
            "google_calendar": self._handle_google_calendar,
            "google_sheets": self._handle_google_sheets,
            # Media
            "audio_transcribe": self._handle_audio,
            "vision": self._handle_vision,
            "ocr": self._handle_ocr,
            # Analysis
            "knowledge_graph": self._handle_knowledge_graph,
            "evaluation": self._handle_evaluation,
        }

    async def execute(
        self,
        context: OrchestratorContext,
        start_from_tool_index: int = 0,
    ) -> AsyncGenerator[AgenticEvent, None]:
        """
        도구 실행 및 결과 스트리밍

        Args:
            context: 실행 컨텍스트
            start_from_tool_index: 재개 시 N번 도구부터 실행 (Human-in-the-loop 재개용)

        Yields:
            AgenticEvent: SSE 이벤트
        """
        try:
            logger.info(
                f"Orchestrator.execute started: query={context.query[:50]}..., tools={len(context.selected_tools)}, start_from={start_from_tool_index}"
            )

            if start_from_tool_index == 0:
                # 1. Intent 이벤트
                logger.info(f"Yielding INTENT event: {context.intent.primary_intent.value}")
                yield AgenticEvent(
                    type=EventType.INTENT,
                    data={
                        "primary_intent": context.intent.primary_intent.value,
                        "confidence": context.intent.confidence,
                        "secondary_intents": [i.value for i in context.intent.secondary_intents],
                        "extracted_entities": context.intent.extracted_entities,
                    },
                )

                # 2. 도구가 없으면 기본 Chat으로 폴백
                if not context.selected_tools:
                    logger.warning("No tools selected, falling back to chat")
                    chat_tool = self._registry.get_best_tool_for_intent(IntentType.CHAT)
                    if chat_tool:
                        context.selected_tools = [chat_tool]
                        logger.info(f"Fallback to chat tool: {chat_tool.tool.name}")
                    else:
                        logger.error("No chat tool available")
                        yield AgenticEvent(
                            type=EventType.ERROR, data={"message": "No tools available"}
                        )
                        return

                # 노드 최대 개수 검증 (최대 MAX_NODES)
                if len(context.selected_tools) > MAX_NODES:
                    logger.warning(
                        f"Tool count {len(context.selected_tools)} exceeds MAX_NODES={MAX_NODES}, capping"
                    )
                    context.selected_tools = context.selected_tools[:MAX_NODES]

                # 3. 도구 선택 이벤트
                tool_names = [t.tool.name for t in context.selected_tools]
                logger.info(f"Yielding TOOL_SELECT event: {tool_names}")
                yield AgenticEvent(
                    type=EventType.TOOL_SELECT,
                    data={
                        "tools": tool_names,
                        "count": len(context.selected_tools),
                    },
                )

                # 3.5 제안 단계: 노드/파이프라인 제안 (챗 전용)
                node_count = min(len(context.selected_tools), MAX_NODES)
                pipeline = tool_names[:MAX_NODES]
                yield AgenticEvent(
                    type=EventType.PROPOSAL,
                    data={
                        "nodes": node_count,
                        "pipeline": pipeline,
                        "reason": f"Suggested pipeline with {node_count} step(s): {' → '.join(pipeline)}. Reply to approve, change, or specify your own.",
                    },
                )

                # 3.6 proposal_action 반영: custom_spec / modified 시 proposal_pipeline 순서로 도구 재구성
                if context.proposal_pipeline and len(context.proposal_pipeline) > 0:
                    name_to_check = {t.tool.name: t for t in context.selected_tools}
                    reordered = []
                    for name in context.proposal_pipeline[:MAX_NODES]:
                        if name in name_to_check:
                            reordered.append(name_to_check[name])
                    if reordered:
                        context.selected_tools = reordered
                        tool_names = [t.tool.name for t in context.selected_tools]
                        logger.info(f"Applied proposal_pipeline: {tool_names}")

            # 4. 각 도구 실행 (재개 시 start_from_tool_index부터)
            all_results = []
            accumulated_usage = {"input_tokens": 0, "output_tokens": 0}
            for idx in range(start_from_tool_index, len(context.selected_tools)):
                tool_check = context.selected_tools[idx]
                tool = tool_check.tool
                logger.info(f"Executing tool {idx+1}/{len(context.selected_tools)}: {tool.name}")

                # 도구 사용 불가 체크
                if not tool_check.is_available:
                    yield AgenticEvent(
                        type=EventType.ERROR,
                        data={
                            "tool": tool.name,
                            "message": f"Tool '{tool.name}' is not available",
                            "missing_keys": tool_check.missing_keys,
                            "missing_packages": tool_check.missing_packages,
                        },
                    )
                    continue

                # Human-in-the-loop: 도구 실행 직전 승인 요청 이벤트 (실행/취소/다른 도구로)
                approval_data = {
                    "tool": tool.name,
                    "query_snippet": context.query[:80]
                    + ("..." if len(context.query) > 80 else ""),
                    "actions": ["run", "cancel", "change_tool"],
                }
                if context.run_id:
                    approval_data["run_id"] = context.run_id
                yield AgenticEvent(type=EventType.HUMAN_APPROVAL, data=approval_data)

                # 승인 대기 모드: Redis에 상태 저장 후 스트림 중단
                if context.require_approval and context.run_id:
                    try:
                        import json

                        from beanllm.infrastructure.distributed.redis.client import get_redis_client

                        redis = get_redis_client()
                        if redis:
                            state = {
                                "query": context.query,
                                "tool_names": [t.tool.name for t in context.selected_tools],
                                "model": context.model,
                                "temperature": context.temperature,
                                "max_tokens": context.max_tokens,
                                "session_id": context.session_id,
                                "user_id": context.user_id,
                                "extra_params": context.extra_params or {},
                                "primary_intent": context.intent.primary_intent.value,
                                "confidence": context.intent.confidence,
                                "tool_index": idx,
                            }
                            key = f"run:approval:{context.run_id}"
                            await redis.setex(key, 3600, json.dumps(state))  # 1시간 TTL
                            logger.info(
                                f"Saved run state for approval: run_id={context.run_id}, tool_index={idx}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save run state for approval: {e}")
                    yield AgenticEvent(
                        type=EventType.STREAM_PAUSED,
                        data={"run_id": context.run_id, "tool": tool.name, "tool_index": idx},
                    )
                    return

                # 도구 시작 이벤트
                yield AgenticEvent(
                    type=EventType.TOOL_START,
                    data={
                        "tool": tool.name,
                        "description": tool.description_ko,
                        "is_streaming": tool.is_streaming,
                    },
                )

                # 도구 실행
                handler = self._tool_handlers.get(tool.name, self._handle_unknown)
                result = None

                try:
                    async for event in handler(context, tool):
                        yield event
                        # 최종 결과 수집
                        if event.type == EventType.TOOL_RESULT:
                            result = event.data.get("result")
                            all_results.append(
                                {
                                    "tool": tool.name,
                                    "result": result,
                                }
                            )
                            # 토큰 메트릭 수집 (handler가 usage를 넘기면 합산)
                            usage = event.data.get("usage")
                            if isinstance(usage, dict):
                                accumulated_usage["input_tokens"] += int(
                                    usage.get("input_tokens") or 0
                                )
                                accumulated_usage["output_tokens"] += int(
                                    usage.get("output_tokens") or 0
                                )
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Tool execution error: {tool.name} - {error_message}")
                    logger.error(traceback.format_exc())

                    # 메모리 부족 에러 등 특정 에러에 대한 사용자 친화적 메시지
                    user_friendly_message = error_message
                    if (
                        "memory" in error_message.lower()
                        or "requires more" in error_message.lower()
                    ):
                        user_friendly_message = f"메모리 부족: {error_message}. 더 작은 모델을 사용하거나 메모리를 확보해주세요."
                    elif "not found" in error_message.lower() or "404" in error_message:
                        user_friendly_message = f"모델을 찾을 수 없습니다: {error_message}. 모델이 설치되어 있는지 확인해주세요."

                    yield AgenticEvent(
                        type=EventType.ERROR,
                        data={
                            "tool": tool.name,
                            "message": user_friendly_message,
                            "original_error": error_message,
                            "traceback": traceback.format_exc(),
                        },
                    )

            # 5. 완료 이벤트 (usage 있으면 메트릭·로깅용)
            done_data = {
                "success": True,
                "tool_count": len(context.selected_tools),
                "results": all_results,
            }
            if accumulated_usage["input_tokens"] or accumulated_usage["output_tokens"]:
                done_data["usage"] = accumulated_usage
            yield AgenticEvent(type=EventType.DONE, data=done_data)

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            logger.error(traceback.format_exc())
            yield AgenticEvent(
                type=EventType.ERROR,
                data={
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    # ===========================================
    # Tool Handlers
    # ===========================================

    async def _handle_chat(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Chat 도구 핸들러 (스트리밍, 컨텍스트 관리 통합)"""
        try:
            logger.info(
                f"_handle_chat started: tool={tool.name}, model={context.model}, session_id={context.session_id}"
            )
            from beanllm.facade.core import Client
            from services.context_manager import context_manager

            # 진행 상황
            logger.info("Yielding TOOL_PROGRESS: initializing")
            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "initializing",
                    "message": "Initializing LLM client...",
                    "progress": 0.1,
                },
            )

            # 세션이 있으면 컨텍스트 관리 사용
            messages = []
            if context.session_id:
                # 요약이 필요한지 확인
                if context_manager.needs_summarization(context.session_id):
                    yield AgenticEvent(
                        type=EventType.TOOL_PROGRESS,
                        data={
                            "tool": tool.name,
                            "step": "summarizing",
                            "message": "Summarizing conversation...",
                            "progress": 0.15,
                        },
                    )

                    # 요약 생성
                    summary = await context_manager.summarize_if_needed(
                        context.session_id, model=context.model
                    )

                    if summary:
                        yield AgenticEvent(
                            type=EventType.TOOL_PROGRESS,
                            data={
                                "tool": tool.name,
                                "step": "summarized",
                                "message": f"Summary done: {summary[:100]}...",
                                "progress": 0.2,
                                "summary_preview": summary[:200],
                            },
                        )

                # 사용자 메시지 추가
                context_manager.add_message(
                    context.session_id,
                    "user",
                    context.query,
                )

                # 요약 포함된 컨텍스트 가져오기 (MongoDB fallback 포함)
                messages = await context_manager.get_context_with_summary_async(context.session_id)
            else:
                # 세션 없으면 단일 메시지
                messages = [{"role": "user", "content": context.query}]

            # Client 생성
            client = Client(model=context.model)

            # 진행 상황
            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "generating",
                    "message": "Generating response...",
                    "progress": 0.3,
                },
            )

            # 스트리밍 응답
            logger.info(f"Starting stream: model={context.model}, messages={len(messages)}")
            full_content = ""
            chunk_count = 0
            async for chunk in client.stream_chat(messages, temperature=context.temperature):
                # stream_chat은 문자열을 yield함
                if chunk:
                    chunk_count += 1
                    full_content += chunk
                    if chunk_count % 10 == 0:  # 10개마다 로깅
                        logger.debug(
                            f"Streaming chunk #{chunk_count}, accumulated: {len(full_content)} chars"
                        )
                    yield AgenticEvent(
                        type=EventType.TEXT,
                        data={
                            "tool": tool.name,
                            "content": chunk,
                            "accumulated_length": len(full_content),
                        },
                    )

            logger.info(f"Stream completed: {chunk_count} chunks, {len(full_content)} total chars")

            # 텍스트 완료
            yield AgenticEvent(
                type=EventType.TEXT_DONE,
                data={
                    "tool": tool.name,
                    "total_length": len(full_content),
                },
            )

            # 세션이 있으면 assistant 응답 저장
            if context.session_id and full_content:
                context_manager.add_message(
                    context.session_id,
                    "assistant",
                    full_content,
                )

            # 결과
            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": {
                        "content": full_content,
                        "model": context.model,
                        "has_context": context.session_id is not None,
                    },
                    "status": "completed",
                },
            )

        except Exception as e:
            logger.error(f"Chat handler error: {e}")
            raise

    async def _handle_rag(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """RAG 도구 핸들러 (MCP 서버 사용)"""
        try:
            # 진행 상황: 검색
            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "searching",
                    "message": "Searching documents...",
                    "progress": 0.2,
                },
            )

            collection_name = context.extra_params.get("collection_name", "default")

            # 진행 상황: 답변 생성
            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "generating",
                    "message": "Generating reply...",
                    "progress": 0.6,
                },
            )

            # MCP 서버의 query_rag_system 호출
            result = await self._mcp_client.call_rag_query(
                query=context.query,
                collection_name=collection_name,
                top_k=context.extra_params.get("top_k", 5),
                model=context.model,
                temperature=context.temperature,
                session_id=context.session_id,
            )

            if result.get("success"):
                # 답변 스트리밍 (청크로 나누어)
                answer = result.get("answer", "")
                if answer:
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        chunk = answer[i : i + chunk_size]
                        yield AgenticEvent(
                            type=EventType.TEXT,
                            data={
                                "tool": tool.name,
                                "content": chunk,
                            },
                        )

                # 결과
                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": {
                            "answer": answer,
                            "sources": result.get("sources", []),
                            "source_count": len(result.get("sources", [])),
                            "collection": collection_name,
                        },
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "RAG query failed"),
                    },
                )

        except Exception as e:
            logger.error(f"RAG handler error: {e}")
            raise

    async def _handle_agent(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Agent 도구 핸들러 (MCP 서버 사용)"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "initializing",
                "message": "Initializing agent...",
                "progress": 0.2,
            },
        )

        try:
            # 기본 Multi-Agent 시스템으로 단일 에이전트 실행
            system_name = context.extra_params.get("system_name", "default_agent")

            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "executing",
                    "message": "Running agent task...",
                    "progress": 0.5,
                },
            )

            # MCP 서버의 run_multiagent_task 호출
            result = await self._mcp_client.call_multiagent_run(
                system_name=system_name,
                task=context.query,
                context=context.extra_params.get("context", {}),
                session_id=context.session_id,
            )

            if result.get("success"):
                # 결과 스트리밍
                final_result = result.get("final_result", "")
                if final_result:
                    chunk_size = 50
                    for i in range(0, len(final_result), chunk_size):
                        chunk = final_result[i : i + chunk_size]
                        yield AgenticEvent(
                            type=EventType.TEXT,
                            data={
                                "tool": tool.name,
                                "content": chunk,
                            },
                        )

                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": result,
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "Agent execution failed"),
                    },
                )

        except Exception as e:
            logger.error(f"Agent handler error: {e}")
            raise

    async def _handle_multi_agent(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Multi-Agent 도구 핸들러 (MCP 서버 사용)"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "initializing",
                "message": "Initializing multi-agent system...",
                "progress": 0.2,
            },
        )

        try:
            system_name = context.extra_params.get("system_name", "default")

            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "executing",
                    "message": "Running multi-agent collaboration...",
                    "progress": 0.5,
                },
            )

            # MCP 서버의 run_multiagent_task 호출
            result = await self._mcp_client.call_multiagent_run(
                system_name=system_name,
                task=context.query,
                context=context.extra_params.get("context", {}),
                session_id=context.session_id,
            )

            if result.get("success"):
                # 에이전트 응답들 스트리밍
                agent_responses = result.get("agent_responses", [])
                for resp in agent_responses:
                    yield AgenticEvent(
                        type=EventType.TOOL_PROGRESS,
                        data={
                            "tool": tool.name,
                            "step": "agent_response",
                            "message": f"Agent '{resp.get('agent', 'unknown')}' responded",
                            "agent": resp.get("agent"),
                            "content": resp.get("content", "")[:200],
                        },
                    )

                # 최종 결과 스트리밍
                final_result = result.get("final_result", "")
                if final_result:
                    chunk_size = 50
                    for i in range(0, len(final_result), chunk_size):
                        chunk = final_result[i : i + chunk_size]
                        yield AgenticEvent(
                            type=EventType.TEXT,
                            data={
                                "tool": tool.name,
                                "content": chunk,
                            },
                        )

                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": {
                            "final_result": final_result,
                            "agent_responses": agent_responses,
                            "total_rounds": result.get("total_rounds", 0),
                            "strategy": result.get("strategy", "unknown"),
                        },
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "Multi-agent execution failed"),
                    },
                )

        except Exception as e:
            logger.error(f"Multi-agent handler error: {e}")
            raise

    async def _handle_web_search(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Web Search 도구 핸들러"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "searching",
                "message": "Searching the web...",
                "progress": 0.3,
            },
        )

        # TODO: beanllm WebSearch Facade 연동
        yield AgenticEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool": tool.name,
                "result": {"message": "Web search tool not yet implemented"},
                "status": "pending",
            },
        )

    async def _handle_code(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Code 도구 핸들러 (Chat과 유사하지만 코드 특화)"""
        # Chat 핸들러 재사용 (코드 모드 시스템 프롬프트 추가)
        async for event in self._handle_chat(context, tool):
            yield event

    async def _handle_google_drive(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Google Drive 도구 핸들러 (파일 목록, 읽기, 저장)."""
        yield self._create_progress_event(
            tool, "authenticating", "Checking Google authentication...", 0.1
        )

        try:
            access_token = await self._get_google_access_token(context, tool)
            if not access_token:
                yield self._create_google_auth_error_event(tool)
                return

            user_id = context.user_id or "default"
            yield self._create_progress_event(
                tool, "processing", "Processing Google Drive request...", 0.3
            )

            # 쿼리 분석 (읽기/학습 vs 목록 조회 vs 저장)
            query_lower = context.query.lower()

            if any(
                kw in query_lower for kw in ["읽어", "가져와", "학습", "import", "인덱싱", "rag"]
            ):
                # 파일 읽기 및 RAG 학습
                file_id = context.extra_params.get("file_id")
                if not file_id:
                    yield AgenticEvent(
                        type=EventType.ERROR,
                        data={
                            "tool": tool.name,
                            "message": "file_id가 필요합니다. tool_options에 file_id를 포함해주세요.",
                        },
                    )
                    return

                yield AgenticEvent(
                    type=EventType.TOOL_PROGRESS,
                    data={
                        "tool": tool.name,
                        "step": "reading",
                        "message": "Google Drive에서 파일을 읽는 중...",
                        "progress": 0.5,
                    },
                )

                from mcp_server.tools.google_tools import import_google_data_to_rag

                result = await import_google_data_to_rag(
                    access_token=access_token,
                    session_id=context.session_id or "default",
                    source_type="drive",
                    source_id=file_id,
                    collection_name=context.extra_params.get("collection_name"),
                )

            elif any(kw in query_lower for kw in ["목록", "리스트", "list", "보여", "파일"]):
                # 파일 목록 조회
                from mcp_server.tools.google_tools import list_google_drive_files

                result = await list_google_drive_files(
                    access_token=access_token,
                    page_size=context.extra_params.get("page_size", 10),
                )
            else:
                # 파일 저장
                from mcp_server.tools.google_tools import save_to_google_drive

                result = await save_to_google_drive(
                    filename=context.extra_params.get("filename", "beanllm_chat.txt"),
                    user_id=user_id,
                    access_token=access_token,
                    session_id=context.session_id,
                    content=context.extra_params.get("content"),
                )

            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": result,
                    "status": "completed" if result.get("success") else "error",
                },
            )

        except Exception as e:
            logger.error(f"Google Drive handler error: {e}")
            raise

    async def _handle_google_docs(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Google Docs 도구 핸들러 (문서 읽기, 내보내기)."""
        yield self._create_progress_event(
            tool, "authenticating", "Checking Google authentication...", 0.1
        )

        try:
            access_token = await self._get_google_access_token(context, tool)
            if not access_token:
                yield self._create_google_auth_error_event(tool)
                return

            user_id = context.user_id or "default"
            query_lower = context.query.lower()

            if any(
                kw in query_lower for kw in ["읽어", "가져와", "학습", "import", "인덱싱", "rag"]
            ):
                # 문서 읽기 및 RAG 학습
                doc_id = context.extra_params.get("doc_id")
                if not doc_id:
                    yield AgenticEvent(
                        type=EventType.ERROR,
                        data={
                            "tool": tool.name,
                            "message": "doc_id가 필요합니다. tool_options에 doc_id를 포함해주세요.",
                        },
                    )
                    return

                yield AgenticEvent(
                    type=EventType.TOOL_PROGRESS,
                    data={
                        "tool": tool.name,
                        "step": "reading",
                        "message": "Google Docs 문서를 읽는 중...",
                        "progress": 0.5,
                    },
                )

                from mcp_server.tools.google_tools import import_google_data_to_rag

                result = await import_google_data_to_rag(
                    access_token=access_token,
                    session_id=context.session_id or "default",
                    source_type="docs",
                    source_id=doc_id,
                    collection_name=context.extra_params.get("collection_name"),
                )
            else:
                # 문서 내보내기
                yield AgenticEvent(
                    type=EventType.TOOL_PROGRESS,
                    data={
                        "tool": tool.name,
                        "step": "creating",
                        "message": "Creating Google Docs document...",
                        "progress": 0.5,
                    },
                )

                from mcp_server.tools.google_tools import export_to_google_docs

                result = await export_to_google_docs(
                    title=context.extra_params.get("title", "beanllm Chat Export"),
                    user_id=user_id,
                    access_token=access_token,
                    session_id=context.session_id,
                    content=context.extra_params.get("content"),
                )

            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": result,
                    "status": "completed" if result.get("success") else "error",
                },
            )

        except Exception as e:
            logger.error(f"Google Docs handler error: {e}")
            raise

    async def _handle_google_gmail(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Gmail 도구 핸들러 (이메일 전송)."""
        yield self._create_progress_event(
            tool, "authenticating", "Checking Google authentication...", 0.1
        )

        try:
            access_token = await self._get_google_access_token(context, tool)
            if not access_token:
                yield self._create_google_auth_error_event(tool)
                return

            user_id = context.user_id or "default"
            recipient_email = context.extra_params.get("recipient_email")
            if not recipient_email:
                # 쿼리에서 이메일 추출 시도
                import re

                email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", context.query)
                if email_match:
                    recipient_email = email_match.group()
                else:
                    yield AgenticEvent(
                        type=EventType.ERROR,
                        data={
                            "tool": tool.name,
                            "message": "Specify recipient email.",
                        },
                    )
                    return

            yield AgenticEvent(
                type=EventType.TOOL_PROGRESS,
                data={
                    "tool": tool.name,
                    "step": "sending",
                    "message": f"Sending email to {recipient_email}...",
                    "progress": 0.5,
                },
            )

            from mcp_server.tools.google_tools import share_via_gmail

            result = await share_via_gmail(
                recipient_email=recipient_email,
                subject=context.extra_params.get("subject", "beanllm Chat History"),
                user_id=user_id,
                access_token=access_token,
                session_id=context.session_id,
                content=context.extra_params.get("content"),
                message=context.extra_params.get("message"),
            )

            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": result,
                    "status": "completed" if result.get("success") else "error",
                },
            )

        except Exception as e:
            logger.error(f"Gmail handler error: {e}")
            raise

    async def _handle_google_calendar(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Google Calendar 도구 핸들러 (이벤트 생성/조회)."""
        yield self._create_progress_event(
            tool, "authenticating", "Checking Google authentication...", 0.1
        )

        try:
            access_token = await self._get_google_access_token(context, tool)
            if not access_token:
                yield self._create_google_auth_error_event(tool)
                return

            # TODO: Google Calendar API 연동 (이벤트 생성/조회)
            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": {
                        "message": "Google Calendar is not available yet.",
                        "authenticated": True,
                    },
                    "status": "pending",
                },
            )

        except Exception as e:
            logger.error(f"Google Calendar handler error: {e}")
            raise

    async def _handle_google_sheets(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Google Sheets 도구 핸들러 (스프레드시트 읽기/내보내기)."""
        yield self._create_progress_event(
            tool, "authenticating", "Checking Google authentication...", 0.1
        )

        try:
            access_token = await self._get_google_access_token(context, tool)
            if not access_token:
                yield self._create_google_auth_error_event(tool)
                return

            query_lower = context.query.lower()

            if any(
                kw in query_lower for kw in ["읽어", "가져와", "학습", "import", "인덱싱", "rag"]
            ):
                # 스프레드시트 읽기 및 RAG 학습
                spreadsheet_id = context.extra_params.get("spreadsheet_id")
                if not spreadsheet_id:
                    yield AgenticEvent(
                        type=EventType.ERROR,
                        data={
                            "tool": tool.name,
                            "message": "spreadsheet_id가 필요합니다. tool_options에 spreadsheet_id를 포함해주세요.",
                        },
                    )
                    return

                yield AgenticEvent(
                    type=EventType.TOOL_PROGRESS,
                    data={
                        "tool": tool.name,
                        "step": "reading",
                        "message": "Google Sheets 데이터를 읽는 중...",
                        "progress": 0.5,
                    },
                )

                from mcp_server.tools.google_tools import import_google_data_to_rag

                result = await import_google_data_to_rag(
                    access_token=access_token,
                    session_id=context.session_id or "default",
                    source_type="sheets",
                    source_id=spreadsheet_id,
                    sheet_name=context.extra_params.get("sheet_name"),
                    collection_name=context.extra_params.get("collection_name"),
                )
            else:
                # TODO: Google Sheets 내보내기 (시트 생성/데이터 입력)
                result = {
                    "success": False,
                    "message": "Google Sheets 내보내기는 아직 지원되지 않습니다. 읽기/학습 기능만 사용 가능합니다.",
                    "authenticated": True,
                }

            yield AgenticEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool": tool.name,
                    "result": result,
                    "status": "completed" if result.get("success") else "pending",
                },
            )

        except Exception as e:
            logger.error(f"Google Sheets handler error: {e}")
            raise

    async def _handle_audio(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Audio 도구 핸들러 (MCP 서버 사용)"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "transcribing",
                "message": "Transcribing audio...",
                "progress": 0.3,
            },
        )

        try:
            audio_path = context.extra_params.get("audio_path")
            if not audio_path:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": "audio_path is required.",
                    },
                )
                return

            # MCP 서버의 transcribe_audio 호출
            result = await self._mcp_client.call_audio_transcribe(
                audio_path=audio_path,
                engine=context.extra_params.get("engine", "whisper"),
                language=context.extra_params.get("language"),
                session_id=context.session_id,
            )

            if result.get("success"):
                # 전사 결과 스트리밍
                text = result.get("text", "")
                if text:
                    chunk_size = 100
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i : i + chunk_size]
                        yield AgenticEvent(
                            type=EventType.TEXT,
                            data={
                                "tool": tool.name,
                                "content": chunk,
                            },
                        )

                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": {
                            "text": text,
                            "language": result.get("language"),
                            "confidence": result.get("confidence"),
                            "duration_seconds": result.get("duration_seconds"),
                        },
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "Audio transcription failed"),
                    },
                )

        except Exception as e:
            logger.error(f"Audio handler error: {e}")
            raise

    async def _handle_vision(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Vision 도구 핸들러"""
        yield AgenticEvent(
            type=EventType.TOOL_RESULT,
            data={
                "tool": tool.name,
                "result": {"message": "Vision tool not yet implemented"},
                "status": "pending",
            },
        )

    async def _handle_ocr(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """OCR 도구 핸들러 (MCP 서버 사용)"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "processing",
                "message": "Extracting text from image...",
                "progress": 0.3,
            },
        )

        try:
            image_path = context.extra_params.get("image_path")
            if not image_path:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": "image_path is required.",
                    },
                )
                return

            # MCP 서버의 recognize_text_ocr 호출
            result = await self._mcp_client.call_ocr(
                image_path=image_path,
                engine=context.extra_params.get("engine", "tesseract"),
                language=context.extra_params.get("language", "eng"),
                session_id=context.session_id,
            )

            if result.get("success"):
                # OCR 결과 스트리밍
                text = result.get("text", "")
                if text:
                    chunk_size = 100
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i : i + chunk_size]
                        yield AgenticEvent(
                            type=EventType.TEXT,
                            data={
                                "tool": tool.name,
                                "content": chunk,
                            },
                        )

                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": {
                            "text": text,
                            "confidence": result.get("confidence"),
                            "engine": result.get("engine"),
                        },
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "OCR failed"),
                    },
                )

        except Exception as e:
            logger.error(f"OCR handler error: {e}")
            raise

    async def _handle_knowledge_graph(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Knowledge Graph 도구 핸들러 (MCP 서버 사용)"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "querying",
                "message": "Querying knowledge graph...",
                "progress": 0.3,
            },
        )

        try:
            graph_name = context.extra_params.get("graph_name", "default")

            # MCP 서버의 query_knowledge_graph 호출
            result = await self._mcp_client.call_kg_query(
                query=context.query,
                graph_name=graph_name,
                model=context.model,
                max_depth=context.extra_params.get("max_depth", 2),
                session_id=context.session_id,
            )

            if result.get("success"):
                # 답변 스트리밍
                answer = result.get("answer", "")
                if answer:
                    chunk_size = 50
                    for i in range(0, len(answer), chunk_size):
                        chunk = answer[i : i + chunk_size]
                        yield AgenticEvent(
                            type=EventType.TEXT,
                            data={
                                "tool": tool.name,
                                "content": chunk,
                            },
                        )

                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": {
                            "answer": answer,
                            "entities": result.get("entities", []),
                            "relations": result.get("relations", []),
                            "graph_name": graph_name,
                        },
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "Knowledge graph query failed"),
                    },
                )

        except Exception as e:
            logger.error(f"Knowledge Graph handler error: {e}")
            raise

    async def _handle_evaluation(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Evaluation 도구 핸들러 (MCP 서버 사용)"""
        yield AgenticEvent(
            type=EventType.TOOL_PROGRESS,
            data={
                "tool": tool.name,
                "step": "evaluating",
                "message": "Evaluating model...",
                "progress": 0.3,
            },
        )

        try:
            model = context.extra_params.get("eval_model", context.model)
            evaluation_type = context.extra_params.get("evaluation_type", "answer_relevancy")

            # MCP 서버의 evaluate_model 호출
            result = await self._mcp_client.call_evaluation(
                model=model,
                evaluation_type=evaluation_type,
                test_data=context.extra_params.get("test_data"),
                session_id=context.session_id,
            )

            if result.get("success"):
                yield AgenticEvent(
                    type=EventType.TOOL_RESULT,
                    data={
                        "tool": tool.name,
                        "result": {
                            "model": result.get("model"),
                            "evaluation_type": result.get("evaluation_type"),
                            "overall_score": result.get("overall_score"),
                            "metric_scores": result.get("metric_scores", {}),
                        },
                        "status": "completed",
                    },
                )
            else:
                yield AgenticEvent(
                    type=EventType.ERROR,
                    data={
                        "tool": tool.name,
                        "message": result.get("error", "Evaluation failed"),
                    },
                )

        except Exception as e:
            logger.error(f"Evaluation handler error: {e}")
            raise

    async def _handle_unknown(
        self, context: OrchestratorContext, tool: Tool
    ) -> AsyncGenerator[AgenticEvent, None]:
        """Unknown 도구 핸들러"""
        yield AgenticEvent(
            type=EventType.ERROR,
            data={
                "tool": tool.name,
                "message": f"Unknown tool handler: {tool.name}",
            },
        )

    # ===========================================
    # Parallel Execution
    # ===========================================

    async def execute_parallel(
        self, context: OrchestratorContext
    ) -> AsyncGenerator[AgenticEvent, None]:
        """
        병렬 도구 실행 및 결과 스트리밍

        여러 도구를 동시에 실행하고 각 작업별 진행 상황을 실시간으로 스트리밍합니다.

        Args:
            context: 실행 컨텍스트

        Yields:
            AgenticEvent: SSE 이벤트 (병렬 진행 상황 포함)
        """
        try:
            # 1. Intent 이벤트
            yield AgenticEvent(
                type=EventType.INTENT,
                data={
                    "primary_intent": context.intent.primary_intent.value,
                    "confidence": context.intent.confidence,
                    "secondary_intents": [i.value for i in context.intent.secondary_intents],
                },
            )

            # 2. 도구가 없으면 기본 Chat으로 폴백
            if not context.selected_tools:
                chat_tool = self._registry.get_best_tool_for_intent(IntentType.CHAT)
                if chat_tool:
                    context.selected_tools = [chat_tool]
                else:
                    yield AgenticEvent(type=EventType.ERROR, data={"message": "No tools available"})
                    return

            available_tools = [t for t in context.selected_tools if t.is_available]

            if not available_tools:
                yield AgenticEvent(
                    type=EventType.ERROR, data={"message": "No available tools to execute"}
                )
                return

            # 3. 병렬 작업 시작 이벤트
            yield AgenticEvent(
                type=EventType.PARALLEL_START,
                data={
                    "tools": [t.tool.name for t in available_tools],
                    "count": len(available_tools),
                    "message": f"{len(available_tools)}개 작업을 병렬로 실행합니다.",
                },
            )

            # 4. 각 도구에 대한 진행 상황 수집을 위한 큐
            event_queue: asyncio.Queue[AgenticEvent] = asyncio.Queue()
            results: Dict[str, Any] = {}
            errors: Dict[str, str] = {}

            async def run_tool_with_progress(tool_check: ToolCheckResult, index: int):
                """개별 도구 실행 및 진행 상황 큐에 추가"""
                tool = tool_check.tool
                tool_name = tool.name

                try:
                    # 시작 이벤트
                    await event_queue.put(
                        AgenticEvent(
                            type=EventType.PARALLEL_PROGRESS,
                            data={
                                "task_index": index,
                                "tool": tool_name,
                                "step": "starting",
                                "message": f"Task {index + 1} started",
                                "progress": 0.0,
                                "total_tasks": len(available_tools),
                            },
                        )
                    )

                    # 도구 핸들러 실행
                    handler = self._tool_handlers.get(tool_name, self._handle_unknown)
                    tool_result = None

                    async for event in handler(context, tool):
                        # 진행 상황 이벤트에 task_index 추가하여 큐에 전송
                        if event.type == EventType.TOOL_PROGRESS:
                            await event_queue.put(
                                AgenticEvent(
                                    type=EventType.PARALLEL_PROGRESS,
                                    data={
                                        "task_index": index,
                                        "tool": tool_name,
                                        **event.data,
                                        "total_tasks": len(available_tools),
                                    },
                                )
                            )
                        elif event.type == EventType.TOOL_RESULT:
                            tool_result = event.data.get("result")
                            await event_queue.put(
                                AgenticEvent(
                                    type=EventType.PARALLEL_PROGRESS,
                                    data={
                                        "task_index": index,
                                        "tool": tool_name,
                                        "step": "completed",
                                        "message": f"Task {index + 1} completed",
                                        "progress": 1.0,
                                        "total_tasks": len(available_tools),
                                    },
                                )
                            )
                        elif event.type in (EventType.TEXT, EventType.TEXT_DONE):
                            # 텍스트 스트리밍 이벤트도 전달
                            event.data["task_index"] = index
                            await event_queue.put(event)
                        elif event.type == EventType.ERROR:
                            errors[tool_name] = event.data.get("message", "Unknown error")
                            await event_queue.put(event)

                    results[tool_name] = tool_result

                except Exception as e:
                    logger.error(f"Parallel tool error: {tool_name} - {e}")
                    errors[tool_name] = str(e)
                    await event_queue.put(
                        AgenticEvent(
                            type=EventType.ERROR,
                            data={
                                "task_index": index,
                                "tool": tool_name,
                                "message": str(e),
                            },
                        )
                    )

            # 5. 병렬 실행 시작
            tasks = [
                asyncio.create_task(run_tool_with_progress(tool_check, i))
                for i, tool_check in enumerate(available_tools)
            ]

            # 완료 신호를 위한 태스크
            async def wait_for_completion():
                await asyncio.gather(*tasks, return_exceptions=True)
                await event_queue.put(None)  # 종료 신호

            asyncio.create_task(wait_for_completion())

            # 6. 이벤트 스트리밍
            completed_count = 0
            while True:
                event = await event_queue.get()
                if event is None:
                    break  # 모든 작업 완료

                yield event

                # 완료된 작업 카운트
                if (
                    event.type == EventType.PARALLEL_PROGRESS
                    and event.data.get("step") == "completed"
                ):
                    completed_count += 1

            # 7. 병렬 작업 완료 이벤트
            yield AgenticEvent(
                type=EventType.PARALLEL_DONE,
                data={
                    "success": len(errors) == 0,
                    "completed_count": completed_count,
                    "total_count": len(available_tools),
                    "results": results,
                    "errors": errors if errors else None,
                },
            )

            # 8. 전체 완료 이벤트
            yield AgenticEvent(
                type=EventType.DONE,
                data={
                    "success": len(errors) == 0,
                    "tool_count": len(available_tools),
                    "parallel": True,
                    "results": [
                        {"tool": name, "result": result} for name, result in results.items()
                    ],
                },
            )

        except Exception as e:
            logger.error(f"Parallel orchestrator error: {e}")
            logger.error(traceback.format_exc())
            yield AgenticEvent(
                type=EventType.ERROR,
                data={
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )


# Singleton instance
orchestrator = AgenticOrchestrator()
