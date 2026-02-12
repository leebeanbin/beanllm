"""
Chat Router

Chat endpoints for LLM conversation including:
- Basic chat
- Agentic chat (auto-routing to appropriate tools)
- Tool status
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from common import get_client
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from services import (
    IntentType,
    OrchestratorContext,
    intent_classifier,
    orchestrator,
    tool_registry,
)
from services.intent_classifier import IntentResult
from services.orchestrator import EventType as AgenticEventType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])


# ===========================================
# Request/Response Models
# ===========================================


class ChatMessage(BaseModel):
    """Chat message"""

    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Basic chat request"""

    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model: str = Field(default="qwen2.5:0.5b", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=32000)
    stream: bool = Field(default=True, description="Enable streaming")


class AgenticChatRequest(BaseModel):
    """Agentic chat request with auto-routing"""

    messages: List[ChatMessage] = Field(..., description="Chat messages")
    model: str = Field(default="qwen2.5:0.5b", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=32000)
    session_id: Optional[str] = Field(None, description="Session ID for message history")
    user_id: Optional[str] = Field(None, description="User ID for logging")
    # Intent overrides
    force_intent: Optional[str] = Field(
        None, description="Force specific intent (skip classification)"
    )
    # Tool options
    tool_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Tool-specific options (e.g., collection_name for RAG)"
    )
    # Parallel execution
    parallel: bool = Field(
        default=False, description="Execute multiple tools in parallel (병렬 처리)"
    )
    # 제안 단계: 유저가 챗으로 승인/수정/직접 스펙 보낸 값 (챗 전용)
    proposal_action: Optional[str] = Field(
        None, description="approved | modified | custom_spec (from user reply to proposal)"
    )
    proposal_pipeline: Optional[List[str]] = Field(
        None, description="For custom_spec: ordered tool names e.g. ['rag','kg','chat']"
    )
    # Human-in-the-loop: 승인 대기 시 True, 재개 요청 시 run_id + approval_response
    require_approval: bool = Field(
        default=False, description="Pause before each tool for user approval"
    )
    run_id: Optional[str] = Field(None, description="Resume: run_id from stream_paused event")
    approval_response: Optional[Dict[str, Any]] = Field(
        None, description="Resume: { run_id, action: 'run'|'cancel'|'change_tool' }"
    )


class IntentClassifyRequest(BaseModel):
    """Intent classification request"""

    query: str = Field(..., description="Query to classify")
    available_intents: Optional[List[str]] = Field(None, description="Limit to these intents")


class IntentClassifyResponse(BaseModel):
    """Intent classification response"""

    primary_intent: str
    confidence: float
    secondary_intents: List[str]
    extracted_entities: Dict[str, Any]
    reasoning: Optional[str]


class ToolStatusResponse(BaseModel):
    """Tool status response"""

    name: str
    description: str
    description_ko: str
    status: str
    is_available: bool
    missing_keys: List[str]
    missing_packages: List[str]
    intent_types: List[str]


# ===========================================
# Basic Chat Endpoints
# ===========================================


@router.post("")
async def chat(request: ChatRequest):
    """
    Basic chat endpoint (non-streaming)

    Simple LLM chat without tool routing.
    """
    try:
        client = get_client(request.model)

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        response = await client.chat(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return {
            "content": response.content,
            "model": request.model,
            "usage": response.usage if hasattr(response, "usage") else None,
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint

    Returns SSE stream of chat chunks.
    """

    async def generate():
        try:
            client = get_client(request.model)
            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            async for chunk in client.stream(
                messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                if hasattr(chunk, "content") and chunk.content:
                    yield f"data: {chunk.content}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===========================================
# Agentic Chat Endpoints
# ===========================================


@router.post("/agentic")
async def agentic_chat(request: AgenticChatRequest, http_request: Request):
    """
    Agentic chat endpoint with auto-routing

    Automatically classifies user intent and routes to appropriate tools.
    Returns SSE stream of events including:
    - intent: Classification result
    - tool_select: Selected tools
    - tool_start: Tool execution started
    - tool_progress: Execution progress
    - tool_result: Tool results
    - text: Streaming text chunks
    - done: Completion

    Example:
        POST /api/chat/agentic
        {
            "messages": [{"role": "user", "content": "문서에서 AI에 대해 찾아줘"}],
            "model": "qwen2.5:0.5b"
        }

        Response (SSE):
        data: {"type": "intent", "data": {"primary_intent": "rag", ...}}
        data: {"type": "tool_select", "data": {"tools": ["rag"]}}
        data: {"type": "tool_progress", "data": {"step": "searching", ...}}
        data: {"type": "text", "data": {"content": "AI는..."}}
        data: {"type": "done", "data": {"success": true}}
    """
    request_id = http_request.headers.get("X-Request-ID") or str(uuid.uuid4())

    async def generate():
        try:
            # Human-in-the-loop 재개: approval_response에 run_id + action=="run" 이면 Redis에서 복원 후 execute 재개
            if (
                request.approval_response
                and request.approval_response.get("action") == "run"
                and request.approval_response.get("run_id")
            ):
                run_id = request.approval_response["run_id"]
                try:
                    import json

                    from beanllm.infrastructure.distributed.redis.client import get_redis_client

                    redis = get_redis_client()
                    if redis:
                        raw = await redis.get(f"run:approval:{run_id}")
                        if raw:
                            state = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                            intent_result = IntentResult(
                                primary_intent=IntentType(state["primary_intent"]),
                                confidence=state.get("confidence", 0.8),
                                secondary_intents=[],
                                extracted_entities={},
                                reasoning="Resumed from approval",
                            )
                            selected_tools = []
                            for name in state.get("tool_names", []):
                                tool = tool_registry.get_tool(name)
                                if tool:
                                    selected_tools.append(tool_registry.check_requirements(tool))
                            if selected_tools:
                                context = OrchestratorContext(
                                    query=state["query"],
                                    intent=intent_result,
                                    selected_tools=selected_tools,
                                    model=state.get("model", request.model),
                                    temperature=state.get("temperature", request.temperature),
                                    max_tokens=state.get("max_tokens", request.max_tokens),
                                    session_id=state.get("session_id") or request.session_id,
                                    user_id=state.get("user_id") or request.user_id,
                                    extra_params=state.get("extra_params")
                                    or request.tool_options
                                    or {},
                                    run_id=run_id,
                                    require_approval=False,  # 재개 후에는 대기 없이 남은 도구 실행
                                )
                                start_from = int(state.get("tool_index", 0))
                                logger.info(
                                    f"Resuming from run_id={run_id}, tool_index={start_from}"
                                )
                                async for event in orchestrator.execute(
                                    context, start_from_tool_index=start_from
                                ):
                                    yield event.to_sse()
                                    if event.type == AgenticEventType.DONE and event.data.get(
                                        "usage"
                                    ):
                                        try:
                                            from monitoring.middleware import ChatMonitoringMixin

                                            u = event.data["usage"]
                                            await ChatMonitoringMixin.log_chat_response(
                                                request_id=request_id,
                                                model=context.model or "default",
                                                response_content="",
                                                input_tokens=u.get("input_tokens"),
                                                output_tokens=u.get("output_tokens"),
                                            )
                                        except Exception as e:
                                            logger.debug(
                                                f"Failed to log agentic token metrics: {e}"
                                            )
                                return
                except Exception as e:
                    logger.warning(f"Resume from approval failed: {e}")
                yield 'data: {"type": "error", "data": {"message": "Resume failed or run expired"}}\n\n'
                return

            logger.info(
                f"Agentic chat request received: model={request.model}, messages={len(request.messages)}, force_intent={request.force_intent}"
            )
            user_messages = [m for m in request.messages if m.role == "user"]
            if not user_messages:
                logger.warning("No user message in request")
                yield 'data: {"type": "error", "data": {"message": "No user message"}}\n\n'
                return

            query = user_messages[-1].content
            logger.info(f"Processing query: {query[:100]}...")

            # 1. Intent 분류 (또는 강제 지정)
            if request.force_intent:
                try:
                    logger.info(f"Forcing intent: {request.force_intent}")
                    primary_intent = IntentType(request.force_intent)
                    intent_result = await intent_classifier.classify(
                        query, available_intents=[primary_intent]
                    )
                    logger.info(
                        f"Intent forced: {intent_result.primary_intent.value}, confidence: {intent_result.confidence}"
                    )
                except ValueError as e:
                    logger.error(f"Invalid intent: {request.force_intent}, error: {e}")
                    logger.info("Falling back to automatic classification")
                    # 유효하지 않은 intent면 자동 분류로 폴백
                    intent_result = await intent_classifier.classify(query)
                    logger.info(
                        f"Intent classified (fallback): {intent_result.primary_intent.value}, confidence: {intent_result.confidence}"
                    )
            else:
                logger.info("Classifying intent automatically...")
                intent_result = await intent_classifier.classify(query)
                logger.info(
                    f"Intent classified: {intent_result.primary_intent.value}, confidence: {intent_result.confidence}"
                )

            # 2. 도구 선택
            logger.info("Selecting tools...")
            selected_tools = []
            best_tool = tool_registry.get_best_tool_for_intent(
                intent_result.primary_intent, only_available=True
            )
            if best_tool:
                selected_tools.append(best_tool)
                logger.info(f"Selected tool: {best_tool.tool.name}")

            # Secondary intents에서도 도구 추가 (선택적)
            for secondary_intent in intent_result.secondary_intents[:1]:  # 최대 1개
                secondary_tool = tool_registry.get_best_tool_for_intent(
                    secondary_intent, only_available=True
                )
                if secondary_tool and secondary_tool.tool.name not in [
                    t.tool.name for t in selected_tools
                ]:
                    selected_tools.append(secondary_tool)
                    logger.info(f"Added secondary tool: {secondary_tool.tool.name}")

            if not selected_tools:
                logger.warning("No tools selected, will fallback to chat")

            # 3. Orchestrator 컨텍스트 생성
            logger.info(f"Creating orchestrator context with {len(selected_tools)} tools")
            run_id = request.run_id or str(uuid.uuid4())
            context = OrchestratorContext(
                query=query,
                intent=intent_result,
                selected_tools=selected_tools,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                session_id=request.session_id,
                user_id=request.user_id,
                extra_params=request.tool_options or {},
                proposal_action=request.proposal_action,
                proposal_pipeline=request.proposal_pipeline,
                run_id=run_id,
                require_approval=request.require_approval,
            )

            # 4. 실행 및 스트리밍 (병렬 또는 순차)
            logger.info(f"Starting execution (parallel={request.parallel})")
            event_count = 0
            if request.parallel and len(selected_tools) > 1:
                # 병렬 처리: 여러 도구를 동시에 실행
                async for event in orchestrator.execute_parallel(context):
                    event_count += 1
                    logger.debug(f"Yielding event #{event_count}: {event.type.value}")
                    yield event.to_sse()
            else:
                # 순차 처리: 도구를 하나씩 실행
                async for event in orchestrator.execute(context):
                    event_count += 1
                    logger.debug(f"Yielding event #{event_count}: {event.type.value}")
                    yield event.to_sse()
                    # DONE 시 usage 있으면 Redis 토큰 메트릭 기록
                    if event.type == AgenticEventType.DONE and event.data.get("usage"):
                        try:
                            from monitoring.middleware import ChatMonitoringMixin

                            u = event.data["usage"]
                            await ChatMonitoringMixin.log_chat_response(
                                request_id=request_id,
                                model=request.model or "default",
                                response_content="",
                                input_tokens=u.get("input_tokens"),
                                output_tokens=u.get("output_tokens"),
                            )
                        except Exception as e:
                            logger.debug(f"Failed to log agentic token metrics: {e}")

            logger.info(f"Execution completed. Total events: {event_count}")

        except Exception as e:
            logger.error(f"Agentic chat error: {e}", exc_info=True)
            # Never expose traceback to client
            safe_msg = str(e).replace('"', '\\"')
            yield f'data: {{"type": "error", "data": {{"message": "{safe_msg}"}}}}\n\n'

        # Always send done signal for SSE termination
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/classify")
async def classify_intent(request: IntentClassifyRequest) -> IntentClassifyResponse:
    """
    Classify user intent

    Returns intent classification without executing any tools.
    """
    try:
        available = None
        if request.available_intents:
            available = [IntentType(i) for i in request.available_intents]

        result = await intent_classifier.classify(request.query, available_intents=available)

        return IntentClassifyResponse(
            primary_intent=result.primary_intent.value,
            confidence=result.confidence,
            secondary_intents=[i.value for i in result.secondary_intents],
            extracted_entities=result.extracted_entities,
            reasoning=result.reasoning,
        )

    except Exception as e:
        logger.error(f"Classify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# Tool Status Endpoints
# ===========================================


@router.get("/tools")
async def list_tools() -> List[ToolStatusResponse]:
    """
    List all available tools with status

    Returns each tool's availability status including missing API keys/packages.
    """
    try:
        results = tool_registry.get_available_tools()

        return [
            ToolStatusResponse(
                name=r.tool.name,
                description=r.tool.description,
                description_ko=r.tool.description_ko,
                status=r.status.value,
                is_available=r.is_available,
                missing_keys=r.missing_keys,
                missing_packages=r.missing_packages,
                intent_types=[it.value for it in r.tool.intent_types],
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"List tools error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/{tool_name}")
async def get_tool_status(tool_name: str) -> ToolStatusResponse:
    """
    Get specific tool status
    """
    try:
        tool = tool_registry.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        check = tool_registry.check_requirements(tool)

        return ToolStatusResponse(
            name=check.tool.name,
            description=check.tool.description,
            description_ko=check.tool.description_ko,
            status=check.status.value,
            is_available=check.is_available,
            missing_keys=check.missing_keys,
            missing_packages=check.missing_packages,
            intent_types=[it.value for it in check.tool.intent_types],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get tool error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intents")
async def list_intents() -> List[Dict[str, Any]]:
    """
    List all supported intents
    """
    return [
        {
            "value": intent.value,
            "name": intent.name,
            "tools": [t.name for t in tool_registry.get_tools_for_intent(intent)],
        }
        for intent in IntentType
    ]
