"""
beanllm Playground Backend - FastAPI

Complete working backend for all 9 beanllm features
"""

import asyncio
import sys
import os
import logging
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    UploadFile,
    File,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logging.info(f"✅ Loaded .env file from {env_path}")
    else:
        logging.info(f"ℹ️  .env file not found at {env_path}, using environment variables")
except ImportError:
    logging.warning("⚠️  python-dotenv not installed, .env file will not be loaded")


class RequestIDFilter(logging.Filter):
    """Request ID를 로그에 추가하는 필터"""

    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = getattr(record, "request_id", "N/A")
        return True


class SafeFormatter(logging.Formatter):
    """request_id가 없어도 안전하게 처리하는 포맷터"""

    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "N/A"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(
    SafeFormatter("%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s")
)
handler.addFilter(RequestIDFilter())

logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

logger = logging.getLogger(__name__)

from monitoring.middleware import MonitoringMiddleware, ChatMonitoringMixin

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from common import (
    get_ollama_model_name_for_chat,
    track_downloaded_model,
    get_downloaded_models,
    _downloaded_models,
)


from beanllm import Client
from beanllm.facade.core.chain_facade import Chain

app = FastAPI(
    title="beanllm Playground API",
    description="Complete backend for all beanllm features",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_DISTRIBUTED = os.getenv("USE_DISTRIBUTED", "false").lower() == "true"
USE_REDIS_MONITORING = (
    os.getenv("USE_REDIS_MONITORING", "true").lower() == "true"
    )
app.add_middleware(
    MonitoringMiddleware,
    enable_kafka=USE_DISTRIBUTED,
    enable_redis=USE_REDIS_MONITORING,
)

from database import get_mongodb_client, close_mongodb_connection, ping_mongodb
from routers.history_router import router as chat_history_router

app.include_router(chat_history_router)


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting beanllm Playground Backend...")
    if await ping_mongodb():
        logger.info("✅ MongoDB connected successfully")
        from database import create_session_indexes
        await create_session_indexes()
    else:
        if not os.getenv("MONGODB_URI"):
            logger.info("ℹ️  MongoDB not configured (MONGODB_URI unset) - chat history will not be saved")
        else:
            logger.warning("⚠️  MongoDB not available - chat history will not be saved")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down beanllm Playground Backend...")
    await close_mongodb_connection()
    logger.info("✅ Shutdown complete")


from common import (
    get_client,
    get_kg,
    get_web_search,
    get_rag_debugger,
    get_optimizer,
    get_multi_agent,
    get_orchestrator,
    get_vision_rag,
    get_audio_rag,
    get_evaluator,
    get_finetuning,
    get_rag_chain,
    set_rag_chain,
    _rag_chains,
)

_chains: Dict[str, Chain] = {}
active_connections: Dict[str, WebSocket] = {}

# Import schemas from centralized location
from schemas import (
    Message,
    ChatRequest,
    BuildGraphRequest,
    QueryGraphRequest,
    GraphRAGRequest,
    RAGBuildRequest,
    RAGQueryRequest,
    RAGDebugRequest,
    AgentRequest,
    WebSearchRequest,
    OptimizeRequest,
    MultiAgentRequest,
    WorkflowRequest,
    ChainRequest,
    VisionRAGBuildRequest,
    VisionRAGQueryRequest,
    AudioTranscribeRequest,
    AudioSynthesizeRequest,
    AudioRAGRequest,
    EvaluationRequest,
    FineTuningCreateRequest,
    FineTuningStatusRequest,
)


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "beanllm-playground-api",
        "version": "1.0.0",
        "features": [
            "chat",
            "knowledge_graph",
            "rag",
            "agent",
            "web_search",
            "rag_debug",
            "optimizer",
            "multi_agent",
            "orchestrator",
        ],
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        client = get_client()
        return {
            "status": "healthy",
            "client_initialized": _client is not None,
            "facades": {
                "kg": _kg is not None,
                "rag_chains": len(_rag_chains),
                "web_search": _web_search is not None,
                "rag_debugger": _rag_debugger is not None,
                "optimizer": _optimizer is not None,
                "multi_agent": _multi_agent is not None,
                "orchestrator": _orchestrator is not None,
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }




@app.get("/api/config/providers")
async def get_active_providers():
    """
    활성화된 Provider 목록 반환 (✅ EnvConfig 활용)

    Returns:
        {
            "providers": ["openai", "anthropic", "ollama"],
            "config": {
                "openai_api_key": "***MASKED***",
                "anthropic_api_key": None,
                ...
            }
        }
    """
    try:
        from beanllm.utils.config import EnvConfig

        return {
            "providers": EnvConfig.get_active_providers(),
            "config": EnvConfig.get_safe_config_dict(),
        }
    except Exception as e:
        logger.error(f"Failed to get active providers: {e}")
        return {
            "providers": ["ollama"],
            "config": {},
        }


@app.get("/api/config/models")
async def get_available_models():
    """
    활성화된 Provider별 사용 가능한 모델 목록

    Returns:
        {
            "openai": ["gpt-4o", "gpt-4o-mini", ...],
            "anthropic": ["claude-sonnet-4", ...],
            "ollama": ["qwen2.5:0.5b", ...]
        }
    """
    try:
        from beanllm.utils.config import EnvConfig

        available_models = {}

        # OpenAI 모델
        if EnvConfig.is_provider_available("openai"):
            available_models["openai"] = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]

        # Anthropic 모델
        if EnvConfig.is_provider_available("anthropic"):
            available_models["anthropic"] = [
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-haiku-4-20250514",
            ]

        # Google 모델
        if EnvConfig.is_provider_available("google"):
            available_models["google"] = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-1.5-pro",
            ]

        try:
            from beanllm.providers.ollama_provider import OllamaProvider
            ollama = OllamaProvider()
            ollama_models = await ollama.list_models()
            # 실제 설치된 모델만 반환
            if ollama_models:
                available_models["ollama"] = [m["name"] if isinstance(m, dict) else str(m) for m in ollama_models]
            else:
                available_models["ollama"] = []
                logger.info("No Ollama models installed.")
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            available_models["ollama"] = []

        return available_models

    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return {"ollama": []}




@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request = None):
    """
    Main chat endpoint - routes to different assistants
    """
    request_id = http_request.headers.get("X-Request-ID") if http_request else str(uuid.uuid4())
    chat_start_time = time.time()

    try:
        messages = []
        for i, msg in enumerate(request.messages):
            is_last_user = i == len(request.messages) - 1 and msg.role == "user"
            has_images = is_last_user and request.images and len(request.images) > 0
            has_files = is_last_user and request.files and len(request.files) > 0

            if has_images or has_files:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if has_images:
                    import base64

                    for img_base64 in request.images:
                        try:
                            if img_base64.startswith("data:image"):
                                image_url = img_base64
                            else:
                                image_url = f"data:image/png;base64,{img_base64}"
                            content.append({"type": "image_url", "image_url": {"url": image_url}})
                        except Exception as e:
                            logger.warning(f"Failed to process image: {e}")
                            continue

                if has_files:
                    file_info = "\n\nAttached files:\n" + "\n".join(
                        [
                            f"- {f.get('name', 'Unknown')} ({f.get('type', 'Unknown type')})"
                            for f in request.files
                        ]
                    )
                    if content and content[0].get("type") == "text":
                        content[0]["text"] += file_info
                    else:
                        content.insert(0, {"type": "text", "text": file_info})

                messages.append({"role": msg.role, "content": content})
            else:
                messages.append({"role": msg.role, "content": msg.content})

        if request.model:
            from beanllm.infrastructure.registry import get_model_registry

            registry = get_model_registry()
            model_info = None
            try:
                model_info = registry.get_model_info(request.model)
            except:
                pass

            model_name_lower = request.model.lower()
            use_ollama = False
            ollama_model_name = None

            if model_info and model_info.provider == "ollama":
                try:
                    from beanllm.providers.ollama_provider import OllamaProvider

                    ollama_provider = OllamaProvider()
                    try:
                        health = await ollama_provider.health_check()
                        logger.info(f"[DEBUG] Ollama health check: {health}")
                    except Exception as health_error:
                        logger.warning(
                            f"[DEBUG] Ollama health check failed: {health_error}", exc_info=True
                        )
                    try:
                        logger.info(f"[DEBUG] Calling ollama_provider.list_models()...")
                        installed_models = await ollama_provider.list_models()
                        logger.info(
                            f"[DEBUG] list_models() returned: {installed_models} (type: {type(installed_models)}, count: {len(installed_models)})"
                        )
                    except Exception as list_error:
                        logger.error(
                            f"[DEBUG] list_models() raised exception: {list_error}", exc_info=True
                        )
                        try:
                            logger.info(f"[DEBUG] Trying direct client.list() call...")
                            raw_response = await ollama_provider.client.list()
                            logger.info(
                                f"[DEBUG] Direct client.list() returned: {raw_response} (type: {type(raw_response)})"
                            )
                            if isinstance(raw_response, dict):
                                installed_models = [
                                    m.get("name") or m.get("model") or m.get("id")
                                    for m in raw_response.get("models", [])
                                ]
                            elif isinstance(raw_response, list):
                                installed_models = [
                                    m.get("name") if isinstance(m, dict) else str(m)
                                    for m in raw_response
                                ]
                            else:
                                installed_models = []
                            logger.info(f"[DEBUG] Parsed models: {installed_models}")
                        except Exception as direct_error:
                            logger.error(
                                f"[DEBUG] Direct client.list() also failed: {direct_error}",
                                exc_info=True,
                            )
                            installed_models = []

                    installed_models_lower = [m.lower() for m in installed_models if m]
                    logger.info(
                        f"[DEBUG] Checking Ollama models for '{request.model}'. Installed models: {installed_models} (count: {len(installed_models)})"
                    )
                    if not installed_models:
                        logger.warning(
                            f"[DEBUG] WARNING: Ollama list_models() returned empty list. This might indicate:"
                        )
                        logger.warning(f"[DEBUG]   1. Ollama daemon is not running")
                        logger.warning(f"[DEBUG]   2. No models are installed")
                        logger.warning(f"[DEBUG]   3. Connection issue with Ollama")

                    mapped_model_name = get_ollama_model_name_for_chat(request.model)
                    mapped_model_name_lower = mapped_model_name.lower()

                    logger.info(
                        f"[DEBUG] Model mapping: '{request.model}' -> '{mapped_model_name}' (lower: '{mapped_model_name_lower}')"
                    )
                    if mapped_model_name_lower in installed_models_lower:
                        use_ollama = True
                        for installed_model in installed_models:
                            if installed_model.lower() == mapped_model_name_lower:
                                ollama_model_name = installed_model
                                break
                        else:
                            ollama_model_name = mapped_model_name
                        logger.info(
                            f"Using Ollama provider for {request.model} -> {ollama_model_name} (mapped and found in Ollama)"
                        )
                    elif model_name_lower in installed_models_lower:
                        use_ollama = True
                        for installed_model in installed_models:
                            if installed_model.lower() == model_name_lower:
                                ollama_model_name = installed_model
                                break
                        else:
                            ollama_model_name = request.model
                        logger.info(
                            f"Using Ollama provider for {request.model} -> {ollama_model_name} (found in Ollama)"
                        )
                    else:
                        logger.info(f"[DEBUG] Exact match not found. Checking similar names...")
                        for installed_model in installed_models:
                            if (
                                model_name_lower in installed_model.lower()
                                or installed_model.lower() in model_name_lower
                            ):
                                use_ollama = True
                                ollama_model_name = installed_model
                                logger.info(
                                    f"Using Ollama provider for {request.model} -> {installed_model} (found similar)"
                                )
                                break
                        if not use_ollama:
                            logger.info(
                                f"[DEBUG] Checking mapped name '{mapped_model_name_lower}' for similar matches..."
                            )
                            for installed_model in installed_models:
                                if (
                                    mapped_model_name_lower in installed_model.lower()
                                    or installed_model.lower() in mapped_model_name_lower
                                ):
                                    use_ollama = True
                                    ollama_model_name = installed_model
                                    logger.info(
                                        f"Using Ollama provider for {request.model} -> {installed_model} (mapped and found similar)"
                                    )
                                    break

                        if not use_ollama:
                            logger.warning(
                                f"[DEBUG] Model '{request.model}' (mapped: '{mapped_model_name}') not found in Ollama. Installed: {installed_models}"
                            )
                            if model_info and model_info.provider == "ollama":
                                logger.info(
                                    f"[DEBUG] list_models() returned empty list, but model '{request.model}' is registered as Ollama."
                                )
                                logger.info(
                                    f"[DEBUG] This might be a connection issue. Will try to use mapped name: {mapped_model_name}"
                                )
                                logger.info(
                                    f"[DEBUG] Chat will attempt to use the model directly - if it works, the model is installed."
                                )
                                use_ollama = True
                                ollama_model_name = mapped_model_name
                except Exception as e:
                    logger.error(f"Ollama check failed for {request.model}: {e}", exc_info=True)
                    if model_info and model_info.provider == "ollama":
                        mapped_model_name = get_ollama_model_name_for_chat(request.model)
                        logger.info(
                            f"[DEBUG] Ollama check exception but model is registered as Ollama. Will try to use mapped name: {mapped_model_name}"
                        )
                        use_ollama = True
                        ollama_model_name = mapped_model_name

            elif model_info and model_info.provider != "ollama":
                opensource_patterns = ["deepseek", "mistral", "mixtral", "gemma", "codellama"]
                is_opensource = any(pattern in model_name_lower for pattern in opensource_patterns)

                if is_opensource:
                    try:
                        from beanllm.providers.ollama_provider import OllamaProvider

                        ollama_provider = OllamaProvider()
                        installed_models = await ollama_provider.list_models()
                        installed_models_lower = [m.lower() for m in installed_models]
                        if model_name_lower in installed_models_lower:
                            use_ollama = True
                            ollama_model_name = request.model
                            logger.info(
                                    f"Using Ollama provider for {request.model} (found in Ollama, overriding Registry provider)"
                            )
                        else:
                            for installed_model in installed_models:
                                if (
                                    model_name_lower in installed_model.lower()
                                    or installed_model.lower() in model_name_lower
                                ):
                                    use_ollama = True
                                    ollama_model_name = installed_model
                                    logger.info(
                                        f"Using Ollama provider for {request.model} (found similar in Ollama: {installed_model})"
                                    )
                                    break
                    except Exception as e:
                        logger.debug(f"Ollama check failed for {request.model}: {e}")

            elif ":" in request.model:
                try:
                    from beanllm.providers.ollama_provider import OllamaProvider

                    ollama_provider = OllamaProvider()
                    installed_models = await ollama_provider.list_models()
                    installed_models_lower = [m.lower() for m in installed_models]

                    if model_name_lower in installed_models_lower:
                        use_ollama = True
                        ollama_model_name = request.model
                        logger.info(
                            f"Using Ollama provider for {request.model} (not in Registry, but found in Ollama)"
                        )
                    else:
                        for installed_model in installed_models:
                            if (
                                model_name_lower in installed_model.lower()
                                or installed_model.lower() in model_name_lower
                            ):
                                use_ollama = True
                                ollama_model_name = installed_model
                                logger.info(
                                    f"Using Ollama provider for {request.model} (found similar: {installed_model})"
                                )
                                break
                except Exception as e:
                    logger.debug(f"Ollama check failed for {request.model}: {e}")

            if use_ollama:
                final_model_name = ollama_model_name or request.model
                logger.info(
                    f"[DEBUG] Creating Client with model='{final_model_name}', provider='ollama' (requested: '{request.model}', use_ollama={use_ollama}, ollama_model_name={ollama_model_name})"
                )
                client = Client(model=final_model_name, provider="ollama")
            else:
                logger.info(
                    f"[DEBUG] Creating Client with model='{request.model}', provider=auto-detect (use_ollama={use_ollama})"
                )
                client = Client(model=request.model)
            chat_kwargs = {}
            if request.temperature is not None:
                chat_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                chat_kwargs["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                chat_kwargs["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                chat_kwargs["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                chat_kwargs["presence_penalty"] = request.presence_penalty

            if request.enable_thinking:
                if request.model.startswith("claude"):
                    chat_kwargs["extra_params"] = {"thinking": True}
                elif not request.model.startswith(("o1", "o3", "gpt-5")):
                    if messages and messages[0].get("role") != "system":
                        messages.insert(
                            0,
                            {
                                "role": "system",
                                "content": "Think step by step. Show your reasoning process using <think>...</think> tags before your final answer.",
                            },
                        )

            try:
                response = await client.chat(messages=messages, **chat_kwargs)
            except Exception as chat_error:
                error_msg = str(chat_error)
                if (
                    "no available llm provider" in error_msg.lower()
                    or "no available provider" in error_msg.lower()
                ):
                    model_name = request.model.lower()
                    if model_name.startswith("deepseek"):
                        try:
                            from beanllm.providers.ollama_provider import OllamaProvider

                            ollama_provider = OllamaProvider()
                            installed_models = await ollama_provider.list_models()
                            if model_name in [m.lower() for m in installed_models]:
                                raise HTTPException(
                                    400,
                                    f"모델 '{request.model}'이 Ollama에 설치되어 있습니다. "
                                    f"Ollama provider를 사용하려면 provider를 'ollama'로 명시하거나 "
                                    f"모델 이름을 '{request.model}' 그대로 사용하세요. "
                                    f"(현재는 DeepSeek API provider를 시도했습니다)",
                                )
                        except Exception as ollama_check_error:
                            logger.debug(f"Ollama check failed: {ollama_check_error}")

                    provider_name = None
                    api_key_env = None

                    if model_name.startswith("deepseek"):
                        provider_name = "DeepSeek"
                        api_key_env = "DEEPSEEK_API_KEY"
                        raise HTTPException(
                            401,
                            f"DeepSeek 모델을 사용하려면 API 키가 필요합니다. "
                            f"환경 변수 'DEEPSEEK_API_KEY'를 설정하거나, "
                            f"Ollama에 모델을 설치하여 로컬에서 사용할 수 있습니다. "
                            f"(예: `ollama pull {request.model}`)",
                        )
                    elif any(
                        pattern in model_name
                        for pattern in [
                            "mistral",
                            "mixtral",
                            "gemma",
                            "codellama",
                            "neural",
                            "starling",
                            "orca",
                            "vicuna",
                            "wizard",
                            "falcon",
                        ]
                    ):
                        model_type = next(
                            pattern
                            for pattern in [
                                "mistral",
                                "mixtral",
                                "gemma",
                                "codellama",
                                "neural",
                                "starling",
                                "orca",
                                "vicuna",
                                "wizard",
                                "falcon",
                            ]
                            if pattern in model_name
                        )
                        raise HTTPException(
                            401,
                            f"{model_type.capitalize()} 모델을 사용하려면 API 키가 필요하거나, "
                            f"Ollama에 모델을 설치하여 로컬에서 사용할 수 있습니다. "
                            f"(예: `ollama pull {request.model}`)",
                        )
                    elif model_name.startswith("claude"):
                        provider_name = "Claude"
                        api_key_env = "ANTHROPIC_API_KEY"
                    elif model_name.startswith("gemini"):
                        provider_name = "Gemini"
                        api_key_env = "GEMINI_API_KEY"
                    elif (
                        model_name.startswith("gpt")
                        or model_name.startswith("o1")
                        or model_name.startswith("o3")
                        or model_name.startswith("o4")
                    ):
                        provider_name = "OpenAI"
                        api_key_env = "OPENAI_API_KEY"
                    elif "perplexity" in model_name or "sonar" in model_name:
                        provider_name = "Perplexity"
                        api_key_env = "PERPLEXITY_API_KEY"
                    elif ":" in request.model or "ollama" in error_msg.lower():
                        raise HTTPException(
                            404,
                            f"모델 '{request.model}'을(를) 찾을 수 없습니다. 모델을 다운로드해주세요. "
                            f"API: POST /api/models/{request.model}/pull",
                        )

                    if provider_name and api_key_env:
                        raise HTTPException(
                            401,
                            f"{provider_name} 모델을 사용하려면 API 키가 필요합니다. "
                            f"환경 변수 '{api_key_env}'를 설정해주세요.",
                        )
                    else:
                        raise HTTPException(
                            500,
                            f"모델 '{request.model}'을(를) 사용할 수 없습니다. "
                            f"API 키가 설정되어 있는지 확인하거나 다른 모델을 선택해주세요.",
                        )

                # Ollama 모델이 없을 때 처리
                elif "not found" in error_msg.lower() and (
                    "ollama" in error_msg.lower() or ":" in request.model
                ):
                    # Registry에서 모델 정보 확인
                    from beanllm.infrastructure.registry import get_model_registry

                    registry = get_model_registry()
                    model_info = None
                    try:
                        model_info = registry.get_model_info(request.model)
                    except:
                        pass

                    # Registry에 등록된 모델이지만 Ollama에 없으면 다운로드 안내
                    if model_info and model_info.provider == "ollama":
                        raise HTTPException(
                            404,
                            f"모델 '{request.model}'을(를) 찾을 수 없습니다. 모델을 다운로드해주세요. "
                            f"API: POST /api/models/{request.model}/pull",
                        )
                    else:
                        # Registry에 등록된 다른 provider 모델인데 Ollama로 시도한 경우
                        # (이미 Client._detect_provider가 올바른 provider를 사용하므로 드문 경우)
                        if model_info:
                            raise HTTPException(
                                404,
                                f"모델 '{request.model}'을(를) 사용할 수 없습니다. "
                                f"Provider: {model_info.provider}, 에러: {error_msg}",
                            )
                        else:
                            raise HTTPException(
                                404,
                                f"모델 '{request.model}'을(를) 찾을 수 없습니다. "
                                f"모델 이름을 확인하거나 다른 모델을 선택해주세요.",
                            )
                # 다른 모델 관련 에러
                elif "model" in error_msg.lower() and (
                    "not found" in error_msg.lower() or "not available" in error_msg.lower()
                ):
                    raise HTTPException(
                        404,
                        f"모델 '{request.model}'을(를) 사용할 수 없습니다. 모델 이름을 확인하거나 다른 모델을 선택해주세요.",
                    )
                # API 키 관련 에러
                elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                    raise HTTPException(
                        401, f"API 키가 필요합니다. 환경 변수에 API 키를 설정해주세요."
                    )
                # 그 외 에러는 그대로 전달
                raise HTTPException(500, f"Chat error: {error_msg}")
        else:
            # Fallback to default client (requires OPENAI_API_KEY)
            client = get_client()
            chat_kwargs = {}
            if request.temperature is not None:
                chat_kwargs["temperature"] = request.temperature
            if request.max_tokens is not None:
                chat_kwargs["max_tokens"] = request.max_tokens
            if request.top_p is not None:
                chat_kwargs["top_p"] = request.top_p
            if request.frequency_penalty is not None:
                chat_kwargs["frequency_penalty"] = request.frequency_penalty
            if request.presence_penalty is not None:
                chat_kwargs["presence_penalty"] = request.presence_penalty

            # Enable thinking mode if requested
            if request.enable_thinking:
                # For Claude models, add thinking parameter
                model_name = request.model or "gpt-4o-mini"
                if model_name.startswith("claude"):
                    chat_kwargs["extra_params"] = {"thinking": True}
                # For OpenAI reasoning models (o1, o3), thinking is automatic
                # For other models, we can add a system prompt to encourage thinking
                elif not model_name.startswith(("o1", "o3", "gpt-5")):
                    if messages and messages[0].get("role") != "system":
                        messages.insert(
                            0,
                            {
                                "role": "system",
                                "content": "Think step by step. Show your reasoning process using <think>...</think> tags before your final answer.",
                            },
                        )

            await ChatMonitoringMixin.log_chat_request(
                request_id=request_id,
                model=request.model or "gpt-4o-mini",
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            try:
                chat_response_start = time.time()
                response = await client.chat(messages=messages, **chat_kwargs)
                chat_duration_ms = (time.time() - chat_response_start) * 1000
            except Exception as chat_error:
                error_msg = str(chat_error)
                if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                    raise HTTPException(
                        401, f"API 키가 필요합니다. OPENAI_API_KEY 환경 변수를 설정해주세요."
                    )
                raise HTTPException(500, f"Chat error: {error_msg}")

        usage = response.usage if hasattr(response, "usage") else None
        await ChatMonitoringMixin.log_chat_response(
            request_id=request_id,
            model=response.model if hasattr(response, "model") else request.model or "default",
            response_content=response.content if hasattr(response, "content") else str(response),
            input_tokens=usage.input_tokens if usage and hasattr(usage, "input_tokens") else None,
            output_tokens=(
                usage.output_tokens if usage and hasattr(usage, "output_tokens") else None
            ),
            duration_ms=chat_duration_ms if "chat_duration_ms" in locals() else None,
        )

        return {
            "role": "assistant",
            "content": response.content,
            "usage": response.usage,
            "model": response.model,
            "provider": response.provider,
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Chat endpoint error: {error_msg}", exc_info=True)
        raise HTTPException(500, f"Chat error: {error_msg}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    print(f"WebSocket connected: {session_id}")

    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to beanllm playground",
        })

        while True:
            try:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    print(f"Received from {session_id}: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    finally:
        if session_id in active_connections:
            del active_connections[session_id]
        print(f"WebSocket disconnected: {session_id}")




from routers.config_router import router as config_router
from routers.chat_router import router as chat_router
from routers.google_auth_router import router as google_auth_router
from routers.monitoring_router import router as monitoring_router
from routers.models_router import router as models_router
from routers.kg_router import router as kg_router
from routers.rag_router import router as rag_router
from routers.agent_router import router as agent_router
from routers.chain_router import router as chain_router
from routers.vision_router import router as vision_router
from routers.audio_router import router as audio_router
from routers.evaluation_router import router as evaluation_router
from routers.finetuning_router import router as finetuning_router
from routers.ocr_router import router as ocr_router
from routers.web_router import router as web_router
from routers.optimizer_router import router as optimizer_router

app.include_router(config_router)
app.include_router(chat_router)
app.include_router(google_auth_router)
app.include_router(monitoring_router)
app.include_router(models_router)
app.include_router(kg_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(chain_router)
app.include_router(vision_router)
app.include_router(audio_router)
app.include_router(evaluation_router)
app.include_router(finetuning_router)
app.include_router(ocr_router)
app.include_router(web_router)
app.include_router(optimizer_router)

logger.info("✅ Modular routers included: config, chat, google_auth, monitoring, models, kg, rag, agent, chain, vision, audio, evaluation, finetuning, ocr, web, optimizer")

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("beanllm Playground API Server")
    print("=" * 60)
    print("Starting on http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\nAvailable Features:")
    print("  - Chat (General conversation)")
    print("  - Knowledge Graph (Build & Query)")
    print("  - RAG (Retrieval-Augmented Generation)")
    print("  - Agent (Autonomous task execution)")
    print("  - OCR (Optical Character Recognition)")
    print("  - Web Search (Multi-engine search)")
    print("  - RAG Debug (Pipeline analysis)")
    print("  - Optimizer (Performance optimization)")
    print("  - Multi-Agent (Collaborative agents)")
    print("  - Orchestrator (Workflow management)")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
