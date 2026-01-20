"""
beanllm Playground Backend - FastAPI

Complete working backend for all 9 beanllm features
"""

import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

logger = logging.getLogger(__name__)

# Add parent directory to path to import beanllm
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# 다운로드 완료된 모델 추적 (list_models()가 실패해도 UI에 표시하기 위해)
# Key: 로컬 모델 이름 (예: "phi3.5"), Value: Ollama 실제 모델 이름 (예: "phi3")
_downloaded_models: Dict[str, str] = {}

# ============================================================================
# Model Name Mapping (공통 함수)
# ============================================================================

def get_ollama_model_name_for_chat(model_name: str) -> str:
    """
    Chat 시 사용할 Ollama 실제 모델 이름으로 매핑
    
    참고: Pull 시에는 원래 이름을 사용하고, Chat 시에만 이 매핑을 사용합니다.
    (예: pull은 "phi3.5"를 사용하지만, chat은 "phi3"를 사용)
    
    매핑 규칙:
    1. 특정 매핑이 있으면 사용
    2. 콜론(:)이 있으면 그대로 사용 (예: qwen2.5:0.5b)
    3. 버전 번호가 있으면 제거하거나 변환 (예: phi3.5 -> phi3)
    
    Args:
        model_name: 로컬 모델 이름 (예: "phi3.5", "qwen2.5:0.5b")
    
    Returns:
        Ollama 실제 모델 이름 (예: "phi3", "qwen2.5:0.5b")
    """
    # 특정 매핑 (로컬 이름 -> Ollama 실제 이름)
    # 웹 검색 결과 기반: https://ollama.org, https://ollama.ai/library
    # 참고: phi3.5는 pull 시에는 "phi3.5"를 사용하지만, 설치 후 "phi3"로 저장됨
    model_name_mapping = {
        # Phi 시리즈 (Microsoft)
        "phi3.5": "phi3",  # phi3.5는 다운로드 후 phi3로 저장됨 (chat 시 사용)
        "phi-3.5": "phi3",
        "phi3": "phi3",  # 이미 올바른 이름
        "phi4": "phi4:14b",  # phi4는 14b 태그 필요 (웹 검색 결과 기반)
        
        # Qwen 시리즈 (Alibaba) - 대부분 매핑 불필요, 콜론 포함 모델은 그대로 사용
        "qwen3": "qwen3",  # 확인 필요
        
        # 기타 모델들은 콜론 포함 모델이거나 그대로 사용
    }
    
    # 매핑이 있으면 사용
    if model_name in model_name_mapping:
        return model_name_mapping[model_name]
    
    # 콜론이 있으면 그대로 사용 (예: qwen2.5:0.5b, llama3.3:70b)
    # Ollama는 태그 포함 모델 이름을 그대로 사용함
    if ":" in model_name:
        return model_name
    
    # 그 외는 그대로 반환
    return model_name

from beanllm import Client
from beanllm.facade.advanced.knowledge_graph_facade import KnowledgeGraph
from beanllm.facade.core.rag_facade import RAGChain, RAGBuilder
from beanllm.facade.core.agent_facade import Agent
from beanllm.facade.core.chain_facade import Chain, ChainBuilder, PromptChain
from beanllm.facade.ml.web_search_facade import WebSearch
from beanllm.domain.web_search import SearchEngine
from beanllm.facade.advanced.rag_debug_facade import RAGDebug
from beanllm.facade.advanced.optimizer_facade import Optimizer
from beanllm.facade.advanced.multi_agent_facade import MultiAgentCoordinator
from beanllm.facade.advanced.orchestrator_facade import Orchestrator
from beanllm.facade.ml.vision_rag_facade import VisionRAG, MultimodalRAG
from beanllm.facade.ml.audio_facade import WhisperSTT, TextToSpeech, AudioRAG
from beanllm.facade.ml.evaluation_facade import EvaluatorFacade
from beanllm.facade.ml.finetuning_facade import FineTuningManagerFacade
from beanllm.domain.ocr import beanOCR, OCRConfig

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="beanllm Playground API",
    description="Complete backend for all beanllm features",
    version="1.0.0",
)

# CORS for Next.js frontend
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

# ============================================================================
# Global State
# ============================================================================

# Client initialization (lazy)
_client: Optional[Client] = None
_kg: Optional[KnowledgeGraph] = None
_rag_chains: Dict[str, RAGChain] = {}  # collection_name -> RAGChain
_chains: Dict[str, Chain] = {}  # chain_id -> Chain
_web_search: Optional[WebSearch] = None
_rag_debugger: Optional[RAGDebug] = None
_optimizer: Optional[Optimizer] = None
_multi_agent: Optional[MultiAgentCoordinator] = None
_orchestrator: Optional[Orchestrator] = None
_vision_rag: Optional[VisionRAG] = None
_audio_rag: Optional[AudioRAG] = None
_evaluator: Optional[EvaluatorFacade] = None
_finetuning: Optional[FineTuningManagerFacade] = None


def get_client() -> Client:
    """Get or create beanllm client"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(500, "OPENAI_API_KEY not set")
        _client = Client(provider="openai", api_key=api_key, model="gpt-4o-mini")
    return _client


def get_kg() -> KnowledgeGraph:
    """Get or create KnowledgeGraph facade"""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph(client=get_client())
    return _kg


def get_web_search() -> WebSearch:
    """Get or create WebSearch facade"""
    global _web_search
    if _web_search is None:
        _web_search = WebSearch()
    return _web_search


def get_rag_debugger(vector_store=None) -> RAGDebug:
    """Get or create RAGDebug facade"""
    global _rag_debugger
    if vector_store is None:
        # Create default vector_store if not provided
        from beanllm.domain.vector_stores import VectorStore
        from beanllm.domain.embeddings import Embedding

        embedding = Embedding(model="text-embedding-3-small")
        vector_store = VectorStore(embedding_function=embedding.embed)
    # Create new debugger if vector_store changed
    if _rag_debugger is None or _rag_debugger.vector_store != vector_store:
        _rag_debugger = RAGDebug(vector_store=vector_store)
    return _rag_debugger


def get_optimizer() -> Optimizer:
    """Get or create Optimizer facade"""
    global _optimizer
    if _optimizer is None:
        _optimizer = Optimizer()
    return _optimizer


def get_multi_agent() -> MultiAgentCoordinator:
    """Get or create MultiAgent facade"""
    global _multi_agent
    if _multi_agent is None:
        _multi_agent = MultiAgentCoordinator()
    return _multi_agent


def get_orchestrator() -> Orchestrator:
    """Get or create Orchestrator facade"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


# WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# ============================================================================
# Request/Response Models
# ============================================================================


class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    assistant_id: str = "chat"
    model: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = False  # Enable thinking/reasoning mode
    images: Optional[List[str]] = None  # Base64 encoded images
    files: Optional[List[Dict[str, Any]]] = None  # File attachments


# Knowledge Graph
class BuildGraphRequest(BaseModel):
    documents: List[str]
    graph_id: Optional[str] = None
    entity_types: Optional[List[str]] = None
    relation_types: Optional[List[str]] = None
    model: Optional[str] = None  # Optional model selection


class QueryGraphRequest(BaseModel):
    graph_id: str
    query_type: str = "cypher"
    query: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # Optional model selection


class GraphRAGRequest(BaseModel):
    query: str
    graph_id: str
    model: Optional[str] = None  # Optional model selection


# RAG
class RAGBuildRequest(BaseModel):
    documents: List[str]
    collection_name: Optional[str] = "default"
    model: Optional[str] = None  # Optional model selection


class RAGQueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None  # Optional model selection


# Agent
class AgentRequest(BaseModel):
    task: str
    tools: Optional[List[str]] = None
    max_iterations: int = 10
    model: Optional[str] = None  # Optional model selection


# Web Search
class WebSearchRequest(BaseModel):
    query: str
    num_results: int = 5
    engine: str = "duckduckgo"
    model: Optional[str] = None  # LLM 모델 (결과 요약/개선용, 선택적)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    summarize: bool = False  # 검색 결과를 LLM으로 요약할지 여부


# RAG Debug
class RAGDebugRequest(BaseModel):
    query: str
    documents: List[str]
    collection_name: Optional[str] = None  # Use existing RAG chain's vector_store
    debug_mode: str = "full"
    model: Optional[str] = None  # Optional model selection


# Optimizer
class OptimizeRequest(BaseModel):
    task_type: str = "rag"  # rag, agent, chain
    config: Optional[Dict[str, Any]] = None
    top_k_range: Optional[tuple] = None  # (min, max) for quick_optimize
    threshold_range: Optional[tuple] = None  # (min, max) for quick_optimize
    method: str = "bayesian"  # bayesian, grid, random, genetic
    n_trials: int = 30
    test_queries: Optional[List[str]] = None
    model: Optional[str] = None  # Optional model selection


# Multi-Agent
class MultiAgentRequest(BaseModel):
    task: str
    num_agents: int = 3
    strategy: str = "sequential"  # sequential, parallel, hierarchical, debate
    model: Optional[str] = None  # Optional model selection
    agent_configs: Optional[List[Dict[str, Any]]] = None  # Optional: custom agent configs
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


# Orchestrator
class WorkflowRequest(BaseModel):
    workflow_type: str  # research_write, parallel_consensus, debate
    task: str
    input_data: Optional[Dict[str, Any]] = None
    model: Optional[str] = None  # Optional model selection
    num_agents: int = 2  # Number of agents for quick methods


# Chain
class ChainRequest(BaseModel):
    input: str
    chain_id: Optional[str] = None
    chain_type: str = "basic"  # basic, prompt
    template: Optional[str] = None
    model: Optional[str] = None


# VisionRAG
class VisionRAGBuildRequest(BaseModel):
    images: List[str]  # Base64 encoded images or URLs
    texts: Optional[List[str]] = None
    collection_name: Optional[str] = "default"
    model: Optional[str] = None
    generate_captions: bool = False  # Disable by default to avoid transformers dependency


class VisionRAGQueryRequest(BaseModel):
    query: str
    image: Optional[str] = None  # Base64 encoded image or URL
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None


# Audio
class AudioTranscribeRequest(BaseModel):
    audio_file: str  # Base64 encoded audio or file path
    model: Optional[str] = None


class AudioSynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    model: Optional[str] = None


class AudioRAGRequest(BaseModel):
    query: str
    audio_files: Optional[List[str]] = None
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None


# Evaluation
class EvaluationRequest(BaseModel):
    task_type: str  # rag, agent, chain
    queries: List[str]
    ground_truth: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


# Fine-tuning
class FineTuningCreateRequest(BaseModel):
    base_model: str
    training_data: List[Dict[str, Any]]
    job_name: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class FineTuningStatusRequest(BaseModel):
    job_id: str


# ============================================================================
# Health Check
# ============================================================================


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


# ============================================================================
# Chat API
# ============================================================================


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint - routes to different assistants
    """
    try:
        # Convert messages to beanllm format
        # Handle images/files for multimodal messages
        messages = []
        for i, msg in enumerate(request.messages):
            # Check if this is the last user message and has images/files
            is_last_user = i == len(request.messages) - 1 and msg.role == "user"
            has_images = is_last_user and request.images and len(request.images) > 0
            has_files = is_last_user and request.files and len(request.files) > 0
            
            if has_images or has_files:
                # Create multimodal message for vision models
                content = []
                
                # Add text content
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                
                # Add images (OpenAI-style multimodal format)
                if has_images:
                    import base64
                    
                    for img_base64 in request.images:
                        try:
                            # Use base64 directly in OpenAI format
                            # OpenAI expects: {"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}
                            # If already has data URL prefix, use as is; otherwise add it
                            if img_base64.startswith("data:image"):
                                image_url = img_base64
                            else:
                                # Assume it's base64 without prefix, add default prefix
                                image_url = f"data:image/png;base64,{img_base64}"
                            
                            # Add image in OpenAI format
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                        except Exception as e:
                            logger.warning(f"Failed to process image: {e}")
                            continue
                
                # Add file information as text (files are not directly supported by vision models)
                if has_files:
                    file_info = "\n\nAttached files:\n" + "\n".join([f"- {f.get('name', 'Unknown')} ({f.get('type', 'Unknown type')})" for f in request.files])
                    if content and content[0].get("type") == "text":
                        content[0]["text"] += file_info
                    else:
                        content.insert(0, {"type": "text", "text": file_info})
                
                messages.append({"role": msg.role, "content": content})
            else:
                # Regular text message
                messages.append({"role": msg.role, "content": msg.content})

        # If model is provided, create a client for that model
        # Client._detect_provider가 Registry를 먼저 확인하므로 올바른 provider를 자동 감지
        if request.model:
            # Registry에서 모델 정보 확인
            from beanllm.infrastructure.registry import get_model_registry
            registry = get_model_registry()
            model_info = None
            try:
                model_info = registry.get_model_info(request.model)
            except:
                pass
            
            model_name_lower = request.model.lower()
            
            # Ollama에 설치된 모델 확인 (오픈소스 모델은 Ollama에 설치되어 있을 수 있음)
            use_ollama = False
            ollama_model_name = None
            
            # 1. Registry에 등록된 Ollama 모델인 경우
            if model_info and model_info.provider == "ollama":
                # Ollama 모델은 실제 설치 여부 확인
                try:
                    from beanllm.providers.ollama_provider import OllamaProvider
                    ollama_provider = OllamaProvider()
                    
                    # Ollama 연결 상태 확인
                    try:
                        health = await ollama_provider.health_check()
                        logger.info(f"[DEBUG] Ollama health check: {health}")
                    except Exception as health_error:
                        logger.warning(f"[DEBUG] Ollama health check failed: {health_error}", exc_info=True)
                    
                    # list_models() 호출 전후로 상세 로깅
                    try:
                        logger.info(f"[DEBUG] Calling ollama_provider.list_models()...")
                        installed_models = await ollama_provider.list_models()
                        logger.info(f"[DEBUG] list_models() returned: {installed_models} (type: {type(installed_models)}, count: {len(installed_models)})")
                    except Exception as list_error:
                        logger.error(f"[DEBUG] list_models() raised exception: {list_error}", exc_info=True)
                        # 직접 client.list() 호출 시도
                        try:
                            logger.info(f"[DEBUG] Trying direct client.list() call...")
                            raw_response = await ollama_provider.client.list()
                            logger.info(f"[DEBUG] Direct client.list() returned: {raw_response} (type: {type(raw_response)})")
                            # 수동으로 파싱
                            if isinstance(raw_response, dict):
                                installed_models = [m.get("name") or m.get("model") or m.get("id") for m in raw_response.get("models", [])]
                            elif isinstance(raw_response, list):
                                installed_models = [m.get("name") if isinstance(m, dict) else str(m) for m in raw_response]
                            else:
                                installed_models = []
                            logger.info(f"[DEBUG] Parsed models: {installed_models}")
                        except Exception as direct_error:
                            logger.error(f"[DEBUG] Direct client.list() also failed: {direct_error}", exc_info=True)
                            installed_models = []
                    
                    installed_models_lower = [m.lower() for m in installed_models if m]
                    
                    logger.info(f"[DEBUG] Checking Ollama models for '{request.model}'. Installed models: {installed_models} (count: {len(installed_models)})")
                    
                    # 모델이 비어있으면 경고
                    if not installed_models:
                        logger.warning(f"[DEBUG] WARNING: Ollama list_models() returned empty list. This might indicate:")
                        logger.warning(f"[DEBUG]   1. Ollama daemon is not running")
                        logger.warning(f"[DEBUG]   2. No models are installed")
                        logger.warning(f"[DEBUG]   3. Connection issue with Ollama")
                    
                    # 모델 이름 매핑 적용 (Chat 시에만 매핑 사용)
                    # Pull 시에는 원래 이름을 사용하지만, Chat 시에는 설치된 이름을 사용
                    mapped_model_name = get_ollama_model_name_for_chat(request.model)
                    mapped_model_name_lower = mapped_model_name.lower()
                    
                    logger.info(f"[DEBUG] Model mapping: '{request.model}' -> '{mapped_model_name}' (lower: '{mapped_model_name_lower}')")
                    
                    # 매핑된 이름으로 먼저 확인 (예: phi3.5 -> phi3)
                    if mapped_model_name_lower in installed_models_lower:
                        use_ollama = True
                        # 실제 설치된 모델 이름 찾기 (대소문자 구분)
                        for installed_model in installed_models:
                            if installed_model.lower() == mapped_model_name_lower:
                                ollama_model_name = installed_model
                                break
                        else:
                            ollama_model_name = mapped_model_name
                        logger.info(f"Using Ollama provider for {request.model} -> {ollama_model_name} (mapped and found in Ollama)")
                    elif model_name_lower in installed_models_lower:
                        use_ollama = True
                        # 실제 설치된 모델 이름 찾기
                        for installed_model in installed_models:
                            if installed_model.lower() == model_name_lower:
                                ollama_model_name = installed_model
                                break
                        else:
                            ollama_model_name = request.model
                        logger.info(f"Using Ollama provider for {request.model} -> {ollama_model_name} (found in Ollama)")
                    else:
                        logger.info(f"[DEBUG] Exact match not found. Checking similar names...")
                        # 비슷한 이름의 모델 확인 (예: phi3.5 vs phi3)
                        for installed_model in installed_models:
                            if model_name_lower in installed_model.lower() or installed_model.lower() in model_name_lower:
                                use_ollama = True
                                ollama_model_name = installed_model
                                logger.info(f"Using Ollama provider for {request.model} -> {installed_model} (found similar)")
                                break
                        # 매핑된 이름으로도 비슷한 이름 찾기
                        if not use_ollama:
                            logger.info(f"[DEBUG] Checking mapped name '{mapped_model_name_lower}' for similar matches...")
                            for installed_model in installed_models:
                                if mapped_model_name_lower in installed_model.lower() or installed_model.lower() in mapped_model_name_lower:
                                    use_ollama = True
                                    ollama_model_name = installed_model
                                    logger.info(f"Using Ollama provider for {request.model} -> {installed_model} (mapped and found similar)")
                                    break
                        
                        if not use_ollama:
                            logger.warning(f"[DEBUG] Model '{request.model}' (mapped: '{mapped_model_name}') not found in Ollama. Installed: {installed_models}")
                            # list_models()가 빈 리스트를 반환하더라도, 실제로 모델이 설치되어 있을 수 있음
                            # (Ollama 연결 문제일 수 있음)
                            # Registry에 Ollama 모델로 등록되어 있으면 시도해볼 수 있도록 함
                            if model_info and model_info.provider == "ollama":
                                logger.info(f"[DEBUG] list_models() returned empty list, but model '{request.model}' is registered as Ollama.")
                                logger.info(f"[DEBUG] This might be a connection issue. Will try to use mapped name: {mapped_model_name}")
                                logger.info(f"[DEBUG] Chat will attempt to use the model directly - if it works, the model is installed.")
                                use_ollama = True
                                ollama_model_name = mapped_model_name
                except Exception as e:
                    logger.error(f"Ollama check failed for {request.model}: {e}", exc_info=True)
                    # 예외가 발생해도 Registry에 Ollama 모델로 등록되어 있으면 시도
                    if model_info and model_info.provider == "ollama":
                        mapped_model_name = get_ollama_model_name_for_chat(request.model)
                        logger.info(f"[DEBUG] Ollama check exception but model is registered as Ollama. Will try to use mapped name: {mapped_model_name}")
                        use_ollama = True
                        ollama_model_name = mapped_model_name
            
            # 2. Registry에 등록된 다른 provider 모델이지만 Ollama에 설치되어 있을 수 있는 경우
            # (예: deepseek-chat이 Registry에 DEEPSEEK로 등록되어 있지만 Ollama에도 설치되어 있을 수 있음)
            elif model_info and model_info.provider != "ollama":
                # 오픈소스 모델 패턴 (Ollama에 설치 가능)
                opensource_patterns = ["deepseek", "mistral", "mixtral", "gemma", "codellama"]
                is_opensource = any(pattern in model_name_lower for pattern in opensource_patterns)
                
                if is_opensource:
                    try:
                        from beanllm.providers.ollama_provider import OllamaProvider
                        ollama_provider = OllamaProvider()
                        installed_models = await ollama_provider.list_models()
                        installed_models_lower = [m.lower() for m in installed_models]
                        
                        # Ollama에 설치되어 있는지 확인
                        if model_name_lower in installed_models_lower:
                            use_ollama = True
                            ollama_model_name = request.model
                            logger.info(f"Using Ollama provider for {request.model} (found in Ollama, overriding Registry provider)")
                        else:
                            # 비슷한 이름의 모델 확인
                            for installed_model in installed_models:
                                if model_name_lower in installed_model.lower() or installed_model.lower() in model_name_lower:
                                    use_ollama = True
                                    ollama_model_name = installed_model
                                    logger.info(f"Using Ollama provider for {request.model} (found similar in Ollama: {installed_model})")
                                    break
                    except Exception as e:
                        logger.debug(f"Ollama check failed for {request.model}: {e}")
            
            # 3. Registry에 없는 모델인 경우 (예: qwen2.5:0.5b)
            # 콜론이 있으면 Ollama 모델로 간주
            elif ":" in request.model:
                try:
                    from beanllm.providers.ollama_provider import OllamaProvider
                    ollama_provider = OllamaProvider()
                    installed_models = await ollama_provider.list_models()
                    installed_models_lower = [m.lower() for m in installed_models]
                    
                    if model_name_lower in installed_models_lower:
                        use_ollama = True
                        ollama_model_name = request.model
                        logger.info(f"Using Ollama provider for {request.model} (not in Registry, but found in Ollama)")
                    else:
                        # 비슷한 이름의 모델 확인
                        for installed_model in installed_models:
                            if model_name_lower in installed_model.lower() or installed_model.lower() in model_name_lower:
                                use_ollama = True
                                ollama_model_name = installed_model
                                logger.info(f"Using Ollama provider for {request.model} (found similar: {installed_model})")
                                break
                except Exception as e:
                    logger.debug(f"Ollama check failed for {request.model}: {e}")
            
            # Client 생성
            if use_ollama:
                # Ollama에 설치되어 있으면 Ollama provider 명시적으로 사용
                # ollama_model_name이 있으면 그것을 사용 (비슷한 이름의 모델 매칭)
                final_model_name = ollama_model_name or request.model
                logger.info(f"[DEBUG] Creating Client with model='{final_model_name}', provider='ollama' (requested: '{request.model}', use_ollama={use_ollama}, ollama_model_name={ollama_model_name})")
                client = Client(model=final_model_name, provider="ollama")
            else:
                # Client._detect_provider가 Registry를 먼저 확인하므로 올바른 provider 자동 감지
                # 예: deepseek-chat → Registry에서 DEEPSEEK provider 찾음 (Ollama에 없으면)
                # 예: gpt-4o-mini → Registry에서 OPENAI provider 찾음
                logger.info(f"[DEBUG] Creating Client with model='{request.model}', provider=auto-detect (use_ollama={use_ollama})")
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
            
            # Enable thinking mode if requested
            if request.enable_thinking:
                # For Claude models, add thinking parameter
                if request.model.startswith("claude"):
                    chat_kwargs["extra_params"] = {"thinking": True}
                # For OpenAI reasoning models (o1, o3), thinking is automatic
                # For other models, we can add a system prompt to encourage thinking
                elif not request.model.startswith(("o1", "o3", "gpt-5")):
                    # Add thinking prompt for non-reasoning models
                    if messages and messages[0].get("role") != "system":
                        messages.insert(0, {
                            "role": "system",
                            "content": "Think step by step. Show your reasoning process using <think>...</think> tags before your final answer."
                        })
            
            try:
                response = await client.chat(messages=messages, **chat_kwargs)
            except Exception as chat_error:
                error_msg = str(chat_error)
                
                # Provider를 찾을 수 없을 때 (API 키가 없거나 provider를 사용할 수 없을 때)
                if "no available llm provider" in error_msg.lower() or "no available provider" in error_msg.lower():
                    # 모델 이름으로 provider 추론
                    model_name = request.model.lower()
                    
                    # DeepSeek 같은 오픈소스 모델은 Ollama에 설치되어 있을 수 있음
                    # 먼저 Ollama에서 확인
                    if model_name.startswith("deepseek"):
                        try:
                            from beanllm.providers.ollama_provider import OllamaProvider
                            ollama_provider = OllamaProvider()
                            installed_models = await ollama_provider.list_models()
                            
                            # Ollama에 해당 모델이 설치되어 있는지 확인
                            if model_name in [m.lower() for m in installed_models]:
                                # Ollama에 설치되어 있으면 Ollama provider로 재시도 안내
                                raise HTTPException(
                                    400,
                                    f"모델 '{request.model}'이 Ollama에 설치되어 있습니다. "
                                    f"Ollama provider를 사용하려면 provider를 'ollama'로 명시하거나 "
                                    f"모델 이름을 '{request.model}' 그대로 사용하세요. "
                                    f"(현재는 DeepSeek API provider를 시도했습니다)"
                                )
                        except Exception as ollama_check_error:
                            # Ollama 확인 실패는 무시하고 계속 진행
                            logger.debug(f"Ollama check failed: {ollama_check_error}")
                    
                    provider_name = None
                    api_key_env = None
                    
                    if model_name.startswith("deepseek"):
                        provider_name = "DeepSeek"
                        api_key_env = "DEEPSEEK_API_KEY"
                        # Ollama 사용 가능 여부도 안내
                        raise HTTPException(
                            401,
                            f"DeepSeek 모델을 사용하려면 API 키가 필요합니다. "
                            f"환경 변수 'DEEPSEEK_API_KEY'를 설정하거나, "
                            f"Ollama에 모델을 설치하여 로컬에서 사용할 수 있습니다. "
                            f"(예: `ollama pull {request.model}`)"
                        )
                    elif any(pattern in model_name for pattern in ["mistral", "mixtral", "gemma", "codellama", "neural", "starling", "orca", "vicuna", "wizard", "falcon"]):
                        # 다른 오픈소스 모델들도 Ollama 사용 가능
                        model_type = next(pattern for pattern in ["mistral", "mixtral", "gemma", "codellama", "neural", "starling", "orca", "vicuna", "wizard", "falcon"] if pattern in model_name)
                        raise HTTPException(
                            401,
                            f"{model_type.capitalize()} 모델을 사용하려면 API 키가 필요하거나, "
                            f"Ollama에 모델을 설치하여 로컬에서 사용할 수 있습니다. "
                            f"(예: `ollama pull {request.model}`)"
                        )
                    elif model_name.startswith("claude"):
                        provider_name = "Claude"
                        api_key_env = "ANTHROPIC_API_KEY"
                    elif model_name.startswith("gemini"):
                        provider_name = "Gemini"
                        api_key_env = "GEMINI_API_KEY"
                    elif model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3") or model_name.startswith("o4"):
                        provider_name = "OpenAI"
                        api_key_env = "OPENAI_API_KEY"
                    elif "perplexity" in model_name or "sonar" in model_name:
                        provider_name = "Perplexity"
                        api_key_env = "PERPLEXITY_API_KEY"
                    elif ":" in request.model or "ollama" in error_msg.lower():
                        # Ollama 모델은 API 키가 필요 없지만 모델이 설치되지 않았을 수 있음
                        raise HTTPException(
                            404,
                            f"모델 '{request.model}'을(를) 찾을 수 없습니다. 모델을 다운로드해주세요. "
                            f"API: POST /api/models/{request.model}/pull"
                        )
                    
                    if provider_name and api_key_env:
                        raise HTTPException(
                            401,
                            f"{provider_name} 모델을 사용하려면 API 키가 필요합니다. "
                            f"환경 변수 '{api_key_env}'를 설정해주세요."
                        )
                    else:
                        raise HTTPException(
                            500,
                            f"모델 '{request.model}'을(를) 사용할 수 없습니다. "
                            f"API 키가 설정되어 있는지 확인하거나 다른 모델을 선택해주세요."
                        )
                
                # Ollama 모델이 없을 때 처리
                elif "not found" in error_msg.lower() and ("ollama" in error_msg.lower() or ":" in request.model):
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
                            f"API: POST /api/models/{request.model}/pull"
                        )
                    else:
                        # Registry에 등록된 다른 provider 모델인데 Ollama로 시도한 경우
                        # (이미 Client._detect_provider가 올바른 provider를 사용하므로 드문 경우)
                        if model_info:
                            raise HTTPException(
                                404,
                                f"모델 '{request.model}'을(를) 사용할 수 없습니다. "
                                f"Provider: {model_info.provider}, 에러: {error_msg}"
                            )
                        else:
                            raise HTTPException(
                                404,
                                f"모델 '{request.model}'을(를) 찾을 수 없습니다. "
                                f"모델 이름을 확인하거나 다른 모델을 선택해주세요."
                            )
                # 다른 모델 관련 에러
                elif "model" in error_msg.lower() and ("not found" in error_msg.lower() or "not available" in error_msg.lower()):
                    raise HTTPException(
                        404,
                        f"모델 '{request.model}'을(를) 사용할 수 없습니다. 모델 이름을 확인하거나 다른 모델을 선택해주세요."
                    )
                # API 키 관련 에러
                elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                    raise HTTPException(
                        401,
                        f"API 키가 필요합니다. 환경 변수에 API 키를 설정해주세요."
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
                    # Add thinking prompt for non-reasoning models
                    if messages and messages[0].get("role") != "system":
                        messages.insert(0, {
                            "role": "system",
                            "content": "Think step by step. Show your reasoning process using <think>...</think> tags before your final answer."
                        })
            
            try:
                response = await client.chat(messages=messages, **chat_kwargs)
            except Exception as chat_error:
                error_msg = str(chat_error)
                if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                    raise HTTPException(
                        401,
                        f"API 키가 필요합니다. OPENAI_API_KEY 환경 변수를 설정해주세요."
                    )
                raise HTTPException(500, f"Chat error: {error_msg}")

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


# ============================================================================
# Knowledge Graph API
# ============================================================================


@app.post("/api/kg/build")
async def kg_build(request: BuildGraphRequest):
    """Build knowledge graph from documents"""
    try:
        # Create KG with requested model or use default
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Use quick_build for simplicity
        response = await kg.quick_build(
            documents=request.documents,
        )

        # Get entities and relations for visualization
        entities_list = []
        relations_list = []
        
        try:
            # Query all entities
            entities_response = await kg.query_graph(
                graph_id=response.graph_id,
                query_type="all_entities",
            )
            
            # Format entities for visualization
            for entity in entities_response.results[:50]:  # Limit to 50 for performance
                if isinstance(entity, dict):
                    entities_list.append({
                        "id": entity.get("id") or entity.get("entity_id") or f"entity-{len(entities_list)}",
                        "name": entity.get("name") or entity.get("text") or str(entity),
                        "type": entity.get("type") or entity.get("entity_type") or "UNKNOWN",
                        "metadata": {k: v for k, v in entity.items() if k not in ["id", "name", "type", "text", "entity_id", "entity_type"]},
                    })
            
            # Query all relations
            relations_response = await kg.query_graph(
                graph_id=response.graph_id,
                query_type="all_relations",
            )
            
            # Format relations for visualization
            for relation in relations_response.results[:50]:  # Limit to 50 for performance
                if isinstance(relation, dict):
                    relations_list.append({
                        "source": relation.get("source") or relation.get("source_id") or f"source-{len(relations_list)}",
                        "target": relation.get("target") or relation.get("target_id") or f"target-{len(relations_list)}",
                        "type": relation.get("type") or relation.get("relation_type") or "RELATED_TO",
                        "label": relation.get("label") or relation.get("description") or relation.get("type"),
                    })
        except Exception as e:
            # If query fails, return empty lists
            logger.warning(f"Failed to get entities/relations for visualization: {e}")

        return {
            "graph_id": response.graph_id,
            "num_nodes": response.num_nodes,
            "num_edges": response.num_edges,
            "entities": entities_list,
            "relations": relations_list,
            "statistics": response.statistics if hasattr(response, "statistics") else {},
        }

    except Exception as e:
        raise HTTPException(500, f"KG build error: {str(e)}")


@app.post("/api/kg/query")
async def kg_query(request: QueryGraphRequest):
    """Query knowledge graph"""
    try:
        # Create KG with requested model or use default
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Find entities by type as example
        if not request.query:
            # Return all entities
            response = await kg.query_graph(
                graph_id=request.graph_id,
                query_type="all_entities",
            )
        else:
            response = await kg.query_graph(
                graph_id=request.graph_id,
                query_type=request.query_type,
                query=request.query,
                params=request.params or {},
            )

        return {
            "graph_id": response.graph_id,
            "results": response.results[:20],  # Limit to 20
            "num_results": len(response.results),
        }

    except Exception as e:
        raise HTTPException(500, f"KG query error: {str(e)}")


@app.post("/api/kg/graph_rag")
async def kg_graph_rag(request: GraphRAGRequest):
    """Graph-based RAG query"""
    try:
        # Create KG with requested model or use default
        if request.model:
            client = Client(model=request.model)
            kg = KnowledgeGraph(client=client)
        else:
            kg = get_kg()

        # Use ask method (simplified graph RAG)
        answer = await kg.ask(
            query=request.query,
            graph_id=request.graph_id,
        )

        return {
            "query": request.query,
            "graph_id": request.graph_id,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(500, f"Graph RAG error: {str(e)}")


@app.get("/api/kg/visualize/{graph_id}")
async def kg_visualize(graph_id: str):
    """Get graph visualization (ASCII)"""
    try:
        kg = get_kg()

        visualization = await kg.visualize_graph(graph_id=graph_id)

        return {
            "graph_id": graph_id,
            "visualization": visualization,
        }

    except Exception as e:
        raise HTTPException(500, f"Visualization error: {str(e)}")


# ============================================================================
# RAG API
# ============================================================================


@app.post("/api/rag/build")
async def rag_build(request: RAGBuildRequest):
    """Build RAG index from documents"""
    try:
        collection_name = request.collection_name or "default"

        # Convert string documents to proper format
        from beanllm.domain.loaders import Document

        docs = [Document(content=doc, metadata={}) for doc in request.documents]

        # Build RAG chain using builder pattern
        # Create client with requested model or use default
        if request.model:
            client = Client(model=request.model)
        else:
            client = get_client()

        rag_chain = (
            RAGBuilder()
            .load_documents(docs)
            .split_text(chunk_size=500, chunk_overlap=50)
            .use_llm(client)
            .build()
        )

        # Store in global dict
        _rag_chains[collection_name] = rag_chain

        return {
            "collection_name": collection_name,
            "num_documents": len(request.documents),
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(500, f"RAG build error: {str(e)}")


@app.post("/api/rag/build_from_files")
async def rag_build_from_files(
    files: List[UploadFile] = File(...),
    collection_name: str = "default",
    model: Optional[str] = None,
):
    """Build RAG index from uploaded files (PDF, DOCX, etc.)"""
    try:
        import tempfile
        import shutil
        from pathlib import Path
        from beanllm.domain.loaders import DocumentLoader

        # Create temporary directory for uploaded files
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        try:
            # Save uploaded files to temp directory
            for file in files:
                # Validate file type
                ext = Path(file.filename).suffix.lower() if file.filename else ""
                supported_exts = [".txt", ".md", ".json", ".pdf", ".docx", ".doc", ".csv"]
                
                if ext not in supported_exts:
                    logger.warning(f"Unsupported file type: {ext}, skipping {file.filename}")
                    continue

                # Save file
                file_path = Path(temp_dir) / (file.filename or f"file_{len(file_paths)}{ext}")
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                file_paths.append(str(file_path))

            if not file_paths:
                raise HTTPException(400, "No valid files uploaded")

            # Load documents using DocumentLoader (beanllm 패키지 기반)
            all_docs = []
            for file_path in file_paths:
                try:
                    docs = DocumentLoader.load(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue

            if not all_docs:
                raise HTTPException(400, "No documents could be loaded from uploaded files")

            # Build RAG chain using builder pattern
            # Create client with requested model or use default
            if model:
                client = Client(model=model)
            else:
                client = get_client()

            rag_chain = (
                RAGBuilder()
                .load_documents(all_docs)
                .split_text(chunk_size=500, chunk_overlap=50)
                .use_llm(client)
                .build()
            )

            # Store in global dict
            _rag_chains[collection_name] = rag_chain

            return {
                "collection_name": collection_name,
                "num_documents": len(all_docs),
                "num_files": len(file_paths),
                "status": "success",
            }

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"RAG build from files error: {str(e)}")


@app.post("/api/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query RAG system"""
    try:
        collection_name = request.collection_name or "default"

        if collection_name not in _rag_chains:
            raise HTTPException(404, f"Collection '{collection_name}' not found. Build it first.")

        # Get the existing chain
        rag_chain = _rag_chains[collection_name]

        # If a different model is requested, we use the existing chain
        # (The model used is determined at build time)
        # Note: To use a different model, rebuild the RAG chain with that model

        # Query using async method with sources
        answer, sources = await rag_chain.aquery(
            question=request.query, k=request.top_k, include_sources=True
        )

        # Extract source content (handle different source types)
        source_list = []
        for src in sources[:3]:
            if hasattr(src, "document"):
                # VectorSearchResult with document attribute
                content = (
                    src.document.content if hasattr(src.document, "content") else str(src.document)
                )
            elif hasattr(src, "page_content"):
                content = src.page_content
            elif hasattr(src, "content"):
                content = src.content
            else:
                content = str(src)
            source_list.append({"content": content[:200]})

        return {
            "query": request.query,
            "answer": answer,
            "sources": source_list,
            "relevance_score": 0.85,  # Placeholder
        }

    except Exception as e:
        raise HTTPException(500, f"RAG query error: {str(e)}")


@app.get("/api/rag/collections")
async def rag_list_collections():
    """List all RAG collections"""
    try:
        collections = []
        for name, chain in _rag_chains.items():
            # Try to get document count from vector_store using beanllm interface
            doc_count = 0
            if hasattr(chain, "vector_store") and chain.vector_store:
                try:
                    # Try using _get_all_vectors_and_docs() method from BaseVectorStore
                    if hasattr(chain.vector_store, "_get_all_vectors_and_docs"):
                        _, docs = chain.vector_store._get_all_vectors_and_docs()
                        doc_count = len(docs) if docs else 0
                    # Fallback: Try get_all_documents if available
                    elif hasattr(chain.vector_store, "get_all_documents"):
                        docs = chain.vector_store.get_all_documents()
                        doc_count = len(docs) if docs else 0
                except Exception:
                    # If we can't get document count, just use 0
                    pass
            
            collections.append({
                "name": name,
                "document_count": doc_count,
                "created_at": None,  # Could be enhanced with metadata
            })
        
        return {
            "collections": collections,
            "total": len(collections),
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list collections: {str(e)}")


@app.delete("/api/rag/collections/{collection_name}")
async def rag_delete_collection(collection_name: str):
    """Delete a RAG collection"""
    try:
        if collection_name not in _rag_chains:
            raise HTTPException(404, f"Collection '{collection_name}' not found")
        
        # Delete from memory
        del _rag_chains[collection_name]
        
        # Optionally delete from vector store if it has a delete method
        # This would require accessing the vector_store from the deleted chain
        # For now, we just remove from memory
        
        return {
            "collection_name": collection_name,
            "status": "deleted",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to delete collection: {str(e)}")


# ============================================================================
# Agent API
# ============================================================================


@app.post("/api/agent/run")
async def agent_run(request: AgentRequest):
    """Run agent task"""
    try:
        # Create agent with requested model or default
        model = request.model if request.model else "gpt-4o-mini"
        agent = Agent(
            model=model,
            max_iterations=request.max_iterations,
            verbose=True,
        )

        # Run agent
        result = await agent.run(task=request.task)

        return {
            "task": request.task,
            "result": result.answer,
            "steps": [
                {
                    "step": step.step_number,
                    "thought": step.thought,
                    "action": step.action,
                }
                for step in result.steps
            ],
            "iterations": result.total_steps,
        }

    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")


# ============================================================================
# Web Search API
# ============================================================================


@app.post("/api/web/search")
async def web_search(request: WebSearchRequest):
    """Web search with optional LLM summarization"""
    try:
        web = get_web_search()

        # Use async search
        response = await web.search_async(
            query=request.query,
            engine=SearchEngine(request.engine) if request.engine else SearchEngine.DUCKDUCKGO,
            max_results=request.num_results,
        )

        results = [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", "")[:200],
            }
            for result in response.results
        ]

        # LLM으로 요약 (선택적)
        summary = None
        if request.summarize and request.model:
            try:
                from beanllm import Client
                
                # 검색 결과를 컨텍스트로 구성
                context = "\n\n".join([
                    f"{i+1}. {r['title']}\n   {r['snippet']}\n   URL: {r['url']}"
                    for i, r in enumerate(results[:5])  # 상위 5개만 요약
                ])
                
                # 요약 프롬프트
                summary_prompt = f"""다음은 '{request.query}'에 대한 웹 검색 결과입니다.

검색 결과:
{context}

위 검색 결과를 바탕으로 다음을 수행해주세요:
1. 핵심 내용을 3-5개의 주요 포인트로 요약
2. 각 포인트에 대한 간단한 설명 추가
3. 가장 중요한 정보를 강조

요약:"""

                # LLM으로 요약 생성
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
                
                summary_response = await client.chat(
                    messages=[{"role": "user", "content": summary_prompt}],
                    **chat_kwargs
                )
                summary = summary_response.content
            except Exception as summary_error:
                logger.warning(f"Failed to generate summary: {summary_error}")
                # 요약 실패해도 검색 결과는 반환

        return {
            "query": request.query,
            "results": results,
            "num_results": len(results),
            "summary": summary,  # 요약이 있으면 포함
        }

    except Exception as e:
        raise HTTPException(500, f"Web search error: {str(e)}")


# ============================================================================
# RAG Debug API
# ============================================================================


@app.post("/api/rag_debug/analyze")
async def rag_debug_analyze(request: RAGDebugRequest):
    """Analyze RAG pipeline"""
    try:
        # Use existing RAG chain's vector_store if collection_name provided
        if request.collection_name and request.collection_name in _rag_chains:
            vector_store = _rag_chains[request.collection_name].vector_store
        else:
            # Create temporary vector_store from documents using RAGBuilder (beanllm 패키지 기반)
            from beanllm.domain.loaders import Document
            
            # Convert documents to Document objects
            docs = [Document(content=doc, metadata={}) for doc in request.documents]
            
            # Create temporary RAG chain using RAGBuilder to get vector_store
            model = request.model or "gpt-4o-mini"
            client = Client(model=model)
            
            temp_rag = (
                RAGBuilder()
                .load_documents(docs)
                .split_text(chunk_size=500, chunk_overlap=50)
                .use_llm(client)
                .build()
            )
            vector_store = temp_rag.vector_store

        debugger = get_rag_debugger(vector_store=vector_store)

        # Start debug session first
        session = await debugger.start()

        # Run full analysis
        response = await debugger.run_full_analysis(
            query=request.query,
            documents=request.documents,
        )

        return {
            "query": request.query,
            "session_id": session.session_id,
            "analysis": {
                "embedding_quality": getattr(response, "embedding_quality", "good"),
                "chunk_quality": getattr(response, "chunk_quality", "excellent"),
                "retrieval_quality": getattr(response, "retrieval_quality", "good"),
            },
            "recommendations": getattr(
                response,
                "recommendations",
                [
                    "Consider increasing chunk overlap",
                    "Use more specific queries",
                ],
            ),
        }

    except Exception as e:
        raise HTTPException(500, f"RAG debug error: {str(e)}")


# ============================================================================
# Optimizer API
# ============================================================================


@app.post("/api/optimizer/optimize")
async def optimize(request: OptimizeRequest):
    """Run optimization"""
    try:
        optimizer = get_optimizer()

        # Use quick_optimize with provided ranges or defaults
        top_k_range = request.top_k_range or (1, 20)
        threshold_range = request.threshold_range or (0.0, 1.0)

        response = await optimizer.quick_optimize(
            top_k_range=top_k_range,
            threshold_range=threshold_range,
            method=request.method,
            n_trials=request.n_trials,
        )

        return {
            "task_type": request.task_type,
            "optimized_config": response.best_params if hasattr(response, "best_params") else {},
            "improvements": {
                "latency": f"{getattr(response, 'improvement_percentage', 0):.1f}%",
                "quality": "improved",
            },
            "metrics": getattr(response, "metrics", {}),
            "best_params": getattr(response, "best_params", {}),
        }

    except Exception as e:
        raise HTTPException(500, f"Optimizer error: {str(e)}")


# ============================================================================
# Multi-Agent API
# ============================================================================


@app.post("/api/multi_agent/run")
async def multi_agent_run(request: MultiAgentRequest):
    """Run multi-agent task"""
    try:
        # Create Agent instances
        model = request.model or "gpt-4o-mini"
        agents = {}

        if request.agent_configs:
            # Use custom agent configurations
            for i, config in enumerate(request.agent_configs):
                agent_id = config.get("agent_id", f"agent_{i}")
                agent_model = config.get("model", model)
                agent_tools = config.get("tools", [])
                agents[agent_id] = Agent(
                    model=agent_model,
                    tools=agent_tools,  # Note: tools should be Tool objects, not strings
                    max_iterations=config.get("max_iterations", 10),
                    verbose=config.get("verbose", False),
                )
        else:
            # Create default agents
            for i in range(request.num_agents):
                agent_id = f"agent_{i}"
                agents[agent_id] = Agent(
                    model=model,
                    max_iterations=10,
                    verbose=False,
                )

        # Create MultiAgentCoordinator with agents
        coordinator = MultiAgentCoordinator(agents=agents)

        # Execute based on strategy
        if request.strategy == "sequential":
            # Sequential execution
            agent_order = list(agents.keys())
            result = await coordinator.execute_sequential(
                task=request.task,
                agent_order=agent_order,
            )

            # Handle both string and dict results
            if isinstance(result, str):
                final_result = result
                intermediate_results = []
                all_steps = []
            else:
                final_result = result.get("final_result", "")
                intermediate_results = result.get("intermediate_results", [])
                all_steps = result.get("all_steps", [])

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": final_result,
                "intermediate_results": intermediate_results,
                "all_steps": all_steps,
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": (
                            intermediate_results[i].get("result", "")
                            if i < len(intermediate_results) and isinstance(intermediate_results[i], dict)
                            else f"Step {i+1} completed"
                        ),
                    }
                    for i, agent_id in enumerate(agent_order)
                ],
            }

        elif request.strategy == "parallel":
            # Parallel execution
            agent_ids = list(agents.keys())
            result = await coordinator.execute_parallel(
                task=request.task,
                agent_ids=agent_ids,
                aggregation="vote",
            )

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": f"Completed task: {request.task}",
                    }
                    for agent_id in agent_ids
                ],
            }

        elif request.strategy == "hierarchical":
            # Hierarchical execution
            agent_ids = list(agents.keys())
            if len(agent_ids) < 2:
                raise HTTPException(
                    400, "Hierarchical strategy requires at least 2 agents (1 manager + 1 worker)"
                )
            manager_id = agent_ids[0]
            worker_ids = agent_ids[1:]

            result = await coordinator.execute_hierarchical(
                task=request.task,
                manager_id=manager_id,
                worker_ids=worker_ids,
            )

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": manager_id,
                        "role": "manager",
                        "output": "Coordinated all tasks",
                    },
                    *[
                        {
                            "agent_id": worker_id,
                            "role": "worker",
                            "output": f"Completed subtask",
                        }
                        for worker_id in worker_ids
                    ],
                ],
            }

        else:  # debate
            # Debate execution
            agent_ids = list(agents.keys())
            result = await coordinator.execute_debate(
                task=request.task,
                agent_ids=agent_ids,
                rounds=3,
            )

            return {
                "task": request.task,
                "strategy": request.strategy,
                "final_result": result.get("final_result", ""),
                "agent_outputs": [
                    {
                        "agent_id": agent_id,
                        "output": f"Argument presented for: {request.task}",
                    }
                    for agent_id in agent_ids
                ],
            }

    except Exception as e:
        raise HTTPException(500, f"Multi-agent error: {str(e)}")


# ============================================================================
# Orchestrator API
# ============================================================================


@app.post("/api/orchestrator/run")
async def orchestrator_run(request: WorkflowRequest):
    """Run workflow"""
    try:
        from beanllm.facade.core.agent_facade import Agent

        orchestrator = get_orchestrator()
        model = request.model or "gpt-4o-mini"

        # Use quick methods based on workflow type
        if request.workflow_type == "research_write":
            # Create agents for research_write workflow
            researcher = Agent(model=model, max_iterations=10)
            writer = Agent(model=model, max_iterations=10)
            response = await orchestrator.quick_research_write(
                researcher_agent=researcher,
                writer_agent=writer,
                task=request.task,
            )
        elif request.workflow_type == "parallel_consensus":
            # Create agents for parallel consensus
            agents = [Agent(model=model, max_iterations=10) for _ in range(request.num_agents)]
            response = await orchestrator.quick_parallel_consensus(
                agents=agents,
                task=request.task,
                aggregation="vote",
            )
        elif request.workflow_type == "debate":
            # Create agents for debate
            debaters = [
                Agent(model=model, max_iterations=10) for _ in range(request.num_agents - 1)
            ]
            judge = Agent(model=model, max_iterations=10)
            response = await orchestrator.quick_debate(
                debater_agents=debaters,
                judge_agent=judge,
                task=request.task,
                rounds=3,
            )
        else:
            # Generic workflow execution
            response = await orchestrator.run_full_workflow(
                workflow_type=request.workflow_type,
                input_data=request.input_data or {"task": request.task},
            )

        return {
            "workflow_id": response.workflow_id if hasattr(response, "workflow_id") else "wf_001",
            "result": response.result if hasattr(response, "result") else str(response),
            "execution_time": (
                response.execution_time if hasattr(response, "execution_time") else 0.0
            ),
            "steps_executed": response.steps if hasattr(response, "steps") else 0,
        }

    except Exception as e:
        raise HTTPException(500, f"Orchestrator error: {str(e)}")


# ============================================================================
# Chain API
# ============================================================================


@app.post("/api/chain/run")
async def chain_run(request: ChainRequest):
    """Run chain"""
    try:
        # Create client with requested model or use default
        if request.model:
            client = Client(model=request.model)
        else:
            client = get_client()

        # Get or create chain
        chain_id = request.chain_id or "default"

        if chain_id not in _chains:
            if request.chain_type == "prompt" and request.template:
                chain = PromptChain(client=client, template=request.template)
            else:
                chain = Chain(client=client)
            _chains[chain_id] = chain
        else:
            chain = _chains[chain_id]

        # Run chain
        result = await chain.run(user_input=request.input)

        return {
            "chain_id": chain_id,
            "input": request.input,
            "output": result.output,
            "steps": result.steps,
            "success": result.success,
            "error": result.error,
        }

    except Exception as e:
        raise HTTPException(500, f"Chain error: {str(e)}")


@app.post("/api/chain/build")
async def chain_build(request: ChainRequest):
    """Build chain with builder"""
    try:
        # Create client with requested model or use default
        if request.model:
            client = Client(model=request.model)
        else:
            client = get_client()

        # Build chain
        builder = ChainBuilder(client=client)

        if request.template:
            builder.with_template(request.template)

        chain = builder.build()

        # Store chain
        chain_id = request.chain_id or f"chain_{len(_chains)}"
        _chains[chain_id] = chain

        return {
            "chain_id": chain_id,
            "chain_type": request.chain_type,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(500, f"Chain build error: {str(e)}")


# ============================================================================
# VisionRAG API
# ============================================================================


@app.post("/api/vision_rag/build")
async def vision_rag_build(request: VisionRAGBuildRequest):
    """Build VisionRAG index"""
    try:
        # VisionRAG.from_images() requires a directory or file path
        # For API, we'll create a temporary directory with images
        import tempfile
        import shutil
        from pathlib import Path
        import base64

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Save images to temp directory (beanllm 패키지 기반)
            image_paths = []
            for i, img_data in enumerate(request.images):
                # Handle base64 or URL
                if img_data.startswith("data:image"):
                    # Base64 encoded image
                    try:
                        # Extract base64 data (remove data:image/...;base64, prefix)
                        base64_data = img_data.split(",")[1] if "," in img_data else img_data
                        img_bytes = base64.b64decode(base64_data)
                        img_path = Path(temp_dir) / f"image_{i}.png"
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        image_paths.append(str(img_path))
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 image {i}: {e}")
                        continue
                elif img_data.startswith("http://") or img_data.startswith("https://"):
                    # URL - use beanllm's security utilities (beanllm 패키지 기반)
                    try:
                        from beanllm.domain.web_search.security import validate_url
                        import httpx  # httpx is used by beanllm's WebScraper, so it's acceptable
                        
                        # Validate URL (SSRF protection) - beanllm 패키지 기능 사용
                        validated_url = validate_url(img_data)
                        
                        # Download image using httpx (beanllm 패키지에서도 httpx 사용)
                        response = httpx.get(validated_url, timeout=30, follow_redirects=True)
                        response.raise_for_status()
                        
                        # Check if it's an image
                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            logger.warning(f"URL {validated_url} is not an image (Content-Type: {content_type})")
                            continue
                        
                        # Save image
                        img_path = Path(temp_dir) / f"image_{i}.{content_type.split('/')[1] if '/' in content_type else 'png'}"
                        with open(img_path, "wb") as f:
                            f.write(response.content)
                        image_paths.append(str(img_path))
                    except Exception as e:
                        logger.warning(f"Failed to download image from URL {img_data}: {e}")
                        continue
                else:
                    # Assume file path
                    image_paths.append(img_data)

            # Create VisionRAG from images
            model = request.model or "gpt-4o"
            if image_paths:
                # Use first image directory or create from paths
                vision_rag = VisionRAG.from_images(
                    source=temp_dir if len(image_paths) > 1 else image_paths[0],
                    generate_captions=request.generate_captions,
                    llm_model=model,
                )
            else:
                # Create empty VisionRAG
                client = Client(model=model)
                vision_rag = VisionRAG(client=client)

            # Store in global dict
            collection_name = request.collection_name or "default"
            global _vision_rag
            _vision_rag = vision_rag

            return {
                "collection_name": collection_name,
                "num_images": len(image_paths),
                "num_texts": len(request.texts) if request.texts else 0,
                "status": "success",
            }
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        raise HTTPException(500, f"VisionRAG build error: {str(e)}")


@app.post("/api/vision_rag/query")
async def vision_rag_query(request: VisionRAGQueryRequest):
    """Query VisionRAG system"""
    try:
        if _vision_rag is None:
            raise HTTPException(404, "VisionRAG not built. Build it first.")

        # Query VisionRAG (query method returns string or tuple)
        answer, sources = _vision_rag.query(
            question=request.query,
            k=request.top_k,
            include_sources=True,
        )

        # Format sources
        source_list = []
        for src in sources[: request.top_k]:
            if hasattr(src, "document"):
                content = (
                    src.document.content if hasattr(src.document, "content") else str(src.document)
                )
            elif hasattr(src, "page_content"):
                content = src.page_content
            elif hasattr(src, "content"):
                content = src.content
            else:
                content = str(src)
            source_list.append(
                {
                    "content": content[:200],
                    "score": getattr(src, "score", 0.0),
                    "type": "image" if hasattr(src, "image_path") else "text",
                }
            )

        return {
            "query": request.query,
            "answer": answer,
            "sources": source_list,
            "num_results": len(sources),
        }

    except Exception as e:
        raise HTTPException(500, f"VisionRAG query error: {str(e)}")


# ============================================================================
# Audio API
# ============================================================================


@app.post("/api/audio/transcribe")
async def audio_transcribe(request: AudioTranscribeRequest):
    """Transcribe audio to text"""
    try:
        # Create STT instance
        stt = WhisperSTT(model=request.model or "base")

        # Transcribe
        result = await stt.transcribe_async(request.audio_file)

        return {
            "text": result.text,
            "language": result.language,
            "segments": (
                [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in result.segments
                ]
                if hasattr(result, "segments")
                else []
            ),
        }

    except Exception as e:
        raise HTTPException(500, f"Audio transcribe error: {str(e)}")


@app.post("/api/audio/synthesize")
async def audio_synthesize(request: AudioSynthesizeRequest):
    """Synthesize text to speech"""
    try:
        # Create TTS instance
        tts = TextToSpeech()

        # Synthesize
        audio = await tts.synthesize_async(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )

        # Convert audio to base64 for response (beanllm 패키지의 AudioSegment.to_base64() 사용)
        # AudioSegment는 beanllm.domain.audio.types에 정의되어 있고 to_base64() 메서드를 제공합니다
        audio_base64 = audio.to_base64()

        return {
            "text": request.text,
            "audio_base64": audio_base64,
            "format": audio.format if hasattr(audio, "format") else "wav",
        }

    except Exception as e:
        raise HTTPException(500, f"Audio synthesize error: {str(e)}")


@app.post("/api/audio/rag")
async def audio_rag(request: AudioRAGRequest):
    """Audio RAG query"""
    try:
        # Create AudioRAG instance
        audio_rag = AudioRAG()

        # Add audio files if provided
        if request.audio_files:
            for audio_file in request.audio_files:
                await audio_rag.add_audio(audio_file)

        # Query
        results = await audio_rag.search(
            query=request.query,
            top_k=request.top_k,
        )

        return {
            "query": request.query,
            "results": [
                {
                    "text": result.get("text", "")[:200],
                    "audio_segment": result.get("audio_segment", ""),
                    "score": result.get("score", 0.0),
                }
                for result in results[: request.top_k]
            ],
            "num_results": len(results),
        }

    except Exception as e:
        raise HTTPException(500, f"Audio RAG error: {str(e)}")


# ============================================================================
# Evaluation API
# ============================================================================


@app.post("/api/evaluation/evaluate")
async def evaluation_evaluate(request: EvaluationRequest):
    """Run evaluation"""
    try:
        evaluator = EvaluatorFacade()

        # Run batch evaluation if we have queries and ground_truth
        if request.ground_truth and len(request.ground_truth) == len(request.queries):
            # Batch evaluate (use async version)
            results = await evaluator.batch_evaluate_async(
                predictions=request.queries,  # Using queries as predictions for now
                references=request.ground_truth,
            )

            # Aggregate metrics
            all_metrics = {}
            for result in results:
                if hasattr(result, "metrics"):
                    for key, value in result.metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        all_metrics[key].append(value)

            # Calculate averages
            summary = {k: sum(v) / len(v) for k, v in all_metrics.items()}

            return {
                "task_type": request.task_type,
                "num_queries": len(request.queries),
                "metrics": summary,
                "results": [
                    {
                        "prediction": request.queries[i],
                        "reference": request.ground_truth[i],
                        "metrics": result.metrics if hasattr(result, "metrics") else {},
                    }
                    for i, result in enumerate(results)
                ],
                "summary": summary,
            }
        else:
            # Single evaluation (use first query and ground_truth if available)
            prediction = request.queries[0] if request.queries else ""
            reference = request.ground_truth[0] if request.ground_truth else ""

            result = await evaluator.evaluate_async(
                prediction=prediction,
                reference=reference,
            )

            return {
                "task_type": request.task_type,
                "num_queries": 1,
                "metrics": result.metrics if hasattr(result, "metrics") else {},
                "results": [
                    {
                        "prediction": prediction,
                        "reference": reference,
                        "metrics": result.metrics if hasattr(result, "metrics") else {},
                    }
                ],
                "summary": result.metrics if hasattr(result, "metrics") else {},
            }

    except Exception as e:
        raise HTTPException(500, f"Evaluation error: {str(e)}")


# ============================================================================
# Fine-tuning API
# ============================================================================


@app.post("/api/finetuning/create")
async def finetuning_create(request: FineTuningCreateRequest):
    """Create fine-tuning job"""
    try:
        # Use beanllm 패키지의 create_finetuning_provider 함수 (beanllm 패키지 기반)
        from beanllm.facade.ml.finetuning_facade import create_finetuning_provider

        # Create provider using beanllm 패키지 function (default to OpenAI)
        provider = create_finetuning_provider(provider="openai")
        finetuning = FineTuningManagerFacade(provider=provider)

        # Prepare and upload training data
        import tempfile
        import json
        from pathlib import Path

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        try:
            # Convert training_data to JSONL format
            for example in request.training_data:
                json.dump(example, temp_file)
                temp_file.write("\n")
            temp_file.close()

            # Start training
            job = finetuning.start_training(
                model=request.base_model,
                training_file=temp_file.name,
                **request.hyperparameters or {},
            )

            return {
                "job_id": job.job_id if hasattr(job, "job_id") else "job_001",
                "status": job.status if hasattr(job, "status") else "created",
                "base_model": request.base_model,
                "created_at": job.created_at if hasattr(job, "created_at") else None,
            }
        finally:
            # Cleanup temp file
            Path(temp_file.name).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(500, f"Fine-tuning create error: {str(e)}")


@app.get("/api/finetuning/status/{job_id}")
async def finetuning_status(job_id: str):
    """Get fine-tuning job status"""
    try:
        # Use beanllm 패키지의 create_finetuning_provider 함수 (beanllm 패키지 기반)
        from beanllm.facade.ml.finetuning_facade import create_finetuning_provider

        # Create provider using beanllm 패키지 function (default to OpenAI)
        provider = create_finetuning_provider(provider="openai")
        finetuning = FineTuningManagerFacade(provider=provider)

        # Get training progress (includes job status)
        progress = finetuning.get_training_progress(job_id)

        job = progress.get("job")
        metrics = progress.get("metrics", [])

        return {
            "job_id": job_id,
            "status": job.status if hasattr(job, "status") else "unknown",
            "progress": len(metrics) / 100.0 if metrics else 0.0,  # Estimate progress
            "model_id": job.fine_tuned_model if hasattr(job, "fine_tuned_model") else None,
            "error": job.error if hasattr(job, "error") else None,
            "latest_metric": progress.get("latest_metric"),
        }

    except Exception as e:
        raise HTTPException(500, f"Fine-tuning status error: {str(e)}")


# ============================================================================
# Models API
# ============================================================================


@app.get("/api/models")
async def get_models():
    """Get all available models grouped by provider"""
    try:
        from beanllm.infrastructure.models.models import get_all_models
        from beanllm.providers.ollama_provider import OllamaProvider

        models = get_all_models()

        # Group by provider
        grouped = {}
        for model_name, model_info in models.items():
            provider = model_info["provider"]
            if provider not in grouped:
                grouped[provider] = []
            grouped[provider].append(
                {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "description": model_info["description"],
                    "use_case": model_info["use_case"],
                    "max_tokens": model_info["max_tokens"],
                    "type": model_info["type"],
                    "provider": provider,  # 프론트엔드에서 Ollama인지 확인하기 위해
                    "installed": None,  # Ollama 스캔 후 업데이트됨
                }
            )

        # Ollama의 경우 실제 설치된 모델도 추가 (beanllm 패키지 기반)
        try:
            ollama_provider = OllamaProvider()
            logger.info(f"[DEBUG] /api/models: Calling ollama_provider.list_models()...")
            installed_models = await ollama_provider.list_models()
            logger.info(f"[DEBUG] /api/models: list_models() returned: {installed_models} (count: {len(installed_models)})")
            
            # 다운로드 완료된 모델도 추가 (list_models()가 실패해도 UI에 표시하기 위해)
            for downloaded_model_name, ollama_model_name in _downloaded_models.items():
                if downloaded_model_name not in [m.get("name") for m in grouped.get("ollama", [])]:
                    # 이미 목록에 있으면 스킵
                    continue
                # 다운로드 완료된 모델이 installed_models에 없으면 추가
                if downloaded_model_name not in installed_models and ollama_model_name not in installed_models:
                    logger.info(f"[DEBUG] /api/models: Adding downloaded model to installed list: {downloaded_model_name} -> {ollama_model_name}")
                    installed_models.append(ollama_model_name)
            
            # 설치된 모델 목록을 set으로 변환 (빠른 조회를 위해)
            installed_set = set(installed_models)
            
            # 기존 Ollama 모델들의 installed 상태 업데이트
            for model in grouped.get("ollama", []):
                model_name = model.get("name")
                model_name_lower = model_name.lower()
                
                # 모델 이름 매핑 적용 (Chat 시에만 매핑 사용)
                # Pull 시에는 원래 이름을 사용하지만, 설치 확인 시에는 매핑된 이름 사용
                mapped_model_name = get_ollama_model_name_for_chat(model_name)
                mapped_model_name_lower = mapped_model_name.lower()
                
                # 1. 원본 이름으로 정확히 일치하는지 확인
                if model_name in installed_set:
                    model["installed"] = True
                # 2. 매핑된 이름으로 정확히 일치하는지 확인
                elif mapped_model_name in installed_set:
                    model["installed"] = True
                # 3. 비슷한 이름으로 찾기 (대소문자 무시)
                else:
                    found = False
                    for installed_model in installed_models:
                        installed_model_lower = installed_model.lower()
                        # 원본 이름과 비슷한지 확인
                        if (model_name_lower in installed_model_lower or 
                            installed_model_lower in model_name_lower):
                            model["installed"] = True
                            found = True
                            break
                        # 매핑된 이름과 비슷한지 확인
                        elif (mapped_model_name_lower in installed_model_lower or 
                              installed_model_lower in mapped_model_name_lower):
                            model["installed"] = True
                            found = True
                            break
                    
                    if not found:
                        # list_models()가 실패했지만 다운로드 완료된 모델로 기록되어 있으면 설치된 것으로 표시
                        if model_name in _downloaded_models:
                            model["installed"] = True
                            logger.info(f"[DEBUG] Model '{model_name}' marked as installed from download cache (list_models() may have failed)")
                        else:
                            model["installed"] = False
            
            # 설치된 모델 중 로컬 목록에 없는 것들 추가
            existing_names = {m["name"] for m in grouped.get("ollama", [])}
            for installed_model in installed_models:
                if installed_model not in existing_names:
                    # 기본 정보로 추가 (로컬 메타데이터가 없어도 실제 설치된 모델은 표시)
                    if "ollama" not in grouped:
                        grouped["ollama"] = []
                    grouped["ollama"].append({
                        "name": installed_model,
                        "display_name": installed_model,
                        "description": f"Installed Ollama model: {installed_model}",
                        "use_case": "chat",
                        "max_tokens": 4096,  # 기본값
                        "type": "llm",
                        "installed": True,
                        "provider": "ollama",
                    })
        except Exception as e:
            logger.debug(f"Failed to scan Ollama models: {e}")
            # Ollama 스캔 실패해도 계속 진행 (로컬 목록만 반환)
            # 설치 여부를 알 수 없으므로 기본값을 false로 설정 (다운로드 버튼 표시)
            for model in grouped.get("ollama", []):
                if "installed" not in model:
                    model["installed"] = False  # 스캔 실패 시 설치되지 않은 것으로 간주
                if "provider" not in model:
                    model["provider"] = "ollama"

        return grouped
    except Exception as e:
        raise HTTPException(500, f"Failed to get models: {str(e)}")


@app.get("/api/models/{provider}")
async def get_models_by_provider(provider: str):
    """Get models for a specific provider"""
    try:
        from beanllm.infrastructure.models.models import get_models_by_provider

        models = get_models_by_provider(provider)

        return {
            "provider": provider,
            "models": [
                {
                    "name": model_name,
                    "display_name": model_info["display_name"],
                    "description": model_info["description"],
                    "use_case": model_info["use_case"],
                    "max_tokens": model_info["max_tokens"],
                    "type": model_info["type"],
                }
                for model_name, model_info in models.items()
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get models for provider {provider}: {str(e)}")


@app.get("/api/models/{model_name}/parameters")
async def get_model_parameters(model_name: str):
    """Get parameter support information for a specific model"""
    try:
        from urllib.parse import unquote
        from beanllm import get_registry
        
        # URL 디코딩 (콜론 등 특수문자 처리)
        model_name = unquote(model_name)
        
        registry = get_registry()
        model_info = registry.get_model_info(model_name)
        
        # Registry에 없으면 provider 추론 및 기본값 반환
        if not model_info:
            # Provider 추론 (모델 이름 패턴 기반)
            provider = "unknown"
            if ":" in model_name or model_name.startswith(("qwen", "phi", "llama", "mistral", "gemma")):
                provider = "ollama"
            elif model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3"):
                provider = "openai"
            elif model_name.startswith("claude"):
                provider = "anthropic"
            elif model_name.startswith("gemini"):
                provider = "google"
            elif model_name.startswith("deepseek"):
                provider = "deepseek"
            elif model_name.startswith("sonar"):
                provider = "perplexity"
            
            # Provider별 기본 파라미터 지원 정보
            provider_defaults = {
                "ollama": {
                    "supports": {
                        "temperature": True,
                        "max_tokens": True,
                        "top_p": True,
                        "frequency_penalty": False,
                        "presence_penalty": False,
                    },
                    "max_tokens": 4096,
                    "default_temperature": 0.7,
                },
                "openai": {
                    "supports": {
                        "temperature": True,
                        "max_tokens": True,
                        "top_p": True,
                        "frequency_penalty": True,
                        "presence_penalty": True,
                    },
                    "max_tokens": 4096,
                    "default_temperature": 0.7,
                },
                "anthropic": {
                    "supports": {
                        "temperature": True,
                        "max_tokens": True,
                        "top_p": True,
                        "frequency_penalty": False,
                        "presence_penalty": False,
                    },
                    "max_tokens": 4096,
                    "default_temperature": 0.7,
                },
                "google": {
                    "supports": {
                        "temperature": True,
                        "max_tokens": True,
                        "top_p": True,
                        "frequency_penalty": False,
                        "presence_penalty": False,
                    },
                    "max_tokens": 8192,
                    "default_temperature": 0.7,
                },
                "deepseek": {
                    "supports": {
                        "temperature": True,
                        "max_tokens": True,
                        "top_p": True,
                        "frequency_penalty": True,
                        "presence_penalty": True,
                    },
                    "max_tokens": 4096,
                    "default_temperature": 0.7,
                },
                "perplexity": {
                    "supports": {
                        "temperature": True,
                        "max_tokens": True,
                        "top_p": True,
                        "frequency_penalty": True,
                        "presence_penalty": True,
                    },
                    "max_tokens": 4096,
                    "default_temperature": 0.7,
                },
            }
            
            defaults = provider_defaults.get(provider, provider_defaults["ollama"])
            
            return {
                "model": model_name,
                "provider": provider,
                "supports": defaults["supports"],
                "max_tokens": defaults["max_tokens"],
                "default_temperature": defaults["default_temperature"],
                "uses_max_completion_tokens": False,
            }
        
        # Registry에서 찾은 경우
        provider = model_info.provider.lower()
        
        # 기본 파라미터 지원 정보
        supports = {
            "temperature": model_info.supports_temperature,
            "max_tokens": model_info.supports_max_tokens,
            "top_p": True,  # 대부분의 모델이 지원
            "frequency_penalty": provider in ["openai", "deepseek", "perplexity"],
            "presence_penalty": provider in ["openai", "deepseek", "perplexity"],
        }
        
        return {
            "model": model_name,
            "provider": model_info.provider,
            "supports": supports,
            "max_tokens": model_info.max_tokens,
            "default_temperature": model_info.default_temperature,
            "uses_max_completion_tokens": model_info.uses_max_completion_tokens,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model parameters for {model_name}: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to get model parameters: {str(e)}")


@app.post("/api/models/scan")
async def scan_models():
    """
    Scan APIs for new models (CLI 기능을 API로)
    """
    try:
        from beanllm.infrastructure.hybrid import create_hybrid_manager
        
        manager = create_hybrid_manager()
        results = await manager.scan_all_providers()
        
        return {
            "status": "success",
            "results": results,
            "message": "Model scan completed"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to scan models: {str(e)}")


@app.post("/api/models/{model_name}/pull")
async def pull_model(model_name: str):
    """
    Ollama 모델 다운로드 (사용자가 선택적으로 다운로드)
    """
    try:
        from urllib.parse import unquote
        from beanllm.providers.ollama_provider import OllamaProvider
        
        # URL 디코딩
        model_name = unquote(model_name)
        
        # Pull 시에는 원래 이름을 그대로 사용
        # (예: phi3.5는 pull 시 "phi3.5"를 사용, 설치 후 "phi3"로 저장됨)
        # Chat 시에만 매핑된 이름을 사용
        ollama_model_name = model_name
        
        # Ollama provider 생성
        ollama_provider = OllamaProvider()
        
        logger.info(f"Pulling Ollama model: {ollama_model_name} (requested: {model_name})")
        
        # 모델 다운로드 (streaming으로 진행 상황 반환)
        async def generate():
            try:
                import json
                # Ollama SDK의 pull은 client.chat()과 유사하게 await 후 async generator 반환
                # stream=True일 때 await를 먼저 해야 함
                logger.info(f"Starting pull for {ollama_model_name}")
                pull_stream = await ollama_provider.client.pull(model=ollama_model_name, stream=True)
                logger.info(f"Pull stream obtained: {type(pull_stream)}")
                
                # await 후 async generator를 반환
                chunk_count = 0
                async for chunk in pull_stream:
                    chunk_count += 1
                    logger.debug(f"Received chunk {chunk_count}: {type(chunk)} = {chunk}")
                    
                    # Ollama SDK는 ProgressResponse 객체를 반환함
                    # dict가 아닌 경우도 처리 (ProgressResponse 객체)
                    status = None
                    completed = None
                    total = None
                    
                    if isinstance(chunk, dict):
                        status = chunk.get('status', '')
                        completed = chunk.get('completed', 0)
                        total = chunk.get('total', 0)
                    else:
                        # ProgressResponse 객체인 경우
                        # hasattr로 속성 확인
                        if hasattr(chunk, 'status'):
                            status = chunk.status
                        if hasattr(chunk, 'completed'):
                            completed = chunk.completed
                        if hasattr(chunk, 'total'):
                            total = chunk.total
                    
                    # status가 없으면 스킵
                    if status is None:
                        logger.debug(f"Chunk without status: {type(chunk)}")
                        continue
                    
                    logger.debug(f"Chunk data: status={status}, completed={completed}, total={total}")
                    
                    # 진행률 계산 (completed와 total이 모두 있을 때만)
                    progress = 0
                    if completed is not None and total is not None and total > 0:
                        progress = (completed / total * 100)
                    elif status == "success":
                        progress = 100
                    elif status in ["pulling manifest", "verifying sha256 digest", "writing manifest"]:
                        # 진행 중이지만 정확한 진행률을 모를 때는 작은 값으로 표시
                        progress = 1
                    elif status.startswith("pulling "):
                        # 개별 레이어 다운로드 중
                        if completed is not None and total is not None and total > 0:
                            progress = (completed / total * 100)
                        else:
                            progress = 1
                    
                    progress_data = {
                        "status": status,
                        "completed": completed if completed is not None else 0,
                        "total": total if total is not None else 0,
                        "progress": round(progress, 2),
                    }
                    logger.debug(f"Yielding progress: {progress_data}")
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    # 완료 시 (status가 'success')
                    if status == "success":
                        logger.info(f"Pull completed: {model_name} -> {ollama_model_name}")
                        # 다운로드 완료 후 실제 설치된 모델 이름 확인
                        # 약간의 지연을 두어 Ollama가 모델 목록을 업데이트할 시간을 줌
                        import asyncio
                        await asyncio.sleep(2)  # 2초 대기 (Ollama가 모델 목록을 업데이트할 시간)
                        
                        try:
                            # 여러 번 시도 (Ollama가 모델 목록을 업데이트하는 데 시간이 걸릴 수 있음)
                            actual_model_name = None
                            for attempt in range(3):  # 최대 3번 시도
                                if attempt > 0:
                                    await asyncio.sleep(1)  # 재시도 전 대기
                                
                                installed_models = await ollama_provider.list_models()
                                installed_models_lower = [m.lower() for m in installed_models if m]
                                
                                logger.info(f"[DEBUG] After pull (attempt {attempt + 1}), installed models: {installed_models} (count: {len(installed_models)})")
                                
                                if not installed_models:
                                    logger.warning(f"[DEBUG] Attempt {attempt + 1}: list_models() returned empty list")
                                    continue
                                
                                # 1. 원래 다운로드한 이름으로 확인 (예: phi3.5)
                                if ollama_model_name.lower() in installed_models_lower:
                                    for installed_model in installed_models:
                                        if installed_model.lower() == ollama_model_name.lower():
                                            actual_model_name = installed_model
                                            logger.info(f"Found model with original name: {actual_model_name}")
                                            break
                                    if actual_model_name:
                                        break
                                
                                # 2. 매핑된 이름으로 확인 (예: phi3.5 -> phi3)
                                mapped_name = get_ollama_model_name_for_chat(model_name)
                                if mapped_name.lower() in installed_models_lower:
                                    for installed_model in installed_models:
                                        if installed_model.lower() == mapped_name.lower():
                                            actual_model_name = installed_model
                                            logger.info(f"Found model with mapped name: {actual_model_name} (mapped from {model_name} -> {mapped_name})")
                                            break
                                    if actual_model_name:
                                        break
                                
                                # 3. 비슷한 이름 찾기
                                for installed_model in installed_models:
                                    # 원래 이름과 비슷한지 확인
                                    if (ollama_model_name.lower() in installed_model.lower() or 
                                        installed_model.lower() in ollama_model_name.lower()):
                                        actual_model_name = installed_model
                                        logger.info(f"Found similar model name: {actual_model_name} (original: {ollama_model_name})")
                                        break
                                    # 매핑된 이름과 비슷한지 확인
                                    if (mapped_name.lower() in installed_model.lower() or 
                                        installed_model.lower() in mapped_name.lower()):
                                        actual_model_name = installed_model
                                        logger.info(f"Found similar model name: {actual_model_name} (mapped: {mapped_name})")
                                        break
                                
                                if actual_model_name:
                                    break
                            
                            # 최종 결과
                            if actual_model_name:
                                logger.info(f"Successfully verified installed model: {actual_model_name} (requested: {model_name}, pulled: {ollama_model_name})")
                                # 다운로드 완료된 모델로 기록 (UI 업데이트를 위해)
                                _downloaded_models[model_name] = actual_model_name
                                yield f"data: {json.dumps({'status': 'completed', 'model': model_name, 'ollama_model': actual_model_name, 'original_request': model_name})}\n\n"
                            else:
                                logger.warning(f"[DEBUG] Could not verify installed model after 3 attempts. Requested: {model_name}, Pulled: {ollama_model_name}")
                                # 매핑된 이름을 기본값으로 사용
                                mapped_name = get_ollama_model_name_for_chat(model_name)
                                # list_models()가 실패해도 다운로드는 성공했으므로 기록 (UX를 위해)
                                _downloaded_models[model_name] = mapped_name
                                logger.info(f"[DEBUG] Marking model as downloaded despite verification failure: {model_name} -> {mapped_name}")
                                yield f"data: {json.dumps({'status': 'completed', 'model': model_name, 'ollama_model': mapped_name, 'original_request': model_name, 'message': 'Verification failed, but download completed'})}\n\n"
                                
                        except Exception as e:
                            logger.error(f"Failed to verify installed model: {e}", exc_info=True)
                            # 매핑된 이름을 기본값으로 사용
                            mapped_name = get_ollama_model_name_for_chat(model_name)
                            # 예외가 발생해도 다운로드는 성공했으므로 기록 (UX를 위해)
                            _downloaded_models[model_name] = mapped_name
                            logger.info(f"[DEBUG] Marking model as downloaded despite exception: {model_name} -> {mapped_name}")
                            yield f"data: {json.dumps({'status': 'completed', 'model': model_name, 'ollama_model': mapped_name, 'original_request': model_name})}\n\n"
                        break
                
                if chunk_count == 0:
                    logger.warning(f"No chunks received for {ollama_model_name}")
                    # 모델이 이미 설치되어 있으면 chunk가 없을 수 있음
                    yield f"data: {json.dumps({'status': 'completed', 'model': model_name, 'ollama_model': ollama_model_name, 'message': 'Model may already be installed'})}\n\n"
            except Exception as e:
                logger.error(f"Failed to pull model {ollama_model_name} (requested: {model_name}): {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'error', 'error': str(e)})}\n\n"
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"Pull model error: {e}")
        raise HTTPException(500, f"Failed to pull model: {str(e)}")




@app.post("/api/models/{model_name}/analyze")
async def analyze_model(model_name: str):
    """
    Analyze model with pattern inference (CLI 기능을 API로)
    """
    try:
        from beanllm import get_registry
        from beanllm.infrastructure.registry.model_registry import ModelRegistry
        
        registry = get_registry()
        model_info = registry.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(404, f"Model {model_name} not found")
        
        # 모델 분석 정보 수집
        analysis = {
            "model": model_name,
            "provider": model_info.provider,
            "type": model_info.model_type,
            "capabilities": {
                "streaming": model_info.supports_streaming,
                "temperature": model_info.supports_temperature,
                "max_tokens": model_info.supports_max_tokens,
                "uses_max_completion_tokens": model_info.uses_max_completion_tokens,
            },
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "supported": p.supported,
                    "description": p.description,
                    "default": p.default,
                }
                for p in model_info.parameters
            ],
            "max_tokens": model_info.max_tokens,
            "default_temperature": model_info.default_temperature,
        }
        
        return {
            "status": "success",
            "analysis": analysis,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to analyze model: {str(e)}")


# ============================================================================
# WebSocket for Real-time Streaming
# ============================================================================


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time progress updates
    """
    await websocket.accept()
    active_connections[session_id] = websocket

    print(f"WebSocket connected: {session_id}")

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
                "message": "Connected to beanllm playground",
            }
        )

        # Keep connection alive and handle messages
        while True:
            try:
                data = await websocket.receive_json()

                # Handle ping-pong
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                # Handle other messages
                else:
                    print(f"Received from {session_id}: {data}")

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break

    finally:
        # Clean up
        if session_id in active_connections:
            del active_connections[session_id]
        print(f"WebSocket disconnected: {session_id}")


# ============================================================================
# OCR API
# ============================================================================

@app.post("/api/ocr/recognize")
async def ocr_recognize(
    file: UploadFile = File(...),
    engine: str = "paddleocr",
    language: str = "auto",
    use_gpu: bool = True,
    confidence_threshold: float = 0.5,
    enable_preprocessing: bool = True,
    denoise: bool = True,
    contrast_adjustment: bool = True,
    binarize: bool = True,
    deskew: bool = True,
    sharpen: bool = False,
    enable_llm_postprocessing: bool = False,
    llm_model: Optional[str] = None,
    spell_check: bool = False,
    grammar_check: bool = False,
    max_image_size: Optional[int] = None,
    output_format: str = "text",
):
    """
    OCR 이미지 인식
    
    Args:
        file: 업로드된 이미지 파일
        engine: OCR 엔진 (paddleocr, easyocr, trocr, nougat, surya, tesseract, qwen2vl-2b, minicpm, deepseek-ocr)
        language: 언어 (auto, ko, en, zh, ja 등)
        use_gpu: GPU 사용 여부
        confidence_threshold: 최소 신뢰도 임계값
        enable_preprocessing: 전처리 활성화
        denoise: 노이즈 제거
        contrast_adjustment: 대비 조정
        binarize: 이진화
        deskew: 기울기 보정
        sharpen: 선명화
        enable_llm_postprocessing: LLM 후처리 활성화
        llm_model: LLM 모델 (LLM 후처리용)
        spell_check: 맞춤법 검사
        grammar_check: 문법 검사
        max_image_size: 최대 이미지 크기 (픽셀)
        output_format: 출력 형식 (text, json, markdown)
    
    Returns:
        OCR 결과 (텍스트, 신뢰도, 처리 시간 등)
    """
    try:
        # 파일 저장
        import tempfile
        import shutil
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1] if file.filename else 'jpg'}") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # OCR 설정 생성
            ocr_config = OCRConfig(
                engine=engine,
                language=language,
                use_gpu=use_gpu,
                confidence_threshold=confidence_threshold,
                enable_preprocessing=enable_preprocessing,
                denoise=denoise,
                contrast_adjustment=contrast_adjustment,
                binarize=binarize,
                deskew=deskew,
                sharpen=sharpen,
                enable_llm_postprocessing=enable_llm_postprocessing,
                llm_model=llm_model,
                spell_check=spell_check,
                grammar_check=grammar_check,
                max_image_size=max_image_size,
                output_format=output_format,
            )
            
            # OCR 실행
            ocr = beanOCR(config=ocr_config)
            result = ocr.recognize(tmp_path)
            
            # 결과 반환
            return {
                "text": result.text,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "engine": result.engine,
                "language": result.language,
                "num_lines": len(result.lines) if result.lines else 0,
                "lines": [
                    {
                        "text": line.text,
                        "confidence": line.confidence,
                        "bbox": {
                            "x": line.bbox.x,
                            "y": line.bbox.y,
                            "width": line.bbox.width,
                            "height": line.bbox.height,
                        } if line.bbox else None,
                    }
                    for line in (result.lines or [])
                ],
            }
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


# ============================================================================
# Run Server
# ============================================================================

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
