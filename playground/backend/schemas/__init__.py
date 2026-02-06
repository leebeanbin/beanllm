"""
Pydantic Schemas for BeanLLM Playground API

Organized by feature domain.
"""

from schemas.agent import AgentRequest
from schemas.audio import AudioRAGRequest, AudioSynthesizeRequest, AudioTranscribeRequest
from schemas.chat import ChatRequest, Message
from schemas.database import (
    PROVIDER_CONFIG,
    AddMessageRequest,
    ApiKeyBase,
    ApiKeyCreate,
    ApiKeyInDB,
    ApiKeyListResponse,
    ApiKeyResponse,
    ApiKeyValidationResult,
    ChatMessage,
    ChatSession,
    CreateSessionRequest,
    GoogleAuthStatus,
    GoogleOAuthToken,
    ProviderInfo,
    ProviderListResponse,
    RequestLog,
    SessionListResponse,
    SessionResponse,
)
from schemas.evaluation import EvaluationRequest
from schemas.finetuning import FineTuningCreateRequest, FineTuningStatusRequest
from schemas.kg import BuildGraphRequest, GraphRAGRequest, QueryGraphRequest
from schemas.multi_agent import ChainRequest, MultiAgentRequest, WorkflowRequest
from schemas.optimizer import OptimizeRequest
from schemas.rag import RAGBuildRequest, RAGDebugRequest, RAGQueryRequest
from schemas.vision import VisionRAGBuildRequest, VisionRAGQueryRequest
from schemas.web import WebSearchRequest

__all__ = [
    # Chat
    "Message",
    "ChatRequest",
    # Knowledge Graph
    "BuildGraphRequest",
    "QueryGraphRequest",
    "GraphRAGRequest",
    # RAG
    "RAGBuildRequest",
    "RAGQueryRequest",
    "RAGDebugRequest",
    # Agent
    "AgentRequest",
    # Web Search
    "WebSearchRequest",
    # Optimizer
    "OptimizeRequest",
    # Multi-Agent
    "MultiAgentRequest",
    "WorkflowRequest",
    "ChainRequest",
    # Vision
    "VisionRAGBuildRequest",
    "VisionRAGQueryRequest",
    # Audio
    "AudioTranscribeRequest",
    "AudioSynthesizeRequest",
    "AudioRAGRequest",
    # Evaluation
    "EvaluationRequest",
    # Fine-tuning
    "FineTuningCreateRequest",
    "FineTuningStatusRequest",
    # Database Models
    "ChatSession",
    "ChatMessage",
    "CreateSessionRequest",
    "AddMessageRequest",
    "SessionListResponse",
    "SessionResponse",
    "ApiKeyBase",
    "ApiKeyCreate",
    "ApiKeyInDB",
    "ApiKeyResponse",
    "ApiKeyListResponse",
    "ApiKeyValidationResult",
    "ProviderInfo",
    "ProviderListResponse",
    "GoogleOAuthToken",
    "GoogleAuthStatus",
    "RequestLog",
    "PROVIDER_CONFIG",
]
