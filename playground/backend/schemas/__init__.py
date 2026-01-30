"""
Pydantic Schemas for BeanLLM Playground API

Organized by feature domain.
"""

from schemas.chat import Message, ChatRequest
from schemas.kg import BuildGraphRequest, QueryGraphRequest, GraphRAGRequest
from schemas.rag import RAGBuildRequest, RAGQueryRequest, RAGDebugRequest
from schemas.agent import AgentRequest
from schemas.web import WebSearchRequest
from schemas.optimizer import OptimizeRequest
from schemas.multi_agent import MultiAgentRequest, WorkflowRequest, ChainRequest
from schemas.vision import VisionRAGBuildRequest, VisionRAGQueryRequest
from schemas.audio import AudioTranscribeRequest, AudioSynthesizeRequest, AudioRAGRequest
from schemas.evaluation import EvaluationRequest
from schemas.finetuning import FineTuningCreateRequest, FineTuningStatusRequest
from schemas.database import (
    ChatSession,
    ChatMessage,
    CreateSessionRequest,
    AddMessageRequest,
    SessionListResponse,
    SessionResponse,
    ApiKeyBase,
    ApiKeyCreate,
    ApiKeyInDB,
    ApiKeyResponse,
    ApiKeyListResponse,
    ApiKeyValidationResult,
    ProviderInfo,
    ProviderListResponse,
    GoogleOAuthToken,
    GoogleAuthStatus,
    RequestLog,
    PROVIDER_CONFIG,
)

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
