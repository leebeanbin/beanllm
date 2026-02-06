"""
Pydantic Models for Chat History

MongoDB schema for chat sessions and messages.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Get current UTC time (Python 3.12+ compatible)"""
    return datetime.now(timezone.utc)


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class ChatMessage(BaseModel):
    """Single chat message"""

    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=utc_now, description="Message timestamp")
    model: Optional[str] = Field(None, description="Model used for this message")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is beanllm?",
                "timestamp": "2026-01-23T10:00:00Z",
                "model": "gpt-4o",
                "metadata": {"feature": "chat"},
            }
        }


class ChatSession(BaseModel):
    """Chat session with messages"""

    id: Optional[PyObjectId] = Field(default=None, alias="_id", description="Session ID")
    session_id: str = Field(..., description="Unique session ID")
    title: str = Field(default="New Chat", description="Session title")
    feature_mode: str = Field(
        default="chat", description="Feature mode: chat, rag, multi-agent, etc."
    )
    model: str = Field(..., description="Primary model used")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat messages")

    # Feature-specific options
    feature_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Feature options"
    )

    # Metadata
    created_at: datetime = Field(default_factory=utc_now, description="Session created time")
    updated_at: datetime = Field(default_factory=utc_now, description="Session last updated time")

    # Statistics
    total_tokens: int = Field(default=0, description="Total tokens used")
    message_count: int = Field(default=0, description="Total message count")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "title": "beanllm Questions",
                "feature_mode": "chat",
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ],
                "created_at": "2026-01-23T10:00:00Z",
                "total_tokens": 150,
            }
        }


class CreateSessionRequest(BaseModel):
    """Request to create a new session"""

    title: Optional[str] = Field(default="New Chat", description="Session title")
    feature_mode: str = Field(default="chat", description="Feature mode")
    model: str = Field(..., description="Model to use")
    feature_options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AddMessageRequest(BaseModel):
    """Request to add a message to a session"""

    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    model: Optional[str] = Field(None, description="Model used")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    """Response for session list"""

    sessions: List[Dict[str, Any]] = Field(..., description="List of sessions")
    total: int = Field(..., description="Total session count")


class SessionResponse(BaseModel):
    """Response for single session"""

    session: Dict[str, Any] = Field(..., description="Session data")


# ===========================================
# API Key Models
# ===========================================


class ProviderType(str):
    """Supported API key providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    SERPAPI = "serpapi"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    WEAVIATE = "weaviate"
    NEO4J = "neo4j"
    GOOGLE_OAUTH = "google_oauth"


# Provider display names and environment variable mappings
PROVIDER_CONFIG = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "placeholder": "sk-...",
        "description": "GPT-4, GPT-4o, o1 등",
    },
    "anthropic": {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "placeholder": "sk-ant-...",
        "description": "Claude 3, Sonnet, Haiku",
    },
    "google": {
        "name": "Google AI",
        "env_var": "GOOGLE_API_KEY",
        "placeholder": "AIza...",
        "description": "Gemini Pro, PaLM",
    },
    "gemini": {
        "name": "Gemini",
        "env_var": "GEMINI_API_KEY",
        "placeholder": "AIza...",
        "description": "Gemini API",
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
        "placeholder": "sk-...",
        "description": "DeepSeek R1, Coder",
    },
    "perplexity": {
        "name": "Perplexity",
        "env_var": "PERPLEXITY_API_KEY",
        "placeholder": "pplx-...",
        "description": "Perplexity AI",
    },
    "tavily": {
        "name": "Tavily",
        "env_var": "TAVILY_API_KEY",
        "placeholder": "tvly-...",
        "description": "Web Search API",
    },
    "serpapi": {
        "name": "SerpAPI",
        "env_var": "SERPAPI_API_KEY",
        "placeholder": "",
        "description": "Google Search API",
    },
    "pinecone": {
        "name": "Pinecone",
        "env_var": "PINECONE_API_KEY",
        "placeholder": "",
        "description": "Vector Database",
    },
    "qdrant": {
        "name": "Qdrant",
        "env_var": "QDRANT_API_KEY",
        "placeholder": "",
        "description": "Vector Database",
    },
    "weaviate": {
        "name": "Weaviate",
        "env_var": "WEAVIATE_API_KEY",
        "placeholder": "",
        "description": "Vector Database",
    },
    "neo4j": {
        "name": "Neo4j",
        "env_var": "NEO4J_PASSWORD",
        "placeholder": "",
        "description": "Graph Database",
    },
}


class ApiKeyBase(BaseModel):
    """Base model for API keys"""

    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ApiKeyCreate(ApiKeyBase):
    """Request to create/update an API key"""

    api_key: str = Field(..., description="The actual API key", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "api_key": "sk-1234567890abcdef",
                "metadata": {"organization": "my-org"},
            }
        }


class ApiKeyInDB(ApiKeyBase):
    """API key as stored in MongoDB"""

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    key_encrypted: str = Field(..., description="Encrypted API key")
    key_hint: str = Field(..., description="Last 4 characters for identification")
    is_valid: bool = Field(default=False, description="Whether the key has been validated")
    last_validated: Optional[datetime] = Field(None, description="Last validation time")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}


class ApiKeyResponse(BaseModel):
    """Response for API key (without the actual key)"""

    provider: str = Field(..., description="Provider name")
    key_hint: str = Field(..., description="Last 4 characters (e.g., '...7890')")
    is_valid: bool = Field(..., description="Whether the key is valid")
    last_validated: Optional[datetime] = Field(None, description="Last validation time")
    created_at: Optional[datetime] = Field(None, description="Creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "key_hint": "...cdef",
                "is_valid": True,
                "last_validated": "2026-01-23T10:00:00Z",
                "created_at": "2026-01-23T09:00:00Z",
                "updated_at": "2026-01-23T10:00:00Z",
            }
        }


class ApiKeyListResponse(BaseModel):
    """Response for listing all API keys"""

    keys: List[ApiKeyResponse] = Field(..., description="List of API keys")
    total: int = Field(..., description="Total count")


class ApiKeyValidationResult(BaseModel):
    """Result of API key validation"""

    provider: str = Field(..., description="Provider name")
    is_valid: bool = Field(..., description="Whether the key is valid")
    error: Optional[str] = Field(None, description="Error message if invalid")
    models_available: Optional[List[str]] = Field(
        None, description="Available models (if applicable)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "is_valid": True,
                "models_available": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            }
        }


class ProviderInfo(BaseModel):
    """Information about a provider"""

    id: str = Field(..., description="Provider ID")
    name: str = Field(..., description="Display name")
    env_var: str = Field(..., description="Environment variable name")
    placeholder: str = Field(..., description="Input placeholder")
    description: str = Field(..., description="Provider description")
    is_configured: bool = Field(..., description="Whether API key is configured")
    is_valid: Optional[bool] = Field(None, description="Whether the key is valid")


class ProviderListResponse(BaseModel):
    """Response for listing providers"""

    providers: List[ProviderInfo] = Field(..., description="List of providers")


# ===========================================
# Google OAuth Models
# ===========================================


class GoogleOAuthToken(BaseModel):
    """Google OAuth token stored in MongoDB"""

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: str = Field(default="default", description="User identifier")
    access_token_encrypted: str = Field(..., description="Encrypted access token")
    refresh_token_encrypted: Optional[str] = Field(None, description="Encrypted refresh token")
    token_type: str = Field(default="Bearer")
    expires_at: datetime = Field(..., description="Token expiration time")
    scopes: List[str] = Field(default_factory=list, description="Granted scopes")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}


class GoogleAuthStatus(BaseModel):
    """Google OAuth authentication status"""

    is_authenticated: bool = Field(..., description="Whether user is authenticated")
    scopes: List[str] = Field(default_factory=list, description="Granted scopes")
    expires_at: Optional[datetime] = Field(None, description="Token expiration")
    available_services: List[str] = Field(
        default_factory=list, description="Available Google services"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ===========================================
# Request Log Models (for monitoring)
# ===========================================


class RequestLog(BaseModel):
    """Request log for monitoring"""

    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    request_id: str = Field(..., description="Unique request ID")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    endpoint: str = Field(..., description="API endpoint")
    method: str = Field(..., description="HTTP method")
    status_code: int = Field(..., description="Response status code")
    duration_ms: float = Field(..., description="Request duration in milliseconds")
    model: Optional[str] = Field(None, description="Model used")
    provider: Optional[str] = Field(None, description="Provider used")
    input_tokens: Optional[int] = Field(None, description="Input tokens")
    output_tokens: Optional[int] = Field(None, description="Output tokens")
    total_tokens: Optional[int] = Field(None, description="Total tokens")
    error: Optional[str] = Field(None, description="Error message if any")
    timestamp: datetime = Field(default_factory=utc_now)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str, datetime: lambda v: v.isoformat()}
