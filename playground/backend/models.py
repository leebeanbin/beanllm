"""
Pydantic Models for Chat History

MongoDB schema for chat sessions and messages.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from bson import ObjectId


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
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    model: Optional[str] = Field(None, description="Model used for this message")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is beanllm?",
                "timestamp": "2026-01-23T10:00:00Z",
                "model": "gpt-4o",
                "metadata": {"feature": "chat"}
            }
        }


class ChatSession(BaseModel):
    """Chat session with messages"""

    id: Optional[PyObjectId] = Field(default=None, alias="_id", description="Session ID")
    session_id: str = Field(..., description="Unique session ID")
    title: str = Field(default="New Chat", description="Session title")
    feature_mode: str = Field(default="chat", description="Feature mode: chat, rag, multi-agent, etc.")
    model: str = Field(..., description="Primary model used")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat messages")

    # Feature-specific options
    feature_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Feature options")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session created time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Session last updated time")

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
                    {"role": "assistant", "content": "Hi!"}
                ],
                "created_at": "2026-01-23T10:00:00Z",
                "total_tokens": 150
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
