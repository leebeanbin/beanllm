"""
Chat Request Schemas
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class Message(BaseModel):
    """Single chat message"""
    role: str  # "user", "assistant", "system"
    content: str


class ChatRequest(BaseModel):
    """Chat API request"""
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
