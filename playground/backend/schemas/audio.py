"""
Audio Request Schemas
"""

from typing import List, Optional

from pydantic import BaseModel


class AudioTranscribeRequest(BaseModel):
    """Request to transcribe audio"""

    audio_file: str  # Base64 encoded audio or file path
    model: Optional[str] = None


class AudioSynthesizeRequest(BaseModel):
    """Request to synthesize speech"""

    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    model: Optional[str] = None


class AudioRAGRequest(BaseModel):
    """Request for Audio RAG query"""

    query: str
    audio_files: Optional[List[str]] = None
    collection_name: Optional[str] = "default"
    top_k: int = 5
    model: Optional[str] = None
