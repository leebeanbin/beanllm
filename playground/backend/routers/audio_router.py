"""
Audio Router

Audio processing endpoints: transcription (STT), synthesis (TTS), and Audio RAG.
Uses Python best practices: duck typing, list comprehensions.
"""

import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["Audio"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AudioTranscribeRequest(BaseModel):
    """Request to transcribe audio to text"""
    audio_file: str = Field(..., description="Base64 encoded audio or file path")
    model: Optional[str] = Field(default="base", description="Whisper model size")


class AudioSynthesizeRequest(BaseModel):
    """Request to synthesize text to speech"""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    model: Optional[str] = Field(None)


class AudioRAGRequest(BaseModel):
    """Request for Audio RAG query"""
    query: str = Field(..., description="Search query")
    audio_files: Optional[List[str]] = Field(None, description="Audio files to add")
    collection_name: Optional[str] = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    model: Optional[str] = Field(None)


class TranscriptSegment(BaseModel):
    """Segment of transcribed audio"""
    start: float
    end: float
    text: str


class TranscribeResponse(BaseModel):
    """Response from audio transcription"""
    text: str
    language: Optional[str] = None
    segments: List[TranscriptSegment] = Field(default_factory=list)


class SynthesizeResponse(BaseModel):
    """Response from text-to-speech synthesis"""
    text: str
    audio_base64: str
    format: str = "wav"


class AudioSearchResult(BaseModel):
    """Result from audio search"""
    text: str
    audio_segment: str = ""
    score: float = 0.0


class AudioRAGResponse(BaseModel):
    """Response from Audio RAG query"""
    query: str
    results: List[AudioSearchResult]
    num_results: int


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_segments(result: Any) -> List[Dict[str, Any]]:
    """Extract transcript segments using duck typing"""
    if not hasattr(result, "segments"):
        return []

    return [
        {
            "start": getattr(seg, "start", 0.0),
            "end": getattr(seg, "end", 0.0),
            "text": getattr(seg, "text", ""),
        }
        for seg in result.segments
    ]


def _extract_search_results(results: List[Any], top_k: int) -> List[Dict[str, Any]]:
    """Extract search results using duck typing"""
    return [
        {
            "text": (result.get("text", "") if isinstance(result, dict) else getattr(result, "text", ""))[:200],
            "audio_segment": result.get("audio_segment", "") if isinstance(result, dict) else getattr(result, "audio_segment", ""),
            "score": result.get("score", 0.0) if isinstance(result, dict) else getattr(result, "score", 0.0),
        }
        for result in results[:top_k]
    ]


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/transcribe", response_model=TranscribeResponse)
async def audio_transcribe(request: AudioTranscribeRequest) -> TranscribeResponse:
    """
    Transcribe audio to text using Whisper.

    Supports base64 encoded audio or file paths.
    """
    try:
        from beanllm.domain.audio.stt import WhisperSTT

        stt = WhisperSTT(model=request.model or "base")
        result = await stt.transcribe_async(request.audio_file)

        segments = [
            TranscriptSegment(**seg)
            for seg in _extract_segments(result)
        ]

        return TranscribeResponse(
            text=getattr(result, "text", str(result)),
            language=getattr(result, "language", None),
            segments=segments,
        )

    except Exception as e:
        logger.error(f"Audio transcribe error: {e}", exc_info=True)
        raise HTTPException(500, f"Audio transcribe error: {str(e)}")


@router.post("/synthesize", response_model=SynthesizeResponse)
async def audio_synthesize(request: AudioSynthesizeRequest) -> SynthesizeResponse:
    """
    Synthesize text to speech.

    Returns audio as base64 encoded string.
    """
    try:
        from beanllm.domain.audio.tts import TextToSpeech

        tts = TextToSpeech()
        audio = await tts.synthesize_async(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
        )

        # Use duck typing for audio format extraction
        audio_base64 = audio.to_base64() if hasattr(audio, "to_base64") else str(audio)
        audio_format = getattr(audio, "format", "wav")

        return SynthesizeResponse(
            text=request.text,
            audio_base64=audio_base64,
            format=audio_format,
        )

    except Exception as e:
        logger.error(f"Audio synthesize error: {e}", exc_info=True)
        raise HTTPException(500, f"Audio synthesize error: {str(e)}")


@router.post("/rag", response_model=AudioRAGResponse)
async def audio_rag(request: AudioRAGRequest) -> AudioRAGResponse:
    """
    Query Audio RAG system.

    Search through transcribed audio content.
    """
    try:
        from beanllm.facade.advanced.audio_rag_facade import AudioRAG

        audio_rag = AudioRAG()

        # Add audio files if provided
        if request.audio_files:
            for audio_file in request.audio_files:
                await audio_rag.add_audio(audio_file)

        results = await audio_rag.search(
            query=request.query,
            top_k=request.top_k,
        )

        extracted = _extract_search_results(results, request.top_k)
        search_results = [AudioSearchResult(**r) for r in extracted]

        return AudioRAGResponse(
            query=request.query,
            results=search_results,
            num_results=len(results),
        )

    except Exception as e:
        logger.error(f"Audio RAG error: {e}", exc_info=True)
        raise HTTPException(500, f"Audio RAG error: {str(e)}")
