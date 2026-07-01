"""
CareerOS AI Adapter Router

Bridges CareerOS's expected API contract to beanllm's internals.

  POST /ai/complete  — text generation (routes through beanllm Client)
  POST /ai/embed     — text embedding (Ollama embedding → OpenAI fallback)

Contract matches BeanllmAiClient and BeanllmEmbeddingClient in CareerOS.
"""

import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["CareerOS Adapter"])

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "qwen3-embedding:8b")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CompleteRequest(BaseModel):
    model: str
    system: Optional[str] = None
    content: str
    image_base64: Optional[str] = None


class CompleteResponse(BaseModel):
    content: str
    tokens_used: Optional[int] = None


class EmbedRequest(BaseModel):
    text: str
    model: Optional[str] = None  # overrides DEFAULT_EMBED_MODEL when set


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimensions: int


# ---------------------------------------------------------------------------
# /ai/complete
# ---------------------------------------------------------------------------


@router.post("/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest):
    """
    Text generation for CareerOS.

    Model routing:
      airllm/<name>  →  strip prefix, use Ollama provider (legacy prefix support)
      <name>         →  beanllm Client auto-detects provider
    """
    try:
        from beanllm import Client

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})

        if req.image_base64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": req.content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{req.image_base64}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": req.content})

        model = req.model
        provider = None
        if req.model.startswith("airllm/"):
            model = req.model[len("airllm/") :]
            provider = "ollama"

        client = Client(model=model, provider=provider)
        response = await client.chat(messages=messages)

        tokens_used = None
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            tokens_used = getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)

        return CompleteResponse(content=response.content, tokens_used=tokens_used)

    except Exception as e:
        logger.error("/ai/complete failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"/ai/complete error: {str(e)}")


# ---------------------------------------------------------------------------
# /ai/embed
# ---------------------------------------------------------------------------


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """
    Text embedding for CareerOS.
    Priority: Ollama (qwen3-embedding:8b) → OpenAI (text-embedding-3-small)
    """
    embed_model = req.model or DEFAULT_EMBED_MODEL

    # 1. Try Ollama
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": embed_model, "prompt": req.text},
            )
            if resp.status_code == 200:
                embedding = resp.json()["embedding"]
                logger.debug("Ollama embed OK model=%s dims=%d", embed_model, len(embedding))
                return EmbedResponse(embedding=embedding, dimensions=len(embedding))
            logger.warning("Ollama embed HTTP %d: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.warning("Ollama embed unavailable (%s), trying OpenAI fallback", e)

    # 2. Fallback: OpenAI
    try:
        import openai

        oai = openai.AsyncOpenAI()
        resp = await oai.embeddings.create(model="text-embedding-3-small", input=req.text)
        embedding = resp.data[0].embedding
        logger.debug("OpenAI embed fallback OK dims=%d", len(embedding))
        return EmbedResponse(embedding=embedding, dimensions=len(embedding))
    except Exception as e:
        logger.error("All embedding providers failed: %s", e)
        raise HTTPException(status_code=503, detail=f"No embedding provider available: {str(e)}")
