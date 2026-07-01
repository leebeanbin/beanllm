"""
CareerOS AI Adapter Router

Bridges CareerOS's expected API contract to beanllm's internals.

  POST /ai/complete         — text generation (routes through beanllm Client)
  POST /ai/embed            — text embedding (Ollama → OpenAI fallback)
  POST /ai/ocr/glm          — per-page OCR via Ollama glm-ocr (image multipart)
  POST /ai/ocr/unlimited    — multi-page OCR via Baidu unlimited-ocr (PDF base64)
  POST /ai/models/pull      — pull an Ollama model (storage management)
  POST /ai/models/warmup    — preload models into Ollama memory (keep_alive=-1)
  DELETE /ai/models/{name}  — delete an Ollama model (storage management)

Contract matches BeanllmAiClient, BeanllmEmbeddingClient, OcrTextExtractor in CareerOS.
"""

import asyncio
import base64
import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["CareerOS Adapter"])

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_EMBED_MODEL = os.getenv("DEFAULT_EMBED_MODEL", "qwen3-embedding:8b")

# Models kept warm in Ollama memory at all times (keep_alive=-1).
# Default is tuned for 16GB RAM: glm-ocr(2GB) + qwen3:7b(4.5GB) = ~6.5GB.
# For 24GB+: add qwen3-embedding:8b (5GB) or qwen3-vl:8b (6GB).
# Set WARMUP_MODELS='' to disable warmup entirely.
_DEFAULT_WARMUP = "glm-ocr,qwen3:7b"
WARMUP_MODELS: list[str] = [
    m.strip() for m in os.getenv("WARMUP_MODELS", _DEFAULT_WARMUP).split(",") if m.strip()
]


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


class OcrGlmResponse(BaseModel):
    text: str
    confidence: float = 1.0


class UnlimitedOcrRequest(BaseModel):
    pdf_base64: str
    language: str = "kor+eng"


class UnlimitedOcrResponse(BaseModel):
    text: str
    page_count: int = 0
    engine: str = "unlimited-ocr"


class PullRequest(BaseModel):
    model: str


class PullResponse(BaseModel):
    model: str
    status: str


class DeleteResponse(BaseModel):
    model: str
    status: str


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


# ---------------------------------------------------------------------------
# /ai/ocr/glm — per-page OCR via Ollama glm-ocr
# ---------------------------------------------------------------------------

GLM_OCR_MODEL = os.getenv("GLM_OCR_MODEL", "glm-ocr")


@router.post("/ocr/glm", response_model=OcrGlmResponse)
async def ocr_glm(file: UploadFile = File(...)):
    """
    Single-page OCR via Ollama glm-ocr.
    CareerOS sends one PNG per PDF page; call once per page and concatenate.
    """
    image_bytes = await file.read()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": GLM_OCR_MODEL,
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Text Recognition:",
                            "images": [b64],
                        }
                    ],
                },
            )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502, detail=f"glm-ocr HTTP {resp.status_code}: {resp.text[:200]}"
            )
        text = resp.json()["message"]["content"]
        logger.debug("glm-ocr OK chars=%d", len(text))
        return OcrGlmResponse(text=text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("glm-ocr failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"glm-ocr error: {str(e)}")


# ---------------------------------------------------------------------------
# /ai/ocr/unlimited — multi-page OCR via Baidu unlimited-ocr
# ---------------------------------------------------------------------------

_unlimited_ocr_pipeline = None  # lazy-loaded on first call


def _load_unlimited_ocr():
    global _unlimited_ocr_pipeline
    if _unlimited_ocr_pipeline is not None:
        return _unlimited_ocr_pipeline
    try:
        from transformers import AutoModel, AutoTokenizer

        model_id = os.getenv("UNLIMITED_OCR_MODEL", "baidu/Unlimited-OCR")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        _unlimited_ocr_pipeline = (tokenizer, model)
        logger.info("unlimited-ocr loaded: %s", model_id)
        return _unlimited_ocr_pipeline
    except Exception as e:
        logger.error("Failed to load unlimited-ocr: %s", e)
        raise RuntimeError(f"unlimited-ocr not available: {e}")


@router.post("/ocr/unlimited", response_model=UnlimitedOcrResponse)
async def ocr_unlimited(req: UnlimitedOcrRequest):
    """
    Multi-page OCR via Baidu unlimited-ocr (handles long documents in one pass).
    Requires: pip install transformers torch && huggingface-cli download baidu/Unlimited-OCR
    """
    import asyncio

    pdf_bytes = base64.b64decode(req.pdf_base64)

    def _run_ocr(pdf_data: bytes) -> tuple[str, int]:
        tokenizer, model = _load_unlimited_ocr()
        # unlimited-ocr expects a file path or bytes; use temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_data)
            tmp_path = f.name
        try:
            result = model.ocr(tmp_path, tokenizer=tokenizer)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            page_count = result.get("page_count", 0) if isinstance(result, dict) else 0
            return text, page_count
        finally:
            import os as _os

            _os.unlink(tmp_path)

    try:
        text, page_count = await asyncio.get_event_loop().run_in_executor(None, _run_ocr, pdf_bytes)
        logger.debug("unlimited-ocr OK chars=%d pages=%d", len(text), page_count)
        return UnlimitedOcrResponse(text=text, page_count=page_count)
    except Exception as e:
        logger.error("unlimited-ocr failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"unlimited-ocr error: {str(e)}")


# ---------------------------------------------------------------------------
# /ai/models — Ollama model management (pull / delete)
# ---------------------------------------------------------------------------


@router.post("/models/pull", response_model=PullResponse)
async def pull_model(req: PullRequest):
    """
    Pull (download) an Ollama model. Waits until download is complete.
    May take minutes for large models — use with appropriate client timeout.
    """
    try:
        async with httpx.AsyncClient(timeout=3600.0) as client:
            # Ollama streaming pull: consume all lines, check final status
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": req.model},
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise HTTPException(
                        status_code=502, detail=f"Ollama pull HTTP {resp.status_code}: {body[:200]}"
                    )
                last_status = ""
                async for line in resp.aiter_lines():
                    if line:
                        import json

                        try:
                            last_status = json.loads(line).get("status", last_status)
                        except Exception:
                            pass
        logger.info("Ollama pull complete: %s status=%s", req.model, last_status)
        return PullResponse(model=req.model, status=last_status or "success")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ollama pull failed model=%s: %s", req.model, e)
        raise HTTPException(status_code=500, detail=f"pull error: {str(e)}")


@router.delete("/models/{model_name}", response_model=DeleteResponse)
async def delete_model(model_name: str):
    """
    Delete a locally installed Ollama model to free storage.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.request(
                "DELETE",
                f"{OLLAMA_BASE_URL}/api/delete",
                json={"name": model_name},
            )
        if resp.status_code not in (200, 204):
            raise HTTPException(
                status_code=502, detail=f"Ollama delete HTTP {resp.status_code}: {resp.text[:200]}"
            )
        logger.info("Ollama model deleted: %s", model_name)
        return DeleteResponse(model=model_name, status="deleted")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ollama delete failed model=%s: %s", model_name, e)
        raise HTTPException(status_code=500, detail=f"delete error: {str(e)}")


# ---------------------------------------------------------------------------
# /ai/models/warmup — preload models into Ollama memory
# ---------------------------------------------------------------------------


class WarmupResponse(BaseModel):
    warmed: list[str]
    skipped: list[str]  # not installed


async def _warmup_one(model: str) -> bool:
    """Send empty prompt with keep_alive=-1 to load model into Ollama memory."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": -1},
            )
        if resp.status_code == 200:
            logger.info("warmup OK: %s is now resident in Ollama memory", model)
            return True
        logger.warning(
            "warmup skipped %s — HTTP %d (model may not be installed)", model, resp.status_code
        )
        return False
    except Exception as e:
        logger.warning("warmup skipped %s — %s", model, e)
        return False


@router.post("/models/warmup", response_model=WarmupResponse)
async def warmup_models(models: list[str] | None = None):
    """
    Preload models into Ollama memory (keep_alive=-1) so first inference is instant.
    If models is omitted, uses WARMUP_MODELS env var (default: glm-ocr, qwen3:7b, qwen3-embedding:8b).
    Models not installed in Ollama are silently skipped.
    """
    targets = models if models else WARMUP_MODELS
    results = await asyncio.gather(*[_warmup_one(m) for m in targets])
    warmed = [m for m, ok in zip(targets, results) if ok]
    skipped = [m for m, ok in zip(targets, results) if not ok]
    return WarmupResponse(warmed=warmed, skipped=skipped)


async def warmup_on_startup() -> None:
    """Called from main.py startup_event — fire-and-forget warmup."""
    if not WARMUP_MODELS:
        return
    logger.info("Starting warmup for: %s", WARMUP_MODELS)
    await warmup_models(WARMUP_MODELS)
