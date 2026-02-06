"""
Vision RAG Router

Vision RAG endpoints for image-based retrieval augmented generation.
Uses Python best practices: context managers, duck typing.
"""

import base64
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision_rag", tags=["Vision RAG"])

# Global VisionRAG instance
_vision_rag: Optional[Any] = None


# ============================================================================
# Request/Response Models
# ============================================================================


class VisionRAGBuildRequest(BaseModel):
    """Request to build Vision RAG index"""

    images: List[str] = Field(..., description="Base64 encoded images or URLs")
    texts: Optional[List[str]] = Field(None, description="Additional text documents")
    collection_name: Optional[str] = Field(default="default", description="Collection name")
    model: Optional[str] = Field(None, description="LLM model for captioning")
    generate_captions: bool = Field(default=False, description="Generate image captions")


class VisionRAGQueryRequest(BaseModel):
    """Request to query Vision RAG"""

    query: str = Field(..., description="Query text")
    image: Optional[str] = Field(None, description="Base64 encoded image for visual query")
    collection_name: Optional[str] = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    model: Optional[str] = Field(None)


class VisionRAGBuildResponse(BaseModel):
    """Response from Vision RAG build"""

    collection_name: str
    num_images: int
    num_texts: int
    status: str = "success"


class SourceResponse(BaseModel):
    """Source document in query response"""

    content: str
    score: float = 0.0
    type: str = "text"


class VisionRAGQueryResponse(BaseModel):
    """Response from Vision RAG query"""

    query: str
    answer: str
    sources: List[SourceResponse] = Field(default_factory=list)
    num_results: int


# ============================================================================
# Helper Functions
# ============================================================================


@asynccontextmanager
async def temp_directory():
    """Context manager for temporary directory with automatic cleanup"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _decode_base64_image(data: str, index: int, temp_dir: str) -> Optional[str]:
    """Decode base64 image and save to temp directory"""
    try:
        base64_data = data.split(",")[1] if "," in data else data
        img_bytes = base64.b64decode(base64_data)
        img_path = Path(temp_dir) / f"image_{index}.png"
        img_path.write_bytes(img_bytes)
        return str(img_path)
    except Exception as e:
        logger.warning(f"Failed to decode base64 image {index}: {e}")
        return None


async def _download_image(url: str, index: int, temp_dir: str) -> Optional[str]:
    """Download image from URL with security validation"""
    try:
        import httpx

        from beanllm.domain.web_search.security import validate_url

        validated_url = validate_url(url)

        async with httpx.AsyncClient() as client:
            response = await client.get(validated_url, timeout=30, follow_redirects=True)
            response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            logger.warning(f"URL {url} is not an image (Content-Type: {content_type})")
            return None

        ext = content_type.split("/")[1] if "/" in content_type else "png"
        img_path = Path(temp_dir) / f"image_{index}.{ext}"
        img_path.write_bytes(response.content)
        return str(img_path)

    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None


async def _process_images(images: List[str], temp_dir: str) -> List[str]:
    """Process image list (base64, URL, or file path) and return file paths"""
    paths = []

    for i, img_data in enumerate(images):
        if img_data.startswith("data:image"):
            path = _decode_base64_image(img_data, i, temp_dir)
        elif img_data.startswith(("http://", "https://")):
            path = await _download_image(img_data, i, temp_dir)
        else:
            path = img_data  # Assume file path

        if path:
            paths.append(path)

    return paths


def _extract_source_content(source: Any) -> Dict[str, Any]:
    """Extract source content using duck typing"""
    # Try various attribute patterns
    if hasattr(source, "document"):
        doc = source.document
        content = getattr(doc, "content", str(doc))
    elif hasattr(source, "page_content"):
        content = source.page_content
    elif hasattr(source, "content"):
        content = source.content
    else:
        content = str(source)

    return {
        "content": content[:200],
        "score": getattr(source, "score", 0.0),
        "type": "image" if hasattr(source, "image_path") else "text",
    }


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/build", response_model=VisionRAGBuildResponse)
async def vision_rag_build(request: VisionRAGBuildRequest) -> VisionRAGBuildResponse:
    """
    Build Vision RAG index from images.

    Supports base64 encoded images, URLs, or file paths.
    """
    global _vision_rag

    try:
        from beanllm.facade.advanced.vision_rag_facade import VisionRAG
        from beanllm.facade.core.client_facade import Client

        async with temp_directory() as temp_dir:
            image_paths = await _process_images(request.images, temp_dir)

            model = request.model or "gpt-4o"

            if image_paths:
                source = temp_dir if len(image_paths) > 1 else image_paths[0]
                vision_rag = VisionRAG.from_images(
                    source=source,
                    generate_captions=request.generate_captions,
                    llm_model=model,
                )
            else:
                client = Client(model=model)
                vision_rag = VisionRAG(client=client)

            _vision_rag = vision_rag

            return VisionRAGBuildResponse(
                collection_name=request.collection_name or "default",
                num_images=len(image_paths),
                num_texts=len(request.texts) if request.texts else 0,
                status="success",
            )

    except Exception as e:
        logger.error(f"VisionRAG build error: {e}", exc_info=True)
        raise HTTPException(500, f"VisionRAG build error: {str(e)}")


@router.post("/query", response_model=VisionRAGQueryResponse)
async def vision_rag_query(request: VisionRAGQueryRequest) -> VisionRAGQueryResponse:
    """
    Query Vision RAG system.

    Returns relevant images/documents based on the query.
    """
    if _vision_rag is None:
        raise HTTPException(404, "VisionRAG not built. Build it first.")

    try:
        answer, sources = _vision_rag.query(
            question=request.query,
            k=request.top_k,
            include_sources=True,
        )

        source_list = [
            SourceResponse(**_extract_source_content(src)) for src in sources[: request.top_k]
        ]

        return VisionRAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=source_list,
            num_results=len(sources),
        )

    except Exception as e:
        logger.error(f"VisionRAG query error: {e}", exc_info=True)
        raise HTTPException(500, f"VisionRAG query error: {str(e)}")


@router.get("/status")
async def vision_rag_status() -> Dict[str, Any]:
    """Check Vision RAG status"""
    return {
        "initialized": _vision_rag is not None,
        "status": "ready" if _vision_rag else "not_built",
    }
