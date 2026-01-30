"""
RAG Router

RAG (Retrieval-Augmented Generation) endpoints.
Uses Python best practices: context managers, generators, and duck typing.
"""

import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from common import get_rag_chain, set_rag_chain, get_client
from schemas.rag import RAGBuildRequest, RAGQueryRequest, RAGDebugRequest
from schemas.responses.rag import (
    RAGBuildResponse,
    RAGQueryResponse,
    CollectionInfo,
    CollectionListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["RAG"])

# Supported file extensions
SUPPORTED_EXTENSIONS = frozenset([".txt", ".md", ".json", ".pdf", ".docx", ".doc", ".csv"])


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_source_content(source: Any, max_length: int = 200) -> str:
    """
    Extract content from source document using duck typing.

    Supports multiple source types:
    - VectorSearchResult with document attribute
    - LangChain Document with page_content
    - Custom Document with content
    """
    # Try various attribute patterns (duck typing)
    if hasattr(source, "document"):
        doc = source.document
        content = getattr(doc, "content", None) or str(doc)
    elif hasattr(source, "page_content"):
        content = source.page_content
    elif hasattr(source, "content"):
        content = source.content
    else:
        content = str(source)

    # Truncate with ellipsis if needed
    return content[:max_length] + "..." if len(content) > max_length else content


def _get_document_count(chain: Any) -> int:
    """Get document count from RAG chain using multiple methods."""
    if not hasattr(chain, "vector_store") or not chain.vector_store:
        return 0

    vs = chain.vector_store

    # Try different methods (duck typing pattern)
    for method_name in ["_get_all_vectors_and_docs", "get_all_documents", "count"]:
        if hasattr(vs, method_name):
            try:
                method = getattr(vs, method_name)
                result = method()

                if method_name == "_get_all_vectors_and_docs":
                    _, docs = result
                    return len(docs) if docs else 0
                elif isinstance(result, (list, tuple)):
                    return len(result)
                elif isinstance(result, int):
                    return result
            except Exception:
                continue

    return 0


@asynccontextmanager
async def temp_directory():
    """Context manager for temporary directory cleanup."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/build", response_model=RAGBuildResponse)
async def rag_build(request: RAGBuildRequest) -> RAGBuildResponse:
    """
    Build RAG index from text documents.

    Uses beanllm's RAGBuilder with builder pattern.
    """
    try:
        from beanllm.facade.core import Client, RAGBuilder
        from beanllm.domain.loaders import Document

        collection_name = request.collection_name or "default"

        # Convert strings to Document objects (list comprehension)
        docs = [Document(content=doc, metadata={}) for doc in request.documents]

        # Create client with model selection
        client = Client(model=request.model) if request.model else get_client()

        # Build RAG using builder pattern (fluent interface)
        rag_chain = (
            RAGBuilder()
            .load_documents(docs)
            .split_text(chunk_size=500, chunk_overlap=50)
            .use_llm(client)
            .build()
        )

        # Store in global state
        set_rag_chain(collection_name, rag_chain)

        return RAGBuildResponse(
            collection_name=collection_name,
            num_documents=len(request.documents),
            status="success",
        )

    except Exception as e:
        logger.error(f"RAG build error: {e}", exc_info=True)
        raise HTTPException(500, f"RAG build error: {str(e)}")


@router.post("/build_from_files")
async def rag_build_from_files(
    files: List[UploadFile] = File(...),
    collection_name: str = "default",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build RAG index from uploaded files (PDF, DOCX, etc.).

    Uses context manager for proper cleanup.
    """
    try:
        from beanllm.facade.core import Client, RAGBuilder
        from beanllm.domain.loaders import DocumentLoader

        async with temp_directory() as temp_dir:
            file_paths = []

            # Save uploaded files with validation
            for file in files:
                ext = Path(file.filename).suffix.lower() if file.filename else ""

                if ext not in SUPPORTED_EXTENSIONS:
                    logger.warning(f"Unsupported file type: {ext}, skipping {file.filename}")
                    continue

                file_path = Path(temp_dir) / (file.filename or f"file_{len(file_paths)}{ext}")
                content = await file.read()

                # Use Path.write_bytes for cleaner file handling
                file_path.write_bytes(content)
                file_paths.append(str(file_path))

            if not file_paths:
                raise HTTPException(400, "No valid files uploaded")

            # Load documents using generator pattern for memory efficiency
            all_docs = []
            for file_path in file_paths:
                try:
                    docs = DocumentLoader.load(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if not all_docs:
                raise HTTPException(400, "No documents could be loaded from uploaded files")

            # Build RAG chain
            client = Client(model=model) if model else get_client()

            rag_chain = (
                RAGBuilder()
                .load_documents(all_docs)
                .split_text(chunk_size=500, chunk_overlap=50)
                .use_llm(client)
                .build()
            )

            set_rag_chain(collection_name, rag_chain)

            return {
                "collection_name": collection_name,
                "num_documents": len(all_docs),
                "num_files": len(file_paths),
                "status": "success",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG build from files error: {e}", exc_info=True)
        raise HTTPException(500, f"RAG build from files error: {str(e)}")


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Query RAG system.

    Returns answer with source documents.
    """
    try:
        collection_name = request.collection_name or "default"
        rag_chain = get_rag_chain(collection_name)

        if rag_chain is None:
            raise HTTPException(404, f"Collection '{collection_name}' not found. Build it first.")

        # Query with sources using async method
        answer, sources = await rag_chain.aquery(
            question=request.query,
            k=request.top_k,
            include_sources=True,
        )

        # Extract source content using list comprehension
        source_list = [
            {"content": _extract_source_content(src)}
            for src in sources[:3]
        ]

        return RAGQueryResponse(
            query=request.query,
            answer=answer,
            sources=source_list,
            relevance_score=0.85,  # Placeholder - could be calculated from similarity scores
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query error: {e}", exc_info=True)
        raise HTTPException(500, f"RAG query error: {str(e)}")


@router.get("/collections", response_model=CollectionListResponse)
async def rag_list_collections() -> CollectionListResponse:
    """List all RAG collections with document counts."""
    try:
        from common import _rag_chains  # Access global state

        collections = [
            CollectionInfo(
                name=name,
                document_count=_get_document_count(chain),
                created_at=None,
            )
            for name, chain in _rag_chains.items()
        ]

        return CollectionListResponse(
            collections=collections,
            total=len(collections),
        )

    except Exception as e:
        logger.error(f"Failed to list collections: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to list collections: {str(e)}")


@router.delete("/collections/{collection_name}")
async def rag_delete_collection(collection_name: str) -> Dict[str, str]:
    """Delete a RAG collection."""
    try:
        from common import _rag_chains  # Access global state

        if collection_name not in _rag_chains:
            raise HTTPException(404, f"Collection '{collection_name}' not found")

        # Delete from memory
        del _rag_chains[collection_name]

        return {
            "status": "deleted",
            "collection_name": collection_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to delete collection: {str(e)}")


@router.post("/debug")
async def rag_debug_analyze(request: RAGDebugRequest) -> Dict[str, Any]:
    """
    Debug RAG pipeline.

    Analyzes retrieval quality, chunk relevance, and answer generation.
    """
    try:
        from common import get_rag_debugger

        debugger = get_rag_debugger()

        # Run analysis
        result = await debugger.analyze(
            query=request.query,
            documents=request.documents,
            collection_name=request.collection_name,
            debug_mode=request.debug_mode,
        )

        return {
            "query": request.query,
            "analysis": result,
            "debug_mode": request.debug_mode,
        }

    except Exception as e:
        logger.error(f"RAG debug error: {e}", exc_info=True)
        raise HTTPException(500, f"RAG debug error: {str(e)}")
