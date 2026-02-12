"""
RAG Router

RAG (Retrieval-Augmented Generation) endpoints.
Uses Python best practices: context managers, generators, and duck typing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import get_client, get_rag_chain, set_rag_chain
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from schemas.rag import RAGBuildRequest, RAGDebugRequest, RAGQueryRequest
from schemas.responses.rag import (
    CollectionInfo,
    CollectionListResponse,
    RAGBuildResponse,
    RAGQueryResponse,
)
from utils.file_upload import save_upload_to_temp, temp_directory

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
        from beanllm.domain.loaders import Document
        from beanllm.facade.core import Client, RAGBuilder

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
        from beanllm.domain.loaders import DocumentLoader
        from beanllm.facade.core import Client, RAGBuilder

        async with temp_directory() as temp_dir:
            file_paths: List[str] = []

            # Save uploaded files with sanitization + streaming + size validation
            for file in files:
                try:
                    saved = await save_upload_to_temp(
                        file,
                        temp_dir,
                        allowed_extensions=SUPPORTED_EXTENSIONS,
                    )
                    file_paths.append(str(saved))
                except HTTPException as exc:
                    logger.warning("Skipping %s: %s", file.filename, exc.detail)
                    continue

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
        source_list = [{"content": _extract_source_content(src)} for src in sources[:3]]

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


# ============================================================================
# Session-based RAG Endpoints (자동 인덱싱)
# ============================================================================


class SessionDocumentRequest(BaseModel):
    """세션 문서 추가 요청"""

    content: str = Field(..., description="문서 내용")
    source: str = Field(default="upload", description="문서 출처")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="추가 메타데이터")


class SessionQueryRequest(BaseModel):
    """세션 RAG 검색 요청"""

    query: str = Field(..., description="검색 쿼리")
    k: int = Field(default=5, description="반환할 결과 수")
    generate_answer: bool = Field(default=False, description="답변 생성 여부")
    model: Optional[str] = Field(default=None, description="답변 생성 모델")


@router.post("/session/{session_id}/upload")
async def session_rag_upload_files(
    session_id: str,
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    """
    세션에 파일 업로드 및 자동 RAG 인덱싱.

    업로드된 파일은 자동으로 세션 RAG 컬렉션에 인덱싱됩니다.
    """
    try:
        from services.session_rag_service import session_rag_service

        from beanllm.domain.loaders import DocumentLoader

        async with temp_directory() as temp_dir:
            added_count = 0
            failed_files = []

            for file in files:
                try:
                    # 파일명 살균 + 스트리밍 저장 + 크기 검증
                    file_path = await save_upload_to_temp(
                        file,
                        temp_dir,
                        allowed_extensions=SUPPORTED_EXTENSIONS,
                    )
                    ext = file_path.suffix.lower()

                    # 문서 로드
                    docs = DocumentLoader.load(str(file_path))

                    # 세션 RAG에 추가
                    for doc in docs:
                        doc_content = getattr(doc, "content", None) or getattr(
                            doc, "page_content", str(doc)
                        )
                        success = await session_rag_service.add_document(
                            session_id=session_id,
                            content=doc_content,
                            source=file.filename or "upload",
                            metadata={"file_type": ext},
                        )
                        if success:
                            added_count += 1

                except HTTPException as exc:
                    failed_files.append({"filename": file.filename, "reason": exc.detail})
                except Exception as e:
                    logger.warning(f"Failed to process file {file.filename}: {e}")
                    failed_files.append({"filename": file.filename, "reason": str(e)})

            # 세션 정보 가져오기
            session_info = session_rag_service.get_session_info(session_id)

            return {
                "session_id": session_id,
                "added_documents": added_count,
                "total_documents": session_info.document_count if session_info else added_count,
                "failed_files": failed_files,
                "status": "success" if added_count > 0 else "partial",
            }

    except Exception as e:
        logger.error(f"Session RAG upload error: {e}", exc_info=True)
        raise HTTPException(500, f"Session RAG upload error: {str(e)}")


@router.post("/session/{session_id}/document")
async def session_rag_add_document(
    session_id: str,
    request: SessionDocumentRequest,
) -> Dict[str, Any]:
    """
    세션에 텍스트 문서 추가 및 자동 인덱싱.

    텍스트 콘텐츠를 직접 세션 RAG에 추가합니다.
    """
    try:
        from services.session_rag_service import session_rag_service

        success = await session_rag_service.add_document(
            session_id=session_id,
            content=request.content,
            source=request.source,
            metadata=request.metadata,
        )

        if not success:
            raise HTTPException(500, "Failed to add document to session RAG")

        session_info = session_rag_service.get_session_info(session_id)

        return {
            "session_id": session_id,
            "document_count": session_info.document_count if session_info else 1,
            "source": request.source,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session RAG add document error: {e}", exc_info=True)
        raise HTTPException(500, f"Session RAG add document error: {str(e)}")


@router.post("/session/{session_id}/query")
async def session_rag_query(
    session_id: str,
    request: SessionQueryRequest,
) -> Dict[str, Any]:
    """
    세션 RAG 검색.

    세션에 인덱싱된 문서에서 검색합니다.
    답변 생성 옵션을 사용하면 검색 결과를 바탕으로 답변을 생성합니다.
    """
    try:
        from services.session_rag_service import session_rag_service

        if request.generate_answer:
            result = await session_rag_service.query_with_generation(
                session_id=session_id,
                query=request.query,
                model=request.model or "qwen2.5:0.5b",
                k=request.k,
            )
        else:
            result = await session_rag_service.query(
                session_id=session_id,
                query=request.query,
                k=request.k,
            )

        if result is None:
            raise HTTPException(
                404,
                f"No RAG data found for session '{session_id}'. Upload documents first.",
            )

        return {
            "session_id": session_id,
            **result,
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session RAG query error: {e}", exc_info=True)
        raise HTTPException(500, f"Session RAG query error: {str(e)}")


@router.get("/session/{session_id}/info")
async def session_rag_info(session_id: str) -> Dict[str, Any]:
    """
    세션 RAG 정보 조회.

    세션에 인덱싱된 문서 수, 출처 목록 등을 반환합니다.
    """
    try:
        from services.session_rag_service import session_rag_service

        info = session_rag_service.get_session_info(session_id)

        if info is None:
            return {
                "session_id": session_id,
                "exists": False,
                "document_count": 0,
                "sources": [],
            }

        return {
            "session_id": session_id,
            "exists": True,
            "collection_name": info.collection_name,
            "document_count": info.document_count,
            "sources": info.sources,
            "created_at": info.created_at.isoformat(),
            "updated_at": info.updated_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Session RAG info error: {e}", exc_info=True)
        raise HTTPException(500, f"Session RAG info error: {str(e)}")


@router.delete("/session/{session_id}")
async def session_rag_delete(session_id: str) -> Dict[str, str]:
    """
    세션 RAG 삭제.

    세션에 인덱싱된 모든 문서를 삭제합니다.
    """
    try:
        from services.session_rag_service import session_rag_service

        success = await session_rag_service.delete_session_rag(session_id)

        if not success:
            raise HTTPException(404, f"Session RAG '{session_id}' not found")

        return {
            "session_id": session_id,
            "status": "deleted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session RAG delete error: {e}", exc_info=True)
        raise HTTPException(500, f"Session RAG delete error: {str(e)}")


@router.get("/sessions")
async def list_session_rags() -> Dict[str, Any]:
    """
    모든 세션 RAG 목록 조회.

    활성화된 모든 세션 RAG의 정보를 반환합니다.
    """
    try:
        from services.session_rag_service import session_rag_service

        sessions = session_rag_service.list_sessions()

        return {
            "sessions": sessions,
            "total": len(sessions),
        }

    except Exception as e:
        logger.error(f"List session RAGs error: {e}", exc_info=True)
        raise HTTPException(500, f"List session RAGs error: {str(e)}")
