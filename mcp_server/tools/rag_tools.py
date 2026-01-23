"""
RAG Tools - 기존 beanllm RAG 기능을 MCP tool로 wrapping

🎯 핵심: 새로운 코드를 만들지 않고 기존 코드를 함수화!
"""
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastmcp import FastMCP

# 기존 beanllm 코드 import (wrapping 대상)
from beanllm.facade.core import RAGChain
from beanllm.domain.loaders import DirectoryLoader, PDFLoader, TextLoader, CSVLoader
from beanllm.domain.embeddings import LocalEmbeddings
from beanllm.domain.vector_stores.local import ChromaVectorStore
from mcp_server.config import MCPServerConfig

# FastMCP 인스턴스 생성
mcp = FastMCP("RAG Tools")

# 전역 RAG 인스턴스 캐시 (세션 관리)
_rag_instances: Dict[str, RAGChain] = {}


@mcp.tool()
async def build_rag_system(
    documents_path: str,
    collection_name: str = "default",
    chunk_size: int = MCPServerConfig.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = MCPServerConfig.DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = MCPServerConfig.DEFAULT_EMBEDDING_MODEL,
    vector_store_type: str = "chroma",
) -> dict:
    """
    RAG 시스템 구축 (기존 코드 재사용)

    Args:
        documents_path: 문서가 있는 디렉토리 또는 파일 경로
        collection_name: 벡터 스토어 컬렉션 이름
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        embedding_model: 임베딩 모델명 (Ollama 모델)
        vector_store_type: 벡터 스토어 타입 (chroma, faiss, etc.)

    Returns:
        dict: 성공 여부, 문서 개수, 청크 개수

    Example:
        User: "이 폴더의 PDF 파일들로 RAG 시스템을 만들어줘"
        → build_rag_system(documents_path="/path/to/pdfs")
    """
    try:
        # 🎯 기존 beanllm 코드 재사용!
        path = Path(documents_path)

        # 1. 문서 로드 (기존 Loader 사용)
        if path.is_dir():
            loader = DirectoryLoader(str(path))
        elif path.suffix == ".pdf":
            loader = PDFLoader(str(path))
        elif path.suffix in [".txt", ".md"]:
            loader = TextLoader(str(path))
        elif path.suffix == ".csv":
            loader = CSVLoader(str(path))
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {path.suffix}",
            }

        documents = await asyncio.to_thread(loader.load)

        # 2. RAG 구축 (기존 RAGChain.from_documents 사용)
        rag = RAGChain.from_documents(
            documents=documents,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            vector_store_type=vector_store_type,
        )

        # 3. 캐시에 저장 (세션 관리)
        _rag_instances[collection_name] = rag

        # 4. 청크 수 계산
        total_chunks = len(rag._vector_store._collection.get()["ids"])

        return {
            "success": True,
            "collection_name": collection_name,
            "document_count": len(documents),
            "chunk_count": total_chunks,
            "embedding_model": embedding_model,
            "vector_store_type": vector_store_type,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def query_rag_system(
    query: str,
    collection_name: str = "default",
    top_k: int = MCPServerConfig.DEFAULT_TOP_K,
    model: str = MCPServerConfig.DEFAULT_CHAT_MODEL,
    temperature: float = 0.7,
) -> dict:
    """
    RAG 시스템에 질의 (기존 코드 재사용)

    Args:
        query: 질문
        collection_name: 사용할 RAG 시스템 이름
        top_k: 검색할 문서 개수
        model: LLM 모델명
        temperature: 생성 온도

    Returns:
        dict: 답변, 출처 문서, 유사도 점수

    Example:
        User: "beanllm이 뭐야?"
        → query_rag_system(query="beanllm이 뭐야?", collection_name="default")
    """
    try:
        # 1. 캐시에서 RAG 인스턴스 가져오기
        if collection_name not in _rag_instances:
            return {
                "success": False,
                "error": f"RAG system '{collection_name}' not found. Please build it first using build_rag_system.",
            }

        rag = _rag_instances[collection_name]

        # 2. 🎯 기존 RAGChain.query() 메서드 사용!
        result = await asyncio.to_thread(
            rag.query, query=query, k=top_k, model=model, temperature=temperature
        )

        # 3. 결과 포매팅
        sources = []
        for i, (doc, score) in enumerate(zip(result.sources, result.scores)):
            sources.append(
                {
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "...",  # 요약
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                }
            )

        return {
            "success": True,
            "answer": result.answer,
            "sources": sources,
            "model": model,
            "query": query,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def get_rag_stats(collection_name: str = "default") -> dict:
    """
    RAG 시스템 통계 조회

    Args:
        collection_name: RAG 시스템 이름

    Returns:
        dict: 문서 개수, 청크 개수, 벡터 차원

    Example:
        User: "현재 RAG 시스템 상태 알려줘"
        → get_rag_stats(collection_name="default")
    """
    try:
        if collection_name not in _rag_instances:
            return {
                "success": False,
                "error": f"RAG system '{collection_name}' not found.",
            }

        rag = _rag_instances[collection_name]

        # 벡터 스토어 통계
        collection = rag._vector_store._collection
        data = collection.get()

        return {
            "success": True,
            "collection_name": collection_name,
            "total_chunks": len(data["ids"]),
            "embedding_dimension": len(data["embeddings"][0]) if data["embeddings"] else 0,
            "vector_store_type": type(rag._vector_store).__name__,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def list_rag_systems() -> dict:
    """
    생성된 RAG 시스템 목록 조회

    Returns:
        dict: RAG 시스템 이름 목록

    Example:
        User: "어떤 RAG 시스템들이 있어?"
        → list_rag_systems()
    """
    return {
        "success": True,
        "collections": list(_rag_instances.keys()),
        "count": len(_rag_instances),
    }


@mcp.tool()
async def delete_rag_system(collection_name: str) -> dict:
    """
    RAG 시스템 삭제

    Args:
        collection_name: 삭제할 RAG 시스템 이름

    Returns:
        dict: 성공 여부

    Example:
        User: "default RAG 시스템 삭제해줘"
        → delete_rag_system(collection_name="default")
    """
    try:
        if collection_name not in _rag_instances:
            return {
                "success": False,
                "error": f"RAG system '{collection_name}' not found.",
            }

        # 캐시에서 제거
        del _rag_instances[collection_name]

        # 벡터 스토어 삭제 (Chroma의 경우)
        vector_store_path = (
            MCPServerConfig.VECTOR_STORE_DIR / f"chroma_{collection_name}"
        )
        if vector_store_path.exists():
            import shutil

            shutil.rmtree(vector_store_path)

        return {
            "success": True,
            "message": f"RAG system '{collection_name}' deleted successfully.",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
