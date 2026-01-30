"""
Message Vector Store Service

메시지를 Vector DB에 저장하고 검색하는 서비스
MongoDB는 세션 메타데이터만 관리, Vector DB는 메시지 내용 저장
"""
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Vector Store (메시지 저장용)
_message_vector_store = None

try:
    from beanllm.domain.vector_stores.local.chroma import ChromaVectorStore
    
    # 임베딩 함수 - beanllm에서 지원하는 오픈소스 임베더 사용
    # 우선순위: 1) Ollama (로컬 서버), 2) HuggingFace (로컬, 오픈소스)
    embedding_func = None
    
    # 1. Ollama 임베딩 시도 (로컬 서버, 빠름)
    try:
        from beanllm.domain.embeddings import OllamaEmbedding
        
        # beanllm에서 지원하는 Ollama 임베딩 모델:
        # - nomic-embed-text (기본값, 빠르고 안정적)
        # - mxbai-embed-large (더 큰 모델, 더 정확)
        # - all-minilm (경량 모델)
        embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        _embedding_function = OllamaEmbedding(model=embedding_model)
        embedding_func = _embedding_function.embed_sync  # 동기 버전 사용
        logger.info(f"✅ Using Ollama embedding model: {embedding_model}")
    except Exception as e:
        logger.info(f"ℹ️  Ollama embedding not available: {e}")
        logger.info("   Trying HuggingFace embedding (open source, local)...")
        
        # 2. HuggingFace 임베딩 fallback (로컬, 오픈소스, API 키 불필요)
        try:
            from beanllm.domain.embeddings import HuggingFaceEmbedding
            
            # beanllm에서 지원하는 HuggingFace 모델:
            # - sentence-transformers/all-MiniLM-L6-v2 (기본값, 빠르고 가벼움)
            # - nvidia/NV-Embed-v2 (MTEB 1위, 최고 성능)
            # - BAAI/bge-large-en-v1.5 (BGE, 높은 성능)
            # - intfloat/e5-large-v2 (E5, 범용)
            hf_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            _embedding_function = HuggingFaceEmbedding(
                model=hf_model,
                use_gpu=os.getenv("USE_GPU", "false").lower() == "true",  # GPU 사용 여부
                normalize=True,
                batch_size=32
            )
            embedding_func = _embedding_function.embed_sync  # 동기 버전 사용
            logger.info(f"✅ Using HuggingFace embedding model: {hf_model}")
            logger.info("   (Open source, local, no API key required)")
        except Exception as e2:
            embedding_func = None
            logger.warning(f"⚠️  HuggingFace embedding not available: {e2}")
            logger.warning("   해결 방법:")
            logger.warning("   1. Ollama: ollama serve && ollama pull nomic-embed-text")
            logger.warning("   2. HuggingFace: pip install sentence-transformers")
    
    # Vector Store 초기화 (메시지 저장용)
    if embedding_func:
        _message_vector_store = ChromaVectorStore(
            collection_name="chat_messages",  # 메시지 전용 컬렉션
            embedding_function=embedding_func,
            persist_directory="./.chroma_messages"  # 메시지 저장 전용
        )
        logger.info("✅ Message vector store initialized")
    else:
        logger.warning("⚠️  Embedding function not available, message vector storage disabled")
except Exception as e:
    logger.warning(f"⚠️  Message vector store not available: {e}")
    _message_vector_store = None


class MessageVectorStore:
    """메시지 Vector DB 저장/검색 서비스"""
    
    @staticmethod
    async def save_message(
        session_id: str,
        message_id: str,
        role: str,
        content: str,
        model: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        메시지를 Vector DB에 저장
        
        Args:
            session_id: 세션 ID
            message_id: 메시지 ID (고유)
            role: 메시지 역할 (user/assistant)
            content: 메시지 내용
            model: 사용된 모델
            timestamp: 타임스탬프
            metadata: 추가 메타데이터
            
        Returns:
            저장 성공 여부
        """
        if not _message_vector_store:
            return False
        
        try:
            from beanllm.domain.loaders import Document
            
            # 메시지를 Document로 변환
            doc = Document(
                content=content,
                metadata={
                    "session_id": session_id,
                    "message_id": message_id,
                    "role": role,
                    "model": model,
                    "timestamp": timestamp.isoformat(),
                    **(metadata or {})
                }
            )
            
            # Vector DB에 저장 (message_id를 ID로 사용)
            # Chroma collection에 직접 추가
            texts = [content]
            metadatas = [{
                "session_id": session_id,
                "message_id": message_id,
                "role": role,
                "model": model,
                "timestamp": timestamp.isoformat(),
                **(metadata or {})
            }]
            
            # 임베딩 생성 (동기 함수이므로 asyncio.to_thread 사용)
            import asyncio
            if embedding_func:
                embeddings = await asyncio.to_thread(embedding_func, texts)
            else:
                embeddings = None
            
            # Chroma에 추가 (upsert로 중복 방지, 동기 함수이므로 asyncio.to_thread 사용)
            if embeddings:
                await asyncio.to_thread(
                    _message_vector_store.collection.upsert,
                    documents=texts,
                    metadatas=metadatas,
                    ids=[message_id],  # message_id를 ID로 사용
                    embeddings=embeddings
                )
            else:
                await asyncio.to_thread(
                    _message_vector_store.collection.upsert,
                    documents=texts,
                    metadatas=metadatas,
                    ids=[message_id]
                )
            
            logger.debug(f"✅ Saved message to vector DB: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save message to vector DB: {e}")
            return False
    
    @staticmethod
    async def get_session_messages(
        session_id: str,
        limit: Optional[int] = None,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        세션의 모든 메시지를 Vector DB에서 가져오기
        
        Args:
            session_id: 세션 ID
            limit: 최대 메시지 수
            role: 역할 필터 (user/assistant)
            
        Returns:
            메시지 리스트
        """
        if not _message_vector_store:
            return []
        
        try:
            import asyncio
            # Chroma에서 session_id로 필터링하여 가져오기
            # Chroma는 where 조건으로 필터링 가능
            where_filter = {"session_id": session_id}
            if role:
                where_filter["role"] = role
            
            # 모든 메시지 가져오기 (limit이 없으면 모두, 동기 함수이므로 asyncio.to_thread 사용)
            results = await asyncio.to_thread(
                _message_vector_store.collection.get,
                where=where_filter,
                limit=limit
            )
            
            # timestamp 순으로 정렬
            messages = []
            for i, message_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                messages.append({
                    "message_id": message_id,
                    "role": metadata.get("role", "user"),
                    "content": content,
                    "model": metadata.get("model", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "metadata": {k: v for k, v in metadata.items() 
                               if k not in ["session_id", "message_id", "role", "model", "timestamp"]}
                })
            
            # timestamp 순으로 정렬
            messages.sort(key=lambda x: x.get("timestamp", ""))
            
            logger.debug(f"✅ Retrieved {len(messages)} messages from vector DB for session: {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get messages from vector DB: {e}")
            return []
    
    @staticmethod
    async def search_messages(
        query: str,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        메시지 내용으로 검색 (의미 기반)
        
        Args:
            query: 검색 쿼리
            session_id: 세션 ID 필터 (옵션)
            role: 역할 필터 (옵션)
            k: 반환할 결과 수
            
        Returns:
            검색 결과 (메시지 + 관련도 점수)
        """
        if not _message_vector_store:
            return []
        
        try:
            import asyncio
            # Vector DB로 유사도 검색 (동기 함수이므로 asyncio.to_thread 사용)
            vector_results = await asyncio.to_thread(
                _message_vector_store.similarity_search, query, k=k * 2
            )
            
            # 필터링
            filtered_results = []
            for result in vector_results:
                # VectorSearchResult 객체 처리
                if hasattr(result, 'metadata'):
                    metadata = result.metadata
                elif hasattr(result, 'document') and hasattr(result.document, 'metadata'):
                    metadata = result.document.metadata
                else:
                    metadata = {}
                
                # session_id 필터
                if session_id and metadata.get("session_id") != session_id:
                    continue
                
                # role 필터
                if role and metadata.get("role") != role:
                    continue
                
                # content 추출
                if hasattr(result, 'document'):
                    content = result.document.content if hasattr(result.document, 'content') else str(result.document)
                else:
                    content = str(result)
                
                # score 추출
                score = getattr(result, 'score', 0.0)
                
                filtered_results.append({
                    "message_id": metadata.get("message_id", ""),
                    "session_id": metadata.get("session_id", ""),
                    "role": metadata.get("role", "user"),
                    "content": content,
                    "model": metadata.get("model", ""),
                    "timestamp": metadata.get("timestamp", ""),
                    "relevance_score": score,
                    "metadata": {k: v for k, v in metadata.items() 
                               if k not in ["session_id", "message_id", "role", "model", "timestamp"]}
                })
            
            # 관련도 점수 순으로 정렬
            filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # k개만 반환
            filtered_results = filtered_results[:k]
            
            logger.debug(f"✅ Found {len(filtered_results)} messages matching query: {query}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to search messages in vector DB: {e}")
            return []
    
    @staticmethod
    async def delete_session_messages(session_id: str) -> bool:
        """
        세션의 모든 메시지를 Vector DB에서 삭제
        
        Args:
            session_id: 세션 ID
            
        Returns:
            삭제 성공 여부
        """
        if not _message_vector_store:
            return False
        
        try:
            import asyncio
            # session_id로 모든 메시지 ID 가져오기 (동기 함수이므로 asyncio.to_thread 사용)
            results = await asyncio.to_thread(
                _message_vector_store.collection.get,
                where={"session_id": session_id}
            )
            
            if results["ids"]:
                # 모든 메시지 삭제 (동기 함수이므로 asyncio.to_thread 사용)
                await asyncio.to_thread(
                    _message_vector_store.collection.delete,
                    ids=results["ids"]
                )
                logger.debug(f"✅ Deleted {len(results['ids'])} messages from vector DB for session: {session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete messages from vector DB: {e}")
            return False
    
    @staticmethod
    async def delete_message(message_id: str) -> bool:
        """
        단일 메시지 삭제
        
        Args:
            message_id: 메시지 ID
            
        Returns:
            삭제 성공 여부
        """
        if not _message_vector_store:
            return False
        
        try:
            import asyncio
            # 동기 함수이므로 asyncio.to_thread 사용
            await asyncio.to_thread(
                _message_vector_store.collection.delete,
                ids=[message_id]
            )
            logger.debug(f"✅ Deleted message from vector DB: {message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete message from vector DB: {e}")
            return False


# 전역 인스턴스
message_vector_store = MessageVectorStore()
