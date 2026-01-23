# Backend Development Patterns

**자동 활성화**: Backend API, DB, 캐싱 작업 시
**모델**: sonnet

## Skill Description

FastAPI, 데이터베이스, 캐싱, 비동기 처리 등 백엔드 개발에서 자주 사용되는 패턴을 제공합니다.

## When to Use

이 스킬은 다음 키워드 감지 시 자동 활성화됩니다:
- "API", "endpoint", "FastAPI", "router"
- "database", "DB", "PostgreSQL", "Redis", "Neo4j"
- "caching", "cache", "캐싱"
- "async", "asyncio", "비동기"
- "WebSocket", "SSE", "streaming"

## FastAPI Patterns

### REST API Endpoints

```python
# ✅ Good: Dependency Injection + DTO
from fastapi import APIRouter, Depends, HTTPException
from beanllm.dto.request.core.chat_request import ChatRequest
from beanllm.dto.response.core.chat_response import ChatResponse
from beanllm.service.factory import ServiceFactory

router = APIRouter(prefix="/api/chat", tags=["chat"])

async def get_chat_service():
    """Dependency: Chat service"""
    return ServiceFactory.create_chat_service()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: IChatService = Depends(get_chat_service)
):
    """
    Chat with LLM.

    - **messages**: List of conversation messages
    - **model**: Model name (e.g., "gpt-4o")
    - **temperature**: Sampling temperature (0.0-2.0)
    """
    try:
        return await service.chat(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Request Validation

```python
# ✅ Good: Pydantic validation
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(
        ...,
        description="Conversation messages",
        min_items=1
    )
    model: str = Field(
        "gpt-4o",
        description="Model name",
        pattern="^[a-z0-9-]+$"
    )
    temperature: float = Field(
        0.7,
        description="Sampling temperature",
        ge=0.0,
        le=2.0
    )

    @validator("messages")
    def validate_messages(cls, v):
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Invalid role: {msg['role']}")
        return v
```

### Error Handling

```python
# ✅ Good: Custom exception handler
from fastapi import Request
from fastapi.responses import JSONResponse
from beanllm.utils.exceptions import RateLimitError, APIError

@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc),
            "retry_after": exc.retry_after
        }
    )

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=502,
        content={
            "error": "External API error",
            "detail": str(exc)
        }
    )
```

### Streaming Responses

```python
# ✅ Good: SSE streaming
from fastapi.responses import StreamingResponse

@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: IChatService = Depends(get_chat_service)
):
    """Stream chat responses using Server-Sent Events."""
    async def event_stream():
        try:
            async for chunk in service.stream_chat(request):
                # SSE format: data: {json}\n\n
                yield f"data: {json.dumps({'content': chunk})}\n\n"

            # Send completion event
            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
```

## Database Patterns

### PostgreSQL (pgvector)

```python
# ✅ Good: Connection pool + async
import asyncpg
from typing import List

class PostgresVectorStore:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20
        )

        # Create tables with pgvector extension
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING ivfflat (embedding vector_cosine_ops)
            """)

    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, any]] = None
    ) -> List[Document]:
        """Search similar documents using cosine similarity."""
        async with self.pool.acquire() as conn:
            # Build query with optional filter
            sql = """
                SELECT id, content, metadata, 1 - (embedding <=> $1) AS similarity
                FROM documents
            """
            params = [query_embedding]

            if filter:
                sql += " WHERE metadata @> $2"
                params.append(json.dumps(filter))

            sql += " ORDER BY embedding <=> $1 LIMIT $" + str(len(params) + 1)
            params.append(k)

            rows = await conn.fetch(sql, *params)

            return [
                Document(
                    id=row["id"],
                    content=row["content"],
                    metadata=row["metadata"],
                    similarity=row["similarity"]
                )
                for row in rows
            ]

    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
```

### Redis (Caching)

```python
# ✅ Good: Redis cache decorator
import redis.asyncio as redis
from functools import wraps
import json

class RedisCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    async def get(self, key: str) -> Optional[any]:
        """Get value from cache."""
        value = await self.redis.get(key)
        return json.loads(value) if value else None

    async def set(self, key: str, value: any, ttl: int = 3600):
        """Set value in cache with TTL."""
        await self.redis.setex(
            key,
            ttl,
            json.dumps(value, default=str)
        )

    async def delete(self, key: str):
        """Delete key from cache."""
        await self.redis.delete(key)

# Cache decorator
def with_redis_cache(
    prefix: str,
    ttl: int = 3600,
    key_func: Optional[Callable] = None
):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"{prefix}:{key_func(*args, **kwargs)}"
            else:
                cache_key = f"{prefix}:{args[0]}"  # First arg as key

            # Check cache
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached

            # Execute function
            result = await func(self, *args, **kwargs)

            # Store in cache
            await self._cache.set(cache_key, result, ttl=ttl)

            return result
        return wrapper
    return decorator

# Usage
class EmbeddingService:
    def __init__(self, cache: RedisCache):
        self._cache = cache

    @with_redis_cache(prefix="embed", ttl=86400)
    async def embed(self, text: str) -> List[float]:
        # Expensive operation
        return await self._model.embed(text)
```

### Neo4j (Knowledge Graph)

```python
# ✅ Good: Async Neo4j driver
from neo4j import AsyncGraphDatabase

class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, auth: tuple):
        self.driver = AsyncGraphDatabase.driver(uri, auth=auth)

    async def add_entity(
        self,
        entity: str,
        entity_type: str,
        properties: Dict[str, any]
    ):
        """Add entity to graph."""
        async with self.driver.session() as session:
            await session.run(
                f"""
                MERGE (e:{entity_type} {{name: $name}})
                SET e += $properties
                """,
                name=entity,
                properties=properties
            )

    async def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        properties: Optional[Dict[str, any]] = None
    ):
        """Add relation between entities."""
        async with self.driver.session() as session:
            query = """
                MATCH (s {name: $source})
                MATCH (t {name: $target})
                MERGE (s)-[r:""" + relation_type + """]->(t)
            """
            if properties:
                query += " SET r += $properties"

            await session.run(
                query,
                source=source,
                target=target,
                properties=properties or {}
            )

    async def query_graph(self, cypher: str, params: Dict[str, any] = None):
        """Execute Cypher query."""
        async with self.driver.session() as session:
            result = await session.run(cypher, params or {})
            return [record.data() async for record in result]

    async def close(self):
        """Close driver."""
        await self.driver.close()
```

## Caching Strategies

### In-Memory Cache (LRU)

```python
# ✅ Good: functools.lru_cache
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_model_info(model_name: str) -> Dict[str, any]:
    """Get model info from registry (cached)."""
    # O(1) lookup after first call
    return MODEL_REGISTRY.get(model_name)

# Async version
from beanllm.utils.core.cache import async_lru_cache

@async_lru_cache(maxsize=1000)
async def get_embedding(text: str) -> List[float]:
    """Get text embedding (cached)."""
    return await embedding_model.embed(text)
```

### Multi-Level Cache

```python
# ✅ Good: L1 (memory) + L2 (Redis)
class MultiLevelCache:
    def __init__(self, redis_cache: RedisCache):
        self._l1_cache = {}  # In-memory
        self._l2_cache = redis_cache  # Redis
        self._max_l1_size = 1000

    async def get(self, key: str) -> Optional[any]:
        # L1: Memory
        if key in self._l1_cache:
            return self._l1_cache[key]

        # L2: Redis
        value = await self._l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self._l1_cache[key] = value
            self._evict_l1_if_needed()

        return value

    async def set(self, key: str, value: any, ttl: int = 3600):
        # Set in both levels
        self._l1_cache[key] = value
        await self._l2_cache.set(key, value, ttl=ttl)
        self._evict_l1_if_needed()

    def _evict_l1_if_needed(self):
        """LRU eviction for L1 cache."""
        if len(self._l1_cache) > self._max_l1_size:
            # Remove oldest entry (simplified)
            oldest_key = next(iter(self._l1_cache))
            del self._l1_cache[oldest_key]
```

## Async Patterns

### Concurrent Requests

```python
# ✅ Good: asyncio.gather for parallel execution
import asyncio

async def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts in parallel."""
    tasks = [embed_single(text) for text in texts]
    return await asyncio.gather(*tasks)

# ✅ Good: Semaphore for rate limiting
async def embed_batch_with_limit(texts: List[str], max_concurrent: int = 5):
    """Embed with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_with_semaphore(text: str):
        async with semaphore:
            return await embed_single(text)

    tasks = [embed_with_semaphore(text) for text in texts]
    return await asyncio.gather(*tasks)
```

### Background Tasks

```python
# ✅ Good: FastAPI background tasks
from fastapi import BackgroundTasks

@router.post("/rag/build")
async def build_rag_index(
    documents: List[str],
    background_tasks: BackgroundTasks
):
    """Build RAG index in background."""
    # Start task
    task_id = str(uuid.uuid4())
    background_tasks.add_task(
        build_index_async,
        task_id,
        documents
    )

    return {"task_id": task_id, "status": "processing"}

async def build_index_async(task_id: str, documents: List[str]):
    """Background task: build index."""
    try:
        # Long-running operation
        await rag_service.build_index(documents)
        # Update status
        await redis.set(f"task:{task_id}", "completed")
    except Exception as e:
        await redis.set(f"task:{task_id}", f"failed:{e}")
```

## WebSocket Patterns

```python
# ✅ Good: WebSocket chat
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request = json.loads(data)

            # Stream response
            async for chunk in chat_service.stream(request):
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": chunk
                }))

            # Send completion
            await websocket.send_text(json.dumps({
                "type": "done"
            }))

    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

## Related Documents

- `.claude/rules/coding-standards.md` - 코딩 스타일
- `playground/backend/main.py` - FastAPI 예시
- `CLAUDE.md` - 프로젝트 컨텍스트
