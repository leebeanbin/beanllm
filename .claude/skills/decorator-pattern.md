# Decorator Pattern for Code Deduplication

**자동 활성화**: 중복 코드 감지 시
**모델**: sonnet

## Skill Description

데코레이터 패턴을 사용하여 중복 코드를 85-90% 감소시킵니다. 캐싱, Rate Limiting, 이벤트 스트리밍 등 반복되는 패턴을 데코레이터로 추출합니다.

## When to Use

이 스킬은 다음 키워드 감지 시 자동 활성화됩니다:
- "중복", "duplicate", "반복", "repetitive"
- "캐싱", "caching", "rate limiting", "이벤트"
- "데코레이터", "decorator", "@"
- "리팩토링", "refactor"

## Pattern Recognition

### 중복 코드 패턴

```python
# ❌ Pattern 1: 캐싱 로직 중복
async def retrieve_a(self, query):
    cache_key = f"rag:{query}"
    cached = await self._cache.get(cache_key)
    if cached:
        return cached

    results = self._vector_store.search(query)

    await self._cache.set(cache_key, results, ttl=3600)
    return results

async def retrieve_b(self, query):
    cache_key = f"vision_rag:{query}"
    cached = await self._cache.get(cache_key)
    if cached:
        return cached

    results = self._vector_store.search(query)

    await self._cache.set(cache_key, results, ttl=3600)
    return results

# ✅ Solution: 데코레이터로 추출
@with_cache(prefix="rag", ttl=3600)
async def retrieve_a(self, query):
    return self._vector_store.search(query)

@with_cache(prefix="vision_rag", ttl=3600)
async def retrieve_b(self, query):
    return self._vector_store.search(query)
```

### 분산 시스템 패턴

```python
# ❌ Before: 각 메서드마다 ~30-50줄
async def retrieve(self, request):
    # 캐싱 로직 20줄
    if self._cache_enabled:
        cache_key = f"vision_rag:{request.query}"
        cached = await self._cache.get(cache_key)
        if cached:
            return cached

    # Rate limiting 로직 15줄
    if self._rate_limiter:
        rate_limit_key = f"vision:embedding"
        await self._rate_limiter.acquire(key=rate_limit_key)

    # 이벤트 발행 로직 10줄
    if self._event_publisher:
        event_data = {
            "query": request.query,
            "timestamp": datetime.now()
        }
        await self._event_publisher.publish("vision_rag.retrieve.start", event_data)

    # 분산 락 로직 10줄
    async with self._distributed_lock.lock(f"vision_rag:{request.query}"):
        # 실제 비즈니스 로직 5줄
        results = self._vector_store.similarity_search(query, k=k)

    # 캐싱 저장 로직 10줄
    if self._cache_enabled:
        await self._cache.set(cache_key, results, ttl=3600)

    # 이벤트 발행 로직 10줄
    if self._event_publisher:
        await self._event_publisher.publish("vision_rag.retrieve.end", results)

    return VisionRAGResponse(results=results)

# ✅ After: 데코레이터 패턴 (~3-5줄)
@with_distributed_features(
    pipeline_type="vision_rag",
    enable_cache=True,
    enable_rate_limiting=True,
    enable_event_streaming=True,
    enable_distributed_lock=True,
    cache_key_prefix="vision_rag:retrieve",
    rate_limit_key="vision:embedding",
    event_type="vision_rag.retrieve",
)
async def retrieve(self, request):
    # 실제 비즈니스 로직만
    results = self._vector_store.similarity_search(query, k=k)
    return VisionRAGResponse(results=results)
```

## Refactoring Steps

### 1. 중복 패턴 식별

```python
# 중복 패턴 찾기
# 1. 같은 코드 블록이 3번 이상 반복
# 2. 약간의 차이만 있음 (변수명, 파라미터)
# 3. 같은 구조의 try-except-finally
```

### 2. 데코레이터 설계

```python
# 공통 패턴 추출
def with_cache(prefix: str, ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Before: 캐싱 체크
            cache_key = f"{prefix}:{args[0]}"  # 첫 번째 인자를 키로 사용
            cached = await self._cache.get(cache_key)
            if cached:
                return cached

            # 실제 함수 실행
            result = await func(self, *args, **kwargs)

            # After: 캐싱 저장
            await self._cache.set(cache_key, result, ttl=ttl)
            return result

        return wrapper
    return decorator
```

### 3. 기존 코드 리팩토링

```python
# Before
async def method_a(self, query):
    cache_key = f"rag:{query}"
    cached = await self._cache.get(cache_key)
    if cached:
        return cached

    results = self._process(query)

    await self._cache.set(cache_key, results, ttl=3600)
    return results

# After
@with_cache(prefix="rag", ttl=3600)
async def method_a(self, query):
    return self._process(query)
```

## Built-in Decorators

beanllm 프로젝트에서 사용 가능한 데코레이터:

### 1. `@with_distributed_features`

**위치**: `beanllm/infrastructure/distributed/decorators.py`

```python
from beanllm.infrastructure.distributed import with_distributed_features

@with_distributed_features(
    pipeline_type="rag",
    enable_cache=True,           # Redis 캐싱
    enable_rate_limiting=True,   # Redis Rate Limiter
    enable_event_streaming=True, # Kafka 이벤트
    enable_distributed_lock=True,# Redis 분산 락
    cache_key_prefix="rag:query",
    cache_ttl=3600,
    rate_limit_key="rag:api",
    rate_limit_max_requests=100,
    rate_limit_window_seconds=60,
    event_type="rag.query",
)
async def query(self, request: RAGRequest) -> RAGResponse:
    # 비즈니스 로직만 작성
    pass
```

### 2. `@with_batch_processing`

**위치**: `beanllm/infrastructure/distributed/decorators.py`

```python
from beanllm.infrastructure.distributed import with_batch_processing

@with_batch_processing(
    batch_size=32,
    max_wait_seconds=1.0
)
async def embed_batch(self, texts: List[str]) -> List[List[float]]:
    # 배치 처리 로직
    return self._embedding_model.embed(texts)
```

### 3. Custom Decorators

프로젝트에 맞는 커스텀 데코레이터 생성:

```python
# utils/decorators.py
from functools import wraps
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)

def with_logging(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise
    return wrapper

def with_retry(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator
```

## Metrics

리팩토링 후 다음 지표를 측정합니다:

- **코드 라인 수 감소**: ~85-90% (각 메서드마다 ~30-50줄 → ~3-5줄)
- **코드 중복 감소**: 중복 블록 수 측정
- **가독성 향상**: 비즈니스 로직만 남음

## Example Output

```
리팩토링 완료:

변경 파일:
- src/beanllm/service/impl/advanced/vision_rag_service_impl.py

Before:
- retrieve() 메서드: 65줄
- build() 메서드: 58줄
- query() 메서드: 72줄
총 195줄

After:
- retrieve() 메서드: 5줄 (@with_distributed_features)
- build() 메서드: 4줄 (@with_distributed_features)
- query() 메서드: 6줄 (@with_distributed_features)
총 15줄

코드 감소: 92% (195줄 → 15줄)
```

## Related Documents

- `.claude/rules/code-quality.md` - 코드 품질 규칙
- `src/beanllm/infrastructure/distributed/README.md` - 분산 아키텍처
- `.cursorrules` - 데코레이터 패턴 가이드
