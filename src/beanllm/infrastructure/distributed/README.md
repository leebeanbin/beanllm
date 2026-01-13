# 분산 아키텍처 모듈

## 개요

모든 분산 처리 컴포넌트를 추상화하여 환경에 따라 분산/인메모리 선택 가능한 아키텍처입니다.

## 사용법

### 기본 사용

```python
from beanllm.infrastructure.distributed import (
    get_rate_limiter,
    get_cache,
    get_task_queue,
    get_event_bus,
    get_distributed_lock
)

# 환경변수로 자동 선택 (USE_DISTRIBUTED=true/false)
rate_limiter = get_rate_limiter()
cache = get_cache()
task_queue = get_task_queue("ocr.tasks")
producer, consumer = get_event_bus()
lock = get_distributed_lock()
```

### Rate Limiting

```python
# Rate Limiting
await rate_limiter.wait("llm:gpt-4o", cost=1.0)
result = await llm.chat(...)
```

### 캐싱

```python
# 캐싱
cached = await cache.get("embedding:123")
if not cached:
    cached = await embed("text")
    await cache.set("embedding:123", cached, ttl=3600)
```

### 작업 큐

```python
from beanllm.infrastructure.distributed import TaskProcessor

# 작업 큐에 추가
processor = TaskProcessor("ocr.tasks")
task_id = await processor.enqueue_task(
    "ocr.recognize",
    {"image_path": "image.jpg", "engine": "paddleocr"}
)

# Worker에서 처리
result = await processor.process_task(
    "ocr.recognize",
    handler=recognize_image
)
```

### 이벤트 스트리밍

```python
from beanllm.infrastructure.distributed import (
    with_event_publishing,
    get_event_logger
)

# 데코레이터 사용
@with_event_publishing("document.loaded")
async def load_document(path: str):
    # 문서 로드
    return document

# 직접 사용
event_logger = get_event_logger()
await event_logger.log_event("rag.query", {
    "question": "What is AI?",
    "result_count": 5
})
```

### 분산 락

```python
from beanllm.infrastructure.distributed import (
    with_distributed_lock,
    get_lock_manager
)

# 데코레이터 사용
@with_distributed_lock("vector_store:update:123")
async def update_vector_store(store_id: str):
    # 벡터 스토어 업데이트
    await vector_store.add_documents(docs)

# Context Manager 사용
lock_manager = get_lock_manager()
async with lock_manager.acquire_resource_lock("vector_store", "123"):
    await vector_store.add_documents(docs)
```

### 동시성 제어

```python
from beanllm.infrastructure.distributed import ConcurrencyController

controller = ConcurrencyController()

async with controller.with_concurrency_control(
    "ocr",
    max_concurrent=5,
    rate_limit_key="ocr:paddleocr"
):
    # OCR 처리
    result = await process_ocr(image)
```

### 메시지 기반 아키텍처

```python
from beanllm.infrastructure.distributed import (
    MessageProducer,
    DistributedErrorHandler,
    RequestMonitor
)

# 메시지 발행
producer = MessageProducer()
request_id = await producer.publish_request(
    "ocr.recognize",
    {"image_path": "image.jpg", "engine": "paddleocr"}
)

# 오류 처리
error_handler = DistributedErrorHandler()
await error_handler.handle_error(
    request_id,
    error,
    "ocr.recognize"
)

# 요청 상태 조회
monitor = RequestMonitor()
status = await monitor.get_request_status(request_id)
```

## 환경 설정

### 개발 환경 (인메모리)

```bash
# .env
USE_DISTRIBUTED=false
```

### 프로덕션 환경 (분산)

```bash
# .env
USE_DISTRIBUTED=true
REDIS_HOST=redis.example.com
REDIS_PORT=6379
REDIS_PASSWORD=your_password
KAFKA_BOOTSTRAP_SERVERS=kafka1.example.com:9092,kafka2.example.com:9092
KAFKA_CLIENT_ID=beanllm
```

## 설치

```bash
# 분산 모드 의존성 설치
pip install beanllm[distributed]
```

## 아키텍처

- **인터페이스**: 모든 컴포넌트의 추상 인터페이스 정의
- **인메모리 구현**: 기존 코드를 래핑한 인메모리 구현
- **Redis 구현**: Redis 기반 분산 구현
- **Kafka 구현**: Kafka 기반 분산 구현
- **팩토리**: 환경변수에 따라 자동 선택
- **통합 모듈**: 기존 코드와의 통합 (데코레이터, Helper 클래스)

## 마이그레이션 완료

### Rate Limiting
- ✅ `facade/core/rag_facade.py`: `RAGChain.batch_query()`
- ✅ `service/impl/ml/evaluation_service_impl.py`: `EvaluationServiceImpl.batch_evaluate()`
- ✅ `domain/evaluation/evaluator.py`: `Evaluator.batch_evaluate_async()`

### 캐싱
- ✅ `domain/embeddings/utils/cache.py`: `EmbeddingCache`
- ✅ `domain/prompts/cache.py`: `PromptCache`
- ✅ `domain/graph/node_cache.py`: `NodeCache`

### 작업 큐
- ✅ `infrastructure/distributed/task_processor.py`: `TaskProcessor`, `BatchProcessor`

### 이벤트 스트리밍
- ✅ `infrastructure/distributed/event_integration.py`: `with_event_publishing`, `EventLogger`

### 분산 락
- ✅ `infrastructure/distributed/lock_integration.py`: `with_distributed_lock`, `LockManager`

## 주의사항

1. **비동기/동기 호환성**: 캐시는 비동기 인터페이스를 사용하지만, `SyncCacheWrapper`로 동기 코드와 호환 가능
2. **Fallback 메커니즘**: Redis/Kafka 연결 실패 시 자동으로 인메모리 모드로 전환
3. **에러 처리**: 모든 분산 컴포넌트는 에러 발생 시 안전하게 처리 (fallback 또는 로깅)
4. **성능**: 분산 모드에서는 네트워크 지연이 발생할 수 있으므로, 단일 서버 환경에서는 인메모리 모드 권장
