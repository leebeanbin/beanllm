# 🏗️ beanllm 개발 가이드

**Claude Code를 위한 개발 원칙 및 최적화 기법 가이드**

이 문서는 beanllm 프로젝트의 아키텍처 원칙, 최적화 기법, 코딩 스타일을 정의합니다. Claude Code가 코드를 작성하거나 리팩토링할 때 이 가이드를 따라야 합니다.

> 📖 **프로젝트 전체 맥락**: [claude.md](../claude.md)를 먼저 읽어주세요.

---

## 📋 목차

1. [아키텍처 원칙](#아키텍처-원칙)
2. [최적화 기법](#최적화-기법)
3. [코딩 스타일](#코딩-스타일)
4. [리팩토링 패턴](#리팩토링-패턴)
5. [데코레이터 패턴](#데코레이터-패턴)
6. [분산 아키텍처](#분산-아키텍처)

---

## 아키텍처 원칙

### 1. Clean Architecture + SOLID 원칙

beanllm은 **Clean Architecture**와 **SOLID 원칙**을 엄격히 준수합니다.

#### 레이어 구조

```
Facade Layer (사용자 API)
    ↓
Handler Layer (입력 검증, 에러 처리)
    ↓
Service Layer (비즈니스 로직)
    ↓
Domain Layer (핵심 비즈니스 규칙)
    ↑
Infrastructure Layer (외부 시스템 구현)
```

#### 의존성 방향 (절대 규칙)

**✅ 허용된 의존성:**
- Facade → Handler, DTO, Utils, Domain/Infrastructure (직접 사용 가능)
- Handler → Service (인터페이스), DTO, Utils
- Service → Domain (인터페이스), Infrastructure (인터페이스), DTO
- Domain → Domain 내부만
- Infrastructure → Domain (인터페이스), Utils

**❌ 금지된 의존성:**
- 순환 의존 (Circular Dependency)
- 역방향 의존 (하위 레이어 → 상위 레이어)
- 구현체 직접 의존 (인터페이스 사용 필수)
- Handler/Facade → Service 구현체
- Domain → Service/Handler/Facade

자세한 내용: [DEPENDENCY_RULES.md](../DEPENDENCY_RULES.md)

#### SOLID 원칙 적용

**1. Single Responsibility Principle (SRP)**
- 각 클래스는 하나의 책임만 가집니다
- Handler: 입력 검증, 에러 처리만
- Service: 비즈니스 로직만
- Domain: 핵심 비즈니스 규칙만

**2. Open/Closed Principle (OCP)**
- 새로운 Provider 추가 시 기존 코드 수정 불필요
- Strategy 패턴으로 확장 가능

**3. Liskov Substitution Principle (LSP)**
- 인터페이스 구현으로 대체 가능
- 모든 Provider는 BaseLLMProvider를 구현

**4. Interface Segregation Principle (ISP)**
- 작은, 특화된 인터페이스
- IChatService, IRAGService 등 분리

**5. Dependency Inversion Principle (DIP)**
- 상위 레이어가 하위 레이어의 인터페이스에 의존
- Factory 패턴으로 의존성 주입

### 2. Domain-Driven Design (DDD)

**핵심 원칙:**
- Domain Layer는 순수 비즈니스 로직만 포함
- Infrastructure Layer는 Domain 인터페이스를 구현
- Domain은 어떤 외부 의존성도 가지지 않음

**예시:**
```python
# ✅ 올바른 예: Domain은 인터페이스만 정의
# domain/vector_stores/base.py
class BaseVectorStore(ABC):
    @abstractmethod
    def similarity_search(self, query: str, k: int) -> List[VectorSearchResult]:
        pass

# ✅ 올바른 예: Infrastructure는 Domain 인터페이스 구현
# infrastructure/vector_stores/chroma.py
class ChromaVectorStore(BaseVectorStore):
    def similarity_search(self, query: str, k: int) -> List[VectorSearchResult]:
        # ChromaDB 구현
        pass

# ❌ 잘못된 예: Domain이 Infrastructure에 의존
# domain/vector_stores/base.py
from infrastructure.vector_stores.chroma import ChromaVectorStore  # ❌ 금지!
```

---

## 최적화 기법

### 1. 데코레이터 패턴으로 중복 코드 제거

**원칙:** 반복되는 패턴은 데코레이터로 추출하여 코드 중복을 85-90% 감소시킵니다.

#### Before (중복 코드 많음)

```python
async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
    # 캐시 확인 (10줄)
    cache_key = f"vision_rag:retrieve:{hashlib.md5(...).hexdigest()}"
    cache = get_rag_cache(ttl=3600, max_size=1000)
    cached_result = await cache.get(cache_key) if USE_DISTRIBUTED else cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit...")
        return VisionRAGResponse(results=cached_result)
    
    # Rate Limiting (5줄)
    rate_limiter = get_rate_limiter()
    await rate_limiter.wait("vision:embedding", cost=1.0)
    
    # 실제 로직 (3줄)
    results = self._vector_store.similarity_search(query, k=k)
    
    # 캐시 저장 (5줄)
    if USE_DISTRIBUTED:
        await cache.set(cache_key, results, ttl=3600)
    else:
        cache.set(cache_key, results, ttl=3600)
    
    # 이벤트 발행 (5줄)
    event_logger = get_event_logger()
    await event_logger.log_event("vision_rag.retrieve", {...}, level="info")
    
    return VisionRAGResponse(results=results)
    # 총 ~28줄
```

#### After (데코레이터 사용)

```python
@with_distributed_features(
    pipeline_type="vision_rag",
    enable_cache=True,
    enable_rate_limiting=True,
    enable_event_streaming=True,
    cache_key_prefix="vision_rag:retrieve",
    rate_limit_key="vision:embedding",
    event_type="vision_rag.retrieve",
)
async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
    # 실제 로직만 (3줄)
    results = self._vector_store.similarity_search(query, k=k)
    return VisionRAGResponse(results=results)
    # 총 ~3줄 (약 90% 감소)
```

### 2. 알고리즘 최적화

#### O(n) → O(1) 최적화

**예시: Model Parameter Lookup**

```python
# ❌ Before: O(n) 선형 검색
def get_model_params(model: str):
    for m in MODELS:
        if m["name"] == model:
            return m["params"]
    return None

# ✅ After: O(1) 딕셔너리 조회
MODEL_PARAMETER_CACHE = {
    "gpt-4o": {"supports_temperature": True, ...},
    "claude-sonnet-4": {"supports_temperature": True, ...},
    # ...
}

def get_model_params(model: str):
    return MODEL_PARAMETER_CACHE.get(model)
```

**성능:** 100× speedup

#### O(n log n) → O(n log k) 최적화

**예시: Hybrid Search Top-k**

```python
# ❌ Before: 전체 정렬 후 상위 k개 선택
results = sorted(all_results, key=lambda x: x.score, reverse=True)[:k]

# ✅ After: heapq.nlargest()로 상위 k개만 선택
import heapq
results = heapq.nlargest(k, all_results, key=lambda x: x.score)
```

**성능:** 10-50% faster (k << n일 때)

#### O(n×m×p) → O(n×m) 최적화

**예시: Directory Loading Pattern Matching**

```python
# ❌ Before: 매 파일마다 패턴 컴파일
for file_path in files:
    for pattern in exclude_patterns:
        if file_path.match(pattern):  # 매번 컴파일
            should_exclude = True

# ✅ After: 패턴 사전 컴파일
from fnmatch import translate
compiled_patterns = [re.compile(translate(p)) for p in exclude_patterns]
for file_path in files:
    for pattern in compiled_patterns:
        if pattern.match(str(file_path)):  # 컴파일된 패턴 사용
            should_exclude = True
```

**성능:** 1000× faster (1000 files, 10 patterns)

### 3. 메모리 최적화

#### Lazy Loading

**원칙:** 모델이나 리소스는 필요할 때만 로드합니다.

```python
class LazyLoadMixin:
    _model = None
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model
```

#### Streaming

**원칙:** 대용량 데이터는 스트리밍으로 처리합니다.

```python
async def load_streaming(self) -> AsyncIterator[Document]:
    """스트리밍 방식으로 문서 로드"""
    with open(self.path, 'r', encoding='utf-8') as f:
        for line in f:
            yield Document(content=line.strip())
```

#### Memory Mapping (mmap)

**원칙:** 10MB 이상 파일은 자동으로 mmap 사용합니다.

```python
if file_size > 10 * 1024 * 1024:  # 10MB
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # 메모리 매핑 사용
            content = mm.read()
```

### 4. 병렬 처리 최적화

#### ProcessPoolExecutor (CPU-bound 작업)

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(process_file, f) for f in files]
    results = [f.result() for f in futures]
```

#### asyncio.gather() (I/O-bound 작업)

```python
results = await asyncio.gather(*[
    process_item(item) for item in items
])
```

#### 분산 큐 (대규모 작업)

```python
from beanllm.infrastructure.distributed import BatchProcessor

batch_processor = BatchProcessor(task_type="ocr.tasks", max_concurrent=10)
results = await batch_processor.process_items(
    task_name="recognize",
    items=images,
    item_to_task_data=lambda img: {"image_path": img},
    handler=process_image,
)
```

---

## 코딩 스타일

### 1. Import 규칙

**절대 규칙:** 모든 import는 절대 경로를 사용합니다.

```python
# ✅ 올바른 예
from beanllm.domain.loaders import Document
from beanllm.infrastructure.distributed import get_rate_limiter
from beanllm.utils.logging import get_logger

# ❌ 잘못된 예
from ...domain.loaders import Document  # 상대 경로 금지
from ..infrastructure import get_rate_limiter  # 상대 경로 금지
```

### 2. 타입 힌트

**원칙:** 모든 함수와 메서드에 타입 힌트를 작성합니다.

```python
# ✅ 올바른 예
async def retrieve(
    self, 
    request: VisionRAGRequest
) -> VisionRAGResponse:
    """이미지 검색"""
    pass

# ❌ 잘못된 예
async def retrieve(self, request):  # 타입 힌트 없음
    pass
```

### 3. Docstring

**원칙:** 모든 클래스와 메서드에 docstring을 작성합니다.

```python
class VisionRAGService:
    """
    Vision RAG 서비스
    
    이미지 기반 질문-답변 시스템을 제공합니다.
    
    Example:
        ```python
        service = VisionRAGService(vector_store, vision_embedding)
        response = await service.query(request)
        ```
    """
    
    async def retrieve(
        self, 
        request: VisionRAGRequest
    ) -> VisionRAGResponse:
        """
        이미지 검색
        
        Args:
            request: Vision RAG 요청 DTO
            
        Returns:
            VisionRAGResponse: 검색 결과
            
        Example:
            ```python
            request = VisionRAGRequest(query="cat", k=5)
            response = await service.retrieve(request)
            ```
        """
        pass
```

### 4. 에러 처리

**원칙:** 에러는 적절한 레이어에서 처리하고, 민감한 정보는 마스킹합니다.

```python
# ✅ 올바른 예: Handler에서 에러 처리
@handle_errors
async def handle_query(self, request: RAGRequest) -> RAGResponse:
    try:
        return await self._service.query(request)
    except ValueError as e:
        raise BadRequestError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {sanitize_error_message(str(e))}")
        raise InternalServerError("Internal server error")

# ✅ 올바른 예: Provider에서 에러 처리
@provider_error_handler(
    operation="chat",
    api_error_types=(openai.APIError,),
)
async def chat(self, messages, **kwargs):
    # 에러는 자동으로 ProviderError로 변환되고 마스킹됨
    pass
```

### 5. 로깅

**원칙:** 구조화된 로깅을 사용하고, 민감한 정보는 자동 마스킹합니다.

```python
from beanllm.utils.logging import get_logger

logger = get_logger(__name__)

# ✅ 올바른 예: 구조화된 로깅
logger.info("Processing request", extra={
    "request_id": request_id,
    "model": model,
    "token_count": token_count,
})

# ✅ 자동 마스킹: API 키는 자동으로 마스킹됨
logger.error(f"API call failed: {error}")  # API 키는 ***MASKED***로 표시
```

---

## 리팩토링 패턴

### 1. 중복 코드 제거

**패턴:** 반복되는 패턴을 헬퍼 메서드나 데코레이터로 추출합니다.

#### Before

```python
# 여러 곳에서 반복되는 패턴
async def method1(self, request):
    cache_key = f"prefix:{hashlib.md5(...).hexdigest()}"
    cache = get_cache()
    cached = await cache.get(cache_key)
    if cached:
        return cached
    # ... 로직
    await cache.set(cache_key, result)

async def method2(self, request):
    cache_key = f"prefix:{hashlib.md5(...).hexdigest()}"
    cache = get_cache()
    cached = await cache.get(cache_key)
    if cached:
        return cached
    # ... 로직
    await cache.set(cache_key, result)
```

#### After

```python
# 데코레이터로 추출
@with_distributed_features(
    pipeline_type="my_pipeline",
    enable_cache=True,
    cache_key_prefix="prefix",
)
async def method1(self, request):
    # 실제 로직만
    pass

@with_distributed_features(
    pipeline_type="my_pipeline",
    enable_cache=True,
    cache_key_prefix="prefix",
)
async def method2(self, request):
    # 실제 로직만
    pass
```

### 2. God Class 분해

**패턴:** 큰 클래스를 책임별로 작은 클래스로 분해합니다.

#### Before

```python
# ❌ God Class (1,845 lines)
class VisionModels:
    def load_sam(self): ...
    def load_yolo(self): ...
    def load_florence(self): ...
    def load_qwen3vl(self): ...
    # ... 100+ methods
```

#### After

```python
# ✅ 분해된 클래스들
class SAM3Model(BaseVisionModel):
    def load(self): ...
    def segment(self): ...

class YOLOv12Model(BaseVisionModel):
    def load(self): ...
    def detect(self): ...

class Florence2Model(BaseVisionModel):
    def load(self): ...
    def process(self): ...
```

### 3. 인터페이스 추출

**패턴:** 공통 동작을 인터페이스로 추출합니다.

```python
# ✅ 인터페이스 정의
class IEmbedding(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        pass

# ✅ 구현체들
class OpenAIEmbedding(IEmbedding):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI 구현
        pass

class HuggingFaceEmbedding(IEmbedding):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        # HuggingFace 구현
        pass
```

---

## 데코레이터 패턴

### 1. 분산 시스템 데코레이터

**원칙:** 분산 시스템 기능(캐싱, Rate Limiting, 이벤트 스트리밍, 분산 락)은 데코레이터로 자동 적용합니다.

#### 기본 사용법

```python
from beanllm.infrastructure.distributed import with_distributed_features

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
async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
    # 실제 로직만 작성
    results = self._vector_store.similarity_search(query, k=k)
    return VisionRAGResponse(results=results)
```

#### 동적 Rate Limiting 키

```python
@with_distributed_features(
    pipeline_type="vision_rag",
    enable_rate_limiting=True,
    rate_limit_key=lambda self, args, kwargs: f"llm:{(args[0] if args else kwargs.get('request')).llm_model if hasattr(args[0] if args else kwargs.get('request'), 'llm_model') else 'default'}",
)
async def query(self, request: VisionRAGRequest) -> VisionRAGResponse:
    # request.llm_model에 따라 동적으로 Rate Limiting 키 생성
    pass
```

#### 동적 분산 락 키

```python
@with_distributed_features(
    pipeline_type="ocr",
    enable_distributed_lock=True,
    lock_key=lambda self, args, kwargs: f"ocr:file:{hashlib.md5(str(args[0]).encode()).hexdigest() if args else 'default'}",
)
async def recognize(self, image_path: str) -> OCRResult:
    # 파일 경로 기반으로 동적으로 락 키 생성
    pass
```

### 2. 배치 처리 데코레이터

**원칙:** 배치 처리는 `@with_batch_processing` 데코레이터로 자동화합니다.

```python
from beanllm.infrastructure.distributed import with_batch_processing

@with_batch_processing(
    pipeline_type="ocr",
    max_concurrent=10,
    use_distributed_queue=True,
)
async def batch_recognize(self, images: List[str]) -> List[OCRResult]:
    # 각 이미지 처리 로직만 작성
    # 데코레이터가 자동으로 분산 큐 사용, 동시성 제어
    pass
```

### 3. 기존 데코레이터 활용

**원칙:** 기존 데코레이터를 적극 활용합니다.

```python
from beanllm.decorators import (
    handle_errors,
    log_execution,
    validate_input,
    provider_error_handler,
)

@handle_errors
@log_execution
@validate_input
async def process(self, request: Request) -> Response:
    pass

@provider_error_handler(
    operation="chat",
    api_error_types=(openai.APIError,),
)
async def chat(self, messages, **kwargs):
    pass
```

---

## 분산 아키텍처

### 1. 환경변수 기반 선택

**원칙:** 환경변수 `USE_DISTRIBUTED`로 분산/인메모리 모드를 자동 선택합니다.

```python
# 환경변수 설정
USE_DISTRIBUTED=true  # 분산 모드 (Redis/Kafka)
USE_DISTRIBUTED=false  # 인메모리 모드 (기본)

# 코드에서는 자동 선택
from beanllm.infrastructure.distributed import get_rate_limiter

rate_limiter = get_rate_limiter()  # USE_DISTRIBUTED에 따라 자동 선택
```

### 2. 동적 설정 변경

**원칙:** 런타임에 파이프라인별 설정을 자유롭게 수정할 수 있습니다.

```python
from beanllm.infrastructure.distributed import (
    update_pipeline_config,
    get_pipeline_config,
    reset_pipeline_config,
)

# 설정 수정
update_pipeline_config("vision_rag", enable_rate_limiting=False)
update_pipeline_config("chain", chain_cache_ttl=7200)

# 설정 조회
config = get_pipeline_config("vision_rag")
print(config.enable_rate_limiting)  # False

# 설정 초기화
reset_pipeline_config("vision_rag")
```

### 3. Fallback 메커니즘

**원칙:** 분산 컴포넌트 실패 시 자동으로 인메모리로 fallback합니다.

```python
# 자동 fallback (데코레이터 내부에서 처리)
@with_distributed_features(...)
async def method(self, request):
    # Redis 실패 시 자동으로 InMemoryRateLimiter 사용
    # Kafka 실패 시 자동으로 InMemoryTaskQueue 사용
    pass
```

### 4. 컴포넌트별 역할

**Redis:**
- Rate Limiting (빠른 응답 필요)
- 캐싱 (빠른 조회 필요)
- 분산 락 (빠른 락 획득 필요)
- 단기 큐 (빠른 작업 처리)

**Kafka:**
- 이벤트 스트리밍 (영구 저장 필요)
- 장기 작업 큐 (영구 저장 필요)
- 로그 수집 (영구 저장 필요)

---

## 코드 작성 체크리스트

새로운 코드를 작성할 때 다음을 확인하세요:

### ✅ 아키텍처 준수

- [ ] Clean Architecture 레이어 구조 준수
- [ ] 의존성 방향 준수 (역방향 의존 없음)
- [ ] SOLID 원칙 준수
- [ ] 인터페이스에 의존 (구현체 직접 의존 없음)

### ✅ 최적화

- [ ] 중복 코드가 있으면 데코레이터나 헬퍼 메서드로 추출
- [ ] 알고리즘 복잡도 최적화 (O(n) → O(1), O(n log n) → O(n log k))
- [ ] 대용량 데이터는 스트리밍 또는 배치 처리
- [ ] 불필요한 메모리 할당 최소화

### ✅ 코드 품질

- [ ] 모든 import는 절대 경로 사용
- [ ] 모든 함수/메서드에 타입 힌트 작성
- [ ] 모든 클래스/메서드에 docstring 작성
- [ ] 에러 처리는 적절한 레이어에서 수행
- [ ] 민감한 정보는 자동 마스킹

### ✅ 분산 시스템

- [ ] 분산 시스템 기능은 데코레이터로 적용
- [ ] 환경변수 기반 자동 선택
- [ ] Fallback 메커니즘 고려
- [ ] 동적 설정 변경 지원

---

## 예시: 올바른 코드 작성

### ✅ 완벽한 예시

```python
"""
Vision RAG Service Implementation

Clean Architecture + SOLID 원칙 준수
- SRP: Vision RAG 비즈니스 로직만 담당
- DIP: 인터페이스에 의존 (IVisionRAGService)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from beanllm.dto.request.ml.vision_rag_request import VisionRAGRequest
from beanllm.dto.response.ml.vision_rag_response import VisionRAGResponse
from beanllm.infrastructure.distributed.pipeline_decorators import (
    with_distributed_features,
)
from beanllm.utils.logging import get_logger

if TYPE_CHECKING:
    from beanllm.domain.vector_stores import BaseVectorStore
    from beanllm.domain.vision.embeddings import BaseVisionEmbedding

from ..vision_rag_service import IVisionRAGService

logger = get_logger(__name__)


class VisionRAGServiceImpl(IVisionRAGService):
    """
    Vision RAG 서비스 구현체
    
    책임:
    - Vision RAG 비즈니스 로직만
    - 검증 없음 (Handler에서 처리)
    - 에러 처리 없음 (Handler에서 처리)
    
    SOLID:
    - SRP: Vision RAG 비즈니스 로직만
    - DIP: 인터페이스에 의존 (의존성 주입)
    
    Example:
        ```python
        service = VisionRAGServiceImpl(
            vector_store=vector_store,
            vision_embedding=vision_embedding,
        )
        response = await service.retrieve(request)
        ```
    """
    
    def __init__(
        self,
        vector_store: "BaseVectorStore",
        vision_embedding: Optional["BaseVisionEmbedding"] = None,
        chat_service: Optional[Any] = None,
        llm: Optional[Any] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        """
        Args:
            vector_store: 벡터 스토어
            vision_embedding: Vision 임베딩 (선택적)
            chat_service: 채팅 서비스 (선택적)
            llm: LLM Client (선택적)
            prompt_template: 프롬프트 템플릿 (선택적)
        """
        self._vector_store = vector_store
        self._vision_embedding = vision_embedding
        self._chat_service = chat_service
        self._llm = llm
        self._prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
    
    @with_distributed_features(
        pipeline_type="vision_rag",
        enable_cache=True,
        enable_rate_limiting=True,
        enable_event_streaming=True,
        cache_key_prefix="vision_rag:retrieve",
        rate_limit_key="vision:embedding",
        event_type="vision_rag.retrieve",
    )
    async def retrieve(self, request: VisionRAGRequest) -> VisionRAGResponse:
        """
        이미지 검색
        
        Args:
            request: Vision RAG 요청 DTO
            
        Returns:
            VisionRAGResponse: 검색 결과
            
        Example:
            ```python
            request = VisionRAGRequest(query="cat", k=5)
            response = await service.retrieve(request)
            ```
        """
        query = request.query or ""
        k = request.k
        
        # 실제 로직만 작성 (캐싱, Rate Limiting, 이벤트 스트리밍 자동 적용)
        results = self._vector_store.similarity_search(query, k=k)
        
        return VisionRAGResponse(results=results)
```

---

## 참고 자료

- [claude.md](../claude.md) - 프로젝트 전체 맥락 및 방향성
- [ARCHITECTURE.md](../ARCHITECTURE.md) - 아키텍처 상세 설명
- [DEPENDENCY_RULES.md](../DEPENDENCY_RULES.md) - 의존성 규칙 상세 가이드
- [src/beanllm/infrastructure/distributed/README.md](../src/beanllm/infrastructure/distributed/README.md) - 분산 아키텍처 상세

---

**최종 업데이트**: 2026-01-XX

