# 코드 품질 규칙

**우선순위**: HIGH
**적용 범위**: 모든 코드 변경

## 중복 코드 제거 (85-90% 감소 목표)

### 데코레이터 패턴 활용

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
        await self._rate_limiter.acquire(key="vision:embedding")

    # 이벤트 발행 로직 10줄
    if self._event_publisher:
        await self._event_publisher.publish("vision_rag.retrieve.start", request)

    # 실제 비즈니스 로직 5줄
    results = self._vector_store.similarity_search(query, k=k)

    # 캐싱 저장 로직 10줄
    if self._cache_enabled:
        await self._cache.set(cache_key, results, ttl=3600)

    return VisionRAGResponse(results=results)

# ✅ After: ~3-5줄 (데코레이터 패턴)
@with_distributed_features(
    pipeline_type="vision_rag",
    enable_cache=True,
    enable_rate_limiting=True,
    enable_event_streaming=True,
    cache_key_prefix="vision_rag:retrieve",
    rate_limit_key="vision:embedding",
    event_type="vision_rag.retrieve",
)
async def retrieve(self, request):
    # 실제 비즈니스 로직만
    results = self._vector_store.similarity_search(query, k=k)
    return VisionRAGResponse(results=results)
```

### 헬퍼 메서드 추출

```python
# ❌ Before: 중복된 CSV 처리 로직
def load_csv(self, file_path):
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"Column1: {row['col1']}, Column2: {row['col2']}"
            metadata = {"file": file_path, "row": row_num}
            # ...

# ✅ After: 헬퍼 메서드로 추출
def _create_content_from_row(self, row: Dict[str, str]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in row.items())

def _create_metadata_from_row(self, file_path: str, row_num: int) -> Dict:
    return {"source": file_path, "row_number": row_num}
```

## 알고리즘 최적화

### O(n) → O(1): 딕셔너리 캐싱

```python
# ❌ Before: O(n) 리스트 순회
def get_model_info(model_name):
    for model in all_models:  # O(n)
        if model["name"] == model_name:
            return model

# ✅ After: O(1) 딕셔너리 조회
MODEL_REGISTRY = {model["name"]: model for model in all_models}  # 사전 캐싱

def get_model_info(model_name):
    return MODEL_REGISTRY.get(model_name)  # O(1)
```

### O(n log n) → O(n log k): heapq 활용

```python
# ❌ Before: O(n log n) 전체 정렬
def get_top_k(scores, k):
    return sorted(scores, reverse=True)[:k]  # 전체 정렬

# ✅ After: O(n log k) 힙 사용
import heapq

def get_top_k(scores, k):
    return heapq.nlargest(k, scores)  # 힙 사용
```

### O(n×m×p) → O(n×m): 패턴 사전 컴파일

```python
# ❌ Before: 매번 정규표현식 컴파일
def exclude_files(files, patterns):
    excluded = []
    for file in files:  # O(n)
        for pattern in patterns:  # O(m)
            if re.match(pattern, file):  # O(p) - 매번 컴파일
                excluded.append(file)

# ✅ After: 사전 컴파일
import re

class DirectoryLoader:
    def __init__(self, exclude_patterns):
        # 초기화 시 한 번만 컴파일
        self._compiled_patterns = [re.compile(p) for p in exclude_patterns]

    def exclude_files(self, files):
        excluded = []
        for file in files:  # O(n)
            for pattern in self._compiled_patterns:  # O(m) - 사전 컴파일됨
                if pattern.match(file):  # O(1) - 이미 컴파일됨
                    excluded.append(file)
```

## Import 규칙

```python
# ✅ 절대 경로만 사용
from beanllm.domain.loaders import DocumentLoader
from beanllm.service.chat_service import IChatService
from beanllm.utils.logger import get_logger

# ❌ 상대 경로 금지
from ...domain.loaders import DocumentLoader
from ..chat_service import IChatService
from ../../utils.logger import get_logger
```

## 타입 힌트 & Docstring

### 모든 함수/메서드에 타입 힌트 작성

```python
# ✅ Good
async def chat(
    self,
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> ChatResponse:
    """
    LLM과 대화합니다.

    Args:
        messages: 대화 메시지 목록 [{"role": "user", "content": "..."}]
        model: 사용할 모델명 (기본: "gpt-4o")
        temperature: 생성 온도 (0.0-2.0, 기본: 0.7)

    Returns:
        ChatResponse: 응답 객체 (content, usage, model 포함)

    Raises:
        ValueError: messages가 비어있는 경우
        APIError: API 호출 실패 시

    Example:
        >>> client = Client(model="gpt-4o")
        >>> response = await client.chat([
        ...     {"role": "user", "content": "Hello"}
        ... ])
        >>> print(response.content)
        "Hello! How can I help you today?"
    """
    pass

# ❌ Bad: 타입 힌트 없음
async def chat(messages, model="gpt-4o"):
    pass
```

### TYPE_CHECKING으로 순환 import 방지

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from beanllm.service.chat_service import IChatService

class ChatHandler:
    def __init__(self, chat_service: "IChatService"):
        self._service = chat_service
```

## 에러 처리

### 적절한 레이어에서 처리

```python
# ✅ Handler에서 입력 검증
class ChatHandler:
    async def handle_chat(self, request: ChatRequest):
        if not request.messages:
            raise ValueError("messages는 비어있을 수 없습니다")

        return await self._service.chat(request)

# ✅ Service에서 비즈니스 로직 에러
class ChatServiceImpl:
    async def chat(self, request: ChatRequest):
        if not self._is_model_available(request.model):
            raise ModelNotAvailableError(f"모델 {request.model}을 사용할 수 없습니다")

        return await self._provider.chat(request.messages)
```

### 민감한 정보 마스킹

```python
from beanllm.utils.integration.security import sanitize_error_message

try:
    response = await provider.chat(messages)
except Exception as e:
    # API 키 등 민감한 정보 마스킹
    safe_message = sanitize_error_message(str(e))
    logger.error(f"Chat failed: {safe_message}")
    raise
```

## 파일 크기 제한

- **목표**: 평균 200줄 이하
- **최대**: 500줄 (초과 시 분할)
- **God Class**: 1,000줄 이상 시 즉시 분해

## 테스트 커버리지

- **목표**: 80% 이상 (현재 61%)
- **필수**: 모든 Public API 테스트
- **권장**: 엣지 케이스, 에러 처리 테스트

```python
# ✅ Good: 엣지 케이스 테스트
def test_chat_with_empty_messages():
    with pytest.raises(ValueError, match="messages는 비어있을 수 없습니다"):
        await client.chat(messages=[])

def test_chat_with_invalid_model():
    with pytest.raises(ModelNotAvailableError):
        await client.chat(messages=[...], model="invalid-model")
```

## 코드 포매팅

### Black (자동 포매팅)

```bash
# 전체 포매팅
black src/beanllm

# 특정 파일
black src/beanllm/facade/core/client_facade.py
```

### Ruff (린팅)

```bash
# 전체 린팅
ruff check src/beanllm

# 자동 수정 가능한 것만
ruff check --fix src/beanllm
```

### MyPy (타입 체크)

```bash
# 전체 타입 체크
mypy src/beanllm

# 특정 파일
mypy src/beanllm/facade/core/client_facade.py
```

## 참고 문서

- `.cursorrules` - 추가 코딩 스타일
- `pyproject.toml` - Black, Ruff, MyPy 설정
- `CLAUDE.md` - 프로젝트 컨텍스트
