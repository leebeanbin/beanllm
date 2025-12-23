# 성능 최적화 가이드

이 문서는 llmkit의 성능 최적화 방법과 개선 기회를 설명합니다.

## 목차

1. [현재 성능 상태](#현재-성능-상태)
2. [최적화 기회](#최적화-기회)
3. [구현된 최적화](#구현된-최적화)
4. [개선 권장 사항](#개선-권장-사항)
5. [벤치마크 및 측정](#벤치마크-및-측정)

---

## 현재 성능 상태

### 강점

1. **NumPy 벡터화 연산**
   - `domain/embeddings/utils.py`: NumPy를 사용한 벡터 연산
   - SIMD 가속 활용
   - `float32` 사용으로 메모리 효율성

2. **비동기 처리**
   - 대부분의 I/O 작업이 비동기
   - `asyncio.gather()`를 통한 병렬 처리

3. **캐싱 전략**
   - 임베딩 캐싱 (`domain/embeddings/cache.py`)
   - 노드 캐싱 (`domain/graph/node_cache.py`)

### 개선 기회

1. **배치 처리 최적화**
   - `batch_query`가 순차 처리
   - 벡터 검색 배치 처리 부족

2. **비동기 루프 관리**
   - `asyncio.run()` 중복 호출
   - 이벤트 루프 재사용 부족

3. **객체 생성 최적화**
   - Factory 패턴의 반복 생성
   - 반복문 내 객체 생성

4. **메모리 최적화**
   - 대용량 데이터 처리 시 메모리 사용량
   - 스트리밍 처리 개선

---

## 최적화 기회

### 1. 배치 처리 최적화

#### 문제점

**현재 구현 (`facade/rag_facade.py:338-365`):**
```python
def batch_query(self, questions: List[str], k: int = 4, ...) -> List[str]:
    answers = []
    for question in questions:  # 순차 처리
        answer = self.query(question, k=k, model=model, **kwargs)
        answers.append(answer)
    return answers
```

**성능 문제:**
- 순차 처리로 인한 지연 시간 누적
- 각 쿼리가 독립적이므로 병렬 처리 가능
- 시간 복잡도: O(n × t) (n: 질문 수, t: 단일 쿼리 시간)

#### 개선 방안

```python
async def batch_query_async(
    self, questions: List[str], k: int = 4, model: Optional[str] = None, **kwargs
) -> List[str]:
    """
    배치 질의 (병렬 처리)
    
    성능:
    - 순차: O(n × t)
    - 병렬: O(t) (이상적)
    """
    tasks = [
        self.aquery(question, k=k, model=model, **kwargs)
        for question in questions
    ]
    answers = await asyncio.gather(*tasks)
    return answers
```

**예상 성능 향상:**
- 10개 질문: ~10배 빠름
- 100개 질문: ~50-100배 빠름 (네트워크 병목 고려)

### 2. 벡터 검색 배치 처리

#### 문제점

**현재 구현 (`domain/vector_stores/search.py`):**
```python
# 단일 쿼리만 처리
def similarity_search(query: str, k: int = 4) -> List[VectorSearchResult]:
    query_vec = embedding.embed_sync([query])[0]
    # ... 단일 벡터 검색
```

**성능 문제:**
- 여러 쿼리를 순차 처리
- 임베딩 계산도 순차 처리

#### 개선 방안

```python
async def batch_similarity_search(
    self, queries: List[str], k: int = 4
) -> List[List[VectorSearchResult]]:
    """
    배치 벡터 검색
    
    최적화:
    1. 배치 임베딩 계산
    2. 행렬 연산으로 유사도 계산
    3. 병렬 검색
    """
    # 1. 배치 임베딩 (한 번에 계산)
    query_vecs = await self.embedding_service.embed_batch(queries)
    
    # 2. 행렬 연산으로 유사도 계산
    # query_vecs: [n, d], candidate_vecs: [m, d]
    # similarities: [n, m] = query_vecs @ candidate_vecs.T
    similarities = np.dot(query_vecs, self.candidate_vecs.T)
    
    # 3. Top-k 선택 (벡터화)
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    
    # 4. 결과 구성
    results = []
    for i, indices in enumerate(top_k_indices):
        query_results = [
            VectorSearchResult(
                document=self.documents[idx],
                score=similarities[i, idx]
            )
            for idx in indices
        ]
        results.append(query_results)
    
    return results
```

**예상 성능 향상:**
- 10개 쿼리: ~5-10배 빠름
- 100개 쿼리: ~20-50배 빠름

### 3. 비동기 루프 관리 최적화

#### 문제점

**현재 구현 (`facade/web_search_facade.py:94`):**
```python
def search(self, query: str, ...) -> SearchResponse:
    # 매번 새 이벤트 루프 생성
    response = asyncio.run(
        self._web_search_handler.handle_search(...)
    )
```

**성능 문제:**
- `asyncio.run()`은 새 이벤트 루프 생성 및 종료
- 오버헤드 발생
- 기존 루프가 있으면 충돌 가능

#### 개선 방안

```python
def search(self, query: str, ...) -> SearchResponse:
    """
    동기 래퍼 (기존 루프 재사용)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 실행 중인 루프가 있으면 executor 사용
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._web_search_handler.handle_search(...)
                )
                response = future.result()
        else:
            # 루프가 없으면 재사용
            response = loop.run_until_complete(
                self._web_search_handler.handle_search(...)
            )
    except RuntimeError:
        # 루프가 없으면 새로 생성
        response = asyncio.run(
            self._web_search_handler.handle_search(...)
        )
    
    return response
```

**또는 더 나은 방법: 동기 메서드 제거**

```python
# 모든 메서드를 비동기로 통일
async def search_async(self, query: str, ...) -> SearchResponse:
    """비동기 검색 (권장)"""
    return await self._web_search_handler.handle_search(...)
```

### 4. Factory 패턴 최적화 (싱글톤)

#### 문제점

**현재 구현 (`facade/rag_facade.py:73-88`):**
```python
def _init_services(self) -> None:
    # 매번 새 Factory 생성
    provider_factory = SourceProviderFactoryAdapter(SourceProviderFactory)
    service_factory = ServiceFactory(provider_factory=provider_factory, ...)
    handler_factory = HandlerFactory(service_factory)
```

**성능 문제:**
- 매번 새 객체 생성
- 의존성 주입 오버헤드

#### 개선 방안

```python
# 싱글톤 패턴 적용
class ServiceFactory:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, provider_factory=None, ...):
        if hasattr(self, '_initialized'):
            return
        # 초기화 로직
        self._initialized = True
```

**또는 의존성 주입 컨테이너 사용:**

```python
# dependency_injection.py
class DIContainer:
    def __init__(self):
        self._provider_factory = None
        self._service_factory = None
        self._handler_factory = None
    
    @property
    def provider_factory(self):
        if self._provider_factory is None:
            self._provider_factory = SourceProviderFactoryAdapter(SourceProviderFactory)
        return self._provider_factory
    
    @property
    def service_factory(self):
        if self._service_factory is None:
            self._service_factory = ServiceFactory(
                provider_factory=self.provider_factory
            )
        return self._service_factory

# 전역 컨테이너
_container = DIContainer()

# 사용
def _init_services(self) -> None:
    self._rag_handler = _container.handler_factory.create_rag_handler()
```

### 5. 메모리 최적화

#### 문제점

**현재 구현 (`service/impl/rag_service_impl.py:150-156`):**
```python
def _build_context(self, results: List[Any]) -> str:
    context_parts = []
    for i, result in enumerate(results, 1):
        content = result.document.content if hasattr(result, "document") else str(result)
        context_parts.append(f"[{i}] {content}")
    return "\n\n".join(context_parts)
```

**성능 문제:**
- 모든 결과를 메모리에 유지
- 대용량 문서 처리 시 메모리 부족 가능

#### 개선 방안

```python
def _build_context(self, results: List[Any], max_length: int = 4000) -> str:
    """
    컨텍스트 생성 (메모리 효율적)
    
    최적화:
    1. 제너레이터 사용
    2. 길이 제한
    3. 스트리밍 처리
    """
    context_parts = []
    total_length = 0
    
    for i, result in enumerate(results, 1):
        content = result.document.content if hasattr(result, "document") else str(result)
        
        # 길이 제한
        if total_length + len(content) > max_length:
            break
        
        context_parts.append(f"[{i}] {content}")
        total_length += len(content)
    
    return "\n\n".join(context_parts)
```

### 6. 벡터 연산 추가 최적화

#### 현재 구현

**`domain/embeddings/utils.py`**는 이미 NumPy를 사용하지만 추가 최적화 가능:

```python
def batch_cosine_similarity(
    query_vec: List[float],
    candidate_vecs: List[List[float]]
) -> List[float]:
    """
    배치 코사인 유사도 (최적화 버전)
    """
    query = np.array(query_vec, dtype=np.float32)
    candidates = np.array(candidate_vecs, dtype=np.float32)
    
    # 정규화된 벡터라면 내적만으로 계산 가능
    if self._are_normalized:
        similarities = np.dot(candidates, query)
    else:
        # 정규화 필요
        query_norm = np.linalg.norm(query)
        candidate_norms = np.linalg.norm(candidates, axis=1)
        similarities = np.dot(candidates, query) / (candidate_norms * query_norm)
    
    return similarities.tolist()
```

**추가 최적화:**
- 정규화 상태 캐싱
- SIMD 명령어 활용 (NumPy가 자동 처리)
- 메모리 정렬 최적화

---

## 구현된 최적화

### 1. NumPy 벡터화

✅ **구현됨** (`domain/embeddings/utils.py`)
- `cosine_similarity()`: NumPy 사용
- `euclidean_distance()`: NumPy 사용
- `batch_cosine_similarity()`: 배치 처리

**성능:**
- 순수 Python: ~100배 느림
- NumPy: SIMD 가속 활용

### 2. 비동기 처리

✅ **구현됨**
- 대부분의 I/O 작업이 비동기
- `asyncio.gather()` 사용

**예시:**
```python
# service/impl/multi_agent_service_impl.py
tasks = [agent.run(task) for agent in agents]
results = await asyncio.gather(*tasks)
```

### 3. 캐싱

✅ **구현됨**
- 임베딩 캐싱 (`domain/embeddings/cache.py`)
- 노드 캐싱 (`domain/graph/node_cache.py`)
- 프롬프트 캐싱 (`domain/prompts/cache.py`)

---

## 개선 권장 사항

### 우선순위 높음

1. **배치 처리 병렬화**
   - `batch_query` → `batch_query_async`
   - 예상 성능 향상: 10-100배

2. **비동기 루프 관리**
   - `asyncio.run()` 제거
   - 기존 루프 재사용
   - 예상 성능 향상: 10-20%

3. **벡터 검색 배치 처리**
   - 배치 임베딩 계산
   - 행렬 연산 활용
   - 예상 성능 향상: 5-50배

### 우선순위 중간

4. **Factory 싱글톤화**
   - 의존성 주입 컨테이너
   - 예상 성능 향상: 5-10%

5. **메모리 최적화**
   - 스트리밍 처리
   - 길이 제한
   - 예상 메모리 절감: 30-50%

### 우선순위 낮음

6. **벡터 연산 추가 최적화**
   - 정규화 상태 캐싱
   - 예상 성능 향상: 5-10%

---

## 벤치마크 및 측정

### 벤치마크 도구

```python
# tests/benchmark_performance.py
import time
import asyncio
from llmkit import RAGChain

async def benchmark_batch_query():
    """배치 쿼리 성능 측정"""
    rag = RAGChain.from_documents("docs/")
    questions = [f"질문 {i}" for i in range(100)]
    
    # 순차 처리
    start = time.time()
    answers_seq = []
    for q in questions:
        answers_seq.append(await rag.aquery(q))
    seq_time = time.time() - start
    
    # 병렬 처리
    start = time.time()
    tasks = [rag.aquery(q) for q in questions]
    answers_par = await asyncio.gather(*tasks)
    par_time = time.time() - start
    
    print(f"순차: {seq_time:.2f}초")
    print(f"병렬: {par_time:.2f}초")
    print(f"속도 향상: {seq_time/par_time:.2f}배")
```

### 성능 프로파일링

```python
# cProfile 사용
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 코드 실행
rag.query("질문")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 상위 20개 함수
```

### 메모리 프로파일링

```python
# memory_profiler 사용
from memory_profiler import profile

@profile
def test_memory():
    rag = RAGChain.from_documents("large_docs/")
    results = rag.batch_query(questions)
```

---

## 참고 자료

- [NumPy Performance Tips](https://numpy.org/doc/stable/user/basics.performance.html)
- [Python Async Best Practices](https://docs.python.org/3/library/asyncio-dev.html)
- [Memory Profiling in Python](https://pypi.org/project/memory-profiler/)
- [cProfile Documentation](https://docs.python.org/3/library/profile.html)
