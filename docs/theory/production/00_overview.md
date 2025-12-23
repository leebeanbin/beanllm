# Production Features Theory: 프로덕션 기능의 수학적 모델

**석사 수준 이론 문서**  
**기반**: llmkit Callbacks, Caching 실제 구현 분석

---

## 목차

### Part I: 캐싱 이론
1. [LRU 캐시의 수학적 모델](#part-i-캐싱-이론)
2. [캐시 히트율과 성능](#12-캐시-히트율과-성능)
3. [TTL과 시간 기반 만료](#13-ttl과-시간-기반-만료)

### Part II: 속도 제한
4. [토큰 버킷 알고리즘](#part-ii-속도-제한)
5. [Leaky Bucket 알고리즘](#42-leaky-bucket-알고리즘)
6. [Rate Limiting의 수학적 분석](#43-rate-limiting의-수학적-분석)

### Part III: 모니터링
7. [메트릭 수집의 통계적 모델](#part-iii-모니터링)
8. [성능 지표와 분포](#72-성능-지표와-분포)
9. [오류율과 신뢰성](#73-오류율과-신뢰성)

---

## Part I: 캐싱 이론

### 1.1 LRU 캐시의 수학적 모델

#### 정의 1.1.1: LRU (Least Recently Used) 캐시

**LRU 캐시**는 가장 오래 사용되지 않은 항목을 제거합니다:

$$
\text{Cache} = \{(k_1, v_1, t_1), (k_2, v_2, t_2), \ldots, (k_n, v_n, t_n)\}
$$

**제거 규칙:**

$$
\text{evict} = \arg\min_{i} t_i
$$

#### 시각적 표현: LRU 캐시 동작

```
┌─────────────────────────────────────────────────────────┐
│                  LRU 캐시 동작                          │
└─────────────────────────────────────────────────────────┘

초기 상태 (max_size=3):
┌─────────┬─────────┬─────────┐
│ "text1" │ "text2" │ "text3" │  (맨 뒤 = 최근 사용)
└─────────┴─────────┴─────────┘
   (맨 앞 = 오래됨)

1. "text1" 조회:
   ┌─────────┬─────────┬─────────┐
   │ "text2" │ "text3" │ "text1" │  ← "text1" 맨 뒤로 이동
   └─────────┴─────────┴─────────┘

2. "text4" 추가 (캐시 가득 참):
   ┌─────────┬─────────┬─────────┐
   │ "text3" │ "text1" │ "text4" │  ← "text2" 제거 (맨 앞)
   └─────────┴─────────┴─────────┘

3. "text3" 조회:
   ┌─────────┬─────────┬─────────┐
   │ "text1" │ "text4" │ "text3" │  ← "text3" 맨 뒤로 이동
   └─────────┴─────────┴─────────┘
```

#### 구체적 수치 예시

**예시 1.1.1: LRU 캐시 동작**

**설정:**
- `max_size = 3`
- `ttl = 3600` (1시간)

**동작 시퀀스:**

1. **초기 상태:**
   ```
   Cache: ["text1", "text2", "text3"]
   ```

2. **"text1" 조회 (Hit):**
   ```
   Cache: ["text2", "text3", "text1"]  ← LRU 업데이트
   Hit Rate: 1/1 = 100%
   ```

3. **"text4" 추가 (Miss):**
   ```
   Cache: ["text3", "text1", "text4"]  ← "text2" 제거
   Hit Rate: 1/2 = 50%
   ```

4. **"text2" 조회 (Miss, 이미 제거됨):**
   ```
   Cache: ["text1", "text4", "text2"]  ← "text3" 제거
   Hit Rate: 1/3 = 33.3%
   ```

**성능 분석:**
- 캐시 조회: $O(1)$ (해시 테이블)
- LRU 업데이트: $O(1)$ (OrderedDict)
- 총 시간: $O(1)$

**llmkit 구현:**
```python
# domain/embeddings/cache.py: EmbeddingCache
# domain/prompts/cache.py: PromptCache
# domain/graph/node_cache.py: NodeCache
from collections import OrderedDict
import time

class EmbeddingCache:
    """
    LRU + TTL 캐시: 가장 오래 사용되지 않은 항목 제거
    
    수학적 모델:
    - Cache = {(k₁, v₁, t₁), (k₂, v₂, t₂), ..., (kₙ, vₙ, tₙ)}
    - evict = argmin_i t_i (LRU)
    - valid(k) = True if t_current - t_stored < TTL else False
    
    시간 복잡도:
    - get: O(1) (OrderedDict 사용)
    - set: O(1) (OrderedDict 사용)
    - evict: O(1) (맨 앞 항목 제거)
    
    실제 구현:
    - domain/embeddings/cache.py: EmbeddingCache (임베딩 캐시)
    - domain/prompts/cache.py: PromptCache (프롬프트 캐시)
    - domain/graph/node_cache.py: NodeCache (그래프 노드 캐시)
    """
    def __init__(self, ttl: int = 3600, max_size: int = 10000):
        """
        Args:
            ttl: Time To Live (초 단위, 기본값: 3600 = 1시간)
            max_size: 최대 캐시 크기 (기본값: 10000)
        """
        self.cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        캐시 조회: O(1) 시간 복잡도
        
        Process:
        1. 키 존재 확인: O(1)
        2. TTL 검증: O(1)
        3. LRU 업데이트: O(1) (move_to_end)
        
        실제 구현:
        - domain/embeddings/cache.py: EmbeddingCache.get()
        """
        if text not in self.cache:
            self.misses += 1
            return None
        
        vector, stored_time = self.cache[text]
        
        # TTL 검증
        if time.time() - stored_time > self.ttl:
            # 만료된 항목 제거
            del self.cache[text]
            self.misses += 1
            return None
        
        # LRU 업데이트: 사용된 항목을 맨 뒤로 이동
        self.cache.move_to_end(text)
        self.hits += 1
        
        return vector
    
    def set(self, text: str, vector: List[float]):
        """
        캐시 저장: O(1) 시간 복잡도
        
        Process:
        1. 크기 확인: O(1)
        2. 필요시 evict: O(1) (맨 앞 항목 제거)
        3. 항목 추가: O(1)
        
        실제 구현:
        - domain/embeddings/cache.py: EmbeddingCache.set()
        """
        # 크기 제한 확인
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거 (맨 앞)
            self.cache.popitem(last=False)
        
        # 항목 추가 (맨 뒤에 추가 = 최근 사용)
        self.cache[text] = (vector, time.time())
    
    def stats(self) -> Dict[str, Any]:
        """
        캐시 통계: H = Hits / (Hits + Misses)
        
        Returns:
            {
                "hits": hits,
                "misses": misses,
                "hit_rate": H,
                "size": len(cache)
            }
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }
```

---

### 1.2 캐시 히트율과 성능

#### 정의 1.2.1: 캐시 히트율 (Cache Hit Rate)

**캐시 히트율:**

$$
H = \frac{\text{Hits}}{\text{Hits} + \text{Misses}}
$$

**성능 향상:**

$$
T_{\text{avg}} = H \cdot T_{\text{cache}} + (1-H) \cdot T_{\text{compute}}
$$

**llmkit 구현:**
```python
# domain/embeddings/cache.py: EmbeddingCache.get_stats()
def get_stats(self) -> Dict[str, Any]:
    """
    캐시 통계: H = Hits / (Hits + Misses)
    
    실제 구현:
    - domain/embeddings/cache.py: EmbeddingCache.get_stats()
    """
    total = self.hits + self.misses
    hit_rate = self.hits / total if total > 0 else 0
    return {
        "hits": self.hits,
        "misses": self.misses,
        "hit_rate": hit_rate,  # H
        "size": len(self.cache)
    }
```

---

### 1.3 TTL과 시간 기반 만료

#### 정의 1.3.1: TTL (Time To Live)

**TTL 기반 만료:**

$$
\text{valid}(k) = \begin{cases}
\text{True} & \text{if } t_{\text{current}} - t_{\text{stored}} < \text{TTL} \\
\text{False} & \text{otherwise}
\end{cases}
$$

**llmkit 구현:**
```python
# domain/embeddings/cache.py: EmbeddingCache.get()
def get(self, text: str) -> Optional[List[float]]:
    """
    TTL 확인: valid(k) = (t_current - t_stored) < TTL
    
    실제 구현:
    - domain/embeddings/cache.py: EmbeddingCache.get()
    """
    if text not in self.cache:
        return None
    
    vector, timestamp = self.cache[text]
    
    # TTL 확인
    if time.time() - timestamp > self.ttl:
        del self.cache[text]  # 만료된 항목 제거
        return None
    
    return vector
```

---

## Part II: 속도 제한

### 2.1 토큰 버킷 알고리즘

#### 정의 2.1.1: 토큰 버킷 (Token Bucket)

**토큰 버킷 알고리즘:**

$$
\text{tokens}(t) = \min(\text{capacity}, \text{tokens}(t-1) + \text{rate} \times \Delta t)
$$

**요청 허용:**

$$
\text{allow} = \begin{cases}
\text{True} & \text{if } \text{tokens} \geq \text{cost} \\
\text{False} & \text{otherwise}
\end{cases}
$$

#### 시각적 표현: 토큰 버킷 동작

```
┌─────────────────────────────────────────────────────────┐
│                  토큰 버킷 동작                          │
└─────────────────────────────────────────────────────────┘

설정: rate=10 tokens/sec, capacity=20 tokens

시간 →
토큰
  ↑
20 │ ████████████████████  (최대 용량)
   │
15 │ ████████████████
   │
10 │ ████████████
   │
 5 │ ████████
   │
 0 │ ░░░░░░░░░░░░░░░░░░░░
   └──────────────────────────────→ 시간

동작:
t=0s:  tokens=20 (초기)
t=1s:  tokens=20 (충전, 최대치)
      요청 1 (cost=5) → tokens=15 ✓
t=2s:  tokens=15+10=20 (충전)
      요청 2 (cost=5) → tokens=15 ✓
t=3s:  tokens=15+10=20 (충전)
      요청 3 (cost=10) → tokens=10 ✓
t=4s:  tokens=10+10=20 (충전)
      요청 4 (cost=15) → tokens=5 ✓
t=5s:  tokens=5+10=15 (충전)
      요청 5 (cost=20) → tokens=15 < 20 ✗ (거부)
```

**llmkit 구현:**
```python
# utils/error_handling.py: RateLimiter
# decorators/rate_limit.py: @rate_limit 데코레이터
class RateLimiter:
    """
    Rate Limiter: 시간 윈도우 기반 속도 제한
    
    수학적 모델:
    - time_window 내 최대 max_calls 허용
    - allow = True if |{calls in window}| < max_calls else False
    
    실제 구현:
    - utils/error_handling.py: RateLimiter (sliding window 방식)
    - decorators/rate_limit.py: @rate_limit 데코레이터
    - 스레드 안전 (threading.Lock 사용)
    """
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Args:
            config: RateLimitConfig
                - max_calls: 최대 호출 수 (기본값: 10)
                - time_window: 시간 윈도우 (초, 기본값: 60.0)
        """
        self.config = config or RateLimitConfig()
        self.calls = deque()  # 호출 타임스탬프 큐
        self._lock = threading.Lock()
    
    def _is_allowed(self) -> bool:
        """
        호출 허용 여부: allow = len(calls) < max_calls
        
        수학적 표현:
        - window = {c ∈ calls : t_current - c < time_window}
        - allow = True if |window| < max_calls else False
        """
        self._clean_old_calls()  # 오래된 호출 제거
        return len(self.calls) < self.config.max_calls
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Rate limit이 적용된 함수 호출
        
        실제 구현:
        - utils/error_handling.py: RateLimiter.call()
        - decorators/rate_limit.py: @rate_limit 데코레이터
        """
        with self._lock:
            if not self._is_allowed():
                wait_time = self._wait_time()
                raise RateLimitError(
                    f"Rate limit exceeded. Wait {wait_time:.2f}s before retry."
                )
            
            # 호출 기록
            self.calls.append(time.time())
        
        return func(*args, **kwargs)
```

#### 구체적 수치 예시

**예시 2.1.1: 토큰 버킷 계산**

**설정:**
- `rate = 10` tokens/sec
- `capacity = 20` tokens
- 초기 토큰: 20

**시뮬레이션:**

| 시간 | 충전 | 요청 | 토큰 변화 | 결과 |
|------|------|------|----------|------|
| 0.0s | - | - | 20 | 초기 |
| 1.0s | +10 | 요청1 (5) | 20 → 15 | 허용 |
| 2.0s | +10 | 요청2 (5) | 20 → 15 | 허용 |
| 3.0s | +10 | 요청3 (10) | 20 → 10 | 허용 |
| 4.0s | +10 | 요청4 (15) | 20 → 5 | 허용 |
| 5.0s | +10 | 요청5 (20) | 15 → -5 | **거부** |

**평균 처리량:**
$$
\text{Throughput} = \frac{4 \text{ requests}}{5 \text{ seconds}} = 0.8 \text{ req/sec}
$$

**제한된 처리량:**
$$
\text{Throughput}_{\max} = \min(10, \infty) = 10 \text{ req/sec}
$$

**llmkit 구현:**
```python
# callbacks.py: RateLimitingCallback (향후 구현)
class RateLimitingCallback(BaseCallback):
    """
    토큰 버킷: tokens(t) = min(capacity, tokens(t-1) + rate × Δt)
    """
    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate  # 토큰 생성 속도
        self.capacity = capacity  # 최대 토큰 수
        self.tokens = capacity  # 현재 토큰 수
        self.last_update = time.time()
    
    def _refill_tokens(self):
        """토큰 충전"""
        now = time.time()
        delta_t = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + self.rate * delta_t
        )
        self.last_update = now
    
    def allow_request(self, cost: float = 1.0) -> bool:
        """
        요청 허용: allow = (tokens >= cost)
        """
        self._refill_tokens()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
```

---

### 2.2 Leaky Bucket 알고리즘

#### 정의 2.2.1: Leaky Bucket

**Leaky Bucket 알고리즘:**

$$
\text{queue}(t) = \max(0, \text{queue}(t-1) + \text{arrivals} - \text{rate} \times \Delta t)
$$

**llmkit 구현:**
```python
# callbacks.py: LeakyBucketRateLimiter (향후 구현)
class LeakyBucketRateLimiter:
    """
    Leaky Bucket: queue(t) = max(0, queue(t-1) + arrivals - rate × Δt)
    """
    def __init__(self, rate: float = 10.0, capacity: float = 20.0):
        self.rate = rate  # 처리 속도
        self.capacity = capacity  # 최대 큐 크기
        self.queue = 0.0
        self.last_update = time.time()
    
    def allow_request(self) -> bool:
        """
        요청 허용: queue < capacity
        """
        self._update_queue()
        if self.queue < self.capacity:
            self.queue += 1.0
            return True
        return False
    
    def _update_queue(self):
        """큐 업데이트"""
        now = time.time()
        delta_t = now - self.last_update
        self.queue = max(0, self.queue - self.rate * delta_t)
        self.last_update = now
```

---

### 2.3 Rate Limiting의 수학적 분석

#### 정리 2.3.1: 처리량 (Throughput)

**토큰 버킷의 평균 처리량:**

$$
\text{Throughput} = \min(\text{rate}, \text{arrival\_rate})
$$

**llmkit 구현:**
```python
# callbacks.py: RateLimitingCallback
def get_stats(self) -> Dict[str, Any]:
    """
    처리량 통계: Throughput = min(rate, arrival_rate)
    """
    return {
        "rate": self.rate,
        "capacity": self.capacity,
        "current_tokens": self.tokens,
        "throughput": min(self.rate, self.arrival_rate)
    }
```

---

## Part III: 모니터링

### 3.1 메트릭 수집의 통계적 모델

#### 정의 3.1.1: 메트릭 (Metrics)

**메트릭 집합:**

$$
\mathcal{M} = \{\text{latency}, \text{throughput}, \text{error\_rate}, \ldots\}
$$

**llmkit 구현:**
```python
# callbacks.py: CallbackManager
class CallbackManager:
    """
    메트릭 수집: M = {latency, throughput, error_rate, ...}
    """
    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks
        self.metrics: Dict[str, List[float]] = {
            "latency": [],
            "tokens": [],
            "errors": []
        }
    
    def on_llm_end(self, model: str, response: str, tokens_used: int, **kwargs):
        """
        메트릭 수집
        """
        # Latency 측정
        latency = time.time() - self.start_time
        self.metrics["latency"].append(latency)
        self.metrics["tokens"].append(tokens_used)
```

---

### 3.2 성능 지표와 분포

#### 정의 3.2.1: 성능 지표

**평균 지연 시간:**

$$
\bar{L} = \frac{1}{n} \sum_{i=1}^n L_i
$$

**분산:**

$$
\sigma^2 = \frac{1}{n-1} \sum_{i=1}^n (L_i - \bar{L})^2
$$

**llmkit 구현:**
```python
# callbacks.py: TimingCallback
class TimingCallback(BaseCallback):
    """
    성능 지표 계산: L̄ = (1/n) Σ L_i
    """
    def __init__(self):
        self.timings: List[float] = []
    
    def on_llm_end(self, **kwargs):
        """지연 시간 기록"""
        latency = time.time() - self.start_time
        self.timings.append(latency)
    
    def get_stats(self) -> Dict[str, float]:
        """
        통계 계산
        """
        if not self.timings:
            return {}
        
        mean = sum(self.timings) / len(self.timings)  # L̄
        variance = sum((x - mean) ** 2 for x in self.timings) / (len(self.timings) - 1)  # σ²
        std_dev = variance ** 0.5  # σ
        
        return {
            "mean": mean,
            "std_dev": std_dev,
            "min": min(self.timings),
            "max": max(self.timings),
            "p95": sorted(self.timings)[int(len(self.timings) * 0.95)]
        }
```

---

### 3.3 오류율과 신뢰성

#### 정의 3.3.1: 오류율 (Error Rate)

**오류율:**

$$
E = \frac{\text{Errors}}{\text{Total Requests}}
$$

**신뢰성 (Reliability):**

$$
R = 1 - E
$$

**llmkit 구현:**
```python
# callbacks.py: ErrorTrackingCallback
class ErrorTrackingCallback(BaseCallback):
    """
    오류율 계산: E = Errors / Total Requests
    """
    def __init__(self):
        self.total_requests = 0
        self.errors = 0
    
    def on_llm_error(self, **kwargs):
        """오류 기록"""
        self.errors += 1
        self.total_requests += 1
    
    def on_llm_end(self, **kwargs):
        """성공 기록"""
        self.total_requests += 1
    
    def get_error_rate(self) -> float:
        """
        오류율: E = Errors / Total Requests
        """
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests
    
    def get_reliability(self) -> float:
        """
        신뢰성: R = 1 - E
        """
        return 1.0 - self.get_error_rate()
```

---

## 참고 문헌

1. **Tanenbaum & Wetherall (2011)**: "Computer Networks" - 캐싱과 속도 제한
2. **Kleinrock (1975)**: "Queueing Systems" - 큐잉 이론
3. **Jain (1991)**: "The Art of Computer Systems Performance Analysis" - 성능 분석

---

**작성일**: 2025-01-XX  
**버전**: 2.0 (석사 수준 확장)
