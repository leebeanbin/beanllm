# Rate Limiting: Token Bucket Algorithm: 속도 제한과 토큰 버킷

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: 속도 제한 이론

---

## 목차

1. [토큰 버킷 알고리즘](#1-토큰-버킷-알고리즘)
2. [Leaky Bucket 알고리즘](#2-leaky-bucket-알고리즘)
3. [수학적 분석](#3-수학적-분석)
4. [CS 관점: 구현과 최적화](#4-cs-관점-구현과-최적화)

---

## 1. 토큰 버킷 알고리즘

### 1.1 토큰 버킷 정의

#### 정의 1.1.1: Token Bucket

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

### 1.2 시각적 표현: 토큰 버킷

```
토큰 버킷 동작 (rate=10/sec, capacity=20):

시간 →
토큰
  ↑
20 │ ████████████████████  (최대)
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
t=0s:  tokens=20
t=1s:  tokens=20 (충전, 최대치)
      요청 1 (cost=5) → tokens=15 ✓
t=2s:  tokens=15+10=20 (충전)
      요청 2 (cost=5) → tokens=15 ✓
```

---

## 2. Leaky Bucket 알고리즘

### 2.1 Leaky Bucket 정의

#### 정의 2.1.1: Leaky Bucket

**Leaky Bucket:**

$$
\text{queue}(t) = \max(0, \text{queue}(t-1) + \text{arrivals} - \text{rate} \times \Delta t)
$$

---

## 3. 수학적 분석

### 3.1 처리량

#### 정리 3.1.1: 처리량

**평균 처리량:**

$$
\text{Throughput} = \min(\text{rate}, \text{arrival\_rate})
$$

---

## 4. CS 관점: 구현과 최적화

### 4.1 구현

#### 구현 4.1.1: Token Bucket

**llmkit 구현:**
```python
# utils/error_handling.py: RateLimiter
# decorators/rate_limit.py: @rate_limit 데코레이터
import time
from collections import deque
import threading

class RateLimiter:
    """
    Rate Limiter: 시간 윈도우 기반 속도 제한
    
    수학적 모델:
    - time_window 내 최대 max_calls 허용
    - allow = True if len(calls) < max_calls else False
    
    시간 복잡도:
    - _is_allowed: O(n) where n = calls in window (최적화 가능)
    - call: O(1) amortized
    
    실제 구현:
    - utils/error_handling.py: RateLimiter (시간 윈도우 기반)
    - decorators/rate_limit.py: @rate_limit 데코레이터
    - sliding window 방식 사용 (deque)
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
        self._lock = threading.Lock()  # 스레드 안전성
    
    def _clean_old_calls(self):
        """
        오래된 호출 기록 제거
        
        Process:
        1. 현재 시간 기준 cutoff 계산
        2. cutoff 이전의 호출 제거
        
        시간 복잡도: O(n) where n = expired calls
        """
        now = time.time()
        cutoff = now - self.config.time_window
        
        while self.calls and self.calls[0] < cutoff:
            self.calls.popleft()
    
    def _is_allowed(self) -> bool:
        """
        호출 허용 여부: allow = len(calls) < max_calls
        
        수학적 표현:
        - allow = True if |{c ∈ calls : t_current - c < time_window}| < max_calls
        - allow = False otherwise
        """
        self._clean_old_calls()
        return len(self.calls) < self.config.max_calls
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Rate limit이 적용된 함수 호출
        
        Process:
        1. 허용 여부 확인: _is_allowed()
        2. 허용되면 호출 기록 추가
        3. 함수 실행
        
        실제 구현:
        - utils/error_handling.py: RateLimiter.call()
        - decorators/rate_limit.py: @rate_limit 데코레이터 (이 클래스 사용)
        """
        with self._lock:
            if not self._is_allowed():
                wait_time = self._wait_time()
                raise RateLimitError(
                    f"Rate limit exceeded. Wait {wait_time:.2f}s before retry."
                )
            
            # 호출 기록 추가
            self.calls.append(time.time())
        
        # 함수 실행
        return func(*args, **kwargs)
    
    def _wait_time(self) -> float:
        """
        대기 시간 계산
        
        수학적 표현:
        - oldest_call = min(calls)
        - elapsed = t_current - oldest_call
        - wait_time = max(0, time_window - elapsed)
        """
        if not self.calls:
            return 0.0
        
        oldest_call = self.calls[0]
        elapsed = time.time() - oldest_call
        remaining = self.config.time_window - elapsed
        
        return max(0.0, remaining)
```

**Token Bucket 구현 (참고용):**
```python
# 참고: llmkit은 현재 sliding window 방식 사용
# Token Bucket은 향후 추가 가능
class TokenBucket:
    """
    토큰 버킷: tokens(t) = min(capacity, tokens(t-1) + rate × Δt)
    
    수학적 모델:
    - tokens(t) = min(capacity, tokens(t-1) + rate × Δt)
    - allow = True if tokens ≥ cost else False
    
    실제 구현:
    - 현재 llmkit은 sliding window 방식 사용
    - Token Bucket은 향후 추가 예정
    """
    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: 토큰 충전 속도 (tokens/sec)
            capacity: 최대 토큰 수
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def allow_request(self, cost: float = 1.0) -> bool:
        """
        요청 허용 여부: allow = tokens ≥ cost
        
        Process:
        1. 토큰 충전: _refill_tokens()
        2. 허용 여부 확인: tokens ≥ cost
        3. 토큰 소비: tokens -= cost
        """
        self._refill_tokens()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
    
    def _refill_tokens(self):
        """
        토큰 충전: tokens(t) = min(capacity, tokens(t-1) + rate × Δt)
        
        수학적 표현:
        - Δt = t_current - t_last_update
        - tokens_new = min(capacity, tokens_old + rate × Δt)
        """
        now = time.time()
        delta_t = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + self.rate * delta_t
        )
        self.last_update = now
```

---

## 질문과 답변 (Q&A)

### Q1: 토큰 버킷 vs Leaky Bucket?

**A:** 비교:

**토큰 버킷:**
- 버스트 허용
- 토큰 축적 가능
- 일반적으로 사용

**Leaky Bucket:**
- 일정한 처리 속도
- 버스트 제한
- 특수한 경우

**권장:** 토큰 버킷

---

## 참고 문헌

1. **Tanenbaum & Wetherall (2011)**: "Computer Networks" - 속도 제한

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

