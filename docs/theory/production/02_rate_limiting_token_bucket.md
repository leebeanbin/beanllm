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

```python
class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
    
    def allow_request(self, cost=1.0):
        self._refill_tokens()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
    
    def _refill_tokens(self):
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

