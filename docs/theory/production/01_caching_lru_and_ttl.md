# Caching: LRU and TTL: 캐싱 전략

**석사 수준 이론 문서**  
**CS 관점 + 수학적 엄밀성**  
**기반**: llmkit EmbeddingCache 실제 구현 분석

---

## 목차

1. [LRU 캐시의 수학적 모델](#1-lru-캐시의-수학적-모델)
2. [TTL과 시간 기반 만료](#2-ttl과-시간-기반-만료)
3. [캐시 히트율과 성능](#3-캐시-히트율과-성능)
4. [CS 관점: 구현과 최적화](#4-cs-관점-구현과-최적화)

---

## 1. LRU 캐시의 수학적 모델

### 1.1 LRU 정의

#### 정의 1.1.1: LRU (Least Recently Used)

**LRU 캐시**는 가장 오래 사용되지 않은 항목을 제거합니다:

$$
\text{Cache} = \{(k_1, v_1, t_1), (k_2, v_2, t_2), \ldots, (k_n, v_n, t_n)\}
$$

**제거 규칙:**

$$
\text{evict} = \arg\min_{i} t_i
$$

### 1.2 시각적 표현: LRU 동작

```
LRU 캐시 동작 (max_size=3):

초기: ["text1", "text2", "text3"]
      (맨 뒤 = 최근 사용)

"text1" 조회:
  ["text2", "text3", "text1"]  ← "text1" 맨 뒤로

"text4" 추가 (가득 참):
  ["text3", "text1", "text4"]  ← "text2" 제거 (맨 앞)
```

---

## 2. TTL과 시간 기반 만료

### 2.1 TTL 정의

#### 정의 2.1.1: TTL (Time To Live)

**TTL 기반 만료:**

$$
\text{valid}(k) = \begin{cases}
\text{True} & \text{if } t_{\text{current}} - t_{\text{stored}} < \text{TTL} \\
\text{False} & \text{otherwise}
\end{cases}
$$

---

## 3. 캐시 히트율과 성능

### 3.1 히트율 정의

#### 정의 3.1.1: Cache Hit Rate

**캐시 히트율:**

$$
H = \frac{\text{Hits}}{\text{Hits} + \text{Misses}}
$$

**성능 향상:**

$$
T_{\text{avg}} = H \cdot T_{\text{cache}} + (1-H) \cdot T_{\text{compute}}
$$

---

## 4. CS 관점: 구현과 최적화

### 4.1 OrderedDict 사용

#### 구현 4.1.1: LRU Cache

**llmkit 구현:**
```python
# domain/embeddings/cache.py: EmbeddingCache
from collections import OrderedDict
from datetime import datetime, timedelta

class EmbeddingCache:
    """
    LRU + TTL 캐시
    
    제거 규칙: evict = argmin_i t_i
    TTL 검증: valid(k) = (t_current - t_stored < TTL)
    """
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Args:
            max_size: 최대 캐시 크기
            ttl_seconds: TTL (초), None이면 만료 없음
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """
        캐시 조회
        
        Process:
        1. 키 존재 확인
        2. TTL 검증
        3. LRU 업데이트 (맨 뒤로 이동)
        """
        if key not in self._cache:
            return None
        
        value, stored_time = self._cache[key]
        
        # TTL 검증
        if self.ttl_seconds is not None:
            if datetime.now() - stored_time > timedelta(seconds=self.ttl_seconds):
                del self._cache[key]
                return None
        
        # LRU 업데이트 (맨 뒤로 이동)
        self._cache.move_to_end(key)
        
        return value
    
    def set(self, key: str, value: Any):
        """
        캐시 저장
        
        Process:
        1. 키가 이미 있으면 업데이트
        2. 캐시가 가득 차면 LRU 제거
        3. 새 항목 추가
        """
        if key in self._cache:
            # 업데이트
            self._cache.move_to_end(key)
        elif len(self._cache) >= self.max_size:
            # LRU 제거 (맨 앞 항목)
            self._cache.popitem(last=False)
        
        self._cache[key] = (value, datetime.now())
        self._cache.move_to_end(key)
```
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # O(1)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # O(1)
        self.cache[key] = value
```

**시간 복잡도:** $O(1)$ (모든 연산)

---

## 질문과 답변 (Q&A)

### Q1: LRU vs FIFO vs Random?

**A:** 비교:

**LRU:**
- 최근 사용 고려
- 일반적으로 최고 성능
- 권장

**FIFO:**
- 단순하지만 성능 낮음

**Random:**
- 구현 간단
- 성능 중간

---

## 참고 문헌

1. **Tanenbaum & Wetherall (2011)**: "Computer Networks" - 캐싱

---

**작성일**: 2025-01-XX  
**버전**: 3.0 (CS 관점 + 석사 수준 확장)

