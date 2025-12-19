# Production Features 실무 가이드: 프로덕션 기능 활용

**실무 적용 문서**

---

## 목차

1. [캐싱](#1-캐싱)
2. [Rate Limiting](#2-rate-limiting)
3. [모니터링](#3-모니터링)
4. [에러 처리](#4-에러-처리)

---

## 1. 캐싱

```python
from llmkit.callbacks import CachingCallback

cache = CachingCallback(ttl=3600, max_size=10000)

# LLM 호출 캐싱
result = llm.chat(messages, callbacks=[cache])
```

---

## 2. Rate Limiting

```python
from llmkit.callbacks import RateLimitingCallback

rate_limiter = RateLimitingCallback(
    rate=10.0,  # 초당 10개
    capacity=20.0
)

# Rate limit 적용
result = llm.chat(messages, callbacks=[rate_limiter])
```

---

## 3. 모니터링

```python
from llmkit.callbacks import TimingCallback, CostTrackingCallback

timing = TimingCallback()
cost = CostTrackingCallback()

result = llm.chat(messages, callbacks=[timing, cost])

# 통계 확인
print(f"Latency: {timing.get_stats()}")
print(f"Cost: {cost.get_stats()}")
```

---

## 4. 에러 처리

```python
from llmkit.callbacks import ErrorTrackingCallback

error_tracker = ErrorTrackingCallback()

try:
    result = llm.chat(messages, callbacks=[error_tracker])
except Exception as e:
    error_tracker.on_error(e)
    print(f"Error rate: {error_tracker.get_error_rate()}")
```

---

**작성일**: 2025-01-XX  
**버전**: 1.0

