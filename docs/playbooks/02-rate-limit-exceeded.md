# Playbook 02 — Rate Limit Exceeded

## 증상

요청이 빈번하게 실패하며 다음 오류가 발생합니다.

```
beanllm.utils.exceptions.RateLimitError: Rate limit exceeded for provider 'openai' — retry after 60s
```

또는 프로바이더 API가 429 응답을 반환하는 경우:

```
httpx.HTTPStatusError: 429 Too Many Requests
x-ratelimit-reset-requests: 60s
```

---

## 원인

- 분당 요청 수(RPM) 또는 분당 토큰 수(TPM)가 프로바이더 한도 초과
- beanllm 내부 RateLimiter가 요청을 차단 (Redis 또는 인메모리)
- 여러 인스턴스에서 동일 API 키를 공유하는 경우 분산 카운트 필요

---

## 프로바이더별 기본 한도 (Tier 1 기준)

| Provider | RPM | TPM | Notes |
|----------|-----|-----|-------|
| OpenAI GPT-4o | 500 | 30,000 | Tier에 따라 상이 |
| OpenAI GPT-4o-mini | 500 | 200,000 | — |
| Claude Sonnet | 50 | 40,000 | — |
| Gemini Flash | 15 | 1,000,000 | 무료 티어 기준 |
| DeepSeek | 300 | — | 매우 관대 |
| Ollama (로컬) | 무제한 | 무제한 | GPU에 따라 처리량 상이 |

실제 한도는 프로바이더 콘솔에서 확인하세요.

---

## 진단

### 1. 오류 발생 빈도 측정

```bash
grep "RateLimitError\|429" logs/beanllm.log | awk '{print $1}' | sort | uniq -c | sort -rn
```

### 2. 현재 요청 속도 확인

```python
# Redis rate limiter 사용 시 현재 카운트 확인
import redis
r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))

# beanllm rate limit 키 패턴: beanllm:ratelimit:{provider}
keys = r.keys("beanllm:ratelimit:*")
for key in keys:
    count = r.get(key)
    ttl = r.ttl(key)
    print(f"{key.decode()}: {count} requests, resets in {ttl}s")
```

### 3. 프로바이더 사용량 확인

- OpenAI: https://platform.openai.com/usage
- Anthropic: https://console.anthropic.com/settings/usage
- Google: https://console.cloud.google.com/apis/

---

## 해결

### 즉각 조치 — 다른 프로바이더로 전환

```python
# 한도가 낮은 프로바이더에서 한도가 높은 프로바이더로 전환
client = Client(model="deepseek-chat", provider="deepseek")  # 관대한 한도
client = Client(model="llama3", provider="ollama")            # 로컬, 무제한
```

### 요청 속도 조절 (클라이언트 측)

```python
import asyncio
from beanllm import Client

client = Client(model="gpt-4o-mini")

async def rate_limited_chat(messages_list: list, delay_seconds: float = 0.5):
    """요청 간 지연을 주어 Rate Limit 방지."""
    results = []
    for messages in messages_list:
        try:
            resp = await client.chat(messages)
            results.append(resp)
        except Exception as e:
            results.append(None)
        await asyncio.sleep(delay_seconds)
    return results
```

### 배치 처리 및 큐잉

```python
import asyncio
from asyncio import Semaphore
from beanllm import Client

client = Client(model="gpt-4o-mini")
semaphore = Semaphore(5)  # 동시 최대 5개 요청

async def limited_chat(messages):
    async with semaphore:
        return await client.chat(messages)

# 병렬 처리 (최대 5개 동시)
tasks = [limited_chat(m) for m in messages_batch]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Redis Rate Limiter 설정

분산 환경에서 여러 인스턴스가 같은 API 키를 공유할 때 Redis로 카운트를 중앙 관리합니다.

```bash
# Redis 설정
REDIS_URL=redis://localhost:6379/0
BEANLLM_RATE_LIMIT_RPM=400      # 분당 최대 요청 수 (한도의 80%)
BEANLLM_RATE_LIMIT_TPM=24000    # 분당 최대 토큰 수
```

```python
# Docker로 Redis 실행
# docker run -d -p 6379:6379 redis:7-alpine
```

### 모델 변경으로 한도 회피

```python
# 고비용·고한도 → 저비용·다른 한도
# 예: gpt-4o (RPM 500) → gpt-4o-mini (RPM 500, TPM 200k)
client = Client(model="gpt-4o-mini")  # 토큰 한도 6배 더 많음
```

---

## 예방

### 지수 백오프 재시도

```python
import asyncio
import random
from beanllm import Client
from beanllm.utils.exceptions import RateLimitError

async def chat_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.chat(messages)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(wait)
```

### 토큰 사용량 모니터링

```python
response = await client.chat(messages)
if response.usage:
    total = response.usage.get("total_tokens", 0)
    if total > 1000:  # 요청당 토큰이 많으면 경고
        logger.warning(f"High token usage: {total} tokens")
```

---

## 관련 문서

- [wiki/providers.md](../../wiki/providers.md#rate-limiting) — Rate Limiting 설정
- [Playbook 01](01-provider-circuit-open.md) — CircuitBreaker OPEN 대응
