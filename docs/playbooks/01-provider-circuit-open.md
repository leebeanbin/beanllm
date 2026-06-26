# Playbook 01 — Provider CircuitBreaker OPEN

## 증상

특정 프로바이더로의 요청이 즉시 실패하며 다음 오류가 발생합니다.

```
beanllm.utils.exceptions.CircuitBreakerError: Circuit is OPEN for provider 'openai'. Try again in 42.3s
```

로그에서 확인:
```
[ERROR] Circuit OPEN for openai — blocking requests for 60s
[WARN]  Consecutive failures: 5/5 — tripping circuit breaker
```

---

## 원인

`BaseLLMProvider`의 CircuitBreaker가 연속 5회 실패를 감지하면 60초간 OPEN 상태로 전환합니다.

OPEN 트리거 원인:
- 프로바이더 API 다운 또는 장애
- API 키 만료·무효
- 네트워크 문제 (DNS, 방화벽)
- Rate Limit 장시간 초과 (연속 429)
- 잘못된 엔드포인트 URL

---

## 진단

### 1. 로그에서 실패 원인 확인

```bash
# 최근 provider 오류 확인
grep -E "Circuit|OPEN|failure" logs/beanllm.log | tail -20

# 연속 실패 직전 오류 메시지 확인
grep -E "ProviderError|HTTPError|ConnectionError" logs/beanllm.log | tail -10
```

### 2. API 키 유효성 확인

```bash
# 환경변수 확인
echo $OPENAI_API_KEY | head -c 10  # 'sk-...' 형태여야 함

# 직접 API 호출 테스트
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models | jq '.data[0].id'
```

### 3. 사용 가능한 프로바이더 확인

```python
from beanllm.providers.provider_factory import ProviderFactory
print(ProviderFactory.get_available_providers())
```

### 4. CircuitBreaker 상태 확인

```python
from beanllm.providers.provider_factory import ProviderFactory
provider = ProviderFactory.get_provider(provider_name="openai")
cb = provider._circuit_breaker
print(f"State: {cb.state}")          # CLOSED / OPEN / HALF_OPEN
print(f"Failures: {cb.failure_count}")
print(f"Opens at: {cb.last_failure_time}")
```

---

## 해결

### 즉각 조치 — Fallback 프로바이더 사용

```python
# 방법 1: Client에서 다른 프로바이더 명시
client = Client(model="claude-sonnet-4-6", provider="claude")

# 방법 2: ProviderFactory fallback
from beanllm.providers.provider_factory import ProviderFactory
provider = ProviderFactory.get_provider(fallback=True)  # 우선순위상 다음 사용 가능 프로바이더

# 방법 3: 환경변수로 우선순위 변경 (재시작 필요)
# OPENAI_API_KEY를 제거하면 ANTHROPIC_API_KEY가 1순위
unset OPENAI_API_KEY
```

### CircuitBreaker 수동 리셋

```python
# 주의: 원인 해결 후에만 리셋
from beanllm.providers.provider_factory import ProviderFactory
provider = ProviderFactory.get_provider(provider_name="openai")
provider._circuit_breaker.reset()  # CLOSED로 강제 복구
```

### API 키 갱신

```bash
# .env 업데이트
OPENAI_API_KEY=sk-new-key-here...

# 실행 중인 서비스 환경변수 갱신 후 재시작
```

### 60초 자동 복구 대기

CircuitBreaker는 60초 후 HALF_OPEN 상태로 전환됩니다.
HALF_OPEN에서 2회 성공하면 자동으로 CLOSED 복구됩니다.

---

## 예방

### CircuitBreaker 임계값 조정

```python
# 민감도를 낮추려면 (일시적 오류에 덜 반응)
config = {
    "circuit_breaker": {
        "failure_threshold": 10,  # 기본 5 → 10으로 상향
        "timeout": 30.0,          # OPEN 유지 시간 30초로 단축
    }
}
```

### 프로바이더 헬스체크 알림

```python
import asyncio
from beanllm.providers.provider_factory import ProviderFactory

async def health_check():
    available = ProviderFactory.get_available_providers()
    if "openai" not in available:
        # 알림 발송 (Slack, PagerDuty 등)
        send_alert("OpenAI provider unavailable")

# 주기적 실행 (예: 30초마다)
```

---

## 관련 문서

- [wiki/providers.md](../../wiki/providers.md#circuitbreaker) — CircuitBreaker 동작 원리
- [DL-003](../decision-log/DL-003-circuit-breaker-per-provider.md) — 프로바이더별 독립 인스턴스 결정
- [Playbook 02](02-rate-limit-exceeded.md) — Rate Limit 대응
