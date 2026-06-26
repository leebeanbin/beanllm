# Providers

beanllm은 8개 LLM 프로바이더를 동일한 인터페이스로 통합합니다.

---

## 프로바이더 목록

| 프로바이더 | 설치 extra | 환경변수 | 주요 모델 |
|-----------|-----------|---------|---------|
| OpenAI | `beanllm[openai]` | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, o1 |
| Claude (Anthropic) | `beanllm[anthropic]` | `ANTHROPIC_API_KEY` | claude-sonnet-4-6, claude-opus-4 |
| Gemini (Google) | `beanllm[gemini]` | `GEMINI_API_KEY` | gemini-2.0-flash, gemini-1.5-pro |
| Grok (xAI) | `beanllm[all]` | `XAI_API_KEY` | grok-3, grok-3-mini |
| DeepSeek | `beanllm[all]` | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-reasoner |
| Perplexity | `beanllm[all]` | `PERPLEXITY_API_KEY` | llama-3.1-sonar-large-128k-online |
| Ollama (로컬) | `beanllm` (기본 포함) | `OLLAMA_HOST` (선택) | llama3, mistral, qwen2.5, ... |
| HuggingFace | `beanllm[all]` | `HUGGINGFACE_API_KEY` | Inference API 모델 |

---

## 설치

```bash
# 핵심만 (~5MB, Ollama 포함)
pip install beanllm

# 특정 클라우드 프로바이더 추가
pip install beanllm[openai]
pip install beanllm[anthropic]
pip install beanllm[gemini]

# ML 기능 (OCR, 음성)
pip install beanllm[ml]

# 모든 기능
pip install beanllm[all]
```

---

## 프로바이더 자동 선택

환경변수가 설정된 첫 번째 프로바이더가 자동으로 선택됩니다.

**우선순위 (앞에서부터):**

1. `OPENAI_API_KEY` → OpenAI
2. `ANTHROPIC_API_KEY` → Claude
3. `GEMINI_API_KEY` → Gemini
4. `DEEPSEEK_API_KEY` → DeepSeek
5. `PERPLEXITY_API_KEY` → Perplexity
6. `XAI_API_KEY` → Grok
7. `OLLAMA_HOST` (또는 localhost:11434 기본값) → Ollama

```python
# provider 미지정 시 환경변수 우선순위로 자동 선택
client = Client(model="gpt-4o-mini")

# 명시적 지정
client = Client(model="claude-sonnet-4-6", provider="claude")
```

모델 ID에서도 프로바이더를 추론합니다 (`gpt-` → OpenAI, `claude-` → Anthropic 등).

---

## BaseLLMProvider 공통 기능

모든 프로바이더는 `BaseLLMProvider`를 상속하며 다음 기능을 공통으로 가집니다.

### CircuitBreaker

연속 장애 시 일정 시간 동안 해당 프로바이더 호출을 차단합니다.

```
CLOSED → (5회 연속 실패) → OPEN (60초)
OPEN   → (60초 경과)     → HALF_OPEN
HALF_OPEN → (2회 성공)   → CLOSED
HALF_OPEN → (실패)       → OPEN (60초 리셋)
```

설정은 config dict로 주입 가능합니다:

```python
config = {
    "circuit_breaker": {
        "failure_threshold": 5,   # OPEN 전환 연속 실패 횟수
        "success_threshold": 2,   # CLOSED 복구 성공 횟수
        "timeout": 60.0,          # OPEN 유지 시간 (초)
    }
}
```

Circuit이 OPEN이면 `CircuitBreakerError`를 발생시킵니다. `fallback=True`로 `ProviderFactory.get_provider()`를 호출하면 다음 우선순위 프로바이더로 자동 전환됩니다.

### Rate Limiting

분산 환경(Redis)과 인메모리 두 가지 모드를 지원합니다.

```bash
# Redis rate limiter 활성화
REDIS_URL=redis://localhost:6379/0
```

Redis 미설정 시 인메모리 토큰 버킷으로 폴백합니다.

### 에러 처리

모든 프로바이더의 HTTP 오류는 공통 `ProviderError`로 정규화됩니다.

```python
from beanllm.utils.exceptions import ProviderError, CircuitBreakerError, RateLimitError

try:
    response = await client.chat(messages)
except CircuitBreakerError:
    # 프로바이더 일시 차단 — fallback 고려
    ...
except RateLimitError:
    # 요청 한도 초과 — 대기 후 재시도
    ...
except ProviderError as e:
    # 기타 API 오류
    ...
```

---

## ModelParameterStrategy

모델별로 지원되는 파라미터가 다릅니다. `ModelParameterStrategy`가 이를 추상화합니다.

| 모델 | temperature | max_tokens | top_p | stream |
|------|-------------|-----------|-------|--------|
| gpt-4o | O | O | O | O |
| gpt-4o-mini | O | O | O | O |
| o1, o1-mini | X (고정) | O | X | X |
| claude-sonnet-4-6 | O | O | O | O |
| gemini-2.0-flash | O | O | O | O |
| deepseek-reasoner | X | O | X | O |
| ollama/* | O | O | O | O |

`ChatService`는 `ModelParameterStrategy`를 통해 지원하지 않는 파라미터를 자동으로 제거한 뒤 API를 호출합니다.

---

## 개별 프로바이더 상세

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

지원 모델: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini`, `o3-mini`

임베딩: `text-embedding-3-small`, `text-embedding-3-large`

### Claude (Anthropic)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

지원 모델: `claude-opus-4-5`, `claude-sonnet-4-6`, `claude-haiku-3-5`

200k 컨텍스트 지원. 시스템 프롬프트는 `system` 파라미터로 전달.

### Gemini (Google)

```bash
export GEMINI_API_KEY="AIza..."
```

지원 모델: `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`

최대 1M 컨텍스트 (gemini-1.5-pro).

### Grok (xAI)

```bash
export XAI_API_KEY="xai-..."
```

지원 모델: `grok-3`, `grok-3-mini`, `grok-beta`

### DeepSeek

```bash
export DEEPSEEK_API_KEY="sk-..."
```

지원 모델: `deepseek-chat`, `deepseek-reasoner` (R1)

`deepseek-reasoner`는 reasoning 토큰을 반환합니다.

### Perplexity

```bash
export PERPLEXITY_API_KEY="pplx-..."
```

지원 모델: `llama-3.1-sonar-large-128k-online`, `llama-3.1-sonar-small-128k-online`

웹 검색 기반 실시간 답변을 지원합니다.

### Ollama (로컬)

별도 API 키 없음. Ollama 데몬이 실행 중이어야 합니다.

```bash
# Ollama 설치 후 모델 pull
ollama pull llama3
ollama pull mistral
ollama pull qwen2.5

export OLLAMA_HOST="http://localhost:11434"  # 기본값
```

지원 모델: Ollama 레지스트리의 모든 모델.

### HuggingFace

```bash
export HUGGINGFACE_API_KEY="hf_..."
```

Inference API를 통해 HuggingFace Hub의 모델을 사용합니다.

---

## Fallback 동작

```python
from beanllm.providers.provider_factory import ProviderFactory

# fallback=True: 현재 provider 실패 시 다음 available provider 자동 전환
provider = ProviderFactory.get_provider(provider_name="openai", fallback=True)

# 사용 가능한 provider 목록 확인
available = ProviderFactory.get_available_providers()
# ['openai', 'claude', 'ollama'] 등
```

---

## 관련 문서

- [Decision Log DL-003](../docs/decision-log/DL-003-circuit-breaker-per-provider.md) — 프로바이더별 독립 CircuitBreaker 결정 이유
- [Playbook: Circuit OPEN](../docs/playbooks/01-provider-circuit-open.md) — CircuitBreaker OPEN 대응
- [Playbook: Rate Limit](../docs/playbooks/02-rate-limit-exceeded.md) — Rate Limit 대응
- [ADR-002](../docs/adr/ADR-002-unified-provider-interface.md) — 통합 프로바이더 인터페이스 결정
