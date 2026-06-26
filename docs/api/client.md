# Client

`beanllm.Client` — 통합 LLM 채팅 클라이언트 (Facade 패턴)

## Import

```python
from beanllm import Client
```

---

## `__init__`

```python
Client(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
)
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model` | `str` | — (필수) | 모델 ID. 예: `"gpt-4o-mini"`, `"claude-sonnet-4-6"` |
| `provider` | `str \| None` | `None` | 프로바이더 이름. `None`이면 모델명으로 자동 감지 |
| `api_key` | `str \| None` | `None` | API 키. `None`이면 환경변수에서 로드 |
| `**kwargs` | `Any` | — | 프로바이더별 추가 설정 |

**프로바이더 이름 허용값:** `"openai"`, `"claude"` (또는 `"anthropic"`), `"gemini"` (또는 `"google"`), `"deepseek"`, `"perplexity"`, `"grok"`, `"ollama"`, `"huggingface"`

---

## `chat`

```python
async def chat(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs: Any,
) -> ChatResponse
```

비스트리밍 채팅 완료.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `messages` | `List[Dict[str, str]]` | — (필수) | 메시지 목록. `[{"role": "user", "content": "..."}]` |
| `system` | `str \| None` | `None` | 시스템 프롬프트 |
| `temperature` | `float \| None` | `None` (프로바이더 기본값) | 생성 다양성. `0.0`~`2.0`. 모델별 지원 여부 상이 |
| `max_tokens` | `int \| None` | `None` | 최대 생성 토큰 수 |
| `top_p` | `float \| None` | `None` | Top-p 누클리어스 샘플링 |

**반환:** `ChatResponse`

```python
response.content    # str — 생성된 텍스트
response.model      # str — 실제 사용된 모델 ID
response.usage      # dict | None — {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
```

**예시:**

```python
import asyncio
from beanllm import Client

async def main():
    client = Client(model="gpt-4o-mini")
    response = await client.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    print(response.content)   # "4"
    print(response.usage)     # {"total_tokens": 20, ...}

asyncio.run(main())
```

---

## `stream_chat`

```python
async def stream_chat(
    messages: List[Dict[str, str]],
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    **kwargs: Any,
) -> AsyncIterator[str]
```

스트리밍 채팅. 토큰이 생성될 때마다 문자열 청크를 yield합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `messages` | `List[Dict[str, str]]` | — (필수) | `chat()`과 동일 |
| `system` | `str \| None` | `None` | 시스템 프롬프트 |
| `temperature` | `float \| None` | `None` | 온도 |
| `max_tokens` | `int \| None` | `None` | 최대 토큰 수 |
| `top_p` | `float \| None` | `None` | Top-p |

**반환:** `AsyncIterator[str]` — 텍스트 청크

**예시:**

```python
async def stream_demo():
    client = Client(model="claude-sonnet-4-6", provider="claude")
    async for chunk in client.stream_chat(
        messages=[{"role": "user", "content": "Tell me a story."}],
        temperature=0.8,
        max_tokens=256,
    ):
        print(chunk, end="", flush=True)
    print()  # newline
```

---

## `_detect_provider` (내부)

모델 ID로 프로바이더를 자동 감지합니다.

우선순위:
1. `ModelRegistry`에서 모델 조회 (가장 정확)
2. `provider_registry`의 패턴 매칭

---

## 멀티턴 대화

```python
async def multiturn():
    client = Client(model="gpt-4o-mini")
    history = []

    # 1번째 발화
    history.append({"role": "user", "content": "My name is Alice."})
    resp = await client.chat(history)
    history.append({"role": "assistant", "content": resp.content})

    # 2번째 발화 (컨텍스트 유지)
    history.append({"role": "user", "content": "What is my name?"})
    resp = await client.chat(history)
    print(resp.content)  # "Alice"
```

---

## 오류 처리

```python
from beanllm.utils.exceptions import ProviderError, CircuitBreakerError, RateLimitError

try:
    response = await client.chat(messages)
except CircuitBreakerError:
    # 해당 프로바이더가 일시 차단됨 (60초). fallback 프로바이더로 전환 고려.
    ...
except RateLimitError:
    # 요청 한도 초과. 잠시 후 재시도.
    import asyncio
    await asyncio.sleep(10)
    response = await client.chat(messages)
except ProviderError as e:
    # API 오류 (400, 401, 500 등)
    print(f"Provider error: {e}")
```

---

## 관련 문서

- [wiki/facade.md](../../wiki/facade.md) — 고수준 사용 가이드
- [wiki/providers.md](../../wiki/providers.md) — 프로바이더별 환경변수·모델 목록
- [models.md](models.md) — 지원 모델 전체 목록
