# ADR-002: 통합 프로바이더 인터페이스 + ModelParameterStrategy

* **Status:** Accepted
* **Date:** 2024-10
* **Author:** leebeanbin

## Context & Problem Statement

8개 LLM API는 제각각 다른 파라미터 이름과 동작 방식을 가진다:

- OpenAI GPT-4o: `max_tokens` 사용, `temperature` 지원
- OpenAI GPT-5, GPT-4.1: `max_tokens` 미지원 → `max_completion_tokens` 사용
- OpenAI o3/o4-mini: `temperature` 미지원 (항상 1)
- Anthropic Claude: `max_tokens` 필수, messages 형식이 OpenAI와 다름
- Ollama: API 키 없음, `OLLAMA_HOST` 환경변수로 접근
- Grok (xAI): `XAI_API_KEY`, OpenAI-compatible API

각 프로바이더를 직접 호출하면 사용자가 프로바이더마다 다른 파라미터를 기억해야 하고, 프로바이더 교체 시 코드 전체를 수정해야 한다.

## Decision Drivers

* 사용자는 `Client(model="any-model")` 하나로 어떤 프로바이더든 동일하게 호출해야 함
* 모델별 파라미터 차이를 라이브러리가 내부에서 처리해야 함 (사용자 코드 불변)
* 새 모델 추가 시 기존 코드 수정 없이 Strategy만 추가하면 됨 (OCP)
* 프로바이더 SDK가 미설치 상태여도 라이브러리가 동작해야 함 (선택적 의존성)

## Considered Options

1. **직접 SDK 호출** — 각 프로바이더 SDK를 직접 사용, 분기문으로 처리
2. **단일 HTTP 어댑터** — 모든 프로바이더를 httpx로 직접 호출
3. **`BaseLLMProvider` 추상화 + `ModelParameterStrategy`** ← 선택

## Decision Outcome

Chosen Option: **Option 3**.

**`BaseLLMProvider` (ABC)**가 모든 프로바이더의 공통 계약을 정의한다:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages, model, system, temperature, max_tokens) -> LLMResponse: ...
    @abstractmethod
    def stream_chat(self, messages, model, ...) -> AsyncGenerator[str, None]: ...
    @abstractmethod
    async def list_models(self) -> List[str]: ...
    @abstractmethod
    def is_available(self) -> bool: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
```

**`ModelParameterStrategy`**가 모델별 파라미터 지원 여부를 캡슐화한다:

```python
# 사용자 코드 — 변경 없음
client = Client(model="gpt-4o")
client = Client(model="gpt-5")
client = Client(model="o3")

# 내부: ModelParameterFactory가 자동으로 올바른 파라미터 선택
config = ModelParameterFactory.get_config("gpt-5")
# → {supports_temperature: True, uses_max_completion_tokens: True}
```

**`ProviderFactory`** 환경변수 우선순위로 자동 선택 + 인스턴스 캐시:
```
OPENAI_API_KEY → OpenAIProvider
ANTHROPIC_API_KEY → ClaudeProvider
GEMINI_API_KEY → GeminiProvider
...
fallback=True → 실패 시 다음 프로바이더 시도
```

**선택적 import** — SDK 미설치 시 `WARNING`만 남기고 해당 프로바이더만 비활성화:
```python
try:
    from .claude_provider import ClaudeProvider
except Exception as e:
    logger.warning(f"Failed to import ClaudeProvider: {e}")
    ClaudeProvider = None
```

### Consequences

* **Positive:**
  - 사용자는 `Client(model="gpt-4o")` → `Client(model="claude-opus-4-8")` 한 줄 변경으로 프로바이더 교체
  - 새 모델 파라미터 지원 추가 = `ModelParameterStrategy` 서브클래스 하나만 추가 (OCP)
  - `BaseLLMProvider` 공통 메서드(`_handle_provider_error`, `_prepare_openai_messages`, `_extract_openai_usage`)로 프로바이더 구현 중복 최소화
  - 모든 프로바이더에 Circuit Breaker + Rate Limiting 자동 적용
* **Negative/Trade-offs:**
  - 각 프로바이더의 고유 기능(Claude의 `thinking_budget`, Grok의 real-time 검색 등)은 `**kwargs`로 전달 — 공통 인터페이스에 없는 기능은 타입 힌트 미지원
  - `ModelParameterStrategy` STRATEGIES 리스트를 순서대로 매칭 → 새 모델 패턴이 기존 패턴과 충돌 가능 (우선순위 관리 필요)
  - `ProviderFactory._instances`는 클래스 레벨 딕셔너리 → 멀티스레드 환경에서 동시 초기화 시 race condition 가능 (현재 async 단일 루프 가정)

---

## Options Comparison Matrix

| Criteria | 직접 SDK 호출 | 단일 HTTP 어댑터 | BaseLLMProvider + Strategy |
|---|---|---|---|
| **프로바이더 교체** | ❌ 코드 수정 | ⚠️ HTTP 파라미터 직접 관리 | ✅ model= 변경만 |
| **새 모델 추가** | ❌ 분기문 추가 | ❌ URL/파라미터 추가 | ✅ Strategy 하나 추가 |
| **타입 안전성** | ✅ SDK 타입 | ❌ dict 기반 | ✅ LLMResponse 타입 |
| **SDK 업데이트 영향** | 높음 | ❌ 없음 (직접 HTTP) | ✅ Provider 클래스만 |
| **Circuit Breaker** | ❌ 직접 구현 | ❌ 직접 구현 | ✅ BaseLLMProvider 내장 |
