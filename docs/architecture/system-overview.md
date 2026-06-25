# beanllm 시스템 아키텍처 개요

beanllm은 8개 LLM 프로바이더를 단일 인터페이스로 통합하는 Python 라이브러리입니다.  
Clean Architecture를 적용하여 프로바이더 교체 시 도메인 코드 변경이 없고, HTTP 모킹 없이 80% 테스트 커버리지를 달성합니다.

---

## 레이어 아키텍처

```mermaid
graph TB
    subgraph User["사용자 코드"]
        UC["client = Client(model='gpt-4o')\nawait client.chat(messages)"]
    end

    subgraph Facade["Facade Layer (공개 API)"]
        F1["Client — 채팅, 스트리밍"]
        F2["RAGChain — 문서 기반 QA"]
        F3["Agent — ReAct 도구 호출"]
        F4["StateGraph — DAG 워크플로우"]
        FB["FacadeBase\n(DI Container 초기화)"]
        F1 & F2 & F3 & F4 --> FB
    end

    subgraph Handler["Handler Layer (검증·데코레이터)"]
        H1["CoreHandler — 기본 채팅 처리"]
        H2["AdvancedHandler — RAG·에이전트"]
        H3["MLHandler — OCR·비전·오디오"]
        HF["HandlerFactory"]
        HF --> H1 & H2 & H3
    end

    subgraph Service["Service Layer (비즈니스 로직)"]
        S1["ChatService"]
        S2["EmbeddingService"]
        S3["RAGService"]
        S4["AgentService"]
        SF["ServiceFactory"]
        SF --> S1 & S2 & S3 & S4
    end

    subgraph Domain["Domain Layer (핵심 엔티티·규칙)"]
        D1["LLMResponse\n(content, model, usage)"]
        D2["ModelParameterStrategy\n(temperature·max_tokens 지원 여부)"]
        D3["Document, Chunk, Embedding"]
        D4["AgentStep, AgentResult"]
    end

    subgraph Infra["Infrastructure Layer (외부 연동)"]
        I1["ProviderFactory\n(환경변수 기반 자동 선택)"]
        I2["BaseLLMProvider\n(CircuitBreaker · RateLimit)"]
        I3["OpenAIProvider"]
        I4["ClaudeProvider"]
        I5["GeminiProvider"]
        I6["GrokProvider"]
        I7["DeepSeekProvider"]
        I8["PerplexityProvider"]
        I9["OllamaProvider"]
        I1 --> I2
        I2 --> I3 & I4 & I5 & I6 & I7 & I8 & I9
    end

    UC --> Facade
    FB -->|"get_container()\nhandler_factory\nservice_factory"| Handler
    Handler -->|interfaces only| Service
    Service --> Domain
    Service --> Infra
```

---

## 요청 흐름 (Client.chat)

```mermaid
sequenceDiagram
    participant U as 사용자
    participant C as Client (Facade)
    participant H as CoreHandler
    participant S as ChatService
    participant PF as ProviderFactory
    participant P as BaseLLMProvider (구현체)
    participant API as LLM API (외부)

    U->>C: await client.chat(messages, model="gpt-4o")
    C->>H: handle_chat(messages, model, system, temperature)
    H->>H: 입력 검증 (messages 비어있으면 ValueError)
    H->>S: chat(messages, model, ...)
    S->>PF: get_provider("openai")
    PF->>PF: _instances 캐시 확인
    PF->>P: OpenAIProvider() (is_available 체크)
    S->>S: ModelParameterFactory.get_config(model)\n→ {supports_temperature, uses_max_completion_tokens}
    S->>P: await provider.chat(messages, model, ...)
    P->>P: _acquire_rate_limit()
    P->>P: _call_with_circuit_breaker(api_fn)
    P->>API: HTTP POST /v1/chat/completions
    API-->>P: ChatCompletion response
    P-->>S: LLMResponse(content, model, usage)
    S-->>H: LLMResponse
    H-->>C: LLMResponse
    C-->>U: LLMResponse
```

---

## 프로바이더 라우팅

```mermaid
graph LR
    PF["ProviderFactory.get_provider(name?)"]

    subgraph "지정 없으면 환경변수 우선순위"
        P1["OPENAI_API_KEY → OpenAIProvider"]
        P2["ANTHROPIC_API_KEY → ClaudeProvider"]
        P3["GEMINI_API_KEY → GeminiProvider"]
        P4["DEEPSEEK_API_KEY → DeepSeekProvider"]
        P5["PERPLEXITY_API_KEY → PerplexityProvider"]
        P6["XAI_API_KEY → GrokProvider"]
        P7["OLLAMA_HOST → OllamaProvider (선택적)"]
    end

    PF --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7

    note1["✓ 인스턴스 캐시 (_instances dict)\n✓ fallback=True: 실패 시 다음 프로바이더\n✓ 선택적 import (try/except): 미설치 패키지 무시"]
```

**모델 이름 → 프로바이더 자동 매핑:**
```python
client = Client(model="gpt-4o")         # → OpenAIProvider
client = Client(model="claude-opus-4-8") # → ClaudeProvider
client = Client(model="grok-4.3")        # → GrokProvider
```

---

## ModelParameterStrategy (Strategy 패턴)

각 모델 시리즈마다 `temperature`, `max_tokens` 파라미터 지원 여부가 다릅니다. 하드코딩 대신 Strategy 패턴으로 확장합니다.

```
ModelParameterFactory.get_config("gpt-5-nano")
  → extract_base_model("gpt-5-nano-2025-08-07") = "gpt-5-nano"
  → 우선순위 패턴 매칭: "gpt-5-nano" → NanoModelStrategy
  → {supports_temperature: False, supports_max_tokens: False}

ModelParameterFactory.get_config("gpt-4o-mini")
  → "mini" 패턴 매칭 → MiniModelStrategy
  → {supports_temperature: False, supports_max_tokens: True}

ModelParameterFactory.get_config("gpt-4o")
  → 매칭 없음 → DefaultModelStrategy
  → {supports_temperature: True, supports_max_tokens: True}
```

| 전략 | 적용 모델 | temperature | max_tokens | max_completion_tokens |
|------|---------|-------------|-----------|----------------------|
| `GPT5Strategy` | gpt-5 | ✓ | ✗ | ✓ |
| `GPT41Strategy` | gpt-4.1 | ✓ | ✗ | ✓ |
| `NanoModelStrategy` | *-nano | ✗ | ✗ | ✗ |
| `MiniModelStrategy` | *-mini | ✗ | ✓ | ✗ |
| `O3ModelStrategy` | o3 | ✗ | ✓ | ✗ |
| `DefaultModelStrategy` | 나머지 | ✓ | ✓ | ✗ |

---

## BaseLLMProvider 공통 기능

모든 프로바이더 구현체가 `BaseLLMProvider`를 상속합니다.

| 기능 | 구현 | 설명 |
|------|------|------|
| **Circuit Breaker** | `CircuitBreaker` (per-provider) | 연속 5회 실패 → OPEN (60초 차단) → HALF_OPEN → 2회 성공 → CLOSED |
| **Rate Limiting** | `_acquire_rate_limit()` | 분산(Redis) 또는 인메모리, 실패 시 경고만 하고 계속 진행 |
| **에러 통합** | `_handle_provider_error()` | API 키 마스킹 + `ProviderError` 래핑 |
| **OpenAI 메시지 변환** | `_prepare_openai_messages()` | system 프롬프트를 messages[0]으로 삽입 |
| **Health Check** | `_safe_health_check()` | 모든 예외 캐치 → False 반환 |

---

## 선택적 의존성 (Optional Extras) 구조

```
pip install beanllm              # 핵심만 (~5MB): httpx, pydantic, tiktoken, PyMuPDF
pip install beanllm[openai]      # + openai SDK
pip install beanllm[anthropic]   # + anthropic SDK
pip install beanllm[gemini]      # + google-generativeai
pip install beanllm[ml]          # + torch, marker-pdf (ML-based PDF)
pip install beanllm[ragpro]      # + semantic + colbert + DB drivers
pip install beanllm[all]         # 전체 (모든 프로바이더 + CLI + MCP)
```

프로바이더 SDK는 `try/except`로 선택적 import — 미설치 시 `WARNING` 로그만 남기고 해당 프로바이더만 비활성화합니다.

---

## 테스트 전략

| 레이어 | 테스트 방식 | HTTP 모킹 필요 |
|--------|-----------|--------------|
| Domain (LLMResponse, Strategy) | 단위 테스트 | ✗ |
| Service (ChatService, RAGService) | 단위 테스트 + MockProvider | ✗ |
| Handler | 단위 테스트 | ✗ |
| Provider (OpenAI 등) | `pytest-mock` + httpx mock | ✓ (Provider 레이어만) |
| Facade (통합) | MockProvider 주입 | ✗ |

Clean Architecture 덕분에 **HTTP 모킹은 Provider 레이어에만 집중** → 전체 6,340개 테스트 중 대다수가 실제 API 호출 없이 실행됩니다.
