# ADR-001: Clean Architecture 채택 (Facade → Handler → Service → Domain → Infrastructure)

* **Status:** Accepted
* **Date:** 2024-09
* **Author:** leebeanbin

## Context & Problem Statement

LLM 라이브러리를 설계할 때 두 가지 주요 선행 사례가 있었다:

- **LangChain**: Flat chain 방식 — `LLMChain`, `SequentialChain` 등 체인을 중첩. 내부 의존성이 뒤섞여 특정 컴포넌트만 교체하거나 테스트하기 어렵다.
- **LlamaIndex**: Index 중심 — 데이터 인덱싱에 최적화되었지만 범용 채팅·에이전트 워크플로우에는 장황하다.

공통 문제: **프로바이더 교체 시 사용자 코드 변경이 필요하고, HTTP 없이 비즈니스 로직을 단위 테스트할 수 없다.**

## Decision Drivers

* 프로바이더(OpenAI → Claude 등) 교체 시 사용자 코드 변경 없어야 함
* 각 레이어를 HTTP 모킹 없이 독립 단위 테스트 가능해야 함
* 8개 프로바이더가 동일한 추상화를 공유해야 함 (코드 중복 최소화)
* Facade만 노출 — 사용자는 내부 레이어 구조를 몰라도 됨

## Considered Options

1. **Flat chain (LangChain 스타일)** — 체인 객체가 모든 책임
2. **Index-centric (LlamaIndex 스타일)** — 데이터 인덱스 중심 설계
3. **Clean Architecture (5-레이어)** ← 선택

## Decision Outcome

Chosen Option: **Option 3**.

```
Facade   (Client, RAGChain, Agent)
  ↓ DI Container (get_container())
Handler  (CoreHandler, AdvancedHandler, MLHandler)
  ↓ interfaces only
Service  (ChatService, EmbeddingService, RAGService, AgentService)
  ↓
Domain   (LLMResponse, ModelParameterStrategy, Document)
  ↓
Infrastructure (ProviderFactory, BaseLLMProvider, 8 구현체)
```

**의존성 방향 규칙**: 외부 레이어 → 내부 레이어. Infrastructure가 Domain을 알지만 Domain은 Infrastructure를 모른다.

**DI Container** (`get_container()`)가 Facade와 Handler 사이에서 `HandlerFactory`와 `ServiceFactory`를 주입한다. 테스트에서는 MockProvider를 주입해 실제 HTTP 없이 Service/Handler 레이어 전체를 검증한다.

### Consequences

* **Positive:**
  - Provider 교체 = Infrastructure 레이어만 수정, 상위 레이어 불변
  - Domain + Service + Handler를 MockProvider로 단위 테스트 → 전체 6,340 테스트 중 대다수가 HTTP 없이 실행
  - 새 프로바이더 추가 = `BaseLLMProvider` 구현 + `ProviderFactory` 등록만
  - `FacadeBase`가 DI 초기화 공통 로직 담당 → 새 Facade 추가 시 `_init_handlers()`만 오버라이드
* **Negative/Trade-offs:**
  - 레이어당 Factory 클래스 필요 → 초기 보일러플레이트 증가
  - 단순 "모델 호출" 유스케이스에도 5개 레이어를 통과 → 간단한 사용 예제가 장황해 보일 수 있음 (Facade API로 은닉)
  - DI Container 개념 학습 필요

---

## Options Comparison Matrix

| Criteria | Flat chain | Index-centric | Clean Architecture |
|---|---|---|---|
| **Provider 교체 용이성** | ❌ 체인 내부 수정 | ❌ 인덱스 재구성 | ✅ Infrastructure만 |
| **단위 테스트 격리** | ❌ 어려움 | ❌ 어려움 | ✅ 레이어별 격리 |
| **RAG·에이전트 확장** | ⚠️ 체인 중첩 | ✅ | ✅ Service 추가 |
| **학습 곡선** | 낮음 | 중간 | 중간 |
| **초기 보일러플레이트** | 낮음 | 중간 | 높음 |
