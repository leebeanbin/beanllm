# 🏗️ llmkit 아키텍처 가이드

## 📋 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [레이어 구조](#레이어-구조)
3. [디렉토리 구조](#디렉토리-구조)
4. [의존성 방향](#의존성-방향)
5. [설계 원칙](#설계-원칙)
6. [주요 패턴](#주요-패턴)
7. [데이터 흐름](#데이터-흐름)

---

## 아키텍처 개요

llmkit은 **Domain-Driven Design (DDD)**과 **Clean Architecture** 원칙을 따르는 계층형 아키텍처를 사용합니다.

### 핵심 원칙

1. **책임 분리 (Separation of Concerns)**
   - 각 레이어는 명확한 책임을 가집니다
   - Handler → Service → Domain → Infrastructure

2. **의존성 역전 (Dependency Inversion)**
   - 상위 레이어가 하위 레이어의 인터페이스에 의존
   - 구체적인 구현은 하위 레이어에 위치

3. **단일 책임 원칙 (Single Responsibility)**
   - 각 클래스는 하나의 책임만 가집니다
   - Handler: 입력 검증 및 에러 처리
   - Service: 비즈니스 로직
   - Domain: 핵심 비즈니스 규칙

---

## 레이어 구조

```
┌─────────────────────────────────────────────────────────┐
│                    Facade Layer                          │
│  (사용자 친화적 API) - 기존 API 유지                     │
│  - Client, RAGChain, Agent, Graph 등                    │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                    Handler Layer                          │
│  (Controller 역할) - 입력 검증, 에러 처리                  │
│  - ChatHandler, RAGHandler, AgentHandler 등              │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                    Service Layer                          │
│  (비즈니스 로직) - 핵심 로직만 포함                       │
│  - IChatService, IRAGService, IAgentService              │
│  - ChatServiceImpl, RAGServiceImpl 등                    │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                    Domain Layer                           │
│  (핵심 비즈니스) - 엔티티, 인터페이스, 규칙              │
│  - Document, Embedding, VectorStore, Graph 등            │
└──────────────────────┬────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────┐
│                Infrastructure Layer                       │
│  (외부 시스템) - Provider, Vector Store 구현              │
│  - OpenAIProvider, ChromaVectorStore 등                  │
└───────────────────────────────────────────────────────────┘
```

---

## 디렉토리 구조

### 전체 구조

```
src/llmkit/
├── __init__.py              # Public API (통합 export)
│
├── facade/                  # Facade Layer
│   ├── __init__.py
│   ├── client_facade.py     # Client (기존 API 유지)
│   ├── rag_facade.py        # RAGChain (기존 API 유지)
│   ├── agent_facade.py      # Agent (기존 API 유지)
│   ├── graph_facade.py      # Graph (기존 API 유지)
│   └── ...
│
├── handler/                  # Handler Layer (Controller)
│   ├── __init__.py
│   ├── chat_handler.py      # ChatHandler
│   ├── rag_handler.py       # RAGHandler
│   ├── agent_handler.py     # AgentHandler
│   ├── graph_handler.py     # GraphHandler
│   └── factory.py           # HandlerFactory
│
├── service/                  # Service Layer
│   ├── __init__.py
│   ├── chat_service.py      # IChatService (인터페이스)
│   ├── rag_service.py       # IRAGService (인터페이스)
│   ├── agent_service.py     # IAgentService (인터페이스)
│   ├── factory.py           # ServiceFactory
│   └── impl/                # Service 구현체
│       ├── __init__.py
│       ├── chat_service_impl.py
│       ├── rag_service_impl.py
│       └── agent_service_impl.py
│
├── dto/                      # Data Transfer Objects
│   ├── __init__.py
│   ├── request/             # 요청 DTO
│   │   ├── __init__.py
│   │   ├── chat_request.py
│   │   ├── rag_request.py
│   │   └── agent_request.py
│   └── response/            # 응답 DTO
│       ├── __init__.py
│       ├── chat_response.py
│       ├── rag_response.py
│       └── agent_response.py
│
├── domain/                   # Domain Layer (핵심 비즈니스)
│   ├── __init__.py          # 모든 domain 모듈 export
│   │
│   ├── loaders/              # Document Loaders
│   │   ├── __init__.py
│   │   ├── base.py          # BaseDocumentLoader
│   │   ├── types.py          # Document
│   │   ├── loaders.py        # PDFLoader, CSVLoader 등
│   │   └── factory.py       # DocumentLoader
│   │
│   ├── embeddings/           # Embeddings
│   │   ├── __init__.py
│   │   ├── base.py          # BaseEmbedding
│   │   ├── providers.py     # OpenAIEmbedding, GeminiEmbedding 등
│   │   ├── factory.py       # Embedding
│   │   ├── cache.py         # EmbeddingCache
│   │   └── advanced.py      # MMR, Query Expansion 등
│   │
│   ├── splitters/            # Text Splitters
│   │   ├── __init__.py
│   │   ├── base.py          # BaseTextSplitter
│   │   ├── splitters.py     # RecursiveCharacterTextSplitter 등
│   │   └── factory.py       # TextSplitter
│   │
│   ├── vector_stores/        # Vector Stores
│   │   ├── __init__.py
│   │   ├── base.py          # BaseVectorStore
│   │   └── implementations.py  # ChromaVectorStore, FAISSVectorStore 등
│   │
│   ├── tools/                # Tools & Agents
│   │   ├── __init__.py
│   │   ├── tool.py          # Tool, ToolParameter
│   │   ├── tool_registry.py # ToolRegistry
│   │   ├── default_tools.py # calculator, search_web 등
│   │   └── advanced/        # Advanced Tools
│   │
│   ├── memory/               # Memory Systems
│   │   ├── __init__.py
│   │   ├── base.py          # BaseMemory
│   │   └── implementations.py  # BufferMemory, WindowMemory 등
│   │
│   ├── graph/                # Graph Workflows
│   │   ├── __init__.py
│   │   ├── base_node.py     # BaseNode
│   │   ├── graph_state.py   # GraphState
│   │   ├── node_cache.py    # NodeCache
│   │   └── nodes.py         # AgentNode, LLMNode 등
│   │
│   ├── multi_agent/          # Multi-Agent Systems
│   │   ├── __init__.py
│   │   ├── communication.py  # CommunicationBus
│   │   └── strategies.py    # SequentialStrategy, ParallelStrategy 등
│   │
│   ├── state_graph/          # State Graph
│   │   ├── __init__.py
│   │   ├── checkpoint.py    # Checkpoint
│   │   └── execution.py     # GraphExecution
│   │
│   ├── vision/               # Vision RAG
│   │   ├── __init__.py
│   │   ├── embeddings.py    # CLIPEmbedding, MultimodalEmbedding
│   │   └── loaders.py       # ImageLoader, PDFWithImagesLoader
│   │
│   ├── web_search/           # Web Search
│   │   ├── __init__.py
│   │   ├── engines.py       # GoogleSearch, BingSearch 등
│   │   └── scraper.py       # WebScraper
│   │
│   ├── evaluation/           # Evaluation
│   │   ├── __init__.py
│   │   ├── base_metric.py   # BaseMetric
│   │   ├── metrics.py       # BLEUMetric, ROUGEMetric 등
│   │   └── evaluator.py    # Evaluator
│   │
│   ├── finetuning/           # Fine-tuning
│   │   ├── __init__.py
│   │   ├── types.py        # FineTuningConfig, FineTuningJob
│   │   └── providers.py    # OpenAIFineTuningProvider
│   │
│   ├── audio/                # Audio Processing
│   │   ├── __init__.py
│   │   ├── types.py        # AudioSegment, TranscriptionResult
│   │   └── providers.py    # TTSProvider, WhisperModel
│   │
│   ├── parsers/              # Output Parsers
│   │   ├── __init__.py
│   │   ├── base.py         # BaseOutputParser
│   │   └── parsers.py      # JSONOutputParser, PydanticOutputParser 등
│   │
│   └── prompts/              # Prompt Templates
│       ├── __init__.py
│       ├── base.py         # BasePromptTemplate
│       └── templates.py   # PromptTemplate, ChatPromptTemplate 등
│
├── infrastructure/           # Infrastructure Layer
│   ├── __init__.py          # 모든 infrastructure 모듈 export
│   │
│   ├── adapter/              # Parameter Adapter
│   │   ├── __init__.py
│   │   └── parameter_adapter.py  # ParameterAdapter
│   │
│   ├── registry/             # Model Registry
│   │   ├── __init__.py
│   │   └── model_registry.py  # ModelRegistry
│   │
│   ├── provider/             # Provider Factory
│   │   ├── __init__.py
│   │   └── provider_factory.py  # ProviderFactory
│   │
│   ├── models/               # Model Definitions
│   │   ├── __init__.py
│   │   └── models.py       # MODELS, ModelCapabilityInfo 등
│   │
│   ├── hybrid/               # Hybrid Model Manager
│   │   ├── __init__.py
│   │   └── hybrid_manager.py  # HybridModelManager
│   │
│   ├── inferrer/             # Metadata Inferrer
│   │   ├── __init__.py
│   │   └── metadata_inferrer.py  # MetadataInferrer
│   │
│   ├── scanner/              # Model Scanner
│   │   ├── __init__.py
│   │   └── model_scanner.py  # ModelScanner
│   │
│   └── ml/                   # ML Models
│       ├── __init__.py
│       └── ml_models.py    # BaseMLModel, PyTorchModel 등
│
├── utils/                    # Utilities
│   ├── __init__.py          # 모든 utils 모듈 export
│   │
│   ├── config.py            # Config, EnvConfig
│   ├── error_handling.py    # ErrorHandler, CircuitBreaker 등
│   ├── streaming.py         # Streaming utilities
│   ├── token_counter.py     # Token counting
│   ├── tracer.py            # Tracing
│   ├── callbacks.py        # Callbacks
│   ├── logger.py           # Logger
│   ├── retry.py            # Retry decorator
│   ├── exceptions.py       # Custom exceptions
│   ├── cli/                # CLI utilities
│   └── rag_debug/          # RAG debugging tools
│
├── _source_providers/        # LLM Providers (외부 시스템)
│   ├── __init__.py
│   ├── base_provider.py     # BaseLLMProvider
│   ├── openai_provider.py   # OpenAIProvider
│   ├── claude_provider.py   # ClaudeProvider
│   ├── gemini_provider.py   # GeminiProvider
│   ├── ollama_provider.py   # OllamaProvider
│   └── provider_factory.py  # ProviderFactory
│
└── decorators/               # Decorators
    ├── __init__.py
    ├── logger.py           # Logging decorators
    ├── error_handler.py    # Error handling decorators
    └── validation.py       # Validation decorators
```

---

## 의존성 방향

### 원칙

1. **의존성은 항상 안쪽으로** (Dependency Rule)
   - Facade → Handler → Service → Domain ← Infrastructure
   - Domain은 어떤 레이어에도 의존하지 않음

2. **인터페이스에 의존**
   - Service는 인터페이스(IChatService)에 의존
   - 구현체(ChatServiceImpl)는 Infrastructure에 위치

3. **의존성 주입 (Dependency Injection)**
   - Factory 패턴으로 의존성 관리
   - 테스트 시 Mock 객체 주입 가능

### 의존성 다이어그램

```
Facade Layer
    ↓ (의존)
Handler Layer
    ↓ (의존)
Service Layer (인터페이스)
    ↓ (의존)
Domain Layer ← Infrastructure Layer (구현체)
```

---

## 설계 원칙

### SOLID 원칙

#### 1. Single Responsibility Principle (SRP)
- **Handler**: 입력 검증, 에러 처리만
- **Service**: 비즈니스 로직만
- **Domain**: 핵심 비즈니스 규칙만

#### 2. Open/Closed Principle (OCP)
- 새로운 Provider 추가 시 기존 코드 수정 불필요
- Strategy 패턴으로 확장 가능

#### 3. Liskov Substitution Principle (LSP)
- 인터페이스 구현으로 대체 가능
- 모든 Provider는 BaseLLMProvider를 구현

#### 4. Interface Segregation Principle (ISP)
- 작은, 특화된 인터페이스
- IChatService, IRAGService 등 분리

#### 5. Dependency Inversion Principle (DIP)
- 상위 레이어가 하위 레이어의 인터페이스에 의존
- Factory 패턴으로 의존성 주입

### Design Patterns

#### 1. Facade Pattern
- `Client`, `RAGChain`, `Agent` 등
- 복잡한 내부 구조를 단순한 API로 제공

#### 2. Factory Pattern
- `ServiceFactory`, `HandlerFactory`
- 의존성 주입 및 객체 생성 관리

#### 3. Strategy Pattern
- 검색 전략 (similarity, mmr, hybrid)
- Coordination 전략 (sequential, parallel, hierarchical)

#### 4. Adapter Pattern
- `ParameterAdapter`: Provider 간 파라미터 변환
- `SourceProviderFactoryAdapter`: ProviderFactory 어댑터

#### 5. Decorator Pattern
- `@log_handler_call`, `@handle_errors`, `@validate_input`
- 공통 기능을 데코레이터로 추출

---

## 데이터 흐름

### 예시: Chat 요청 처리

```
1. 사용자 호출
   ↓
   from llmkit import Client
   client = Client(model="gpt-4o")
   response = client.chat("Hello")

2. Facade Layer (client_facade.py)
   ↓
   - 기존 API 유지
   - 내부적으로 Handler 호출

3. Handler Layer (chat_handler.py)
   ↓
   - 입력 검증 (@validate_input)
   - DTO 변환 (ChatRequest 생성)
   - 에러 처리 (@handle_errors)
   - Service 호출

4. Service Layer (chat_service_impl.py)
   ↓
   - 비즈니스 로직 실행
   - Provider 생성 (ProviderFactory)
   - 파라미터 변환 (ParameterAdapter)
   - LLM 호출

5. Infrastructure Layer
   ↓
   - OpenAIProvider.chat() 호출
   - 실제 API 요청

6. 응답 반환
   ↓
   Service → Handler → Facade → 사용자
   ChatResponse 반환
```

### 예시: RAG 요청 처리

```
1. 사용자 호출
   ↓
   rag = RAGChain.from_documents("docs/")
   answer = rag.query("What is this about?")

2. Facade Layer (rag_facade.py)
   ↓
   - 문서 로딩 (Domain.loaders)
   - 임베딩 생성 (Domain.embeddings)
   - 벡터 스토어 생성 (Domain.vector_stores)
   - Handler 호출

3. Handler Layer (rag_handler.py)
   ↓
   - 입력 검증
   - DTO 변환 (RAGRequest)
   - Service 호출

4. Service Layer (rag_service_impl.py)
   ↓
   - 벡터 검색 (Domain.vector_stores)
   - 컨텍스트 구성
   - LLM 호출 (Service.chat_service)

5. Domain Layer
   ↓
   - VectorStore.similarity_search()
   - Embedding.embed()
   - Document 처리

6. Infrastructure Layer
   ↓
   - ChromaVectorStore 구현
   - OpenAIEmbedding 구현

7. 응답 반환
   ↓
   RAGResponse 반환
```

---

## Import 방법

### 통합 Import (권장)

```python
from llmkit import Client, Embedding, Document, Agent, RAGChain
```

### 레이어별 Import

```python
# Domain Layer
from llmkit.domain import Document, Embedding, VectorStore

# Infrastructure Layer
from llmkit.infrastructure import ModelRegistry, ParameterAdapter

# Utils
from llmkit.utils import Config, ErrorHandler, retry
```

### Facade Import

```python
from llmkit.facade import Client, RAGChain, Agent
```

---

## 확장 방법

### 새로운 Provider 추가

1. **Infrastructure Layer에 Provider 구현**
   ```python
   # _source_providers/new_provider.py
   class NewProvider(BaseLLMProvider):
       ...
   ```

2. **ProviderFactory에 등록**
   ```python
   # _source_providers/provider_factory.py
   PROVIDER_PRIORITY.append(("new", NewProvider, "NEW_API_KEY"))
   ```

3. **자동으로 사용 가능**
   - 기존 코드 수정 불필요
   - Client(model="new-model")로 사용 가능

### 새로운 기능 추가

1. **Domain Layer에 엔티티/인터페이스 정의**
2. **Infrastructure Layer에 구현체 생성**
3. **Service Layer에 비즈니스 로직 추가**
4. **Handler Layer에 요청 처리 추가**
5. **Facade Layer에 사용자 API 추가**

---

## 테스트 전략

### 단위 테스트

- **Domain Layer**: 순수 함수 테스트 (의존성 없음)
- **Service Layer**: Mock 객체로 테스트
- **Handler Layer**: Mock Service로 테스트

### 통합 테스트

- **Facade → Handler → Service → Infrastructure** 전체 흐름 테스트
- 실제 Provider는 선택적으로 테스트

---

## 성능 최적화

### 1. Lazy Loading
- Embedding 모델은 필요 시 로드
- Vector Store는 필요 시 초기화

### 2. Caching
- EmbeddingCache: 임베딩 결과 캐싱
- NodeCache: Graph 노드 결과 캐싱
- Model Registry: 모델 정보 캐싱

### 3. 비동기 처리
- 모든 LLM 호출은 async/await
- Streaming 지원

---

## 보안 고려사항

### 1. API 키 관리
- 환경 변수로 관리 (.env 파일)
- 절대 코드에 하드코딩하지 않음

### 2. 입력 검증
- Handler Layer에서 모든 입력 검증
- DTO를 통한 타입 안전성

### 3. 에러 처리
- 민감한 정보 노출 방지
- 적절한 에러 메시지

---

## 마이그레이션 가이드

기존 코드는 **하위 호환성**을 유지합니다:

```python
# 기존 코드 (여전히 작동)
from llmkit import Client
client = Client(model="gpt-4o")
response = client.chat("Hello")

# 내부적으로는 새로운 아키텍처 사용
# Facade → Handler → Service → Infrastructure
```

---

## 참고 자료

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)

---

**최종 업데이트**: 2025-12-22
