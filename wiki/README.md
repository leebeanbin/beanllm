# beanllm Wiki

8개 LLM 프로바이더를 단일 인터페이스로 통합하는 Python 라이브러리.

## Architecture

```mermaid
graph TB
    subgraph FACADE["Facade Layer"]
        CLIENT[Client<br/>chat / embed / stream]
        RAGCHAIN[RAGChain<br/>chain.ask]
        AGENT[Agent<br/>agent.run]
    end

    subgraph HANDLER["Handler Layer"]
        CH[ChatHandler]
        EH[EmbedHandler]
        RAG_H[RAGHandler]
    end

    subgraph SERVICE["Service Layer"]
        CS[CompletionService]
        ES[EmbeddingService]
        RS[RetrievalService]
    end

    subgraph DOMAIN["Domain Layer"]
        MSG[Message]
        TOKEN[TokenCounter]
        CHUNK[Chunker]
    end

    subgraph INFRA["Infrastructure Layer"]
        PF[ProviderFactory<br/>+ CircuitBreaker]
        OAI[OpenAIProvider]
        ANT[AnthropicProvider]
        GEM[GeminiProvider]
        MORE[Grok / DeepSeek<br/>Perplexity / Ollama / HF]
        VEC[VectorStoreAdapter<br/>Chroma / FAISS]
    end

    CLIENT --> CH & EH
    RAGCHAIN --> RAG_H
    AGENT --> CH
    CH --> CS
    EH --> ES
    RAG_H --> RS
    CS --> DOMAIN
    ES --> DOMAIN
    CS --> PF
    ES --> PF
    PF --> OAI & ANT & GEM & MORE
    RS --> VEC
```

## Request Flow

```mermaid
sequenceDiagram
    participant U as User Code
    participant C as Client
    participant H as ChatHandler
    participant S as CompletionService
    participant PF as ProviderFactory
    participant P as Provider

    U->>C: client.chat(messages, model="gpt-4o")
    C->>H: handle(ChatRequest)
    H->>S: complete(messages, model)
    S->>PF: get_provider(model)
    PF->>PF: check CircuitBreaker state
    alt Circuit OPEN
        PF->>PF: skip → next provider (fallback=True)
    end
    PF-->>S: provider instance
    S->>P: HTTP /chat/completions
    P-->>S: ChatCompletion
    S->>S: TokenCounter.count()
    S-->>H: CompletionResult
    H-->>C: ChatResponse
    C-->>U: response.content
```

## Domain Pages

| 도메인 | 설명 | 문서 |
|--------|------|------|
| Architecture | Clean Architecture + 레이어 규칙 | [architecture.md](architecture.md) |
| Providers | 8개 프로바이더 + CircuitBreaker | [providers.md](providers.md) |
| Facade API | Client, RAGChain, Agent, StateGraph | [facade.md](facade.md) |
| Testing | 레이어별 테스트 전략 | [testing.md](testing.md) |
| Contributing | 새 프로바이더 추가 방법 | [contributing.md](contributing.md) |
